"""Test: take N leads from DB, refetch profiles, run face detection.

Three modes (mutually exclusive):
  default (all):      EVERY non-private lead with a fetched profile,
                      regardless of current faces_count / avatar_path.
                      Overwrites stale counts — this is what you want
                      after a detector migration (e.g. MediaPipe → SCRFD)
                      or whenever you're doing a full re-validation pass.
  --only-new:         leads that never had face detection
                      (profile_fetched=1, avatar_path IS NULL, not private).
                      Useful for incremental daily runs when the rest of
                      the base is already verified.
  --retest-nonsingle: leads already scanned whose faces_count != 1
                      (i.e. 0 faces or group photos) — narrow re-check
                      without touching clean single-face leads.

Each picked lead is re-fetched via Apify (CDN URLs expire in 1-2 days),
avatar is downloaded, SCRFD counts faces, result is written back to
`faces_count` in the DB. A per-run summary is saved as JSON under `logs/`.

Unlike the main pipeline (which stores avatars at
``data/avatars/<user_id>.jpg`` — a numeric Instagram pk — so that
username changes don't break dedup), this test script downloads to
``data/avatars_review/<username>.jpg``. That way the filenames line up
with the console output and the JSON log, making manual spot-checking
straightforward. The DB's ``avatar_path`` field will point at the
review folder after the test — on the next full pipeline run
``download_avatar`` will simply re-download into the canonical
``data/avatars/`` location (cheap, no Apify re-fetch for the image).

By default downloaded avatars are kept on disk so you can manually
eyeball edge cases. Pass ``--delete-photos`` to wipe them per-lead,
``--clean-start`` to wipe the review folder before the run. Downloads
are idempotent — re-running without ``--clean-start`` reuses existing
files (useful for re-detecting with a different SCRFD threshold).

Usage:
    python scripts/test_avatar_batch.py                   # re-detect every lead, keep photos
    python scripts/test_avatar_batch.py --limit 20        # cap
    python scripts/test_avatar_batch.py --yes             # skip prompt
    python scripts/test_avatar_batch.py --delete-photos   # wipe per-lead after detect
    python scripts/test_avatar_batch.py --clean-start     # wipe review folder first
    python scripts/test_avatar_batch.py --only-new        # only leads without avatar_path
    python scripts/test_avatar_batch.py --retest-nonsingle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv

from src.avatar_downloader import download_avatar
from src.config import load_config
from src.db import LeadDB
from src.face_embedder import make_face_embedder
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("test_avatar_batch")

PROFILE_BATCH_SIZE = 50
COST_PER_PROFILE = 0.0023
LOGS_DIR = Path("logs")
# Dedicated review folder for this test: filenames are username-based
# (human-readable) so you can cross-reference the console table and JSON
# log against the actual image files. Separate from the main pipeline's
# ``data/avatars/`` which uses numeric user_id-based names.
REVIEW_DIR = Path("data/avatars_review")


def verdict(faces: int) -> str:
    if faces == 0:
        return "no face"
    if faces == 1:
        return "single face"
    return f"{faces} faces"


def cleanup_review_dir() -> tuple[int, int]:
    """Remove every file in ``REVIEW_DIR``. Returns (deleted, failed)."""
    if not REVIEW_DIR.exists():
        return 0, 0
    deleted = 0
    failed = 0
    for f in REVIEW_DIR.iterdir():
        if not f.is_file():
            continue
        try:
            f.unlink()
            deleted += 1
        except OSError:
            failed += 1
    return deleted, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-test face detection on leads from DB"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max leads to process (default: 0 = all in DB).",
    )
    parser.add_argument("--yes", action="store_true",
                        help="Skip cost confirmation prompt.")
    parser.add_argument(
        "--delete-photos",
        action="store_true",
        help="Delete each avatar right after its face count is written "
             "to the DB. Default is to keep them for manual review.",
    )
    parser.add_argument(
        "--clean-start",
        action="store_true",
        help=f"Wipe {REVIEW_DIR} before the run. Default is to leave "
             "existing files in place (download_avatar is idempotent, "
             "so unchanged usernames won't be re-fetched).",
    )
    # Default mode = "all" (process every eligible lead, overwriting
    # stale faces_count). The two narrow modes below are opt-in.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--only-new",
        action="store_true",
        help="Narrow scope: only leads that never had face detection "
             "(avatar_path IS NULL). Default is to re-process EVERY "
             "non-private lead with a fetched profile.",
    )
    mode_group.add_argument(
        "--retest-nonsingle",
        action="store_true",
        help="Narrow scope: only leads whose faces_count != 1 "
             "(0 faces or group photos). Default is to re-process "
             "every non-private lead.",
    )
    args = parser.parse_args()

    load_dotenv()
    db = LeadDB("data/leads.db")

    db_stats = db.get_stats()

    effective_limit = args.limit if args.limit > 0 else 10**9
    if args.only_new:
        mode = "only-new"
        leads = db.get_leads_needing_avatar(limit=effective_limit)
    elif args.retest_nonsingle:
        mode = "retest-nonsingle"
        leads = db.get_leads_with_non_single_face(limit=effective_limit)
    else:
        mode = "all"
        leads = db.get_all_face_detection_candidates(limit=effective_limit)

    if not leads:
        if args.only_new:
            msg = "No leads need avatar processing."
        elif args.retest_nonsingle:
            msg = "No leads with faces_count != 1."
        else:
            msg = "No non-private leads with a fetched profile in the DB."
        print(f"{msg} Nothing to do.")
        return

    # In the default ("all") mode we overwrite stale faces_count values
    # from a previous detector run. Snapshot prior values so the summary
    # can show how many leads flipped verdict.
    prior_counts: dict[str, int | None] = {}
    if mode == "all":
        prior_counts = {
            l["username"]: l.get("faces_count")
            for l in leads
        }

    if args.clean_start:
        cleaned_pre, cleaned_pre_failed = cleanup_review_dir()
    else:
        cleaned_pre, cleaned_pre_failed = 0, 0

    estimated_cost = len(leads) * COST_PER_PROFILE
    print(f"\n{'='*60}")
    print(f"Batch avatar test  [mode: {mode}]")
    print(f"{'='*60}")
    print(f"DB totals:")
    print(f"  leads total:          {db_stats['leads_total']}")
    print(f"  profile fetched:      {db_stats['leads_with_profile']}")
    print(f"  with avatar already:  {db_stats['leads_with_avatar']}")
    print(f"  single-face so far:   {db_stats['leads_with_single_face']}")
    print(f"Review folder:          {REVIEW_DIR} (filenames: <username>.jpg)")
    if args.clean_start:
        print(f"Pre-cleanup:")
        print(f"  files removed:        {cleaned_pre}"
              + (f" (failed: {cleaned_pre_failed})"
                 if cleaned_pre_failed else ""))
    else:
        print(f"Pre-cleanup:            skipped (pass --clean-start to wipe)")
    print(f"Leads to process:       {len(leads)}"
          + ("" if args.limit == 0 else f" (capped at --limit {args.limit})"))
    print(f"Apify (profile):        ~${estimated_cost:.3f}")
    print(f"Photos:                 "
          + ("DELETE after detection" if args.delete_photos
             else "KEEP on disk for manual review"))
    print(f"{'='*60}")
    if not args.yes:
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    cfg = load_config()

    apify = ApifyClient(os.environ["APIFY_API_TOKEN"])
    pipeline = PipelineLogger("logs", "test_avatar_batch")
    # Avatar-only test — single 320x320 SCRFD instance is enough.
    face_embedder = make_face_embedder(cfg, kind="avatar")

    total_start = time.perf_counter()
    print(f"\nLoading SCRFD model "
          f"(min_det_score={face_embedder.min_det_score}, "
          f"det_size={face_embedder.det_size[0]})...")
    t0 = time.perf_counter()
    face_embedder._ensure_loaded()
    print(f"Model loaded in {(time.perf_counter() - t0) * 1000:.1f} ms\n")

    usernames = [l["username"] for l in leads]

    refetch_start = time.perf_counter()
    all_profiles: list[dict] = []
    apify_cost = 0.0
    for i in range(0, len(usernames), PROFILE_BATCH_SIZE):
        batch = usernames[i:i + PROFILE_BATCH_SIZE]
        print(f"Apify batch {i // PROFILE_BATCH_SIZE + 1}: "
              f"fetching {len(batch)} profiles...")
        run = apify.actor("apify/instagram-profile-scraper").call(run_input={
            "usernames": batch,
        })
        detail = apify.run(run["id"]).get()
        items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
        run_cost = detail.get("usageTotalUsd", 0) or 0
        apify_cost += run_cost
        pipeline.log_run(
            actor_id="apify/instagram-profile-scraper",
            run_id=run["id"], status=run["status"],
            input_params={"batch_size": len(batch), "mode": mode},
            items_count=len(items),
            cost_usd=run_cost,
            duration_ms=detail.get("stats", {}).get("durationMillis"),
        )
        all_profiles.extend(items)
    refetch_time = time.perf_counter() - refetch_start
    print(f"\nApify refetch done in {refetch_time:.1f}s, "
          f"got {len(all_profiles)} profiles.\n")

    print(f"{'#':>4} {'username':<30} {'priv':>4} {'faces':>6}  {'ms':>5}  verdict")
    print("-" * 70)

    face_counts: Counter[int] = Counter()
    download_fail = 0
    private_count = 0
    detect_times: list[float] = []
    deleted_files = 0
    delete_failed = 0
    per_lead: list[dict] = []
    # Only populated in --all mode: how many leads' faces_count changed
    # compared to what was stored from the previous detector run.
    flipped_total = 0
    flipped_1_to_multi = 0
    flipped_multi_to_1 = 0
    flipped_examples: list[dict] = []

    for idx, profile in enumerate(all_profiles, start=1):
        username = profile.get("username") or "?"
        uid = profile.get("id") or profile.get("pk")
        is_private = bool(profile.get("private"))

        if is_private:
            private_count += 1
            per_lead.append({
                "username": username,
                "user_id": str(uid) if uid else None,
                "status": "private",
                "faces_count": None,
                "detect_ms": None,
            })
            print(f"{idx:>4} {username:<30} {'yes':>4} {'-':>6}  {'-':>5}  private, skipped")
            continue

        avatar_url = profile.get("profilePicUrlHD") or profile.get("profilePicUrl")
        # Force username-based filename by passing user_id=None, and route
        # to the review-only folder so the main pipeline's canonical
        # ``data/avatars/`` store stays pristine.
        avatar_path = download_avatar(
            avatar_url,
            user_id=None,
            username=username,
            avatars_dir=REVIEW_DIR,
        )
        if not avatar_path:
            download_fail += 1
            per_lead.append({
                "username": username,
                "user_id": str(uid) if uid else None,
                "status": "download_failed",
                "faces_count": None,
                "detect_ms": None,
            })
            print(f"{idx:>4} {username:<30} {'no':>4} {'-':>6}  {'-':>5}  download failed")
            continue

        t0 = time.perf_counter()
        faces = face_embedder.count_faces(avatar_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        detect_times.append(elapsed_ms)
        face_counts[faces] += 1

        # Track prior vs new verdict BEFORE writing — otherwise
        # ``prior`` would be whatever we just wrote.
        prior = prior_counts.get(username) if mode == "all" else None

        db.update_lead_avatar(username, avatar_path, faces)

        if mode == "all" and prior is not None and prior != faces:
            flipped_total += 1
            if prior == 1 and faces != 1:
                flipped_1_to_multi += 1
            elif prior != 1 and faces == 1:
                flipped_multi_to_1 += 1
            if len(flipped_examples) < 20:
                flipped_examples.append({
                    "username": username,
                    "prior": prior,
                    "new": faces,
                })

        suffix = ""
        if args.delete_photos:
            try:
                Path(avatar_path).unlink(missing_ok=True)
                deleted_files += 1
                suffix = "  [deleted]"
            except OSError as e:
                delete_failed += 1
                suffix = f"  [delete failed: {e}]"

        per_lead.append({
            "username": username,
            "user_id": str(uid) if uid else None,
            "status": "ok",
            "faces_count": faces,
            "detect_ms": round(elapsed_ms, 2),
        })

        print(f"{idx:>4} {username:<30} {'no':>4} {faces:>6}  "
              f"{elapsed_ms:>5.1f}  {verdict(faces)}{suffix}")

    face_embedder.close()
    total_time = time.perf_counter() - total_start

    print("-" * 70)
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Leads requested:      {len(leads)}")
    print(f"Profiles returned:    {len(all_profiles)}")
    print(f"  private (skipped):  {private_count}")
    print(f"  download failed:    {download_fail}")
    print(f"  detected:           {sum(face_counts.values())}")
    print(f"\nFace count distribution:")
    for count in sorted(face_counts):
        verdict_str = verdict(count)
        print(f"  {count} faces: {face_counts[count]:>4}  ({verdict_str})")
    if detect_times:
        print(f"\nDetection timing:")
        print(f"  avg:     {sum(detect_times) / len(detect_times):.1f} ms")
        print(f"  min/max: {min(detect_times):.1f} / {max(detect_times):.1f} ms")
    print(f"\nTotal time:           {total_time:.1f}s")
    print(f"  Apify refetch:      {refetch_time:.1f}s")
    print(f"  Download + detect:  {total_time - refetch_time:.1f}s")
    print(f"Apify cost (actual):  ${apify_cost:.4f}")

    if args.delete_photos or args.clean_start:
        print(f"\nPhoto cleanup:")
        print(f"  pre-run deleted:    {cleaned_pre}"
              + ("" if args.clean_start else " (skipped)"))
        print(f"  per-lead deleted:   {deleted_files}"
              + ("" if args.delete_photos else " (skipped — kept for review)"))
        if delete_failed:
            print(f"  delete failed:      {delete_failed}")
    else:
        print(f"\nPhotos kept in:       {REVIEW_DIR} (one <username>.jpg per lead)")

    single_face = face_counts.get(1, 0)
    detected_total = sum(face_counts.values())
    print(f"\nSherlock candidates (single-face): {single_face}/{detected_total}")

    flipped_other = 0
    if mode == "all":
        # How many previously-stored faces_count values were overwritten
        # by the new detector. "flipped_other" = changes that aren't a
        # simple {1 <-> not 1} swap (e.g. 2 -> 3, 0 -> 2).
        flipped_other = flipped_total - flipped_1_to_multi - flipped_multi_to_1
        print(f"\nDetector diff vs previous run:")
        print(f"  total flipped:            {flipped_total}")
        print(f"    1 -> not 1 (lost):      {flipped_1_to_multi}")
        print(f"    not 1 -> 1 (gained):    {flipped_multi_to_1}")
        print(f"    other shifts:           {flipped_other}")
        if flipped_examples:
            shown = min(10, len(flipped_examples))
            print(f"  sample (first {shown}):")
            for ex in flipped_examples[:shown]:
                print(f"    {ex['username']:<30} {ex['prior']} -> {ex['new']}")

    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = LOGS_DIR / f"test_avatar_batch_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "limit": args.limit,
        "delete_photos": args.delete_photos,
        "clean_start": args.clean_start,
        "review_dir": str(REVIEW_DIR),
        "db_stats": db_stats,
        "leads_requested": len(leads),
        "profiles_returned": len(all_profiles),
        "private_skipped": private_count,
        "download_failed": download_fail,
        "detected_total": detected_total,
        "face_counts": {str(k): v for k, v in sorted(face_counts.items())},
        "detect_timing_ms": {
            "avg": round(sum(detect_times) / len(detect_times), 2) if detect_times else None,
            "min": round(min(detect_times), 2) if detect_times else None,
            "max": round(max(detect_times), 2) if detect_times else None,
        },
        "time_s": {
            "total": round(total_time, 2),
            "apify_refetch": round(refetch_time, 2),
            "download_detect": round(total_time - refetch_time, 2),
        },
        "apify_cost_usd": round(apify_cost, 6),
        "photos": {
            "pre_run_deleted": cleaned_pre,
            "pre_run_delete_failed": cleaned_pre_failed,
            "per_lead_deleted": deleted_files,
            "per_lead_delete_failed": delete_failed,
        },
        "detector_diff": ({
            "flipped_total": flipped_total,
            "flipped_1_to_not1": flipped_1_to_multi,
            "flipped_not1_to_1": flipped_multi_to_1,
            "flipped_other": flipped_other,
            "examples": flipped_examples,
        } if mode == "all" else None),
        "leads": per_lead,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\nJSON log: {json_path}")


if __name__ == "__main__":
    main()
