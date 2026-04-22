"""Test: take N leads from DB, refetch profiles, run face detection.

Two modes:
  default:            leads that never had face detection
                      (profile_fetched=1, avatar_path IS NULL, not private)
  --retest-nonsingle: leads already scanned whose faces_count != 1
                      (i.e. 0 faces or group photos) — useful when tuning
                      the detector or swapping models.

Each picked lead is re-fetched via Apify (CDN URLs expire in 1-2 days),
avatar is downloaded, MediaPipe counts faces, result is written back to
`faces_count` in the DB, and the photo is deleted. A per-run summary is
also saved as JSON under `logs/`.

Usage:
    python scripts/test_avatar_batch.py                   # all new leads
    python scripts/test_avatar_batch.py --limit 20        # cap
    python scripts/test_avatar_batch.py --yes             # skip prompt
    python scripts/test_avatar_batch.py --keep-photos     # don't delete
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

from src.avatar_downloader import AVATARS_DIR, download_avatar
from src.db import LeadDB
from src.face_detector import FaceDetector
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("test_avatar_batch")

PROFILE_BATCH_SIZE = 50
COST_PER_PROFILE = 0.0023
LOGS_DIR = Path("logs")


def verdict(faces: int) -> str:
    if faces == 0:
        return "no face"
    if faces == 1:
        return "single face"
    return f"{faces} faces"


def cleanup_avatars_dir() -> tuple[int, int]:
    """Remove every file in data/avatars/. Returns (deleted, failed)."""
    if not AVATARS_DIR.exists():
        return 0, 0
    deleted = 0
    failed = 0
    for f in AVATARS_DIR.iterdir():
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
        "--keep-photos",
        action="store_true",
        help="Don't delete downloaded avatars after detection.",
    )
    parser.add_argument(
        "--retest-nonsingle",
        action="store_true",
        help="Only process leads whose faces_count != 1 (re-detect).",
    )
    args = parser.parse_args()

    load_dotenv()
    db = LeadDB("data/leads.db")

    db_stats = db.get_stats()

    effective_limit = args.limit if args.limit > 0 else 10**9
    mode = "retest-nonsingle" if args.retest_nonsingle else "needing"
    if args.retest_nonsingle:
        leads = db.get_leads_with_non_single_face(limit=effective_limit)
    else:
        leads = db.get_leads_needing_avatar(limit=effective_limit)

    if not leads:
        msg = ("No leads with faces_count != 1."
               if args.retest_nonsingle
               else "No leads need avatar processing.")
        print(f"{msg} Nothing to do.")
        return

    cleaned_pre, cleaned_pre_failed = cleanup_avatars_dir()

    estimated_cost = len(leads) * COST_PER_PROFILE
    print(f"\n{'='*60}")
    print(f"Batch avatar test  [mode: {mode}]")
    print(f"{'='*60}")
    print(f"DB totals:")
    print(f"  leads total:          {db_stats['leads_total']}")
    print(f"  profile fetched:      {db_stats['leads_with_profile']}")
    print(f"  with avatar already:  {db_stats['leads_with_avatar']}")
    print(f"  single-face so far:   {db_stats['leads_with_single_face']}")
    print(f"Pre-cleanup:")
    print(f"  files removed:        {cleaned_pre}"
          + (f" (failed: {cleaned_pre_failed})"
             if cleaned_pre_failed else ""))
    print(f"Leads to process:       {len(leads)}"
          + ("" if args.limit == 0 else f" (capped at --limit {args.limit})"))
    print(f"Apify (profile):        ~${estimated_cost:.3f}")
    print(f"Photos:                 "
          + ("KEEP on disk" if args.keep_photos
             else "DELETE after detection"))
    print(f"{'='*60}")
    if not args.yes:
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    apify = ApifyClient(os.environ["APIFY_API_TOKEN"])
    pipeline = PipelineLogger("logs", "test_avatar_batch")
    face_detector = FaceDetector()

    total_start = time.perf_counter()
    print("\nLoading MediaPipe model...")
    t0 = time.perf_counter()
    face_detector._ensure_loaded()
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
        avatar_path = download_avatar(
            avatar_url,
            user_id=str(uid) if uid else None,
            username=username,
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
        faces = face_detector.count_faces(avatar_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        detect_times.append(elapsed_ms)
        face_counts[faces] += 1

        db.update_lead_avatar(username, avatar_path, faces)

        suffix = ""
        if not args.keep_photos:
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

    face_detector.close()
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

    if not args.keep_photos:
        print(f"\nPhoto cleanup:")
        print(f"  pre-run deleted:    {cleaned_pre}")
        print(f"  per-lead deleted:   {deleted_files}")
        if delete_failed:
            print(f"  delete failed:      {delete_failed}")

    single_face = face_counts.get(1, 0)
    detected_total = sum(face_counts.values())
    print(f"\nSherlock candidates (single-face): {single_face}/{detected_total}")

    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = LOGS_DIR / f"test_avatar_batch_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "limit": args.limit,
        "keep_photos": args.keep_photos,
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
        "leads": per_lead,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\nJSON log: {json_path}")


if __name__ == "__main__":
    main()
