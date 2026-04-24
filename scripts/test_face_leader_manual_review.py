"""Manual-review run of the face-leader fallback.

Picks every lead in the DB whose avatar face count is NOT exactly one
(``faces_count != 1``, so group shots + no-face avatars) and replays
the last-N-posts face-leader algorithm on them — but instead of writing
the decision back to the DB and deleting the candidate photos, it lays
everything out on disk for eyeball verification.

Per processed lead the script produces:

    data/manual_review/<username>/
        _avatar.jpg   # the owner's current avatar (context)
        00.jpg        # candidate photo #1 from latestPosts
        01.jpg        # candidate photo #2
        ...           # all candidates are kept, nothing is deleted
    data/manual_review/<username>.jpg   # <-- the algorithm's winner

That layout lets you open a folder, see every photo the algorithm saw,
and compare it against the single ``<username>.jpg`` file sitting next
to the folder. If the winner file is missing, the algorithm returned
"no unambiguous leader" for that lead.

Side-effects:
  * Apify profile refetch (paid, ~$0.0023 per lead — confirmation prompt).
  * Writes files under ``data/manual_review/``.
  * Does NOT update ``faces_count`` / ``face_photo_path`` in the DB.
  * Does NOT delete any downloaded photo.

Usage:
    python scripts/test_face_leader_manual_review.py
    python scripts/test_face_leader_manual_review.py --limit 30
    python scripts/test_face_leader_manual_review.py --yes
    python scripts/test_face_leader_manual_review.py --only-multi   # skip 0-face avatars
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv

from scripts.pipeline import _pick_post_images
from src.avatar_downloader import _safe_stem, download_avatar, download_post_photos
from src.config import load_config
from src.db import LeadDB
from src.face_embedder import FaceEmbedder
from src.face_leader import resolve_face_leader
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("test_face_leader_manual_review")

PROFILE_BATCH_SIZE = 50
COST_PER_PROFILE = 0.0023
LOGS_DIR = Path("logs")
REVIEW_DIR = Path("data/manual_review")


def _fmt_s(s: float) -> str:
    return f"{s:.2f} s"


def _username_slug(username: str) -> str:
    """Filesystem-safe version of the Instagram username."""
    return _safe_stem(username) or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Manual-review run of the face-leader fallback for leads "
            "whose avatar face count is not exactly 1."
        )
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max leads to process (0 = all).",
    )
    parser.add_argument("--yes", action="store_true",
                        help="Skip cost confirmation prompt.")
    parser.add_argument(
        "--only-multi", action="store_true",
        help="Only process leads whose avatar has >=2 faces "
             "(skip 0-face avatars).",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()
    fb_cfg = cfg.get("face_fallback") or {}
    fb_limit = int(fb_cfg.get("latest_posts_limit", 5))
    fb_min_cluster = int(fb_cfg.get("min_cluster_size", 2))
    fb_threshold = float(fb_cfg.get("cluster_threshold", 0.5))
    fb_skip_videos = bool(fb_cfg.get("skip_videos", True))
    fd_cfg = cfg.get("face_detection") or {}
    min_det_score = float(fd_cfg.get("min_det_score", 0.7))

    db = LeadDB("data/leads.db")
    stats_before = db.get_stats()

    effective_limit = args.limit if args.limit > 0 else 10**9
    leads = db.get_leads_with_non_single_face(limit=effective_limit)

    if args.only_multi:
        leads = [l for l in leads if (l.get("faces_count") or 0) >= 2]

    if not leads:
        print("No leads with faces_count != 1 in the DB. Nothing to do.")
        return

    estimated_cost = len(leads) * COST_PER_PROFILE

    print(f"\n{'='*64}")
    print("Face-leader manual-review test")
    print(f"{'='*64}")
    print(f"Review folder:          {REVIEW_DIR}")
    print(f"DB totals:")
    print(f"  leads total:          {stats_before['leads_total']}")
    print(f"  single-face avatars:  {stats_before['leads_with_single_face']}")
    print(f"  face photo ready:     {stats_before['leads_with_face_photo']}")
    print(f"Config:")
    print(f"  N (latest_posts):     {fb_limit}")
    print(f"  M (min_cluster):      {fb_min_cluster}")
    print(f"  threshold:            {fb_threshold}")
    print(f"  skip videos:          {fb_skip_videos}")
    print(f"  only-multi filter:    {args.only_multi}")
    print(f"Leads to process:       {len(leads)}"
          + ("" if args.limit == 0 else f" (capped at --limit {args.limit})"))
    print(f"Apify (profile):        ~${estimated_cost:.3f}")
    print("Policy:                 DB is NOT modified, photos are NOT deleted.")
    print(f"{'='*64}")

    if not args.yes:
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    apify = ApifyClient(os.environ["APIFY_API_TOKEN"])
    pipeline = PipelineLogger("logs", "test_face_leader_manual_review")
    face_embedder = FaceEmbedder(min_det_score=min_det_score)

    print(f"\nLoading SCRFD + ArcFace (min_det_score={min_det_score})...")
    t0 = time.perf_counter()
    face_embedder._ensure_loaded()
    print(f"Models loaded in {_fmt_s(time.perf_counter() - t0)}")

    usernames = [l["username"] for l in leads]
    total_start = time.perf_counter()

    refetch_start = time.perf_counter()
    all_profiles: list[dict] = []
    apify_cost = 0.0
    for i in range(0, len(usernames), PROFILE_BATCH_SIZE):
        batch = usernames[i:i + PROFILE_BATCH_SIZE]
        print(f"\nApify batch {i // PROFILE_BATCH_SIZE + 1}: "
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
            input_params={"batch_size": len(batch), "mode": "manual_review"},
            items_count=len(items),
            cost_usd=run_cost,
            duration_ms=detail.get("stats", {}).get("durationMillis"),
        )
        all_profiles.extend(items)
    refetch_time = time.perf_counter() - refetch_start
    print(f"\nApify refetch done in {_fmt_s(refetch_time)}, "
          f"got {len(all_profiles)} profiles.\n")

    print(f"{'#':>4} {'username':<28} {'avF':>3} {'tried':>5} {'1face':>5} "
          f"{'clust':>5}  verdict")
    print("-" * 74)

    decisions: Counter[str] = Counter()
    per_lead: list[dict] = []
    leader_times: list[float] = []

    for idx, p in enumerate(all_profiles, start=1):
        username = p.get("username") or "?"
        uid = p.get("id") or p.get("pk")
        uid_str = str(uid) if uid else None
        is_private = bool(p.get("private"))

        slug = _username_slug(username)
        lead_dir = REVIEW_DIR / slug

        if is_private or not uid_str:
            decisions["skipped_meta"] += 1
            per_lead.append({
                "username": username, "user_id": uid_str,
                "status": "private_or_no_uid",
            })
            print(f"{idx:>4} {username:<28} {'-':>3} {'-':>5} {'-':>5} "
                  f"{'-':>5}  {'private' if is_private else 'no uid'}, skipped")
            continue

        lead_dir.mkdir(parents=True, exist_ok=True)

        # Re-download the avatar so we have it for visual context AND to
        # report the fresh SCRFD count (the stored faces_count may be
        # out of date by the time we get here).
        avatar_url = p.get("profilePicUrlHD") or p.get("profilePicUrl")
        avatar_path = download_avatar(
            avatar_url, user_id=uid_str, username=username,
        )
        avatar_faces: int | None = None
        if avatar_path:
            try:
                avatar_faces = face_embedder.count_faces(avatar_path)
            except Exception as e:
                log.warning("avatar_detect_error",
                            username=username, error=str(e))
            try:
                shutil.copy2(avatar_path, lead_dir / "_avatar.jpg")
            except OSError as e:
                log.warning("avatar_copy_error",
                            username=username, error=str(e))

        if avatar_faces == 1:
            # Avatar is now clean — nothing to disambiguate. Skip the
            # fallback entirely but keep the folder so you can see that
            # the old DB row is stale.
            decisions["avatar_now_single"] += 1
            per_lead.append({
                "username": username, "user_id": uid_str,
                "status": "avatar_now_single",
                "avatar_faces": avatar_faces,
                "review_dir": str(lead_dir),
            })
            print(f"{idx:>4} {username:<28} {avatar_faces:>3} {'-':>5} "
                  f"{'-':>5} {'-':>5}  avatar now single, skipped")
            continue

        # Download the last N post photos straight into the review folder
        # (download_post_photos uses its second arg as the subfolder name,
        # so passing the username-slug gives us data/manual_review/<slug>/).
        post_urls = _pick_post_images(
            p.get("latestPosts"), limit=fb_limit, skip_videos=fb_skip_videos,
        )
        local_paths = download_post_photos(
            post_urls, user_id=slug, dest_root=REVIEW_DIR,
        )

        t0 = time.perf_counter()
        result = resolve_face_leader(
            local_paths,
            face_embedder,
            min_cluster_size=fb_min_cluster,
            cluster_threshold=fb_threshold,
        )
        leader_times.append(time.perf_counter() - t0)

        winner_copy_path: Path | None = None
        if result:
            decisions["fallback_resolved"] += 1
            winner_copy_path = REVIEW_DIR / f"{slug}.jpg"
            try:
                shutil.copy2(result.photo_path, winner_copy_path)
            except OSError as e:
                log.warning("winner_copy_error",
                            username=username, error=str(e))
                winner_copy_path = None
            verdict = f"leader size={result.cluster_size}"
            per_lead.append({
                "username": username, "user_id": uid_str,
                "status": "resolved",
                "avatar_faces": avatar_faces,
                "photos_tried": result.photos_tried,
                "photos_single_face": result.photos_single_face,
                "cluster_size": result.cluster_size,
                "det_score": round(result.det_score, 3),
                "review_dir": str(lead_dir),
                "winner_source": str(result.photo_path),
                "winner_copy": str(winner_copy_path) if winner_copy_path else None,
            })
            print(f"{idx:>4} {username:<28} "
                  f"{(avatar_faces if avatar_faces is not None else '?'):>3} "
                  f"{result.photos_tried:>5} "
                  f"{result.photos_single_face:>5} "
                  f"{result.cluster_size:>5}  {verdict}")
        else:
            decisions["fallback_skipped"] += 1
            per_lead.append({
                "username": username, "user_id": uid_str,
                "status": "skipped_no_leader",
                "avatar_faces": avatar_faces,
                "photos_tried": len(local_paths),
                "review_dir": str(lead_dir),
            })
            print(f"{idx:>4} {username:<28} "
                  f"{(avatar_faces if avatar_faces is not None else '?'):>3} "
                  f"{len(local_paths):>5} {'-':>5} {'-':>5}  "
                  f"no leader, skipped")

    total_time = time.perf_counter() - total_start
    face_embedder.close()

    print("-" * 74)
    print(f"\n{'='*64}")
    print("Summary")
    print(f"{'='*64}")
    print(f"Leads requested:      {len(leads)}")
    print(f"Profiles returned:    {len(all_profiles)}")
    for k, v in decisions.most_common():
        print(f"  {k:<22}{v:>4}")
    if leader_times:
        avg = sum(leader_times) / len(leader_times)
        print(f"\nLeader resolution timing:")
        print(f"  avg per lead:       {avg * 1000:.1f} ms")
        print(f"  min/max:            {min(leader_times) * 1000:.1f} / "
              f"{max(leader_times) * 1000:.1f} ms")
    print(f"\nTotal time:           {_fmt_s(total_time)}")
    print(f"  Apify refetch:      {_fmt_s(refetch_time)}")
    print(f"  Download + resolve: {_fmt_s(total_time - refetch_time)}")
    print(f"Apify cost (actual):  ${apify_cost:.4f}")
    print(f"\nReview artifacts in:  {REVIEW_DIR.resolve()}")
    print("  <username>/        folder with all candidate photos + _avatar.jpg")
    print("  <username>.jpg     algorithm's chosen winner (missing = no leader)")

    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = LOGS_DIR / f"test_face_leader_manual_review_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": args.limit,
        "only_multi": args.only_multi,
        "review_dir": str(REVIEW_DIR.resolve()),
        "config": {
            "latest_posts_limit": fb_limit,
            "min_cluster_size": fb_min_cluster,
            "cluster_threshold": fb_threshold,
            "skip_videos": fb_skip_videos,
        },
        "db_before": stats_before,
        "decisions": dict(decisions),
        "time_s": {
            "total": round(total_time, 2),
            "apify_refetch": round(refetch_time, 2),
            "download_resolve": round(total_time - refetch_time, 2),
        },
        "apify_cost_usd": round(apify_cost, 6),
        "leads": per_lead,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON log: {json_path}")


if __name__ == "__main__":
    main()
