"""Dev smoke test for the last-N-posts face-leader fallback.

Picks leads whose avatar did NOT resolve to a single face
(``faces_count != 1``) and whose canonical face photo is still
unresolved, re-fetches their profiles via Apify (stored post URLs
expire in 1-2 days), and runs the same inline fallback as the daily
pipeline. Results are written to the DB and summarized in a JSON log.

Usage:
    python scripts/test_face_leader.py
    python scripts/test_face_leader.py --limit 20
    python scripts/test_face_leader.py --yes
    python scripts/test_face_leader.py --keep-photos
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

from scripts.pipeline import _pick_post_images
from src.avatar_downloader import (
    cleanup_lead_photos,
    download_avatar,
    download_post_photos,
)
from src.config import load_config
from src.db import LeadDB
from src.face_detector import FaceDetector
from src.face_embedder import FaceEmbedder
from src.face_leader import resolve_face_leader
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("test_face_leader")

PROFILE_BATCH_SIZE = 50
COST_PER_PROFILE = 0.0023
LOGS_DIR = Path("logs")


def _fmt_s(s: float) -> str:
    return f"{s:.2f} s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face-leader fallback test: probe last N posts for leads with "
                    "faces_count != 1, find dominant face, store canonical photo."
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max leads to process (0 = all).",
    )
    parser.add_argument("--yes", action="store_true",
                        help="Skip cost confirmation prompt.")
    parser.add_argument(
        "--keep-photos", action="store_true",
        help="Don't delete downloaded post photos afterwards.",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()
    fb_cfg = cfg.get("face_fallback") or {}
    fb_limit = int(fb_cfg.get("latest_posts_limit", 5))
    fb_min_cluster = int(fb_cfg.get("min_cluster_size", 2))
    fb_threshold = float(fb_cfg.get("cluster_threshold", 0.5))
    fb_skip_videos = bool(fb_cfg.get("skip_videos", True))
    fb_keep_photos = args.keep_photos or bool(fb_cfg.get("keep_photos", False))

    db = LeadDB("data/leads.db")
    stats_before = db.get_stats()

    effective_limit = args.limit if args.limit > 0 else 10**9
    leads = db.get_leads_needing_face_fallback(limit=effective_limit)

    if not leads:
        print("No leads needing face fallback. Nothing to do.")
        return

    estimated_cost = len(leads) * COST_PER_PROFILE

    print(f"\n{'='*60}")
    print("Face-leader fallback test")
    print(f"{'='*60}")
    print(f"DB totals:")
    print(f"  leads total:          {stats_before['leads_total']}")
    print(f"  single-face avatars:  {stats_before['leads_with_single_face']}")
    print(f"  face photo ready:     {stats_before['leads_with_face_photo']}")
    print(f"Config:")
    print(f"  N (latest_posts):     {fb_limit}")
    print(f"  M (min_cluster):      {fb_min_cluster}")
    print(f"  threshold:            {fb_threshold}")
    print(f"  skip videos:          {fb_skip_videos}")
    print(f"  keep photos:          {fb_keep_photos}")
    print(f"Leads to process:       {len(leads)}"
          + ("" if args.limit == 0 else f" (capped at --limit {args.limit})"))
    print(f"Apify (profile):        ~${estimated_cost:.3f}")
    print(f"{'='*60}")

    if not args.yes:
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    apify = ApifyClient(os.environ["APIFY_API_TOKEN"])
    pipeline = PipelineLogger("logs", "test_face_leader")
    face_detector = FaceDetector()
    face_embedder = FaceEmbedder()

    print("\nLoading face models (first run may download ~155 MB)...")
    t0 = time.perf_counter()
    face_detector._ensure_loaded()
    face_embedder._ensure_loaded()
    print(f"Models loaded in {_fmt_s(time.perf_counter() - t0)}")

    usernames = [l["username"] for l in leads]
    total_start = time.perf_counter()

    # Refetch profiles (URLs in DB are likely expired).
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
            input_params={"batch_size": len(batch)},
            items_count=len(items),
            cost_usd=run_cost,
            duration_ms=detail.get("stats", {}).get("durationMillis"),
        )
        all_profiles.extend(items)
    refetch_time = time.perf_counter() - refetch_start
    print(f"\nApify refetch done in {_fmt_s(refetch_time)}, "
          f"got {len(all_profiles)} profiles.\n")

    print(f"{'#':>4} {'username':<28} {'tried':>5} {'1face':>5} "
          f"{'clust':>5}  verdict")
    print("-" * 72)

    decisions: Counter[str] = Counter()
    per_lead: list[dict] = []
    leader_times: list[float] = []

    for idx, p in enumerate(all_profiles, start=1):
        username = p.get("username") or "?"
        uid = p.get("id") or p.get("pk")
        uid_str = str(uid) if uid else None
        is_private = bool(p.get("private"))

        if is_private or not uid_str:
            decisions["skipped_meta"] += 1
            per_lead.append({
                "username": username, "user_id": uid_str,
                "status": "private_or_no_uid",
            })
            print(f"{idx:>4} {username:<28} {'-':>5} {'-':>5} {'-':>5}  "
                  f"{'private' if is_private else 'no uid'}, skipped")
            continue

        # Re-run the avatar single-face check — the profile may have
        # changed since the last scan, and we want the fresh decision.
        avatar_url = p.get("profilePicUrlHD") or p.get("profilePicUrl")
        avatar_path = download_avatar(
            avatar_url, user_id=uid_str, username=username,
        )
        if avatar_path:
            avatar_faces = face_detector.count_faces(avatar_path)
            db.update_lead_avatar(username, avatar_path, avatar_faces)
            if avatar_faces == 1:
                db.update_lead_face(username, avatar_path)
                decisions["avatar_now_single"] += 1
                per_lead.append({
                    "username": username, "user_id": uid_str,
                    "status": "avatar_now_single", "avatar_faces": avatar_faces,
                })
                print(f"{idx:>4} {username:<28} {'-':>5} {'-':>5} {'-':>5}  "
                      f"avatar now has 1 face, stored")
                continue

        # Fallback: last N posts.
        post_urls = _pick_post_images(
            p.get("latestPosts"), limit=fb_limit, skip_videos=fb_skip_videos,
        )
        local_paths = download_post_photos(post_urls, user_id=uid_str)

        t0 = time.perf_counter()
        result = resolve_face_leader(
            local_paths,
            face_detector,
            face_embedder,
            min_cluster_size=fb_min_cluster,
            cluster_threshold=fb_threshold,
        )
        leader_times.append(time.perf_counter() - t0)

        if result:
            decisions["fallback_resolved"] += 1
            db.update_lead_face(username, str(result.photo_path))
            verdict = f"leader size={result.cluster_size}"
            per_lead.append({
                "username": username, "user_id": uid_str,
                "status": "resolved",
                "photos_tried": result.photos_tried,
                "photos_single_face": result.photos_single_face,
                "cluster_size": result.cluster_size,
                "det_score": round(result.det_score, 3),
                "photo_path": str(result.photo_path),
            })
            print(f"{idx:>4} {username:<28} "
                  f"{result.photos_tried:>5} "
                  f"{result.photos_single_face:>5} "
                  f"{result.cluster_size:>5}  {verdict}")
        else:
            decisions["fallback_skipped"] += 1
            per_lead.append({
                "username": username, "user_id": uid_str,
                "status": "skipped_no_leader",
                "photos_tried": len(local_paths),
            })
            print(f"{idx:>4} {username:<28} "
                  f"{len(local_paths):>5} {'-':>5} {'-':>5}  "
                  f"no leader, skipped")

        if not fb_keep_photos:
            cleanup_lead_photos(
                uid_str,
                keep=(result.photo_path if result else None),
            )

    total_time = time.perf_counter() - total_start
    face_detector.close()
    face_embedder.close()

    stats_after = db.get_stats()

    print("-" * 72)
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Leads requested:      {len(leads)}")
    print(f"Profiles returned:    {len(all_profiles)}")
    for k, v in decisions.most_common():
        print(f"  {k:<20}{v:>4}")
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

    delta_photos = (stats_after["leads_with_face_photo"]
                    - stats_before["leads_with_face_photo"])
    print(f"\nDB deltas:")
    print(f"  face photo ready:   +{delta_photos}")

    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = LOGS_DIR / f"test_face_leader_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": args.limit,
        "keep_photos": fb_keep_photos,
        "config": {
            "latest_posts_limit": fb_limit,
            "min_cluster_size": fb_min_cluster,
            "cluster_threshold": fb_threshold,
            "skip_videos": fb_skip_videos,
        },
        "db_before": stats_before,
        "db_after": stats_after,
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
