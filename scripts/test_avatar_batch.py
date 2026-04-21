"""Test: take N leads from DB, refetch profiles, run face detection.

Picks leads that haven't had avatars processed yet, refetches their
profiles via Apify (to get fresh CDN URLs), downloads each avatar,
runs MediaPipe face detection, and prints a per-lead breakdown plus
aggregate stats.

Usage:
    python scripts/test_avatar_batch.py                 # 100 leads, with prompt
    python scripts/test_avatar_batch.py --limit 20      # fewer
    python scripts/test_avatar_batch.py --yes           # skip cost prompt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv

from src.avatar_downloader import download_avatar
from src.db import LeadDB
from src.face_detector import FaceDetector
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("test_avatar_batch")

PROFILE_BATCH_SIZE = 50
COST_PER_PROFILE = 0.0023


def verdict(faces: int) -> str:
    if faces == 0:
        return "no face"
    if faces == 1:
        return "single face"
    return f"{faces} faces"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-test face detection on N leads from DB"
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--yes", action="store_true",
                        help="Skip cost confirmation prompt.")
    args = parser.parse_args()

    load_dotenv()
    db = LeadDB("data/leads.db")

    leads = db.get_leads_needing_avatar(limit=args.limit)
    if not leads:
        print("No leads need avatar processing. Nothing to test.")
        return

    estimated_cost = len(leads) * COST_PER_PROFILE
    print(f"\n{'='*60}")
    print(f"Batch avatar test")
    print(f"{'='*60}")
    print(f"Leads to process:     {len(leads)}")
    print(f"Apify (profile):      ~${estimated_cost:.3f}")
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
    all_profiles = []
    for i in range(0, len(usernames), PROFILE_BATCH_SIZE):
        batch = usernames[i:i + PROFILE_BATCH_SIZE]
        print(f"Apify batch {i // PROFILE_BATCH_SIZE + 1}: "
              f"fetching {len(batch)} profiles...")
        run = apify.actor("apify/instagram-profile-scraper").call(run_input={
            "usernames": batch,
        })
        detail = apify.run(run["id"]).get()
        items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
        pipeline.log_run(
            actor_id="apify/instagram-profile-scraper",
            run_id=run["id"], status=run["status"],
            input_params={"batch_size": len(batch), "mode": "test_avatar_batch"},
            items_count=len(items),
            cost_usd=detail.get("usageTotalUsd", 0),
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

    for idx, profile in enumerate(all_profiles, start=1):
        username = profile.get("username") or "?"
        is_private = bool(profile.get("private"))
        if is_private:
            private_count += 1
            print(f"{idx:>4} {username:<30} {'yes':>4} {'-':>6}  {'-':>5}  private, skipped")
            continue

        avatar_url = profile.get("profilePicUrlHD") or profile.get("profilePicUrl")
        uid = profile.get("id") or profile.get("pk")
        avatar_path = download_avatar(
            avatar_url,
            user_id=str(uid) if uid else None,
            username=username,
        )
        if not avatar_path:
            download_fail += 1
            print(f"{idx:>4} {username:<30} {'no':>4} {'-':>6}  {'-':>5}  download failed")
            continue

        t0 = time.perf_counter()
        faces = face_detector.count_faces(avatar_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        detect_times.append(elapsed_ms)
        face_counts[faces] += 1

        db.update_lead_avatar(username, avatar_path, faces)

        print(f"{idx:>4} {username:<30} {'no':>4} {faces:>6}  "
              f"{elapsed_ms:>5.1f}  {verdict(faces)}")

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

    single_face = face_counts.get(1, 0)
    print(f"\nSherlock candidates (single-face): {single_face}/{sum(face_counts.values())}")


if __name__ == "__main__":
    main()
