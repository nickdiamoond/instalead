"""Backfill avatars + face detection for existing leads.

Two modes:
  --refetch (default): re-run apify/instagram-profile-scraper to get
                       fresh avatar URLs (Instagram CDN URLs expire in
                       1-2 days), then download + detect faces.
  --no-refetch:        try existing profile_pic_url_hd values stored
                       in the DB; most will 403 if older than ~2 days.

Usage:
    python scripts/backfill_avatars.py              # refetch
    python scripts/backfill_avatars.py --no-refetch # try stale URLs
    python scripts/backfill_avatars.py --limit 100  # cap number of leads
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv

from src.avatar_downloader import download_avatar
from src.config import load_config
from src.db import LeadDB
from src.face_embedder import FaceEmbedder, make_face_embedder
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("backfill_avatars")

PROFILE_BATCH_SIZE = 50
COST_PER_PROFILE = 0.0023


def process_profile(db: LeadDB, face_embedder: FaceEmbedder, profile: dict) -> tuple[bool, int]:
    """Download avatar + detect faces. Returns (success, faces_count)."""
    username = profile.get("username")
    if not username:
        return False, 0

    avatar_url = profile.get("profilePicUrlHD") or profile.get("profilePicUrl")
    uid = profile.get("id") or profile.get("pk")

    if profile.get("private"):
        return False, 0

    avatar_path = download_avatar(
        avatar_url,
        user_id=str(uid) if uid else None,
        username=username,
    )
    if not avatar_path:
        return False, 0

    faces_count = face_embedder.count_faces(avatar_path)
    db.update_lead_avatar(username, avatar_path, faces_count)
    return True, faces_count


def run_refetch(
    db: LeadDB,
    apify: ApifyClient,
    pipeline: PipelineLogger,
    face_embedder: FaceEmbedder,
    leads: list[dict],
) -> None:
    usernames = [l["username"] for l in leads]
    estimated_cost = len(usernames) * COST_PER_PROFILE

    print(f"\n{'='*50}")
    print(f"Backfill avatars (refetch mode)")
    print(f"Leads to process:     {len(usernames)}")
    print(f"Estimated cost:       ${estimated_cost:.2f}")
    print(f"{'='*50}")
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    downloaded = 0
    single_face = 0

    for i in range(0, len(usernames), PROFILE_BATCH_SIZE):
        batch = usernames[i:i + PROFILE_BATCH_SIZE]
        log.info("refetch_batch", start=i, size=len(batch))

        run = apify.actor("apify/instagram-profile-scraper").call(run_input={
            "usernames": batch,
        })
        detail = apify.run(run["id"]).get()
        items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())

        pipeline.log_run(
            actor_id="apify/instagram-profile-scraper",
            run_id=run["id"], status=run["status"],
            input_params={"batch_size": len(batch), "mode": "backfill_avatars"},
            items_count=len(items),
            cost_usd=detail.get("usageTotalUsd", 0),
            duration_ms=detail.get("stats", {}).get("durationMillis"),
        )

        for p in items:
            ok, faces = process_profile(db, face_embedder, p)
            if ok:
                downloaded += 1
                if faces == 1:
                    single_face += 1

    log.info("refetch_done", downloaded=downloaded, single_face=single_face)


def run_stale(
    db: LeadDB,
    face_embedder: FaceEmbedder,
    leads: list[dict],
) -> None:
    print(f"\nTrying {len(leads)} stale CDN URLs (expect most to 403)...")
    downloaded = 0
    single_face = 0

    for lead in leads:
        avatar_url = lead.get("profile_pic_url_hd") or lead.get("profile_pic_url")
        if not avatar_url:
            continue
        avatar_path = download_avatar(
            avatar_url,
            user_id=lead.get("user_id"),
            username=lead.get("username"),
        )
        if not avatar_path:
            continue
        faces_count = face_embedder.count_faces(avatar_path)
        db.update_lead_avatar(lead["username"], avatar_path, faces_count)
        downloaded += 1
        if faces_count == 1:
            single_face += 1

    log.info("stale_done", downloaded=downloaded, single_face=single_face)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill avatars + face detection")
    parser.add_argument(
        "--no-refetch",
        action="store_true",
        help="Don't call Apify; just try stale URLs already in DB.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Maximum leads to process (default: 10000).",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()

    db = LeadDB("data/leads.db")
    # Avatar-only script — single 320x320 SCRFD instance is enough.
    face_embedder = make_face_embedder(cfg, kind="avatar")

    if args.no_refetch:
        leads = db.get_leads_needing_avatar(limit=args.limit)
        if not leads:
            print("No leads need avatar backfill.")
            return
        run_stale(db, face_embedder, leads)
    else:
        leads_dicts = db.get_leads_needing_avatar(limit=args.limit)
        if not leads_dicts:
            print("No leads need avatar backfill.")
            return
        apify = ApifyClient(os.environ["APIFY_API_TOKEN"])
        pipeline = PipelineLogger("logs", "backfill_avatars")
        run_refetch(db, apify, pipeline, face_embedder, leads_dicts)

    face_embedder.close()

    stats = db.get_stats()
    print(f"\nLeads with avatar:    {stats['leads_with_avatar']}")
    print(f"Leads single-face:    {stats['leads_with_single_face']}")


if __name__ == "__main__":
    main()
