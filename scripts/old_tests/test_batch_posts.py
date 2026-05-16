"""Fetch posts/reels from found realtors in one batch call.

Loads accounts from the latest related_realtors JSON,
fetches their content for the last 7 days,
saves only posts with 10+ comments.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.apify_client_wrapper import ApifyWrapper
from src.config import load_config
from src.db import LeadDB
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("batch_posts")

MIN_COMMENTS = 10
MAX_AGE_DAYS = 7


def main():
    cfg = load_config()
    db = LeadDB(cfg["db"]["path"])
    pipeline = PipelineLogger(cfg["logging"]["pipeline_log_dir"], "batch_posts")
    apify = ApifyWrapper(cfg, db, pipeline)

    # Load realtors from latest file
    data_dir = Path("data")
    realtor_files = sorted(data_dir.glob("related_realtors_*.json"), reverse=True)
    if not realtor_files:
        log.error("no_realtor_file", msg="Run test_related_realtors.py first")
        return

    with open(realtor_files[0], encoding="utf-8") as f:
        realtors = json.load(f)

    usernames = [r["username"] for r in realtors if not r.get("is_private")]
    log.info("loaded_realtors", count=len(usernames), source=str(realtor_files[0].name))

    # One batch call
    all_posts = apify.get_accounts_posts_batch(usernames, max_age_days=MAX_AGE_DAYS)
    log.info("total_posts_fetched", count=len(all_posts))

    # Filter: 10+ comments
    good_posts = []
    for p in all_posts:
        comments = p.get("commentsCount") or 0
        is_reel = p.get("type") == "Video" or p.get("productType") == "clips"

        if comments >= MIN_COMMENTS:
            good_posts.append({
                "url": p.get("url"),
                "shortcode": p.get("shortCode"),
                "content_type": "reel" if is_reel else "post",
                "owner_username": p.get("ownerUsername"),
                "caption": p.get("caption"),
                "comments_count": comments,
                "likes_count": p.get("likesCount") or 0,
                "views_count": p.get("videoViewCount") or 0,
                "timestamp": p.get("timestamp"),
                "location_name": p.get("locationName"),
                "hashtags": p.get("hashtags") or [],
            })

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = data_dir / f"posts_with_comments_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(good_posts, f, ensure_ascii=False, indent=2)

    by_type = {"post": 0, "reel": 0}
    for item in good_posts:
        by_type[item["content_type"]] += 1

    ps = pipeline.summary()
    print(f"\nFetched {len(all_posts)} posts/reels from {len(usernames)} accounts (last {MAX_AGE_DAYS} days)")
    print(f"With {MIN_COMMENTS}+ comments: {len(good_posts)} ({by_type['post']} posts, {by_type['reel']} reels)")
    print(f"Apify cost: ${ps['total_cost_usd']:.4f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
