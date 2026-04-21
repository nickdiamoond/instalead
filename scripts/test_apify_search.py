"""Test script: search Instagram posts by hashtags via instagram-hashtag-scraper.

Uses the dedicated hashtag scraper which returns ACTUAL POSTS (not hashtag URLs).
Config is set for SPB real estate hashtags.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.apify_client_wrapper import ApifyWrapper
from src.config import load_config
from src.db import LeadDB
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("test_search")


def main():
    cfg = load_config()
    db = LeadDB(cfg["db"]["path"])
    pipeline = PipelineLogger(cfg["logging"]["pipeline_log_dir"], "test_hashtag_search")
    apify = ApifyWrapper(cfg, db, pipeline)

    hashtags = cfg["search"]["hashtags"]
    limit = cfg["apify"]["test_limits"]["results_limit"]

    log.info("starting_hashtag_search", hashtags=hashtags, limit_per_hashtag=limit)

    # Search posts by hashtags (one call, all hashtags at once)
    posts = apify.search_by_hashtag(hashtags, results_type="posts", limit=limit)

    log.info("total_posts_found", count=len(posts))

    # Analyze results
    reels_count = 0
    image_count = 0
    qualifying = 0

    for p in posts:
        is_reel = p.get("type") == "Video" or p.get("productType") == "clips"
        if is_reel:
            reels_count += 1
        else:
            image_count += 1

        comments = p.get("commentsCount") or 0
        likes = p.get("likesCount") or 0
        passes_filter = comments >= cfg["filters"]["min_comments"]

        if passes_filter:
            qualifying += 1

        log.info(
            "post",
            url=p.get("url"),
            type="reel" if is_reel else p.get("type", "unknown"),
            comments=comments,
            likes=likes,
            owner=p.get("ownerUsername"),
            location=p.get("locationName"),
            passes_filter=passes_filter,
            caption=str(p.get("caption", ""))[:100],
        )

        # Show latest comments if available (potential leads)
        latest = p.get("latestComments") or []
        for c in latest[:3]:
            log.info(
                "  latest_comment",
                username=c.get("ownerUsername"),
                text=str(c.get("text", ""))[:80],
            )

    log.info(
        "search_summary",
        total_posts=len(posts),
        reels=reels_count,
        images=image_count,
        qualifying_posts=qualifying,
        min_comments_filter=cfg["filters"]["min_comments"],
    )

    # Log first post's ALL keys to understand schema
    if posts:
        log.info("first_post_keys", keys=sorted(posts[0].keys()))

    summary = pipeline.summary()
    log.info("session_summary", **summary)
    print(f"\nPipeline log saved to: {pipeline.file_path}")


if __name__ == "__main__":
    main()
