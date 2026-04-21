"""Test script: get posts/reels for a specific Instagram account via Apify.

Tests Method 2 — working from a curated list of realtor accounts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.db import LeadDB
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger
from src.apify_client_wrapper import ApifyWrapper

setup_logging()
log = get_logger("test_account_posts")


def main():
    cfg = load_config()
    db = LeadDB(cfg["db"]["path"])
    pipeline = PipelineLogger(cfg["logging"]["pipeline_log_dir"], "test_account_posts")

    apify = ApifyWrapper(cfg, db, pipeline)

    accounts = cfg["search"]["realtor_accounts"]
    if not accounts:
        log.error("no_realtor_accounts", msg="Add realtor accounts to config.yaml")
        return

    for username in accounts:
        log.info("fetching_posts", username=username)
        posts = apify.get_account_posts(username)

        for p in posts:
            is_video = p.get("type") == "Video" or p.get("videoUrl") is not None
            log.info(
                "post",
                shortcode=p.get("shortCode"),
                type="reel/video" if is_video else p.get("type", "unknown"),
                comments=p.get("commentsCount"),
                likes=p.get("likesCount"),
                timestamp=p.get("timestamp"),
                caption=str(p.get("caption", ""))[:80],
                location=p.get("locationName"),
            )

            # Show which posts pass filters
            comments_count = p.get("commentsCount") or 0
            likes_count = p.get("likesCount") or 0
            passes_filter = comments_count >= cfg["filters"]["min_comments"]
            log.info(
                "filter_check",
                shortcode=p.get("shortCode"),
                passes=passes_filter,
                comments=comments_count,
                min_required=cfg["filters"]["min_comments"],
            )

        log.info("account_done", username=username, posts_count=len(posts))

    summary = pipeline.summary()
    log.info("session_summary", **summary)
    print(f"\nPipeline log saved to: {pipeline.file_path}")


if __name__ == "__main__":
    main()
