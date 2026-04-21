"""Test script: get comments for a post/reel and extract commenter usernames.

This is the most business-critical script — commenter usernames are the leads.
Pass a post URL as argument or it will use a hardcoded example.
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
log = get_logger("test_comments")

# Put a real post URL here for testing
DEFAULT_POST_URL = "https://www.instagram.com/p/EXAMPLE_SHORTCODE/"


def main():
    post_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_POST_URL

    if "EXAMPLE" in post_url:
        log.error(
            "no_post_url",
            msg="Pass a real Instagram post URL as argument: "
                "python scripts/test_apify_comments.py https://www.instagram.com/p/XXX/",
        )
        return

    cfg = load_config()
    db = LeadDB(cfg["db"]["path"])
    pipeline = PipelineLogger(cfg["logging"]["pipeline_log_dir"], "test_comments")

    apify = ApifyWrapper(cfg, db, pipeline)

    log.info("fetching_comments", post_url=post_url)
    comments = apify.get_comments(post_url)

    lead_usernames = []
    for c in comments:
        username = c.get("ownerUsername")
        if not username:
            continue

        is_new = db.add_lead_account(
            username=username,
            user_id=str(c.get("ownerId", "")),
            profile_pic_url=c.get("ownerProfilePicUrl"),
            source_post_id=post_url,
            comment_text=str(c.get("text", ""))[:500],
            comment_timestamp=c.get("timestamp"),
        )

        lead_usernames.append(username)
        log.info(
            "commenter",
            username=username,
            is_new=is_new,
            text=str(c.get("text", ""))[:100],
            timestamp=c.get("timestamp"),
        )

    log.info(
        "comments_done",
        total_comments=len(comments),
        unique_leads=len(set(lead_usernames)),
    )

    summary = pipeline.summary()
    log.info("session_summary", **summary)
    db_stats = db.get_stats()
    log.info("db_stats", **db_stats)
    print(f"\nPipeline log saved to: {pipeline.file_path}")


if __name__ == "__main__":
    main()
