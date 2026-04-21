"""Collect fresh posts & reels from all methods, save to JSON for manual relevance review.

Methods:
  1. Hashtag search (posts + reels) — instagram-hashtag-scraper
  2. Realtor account posts — instagram-post-scraper
  3. User search for new realtors — instagram-scraper (universal)

Output: data/collected_content_YYYYMMDD_HHMMSS.json — flat list for review.
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.apify_client_wrapper import ApifyWrapper
from src.config import load_config
from src.db import LeadDB
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("collect")


def is_fresh(post: dict, max_age_days: int) -> bool:
    """Check if post is newer than max_age_days."""
    ts = post.get("timestamp")
    if not ts:
        return False
    try:
        if isinstance(ts, str):
            post_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, (int, float)):
            post_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        return post_dt >= cutoff
    except (ValueError, OSError):
        return False


def normalize_post(post: dict, source_method: str, source_query: str) -> dict:
    """Extract key fields into a flat dict for review."""
    is_reel = post.get("type") == "Video" or post.get("productType") == "clips"
    return {
        "source_method": source_method,
        "source_query": source_query,
        "url": post.get("url"),
        "shortcode": post.get("shortCode"),
        "content_type": "reel" if is_reel else "post",
        "owner_username": post.get("ownerUsername"),
        "owner_full_name": post.get("ownerFullName"),
        "caption": post.get("caption"),
        "comments_count": post.get("commentsCount") or 0,
        "likes_count": post.get("likesCount") or 0,
        "timestamp": post.get("timestamp"),
        "location_name": post.get("locationName"),
        "hashtags": post.get("hashtags") or [],
        "latest_comments": [
            {
                "username": c.get("ownerUsername"),
                "text": c.get("text"),
            }
            for c in (post.get("latestComments") or [])[:5]
        ],
    }


def main():
    cfg = load_config()
    db = LeadDB(cfg["db"]["path"])
    pipeline = PipelineLogger(cfg["logging"]["pipeline_log_dir"], "collect_content")
    apify = ApifyWrapper(cfg, db, pipeline)

    max_age = cfg["filters"]["max_post_age_days"]
    all_content = []

    # ===== METHOD 1: Hashtags (posts + reels) =====
    hashtags = cfg["search"]["hashtags"]
    limit = cfg["apify"]["test_limits"]["results_limit"]

    log.info("method1_hashtag_posts", hashtags=hashtags)
    posts = apify.search_by_hashtag(hashtags, results_type="posts", limit=limit)
    fresh_posts = [p for p in posts if is_fresh(p, max_age)]
    log.info("method1_posts_result", total=len(posts), fresh=len(fresh_posts))

    for p in fresh_posts:
        all_content.append(normalize_post(p, "hashtag_posts", ",".join(hashtags)))

    log.info("method1_hashtag_reels", hashtags=hashtags)
    reels = apify.search_by_hashtag(hashtags, results_type="reels", limit=limit)
    fresh_reels = [r for r in reels if is_fresh(r, max_age)]
    log.info("method1_reels_result", total=len(reels), fresh=len(fresh_reels))

    for r in fresh_reels:
        all_content.append(normalize_post(r, "hashtag_reels", ",".join(hashtags)))

    # ===== METHOD 2: Realtor accounts =====
    realtor_accounts = cfg["search"]["realtor_accounts"]
    log.info("method2_realtor_accounts", accounts=realtor_accounts)

    for username in realtor_accounts:
        log.info("method2_fetching", username=username)
        account_posts = apify.get_account_posts(username, limit=limit)
        fresh = [p for p in account_posts if is_fresh(p, max_age)]
        log.info("method2_result", username=username, total=len(account_posts), fresh=len(fresh))

        for p in fresh:
            all_content.append(normalize_post(p, "realtor_account", username))

    # ===== DEDUP by URL =====
    seen_urls = set()
    deduped = []
    for item in all_content:
        url = item.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(item)
    log.info("dedup", before=len(all_content), after=len(deduped))
    all_content = deduped

    # ===== SAVE JSON for review =====
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path("data") / f"collected_content_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_content, f, ensure_ascii=False, indent=2, default=str)

    # ===== SUMMARY =====
    by_method = {}
    by_type = {"post": 0, "reel": 0}
    for item in all_content:
        m = item["source_method"]
        by_method[m] = by_method.get(m, 0) + 1
        by_type[item["content_type"]] += 1

    log.info(
        "collection_done",
        total=len(all_content),
        by_method=by_method,
        by_type=by_type,
        output_file=str(out_path),
    )

    pipeline_summary = pipeline.summary()
    log.info("cost_summary", **pipeline_summary)

    print(f"\nCollected {len(all_content)} items")
    print(f"  Posts: {by_type['post']}, Reels: {by_type['reel']}")
    print(f"  By method: {by_method}")
    print(f"  Total cost: ${pipeline_summary['total_cost_usd']:.4f}")
    print(f"\nReview file: {out_path}")
    print(f"Pipeline log: {pipeline.file_path}")


if __name__ == "__main__":
    main()
