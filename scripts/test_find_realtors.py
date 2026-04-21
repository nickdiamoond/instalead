"""Find realtor accounts by keyword search, then get their fresh posts/reels.

Flow:
  1. Search users by keywords ("квартиры спб", "новостройки спб", etc.)
  2. Get their recent posts/reels (14 days)
  3. Save everything to JSON for relevance review
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
log = get_logger("find_realtors")

SEARCH_QUERIES = [
    "квартиры спб",
    "новостройки спб",
    "недвижимость петербург",
    "купить квартиру спб",
    "риелтор спб",
]


def is_fresh(post: dict, max_age_days: int) -> bool:
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


def main():
    cfg = load_config()
    db = LeadDB(cfg["db"]["path"])
    pipeline = PipelineLogger(cfg["logging"]["pipeline_log_dir"], "find_realtors")
    apify = ApifyWrapper(cfg, db, pipeline)

    max_age = cfg["filters"]["max_post_age_days"]
    limit_per_query = cfg["apify"]["test_limits"]["search_limit"]
    posts_limit = cfg["apify"]["test_limits"]["results_limit"]

    # ===== STEP 1: Find realtor accounts =====
    all_accounts = {}  # username -> profile dict (dedup)

    for query in SEARCH_QUERIES:
        log.info("searching_users", query=query, limit=limit_per_query)
        profiles = apify.search_users(query, limit=limit_per_query)

        for p in profiles:
            username = p.get("username")
            if username and username not in all_accounts:
                all_accounts[username] = {
                    "username": username,
                    "full_name": p.get("fullName"),
                    "biography": p.get("biography"),
                    "followers_count": p.get("followersCount"),
                    "posts_count": p.get("postsCount"),
                    "is_private": p.get("private"),
                    "is_verified": p.get("verified"),
                    "is_business": p.get("isBusinessAccount"),
                    "business_category": p.get("businessCategoryName"),
                    "external_url": p.get("externalUrl"),
                    "profile_pic_url": p.get("profilePicUrlHD") or p.get("profilePicUrl"),
                    "found_by_query": query,
                }
                log.info(
                    "account_found",
                    username=username,
                    followers=p.get("followersCount"),
                    posts=p.get("postsCount"),
                    bio=str(p.get("biography", ""))[:100],
                    query=query,
                )

    log.info("total_unique_accounts", count=len(all_accounts))

    # Save accounts
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    accounts_path = Path("data") / f"found_realtors_{ts}.json"
    with open(accounts_path, "w", encoding="utf-8") as f:
        json.dump(list(all_accounts.values()), f, ensure_ascii=False, indent=2)
    log.info("accounts_saved", path=str(accounts_path))

    # ===== STEP 2: Get posts/reels from found accounts =====
    all_content = []

    for username, account in all_accounts.items():
        if account.get("is_private"):
            log.info("skipping_private", username=username)
            continue

        log.info("fetching_posts", username=username)
        posts = apify.get_account_posts(username, limit=posts_limit)
        fresh = [p for p in posts if is_fresh(p, max_age)]

        for p in fresh:
            is_reel = p.get("type") == "Video" or p.get("productType") == "clips"
            all_content.append({
                "source_method": "user_search",
                "source_query": account["found_by_query"],
                "source_account": username,
                "url": p.get("url"),
                "shortcode": p.get("shortCode"),
                "content_type": "reel" if is_reel else "post",
                "caption": p.get("caption"),
                "comments_count": p.get("commentsCount") or 0,
                "likes_count": p.get("likesCount") or 0,
                "timestamp": p.get("timestamp"),
                "location_name": p.get("locationName"),
                "hashtags": p.get("hashtags") or [],
            })

        log.info(
            "account_posts",
            username=username,
            total=len(posts),
            fresh=len(fresh),
            reels=sum(1 for p in fresh if p.get("type") == "Video" or p.get("productType") == "clips"),
        )

    # ===== SAVE =====
    content_path = Path("data") / f"realtor_content_{ts}.json"
    with open(content_path, "w", encoding="utf-8") as f:
        json.dump(all_content, f, ensure_ascii=False, indent=2)

    # ===== SUMMARY =====
    by_type = {"post": 0, "reel": 0}
    for item in all_content:
        by_type[item["content_type"]] += 1

    with_comments = sum(1 for item in all_content if item["comments_count"] >= 2)

    log.info(
        "done",
        accounts_found=len(all_accounts),
        total_content=len(all_content),
        posts=by_type["post"],
        reels=by_type["reel"],
        with_2plus_comments=with_comments,
    )

    ps = pipeline.summary()
    log.info("cost", **ps)

    print(f"\nFound {len(all_accounts)} realtor accounts")
    print(f"Collected {len(all_content)} fresh items ({by_type['post']} posts, {by_type['reel']} reels)")
    print(f"Items with 2+ comments: {with_comments}")
    print(f"Total cost: ${ps['total_cost_usd']:.4f}")
    print(f"\nAccounts: {accounts_path}")
    print(f"Content:  {content_path}")
    print(f"Pipeline: {pipeline.file_path}")


if __name__ == "__main__":
    main()
