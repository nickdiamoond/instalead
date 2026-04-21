"""Fetch comments for scored relevant posts via louisdeconinck/instagram-comments-scraper.

Takes the latest posts_scored JSON, selects posts with relevance="relevant" and CTA,
fetches comments in one batch call, deduplicates, saves leads.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv

from src.db import LeadDB
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("fetch_comments")

COST_PER_COMMENT = 0.0005  # approximate, based on tests


def main():
    load_dotenv()
    client = ApifyClient(os.environ["APIFY_API_TOKEN"])
    db = LeadDB("data/leads.db")
    pipeline = PipelineLogger("logs", "fetch_comments")

    # Load latest scored posts
    data_dir = Path("data")
    scored_files = sorted(data_dir.glob("posts_scored_*.json"), reverse=True)
    if not scored_files:
        log.error("no_scored_file", msg="Run test_score_posts.py first")
        return

    with open(scored_files[0], encoding="utf-8") as f:
        posts = json.load(f)

    # Select relevant posts with CTA=comment
    target_posts = [
        p for p in posts
        if p.get("relevance") == "relevant"
        and p.get("call_to_action_type") == "comment"
        and p.get("url")
    ]

    if not target_posts:
        log.warning("no_target_posts")
        return

    total_comments = sum(p["comments_count"] for p in target_posts)
    estimated_cost = total_comments * COST_PER_COMMENT

    log.info(
        "selected_posts",
        total_scored=len(posts),
        target=len(target_posts),
        total_comments=total_comments,
    )

    print(f"\n{'='*50}")
    print(f"Posts to process:     {len(target_posts)}")
    print(f"Total comments:      {total_comments}")
    print(f"Estimated cost:      ${estimated_cost:.2f}")
    print(f"{'='*50}")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    # Build URL list
    urls = [p["url"] for p in target_posts]

    # One batch call — no maxComments limit, get everything
    log.info("fetching_comments", urls_count=len(urls))
    run = client.actor("louisdeconinck/instagram-comments-scraper").call(run_input={
        "urls": urls,
    })

    run_id = run["id"]
    status = run["status"]
    log.info("run_finished", run_id=run_id, status=status)

    # Get cost
    detail = client.run(run_id).get()
    cost_usd = detail.get("usageTotalUsd", 0)
    duration_ms = detail.get("stats", {}).get("durationMillis", 0)

    # Get items
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    log.info("raw_items", count=len(items), cost=cost_usd, duration_s=round(duration_ms / 1000, 1))

    # Log to pipeline
    pipeline.log_run(
        actor_id="louisdeconinck/instagram-comments-scraper",
        run_id=run_id,
        status=status,
        input_params={"urls": urls, "maxComments": MAX_COMMENTS_PER_POST},
        items_count=len(items),
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        dataset_id=run.get("defaultDatasetId"),
    )

    # Dedup by pk
    unique = {}
    for c in items:
        pk = str(c.get("pk", ""))
        if pk and pk not in unique:
            unique[pk] = c

    log.info("dedup", raw=len(items), unique=len(unique), dupes=len(items) - len(unique))

    # Build media_id -> (post_url, shortcode) mapping via shortcode conversion
    # media_id from JSON loses precision (JS float64), so we convert shortcode -> id
    # and build a fuzzy match (within ±1000 tolerance)
    CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

    def shortcode_to_id(sc: str) -> int:
        mid = 0
        for ch in sc:
            mid = mid * 64 + CHARSET.index(ch)
        return mid

    # shortcode -> (real_media_id, post_url)
    post_lookup = {}
    for p in target_posts:
        sc = p.get("shortcode")
        if sc:
            real_id = shortcode_to_id(sc)
            post_lookup[real_id] = (p["url"], sc)

    # Map imprecise media_id from comments to post_url
    media_to_post: dict[str, tuple[str, str]] = {}  # media_id_str -> (url, shortcode)
    for c in unique.values():
        mid = c.get("media_id")
        if not mid:
            continue
        mid_str = str(mid)
        if mid_str in media_to_post:
            continue
        # Fuzzy match: JS float64 loses ~100 precision on large ints
        for real_id, (url, sc) in post_lookup.items():
            if abs(real_id - mid) < 1000:
                media_to_post[mid_str] = (url, sc)
                break

    log.info("media_mapping", mapped=len(media_to_post), total_media_ids=len(set(
        str(c.get("media_id", "")) for c in unique.values()
    )))

    # Extract leads
    leads = []
    seen_usernames = set()
    for c in unique.values():
        user = c.get("user", {})
        username = user.get("username")
        if not username or username in seen_usernames:
            continue
        seen_usernames.add(username)

        mid_str = str(c.get("media_id", ""))
        post_info = media_to_post.get(mid_str)
        post_url = post_info[0] if post_info else None
        post_shortcode = post_info[1] if post_info else None
        uid = str(user.get("pk", ""))

        # Save lead to DB
        db.add_lead_account(
            username=username,
            user_id=uid,
            full_name=user.get("full_name", ""),
            profile_pic_url=user.get("profile_pic_url", ""),
            is_private=1 if user.get("is_private") else 0,
            is_verified=1 if user.get("is_verified") else 0,
        )

        # Link lead to post
        if post_url:
            db.add_lead_post_link(
                username=username,
                post_url=post_url,
                user_id=uid,
                post_shortcode=post_shortcode,
                comment_text=c.get("text", "")[:500],
                comment_at=str(c.get("created_at_utc", "")),
            )

        leads.append({
            "username": username,
            "full_name": user.get("full_name", ""),
            "user_id": uid,
            "is_private": user.get("is_private", False),
            "profile_pic_url": user.get("profile_pic_url", ""),
            "comment_text": c.get("text", ""),
            "comment_created_at": c.get("created_at_utc"),
            "post_url": post_url,
            "post_shortcode": post_shortcode,
        })

    # Save leads JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    leads_path = data_dir / f"leads_{ts}.json"
    with open(leads_path, "w", encoding="utf-8") as f:
        json.dump(leads, f, ensure_ascii=False, indent=2)

    # Save raw comments
    raw_path = data_dir / f"comments_raw_{ts}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(list(unique.values()), f, ensure_ascii=False, indent=2, default=str)

    # Summary
    private_count = sum(1 for l in leads if l["is_private"])
    public_count = len(leads) - private_count

    ps = pipeline.summary()
    db_stats = db.get_stats()

    print(f"\nFetched comments from {len(target_posts)} posts")
    print(f"Raw comments: {len(items)}, unique: {len(unique)}")
    print(f"Unique leads: {len(leads)} ({public_count} public, {private_count} private)")
    print(f"Cost: ${cost_usd:.4f}")
    print(f"DB total leads: {db_stats['leads_total']}")
    print(f"\nLeads: {leads_path}")
    print(f"Raw comments: {raw_path}")
    print(f"Pipeline: {pipeline.file_path}")


if __name__ == "__main__":
    main()
