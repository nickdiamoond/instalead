"""Fetch full profiles for leads and extract contacts from bio/links.

Takes leads with profile_fetched=0 from DB, fetches via profile-scraper in batches,
extracts phone/telegram/whatsapp/email from bio, updates DB.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv

from src.contact_extractor import extract_contacts
from src.db import LeadDB
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("fetch_profiles")

BATCH_SIZE = 50
MAX_COST_USD = 5.0


def extract_media_urls(latest_posts: list) -> list[str]:
    """Extract photo/video URLs from latestPosts."""
    urls = []
    for post in (latest_posts or []):
        # Photo posts: images array or displayUrl
        for img in (post.get("images") or []):
            if img:
                urls.append(img)
        display = post.get("displayUrl")
        if display and display not in urls:
            urls.append(display)
        # Video posts: videoUrl
        video = post.get("videoUrl")
        if video:
            urls.append(video)
    return urls


def main():
    load_dotenv()
    client = ApifyClient(os.environ["APIFY_API_TOKEN"])
    db = LeadDB("data/leads.db")
    pipeline = PipelineLogger("logs", "fetch_profiles")

    # Get leads that need profile fetching (public only)
    leads = db.get_leads_without_profile(limit=1000)
    if not leads:
        print("No leads to fetch profiles for.")
        return

    usernames = [l["username"] for l in leads]
    log.info("leads_to_fetch", count=len(usernames))

    total_cost = 0
    contacts_found = 0
    profiles_fetched = 0
    batch_num = 0

    for i in range(0, len(usernames), BATCH_SIZE):
        batch = usernames[i:i + BATCH_SIZE]
        batch_num += 1

        # Budget check
        if total_cost >= MAX_COST_USD:
            log.warning("budget_exceeded", cost=total_cost, limit=MAX_COST_USD)
            break

        log.info("batch_start", batch=batch_num, size=len(batch), total_cost=round(total_cost, 4))

        run = client.actor("apify/instagram-profile-scraper").call(run_input={
            "usernames": batch,
        })

        detail = client.run(run["id"]).get()
        cost = detail.get("usageTotalUsd", 0)
        total_cost += cost
        duration_ms = detail.get("stats", {}).get("durationMillis", 0)

        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        log.info("batch_done", batch=batch_num, items=len(items), cost=cost)

        # Log to pipeline
        pipeline.log_run(
            actor_id="apify/instagram-profile-scraper",
            run_id=run["id"],
            status=run["status"],
            input_params={"usernames_count": len(batch)},
            items_count=len(items),
            cost_usd=cost,
            duration_ms=duration_ms,
        )

        for p in items:
            username = p.get("username")
            if not username:
                continue

            # Extract media URLs from latest posts
            media_urls = extract_media_urls(p.get("latestPosts"))

            # Update profile in DB
            db.update_lead_profile(
                username=username,
                full_name=p.get("fullName"),
                biography=p.get("biography"),
                profile_pic_url_hd=p.get("profilePicUrlHD"),
                is_private=1 if p.get("private") else 0,
                is_verified=1 if p.get("verified") else 0,
                is_business=1 if p.get("isBusinessAccount") else 0,
                business_category=p.get("businessCategoryName"),
                followers_count=p.get("followersCount"),
                following_count=p.get("followsCount"),
                posts_count=p.get("postsCount"),
                external_url=p.get("externalUrl"),
                latest_media_urls=json.dumps(media_urls[:20], ensure_ascii=False) if media_urls else None,
            )
            profiles_fetched += 1

            # Extract contacts from bio and URLs
            contacts = extract_contacts(
                bio=p.get("biography"),
                external_url=p.get("externalUrl"),
                external_urls=p.get("externalUrls"),
            )

            has_any = any(v for v in contacts.values())
            if has_any:
                db.update_lead_contacts(username=username, **{k: v for k, v in contacts.items() if v})
                contacts_found += 1
                log.info(
                    "contacts_extracted",
                    username=username,
                    **{k: v for k, v in contacts.items() if v},
                )

    # Summary
    ps = pipeline.summary()
    db_stats = db.get_stats()

    print(f"\nProfiles fetched: {profiles_fetched}")
    print(f"Contacts found from bio: {contacts_found}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"\nDB stats:")
    print(f"  Leads total: {db_stats['leads_total']}")
    print(f"  With profile: {db_stats['leads_with_profile']}")
    print(f"  With contacts: {db_stats['leads_with_contacts']}")
    print(f"\nPipeline: {pipeline.file_path}")


if __name__ == "__main__":
    main()
