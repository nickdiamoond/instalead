"""Daily lead collection pipeline.

Steps:
  1. Fetch recent posts/reels from tracked realtors
  2. Score new posts via DeepSeek (relevance + CTA)
  3. Fetch comments for relevant posts (new + grown)
  4. Fetch profiles for new leads, extract contacts from bio

Uses DB for deduplication — safe to run repeatedly.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv
from openai import OpenAI

from src.avatar_downloader import (
    cleanup_lead_photos,
    download_avatar,
    download_post_photos,
)
from src.config import load_config
from src.contact_extractor import extract_contacts
from src.db import LeadDB
from src.face_embedder import FaceEmbedder
from src.face_leader import resolve_face_leader
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger

setup_logging()
log = get_logger("pipeline")

POSTS_MAX_AGE_DAYS = 7
MIN_COMMENTS = 10
COMMENTS_GROWTH_PCT = 5.0
PROFILE_BATCH_SIZE = 50
COST_PER_COMMENT = 0.0005

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

RELEVANCE_PROMPT = """\
Ты анализируешь описание поста/рилса из Instagram от риелтора. Определи:

1. is_real_estate — пост про недвижимость (продажа/покупка квартир, обзоры ЖК, ипотека)?
2. has_call_to_action — есть ли призыв заинтересованным покупателям писать в комментарии/директ?
3. call_to_action_type — тип призыва: "comment" / "direct" / "link" / "none"

Если описание слишком короткое или непонятное — верни is_real_estate: null.

Ответь ТОЛЬКО валидным JSON без markdown:
{"is_real_estate": true/false/null, "has_call_to_action": true/false, "call_to_action_type": "comment"|"direct"|"link"|"none"}
"""


def shortcode_to_id(sc: str) -> int:
    mid = 0
    for ch in sc:
        mid = mid * 64 + CHARSET.index(ch)
    return mid


def caption_is_empty(caption: str | None) -> bool:
    if not caption:
        return True
    without_hashtags = " ".join(w for w in caption.strip().split() if not w.startswith("#"))
    return len(without_hashtags.strip()) < 15


def _pick_post_images(
    latest_posts: list[dict] | None,
    limit: int,
    *,
    skip_videos: bool = True,
) -> list[str]:
    """Pick at most one representative image URL from each of the first
    ``limit`` posts in ``latestPosts``.

    We intentionally take one image per post (not every carousel slide)
    so that clustering counts *distinct post appearances* — if the same
    person posts a 10-slide carousel of themselves, it shouldn't drown
    out four separate posts showing someone else.

    Preference per post:
      1. ``images[0]`` — carousel cover / first slide (always a photo).
      2. ``displayUrl`` — the single photo of a photo post.
      3. Otherwise skip (videos, empties).
    """
    if not latest_posts:
        return []

    urls: list[str] = []
    for post in latest_posts[:limit]:
        images = post.get("images") or []
        if images and images[0]:
            urls.append(images[0])
            continue
        display_url = post.get("displayUrl")
        video_url = post.get("videoUrl")
        if not display_url:
            continue
        if skip_videos and video_url:
            # Pure video post: displayUrl is just a cover frame, often
            # low-quality / motion-blurred. Skip.
            continue
        urls.append(display_url)
    return urls


def score_caption(client: OpenAI, caption: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": RELEVANCE_PROMPT},
                {"role": "user", "content": caption[:2000]},
            ],
            temperature=0,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content
        if not raw:
            return {"error": "empty"}
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


def main():
    load_dotenv()
    cfg = load_config()
    apify = ApifyClient(os.environ["APIFY_API_TOKEN"])
    deepseek = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    db = LeadDB("data/leads.db")
    pipeline = PipelineLogger("logs", "pipeline")

    fd_cfg = cfg.get("face_detection") or {}
    min_det_score = float(fd_cfg.get("min_det_score", 0.7))
    face_embedder = FaceEmbedder(min_det_score=min_det_score)

    fb_cfg = cfg.get("face_fallback") or {}
    fb_limit = int(fb_cfg.get("latest_posts_limit", 5))
    fb_min_cluster = int(fb_cfg.get("min_cluster_size", 2))
    fb_threshold = float(fb_cfg.get("cluster_threshold", 0.5))
    fb_skip_videos = bool(fb_cfg.get("skip_videos", True))
    fb_keep_photos = bool(fb_cfg.get("keep_photos", False))

    stats_before = db.get_stats()
    log.info("pipeline_start", **stats_before)

    # ============================================================
    # STEP 1: Fetch posts from tracked realtors
    # ============================================================
    realtors = db.get_active_realtors()
    if not realtors:
        log.error("no_realtors", msg="Add realtors to tracked_realtors table first")
        return

    log.info("step1_fetch_posts", realtors=len(realtors), max_age_days=POSTS_MAX_AGE_DAYS)

    run = apify.actor("apify/instagram-post-scraper").call(run_input={
        "username": realtors,
        "resultsLimit": 20,
        "onlyPostsNewerThan": f"{POSTS_MAX_AGE_DAYS} days",
        "dataDetailLevel": "basicData",
        "proxy": {"useApifyProxy": True},
    })
    detail = apify.run(run["id"]).get()
    all_posts = list(apify.dataset(run["defaultDatasetId"]).iterate_items())

    pipeline.log_run(
        actor_id="apify/instagram-post-scraper",
        run_id=run["id"], status=run["status"],
        input_params={"realtors": len(realtors)},
        items_count=len(all_posts),
        cost_usd=detail.get("usageTotalUsd", 0),
        duration_ms=detail.get("stats", {}).get("durationMillis"),
    )

    # Filter by min comments and register in DB
    new_posts = 0
    updated_posts = 0
    for p in all_posts:
        shortcode = p.get("shortCode", "")
        comments_count = p.get("commentsCount") or 0
        if comments_count < MIN_COMMENTS:
            continue

        is_reel = p.get("type") == "Video" or p.get("productType") == "clips"
        existing = db.get_post(shortcode)

        if existing:
            # Update comments_count if changed
            if comments_count != (existing.get("comments_count") or 0):
                db.upsert_post(shortcode, comments_count=comments_count)
                updated_posts += 1
        else:
            db.upsert_post(
                shortcode,
                post_url=p.get("url", ""),
                shortcode=shortcode,
                owner_username=p.get("ownerUsername"),
                comments_count=comments_count,
                likes_count=p.get("likesCount") or 0,
                views_count=p.get("videoViewCount") or 0,
                post_type="reel" if is_reel else "post",
                caption=p.get("caption"),
                timestamp=p.get("timestamp"),
            )
            new_posts += 1

    log.info("step1_done", total_posts=len(all_posts), new=new_posts, updated=updated_posts,
             cost=detail.get("usageTotalUsd", 0))

    # ============================================================
    # STEP 2: Score new posts via DeepSeek
    # ============================================================
    with db._conn() as conn:
        unscored = conn.execute(
            "SELECT post_id, caption FROM processed_posts WHERE relevance IS NULL"
        ).fetchall()
        unscored = [dict(r) for r in unscored]

    log.info("step2_score_posts", count=len(unscored))

    for p in unscored:
        caption = p.get("caption")
        if caption_is_empty(caption):
            db.upsert_post(p["post_id"], relevance="unknown", has_cta=0, cta_type="none")
            continue

        score = score_caption(deepseek, caption)
        if "error" in score:
            db.upsert_post(p["post_id"], relevance="unknown", has_cta=0, cta_type="none")
        elif score.get("is_real_estate") is None:
            db.upsert_post(p["post_id"], relevance="unknown",
                           has_cta=1 if score.get("has_call_to_action") else 0,
                           cta_type=score.get("call_to_action_type", "none"))
        else:
            relevance = "relevant" if score["is_real_estate"] else "irrelevant"
            db.upsert_post(p["post_id"], relevance=relevance,
                           has_cta=1 if score.get("has_call_to_action") else 0,
                           cta_type=score.get("call_to_action_type", "none"))

    log.info("step2_done", scored=len(unscored))

    # ============================================================
    # STEP 3: Fetch comments
    # ============================================================
    posts_to_scan = db.get_posts_needing_comments(min_growth_pct=COMMENTS_GROWTH_PCT)

    if not posts_to_scan:
        log.info("step3_no_posts_to_scan")
    else:
        total_comments = sum(p.get("comments_count") or 0 for p in posts_to_scan)
        estimated_cost = total_comments * COST_PER_COMMENT

        log.info("step3_fetch_comments", posts=len(posts_to_scan),
                 total_comments=total_comments, estimated_cost=round(estimated_cost, 2))

        print(f"\n{'='*50}")
        print(f"Step 3: Fetch comments")
        print(f"Posts to scan:        {len(posts_to_scan)}")
        print(f"Estimated comments:   {total_comments}")
        print(f"Estimated cost:       ${estimated_cost:.2f}")
        print(f"{'='*50}")
        confirm = input("Proceed? (y/n): ").strip().lower()

        if confirm == "y":
            urls = [p["post_url"] for p in posts_to_scan if p.get("post_url")]

            run = apify.actor("louisdeconinck/instagram-comments-scraper").call(run_input={
                "urls": urls,
            })
            detail = apify.run(run["id"]).get()
            items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
            cost = detail.get("usageTotalUsd", 0)

            pipeline.log_run(
                actor_id="louisdeconinck/instagram-comments-scraper",
                run_id=run["id"], status=run["status"],
                input_params={"urls_count": len(urls)},
                items_count=len(items),
                cost_usd=cost,
                duration_ms=detail.get("stats", {}).get("durationMillis"),
            )

            # Dedup by pk
            unique = {}
            for c in items:
                pk = str(c.get("pk", ""))
                if pk and pk not in unique:
                    unique[pk] = c

            # Build media_id -> post mapping via shortcode
            post_lookup = {}
            for p in posts_to_scan:
                sc = p.get("shortcode")
                if sc:
                    post_lookup[shortcode_to_id(sc)] = (p["post_url"], sc)

            media_to_post = {}
            for c in unique.values():
                mid = c.get("media_id")
                if not mid:
                    continue
                mid_str = str(mid)
                if mid_str in media_to_post:
                    continue
                for real_id, (url, sc) in post_lookup.items():
                    if abs(real_id - mid) < 1000:
                        media_to_post[mid_str] = (url, sc)
                        break

            # Save leads
            new_leads = 0
            for c in unique.values():
                user = c.get("user", {})
                username = user.get("username")
                if not username:
                    continue
                uid = str(user.get("pk", ""))

                is_new = db.add_lead_account(
                    username=username,
                    user_id=uid,
                    full_name=user.get("full_name", ""),
                    profile_pic_url=user.get("profile_pic_url", ""),
                    is_private=1 if user.get("is_private") else 0,
                    is_verified=1 if user.get("is_verified") else 0,
                )
                if is_new:
                    new_leads += 1

                mid_str = str(c.get("media_id", ""))
                post_info = media_to_post.get(mid_str)
                if post_info:
                    db.add_lead_post_link(
                        username=username,
                        post_url=post_info[0],
                        user_id=uid,
                        post_shortcode=post_info[1],
                        comment_text=c.get("text", "")[:500],
                        comment_at=str(c.get("created_at_utc", "")),
                    )

            # Mark posts as scanned
            for p in posts_to_scan:
                db.mark_post_comments_scanned(
                    p["post_id"],
                    p.get("comments_count") or 0,
                )

            log.info("step3_done", raw=len(items), unique=len(unique),
                     new_leads=new_leads, cost=cost)
        else:
            log.info("step3_skipped")

    # ============================================================
    # STEP 4: Fetch profiles for new leads
    # ============================================================
    leads_to_fetch = db.get_leads_without_profile(limit=1000)

    if not leads_to_fetch:
        log.info("step4_no_profiles_to_fetch")
    else:
        usernames = [l["username"] for l in leads_to_fetch]
        log.info("step4_fetch_profiles", count=len(usernames))

        contacts_found = 0
        profiles_fetched = 0
        avatars_downloaded = 0
        single_face_new = 0
        fallback_resolved = 0
        fallback_skipped = 0

        for i in range(0, len(usernames), PROFILE_BATCH_SIZE):
            batch = usernames[i:i + PROFILE_BATCH_SIZE]

            run = apify.actor("apify/instagram-profile-scraper").call(run_input={
                "usernames": batch,
            })
            detail = apify.run(run["id"]).get()
            items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())

            pipeline.log_run(
                actor_id="apify/instagram-profile-scraper",
                run_id=run["id"], status=run["status"],
                input_params={"batch_size": len(batch)},
                items_count=len(items),
                cost_usd=detail.get("usageTotalUsd", 0),
                duration_ms=detail.get("stats", {}).get("durationMillis"),
            )

            for p in items:
                username = p.get("username")
                if not username:
                    continue

                # Media URLs from latest posts
                media_urls = []
                for post in (p.get("latestPosts") or []):
                    for img in (post.get("images") or []):
                        if img:
                            media_urls.append(img)
                    if post.get("displayUrl"):
                        media_urls.append(post["displayUrl"])
                    if post.get("videoUrl"):
                        media_urls.append(post["videoUrl"])

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

                if not p.get("private"):
                    avatar_url = p.get("profilePicUrlHD") or p.get("profilePicUrl")
                    uid = p.get("id") or p.get("pk")
                    uid_str = str(uid) if uid else None
                    avatar_path = download_avatar(
                        avatar_url,
                        user_id=uid_str,
                        username=username,
                    )
                    if avatar_path:
                        avatars_downloaded += 1
                        faces_count = face_embedder.count_faces(avatar_path)
                        db.update_lead_avatar(username, avatar_path, faces_count)

                        if faces_count == 1:
                            single_face_new += 1
                            # Avatar itself is a clean single-face photo.
                            db.update_lead_face(username, avatar_path)
                        elif uid_str:
                            # Fallback: probe the last N posts, pick the
                            # dominant face if there's an unambiguous leader.
                            post_urls = _pick_post_images(
                                p.get("latestPosts"),
                                limit=fb_limit,
                                skip_videos=fb_skip_videos,
                            )
                            local_paths = download_post_photos(
                                post_urls, user_id=uid_str
                            )
                            result = resolve_face_leader(
                                local_paths,
                                face_embedder,
                                min_cluster_size=fb_min_cluster,
                                cluster_threshold=fb_threshold,
                            )
                            if result:
                                fallback_resolved += 1
                                db.update_lead_face(
                                    username, str(result.photo_path)
                                )
                            else:
                                fallback_skipped += 1

                            if not fb_keep_photos:
                                cleanup_lead_photos(
                                    uid_str,
                                    keep=(result.photo_path if result else None),
                                )

                contacts = extract_contacts(
                    bio=p.get("biography"),
                    external_url=p.get("externalUrl"),
                    external_urls=p.get("externalUrls"),
                )
                if any(v for v in contacts.values()):
                    db.update_lead_contacts(username=username, **{k: v for k, v in contacts.items() if v})
                    contacts_found += 1

        log.info("step4_done", profiles=profiles_fetched, contacts_from_bio=contacts_found,
                 avatars=avatars_downloaded, single_face=single_face_new,
                 fallback_resolved=fallback_resolved,
                 fallback_skipped=fallback_skipped)

    # ============================================================
    # SUMMARY
    # ============================================================
    stats_after = db.get_stats()
    ps = pipeline.summary()

    print(f"\n{'='*50}")
    print(f"Pipeline complete")
    print(f"{'='*50}")
    print(f"Tracked realtors:     {stats_after['tracked_realtors']}")
    print(f"Leads total:          {stats_after['leads_total']} (+{stats_after['leads_total'] - stats_before['leads_total']})")
    print(f"  with profile:       {stats_after['leads_with_profile']}")
    print(f"  with contacts:      {stats_after['leads_with_contacts']}")
    print(f"  with avatar:        {stats_after['leads_with_avatar']}")
    print(f"  single-face:        {stats_after['leads_with_single_face']}")
    print(f"  face photo ready:   {stats_after['leads_with_face_photo']}")
    print(f"Processed posts:      {stats_after['processed_posts']}")
    print(f"Post links:           {stats_after['post_links']}")
    print(f"Total API cost:       ${ps['total_cost_usd']:.4f}")
    print(f"Pipeline log:         {pipeline.file_path}")

    face_embedder.close()


if __name__ == "__main__":
    main()
