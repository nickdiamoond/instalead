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
from src.comment_normalizer import normalize_apidojo_api
from src.config import load_config
from src.contact_extractor import extract_contacts
from src.db import LeadDB
from src.face_embedder import make_face_embedder
from src.face_leader import resolve_face_leader
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger
from src.transcriber import NexaraTranscriber

setup_logging()
log = get_logger("pipeline")

POSTS_MAX_AGE_DAYS = 7
MIN_COMMENTS = 10
COMMENTS_GROWTH_PCT = 5.0
PROFILE_BATCH_SIZE = 50
COST_PER_COMMENT = 0.0005

# Step 3 comment scrapers. louisdeconinck is the primary because its
# snake_case Instagram-raw output maps 1:1 to ``lead_accounts`` columns
# and to ``apify/instagram-profile-scraper`` (Step 4) -- no field
# remapping needed downstream. apidojo-api is the fallback: it has been
# observed to keep working when louisdeconinck silently returns 0 items
# with status=SUCCEEDED. Its camelCase output is normalized via
# :func:`src.comment_normalizer.normalize_apidojo_api` before saving.
#
# These constants are the *defaults*. ``main()`` overrides them from
# ``config.yaml`` (``apify.actors.comments_primary`` /
# ``apify.actors.comments_fallback``) so a switch to a different actor
# is a config edit instead of a code change.
DEFAULT_COMMENTS_PRIMARY_ACTOR = "louisdeconinck/instagram-comments-scraper"
DEFAULT_COMMENTS_FALLBACK_ACTOR = "apidojo/instagram-comments-scraper-api"

# louisdeconinck silently returns 0 items with status=SUCCEEDED if its
# input is missing a per-post comment cap -- bisected via
# ``scripts/test_comment_scrapers.py`` (recipe 1 -> recipe 3). The
# fallback (apidojo-api) has no such requirement and is left alone.
#
# 10_000 is a *ceiling*, not a target: the actor returns only
# comments that actually exist on the post, so a higher cap doesn't
# raise our bill -- it just protects against losing the tail on a
# viral post. Max ``comments_count`` observed in our DB is ~2_200
# (avg ~130), so 10_000 leaves ~5x headroom for unexpected spikes.
# The cap is applied on the primary's call only -- see
# ``_fetch_comments_with_fallback``.
LOUISDECONINCK_COMMENTS_CAP_PER_POST = 10_000

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


def _apply_score(db: LeadDB, post_id: str, score: dict | None) -> str:
    """Persist a DeepSeek score result. Returns the resolved relevance.

    Centralizes the upsert mapping so step 2 can call it from any branch
    (caption-only, transcript fallback, terminal-unknown).
    """
    if not score or "error" in score:
        db.upsert_post(
            post_id, relevance="unknown", has_cta=0, cta_type="none"
        )
        return "unknown"
    has_cta = 1 if score.get("has_call_to_action") else 0
    cta_type = score.get("call_to_action_type") or "none"
    is_re = score.get("is_real_estate")
    if is_re is None:
        db.upsert_post(
            post_id, relevance="unknown", has_cta=has_cta, cta_type=cta_type
        )
        return "unknown"
    relevance = "relevant" if is_re else "irrelevant"
    db.upsert_post(
        post_id, relevance=relevance, has_cta=has_cta, cta_type=cta_type
    )
    return relevance


def _build_scoring_text(caption: str | None, transcript: str | None) -> str:
    """Concatenate caption and video transcript into a single payload.

    Order is fixed: caption first, transcript second, separated by a
    blank line. Either part may be missing. The result is what gets
    sent to ``RELEVANCE_PROMPT``.
    """
    parts: list[str] = []
    if caption and caption.strip():
        parts.append(caption.strip())
    if transcript and transcript.strip():
        parts.append(transcript.strip())
    return "\n\n".join(parts)


def _banner(title: str, char: str = "=") -> None:
    """Print a wide stdout banner — survives the structlog stderr scroll
    on Windows PowerShell, so per-step status remains readable after the
    run finishes."""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")


def _run_apify_actor(
    apify: ApifyClient,
    pipeline: PipelineLogger,
    actor_id: str,
    run_input: dict,
    *,
    log_input: dict | None = None,
) -> tuple[list[dict], float, dict]:
    """Run an Apify actor and return ``(items, cost_usd, run_meta)``.

    Centralizes the boilerplate of ``actor.call`` -> ``run.get`` ->
    ``dataset.iterate_items`` -> ``pipeline.log_run`` so Step 3's
    primary/fallback split doesn't duplicate it. ``log_input`` is what
    gets persisted to the pipeline JSON log -- usually a sanitized
    summary like ``{"urls_count": N}`` rather than the full URL list.
    """
    run = apify.actor(actor_id).call(run_input=run_input)
    detail = apify.run(run["id"]).get() or {}
    cost = detail.get("usageTotalUsd") or 0.0
    items: list[dict] = []
    dataset_id = run.get("defaultDatasetId")
    if dataset_id:
        items = list(apify.dataset(dataset_id).iterate_items())
    pipeline.log_run(
        actor_id=actor_id,
        run_id=run["id"],
        status=run["status"],
        input_params=log_input or run_input,
        items_count=len(items),
        cost_usd=cost,
        duration_ms=detail.get("stats", {}).get("durationMillis"),
    )
    return items, cost, run


def _fetch_comments_with_fallback(
    apify: ApifyClient,
    pipeline: PipelineLogger,
    urls: list[str],
    *,
    primary_actor: str,
    fallback_actor: str,
) -> tuple[list[dict], float, str, dict]:
    """Pull comments for ``urls`` with primary -> apidojo-api fallback.

    Returns ``(items, total_cost, source, debug)`` where:

    * ``items`` is a list of louisdeconinck-shaped dicts (the apidojo-api
      branch normalizes via
      :func:`src.comment_normalizer.normalize_apidojo_api` so the
      caller's dedup / save loop is actor-agnostic).
    * ``total_cost`` is primary + fallback ``usageTotalUsd`` summed.
    * ``source`` is one of ``"primary"`` / ``"fallback"`` /
      ``"both-empty"`` -- the caller uses ``"both-empty"`` to leave
      ``processed_posts.last_scanned_at`` untouched so the queue keeps
      retrying instead of silently freezing (the same guard the script
      had before the fallback was added).
    * ``debug`` carries metadata each branch may want to surface in
      banners / issues -- ``primary_run_id``, ``primary_cost``,
      ``primary_items``, plus ``fallback_*`` if the fallback fired.

    Both Apify runs are logged separately via ``pipeline.log_run`` so
    the per-actor cost split stays explicit in ``logs/pipeline_*.json``.

    The actor ids are passed in (rather than read from module-level
    constants) so ``main()`` can override them from ``config.yaml``
    without touching this function.
    """
    # Two louisdeconinck-specific guardrails baked into the primary
    # call -- both bisected via ``scripts/test_comment_scrapers.py``:
    #
    # * ``proxy: useApifyProxy`` keeps Apify infra IPs off Instagram's
    #   block list. Without it the actor finishes ~9s with 0 items.
    #
    # * ``resultsLimit`` + ``maxComments`` are MANDATORY for this
    #   actor: omitting them is the actual reason Step 3 has been
    #   silently returning 0 items even with proxy on (recipe 3
    #   confirmed it -- the cap is what makes the actor commit
    #   instead of bailing out). The fallback (apidojo-api) does
    #   NOT need this and intentionally keeps its uncapped shape.
    #   ``LOUISDECONINCK_COMMENTS_CAP_PER_POST`` is set well above
    #   any per-post comment count we've ever seen, so it acts as
    #   a safety ceiling rather than a real cap.
    primary_items, primary_cost, primary_run = _run_apify_actor(
        apify,
        pipeline,
        primary_actor,
        run_input={
            "urls": urls,
            "proxy": {"useApifyProxy": True},
            "resultsLimit": LOUISDECONINCK_COMMENTS_CAP_PER_POST,
            "maxComments": LOUISDECONINCK_COMMENTS_CAP_PER_POST,
        },
        log_input={
            "urls_count": len(urls),
            "results_limit": LOUISDECONINCK_COMMENTS_CAP_PER_POST,
        },
    )
    debug = {
        "primary_actor": primary_actor,
        "primary_run_id": primary_run["id"],
        "primary_status": primary_run["status"],
        "primary_items": len(primary_items),
        "primary_cost": primary_cost,
    }

    if primary_items:
        return primary_items, primary_cost, "primary", debug

    log.warning(
        "step3_primary_empty_falling_back",
        actor=primary_actor,
        fallback=fallback_actor,
        urls=len(urls),
        run_id=primary_run["id"],
        primary_cost=primary_cost,
        msg="primary returned 0 items, retrying via fallback",
    )

    fb_raw, fb_cost, fb_run = _run_apify_actor(
        apify,
        pipeline,
        fallback_actor,
        # apidojo-api takes ``startUrls`` (flat string array) + ``maxItems``.
        # Omitting maxItems lets it fetch every comment, matching the
        # primary's "no per-post cap" behavior. ``proxy: useApifyProxy``
        # is harmless if the actor's input schema doesn't declare it
        # (Apify silently drops unknown fields) and matches the rest of
        # the pipeline's Apify calls -- see the primary above.
        run_input={
            "startUrls": urls,
            "proxy": {"useApifyProxy": True},
        },
        log_input={"startUrls_count": len(urls), "fallback": True},
    )
    debug.update(
        {
            "fallback_actor": fallback_actor,
            "fallback_run_id": fb_run["id"],
            "fallback_status": fb_run["status"],
            "fallback_raw_items": len(fb_raw),
            "fallback_cost": fb_cost,
        }
    )

    fb_items = [
        normalized
        for normalized in (normalize_apidojo_api(it) for it in fb_raw)
        if normalized is not None
    ]
    debug["fallback_normalized_items"] = len(fb_items)
    total_cost = primary_cost + fb_cost

    if not fb_items:
        log.error(
            "step3_fallback_also_empty",
            primary=primary_actor,
            fallback=fallback_actor,
            primary_run_id=primary_run["id"],
            fallback_run_id=fb_run["id"],
            total_cost=total_cost,
        )
        return [], total_cost, "both-empty", debug

    log.info(
        "step3_fallback_recovered",
        actor=fallback_actor,
        raw=len(fb_raw),
        normalized=len(fb_items),
        primary_cost=primary_cost,
        fallback_cost=fb_cost,
    )
    return fb_items, total_cost, "fallback", debug


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
    transcriber = NexaraTranscriber(
        os.environ.get("NEXARA_API_KEY"),
        pipeline=pipeline,
    )

    # Issues are surfaced both per-step (loud banner at the failure point)
    # and again in the final summary. Each entry is a (step, hint) tuple
    # — when the list is non-empty at the end we hold the script open
    # with `input()` so the operator can read the diagnostic instead of
    # losing it to terminal scrollback.
    issues: list[tuple[str, str]] = []

    # Two SCRFD instances with different det_size:
    #   * avatar_embedder (320x320) for the avatar single-face check
    #   * post_embedder (640x640) for the last-N-posts leader fallback
    # See make_face_embedder docstring / config.yaml for the rationale.
    avatar_embedder = make_face_embedder(cfg, kind="avatar")
    post_embedder = make_face_embedder(cfg, kind="post")

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
    _banner(f"STEP 1: Fetch posts (last {POSTS_MAX_AGE_DAYS} days)")
    realtors = db.get_active_realtors()
    if not realtors:
        log.error("no_realtors", msg="Add realtors to tracked_realtors table first")
        print("FAILED: no active realtors in DB. Add rows to tracked_realtors first.")
        issues.append(("Step 1", "no active realtors in tracked_realtors"))
        return

    print(f"  Realtors:       {len(realtors)}")
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
    # In-memory bridge to step 2: shortcode -> fresh IG videoUrl. Used by
    # the transcription fallback. Stored only for the lifetime of this
    # run because IG CDN URLs are signed and expire in ~1-2 days.
    post_videos: dict[str, str] = {}
    for p in all_posts:
        shortcode = p.get("shortCode", "")
        comments_count = p.get("commentsCount") or 0
        if comments_count < MIN_COMMENTS:
            continue

        is_reel = p.get("type") == "Video" or p.get("productType") == "clips"
        existing = db.get_post(shortcode)

        video_url = p.get("videoUrl")
        if shortcode and video_url:
            post_videos[shortcode] = video_url

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
             videos=len(post_videos), cost=detail.get("usageTotalUsd", 0))
    print(f"  DONE: fetched {len(all_posts)} posts "
          f"(new={new_posts}, updated={updated_posts}, "
          f"with_video={len(post_videos)}) "
          f"cost=${detail.get('usageTotalUsd', 0):.4f}")
    if len(all_posts) == 0:
        issues.append(("Step 1", "post-scraper returned 0 items"))

    # ============================================================
    # STEP 2: Score new posts via DeepSeek (caption + transcript)
    # ============================================================
    _banner("STEP 2: Score new posts (DeepSeek over caption + transcript)")
    with db._conn() as conn:
        unscored = conn.execute(
            "SELECT post_id, caption FROM processed_posts WHERE relevance IS NULL"
        ).fetchall()
        unscored = [dict(r) for r in unscored]

    print(f"  Unscored posts:    {len(unscored)}")
    print(f"  Posts with video:  {len(post_videos)}")
    log.info(
        "step2_score_posts", count=len(unscored), with_videos=len(post_videos)
    )

    transcribed = 0
    transcribe_failed = 0
    empty_skipped = 0

    for p in unscored:
        post_id = p["post_id"]
        caption = p.get("caption")
        video_url = post_videos.get(post_id)

        # Always transcribe when a fresh videoUrl is available -- the
        # transcript is concatenated with the caption (caption first,
        # transcript second) and the combined payload is scored in a
        # single DeepSeek call. IG videoUrls are signed and expire in
        # ~1-2 days, so transcription only fires for posts pulled in
        # the *current* run; older ``relevance IS NULL`` leftovers
        # fall back to caption-only scoring on subsequent runs.
        transcript: str | None = None
        if video_url:
            transcript = transcriber.transcribe(video_url)
            if transcript:
                transcribed += 1
            else:
                transcribe_failed += 1

        combined = _build_scoring_text(caption, transcript)

        # Nothing meaningful to send to DeepSeek (no caption / just
        # hashtags AND no usable transcript) -> mark unknown without
        # spending a DeepSeek call.
        if caption_is_empty(combined):
            empty_skipped += 1
            _apply_score(db, post_id, None)
            continue

        _apply_score(db, post_id, score_caption(deepseek, combined))

    log.info(
        "step2_done",
        scored=len(unscored),
        transcribed=transcribed,
        transcribe_failed=transcribe_failed,
        empty_skipped=empty_skipped,
    )
    print(f"  DONE: scored {len(unscored)} "
          f"(transcribed={transcribed}, "
          f"transcribe_failed={transcribe_failed}, "
          f"empty_skipped={empty_skipped})")
    if transcribe_failed and transcribe_failed > transcribed:
        issues.append((
            "Step 2",
            f"transcription failed on {transcribe_failed} posts vs "
            f"{transcribed} succeeded — check NEXARA_API_KEY and IG video URL freshness",
        ))

    # ============================================================
    # STEP 3: Fetch comments
    # ============================================================
    _banner("STEP 3: Fetch comments (Apify)")
    posts_to_scan = db.get_posts_needing_comments(min_growth_pct=COMMENTS_GROWTH_PCT)

    if not posts_to_scan:
        print("  SKIPPED: no relevant posts in the queue.")
        log.info("step3_no_posts_to_scan")
    else:
        total_comments = sum(p.get("comments_count") or 0 for p in posts_to_scan)
        estimated_cost = total_comments * COST_PER_COMMENT

        log.info("step3_fetch_comments", posts=len(posts_to_scan),
                 total_comments=total_comments, estimated_cost=round(estimated_cost, 2))

        print(f"  Posts to scan:        {len(posts_to_scan)}")
        print(f"  Estimated comments:   {total_comments}")
        print(f"  Estimated cost:       ${estimated_cost:.2f}")
        confirm = input("  Proceed? (y/n): ").strip().lower()

        if confirm == "y":
            urls = [p["post_url"] for p in posts_to_scan if p.get("post_url")]

            # Read actor IDs from config so swapping primary/fallback
            # is a config edit, not a code change. Defaults preserve
            # the historical behavior if the keys are missing.
            actor_cfg = (cfg.get("apify") or {}).get("actors") or {}
            primary_actor = actor_cfg.get(
                "comments_primary", DEFAULT_COMMENTS_PRIMARY_ACTOR
            )
            fallback_actor = actor_cfg.get(
                "comments_fallback", DEFAULT_COMMENTS_FALLBACK_ACTOR
            )

            items, cost, source, debug = _fetch_comments_with_fallback(
                apify,
                pipeline,
                urls,
                primary_actor=primary_actor,
                fallback_actor=fallback_actor,
            )

            # Bail out *before* marking anything as scanned if both the
            # primary and the fallback returned an empty dataset.
            # Background: louisdeconinck has been observed to silently
            # "succeed" with 0/null comments per page (its own log says
            # ``fetched 0/null comments``). Marking those posts as
            # scanned would freeze them out of the queue until comments
            # grow another COMMENTS_GROWTH_PCT% — i.e. silently lose
            # tens of thousands of real commenters. Treat as a transient
            # failure and leave the queue untouched so the next run
            # retries them.
            if source == "both-empty":
                log.warning(
                    "step3_empty_after_fallback",
                    posts=len(posts_to_scan),
                    urls=len(urls),
                    debug=debug,
                    msg=(
                        "primary AND fallback returned 0 items -- "
                        "leaving posts unscanned for retry"
                    ),
                )
                primary_url = (
                    f"https://console.apify.com/actors/runs/"
                    f"{debug.get('primary_run_id')}"
                )
                fallback_url = (
                    f"https://console.apify.com/actors/runs/"
                    f"{debug.get('fallback_run_id')}"
                )
                print(f"\n{'!' * 60}")
                print("  STEP 3 FAILED: both scrapers returned 0 items")
                print(f"  URLs sent:        {len(urls)}")
                print(
                    f"  Primary run:      {debug.get('primary_run_id')} "
                    f"-- {primary_url}"
                )
                print(
                    f"  Fallback run:     {debug.get('fallback_run_id')} "
                    f"-- {fallback_url}"
                )
                print(f"  Combined cost:    ${cost:.4f}")
                print("  Queue NOT marked scanned -- re-run the pipeline")
                print("  once at least one of the scrapers recovers.")
                print(f"{'!' * 60}")
                issues.append((
                    "Step 3",
                    f"primary+fallback returned 0 items on {len(urls)} URLs "
                    f"(primary {debug.get('primary_run_id')}, "
                    f"fallback {debug.get('fallback_run_id')}); "
                    f"queue preserved for retry",
                ))
            else:
                # Surface the fallback path (if it fired) at the top of
                # the success block so the operator sees right away that
                # we paid twice -- once for the empty primary, once for
                # the working fallback. ``debug`` already carries the
                # split costs for the JSON pipeline log; banner is for
                # the human watching the terminal.
                if source == "fallback":
                    log.warning(
                        "step3_used_fallback",
                        actor=fallback_actor,
                        primary_cost=debug.get("primary_cost"),
                        fallback_cost=debug.get("fallback_cost"),
                    )
                    print(
                        f"  NOTE: primary returned 0 -- fell back to "
                        f"{fallback_actor} "
                        f"(primary cost ${debug.get('primary_cost', 0):.4f} "
                        f"wasted, fallback "
                        f"${debug.get('fallback_cost', 0):.4f})"
                    )
                    issues.append((
                        "Step 3",
                        f"primary {primary_actor} returned 0 items; "
                        f"fallback {fallback_actor} recovered "
                        f"{debug.get('fallback_normalized_items', 0)} items",
                    ))

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

                # Mark posts as scanned only on a non-empty dataset --
                # see the bail-out comment above for the rationale.
                for p in posts_to_scan:
                    db.mark_post_comments_scanned(
                        p["post_id"],
                        p.get("comments_count") or 0,
                    )

                log.info(
                    "step3_done",
                    raw=len(items),
                    unique=len(unique),
                    new_leads=new_leads,
                    cost=cost,
                    source=source,
                )
                print(
                    f"  DONE: {new_leads} new leads "
                    f"({len(unique)} unique commenters / {len(items)} raw) "
                    f"via {source} cost=${cost:.4f}"
                )
        else:
            log.info("step3_skipped")
            print("  SKIPPED by user.")

    # ============================================================
    # STEP 4: Fetch profiles for new leads
    # ============================================================
    _banner("STEP 4: Fetch profiles for new leads")
    leads_to_fetch = db.get_leads_without_profile(limit=1000)

    if not leads_to_fetch:
        print("  SKIPPED: no leads without profile.")
        log.info("step4_no_profiles_to_fetch")
    else:
        usernames = [l["username"] for l in leads_to_fetch]
        print(f"  Leads to fetch:  {len(usernames)}")
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
                        faces_count = avatar_embedder.count_faces(avatar_path)
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
                                post_embedder,
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
        print(f"  DONE: profiles={profiles_fetched} "
              f"contacts={contacts_found} "
              f"avatars={avatars_downloaded} "
              f"single_face={single_face_new} "
              f"fallback_resolved={fallback_resolved} "
              f"fallback_skipped={fallback_skipped}")

    # ============================================================
    # SUMMARY
    # ============================================================
    stats_after = db.get_stats()
    ps = pipeline.summary()

    _banner("PIPELINE COMPLETE")
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

    if issues:
        print(f"\n{'!' * 60}")
        print("  ISSUES DETECTED — review before re-running:")
        for step, hint in issues:
            print(f"    [{step}] {hint}")
        print(f"{'!' * 60}")
        # Hold the terminal open so the operator actually reads the
        # diagnostic instead of losing it to PowerShell scrollback when
        # the prompt returns. EOF (Ctrl-Z / closed pipe) is fine — we
        # swallow it to keep the script non-interactive-friendly.
        try:
            input("\nPress Enter to exit... ")
        except EOFError:
            pass

    avatar_embedder.close()
    post_embedder.close()


if __name__ == "__main__":
    main()
