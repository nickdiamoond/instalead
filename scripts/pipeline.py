"""Daily lead collection pipeline.

Steps:
  1. Fetch recent posts/reels (one or more of: realtor accounts, hashtags,
     cookie keyword search — ``pipeline.step1.discovery_mode`` str or list)
  2. Score new posts via DeepSeek (relevance + CTA)
  3. Fetch comments for relevant posts (new + grown)
  4. Fetch profiles for new leads, extract contacts from bio
  5. Resolve Telegram contacts for naked leads via Sherlock
     (nick search first, photo fallback with DeepSeek disambiguation)

Uses DB for deduplication — safe to run repeatedly.
"""

import asyncio
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv
from lingua import Language
from openai import OpenAI

from src.avatar_downloader import (
    cleanup_lead_photos,
    download_avatar,
    download_post_photos,
)
from src.config import (
    deepseek_relevance_prompt,
    deepseek_usermatch_prompt,
    load_config,
    step1_cookie_search_section,
    step1_min_comments_per_post,
    step1_posts_max_age_days,
)
from src.contact_extractor import extract_contacts
from src.db import LeadDB
from src.face_embedder import make_face_embedder
from src.face_leader import resolve_face_leader
from src.ig_media_payload import (
    extract_video_url,
    filter_items_within_max_age,
    is_reel_payload,
    is_valid_video_url,
    merge_hashtag_items_by_shortcode,
    post_location_label_from_item,
)
from src.instagram_cookie_search import (
    cookies_json_string_for_actor,
    dedupe_keyword_items_by_shortcode,
    normalize_keyword_search_item,
)
from src.logger import setup_logging
from src.pipeline_logger import PipelineLogger
from src.regions import (
    parse_active_regions,
    region_cookie_keywords,
    region_hashtags,
    region_realtor_accounts,
    region_result_chat_id,
)
from src.telegram_notifier import (
    PipelineTelegramNotifier,
    build_step1_date_filter_section_lines,
    build_step1_pipeline_summary_telegram_text,
)
from src.transcriber import NexaraTranscriber

from scripts.pipeline_lib.apify_runner import _fetch_comments_with_fallback
from scripts.pipeline_lib.cli import _parse_cli_args
from scripts.pipeline_lib.config_parse import _cfg_prompt_terminal_confirmation
from scripts.pipeline_lib.defaults import (
    DEFAULT_APIFY_COMMENTS_CAP_PER_POST,
    DEFAULT_COMMENTS_FALLBACK_ACTOR,
    DEFAULT_COMMENTS_GROWTH_PCT,
    DEFAULT_COMMENTS_PRIMARY_ACTOR,
    DEFAULT_COST_PER_COMMENT,
    DEFAULT_LOUISDECONINCK_COMMENTS_CAP_PER_POST,
    DEFAULT_MIN_AVATAR_FACE_AREA_PCT,
    DEFAULT_POST_SCRAPER_RESULTS_LIMIT,
    DEFAULT_PROFILE_BATCH_SIZE,
    DEFAULT_PROMPT_TERMINAL_CONFIRMATION,
    DEFAULT_SHERLOCK_BATCH_LIMIT,
    DEFAULT_SHERLOCK_REQUEST_GAP_SECS,
    DEFAULT_SHERLOCK_SEQUENTIAL,
    DEFAULT_STEP1_DISCOVERY_MODE,
    DEFAULT_STEP4_BATCH_LIMIT,
)
from scripts.pipeline_lib.ig_shortcode import caption_is_empty, shortcode_to_id
from scripts.pipeline_lib.io_utils import _banner
from scripts.pipeline_lib.logging import log
from scripts.pipeline_lib.scoring import (
    _apply_language_gate_irrelevant,
    _apply_score,
    _build_scoring_text,
    _run_step2_human_confirmations,
    detect_scoring_text_language,
    score_caption,
)
from scripts.pipeline_lib.step4_faces import (
    _pick_post_images,
    _reconcile_step4_ephemeral_avatar,
    face_bbox_percent_of_image,
)
from scripts.pipeline_lib.step5_sherlock import (
    SHERLOCK_HEALTH_PROBE_MAX_ATTEMPTS,
    _step_5_resolve_contacts_via_sherlock,
)
from scripts.pipeline_lib.step1_discovery import (
    VALID_STEP1_DISCOVERY_MODES,
    build_step1_searched_summary,
    format_step1_discovery_modes_label,
    parse_step1_discovery_modes,
)
from scripts.pipeline_lib.step6_cleanup import _step_6_cleanup_spent_face_assets

setup_logging()


def _step1_apify_actor_call(
    apify: ApifyClient,
    tg_notifier: PipelineTelegramNotifier,
    issues: list[tuple[str, str]],
    *,
    actor_id: str,
    run_input: dict,
) -> dict | None:
    """Run a Step 1 Apify actor; on exception alert, log, and return None."""
    try:
        return apify.actor(actor_id).call(run_input=run_input)
    except Exception as e:
        log.error("step1_apify_call_failed", actor_id=actor_id, error=str(e))
        print(f"FAILED: Step 1 Apify call ({actor_id}): {e}")
        tg_notifier.notify_step1_apify_call_error(e)
        issues.append(("Step 1", f"Apify call error ({actor_id}): {e}"))
        return None


def main():
    args = _parse_cli_args()
    load_dotenv()
    cfg = load_config()
    relevance_prompt = deepseek_relevance_prompt(cfg)
    usermatch_prompt = deepseek_usermatch_prompt(cfg)
    tg_notifier = PipelineTelegramNotifier.from_config(cfg)
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
    # and again in the final summary. Each entry is a (step, hint) tuple.
    # When ``pipeline.prompt_terminal_confirmation`` is true and the list
    # is non-empty at the end we hold the script open with `input()` so the
    # operator can read the diagnostic instead of losing it to terminal
    # scrollback.
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

    fd_cfg = cfg.get("face_detection") or {}
    min_avatar_face_area_pct = float(
        fd_cfg.get("min_avatar_face_area_pct", DEFAULT_MIN_AVATAR_FACE_AREA_PCT)
    )

    # Per-step tuning knobs from config.yaml (``pipeline.stepN.*``).
    # Every value falls back to its DEFAULT_* module constant so a
    # missing section / key boots the pipeline with safe values.
    pipe_cfg = cfg.get("pipeline") or {}
    s1_cfg = pipe_cfg.get("step1") or {}
    s3_cfg = pipe_cfg.get("step3") or {}
    s4_cfg = pipe_cfg.get("step4") or {}
    s5_cfg = pipe_cfg.get("step5") or {}

    posts_max_age_days = step1_posts_max_age_days(cfg)
    min_comments = step1_min_comments_per_post(cfg)
    post_scraper_results_limit = int(
        s1_cfg.get(
            "post_scraper_results_limit",
            DEFAULT_POST_SCRAPER_RESULTS_LIMIT,
        )
    )
    discovery_modes = parse_step1_discovery_modes(
        s1_cfg.get("discovery_mode", DEFAULT_STEP1_DISCOVERY_MODE)
    )
    active_regions = parse_active_regions(cfg)
    hashtag_results_limit = int(
        s1_cfg.get("hashtag_results_limit", post_scraper_results_limit)
    )
    comments_growth_pct = float(
        s3_cfg.get("comments_growth_pct", DEFAULT_COMMENTS_GROWTH_PCT)
    )
    cost_per_comment = float(
        s3_cfg.get("cost_per_comment_usd", DEFAULT_COST_PER_COMMENT)
    )
    louisdeconinck_cap = int(
        s3_cfg.get(
            "louisdeconinck_comments_cap_per_post",
            DEFAULT_LOUISDECONINCK_COMMENTS_CAP_PER_POST,
        )
    )
    apidojo_cap = int(
        s3_cfg.get(
            "apidojo_comments_cap_per_post",
            DEFAULT_APIFY_COMMENTS_CAP_PER_POST,
        )
    )
    profile_batch_size = int(
        s4_cfg.get("profile_batch_size", DEFAULT_PROFILE_BATCH_SIZE)
    )
    step4_batch_limit = int(
        s4_cfg.get("batch_limit", DEFAULT_STEP4_BATCH_LIMIT)
    )
    sherlock_batch_limit = int(
        s5_cfg.get("batch_limit", DEFAULT_SHERLOCK_BATCH_LIMIT)
    )
    sherlock_sequential = bool(
        s5_cfg.get("sequential", DEFAULT_SHERLOCK_SEQUENTIAL)
    )
    sherlock_request_gap_secs = float(
        s5_cfg.get(
            "request_gap_secs",
            DEFAULT_SHERLOCK_REQUEST_GAP_SECS,
        )
    )
    prompt_terminal_confirmation = _cfg_prompt_terminal_confirmation(
        pipe_cfg.get("prompt_terminal_confirmation"),
        DEFAULT_PROMPT_TERMINAL_CONFIRMATION,
    )

    stats_before = db.get_stats()
    log.info("pipeline_start", **stats_before)

    # ============================================================
    # STEP 1: Fetch posts (realtors, hashtags, or cookie keyword search)
    # ============================================================
    actors_cfg = (cfg.get("apify") or {}).get("actors") or {}
    hashtag_actor_id = actors_cfg.get("hashtag", "apify/instagram-hashtag-scraper")
    cookie_search_actor_id = actors_cfg.get(
        "cookie_search_posts", "crawlerbros/instagram-keyword-search-scraper"
    )

    invalid_modes = [m for m in discovery_modes if m not in VALID_STEP1_DISCOVERY_MODES]
    if invalid_modes:
        log.error("step1_invalid_discovery_mode", modes=invalid_modes)
        print(
            "FAILED: pipeline.step1.discovery_mode entries must be "
            "'realtors', 'hashtags', and/or 'cookie_keywords'; "
            f"invalid: {invalid_modes!r}."
        )
        issues.append(("Step 1", f"invalid discovery_mode: {invalid_modes}"))
        return

    if not active_regions:
        log.error("step1_no_active_regions")
        print(
            "FAILED: pipeline.regions is empty -- list at least one region "
            "defined under region_definitions (e.g. moscow, rostov)."
        )
        issues.append(("Step 1", "pipeline.regions empty"))
        return

    region_catalog = cfg.get("region_definitions") or {}
    unknown_regions = [r for r in active_regions if r not in region_catalog]
    if unknown_regions:
        log.error("step1_unknown_regions", regions=unknown_regions)
        print(
            "FAILED: pipeline.regions references region(s) absent from "
            f"region_definitions: {unknown_regions!r}."
        )
        issues.append(("Step 1", f"unknown region(s): {unknown_regions}"))
        return

    _banner(
        f"STEP 1: Fetch posts (≤{posts_max_age_days}d) "
        f"[{format_step1_discovery_modes_label(discovery_modes)}] "
        f"regions=[{', '.join(active_regions)}]"
    )

    step1_cost_usd = 0.0
    posts_age_stats: dict[str, int] | None = None
    reels_age_stats: dict[str, int] | None = None
    step1_empty_issue = "Step 1 discovery returned 0 items after filters"
    source_counts: dict[str, int] = {}
    modes_ran: list[str] = []
    regions_ran: list[str] = []
    step1_age_dropped_client: int | None = None
    step1_age_kept_missing_ts: int | None = None
    all_posts: list[dict] = []
    # shortCode -> first region that discovered it (authoritative region tag).
    # Decoupled from merge_hashtag_items_by_shortcode (whose tie-break can swap
    # the in-memory item) so a shortcode seen in two regions keeps the first.
    shortcode_region: dict[str, str] = {}

    # Step 1 loops region x discovery_mode: each region supplies its own source
    # content; region tag flows to processed_posts (first region wins).
    region_mode_pairs = [
        (region, mode) for region in active_regions for mode in discovery_modes
    ]
    for region, discovery_mode in region_mode_pairs:
        if discovery_mode == "realtors":
            print(f"\n  --- {region} / realtors ---")
            realtors = region_realtor_accounts(cfg, region)
            if not realtors:
                log.warning(
                    "step1_skip_realtors",
                    region=region,
                    reason="region_definitions realtor_accounts empty",
                )
                print(
                    f"  SKIP {region}/realtors: region_definitions.{region}."
                    "realtor_accounts is empty."
                )
                continue

            print(f"  Realtors:       {len(realtors)}")
            log.info(
                "step1_fetch_posts",
                region=region,
                discovery_mode=discovery_mode,
                realtors=len(realtors),
                max_age_days=posts_max_age_days,
                post_scraper_results_limit=post_scraper_results_limit,
            )

            run = _step1_apify_actor_call(
                apify,
                tg_notifier,
                issues,
                actor_id="apify/instagram-post-scraper",
                run_input={
                    "username": realtors,
                    "resultsLimit": post_scraper_results_limit,
                    "onlyPostsNewerThan": f"{posts_max_age_days} days",
                    "dataDetailLevel": "basicData",
                    "proxy": {"useApifyProxy": True},
                },
            )
            if run is None:
                return
            tg_notifier.maybe_notify_apify_run_failure(
                run,
                actor_id="apify/instagram-post-scraper",
                step="Step 1",
            )
            detail = apify.run(run["id"]).get()
            posts_fetched = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
            mode_posts, mode_age_stats = filter_items_within_max_age(
                posts_fetched, posts_max_age_days
            )
            mode_dropped = int((mode_age_stats or {}).get("dropped_too_old") or 0)
            mode_kept_missing = int(
                (mode_age_stats or {}).get("kept_missing_timestamp") or 0
            )
            step1_age_dropped_client = (step1_age_dropped_client or 0) + mode_dropped
            step1_age_kept_missing_ts = (step1_age_kept_missing_ts or 0) + mode_kept_missing
            posts_age_stats = mode_age_stats
            step1_cost_usd += float(detail.get("usageTotalUsd") or 0)
            for _it in mode_posts:
                _sc = (_it.get("shortCode") or "").strip()
                if _sc:
                    shortcode_region.setdefault(_sc, region)
            all_posts = merge_hashtag_items_by_shortcode(all_posts, mode_posts)
            source_counts["realtors"] = source_counts.get("realtors", 0) + len(realtors)
            modes_ran.append(discovery_mode)
            regions_ran.append(region)

            pipeline.log_run(
                actor_id="apify/instagram-post-scraper",
                run_id=run["id"],
                status=run["status"],
                input_params={
                    "realtors": len(realtors),
                    "resultsLimit": post_scraper_results_limit,
                    "max_age_days": posts_max_age_days,
                },
                items_count=len(posts_fetched),
                cost_usd=detail.get("usageTotalUsd", 0),
                duration_ms=detail.get("stats", {}).get("durationMillis"),
            )
            log.info(
                "step1_realtor_age_filter",
                region=region,
                raw=len(posts_fetched),
                after_filter=len(mode_posts),
                merged_total=len(all_posts),
                posts_age_stats=posts_age_stats,
            )

        elif discovery_mode == "hashtags":
            print(f"\n  --- {region} / hashtags ---")
            hashtags = region_hashtags(cfg, region)
            if not hashtags:
                log.warning(
                    "step1_skip_hashtags",
                    region=region,
                    reason="region_definitions hashtags empty",
                )
                print(
                    f"  SKIP {region}/hashtags: region_definitions.{region}."
                    "hashtags is empty."
                )
                continue

            print(f"  Hashtags:       {len(hashtags)}")
            log.info(
                "step1_fetch_posts",
                region=region,
                discovery_mode=discovery_mode,
                hashtags=len(hashtags),
                hashtag_results_limit=hashtag_results_limit,
                max_age_days=posts_max_age_days,
            )

            proxy_in = {"useApifyProxy": True}
            run_base = {
                "hashtags": hashtags,
                "resultsLimit": hashtag_results_limit,
                "proxy": proxy_in,
            }

            run_p = _step1_apify_actor_call(
                apify,
                tg_notifier,
                issues,
                actor_id=hashtag_actor_id,
                run_input={**run_base, "resultsType": "posts"},
            )
            if run_p is None:
                return
            tg_notifier.maybe_notify_apify_run_failure(
                run_p, actor_id=hashtag_actor_id, step="Step 1 (hashtag posts)"
            )
            detail_p = apify.run(run_p["id"]).get()
            posts_fetched = list(apify.dataset(run_p["defaultDatasetId"]).iterate_items())
            posts_filtered, posts_age_stats = filter_items_within_max_age(
                posts_fetched, posts_max_age_days
            )
            cost_p = float(detail_p.get("usageTotalUsd") or 0)
            step1_cost_usd += cost_p
            pipeline.log_run(
                actor_id=hashtag_actor_id,
                run_id=run_p["id"],
                status=run_p["status"],
                input_params={
                    "hashtags": len(hashtags),
                    "resultsType": "posts",
                    "resultsLimit": hashtag_results_limit,
                },
                items_count=len(posts_fetched),
                cost_usd=cost_p,
                duration_ms=detail_p.get("stats", {}).get("durationMillis"),
            )

            run_r = _step1_apify_actor_call(
                apify,
                tg_notifier,
                issues,
                actor_id=hashtag_actor_id,
                run_input={**run_base, "resultsType": "reels"},
            )
            if run_r is None:
                return
            tg_notifier.maybe_notify_apify_run_failure(
                run_r, actor_id=hashtag_actor_id, step="Step 1 (hashtag reels)"
            )
            detail_r = apify.run(run_r["id"]).get()
            reels_fetched = list(apify.dataset(run_r["defaultDatasetId"]).iterate_items())
            reels_filtered, reels_age_stats = filter_items_within_max_age(
                reels_fetched, posts_max_age_days
            )
            cost_r = float(detail_r.get("usageTotalUsd") or 0)
            step1_cost_usd += cost_r
            pipeline.log_run(
                actor_id=hashtag_actor_id,
                run_id=run_r["id"],
                status=run_r["status"],
                input_params={
                    "hashtags": len(hashtags),
                    "resultsType": "reels",
                    "resultsLimit": hashtag_results_limit,
                },
                items_count=len(reels_fetched),
                cost_usd=cost_r,
                duration_ms=detail_r.get("stats", {}).get("durationMillis"),
            )

            mode_posts = merge_hashtag_items_by_shortcode(posts_filtered, reels_filtered)
            mode_dropped = int(
                (posts_age_stats or {}).get("dropped_too_old") or 0
            ) + int((reels_age_stats or {}).get("dropped_too_old") or 0)
            mode_kept_missing = int(
                (posts_age_stats or {}).get("kept_missing_timestamp") or 0
            ) + int((reels_age_stats or {}).get("kept_missing_timestamp") or 0)
            step1_age_dropped_client = (step1_age_dropped_client or 0) + mode_dropped
            step1_age_kept_missing_ts = (step1_age_kept_missing_ts or 0) + mode_kept_missing
            for _it in mode_posts:
                _sc = (_it.get("shortCode") or "").strip()
                if _sc:
                    shortcode_region.setdefault(_sc, region)
            all_posts = merge_hashtag_items_by_shortcode(all_posts, mode_posts)
            source_counts["hashtags"] = source_counts.get("hashtags", 0) + len(hashtags)
            modes_ran.append(discovery_mode)
            regions_ran.append(region)
            log.info(
                "step1_hashtag_merge",
                region=region,
                posts_raw=len(posts_fetched),
                reels_raw=len(reels_fetched),
                mode_posts=len(mode_posts),
                merged_total=len(all_posts),
                posts_age_stats=posts_age_stats,
                reels_age_stats=reels_age_stats,
            )

        elif discovery_mode == "cookie_keywords":
            print(f"\n  --- {region} / cookie_keywords ---")
            cs_cfg = step1_cookie_search_section(cfg)
            keywords = region_cookie_keywords(cfg, region)
            if not keywords:
                log.warning(
                    "step1_skip_cookie_keywords",
                    region=region,
                    reason="region_definitions cookie_search_keywords empty",
                )
                print(
                    f"  SKIP {region}/cookie_keywords: region_definitions."
                    f"{region}.cookie_search_keywords is empty."
                )
                continue

            cookie_var = str(
                cs_cfg.get("session_cookie_env_var", "INSTAGRAM_SESSION_COOKIE")
            )
            cookies_raw = (os.environ.get(cookie_var) or "").strip()
            if not cookies_raw:
                log.warning(
                    "step1_skip_cookie_keywords",
                    reason="missing session cookie",
                    env_var=cookie_var,
                )
                print(
                    f"  SKIP cookie_keywords: {cookie_var} is empty or unset."
                )
                issues.append(
                    (
                        "Step 1",
                        f"missing env {cookie_var} for cookie keyword search (skipped)",
                    )
                )
                continue

            try:
                cookies_payload = cookies_json_string_for_actor(cookies_raw)
            except (json.JSONDecodeError, ValueError) as e:
                log.error("step1_cookie_parse_failed", error=str(e))
                print(f"  SKIP cookie_keywords: could not normalize cookies: {e}")
                issues.append(("Step 1", f"cookie parse error: {e}"))
                continue

            max_posts = int(cs_cfg.get("size_per_keyword", 5))
            session_name = str(cs_cfg.get("session_name", "instalead_cookie_search"))

            print(f"  Keywords:       {len(keywords)}")
            log.info(
                "step1_fetch_posts",
                region=region,
                discovery_mode=discovery_mode,
                keywords=len(keywords),
                max_posts_per_keyword=max_posts,
                max_age_days=posts_max_age_days,
                cookie_env_var=cookie_var,
            )

            run_kw = _step1_apify_actor_call(
                apify,
                tg_notifier,
                issues,
                actor_id=cookie_search_actor_id,
                run_input={
                    "keywords": keywords,
                    "maxPosts": max_posts,
                    "cookies": cookies_payload,
                    "sessionName": session_name,
                },
            )
            if run_kw is None:
                return
            tg_notifier.maybe_notify_apify_run_failure(
                run_kw, actor_id=cookie_search_actor_id, step="Step 1"
            )
            detail_kw = apify.run(run_kw["id"]).get()
            raw_items = list(apify.dataset(run_kw["defaultDatasetId"]).iterate_items())
            step1_cost_usd += float(detail_kw.get("usageTotalUsd") or 0)

            normalized: list[dict] = []
            for row in raw_items:
                if not isinstance(row, dict):
                    continue
                n = normalize_keyword_search_item(row)
                if n is not None:
                    normalized.append(n)

            deduped = dedupe_keyword_items_by_shortcode(normalized)
            mode_posts, posts_age_stats = filter_items_within_max_age(
                deduped, posts_max_age_days
            )
            mode_dropped = int((posts_age_stats or {}).get("dropped_too_old") or 0)
            mode_kept_missing = int(
                (posts_age_stats or {}).get("kept_missing_timestamp") or 0
            )
            step1_age_dropped_client = (step1_age_dropped_client or 0) + mode_dropped
            step1_age_kept_missing_ts = (step1_age_kept_missing_ts or 0) + mode_kept_missing
            reels_age_stats = None
            for _it in mode_posts:
                _sc = (_it.get("shortCode") or "").strip()
                if _sc:
                    shortcode_region.setdefault(_sc, region)
            all_posts = merge_hashtag_items_by_shortcode(all_posts, mode_posts)
            source_counts["cookie_keywords"] = (
                source_counts.get("cookie_keywords", 0) + len(keywords)
            )
            modes_ran.append(discovery_mode)
            regions_ran.append(region)

            pipeline.log_run(
                actor_id=cookie_search_actor_id,
                run_id=run_kw["id"],
                status=run_kw["status"],
                input_params={
                    "keywords": len(keywords),
                    "maxPosts": max_posts,
                },
                items_count=len(raw_items),
                cost_usd=detail_kw.get("usageTotalUsd", 0),
                duration_ms=detail_kw.get("stats", {}).get("durationMillis"),
            )
            log.info(
                "step1_cookie_keyword_merge",
                region=region,
                raw_dataset_rows=len(raw_items),
                normalized=len(normalized),
                deduped=len(deduped),
                after_age_filter=len(mode_posts),
                merged_total=len(all_posts),
                posts_age_stats=posts_age_stats,
            )

    if not modes_ran:
        print(
            "FAILED: no Step 1 discovery modes ran "
            f"(configured: {format_step1_discovery_modes_label(discovery_modes)})."
        )
        issues.append(("Step 1", "all discovery modes skipped or failed to start"))
        return

    discovery_mode_report = ",".join(dict.fromkeys(modes_ran))
    regions_ran_unique = list(dict.fromkeys(regions_ran))
    searched_summary = build_step1_searched_summary(source_counts)
    notify_secondary_count = sum(source_counts.values())

    # Filter by min comments and register in DB
    new_posts = 0
    step1_new_post_items: list[dict] = []
    updated_posts = 0
    skipped_no_video_url = 0
    step1_skip_low_comments = 0
    step1_skip_no_shortcode = 0
    step1_existing_unchanged = 0
    # In-memory bridge to step 2: shortcode -> fresh IG videoUrl. Used by
    # the transcription fallback. Stored only for the lifetime of this
    # run because IG CDN URLs are signed and expire in ~1-2 days.
    post_videos: dict[str, str] = {}
    for p in all_posts:
        shortcode = (p.get("shortCode") or "").strip()
        comments_count = p.get("commentsCount") or 0
        if comments_count < min_comments:
            step1_skip_low_comments += 1
            continue
        if not shortcode:
            step1_skip_no_shortcode += 1
            continue

        is_reel = is_reel_payload(p)
        video_url = extract_video_url(p)
        if is_reel:
            if not is_valid_video_url(video_url):
                skipped_no_video_url += 1
                continue
            post_videos[shortcode] = video_url

        loc_label = post_location_label_from_item(p)

        existing = db.get_post(shortcode)

        if existing:
            update_fields: dict = {}
            if comments_count != (existing.get("comments_count") or 0):
                update_fields["comments_count"] = comments_count
            if loc_label is not None:
                update_fields["location"] = loc_label
            if update_fields:
                db.upsert_post(shortcode, **update_fields)
                if "comments_count" in update_fields:
                    updated_posts += 1
            else:
                step1_existing_unchanged += 1
        else:
            post_region = shortcode_region.get(shortcode)
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
                **({"location": loc_label} if loc_label is not None else {}),
                **({"region": post_region} if post_region is not None else {}),
            )
            new_posts += 1
            step1_new_post_items.append(p)

    log.info(
        "step1_done",
        discovery_modes=discovery_modes,
        modes_ran=modes_ran,
        active_regions=active_regions,
        regions_ran=regions_ran_unique,
        total_posts=len(all_posts),
        new=new_posts,
        updated=updated_posts,
        videos=len(post_videos),
        skipped_no_video_url=skipped_no_video_url,
        skip_low_comments=step1_skip_low_comments,
        skip_no_shortcode=step1_skip_no_shortcode,
        existing_unchanged=step1_existing_unchanged,
        cost=step1_cost_usd,
        posts_age_stats=posts_age_stats,
        reels_age_stats=reels_age_stats,
        posts_max_age_days=posts_max_age_days,
        age_dropped_client=step1_age_dropped_client,
        age_kept_missing_ts=step1_age_kept_missing_ts,
    )
    print(
        f"  DONE: fetched {len(all_posts)} posts "
        f"(new={new_posts}, updated={updated_posts}, "
        f"with_video={len(post_videos)}, skipped_no_video_url={skipped_no_video_url}) "
        f"cost=${step1_cost_usd:.4f}"
    )
    print(
        f"  Regions ran: {', '.join(regions_ran_unique) if regions_ran_unique else '(none)'}"
    )
    print("  Step 1 · gate breakdown:")
    print(
        f"    skipped (comments < min_comments_per_post={min_comments}): "
        f"{step1_skip_low_comments}"
    )
    print(f"    skipped (empty shortCode):              {step1_skip_no_shortcode}")
    print(
        f"    skipped (reel, no valid videoUrl):      {skipped_no_video_url}"
    )
    print(
        f"    already in DB, comments unchanged:     {step1_existing_unchanged}"
    )
    print(f"    already in DB, comments_count updated: {updated_posts}")
    print("  Step 1 · date filter:")
    for df_line in build_step1_date_filter_section_lines(
        discovery_mode=discovery_mode_report,
        posts_max_age_days=posts_max_age_days,
        age_dropped_client=step1_age_dropped_client,
        age_kept_missing_ts=step1_age_kept_missing_ts,
    ):
        print(f"    {df_line}")
    if len(all_posts) == 0:
        issues.append(("Step 1", step1_empty_issue))

    tg_notifier.notify_step1(
        new_posts,
        notify_secondary_count,
        discovery_mode=discovery_mode_report,
        full_message=build_step1_pipeline_summary_telegram_text(
            new_posts=new_posts,
            source_count=notify_secondary_count,
            discovery_mode=discovery_mode_report,
            searched_line=searched_summary,
            min_comments=min_comments,
            fetched_total=len(all_posts),
            updated_posts=updated_posts,
            with_video_count=len(post_videos),
            skipped_no_video_url=skipped_no_video_url,
            step1_skip_low_comments=step1_skip_low_comments,
            step1_skip_no_shortcode=step1_skip_no_shortcode,
            step1_existing_unchanged=step1_existing_unchanged,
            cost_usd=step1_cost_usd,
            posts_max_age_days=posts_max_age_days,
            age_dropped_client=step1_age_dropped_client,
            age_kept_missing_ts=step1_age_kept_missing_ts,
        ),
    )
    tg_notifier.notify_step1_new_posts(step1_new_post_items)

    # ============================================================
    # STEP 2: Language-gate posts, then score Russian text via DeepSeek.
    # ============================================================
    _banner("STEP 2: Score new posts (Lingua gate + DeepSeek)")
    with db._conn() as conn:
        unscored = conn.execute(
            "SELECT post_id, caption, post_url, location, region "
            "FROM processed_posts "
            "WHERE relevance IS NULL"
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
    non_russian_skipped = 0
    language_detect_failed = 0
    deepseek_calls = 0
    deepseek_failed = 0
    step2_is_re_audit: list[tuple[str, object]] = []
    human_confirm_queue: list[dict] = []

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
        post_link = (p.get("post_url") or "").strip()
        if not post_link:
            post_link = f"https://www.instagram.com/p/{post_id}/"

        # Nothing meaningful to send to DeepSeek (no caption / just
        # hashtags AND no usable transcript) -> mark unknown without
        # spending a DeepSeek call.
        if caption_is_empty(combined):
            empty_skipped += 1
            _apply_score(db, post_id, None)
            continue

        detected_language, russian_confidence, language_error = (
            detect_scoring_text_language(combined)
        )
        if language_error:
            language_detect_failed += 1
            log.warning(
                "step2_language_detection_failed",
                post_id=post_id,
                error=language_error,
            )
        elif detected_language != Language.RUSSIAN:
            non_russian_skipped += 1
            relevance = _apply_language_gate_irrelevant(db, post_id)
            detected_label = (
                detected_language.name if detected_language is not None else "None"
            )
            confidence_text = (
                f"{russian_confidence:.4f}"
                if russian_confidence is not None
                else "n/a"
            )
            tg_notifier.notify_step2_scored_post(
                post_url=post_link,
                raw_score={
                    "error": (
                        "skipped DeepSeek: Lingua detected "
                        f"{detected_label}; russian_confidence={confidence_text}"
                    )
                },
                resolved_relevance=relevance,
                combined_text=combined,
            )
            log.info(
                "step2_skipped_non_russian",
                post_id=post_id,
                detected_language=detected_label,
                russian_confidence=russian_confidence,
            )
            continue

        deepseek_calls += 1
        raw_score = score_caption(
            deepseek, combined, relevance_prompt=relevance_prompt
        )
        if "error" in raw_score:
            deepseek_failed += 1
        else:
            step2_is_re_audit.append((post_id, raw_score.get("is_real_estate")))

        relevance = _apply_score(db, post_id, raw_score)
        tg_notifier.notify_step2_scored_post(
            post_url=post_link,
            raw_score=raw_score,
            resolved_relevance=relevance,
            combined_text=combined,
        )

        if raw_score.get("is_real_estate") is True and "error" not in raw_score:
            human_confirm_queue.append(
                {
                    "post_id": post_id,
                    "post_link": post_link,
                    "combined": combined,
                    "raw_score": dict(raw_score),
                    "location": p.get("location"),
                    "region": p.get("region"),
                }
            )

    is_re_ctr = Counter(v for _, v in step2_is_re_audit)
    human_stats = {"approved": 0, "denied": 0, "timeout": 0}
    creds = tg_notifier.inline_confirm_token_and_chat()
    if human_confirm_queue and creds:
        token, fallback_chat_id = creds  # fallback = report chat
        # Route each post's confirmation to its region's result_chat
        # (falling back to the report chat for region=NULL or a region
        # without its own result_chat_id). Group by destination chat
        # so posts sharing a chat keep their original sequential ordering.
        chat_groups: dict[int, list[dict]] = {}
        for item in human_confirm_queue:
            chat = (
                region_result_chat_id(cfg, item.get("region"))
                or fallback_chat_id
            )
            chat_groups.setdefault(chat, []).append(item)
        for chat_id, group in chat_groups.items():
            group_stats = asyncio.run(
                _run_step2_human_confirmations(db, group, token, chat_id)
            )
            for k in human_stats:
                human_stats[k] += group_stats[k]
    elif human_confirm_queue and not creds:
        log.warning(
            "step2_human_confirm_skipped",
            reason="telegram_disabled_or_unconfigured",
            queued=len(human_confirm_queue),
        )

    log.info(
        "step2_done",
        scored=len(unscored),
        deepseek_calls=deepseek_calls,
        deepseek_failed=deepseek_failed,
        transcribed=transcribed,
        transcribe_failed=transcribe_failed,
        empty_skipped=empty_skipped,
        non_russian_skipped=non_russian_skipped,
        language_detect_failed=language_detect_failed,
        is_re_true=is_re_ctr.get(True, 0),
        is_re_false=is_re_ctr.get(False, 0),
        is_re_none=is_re_ctr.get(None, 0),
        human_confirm_queued=len(human_confirm_queue),
        human_confirm_approved=human_stats["approved"],
        human_confirm_denied=human_stats["denied"],
        human_confirm_timeout=human_stats["timeout"],
    )
    print(f"  DONE: scored {len(unscored)} "
          f"(deepseek_calls={deepseek_calls}, "
          f"transcribed={transcribed}, "
          f"transcribe_failed={transcribe_failed}, "
          f"empty_skipped={empty_skipped}, "
          f"non_russian_skipped={non_russian_skipped}, "
          f"language_detect_failed={language_detect_failed})")
    if human_confirm_queue:
        print(
            "  Step 2 human confirm: "
            f"queued={len(human_confirm_queue)} "
            f"approved={human_stats['approved']} "
            f"denied={human_stats['denied']} "
            f"timeout={human_stats['timeout']}"
        )
    nexara_attempts = transcribed + transcribe_failed
    tg_notifier.maybe_notify_nexara_batch_all_failed(
        transcription_attempts=nexara_attempts,
        transcribed_count=transcribed,
    )
    tg_notifier.maybe_notify_deepseek_batch_all_failed(
        deepseek_calls=deepseek_calls,
        deepseek_succeeded=deepseek_calls - deepseek_failed,
    )

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
    posts_to_scan = db.get_posts_needing_comments(min_growth_pct=comments_growth_pct)
    step3_new_commenters = 0

    if not posts_to_scan:
        print("  SKIPPED: no relevant posts in the queue.")
        log.info("step3_no_posts_to_scan")
    else:
        total_comments = sum(p.get("comments_count") or 0 for p in posts_to_scan)
        estimated_cost = total_comments * cost_per_comment

        log.info("step3_fetch_comments", posts=len(posts_to_scan),
                 total_comments=total_comments, estimated_cost=round(estimated_cost, 2))

        print(f"  Posts to scan:        {len(posts_to_scan)}")
        print(f"  Estimated comments:   {total_comments}")
        print(f"  Estimated cost:       ${estimated_cost:.2f}")
        if prompt_terminal_confirmation:
            confirm = input("  Proceed? (y/n): ").strip().lower()
        else:
            confirm = "y"

        if confirm == "y":
            urls = [p["post_url"] for p in posts_to_scan if p.get("post_url")]

            posts_over_comment_cap = [
                p
                for p in posts_to_scan
                if (p.get("comments_count") or 0) > louisdeconinck_cap
            ]
            if posts_over_comment_cap:
                tg_notifier.notify_step3_posts_over_comment_cap(
                    posts_over_comment_cap,
                    cap_per_post=louisdeconinck_cap,
                )

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

            step3_fetch_ok = True
            try:
                items, cost, source, debug = _fetch_comments_with_fallback(
                    apify,
                    pipeline,
                    urls,
                    primary_actor=primary_actor,
                    fallback_actor=fallback_actor,
                    louisdeconinck_cap_per_post=louisdeconinck_cap,
                    apidojo_cap_per_post=apidojo_cap,
                    tg_notifier=tg_notifier,
                )
            except Exception as e:
                step3_fetch_ok = False
                log.error("step3_fetch_comments_failed", error=str(e))
                print(f"FAILED: Step 3 comment fetch: {e}")
                tg_notifier.notify_step3_apify_call_error(e)
                issues.append(("Step 3", f"Apify comment fetch error: {e}"))

            # Bail out *before* marking anything as scanned if both the
            # primary and the fallback returned an empty dataset.
            # Background: louisdeconinck has been observed to silently
            # "succeed" with 0/null comments per page (its own log says
            # ``fetched 0/null comments``). Marking those posts as
            # scanned would freeze them out of the queue until comments
            # grow another ``comments_growth_pct`` — i.e. silently lose
            # tens of thousands of real commenters. Treat as a transient
            # failure and leave the queue untouched so the next run
            # retries them.
            if step3_fetch_ok and source == "both-empty":
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
            elif step3_fetch_ok:
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

                # Build media_id -> post mapping via shortcode. Each post
                # carries its region (from processed_posts) so the lead and
                # lead_post_link rows inherit it -- this is what lets Step 5
                # route a lead's result to the right per-region chat.
                post_lookup = {}
                for p in posts_to_scan:
                    sc = p.get("shortcode")
                    if sc:
                        post_lookup[shortcode_to_id(sc)] = (
                            p["post_url"],
                            sc,
                            p.get("region"),
                        )

                media_to_post = {}
                for c in unique.values():
                    mid = c.get("media_id")
                    if not mid:
                        continue
                    mid_str = str(mid)
                    if mid_str in media_to_post:
                        continue
                    for real_id, (url, sc, reg) in post_lookup.items():
                        if abs(real_id - mid) < 1000:
                            media_to_post[mid_str] = (url, sc, reg)
                            break

                # Save leads
                new_leads = 0
                for c in unique.values():
                    user = c.get("user", {})
                    username = user.get("username")
                    if not username:
                        continue
                    uid = str(user.get("pk", ""))

                    mid_str = str(c.get("media_id", ""))
                    post_info = media_to_post.get(mid_str)
                    lead_region = post_info[2] if post_info else None

                    is_new = db.add_lead_account(
                        username=username,
                        user_id=uid,
                        full_name=user.get("full_name", ""),
                        profile_pic_url=user.get("profile_pic_url", ""),
                        is_private=1 if user.get("is_private") else 0,
                        is_verified=1 if user.get("is_verified") else 0,
                        **({"region": lead_region} if lead_region is not None else {}),
                    )
                    if is_new:
                        new_leads += 1

                    if post_info:
                        db.add_lead_post_link(
                            username=username,
                            post_url=post_info[0],
                            user_id=uid,
                            post_shortcode=post_info[1],
                            comment_pk=str(c.get("pk") or ""),
                            comment_text=c.get("text", "")[:500],
                            comment_at=str(c.get("created_at_utc", "")),
                            **(
                                {"region": lead_region}
                                if lead_region is not None
                                else {}
                            ),
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
                step3_new_commenters = new_leads
        else:
            log.info("step3_skipped")
            print("  SKIPPED by user.")

    tg_notifier.notify_step3(step3_new_commenters)

    # ============================================================
    # STEP 4: Fetch profiles for new leads
    # ============================================================
    _banner("STEP 4: Fetch profiles for new leads")
    leads_to_fetch = db.get_leads_without_profile(limit=step4_batch_limit)
    profiles_queued = 0
    single_face_new = 0
    fallback_resolved = 0
    no_suitable_photo = 0
    contacts_found = 0

    if not leads_to_fetch:
        print("  SKIPPED: no leads without profile.")
        log.info("step4_no_profiles_to_fetch")
    else:
        usernames = [l["username"] for l in leads_to_fetch]
        profiles_queued = len(usernames)
        print(f"  Leads to fetch:  {profiles_queued}")
        log.info("step4_fetch_profiles", count=profiles_queued)

        profiles_fetched = 0
        avatars_downloaded = 0
        fallback_skipped = 0

        for i in range(0, len(usernames), profile_batch_size):
            batch = usernames[i:i + profile_batch_size]

            try:
                run = apify.actor("apify/instagram-profile-scraper").call(
                    run_input={"usernames": batch}
                )
            except Exception as e:
                log.error(
                    "step4_fetch_profiles_failed",
                    batch_size=len(batch),
                    error=str(e),
                )
                print(f"FAILED: Step 4 profile fetch (batch): {e}")
                tg_notifier.notify_step4_apify_call_error(e)
                issues.append(("Step 4", f"Apify profile fetch error: {e}"))
                continue
            tg_notifier.maybe_notify_apify_run_failure(
                run,
                actor_id="apify/instagram-profile-scraper",
                step="Step 4",
            )
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
                        avatar_faces = avatar_embedder.embed_faces(avatar_path)
                        faces_count = len(avatar_faces)
                        db.update_lead_avatar(username, avatar_path, faces_count)
                        final_face_path: str | None = None

                        avatar_area_ok = False
                        if faces_count == 1:
                            img_bgr = cv2.imread(str(avatar_path))
                            if img_bgr is not None:
                                ih, iw = img_bgr.shape[:2]
                                area_pct, _, _ = face_bbox_percent_of_image(
                                    avatar_faces[0].bbox, iw, ih
                                )
                                avatar_area_ok = area_pct >= min_avatar_face_area_pct

                        if faces_count == 1 and avatar_area_ok:
                            single_face_new += 1
                            db.update_lead_face(username, avatar_path)
                            final_face_path = str(avatar_path)
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
                                final_face_path = str(result.photo_path)
                            else:
                                fallback_skipped += 1
                                no_suitable_photo += 1

                            if not fb_keep_photos:
                                cleanup_lead_photos(
                                    uid_str,
                                    keep=(result.photo_path if result else None),
                                )
                        else:
                            no_suitable_photo += 1

                        _reconcile_step4_ephemeral_avatar(
                            db,
                            log,
                            username=username,
                            downloaded_avatar_path=str(avatar_path),
                            final_face_path=final_face_path,
                        )
                    else:
                        no_suitable_photo += 1
                else:
                    no_suitable_photo += 1

                contacts = extract_contacts(
                    bio=p.get("biography"),
                    external_url=p.get("externalUrl"),
                    external_urls=p.get("externalUrls"),
                )
                if any(v for v in contacts.values()):
                    #db.update_lead_contacts(username=username, **{k: v for k, v in contacts.items() if v})
                    contacts_found += 1

        log.info(
            "step4_done",
            profiles=profiles_fetched,
            contacts_from_bio=contacts_found,
            avatars=avatars_downloaded,
            single_face=single_face_new,
            fallback_resolved=fallback_resolved,
            fallback_skipped=fallback_skipped,
            no_suitable_photo=no_suitable_photo,
        )
        print(f"  DONE: profiles={profiles_fetched} "
              f"contacts={contacts_found} "
              f"avatars={avatars_downloaded} "
              f"single_face={single_face_new} "
              f"fallback_resolved={fallback_resolved} "
              f"fallback_skipped={fallback_skipped} "
              f"no_suitable_photo={no_suitable_photo}")

    tg_notifier.notify_step4(
        profiles_queued=profiles_queued,
        single_face_avatar=single_face_new,
        face_leader_resolved=fallback_resolved,
        without_suitable_photo=no_suitable_photo,
        contacts_from_bio=contacts_found,
    )

    # ============================================================
    # STEP 5: Resolve Telegram contacts via Sherlock
    # ============================================================
    # Only runs for "naked" leads -- profile fetched but bio gave us
    # no phone / telegram. Step 4's contact_extractor wins ties; we
    # don't overwrite anything it found. Skipped entirely under
    # --skip-sherlock or if the SHERLOCK_API_KEY is missing.
    #
    # Pre-flight: GET /v1/health (up to SHERLOCK_HEALTH_PROBE_MAX_ATTEMPTS
    # retries). Skip Step 5 when pool.by_status.idle is 0 or the API never
    # answers; alerts go to telegram.alert_chat_id (not the terminal).
    if args.skip_sherlock:
        _banner("STEP 5: Resolve contacts via Sherlock")
        print("  SKIPPED by --skip-sherlock.")
        log.info("step5_skipped_by_flag")
    else:
        _step_5_resolve_contacts_via_sherlock(
            db,
            cfg,
            batch_limit=sherlock_batch_limit,
            workers_override=args.workers,
            sequential=sherlock_sequential,
            request_gap_secs=sherlock_request_gap_secs,
            auto_yes=args.yes or not prompt_terminal_confirmation,
            log=log,
            issues=issues,
            tg_notifier=tg_notifier,
            deepseek=deepseek,
            usermatch_prompt=usermatch_prompt,
            health_probe_max_attempts=SHERLOCK_HEALTH_PROBE_MAX_ATTEMPTS,
        )

    # ============================================================
    # STEP 6: Cleanup spent face assets
    # ============================================================
    # Runs even under --skip-sherlock to drain the backlog of leads
    # already Sherlock'd in prior runs. --keep-photos disables it
    # for debugging / forensic work.

    if args.keep_photos:
        _banner("STEP 6: Cleanup spent face assets")
        print("  SKIPPED by --keep-photos.")
        log.info("step6_skipped_by_flag")
    else:
        _step_6_cleanup_spent_face_assets(
            db,
            log=log,
            issues=issues,
        )

    # ============================================================
    # SUMMARY
    # ============================================================
    stats_after = db.get_stats()
    ps = pipeline.summary()

    _banner("PIPELINE COMPLETE")
    realtor_total = sum(
        len(region_realtor_accounts(cfg, r)) for r in active_regions
    )
    print(
        "Realtor accounts (config, active regions): "
        f"{realtor_total}"
    )
    print(f"Leads total:          {stats_after['leads_total']} (+{stats_after['leads_total'] - stats_before['leads_total']})")
    print(f"  with profile:       {stats_after['leads_with_profile']}")
    print(f"  with contacts:      {stats_after['leads_with_contacts']} "
          f"(+{stats_after['leads_with_contacts'] - stats_before['leads_with_contacts']})")
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
        if prompt_terminal_confirmation:
            try:
                input("\nPress Enter to exit... ")
            except EOFError:
                pass

    avatar_embedder.close()
    post_embedder.close()


if __name__ == "__main__":
    main()
