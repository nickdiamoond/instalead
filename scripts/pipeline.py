"""Daily lead collection pipeline.

Steps:
  1. Fetch recent posts/reels (realtor accounts from config, hashtags, or
     cookie keyword search — ``pipeline.step1.discovery_mode``)
  2. Score new posts via DeepSeek (relevance + CTA)
  3. Fetch comments for relevant posts (new + grown)
  4. Fetch profiles for new leads, extract contacts from bio
  5. Resolve Telegram contacts for naked leads via Sherlock
     (nick search first, photo fallback with exact-match or DeepSeek pick)

Uses DB for deduplication — safe to run repeatedly.
"""

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv
from lingua import Language, LanguageDetectorBuilder
from openai import OpenAI

from src.avatar_downloader import (
    cleanup_lead_face_assets,
    cleanup_lead_photos,
    download_avatar,
    download_post_photos,
)
from src.comment_normalizer import normalize_apidojo_api
from src.config import (
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
from src.logger import get_logger, setup_logging
from src.pipeline_logger import PipelineLogger
from src.sherlock_client import SherlockError, make_sherlock_client
from src.telegram_inline_confirm import await_single_yes_no
from src.telegram_notifier import (
    PipelineTelegramNotifier,
    STEP2_INLINE_BTN_CONFIRM,
    STEP2_INLINE_BTN_DENY,
    STEP2_INLINE_SUFFIX_APPROVED,
    STEP2_INLINE_SUFFIX_DENIED,
    build_step1_date_filter_section_lines,
    build_step1_pipeline_summary_telegram_text,
    build_step2_human_confirm_body,
    truncate_step2_human_confirm_body,
)
from src.transcriber import NexaraTranscriber

setup_logging()
log = get_logger("pipeline")

# Per-step tuning knobs. These are *defaults*; ``main()`` overrides
# every value from ``config.yaml`` (``pipeline.stepN.*``) so the daily
# run picks up changes without a code edit. Constants are kept in the
# module (rather than only in YAML) so direct imports / unit tests
# don't have to pull in the config loader to know reasonable values.
#
# Don't touch these to "tune the pipeline"; edit ``config.yaml``
# instead. They exist solely to keep the script bootable when a key
# is missing from a fresh config (e.g. on a brand-new machine before
# the operator has copied the canonical YAML over).

# Step 1: numeric defaults for ``posts_max_age_days`` and
# ``min_comments_per_post`` live in ``src.config`` (``step1_*`` helpers).
# ``apify/instagram-post-scraper`` input ``resultsLimit`` (per username).
# Override via ``pipeline.step1.post_scraper_results_limit``.
DEFAULT_POST_SCRAPER_RESULTS_LIMIT = 20
# Step 1: ``pipeline.step1.discovery_mode`` — ``realtors`` | ``hashtags`` |
# ``cookie_keywords``.
DEFAULT_STEP1_DISCOVERY_MODE = "realtors"

# Step 3: comment re-scan growth threshold + the displayed cost
# estimate per fetched comment (real bill comes from
# ``run.usageTotalUsd`` regardless of this value).
DEFAULT_COMMENTS_GROWTH_PCT = 5.0
DEFAULT_COST_PER_COMMENT = 0.0005

# Step 4: profile-scraper batch size + max new leads pulled per run.
DEFAULT_PROFILE_BATCH_SIZE = 50
DEFAULT_STEP4_BATCH_LIMIT = 1000
# Minimum bbox area (percent of full raster) to accept the avatar as the
# canonical face photo; below this, Step 4 uses the post-photo leader path.
DEFAULT_MIN_AVATAR_FACE_AREA_PCT = 2.0

# Step 5: max leads pulled per run from get_leads_for_sherlock.
# Smaller than Step 4's effective rate because Sherlock tasks are
# slow (~30 s nick / ~135 s photo each), so 1000 leads on a 3-account
# pool is already a multi-hour run; bigger pools could safely raise
# this. The daily run has no CLI flag for it on purpose, to keep
# ``python scripts/pipeline.py`` behavior reproducible across
# machines / cron jobs -- override via config.yaml.
DEFAULT_SHERLOCK_BATCH_LIMIT = 1000
DEFAULT_SHERLOCK_SEQUENTIAL = True
DEFAULT_SHERLOCK_REQUEST_GAP_SECS = 5.0

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
# ``_fetch_comments_with_fallback``. Override via
# ``pipeline.step3.louisdeconinck_comments_cap_per_post`` in config.
DEFAULT_LOUISDECONINCK_COMMENTS_CAP_PER_POST = 10_000

# When true, Step 3 / Step 5 ask ``Proceed? (y/n)`` before spendy work, and
# the script waits for Enter after reporting issues. Set false in config for
# cron / unattended runs (``pipeline.prompt_terminal_confirmation``).
DEFAULT_PROMPT_TERMINAL_CONFIRMATION = True


def _cfg_prompt_terminal_confirmation(value, default: bool = True) -> bool:
    """Parse ``pipeline.prompt_terminal_confirmation`` (bool / int / str)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("false", "no", "n", "0", "off"):
            return False
        if s in ("true", "yes", "y", "1", "on"):
            return True
        return default
    return default


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

RUSSIAN_LANGUAGE_DETECTOR = (
    LanguageDetectorBuilder.from_all_spoken_languages().build()
)


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


def face_bbox_percent_of_image(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float]:
    """BBox vs full raster: ``(area_percent, width_percent, height_percent)``."""
    x1, y1, x2, y2 = bbox
    bw = max(0.0, float(x2 - x1))
    bh = max(0.0, float(y2 - y1))
    iw = float(image_width)
    ih = float(image_height)
    if iw <= 0.0 or ih <= 0.0:
        return (0.0, 0.0, 0.0)
    area_pct = 100.0 * (bw * bh) / (iw * ih)
    w_pct = 100.0 * bw / iw
    h_pct = 100.0 * bh / ih
    return (area_pct, w_pct, h_pct)


def _same_disk_face_file(path_a: str, path_b: str) -> bool:
    """True if both strings refer to the same on-disk file."""
    try:
        return Path(path_a).resolve() == Path(path_b).resolve()
    except OSError:
        return os.path.normcase(os.path.abspath(path_a)) == os.path.normcase(
            os.path.abspath(path_b)
        )


def _reconcile_step4_ephemeral_avatar(
    db: LeadDB,
    log,
    *,
    username: str,
    downloaded_avatar_path: str,
    final_face_path: str | None,
) -> None:
    """Drop avatar file if it is not the canonical ``face_photo_path`` target.

    When the avatar *is* the canonical photo, keep the file and ``avatar_path``
    until Step 6 post-Sherlock cleanup.
    """
    if final_face_path is not None and _same_disk_face_file(
        downloaded_avatar_path, final_face_path
    ):
        return
    p = Path(downloaded_avatar_path)
    try:
        p.unlink(missing_ok=True)
    except OSError as e:
        log.warning(
            "step4_avatar_unlink_failed",
            username=username,
            path=str(p),
            error=str(e),
        )
    db.clear_lead_avatar_path(username)
    reason = "no_canonical_face" if final_face_path is None else "canonical_elsewhere"
    log.info("step4_avatar_disk_released", username=username, reason=reason)


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
                {"role": "user", "content": caption[:3000]},
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


def detect_scoring_text_language(
    text: str,
) -> tuple[Language | None, float | None, str | None]:
    """Return Lingua's best guess plus Russian confidence for Step 2.

    ``error`` is reserved for detector/runtime failures. A ``None``
    language with ``error=None`` means Lingua could not decide reliably.
    """
    try:
        detected = RUSSIAN_LANGUAGE_DETECTOR.detect_language_of(text)
        russian_confidence = RUSSIAN_LANGUAGE_DETECTOR.compute_language_confidence(
            text, Language.RUSSIAN
        )
        return detected, russian_confidence, None
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"


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


def _apply_language_gate_irrelevant(db: LeadDB, post_id: str) -> str:
    """Persist Step 2 language-gate rejection as irrelevant."""
    db.upsert_post(
        post_id, relevance="irrelevant", has_cta=0, cta_type="none"
    )
    return "irrelevant"


def _apply_human_irrelevant_override(db: LeadDB, post_id: str, raw_score: dict) -> None:
    """Operator rejected ``is_real_estate=True``; force irrelevant, keep CTA columns."""
    has_cta = 1 if raw_score.get("has_call_to_action") else 0
    cta_type = raw_score.get("call_to_action_type") or "none"
    db.upsert_post(
        post_id, relevance="irrelevant", has_cta=has_cta, cta_type=cta_type
    )


async def _run_step2_human_confirmations(
    db: LeadDB,
    items: list[dict],
    token: str,
    chat_id: int,
) -> dict[str, int]:
    """Sequential inline confirm per post; ``2s`` pause between items."""
    approved = 0
    denied = 0
    timed_out = 0
    total = len(items)
    for i, item in enumerate(items, start=1):
        body = build_step2_human_confirm_body(
            index=i,
            total=total,
            post_url=str(item.get("post_link") or ""),
            combined_text=str(item.get("combined") or ""),
        )
        text = truncate_step2_human_confirm_body(body)
        result = await await_single_yes_no(
            token,
            chat_id,
            text,
            confirm_button_text=STEP2_INLINE_BTN_CONFIRM,
            deny_button_text=STEP2_INLINE_BTN_DENY,
            suffix_yes=STEP2_INLINE_SUFFIX_APPROVED,
            suffix_no=STEP2_INLINE_SUFFIX_DENIED,
        )
        if result == "no":
            _apply_human_irrelevant_override(
                db, str(item["post_id"]), item["raw_score"]
            )
            denied += 1
        elif result == "yes":
            approved += 1
        else:
            timed_out += 1
            log.warning(
                "step2_human_confirm_timeout",
                post_id=item.get("post_id"),
                index=i,
                total=total,
            )
        if i < total:
            await asyncio.sleep(2.0)
    return {"approved": approved, "denied": denied, "timeout": timed_out}


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


def _realtor_usernames_from_cfg(cfg: dict) -> list[str]:
    """Instagram usernames for Step 1 ``discovery_mode=realtors``.

    Reads ``search.realtor_accounts`` from config (same contract as
    ``search.hashtags`` for the hashtag path): non-strings skipped,
    stripped, empties dropped, duplicates removed with order preserved.
    """
    raw = list((cfg.get("search") or {}).get("realtor_accounts") or [])
    out: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue
        u = x.strip()
        if u:
            out.append(u)
    return list(dict.fromkeys(out))


def _run_apify_actor(
    apify: ApifyClient,
    pipeline: PipelineLogger,
    actor_id: str,
    run_input: dict,
    *,
    log_input: dict | None = None,
    tg_notifier: PipelineTelegramNotifier | None = None,
    apify_step: str | None = None,
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
    if tg_notifier is not None and apify_step:
        tg_notifier.maybe_notify_apify_run_failure(
            run, actor_id=actor_id, step=apify_step
        )
    return items, cost, run


def _fetch_comments_with_fallback(
    apify: ApifyClient,
    pipeline: PipelineLogger,
    urls: list[str],
    *,
    primary_actor: str,
    fallback_actor: str,
    louisdeconinck_cap_per_post: int,
    tg_notifier: PipelineTelegramNotifier | None = None,
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

    The actor ids and the per-post comment cap are passed in (rather
    than read from module-level constants) so ``main()`` can override
    them from ``config.yaml`` (``pipeline.step3.*``) without touching
    this function.
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
    #   ``louisdeconinck_cap_per_post`` is set well above any
    #   per-post comment count we've ever seen, so it acts as a
    #   safety ceiling rather than a real cap.
    primary_items, primary_cost, primary_run = _run_apify_actor(
        apify,
        pipeline,
        primary_actor,
        run_input={
            "urls": urls,
            "proxy": {"useApifyProxy": True},
            "resultsLimit": louisdeconinck_cap_per_post,
            "maxComments": louisdeconinck_cap_per_post,
        },
        log_input={
            "urls_count": len(urls),
            "results_limit": louisdeconinck_cap_per_post,
        },
        tg_notifier=tg_notifier,
        apify_step="Step 3 (comments primary)",
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
        tg_notifier=tg_notifier,
        apify_step="Step 3 (comments fallback)",
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


# =========================================================================
# Step 5: Sherlock contact resolution (parallel)
# =========================================================================
#
# For each "naked" lead (profile fetched, but bio gave us no contact),
# we try Sherlock in two stages -- nick first because it's cheap and
# definitive, photo only as fallback because it's slow:
#
#   1) POST /v1/search/nick { nick = ig_username, search_in = "telegram" }
#      If `result.results` is non-empty, the IG handle exists in TG and
#      we save the matched username + t.me link. Done.
#
#   2) Else, if face_photo_path is set, POST /v1/search/photo with the
#      file. If ``results[0].status`` contains the Cyrillic substring
#      "точное совпадение", we save ``phone`` + ``link`` from that row.
#      Otherwise we collect every result row with a non-null ``person``,
#      format names for DeepSeek (same prompt as
#      ``scripts/test_profile_face_pick.py``), and save ``phone`` + ``link``
#      only for the candidate the model picks (confidence gate in the
#      prompt). No pick / no candidates / missing DeepSeek client →
#      ``no_match`` without writing contacts.
#
# By default Step 5 runs one lead at a time (sequential) with a short
# pause between leads; set ``pipeline.step5.sequential: false`` to fan
# out across a thread pool again. DB writes stay on the main thread.

# Sherlock outcome labels stored in lead_accounts.sherlock_status. Kept
# as a centralized vocabulary so the pipeline summary banner and
# downstream tooling can rely on a closed set of values.
SH_STATUS_FOUND_NICK = "found_nick"
SH_STATUS_FOUND_PHOTO = "found_photo"
SH_STATUS_NO_MATCH = "no_match"
SH_STATUS_NO_FACE_PHOTO = "no_face_photo"
SH_STATUS_ERROR = "error"

# First-row ``status`` substring for Sherlock photo ``result.results``;
# mirrors ``scripts/test_profile_face_pick.py``.
SHERLOCK_EXACT_MATCH_SUBSTRING = "точное совпадение"
USERMATCH_PROMPT = """\
#Задача: 
Ты анализируешь Ник пользователя из Instagram(username) и его ФИО из профиля(если оно присутствует). А также пронумерованный список потенциальных кандидатов.
Ты должен определить, кому из кандидатов принадлежит этот аккаунт.
Может оказатья так, что ник пользователя из Instagram и его ФИО не принадлежат ни одному кандидату.
Иногда бывает так, что ник ползователя(username) содержит в себе как минимум частично имя или фамилию кандидата. И если ФИО из профиля пустое, ты должен сопоставить юзернейм с кандидатами. Но если ФИО пользователя из Instagram присутствует, отдавай приоритет ему.

#Данные:
Ник пользователя из Instagram: "{username}"
ФИО пользователя из Instagram: "{full_name}"

Список потенциальных кандидатов: {candidates}

#Формат ответа:
В ответе ты должен указать номер кандидата, которому принадлежит этот аккаунт только в том случае, если ты уверен, что этот аккаунт принадлежит этому кандидату минимум на 7/10. Если ты не уверен, отдай 0.
Если ты нашел совпадение, но в списке потенциальных кандидатов есть одинаковые ФИО, отдай номер первого из них.
Ответь ТОЛЬКО одной цифрой, которая соответствует номеру кандидата, либо 0, если ник пользователя из Instagram и его ФИО не принадлежат ни одному кандидату.
"""


def _sherlock_photo_results_list(result: dict | None) -> list:
    """Normalize ``result["results"]`` for photo search (dict or list)."""
    if not result or not isinstance(result, dict):
        return []
    raw = result.get("results")
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    return []


def _person_for_digest_list(person: object) -> object:
    """Keep the substring before the last space (trim trailing tokens like a DOB)."""
    if not isinstance(person, str):
        return person
    if " " not in person:
        return person
    return person.rsplit(" ", 1)[0]


def _format_candidates_for_prompt(persons: list) -> str:
    """``1) `` + name + ``\\n`` for each entry (1-based)."""
    return "".join(f"{i}) {p}\n" for i, p in enumerate(persons, start=1))


def _parse_usermatch_digit(raw: str) -> int | None:
    """First signed integer in model output, or ``None`` if unparseable."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    m = re.search(r"-?\d+", text)
    if not m:
        return None
    return int(m.group(0))


def _deepseek_usermatch_pick_index(
    client: OpenAI,
    *,
    ig_username: str,
    ig_full_name: str,
    persons: list,
) -> tuple[int | None, str | None]:
    """Return ``(pick, api_error)``.

    * ``pick`` — 1-based candidate index, or ``None`` if the model declines
      (digit ``0``) or the call failed.
    * ``api_error`` — set when the HTTP call or response parsing failed;
      ``None`` when the API returned a usable answer (including decline).
    """
    candidates_block = _format_candidates_for_prompt(persons)
    system_prompt = USERMATCH_PROMPT.format(
        username=ig_username,
        full_name=ig_full_name,
        candidates=candidates_block,
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Ответь одной цифрой."},
            ],
            temperature=0,
            max_tokens=16,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "step5_deepseek_usermatch_failed",
            username=ig_username,
            error=str(exc),
        )
        return None, str(exc)

    pick = _parse_usermatch_digit(raw)
    if pick is None:
        log.warning(
            "step5_deepseek_usermatch_unparseable",
            username=ig_username,
            raw=raw[:200],
        )
        return None, f"unparseable response: {raw[:200]}"
    if pick == 0:
        log.info(
            "step5_deepseek_usermatch_zero",
            username=ig_username,
        )
        return None, None
    if pick < 1 or pick > len(persons):
        log.warning(
            "step5_deepseek_usermatch_out_of_range",
            username=ig_username,
            pick=pick,
            n=len(persons),
            raw=raw[:200],
        )
        return None, f"out of range: pick={pick} n={len(persons)}"
    return pick, None


def _resolve_one_lead_via_sherlock(
    sherlock,
    lead: dict,
    *,
    nick_cfg: dict,
    photo_cfg: dict,
    task_cfg: dict,
    deepseek: OpenAI | None = None,
) -> dict:
    """Run the full nick->photo flow for one lead.

    Pure function w.r.t. the DB: returns a dict that the orchestrator
    persists via :py:meth:`LeadDB.mark_lead_sherlock`. Never raises --
    every exception path resolves to ``status=error`` with a populated
    ``error`` message so a single misbehaving lead doesn't sink the
    whole batch.

    Photo stage: if ``results[0].status`` contains "точное совпадение",
    ``phone`` / ``link`` are taken from that row. Otherwise every row
    with a non-null ``person`` is sent to DeepSeek (``USERMATCH_PROMPT``)
    with the lead's Instagram ``username`` and ``full_name``; the model
    picks at most one candidate. Missing client, no candidates, or no
    confident pick → ``no_match`` without contact fields.

    Returned shape (orchestrator + Telegram):
      {
        "username": str,
        "status":   str,           # SH_STATUS_* label
        "telegram_username": str | None,
        "phone":            str | None,
        "sherlock_link":    str | None,
        "error":            str | None,   # debug message, not stored
        "nick_skipped_dot": bool,
        "nick_search_ran": bool,
        "nick_hit": bool,
        "photo_search_ran": bool,
        "photo_task": dict | None,       # snapshot after photo wait_for_task
        "nick_query": str | None,
      }
    """
    username = lead["username"]
    out: dict = {
        "username": username,
        "status": SH_STATUS_ERROR,
        "telegram_username": None,
        "phone": None,
        "sherlock_link": None,
        "error": None,
        "nick_skipped_dot": "." in username,
        "nick_search_ran": False,
        "nick_hit": False,
        "photo_search_ran": False,
        "photo_task": None,
        "nick_query": None if "." in username else f"@{username}",
    }
    poll_interval = float(task_cfg.get("poll_interval_secs", 3))
    max_wait = float(task_cfg.get("max_wait_secs", 300))

    # ---- Stage 1: nick search ------------------------------------
    # Nick is invalid for Sherlock if it contains dot. In that case we
    # skip nick stage entirely and go straight to photo fallback.
    if "." not in username:
        # Sherlock expects the leading '@' in nick queries.
        nick_query = f"@{username}"
        out["nick_search_ran"] = True
        try:
            enq = sherlock.enqueue_nick(
                nick=nick_query,
                search_in="telegram",
                max_pages=int(nick_cfg.get("max_pages", 1)),
                max_attempts=int(nick_cfg.get("max_attempts", 3)),
            )
            task = sherlock.wait_for_task(
                enq["id"],
                poll_interval=poll_interval,
                max_wait=max_wait,
            )
            if (task.get("status") or "").lower() == "completed":
                results = ((task.get("result") or {}).get("results")) or []
                if isinstance(results, dict):
                    results = [results]
                # Nick hit is valid ONLY when profile_url exists.
                match = next(
                    (
                        item for item in results
                        if isinstance(item, dict) and item.get("profile_url")
                    ),
                    None,
                )
                if match:
                    # Prefer the result's reported handle (case might
                    # differ from the IG one, e.g. CamelCase). Fall back
                    # to the queried IG username if missing.
                    tg_username = match.get("username") or username
                    tg_link = (
                        match.get("link")
                        or f"https://t.me/{tg_username}"
                    )
                    out.update({
                        "status": SH_STATUS_FOUND_NICK,
                        "telegram_username": tg_username,
                        "sherlock_link": tg_link,
                        "nick_hit": True,
                    })
                    return out
            # Anything else (failed / timeout / completed-but-empty) -> fall
            # through to photo. We don't return early on a nick failure
            # because photo might still rescue the lead.
        except (SherlockError, TimeoutError) as exc:
            # Best-effort: log and try photo anyway. If photo also fails
            # the lead ends up as `error` further down.
            out["error"] = f"nick: {exc}"
        except Exception as exc:  # noqa: BLE001 - never let a thread die
            out["error"] = f"nick: unexpected {type(exc).__name__}: {exc}"

    # ---- Stage 2: photo fallback ---------------------------------
    face_path_str = lead.get("face_photo_path")
    if not face_path_str:
        # Spec: skip leads without a usable photo. They keep
        # `sherlock_processed_at` set so we don't retry next run.
        out["status"] = SH_STATUS_NO_FACE_PHOTO
        # Clear the nick-stage error message: the *outcome* is
        # cleanly "no face photo to try", not an error.
        out["error"] = None
        return out

    face_path = Path(face_path_str)
    if not face_path.is_file():
        # Path is recorded but the file is gone (manual cleanup,
        # disk hiccup). Treat the same as "no photo".
        out["status"] = SH_STATUS_NO_FACE_PHOTO
        out["error"] = f"face_photo missing on disk: {face_path}"
        return out

    try:
        enq = sherlock.enqueue_photo(
            face_path,
            max_pages=int(photo_cfg.get("max_pages", 20)),
            max_attempts=int(photo_cfg.get("max_attempts", 3)),
        )
        task = sherlock.wait_for_task(
            enq["id"],
            poll_interval=poll_interval,
            max_wait=max_wait,
        )
        out["photo_search_ran"] = True
        out["photo_task"] = copy.deepcopy(task)
        final_status = (task.get("status") or "").lower()
        if final_status != "completed":
            out["status"] = SH_STATUS_ERROR
            out["error"] = (
                task.get("error_message")
                or f"photo task ended status={final_status!r}"
            )
            return out
        result_block = task.get("result") or {}
        results = _sherlock_photo_results_list(result_block)
        if not results:
            out["status"] = SH_STATUS_NO_MATCH
            out["error"] = None
            return out
        first_raw = results[0]
        first = first_raw if isinstance(first_raw, dict) else {}
        first_status = str(first.get("status") or "")

        if SHERLOCK_EXACT_MATCH_SUBSTRING in first_status:
            out.update({
                "status": SH_STATUS_FOUND_PHOTO,
                "phone": first.get("phone"),
                "sherlock_link": first.get("link"),
                "error": None,
                "photo_match_kind": "exact",
                "sherlock_person": first.get("person"),
            })
            log.info(
                "step5_sherlock_photo_outcome",
                username=username,
                branch="exact_match",
            )
            return out

        persons: list = []
        raw_persons: list = []
        phones: list = []
        links: list = []
        for item in results:
            if not isinstance(item, dict):
                continue
            if "person" not in item or item.get("person") is None:
                continue
            persons.append(_person_for_digest_list(item.get("person")))
            raw_persons.append(item.get("person"))
            phones.append(item.get("phone"))
            links.append(item.get("link"))

        if not persons:
            out["status"] = SH_STATUS_NO_MATCH
            out["error"] = None
            log.info(
                "step5_sherlock_photo_outcome",
                username=username,
                branch="no_person_candidates",
            )
            return out

        if deepseek is None:
            out["status"] = SH_STATUS_NO_MATCH
            out["error"] = None
            log.warning(
                "step5_sherlock_photo_no_deepseek_client",
                username=username,
            )
            return out

        pick, deepseek_api_error = _deepseek_usermatch_pick_index(
            deepseek,
            ig_username=username,
            ig_full_name=str(lead.get("full_name") or ""),
            persons=persons,
        )
        out["step5_deepseek_called"] = True
        out["step5_deepseek_api_failed"] = deepseek_api_error is not None
        if pick is None:
            out["status"] = SH_STATUS_NO_MATCH
            out["error"] = None
            log.info(
                "step5_sherlock_photo_outcome",
                username=username,
                branch="no_contact_after_disambiguation",
            )
            return out

        idx = pick - 1
        out.update({
            "status": SH_STATUS_FOUND_PHOTO,
            "phone": phones[idx],
            "sherlock_link": links[idx],
            "error": None,
            "photo_match_kind": "deepseek",
            "sherlock_person": raw_persons[idx],
        })
        log.info(
            "step5_sherlock_photo_outcome",
            username=username,
            branch="deepseek_pick",
            pick=pick,
        )
        return out
    except (SherlockError, TimeoutError) as exc:
        out["status"] = SH_STATUS_ERROR
        out["error"] = f"photo: {exc}"
        return out
    except Exception as exc:  # noqa: BLE001
        out["status"] = SH_STATUS_ERROR
        out["error"] = f"photo: unexpected {type(exc).__name__}: {exc}"
        return out


def _format_eta(seconds: float) -> str:
    """Render a duration as ``Xh Ym`` / ``Ym Zs`` for the cost banner."""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    return f"{seconds:.0f}s"


# Per-task wallclock estimates from our smoke tests against the live
# service. Used only for the cost-confirmation banner -- actual times
# fluctuate with TG-side latency and pool saturation.
NICK_TASK_ETA_S = 30
PHOTO_TASK_ETA_S = 135


def _step_5_resolve_contacts_via_sherlock(
    db: "LeadDB",
    cfg: dict,
    *,
    batch_limit: int,
    workers_override: int | None,
    sequential: bool,
    request_gap_secs: float,
    auto_yes: bool,
    log,
    issues: list[tuple[str, str]],
    tg_notifier: PipelineTelegramNotifier,
    deepseek: OpenAI | None,
) -> None:
    """Run Sherlock contact resolution for naked leads (parallel or sequential).

    Pulls up to ``batch_limit`` candidates via
    :py:meth:`LeadDB.get_leads_for_sherlock` (mirrors Step 4's
    ``get_leads_without_profile`` pattern -- one pipeline run eats
    a bounded chunk of the Sherlock backlog, leftovers are picked
    up next run since ``sherlock_processed_at`` is set on every
    terminal outcome).

    By default (``pipeline.step5.sequential``) each lead is fully
    resolved before the next starts, with ``request_gap_secs`` between
    leads (after the first). With ``sequential: false``, leads are fanned
    out across a thread pool sized to either ``workers_override`` (CLI)
    or ``/v1/health.pool.idle`` (fallback 3 if unreachable).

    DB writes always happen on the main thread so SQLite stays
    single-writer.

    ``batch_limit`` is sourced by ``main()`` from
    ``config.yaml`` (``pipeline.step5.batch_limit``), defaulting
    to :data:`DEFAULT_SHERLOCK_BATCH_LIMIT`.

    The function is self-contained: builds its own SherlockClient
    from ``cfg`` and tears it down before returning. Never raises;
    user-visible failures land in ``issues`` so the final summary
    banner highlights them.

    ``tg_notifier``: after each ``mark_lead_sherlock``, sends Telegram
    per lead from the main thread (photo + InsightFace caption, then text
    when photo stage ran — see :meth:`PipelineTelegramNotifier.notify_sherlock_lead`).

    ``deepseek``: OpenAI client pointed at DeepSeek; used to pick among
    photo-search rows when the first row is not an exact match. Pass
    ``None`` only in tests — production ``main()`` always supplies the
    same client as Step 2.
    """
    _banner("STEP 5: Resolve contacts via Sherlock")

    sh_cfg = cfg.get("sherlock") or {}
    nick_cfg = sh_cfg.get("nick_search") or {}
    photo_cfg = sh_cfg.get("photo_search") or {}
    task_cfg = sh_cfg.get("task") or {}
    conc_cfg = sh_cfg.get("concurrency") or {}

    try:
        sherlock = make_sherlock_client(cfg)
    except EnvironmentError as exc:
        # Most common cause: SHERLOCK_API_KEY missing in .env.
        # Don't crash the whole pipeline -- skip the step loudly.
        print(f"  SKIPPED: cannot build Sherlock client ({exc}).")
        log.warning("step5_skip_no_client", error=str(exc))
        issues.append(("Step 5", f"Sherlock client missing: {exc}"))
        return

    try:
        gap_s = max(0.0, float(request_gap_secs))
        # Workers: sequential forces 1; else CLI override beats config
        # beats live pool probe.
        if sequential:
            workers = 1
            workers_source = "sequential (pipeline.step5.sequential)"
        elif workers_override is not None:
            workers = max(1, int(workers_override))
            workers_source = "--workers"
        elif conc_cfg.get("workers"):
            workers = max(1, int(conc_cfg["workers"]))
            workers_source = "config.yaml sherlock.concurrency.workers"
        else:
            workers = sherlock.get_pool_idle(fallback=3)
            workers_source = "/v1/health pool.idle"

        candidates = db.get_leads_for_sherlock(limit=batch_limit)
        with_face = sum(1 for c in candidates if c.get("face_photo_path"))

        # Best- and worst-case ETAs:
        #   best  = every nick hits (no photo runs)
        #   worst = every nick misses, every face_photo'd lead burns a
        #           full photo task
        n = len(candidates)
        if sequential:
            best_eta = n * NICK_TASK_ETA_S
            worst_eta = n * NICK_TASK_ETA_S + with_face * PHOTO_TASK_ETA_S
            if gap_s > 0 and n > 1:
                best_eta += gap_s * (n - 1)
                worst_eta += gap_s * (n - 1)
        else:
            best_eta = (n * NICK_TASK_ETA_S) / max(workers, 1)
            worst_eta = (
                n * NICK_TASK_ETA_S + with_face * PHOTO_TASK_ETA_S
            ) / max(workers, 1)

        if not sequential and gap_s > 0:
            print(
                "  NOTE: request_gap_secs is ignored unless sequential mode "
                "(pipeline.step5.sequential)."
            )
            log.warning(
                "step5_gap_ignored_not_sequential",
                request_gap_secs=gap_s,
            )

        print(f"  Candidates:        {n}")
        print(f"  With face photo:   {with_face}  (eligible for photo fallback)")
        print(f"  Workers:           {workers}  (from {workers_source})")
        if sequential and gap_s > 0:
            print(
                f"  Gap between leads: {gap_s}s  (pipeline.step5.request_gap_secs)"
            )
        print(f"  Best-case ETA:     {_format_eta(best_eta)}  "
              f"(every nick search hits)")
        print(f"  Worst-case ETA:    {_format_eta(worst_eta)}  "
              f"(every photo fallback runs)")

        if n == 0:
            print("  SKIPPED: no candidates.")
            log.info("step5_no_candidates")
            return

        if not auto_yes:
            confirm = input("  Proceed? (y/n): ").strip().lower()
            if confirm != "y":
                print("  SKIPPED by user.")
                log.info("step5_skipped_by_user")
                return

        log.info(
            "step5_start",
            candidates=n,
            with_face=with_face,
            workers=workers,
            workers_source=workers_source,
            sequential=sequential,
            request_gap_secs=gap_s if sequential else 0.0,
        )

        counters: dict[str, int] = {
            SH_STATUS_FOUND_NICK: 0,
            SH_STATUS_FOUND_PHOTO: 0,
            SH_STATUS_NO_MATCH: 0,
            SH_STATUS_NO_FACE_PHOTO: 0,
            SH_STATUS_ERROR: 0,
        }
        step5_deepseek_calls = 0
        step5_deepseek_api_ok = 0

        def _worker_error_payload(username: str, exc: Exception) -> dict:
            return {
                "username": username,
                "status": SH_STATUS_ERROR,
                "telegram_username": None,
                "phone": None,
                "sherlock_link": None,
                "error": f"worker crashed: {type(exc).__name__}: {exc}",
                "nick_skipped_dot": "." in username,
                "nick_search_ran": False,
                "nick_hit": False,
                "photo_search_ran": False,
                "photo_task": None,
                "nick_query": None if "." in username else f"@{username}",
            }

        def _apply_step5_result(lead: dict, res: dict, progress_i: int) -> None:
            nonlocal step5_deepseek_calls, step5_deepseek_api_ok
            username = lead["username"]
            db.mark_lead_sherlock(
                username=username,
                status=res["status"],
                telegram_username=res.get("telegram_username"),
                phone=res.get("phone"),
                sherlock_link=res.get("sherlock_link"),
            )
            tg_notifier.notify_sherlock_lead(lead, res, cfg=cfg)
            if res.get("step5_deepseek_called"):
                step5_deepseek_calls += 1
                if not res.get("step5_deepseek_api_failed"):
                    step5_deepseek_api_ok += 1
            counters[res["status"]] = counters.get(res["status"], 0) + 1
            tag = res["status"]
            detail_bits: list[str] = []
            if res.get("telegram_username"):
                detail_bits.append(f"tg=@{res['telegram_username']}")
            if res.get("phone"):
                detail_bits.append(f"phone={res['phone']}")
            if res.get("sherlock_link"):
                detail_bits.append(f"link={res['sherlock_link']}")
            if res.get("error") and res["status"] == SH_STATUS_ERROR:
                detail_bits.append(f"err={res['error'][:80]}")
            detail = "  " + " ".join(detail_bits) if detail_bits else ""
            print(f"  [{progress_i:>4}/{n}] @{username:<25} -> {tag}{detail}")
            log.info(
                "step5_lead_done",
                username=username,
                status=tag,
                telegram_username=res.get("telegram_username"),
                phone=bool(res.get("phone")),
                error=res.get("error"),
            )

        if sequential:
            for idx, lead in enumerate(candidates, 1):
                if idx > 1 and gap_s > 0:
                    time.sleep(gap_s)
                username = lead["username"]
                try:
                    res = _resolve_one_lead_via_sherlock(
                        sherlock,
                        lead,
                        nick_cfg=nick_cfg,
                        photo_cfg=photo_cfg,
                        task_cfg=task_cfg,
                        deepseek=deepseek,
                    )
                except Exception as exc:  # noqa: BLE001
                    res = _worker_error_payload(username, exc)
                _apply_step5_result(lead, res, idx)
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="sherlock"
            ) as pool:
                futures = {
                    pool.submit(
                        _resolve_one_lead_via_sherlock,
                        sherlock,
                        lead,
                        nick_cfg=nick_cfg,
                        photo_cfg=photo_cfg,
                        task_cfg=task_cfg,
                        deepseek=deepseek,
                    ): lead
                    for lead in candidates
                }
                for i, fut in enumerate(as_completed(futures), 1):
                    lead = futures[fut]
                    username = lead["username"]
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        res = _worker_error_payload(username, exc)
                    _apply_step5_result(lead, res, i)

        # Final breakdown.
        print()
        print(f"  DONE: {n} processed")
        for label in (
            SH_STATUS_FOUND_NICK,
            SH_STATUS_FOUND_PHOTO,
            SH_STATUS_NO_MATCH,
            SH_STATUS_NO_FACE_PHOTO,
            SH_STATUS_ERROR,
        ):
            print(f"    {label:<18} {counters.get(label, 0)}")

        log.info(
            "step5_done",
            step5_deepseek_calls=step5_deepseek_calls,
            step5_deepseek_api_ok=step5_deepseek_api_ok,
            **{f"count_{k}": v for k, v in counters.items()},
        )

        error_count = counters.get(SH_STATUS_ERROR, 0)
        tg_notifier.maybe_notify_sherlock_batch_all_failed(
            leads_processed=n,
            error_count=error_count,
        )
        tg_notifier.maybe_notify_deepseek_batch_all_failed(
            deepseek_calls=step5_deepseek_calls,
            deepseek_succeeded=step5_deepseek_api_ok,
            step="Step 5",
            call_kind="usermatch call(s)",
            outcome_label="usermatch picks",
        )

        if error_count:
            issues.append((
                "Step 5",
                f"{error_count} leads finished as error -- "
                "check logs for Sherlock task failures / timeouts",
            ))
    finally:
        sherlock.close()


# =========================================================================
# Step 6: cleanup spent face assets
# =========================================================================
#
# Once Sherlock has produced a terminal outcome for a lead, its avatar
# and face_photo are no longer needed by the pipeline -- the contact
# (or "no_match") is recorded in lead_accounts. Step 6 walks
# get_leads_with_spent_photos() (Sherlock done, status != 'error',
# at least one path still set) and unlinks the files, then NULLs the
# columns. error-status leads keep their photos so a manual retry
# (clear sherlock_processed_at) doesn't have to re-pay Apify for
# Step 4.
#
# The four face-detection queries used by backfill / dev scripts
# (get_leads_needing_avatar &c) gate on `sherlock_processed_at IS NULL`,
# so cleaned leads aren't re-fetched via Apify after Step 6 wipes
# their avatar_path. See db.mark_lead_photos_cleaned for the
# implicit "cleaned" predicate.
#
# Runs even with --skip-sherlock so the operator can drain the
# accumulated backlog of already-Sherlock'd leads from prior runs.
# Suppress with --keep-photos for forensic / debugging sessions.


def _step_6_cleanup_spent_face_assets(
    db: "LeadDB",
    *,
    log,
    issues: list[tuple[str, str]],
) -> None:
    """Delete avatars / face photos for leads Sherlock has finished with.

    Idempotent: once both columns are NULL the lead is excluded from
    :py:meth:`LeadDB.get_leads_with_spent_photos`, so subsequent runs
    only touch new spent leads. Never raises -- per-lead unlink
    failures land in ``issues`` for the summary banner.
    """
    _banner("STEP 6: Cleanup spent face assets")

    # No limit pagination here on purpose: cleanup is local and cheap
    # (one unlink per file, no network), and the natural cap is the
    # backlog size on first run after this feature ships. The DB
    # method's default `limit=10000` is just a paranoia ceiling.
    candidates = db.get_leads_with_spent_photos()
    if not candidates:
        print("  SKIPPED: no spent face assets to clean.")
        log.info("step6_no_candidates")
        return

    print(f"  Leads to clean:    {len(candidates)}")
    log.info("step6_cleanup_spent_assets", count=len(candidates))

    files_deleted = 0
    files_failed = 0
    leads_cleaned = 0

    for lead in candidates:
        username = lead["username"]
        deleted, failed = cleanup_lead_face_assets(
            lead.get("avatar_path"),
            lead.get("face_photo_path"),
            user_id=lead.get("user_id"),
        )
        # Always mark the lead cleaned in DB even if every file was
        # already missing on disk -- the goal is to converge the DB
        # state to "no asset paths set" so future runs skip this lead.
        db.mark_lead_photos_cleaned(username)
        files_deleted += deleted
        files_failed += failed
        leads_cleaned += 1

    print(
        f"  DONE: cleaned {leads_cleaned} leads, "
        f"{files_deleted} files removed, "
        f"{files_failed} failed"
    )
    log.info(
        "step6_done",
        leads_cleaned=leads_cleaned,
        files_deleted=files_deleted,
        files_failed=files_failed,
    )

    if files_failed:
        issues.append((
            "Step 6",
            f"{files_failed} files failed to unlink during cleanup -- "
            "check warnings in logs (avatar_downloader). Lead rows "
            "were still marked cleaned to avoid re-trying.",
        ))


def _parse_cli_args() -> argparse.Namespace:
    """Pipeline-level CLI flags. Kept minimal -- the daily run uses
    no flags; flags exist for ad-hoc Step 5 / Step 6 runs."""
    parser = argparse.ArgumentParser(
        description="Daily lead collection pipeline (Apify + DeepSeek + Sherlock)."
    )
    parser.add_argument(
        "--skip-sherlock",
        action="store_true",
        help="Skip Step 5 (Sherlock contact resolution). "
             "Useful when only Steps 1-4 are needed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override Step 5 worker count when pipeline.step5.sequential "
             "is false. Default: probe /v1/health and use pool.idle "
             "(fallback 3).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Auto-confirm Step 5's cost prompt (skip the y/n input). "
             "Use only when running unattended. Same effect as "
             "pipeline.prompt_terminal_confirmation: false for Step 5 only "
             "(--yes does not skip Step 3 or the post-issues Enter pause).",
    )
    parser.add_argument(
        "--keep-photos",
        action="store_true",
        help="Skip Step 6 (cleanup of spent face assets). Use for "
             "debugging / forensic sessions where you need avatars "
             "and face photos to stay on disk after Sherlock.",
    )
    return parser.parse_args()


def main():
    args = _parse_cli_args()
    load_dotenv()
    cfg = load_config()
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
    discovery_mode = str(
        s1_cfg.get("discovery_mode", DEFAULT_STEP1_DISCOVERY_MODE)
    ).strip().lower()
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

    if discovery_mode not in ("realtors", "hashtags", "cookie_keywords"):
        log.error("step1_invalid_discovery_mode", mode=discovery_mode)
        print(
            "FAILED: pipeline.step1.discovery_mode must be 'realtors', "
            f"'hashtags', or 'cookie_keywords', got {discovery_mode!r}."
        )
        issues.append(("Step 1", f"invalid discovery_mode: {discovery_mode}"))
        return

    step1_cost_usd = 0.0
    posts_age_stats: dict[str, int] | None = None
    reels_age_stats: dict[str, int] | None = None
    step1_empty_issue = "post-scraper returned 0 items"
    notify_secondary_count = 0
    step1_age_dropped_client: int | None = None
    step1_age_kept_missing_ts: int | None = None

    if discovery_mode == "realtors":
        _banner(f"STEP 1: Fetch posts (last {posts_max_age_days} days) [realtors]")
        realtors = _realtor_usernames_from_cfg(cfg)
        if not realtors:
            log.error("step1_no_realtor_accounts")
            print(
                "FAILED: search.realtor_accounts is empty in config for "
                "discovery_mode=realtors."
            )
            issues.append(("Step 1", "search.realtor_accounts empty"))
            return

        notify_secondary_count = len(realtors)
        print(f"  Realtors:       {len(realtors)}")
        log.info(
            "step1_fetch_posts",
            discovery_mode=discovery_mode,
            realtors=len(realtors),
            max_age_days=posts_max_age_days,
            post_scraper_results_limit=post_scraper_results_limit,
        )

        run = apify.actor("apify/instagram-post-scraper").call(run_input={
            "username": realtors,
            "resultsLimit": post_scraper_results_limit,
            "onlyPostsNewerThan": f"{posts_max_age_days} days",
            "dataDetailLevel": "basicData",
            "proxy": {"useApifyProxy": True},
        })
        tg_notifier.maybe_notify_apify_run_failure(
            run,
            actor_id="apify/instagram-post-scraper",
            step="Step 1",
        )
        detail = apify.run(run["id"]).get()
        all_posts = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
        step1_cost_usd = float(detail.get("usageTotalUsd") or 0)

        pipeline.log_run(
            actor_id="apify/instagram-post-scraper",
            run_id=run["id"],
            status=run["status"],
            input_params={
                "realtors": len(realtors),
                "resultsLimit": post_scraper_results_limit,
            },
            items_count=len(all_posts),
            cost_usd=detail.get("usageTotalUsd", 0),
            duration_ms=detail.get("stats", {}).get("durationMillis"),
        )
    elif discovery_mode == "hashtags":
        hashtags = list((cfg.get("search") or {}).get("hashtags") or [])
        _banner(
            f"STEP 1: Fetch posts via hashtags "
            f"(≤{posts_max_age_days}d by timestamp) [hashtags]"
        )
        if not hashtags:
            log.error("step1_no_hashtags")
            print(
                "FAILED: search.hashtags is empty in config for discovery_mode=hashtags."
            )
            issues.append(("Step 1", "search.hashtags empty"))
            return

        notify_secondary_count = len(hashtags)
        print(f"  Hashtags:       {len(hashtags)}")
        log.info(
            "step1_fetch_posts",
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

        run_p = apify.actor(hashtag_actor_id).call(
            run_input={**run_base, "resultsType": "posts"}
        )
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

        run_r = apify.actor(hashtag_actor_id).call(
            run_input={**run_base, "resultsType": "reels"}
        )
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

        all_posts = merge_hashtag_items_by_shortcode(posts_filtered, reels_filtered)
        step1_age_dropped_client = int(
            (posts_age_stats or {}).get("dropped_too_old") or 0
        ) + int((reels_age_stats or {}).get("dropped_too_old") or 0)
        step1_age_kept_missing_ts = int(
            (posts_age_stats or {}).get("kept_missing_timestamp") or 0
        ) + int((reels_age_stats or {}).get("kept_missing_timestamp") or 0)
        step1_empty_issue = "hashtag-scraper returned 0 items after merge/filter"
        log.info(
            "step1_hashtag_merge",
            posts_raw=len(posts_fetched),
            reels_raw=len(reels_fetched),
            merged=len(all_posts),
            posts_age_stats=posts_age_stats,
            reels_age_stats=reels_age_stats,
        )

    elif discovery_mode == "cookie_keywords":
        search_cfg = cfg.get("search") or {}
        cs_cfg = step1_cookie_search_section(cfg)
        keywords = [str(k).strip() for k in (search_cfg.get("cookie_search_keywords") or []) if str(k).strip()]
        _banner(
            f"STEP 1: Fetch posts via cookie keyword search "
            f"(≤{posts_max_age_days}d by timestamp) [cookie_keywords]"
        )
        if not keywords:
            log.error("step1_no_cookie_search_keywords")
            print(
                "FAILED: search.cookie_search_keywords is empty in config "
                "for discovery_mode=cookie_keywords."
            )
            issues.append(("Step 1", "search.cookie_search_keywords empty"))
            return

        cookie_var = str(cs_cfg.get("session_cookie_env_var", "INSTAGRAM_SESSION_COOKIE"))
        cookies_raw = (os.environ.get(cookie_var) or "").strip()
        if not cookies_raw:
            log.error("step1_missing_instagram_session_cookie", env_var=cookie_var)
            print(
                f"FAILED: {cookie_var} is empty or unset. "
                "Paste Instagram cookies into .env (see .env.example)."
            )
            issues.append(("Step 1", f"missing env {cookie_var} for cookie keyword search"))
            return

        try:
            cookies_payload = cookies_json_string_for_actor(cookies_raw)
        except (json.JSONDecodeError, ValueError) as e:
            log.error("step1_cookie_parse_failed", error=str(e))
            print(f"FAILED: could not normalize cookies for the actor: {e}")
            issues.append(("Step 1", f"cookie parse error: {e}"))
            return

        max_posts = int(cs_cfg.get("size_per_keyword", 5))
        session_name = str(cs_cfg.get("session_name", "instalead_cookie_search"))

        notify_secondary_count = len(keywords)
        print(f"  Keywords:       {len(keywords)}")
        log.info(
            "step1_fetch_posts",
            discovery_mode=discovery_mode,
            keywords=len(keywords),
            max_posts_per_keyword=max_posts,
            max_age_days=posts_max_age_days,
            cookie_env_var=cookie_var,
        )

        run_kw = apify.actor(cookie_search_actor_id).call(
            run_input={
                "keywords": keywords,
                "maxPosts": max_posts,
                "cookies": cookies_payload,
                "sessionName": session_name,
            }
        )
        tg_notifier.maybe_notify_apify_run_failure(
            run_kw, actor_id=cookie_search_actor_id, step="Step 1"
        )
        detail_kw = apify.run(run_kw["id"]).get()
        raw_items = list(apify.dataset(run_kw["defaultDatasetId"]).iterate_items())
        step1_cost_usd = float(detail_kw.get("usageTotalUsd") or 0)

        normalized: list[dict] = []
        for row in raw_items:
            if not isinstance(row, dict):
                continue
            n = normalize_keyword_search_item(row)
            if n is not None:
                normalized.append(n)

        deduped = dedupe_keyword_items_by_shortcode(normalized)
        all_posts, posts_age_stats = filter_items_within_max_age(
            deduped, posts_max_age_days
        )
        step1_age_dropped_client = int((posts_age_stats or {}).get("dropped_too_old") or 0)
        step1_age_kept_missing_ts = int(
            (posts_age_stats or {}).get("kept_missing_timestamp") or 0
        )
        reels_age_stats = None
        step1_empty_issue = (
            "cookie keyword search returned 0 posts after normalize/dedupe/age-filter"
        )

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
            raw_dataset_rows=len(raw_items),
            normalized=len(normalized),
            deduped=len(deduped),
            after_age_filter=len(all_posts),
            posts_age_stats=posts_age_stats,
        )

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
            )
            new_posts += 1
            step1_new_post_items.append(p)

    log.info(
        "step1_done",
        discovery_mode=discovery_mode,
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
        discovery_mode=discovery_mode,
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
        discovery_mode=discovery_mode,
        full_message=build_step1_pipeline_summary_telegram_text(
            new_posts=new_posts,
            source_count=notify_secondary_count,
            discovery_mode=discovery_mode,
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
            "SELECT post_id, caption, post_url FROM processed_posts "
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
        raw_score = score_caption(deepseek, combined)
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
                }
            )

    is_re_ctr = Counter(v for _, v in step2_is_re_audit)
    human_stats = {"approved": 0, "denied": 0, "timeout": 0}
    creds = tg_notifier.inline_confirm_token_and_chat()
    if human_confirm_queue and creds:
        token, chat_id = creds
        human_stats = asyncio.run(
            _run_step2_human_confirmations(
                db, human_confirm_queue, token, chat_id
            )
        )
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
                louisdeconinck_cap_per_post=louisdeconinck_cap,
                tg_notifier=tg_notifier,
            )

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
                            comment_pk=str(c.get("pk") or ""),
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

            run = apify.actor("apify/instagram-profile-scraper").call(run_input={
                "usernames": batch,
            })
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
    print(
        "Realtor accounts (config): "
        f"{len(_realtor_usernames_from_cfg(cfg))}"
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
