"""Optional Telegram reports after pipeline Step 1–5.

Uses Bot API ``sendMessage`` via aiogram. Disabled when ``TELEGRAM_BOT_TOKEN``
is missing or ``telegram.report_chat_id`` is absent/invalid — the pipeline
keeps running."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from aiogram import Bot
from aiogram.exceptions import TelegramNetworkError, TelegramServerError
from aiogram.types import FSInputFile

from src.ig_media_payload import parse_item_timestamp_utc
from src.logger import get_logger

log = get_logger("telegram_notifier")

# Pause between Step 1 per-post Telegram messages (rate limits / UX).
_STEP1_NEW_POST_MESSAGE_DELAY_SEC = 0.5

TOKEN_ENV_VAR = "TELEGRAM_BOT_TOKEN"

# One initial attempt + this many retries; 20s pause after each failure.
_TELEGRAM_SEND_RETRIES = 3
_TELEGRAM_RETRY_DELAY_SEC = 40.0
_TELEGRAM_REQUEST_TIMEOUT_SEC = 40.0


def _is_retryable_telegram_send_error(exc: BaseException) -> bool:
    """Transient transport / Telegram infra errors worth re-trying."""
    if isinstance(exc, (TelegramNetworkError, TelegramServerError)):
        return True
    if isinstance(exc, (TimeoutError, OSError, ConnectionError)):
        return True
    return False

# Must match scripts/pipeline.py SH_STATUS_* labels (avoid importing pipeline).
_SH_FOUND_NICK = "found_nick"
_SH_FOUND_PHOTO = "found_photo"
_SH_NO_MATCH = "no_match"
_SH_NO_FACE_PHOTO = "no_face_photo"
_SH_ERROR = "error"
_TELEGRAM_MESSAGE_HARD_LIMIT = 4096
_TELEGRAM_PHOTO_CAPTION_LIMIT = 1024


def _telegram_handle(handle: str) -> str:
    h = (handle or "").strip()
    if not h:
        return ""
    return h if h.startswith("@") else f"@{h}"


def truncate_for_telegram(text: str, limit: int = _TELEGRAM_MESSAGE_HARD_LIMIT) -> str:
    if len(text) <= limit:
        return text
    suffix = "\n...(truncated for Telegram limit)"
    return text[: max(0, limit - len(suffix))] + suffix


def build_step1_date_filter_section_lines(
    *,
    discovery_mode: str,
    posts_max_age_days: int,
    age_dropped_client: int | None,
    age_kept_missing_ts: int | None,
) -> list[str]:
    """Human-readable date / age filter lines for Step 1 (Telegram + terminal)."""
    mode = (discovery_mode or "realtors").strip().lower()
    out: list[str] = [
        "Date filter",
        f"Config: pipeline.step1.posts_max_age_days = {posts_max_age_days}",
    ]
    if mode == "realtors":
        out.append(
            "Applied on Apify as onlyPostsNewerThan "
            f"(posts newer than {posts_max_age_days} day(s)); "
            "client-side timestamp drop count is not available for this mode."
        )
        return out
    if posts_max_age_days <= 0:
        out.append("Client-side max-age filter: disabled (no items dropped for age).")
        if age_dropped_client is not None:
            out.append(f"Dropped — older than window: {age_dropped_client}")
        return out
    out.append(
        "Client-side: items with parseable timestamp older than "
        f"{posts_max_age_days} day(s) (UTC) are dropped after the Apify run."
    )
    if age_dropped_client is not None:
        out.append(f"Dropped — older than window: {age_dropped_client}")
    if age_kept_missing_ts is not None and age_kept_missing_ts > 0:
        out.append(
            "Kept — missing or unparseable timestamp "
            f"(not evaluated against age window): {age_kept_missing_ts}"
        )
    return out


def build_step1_pipeline_summary_telegram_text(
    *,
    new_posts: int,
    source_count: int,
    discovery_mode: str,
    min_comments: int,
    fetched_total: int,
    updated_posts: int,
    with_video_count: int,
    skipped_no_video_url: int,
    step1_skip_low_comments: int,
    step1_skip_no_shortcode: int,
    step1_existing_unchanged: int,
    cost_usd: float,
    posts_max_age_days: int,
    age_dropped_client: int | None,
    age_kept_missing_ts: int | None,
) -> str:
    """Single Step 1 Telegram message: headline + run totals + gate breakdown (``\\n``-separated)."""
    mode = (discovery_mode or "realtors").strip().lower()
    if mode == "hashtags":
        searched = f"Searched {source_count} hashtag(s)."
    elif mode == "cookie_keywords":
        searched = f"Searched {source_count} keyword(s)."
    else:
        searched = f"Searched {source_count} active realtor(s)."

    lines: list[str] = [
        "Step 1",
        "",
        "Summary",
        f"New post(s) saved: {new_posts}",
        searched,
        "",
        "Run totals",
        f"Posts fetched from Apify (this run): {fetched_total}",
        f"Reels with usable video URL: {with_video_count}",
        f"Step 1 Apify cost (USD): {cost_usd:.4f}",
        "",
        *build_step1_date_filter_section_lines(
            discovery_mode=discovery_mode,
            posts_max_age_days=posts_max_age_days,
            age_dropped_client=age_dropped_client,
            age_kept_missing_ts=age_kept_missing_ts,
        ),
        "",
        "Gate breakdown",
        f"Skipped — comments below min_comments_per_post ({min_comments}): "
        f"{step1_skip_low_comments}",
        f"Skipped — empty shortCode: {step1_skip_no_shortcode}",
        f"Skipped — reel without valid videoUrl: {skipped_no_video_url}",
        f"Already in DB — unchanged: {step1_existing_unchanged}",
        f"Already in DB — comments_count updated: {updated_posts}",
    ]
    return "\n".join(lines)


def step1_display_content_type(item: dict[str, Any]) -> str:
    """Map Apify ``type`` / ``productType`` to Image | Video | Sidecar."""
    raw = (item.get("type") or "").strip()
    if raw in ("Image", "Video", "Sidecar"):
        return raw
    if raw:
        return raw
    return "Unknown"


def build_step1_new_post_message(item: dict[str, Any]) -> str:
    """One Telegram text per newly upserted Step 1 post (Apify-shaped dict)."""
    shortcode = (item.get("shortCode") or "").strip()
    url = (item.get("url") or "").strip()
    if not url and shortcode:
        url = f"https://www.instagram.com/p/{shortcode}/"
    if not url:
        url = "(unknown)"

    ts_raw = item.get("timestamp")
    ts_parsed = parse_item_timestamp_utc(ts_raw)
    if ts_parsed is not None:
        ts_line = ts_parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        ts_line = str(ts_raw).strip() if ts_raw is not None else "(unknown)"

    owner = (item.get("ownerUsername") or "").strip() or "(unknown)"

    raw_tags = item.get("hashtags")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    elif isinstance(raw_tags, str) and raw_tags.strip():
        tags = [raw_tags.strip()]

    hashtag_lines: list[str]
    if tags:
        hashtag_lines = []
        for t in tags:
            hashtag_lines.append(t if t.startswith("#") else f"#{t}")
    else:
        hashtag_lines = ["(none)"]

    comments_n = item.get("commentsCount")
    likes_n = item.get("likesCount")
    ctype = step1_display_content_type(item)

    loc_name = item.get("locationName")
    loc_id = item.get("locationId")
    loc_name_s = (
        str(loc_name).strip()
        if loc_name is not None and str(loc_name).strip()
        else ""
    )
    loc_id_s = (
        str(loc_id).strip()
        if loc_id is not None and str(loc_id).strip()
        else ""
    )

    lines: list[str] = [
        "Step 1 · new post",
        "────────",
        "",
        "url",
        url,
        "",
        "timestamp",
        ts_line,
        "",
        "ownerUsername",
        owner,
        "",
        "hashtags",
        *hashtag_lines,
        "",
        "commentsCount",
        str(comments_n if comments_n is not None else "—"),
        "",
        "likesCount",
        str(likes_n if likes_n is not None else "—"),
        "",
        "type",
        ctype,
    ]

    if loc_name_s or loc_id_s:
        lines.extend(
            [
                "",
                "locationName",
                loc_name_s if loc_name_s else "—",
                "",
                "locationId",
                loc_id_s if loc_id_s else "—",
            ]
        )

    sk = item.get("searchKeyword")
    if isinstance(sk, str) and sk.strip():
        lines.extend(["", "searchKeyword", sk.strip()])

    csk = item.get("cookieSearchKeywords")
    if isinstance(csk, list) and len(csk) > 1:
        lines.extend(["", "cookieSearchKeywords", ", ".join(str(x) for x in csk if str(x).strip())])

    cmt = item.get("cookieMediaType")
    if isinstance(cmt, str) and cmt.strip():
        lines.extend(["", "cookieMediaType", cmt.strip()])

    prev = item.get("cookieMediaUrlsPreview")
    if isinstance(prev, list) and prev:
        lines.extend(["", "cookieMediaUrlsPreview"])
        for i, u in enumerate(prev[:5]):
            uu = str(u).strip()
            if len(uu) > 120:
                uu = uu[:119] + "…"
            lines.append(f"  [{i}] {uu}")
        if len(prev) > 5:
            lines.append(f"  … +{len(prev) - 5} more (truncated in item)")

    cap_prev = item.get("captionPreview")
    if isinstance(cap_prev, str) and cap_prev.strip():
        lines.extend(["", "captionPreview", cap_prev.strip()])

    mentions = item.get("cookieMentions")
    if isinstance(mentions, list) and mentions:
        lines.extend(["", "mentions", ", ".join(mentions[:40])])
        if len(mentions) > 40:
            lines.append(f"… +{len(mentions) - 40} more")

    vu = item.get("videoUrl")
    if isinstance(vu, str) and vu.strip():
        vv = vu.strip()
        if len(vv) > 100:
            vv = vv[:99] + "…"
        lines.extend(["", "videoUrl", vv])

    return "\n".join(lines)


def build_step2_scored_post_message(
    post_url: str,
    raw_score: dict[str, Any],
    resolved_relevance: str,
    combined_text: str,
) -> str:
    """One Telegram message per post scored in Step 2 (caption and/or transcript)."""
    link = (post_url or "").strip() or "(unknown)"
    lines: list[str] = [
        "Step 2: scored post",
        "",
        f"Link: {link}",
        "",
        "Assessment (DeepSeek relevance prompt):",
    ]
    if not raw_score:
        lines.append("(empty score)")
    elif raw_score.get("error"):
        lines.append(f"error: {raw_score.get('error')}")
    else:
        lines.append(f"is_real_estate: {raw_score.get('is_real_estate')}")
        lines.append(f"has_call_to_action: {raw_score.get('has_call_to_action')}")
        lines.append(f"call_to_action_type: {raw_score.get('call_to_action_type')}")
    lines.extend(
        [
            "",
            f"Pipeline relevance: {resolved_relevance}",
            "",
            "Scoring text (caption; transcript when available):",
            (combined_text or "").strip() or "(empty)",
        ]
    )
    return "\n".join(lines)


def build_sherlock_face_photo_caption(
    lead: dict, insightface_detection_percent: float | None
) -> str:
    """Caption under the Sherlock face image (Instagram handle + SCRFD score)."""
    ig = str(lead.get("username") or "").strip() or "(unknown)"
    handle = f"@{ig}" if ig != "(unknown)" else ig
    if insightface_detection_percent is None:
        score = "n/a"
    else:
        score = f"{insightface_detection_percent:.1f}%"
    return f"{handle}\nInsightFace (SCRFD det. confidence): {score}"


def compute_insightface_best_det_percent(
    face_photo_path: str | Path, cfg: dict[str, Any]
) -> float | None:
    """Best detection score among faces above ``min_det_score`` (0–100).

    Re-runs InsightFace immediately before Telegram send — temporary QA hook.
    Uses ``avatar`` calibration first (most ``face_photo_path`` files are
    avatars), then ``post`` fallback for face-leader crops.
    """
    from src.face_embedder import make_face_embedder

    path = Path(face_photo_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        for kind in ("avatar", "post"):
            embedder = make_face_embedder(cfg, kind=kind)
            faces = embedder.embed_faces(path)
            if faces:
                return max(f.det_score for f in faces) * 100.0
    except Exception:
        log.warning(
            "telegram_insightface_det_percent_failed",
            path=str(path),
            exc_info=True,
        )
    return None


def build_sherlock_lead_notification_text(lead: dict, res: dict) -> str:
    """One message per Sherlock lead — used by Step 5 and tests."""
    res = res or {}
    ig = str(lead.get("username") or "").strip() or "(unknown)"

    nick_skipped = bool(res.get("nick_skipped_dot"))
    nick_hit = bool(res.get("nick_hit"))
    nick_search_ran = bool(res.get("nick_search_ran"))
    nick_query = res.get("nick_query")

    photo_ran = bool(res.get("photo_search_ran"))
    photo_task = res.get("photo_task")
    if not isinstance(photo_task, dict):
        photo_task = {}

    lines: list[str] = []
    lines.append(f"Step 5 (Sherlock): Insta username - {ig}\n")
    post_u = lead.get("context_post_url")
    lines.append(f"Post: {post_u}" if post_u else "Post: (unknown)\n")
    if ig != "(unknown)":
        lines.append(f"Profile: https://www.instagram.com/{ig}/ \n")
    else:
        lines.append("Profile: (unknown)")

    sc = str(lead.get("context_post_shortcode") or "").strip()
    cpk = str(lead.get("context_comment_pk") or "").strip()
    if sc and cpk:
        lines.append(f"Comment: https://www.instagram.com/p/{sc}/c/{cpk}/")
    else:
        lines.append(
            "Comment link: not stored (post URL above is the minimum context)\n"
        )

    if nick_hit:
        tg_raw = str(res.get("telegram_username") or "").strip()
        lines.append(f"Telegram match (nick search): {_telegram_handle(tg_raw)}")
    elif photo_ran:
        if nick_skipped:
            lines.append(
                f"Telegram nick not searched: Instagram username `{ig}` "
                "contains '.' (Sherlock skips nick stage).\n"
            )
        elif nick_search_ran and nick_query:
            lines.append(
                f"Telegram nick not found for {nick_query}; photo search.\n"
            )
        lines.append("")
        lines.append("Photo search — full Sherlock task JSON:\n\n")
        lines.append(json.dumps(photo_task, ensure_ascii=False, indent=2, default=str))
    else:
        st_face = str(res.get("status") or "")
        if st_face == "no_face_photo":
            lines.append(
                "No usable face photo for Sherlock photo search (skipped photo stage)."
            )
        elif nick_skipped:
            lines.append(
                "Sherlock nick search skipped (username contains '.'). "
                "Photo stage did not run or did not finish."
            )
        elif nick_search_ran and nick_query and not nick_hit:
            lines.append(
                f"Sherlock nick search queried {nick_query} but did not confirm "
                "a Telegram profile; photo stage did not run or did not finish."
            )

    return "\n".join(lines)


def _sherlock_result_summary_title_line(lead: dict, res: dict) -> str:
    """First line of the Russian Step 5 summary: ``Результат по "handle"``."""
    res = res or {}
    status = str(res.get("status") or "")
    ig = str(lead.get("username") or "").strip() or "unknown"
    if status == _SH_FOUND_NICK:
        h = _telegram_handle(str(res.get("telegram_username") or ""))
        return f'Результат по "{h}"' if h else f'Результат по "{ig}"'
    if status == _SH_FOUND_PHOTO:
        link = str(res.get("sherlock_link") or "").strip()
        m = re.search(
            r"(?:https?://)?(?:t\.me|telegram\.me)/([A-Za-z0-9_]+)", link, re.I
        )
        if m:
            return f'Результат по "{_telegram_handle(m.group(1))}"'
    return f'Результат по "{ig}"'


def _sherlock_match_label_ru(res: dict) -> str:
    """Value for the ``совпадение:`` line in the Russian summary."""
    st = str(res.get("status") or "")
    if st == _SH_FOUND_NICK:
        return "найден по нику"
    if st == _SH_FOUND_PHOTO:
        kind = str(res.get("photo_match_kind") or "")
        if kind == "exact":
            return "точное совпадение"
        if kind == "deepseek":
            return "вероятное совпадение"
        return "вероятное совпадение"
    if st in (_SH_NO_MATCH, _SH_NO_FACE_PHOTO):
        return "пользователь не найден"
    if st == _SH_ERROR:
        return "ошибка поиска"
    return "пользователь не найден"


def build_sherlock_lead_result_summary_text(lead: dict, res: dict) -> str:
    """Russian follow-up for operators: Insta context + match outcome."""
    res = res or {}
    ig = str(lead.get("username") or "").strip() or "(unknown)"
    lines: list[str] = []
    lines.append(_sherlock_result_summary_title_line(lead, res))
    lines.append("")
    lines.append(f"Юзернейм Instagram: {ig}" if ig != "(unknown)" else "Юзернейм Instagram: —")
    post_u = lead.get("context_post_url")
    lines.append(f"Пост: {post_u}" if post_u else "Пост: —")
    sc = str(lead.get("context_post_shortcode") or "").strip()
    cpk = str(lead.get("context_comment_pk") or "").strip()
    if sc and cpk:
        lines.append(f"Комментарий: https://www.instagram.com/p/{sc}/c/{cpk}/")
    else:
        lines.append("Комментарий: нет")
    lines.append("")

    status = str(res.get("status") or "")
    if status == _SH_FOUND_NICK:
        tg_raw = str(res.get("telegram_username") or "").strip()
        lines.append(f"Ник в тг: {_telegram_handle(tg_raw)}")
        fn = str(lead.get("full_name") or "").strip()
        lines.append(f"Имя пользователя из био инсты: {fn if fn else '—'}")
    elif status == _SH_FOUND_PHOTO:
        person = res.get("sherlock_person")
        lines.append(f"person: {person if person is not None else '—'}")
        ph = str(res.get("phone") or "").strip()
        lines.append(f"Телефон: {ph if ph else '—'}")
    lines.append(f"совпадение: {_sherlock_match_label_ru(res)}")
    return "\n".join(lines)


def _parse_report_chat_id(cfg: dict[str, Any]) -> int | None:
    tg = cfg.get("telegram") or {}
    raw = tg.get("report_chat_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        log.warning("telegram_invalid_report_chat_id", raw=raw)
        return None


class PipelineTelegramNotifier:
    """Fire-and-forget messages to ``telegram.report_chat_id``."""

    def __init__(
        self,
        token: str | None,
        chat_id: int | None,
        *,
        enabled: bool = True,
    ) -> None:
        self._token = (token or "").strip() or None
        self._chat_id = chat_id
        has_pair = bool(self._token and self._chat_id is not None)
        self._enabled = bool(enabled and has_pair)

        if not enabled:
            log.debug("telegram_notifier_disabled", reason="explicitly_disabled")
        elif not has_pair:
            if self._token and self._chat_id is None:
                log.warning("telegram_notifier_disabled", reason="missing_report_chat_id")
            elif not self._token and self._chat_id is not None:
                log.warning(
                    "telegram_notifier_disabled",
                    reason="missing_env",
                    env_var=TOKEN_ENV_VAR,
                )
            else:
                log.debug(
                    "telegram_notifier_disabled",
                    reason="no_token_and_no_chat_id",
                )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> PipelineTelegramNotifier:
        token = os.environ.get(TOKEN_ENV_VAR)
        chat_id = _parse_report_chat_id(cfg)
        return cls(token, chat_id)

    def notify_step1(
        self,
        new_posts: int,
        source_count: int,
        *,
        discovery_mode: str = "realtors",
        full_message: str | None = None,
    ) -> None:
        if not self._enabled:
            return
        if full_message is not None:
            text = truncate_for_telegram(full_message)
        else:
            if discovery_mode == "hashtags":
                tail = f"Searched {source_count} hashtag(s)."
            elif discovery_mode == "cookie_keywords":
                tail = f"Searched {source_count} keyword(s)."
            else:
                tail = f"Searched {source_count} active realtor(s)."
            text = truncate_for_telegram(
                "\n".join(
                    [
                        "Step 1",
                        f"New post(s) saved: {new_posts}",
                        tail,
                    ]
                )
            )
        self._send_sync(text)

    def notify_step1_new_posts(self, items: list[dict[str, Any]]) -> None:
        """Send one message per new post; ``0.5s`` pause between sends."""
        if not self._enabled or not items:
            return
        for i, raw in enumerate(items):
            text = truncate_for_telegram(build_step1_new_post_message(raw))
            self._send_sync(text)
            if i + 1 < len(items):
                time.sleep(_STEP1_NEW_POST_MESSAGE_DELAY_SEC)

    def notify_step2_scored_post(
        self,
        *,
        post_url: str,
        raw_score: dict[str, Any],
        resolved_relevance: str,
        combined_text: str,
    ) -> None:
        if not self._enabled:
            return
        text = truncate_for_telegram(
            build_step2_scored_post_message(
                post_url, raw_score, resolved_relevance, combined_text
            )
        )
        self._send_sync(text)

    def notify_step3(self, new_commenters: int) -> None:
        if not self._enabled:
            return
        text = (
            f"Step 3: added {new_commenters} new commenter(s) "
            "to the database (this run)."
        )
        self._send_sync(text)

    def notify_step4(
        self,
        *,
        profiles_queued: int,
        single_face_avatar: int,
        face_leader_resolved: int,
        without_suitable_photo: int,
        contacts_from_bio: int,
    ) -> None:
        if not self._enabled:
            return
        text = (
            "Step 4: "
            f"{profiles_queued} profile(s) queued; \n"
            f"{single_face_avatar} single-face avatar(s); \n"
            f"{face_leader_resolved} face photo(s) via face_leader; \n"
            f"{without_suitable_photo} lead(s) without suitable photo; \n"
            f"{contacts_from_bio} lead(s) with bio/contact fields from "
            "extract_contacts."
        )
        self._send_sync(text)

    def notify_sherlock_lead(
        self, lead: dict, res: dict, *, cfg: dict[str, Any] | None = None
    ) -> None:
        if not self._enabled:
            return
        text2 = truncate_for_telegram(
            build_sherlock_lead_notification_text(lead, res)
        )
        text3 = truncate_for_telegram(
            build_sherlock_lead_result_summary_text(lead, res)
        )
        if (
            bool(res.get("photo_search_ran"))
            and cfg is not None
            and lead.get("face_photo_path")
        ):
            photo_path = str(lead["face_photo_path"])
            if Path(photo_path).is_file():
                pct = compute_insightface_best_det_percent(photo_path, cfg)
                caption = truncate_for_telegram(
                    build_sherlock_face_photo_caption(lead, pct),
                    limit=_TELEGRAM_PHOTO_CAPTION_LIMIT,
                )
                if self._send_sync_photo_then_text(
                    photo_path, caption, text2, text3
                ):
                    return
        self._send_sync(text2)
        self._send_sync(text3)

    async def _send_message_once(self, text: str) -> None:
        assert self._token is not None and self._chat_id is not None
        async with Bot(token=self._token) as bot:
            await asyncio.wait_for(
                bot.send_message(chat_id=self._chat_id, text=text),
                timeout=_TELEGRAM_REQUEST_TIMEOUT_SEC,
            )

    async def _send_photo_then_messages_once(
        self, photo_path: str, caption: str, *message_texts: str
    ) -> None:
        assert self._token is not None and self._chat_id is not None
        async with Bot(token=self._token) as bot:
            await asyncio.wait_for(
                bot.send_photo(
                    chat_id=self._chat_id,
                    photo=FSInputFile(photo_path),
                    caption=caption,
                ),
                timeout=_TELEGRAM_REQUEST_TIMEOUT_SEC,
            )
            for msg in message_texts:
                await asyncio.wait_for(
                    bot.send_message(chat_id=self._chat_id, text=msg),
                    timeout=_TELEGRAM_REQUEST_TIMEOUT_SEC,
                )

    def _send_sync_photo_then_text(
        self, photo_path: str, caption: str, *message_texts: str
    ) -> bool:
        max_attempts = 1 + _TELEGRAM_SEND_RETRIES
        for attempt in range(1, max_attempts + 1):
            try:
                asyncio.run(
                    self._send_photo_then_messages_once(
                        photo_path, caption, *message_texts
                    )
                )
                if attempt > 1:
                    log.info(
                        "telegram_send_photo_bundle_recovered_after_retry",
                        attempt=attempt,
                    )
                return True
            except Exception as exc:
                if not _is_retryable_telegram_send_error(exc):
                    log.exception("telegram_send_photo_bundle_failed_nonretryable")
                    return False
                if attempt >= max_attempts:
                    log.exception(
                        "telegram_send_photo_bundle_failed_after_retries",
                        attempts=max_attempts,
                    )
                    return False
                log.warning(
                    "telegram_send_photo_bundle_retry_scheduled",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    sleep_s=_TELEGRAM_RETRY_DELAY_SEC,
                    error=f"{type(exc).__name__}: {exc}",
                )
                time.sleep(_TELEGRAM_RETRY_DELAY_SEC)
        return False

    def _send_sync(self, text: str) -> None:
        max_attempts = 1 + _TELEGRAM_SEND_RETRIES
        for attempt in range(1, max_attempts + 1):
            try:
                asyncio.run(self._send_message_once(text))
                if attempt > 1:
                    log.info("telegram_send_recovered_after_retry", attempt=attempt)
                return
            except Exception as exc:
                if not _is_retryable_telegram_send_error(exc):
                    log.exception("telegram_send_failed_nonretryable")
                    return
                if attempt >= max_attempts:
                    log.exception(
                        "telegram_send_failed_after_retries",
                        attempts=max_attempts,
                    )
                    return
                log.warning(
                    "telegram_send_retry_scheduled",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    sleep_s=_TELEGRAM_RETRY_DELAY_SEC,
                    error=f"{type(exc).__name__}: {exc}",
                )
                time.sleep(_TELEGRAM_RETRY_DELAY_SEC)
