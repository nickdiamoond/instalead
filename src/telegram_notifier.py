"""Optional Telegram reports after pipeline Step 1–5.

Uses Bot API ``sendMessage`` via aiogram. Disabled when ``TELEGRAM_BOT_TOKEN``
is missing or ``telegram.report_chat_id`` is absent/invalid — the pipeline
keeps running."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from aiogram import Bot
from aiogram.exceptions import TelegramNetworkError, TelegramServerError
from aiogram.types import FSInputFile

from src.logger import get_logger

log = get_logger("telegram_notifier")

TOKEN_ENV_VAR = "TELEGRAM_BOT_TOKEN"

# One initial attempt + this many retries; 20s pause after each failure.
_TELEGRAM_SEND_RETRIES = 3
_TELEGRAM_RETRY_DELAY_SEC = 20.0
_TELEGRAM_REQUEST_TIMEOUT_SEC = 20.0


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


def build_step2_scored_video_message(
    post_url: str,
    raw_score: dict[str, Any],
    resolved_relevance: str,
    combined_text: str,
) -> str:
    """One Telegram message per video scored in Step 2 (caption + transcript)."""
    link = (post_url or "").strip() or "(unknown)"
    lines: list[str] = [
        "Step 2: scored video",
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
            "Video text (caption + transcript):",
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

    status = str(res.get("status") or "error")

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

    contact_saved_yes = status == _SH_FOUND_NICK or (
        status == _SH_FOUND_PHOTO
        and bool(res.get("phone") or res.get("telegram_username"))
    )
    lines.append("")
    lines.append(f"Sherlock contact saved to DB: {'yes' if contact_saved_yes else 'no'}")

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

    def notify_step1(self, new_posts: int, realtors_count: int) -> None:
        if not self._enabled:
            return
        text = (
            "Step 1: "
            f"{new_posts} new post(s) saved;\n "
            f"searched {realtors_count} active realtor(s)."
        )
        self._send_sync(text)

    def notify_step2_scored_video(
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
            build_step2_scored_video_message(
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
        text = truncate_for_telegram(build_sherlock_lead_notification_text(lead, res))
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
                if self._send_sync_photo_then_text(photo_path, caption, text):
                    return
        self._send_sync(text)

    async def _send_message_once(self, text: str) -> None:
        assert self._token is not None and self._chat_id is not None
        async with Bot(token=self._token) as bot:
            await asyncio.wait_for(
                bot.send_message(chat_id=self._chat_id, text=text),
                timeout=_TELEGRAM_REQUEST_TIMEOUT_SEC,
            )

    async def _send_photo_then_message_once(
        self, photo_path: str, caption: str, message_text: str
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
            await asyncio.wait_for(
                bot.send_message(chat_id=self._chat_id, text=message_text),
                timeout=_TELEGRAM_REQUEST_TIMEOUT_SEC,
            )

    def _send_sync_photo_then_text(
        self, photo_path: str, caption: str, message_text: str
    ) -> bool:
        max_attempts = 1 + _TELEGRAM_SEND_RETRIES
        for attempt in range(1, max_attempts + 1):
            try:
                asyncio.run(
                    self._send_photo_then_message_once(
                        photo_path, caption, message_text
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
