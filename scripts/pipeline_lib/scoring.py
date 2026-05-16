import asyncio
import json

from lingua import Language, LanguageDetectorBuilder
from openai import OpenAI

from src.db import LeadDB
from src.telegram_inline_confirm import await_single_yes_no
from src.telegram_notifier import (
    STEP2_INLINE_BTN_CONFIRM,
    STEP2_INLINE_BTN_DENY,
    STEP2_INLINE_SUFFIX_APPROVED,
    STEP2_INLINE_SUFFIX_DENIED,
    build_step2_human_confirm_body,
    truncate_step2_human_confirm_body,
)

from scripts.pipeline_lib.logging import log

RUSSIAN_LANGUAGE_DETECTOR = (
    LanguageDetectorBuilder.from_all_spoken_languages().build()
)


def score_caption(
    client: OpenAI, caption: str, *, relevance_prompt: str
) -> dict:
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": relevance_prompt},
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
            location=item.get("location"),
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
    sent to ``deepseek.relevance_prompt``.
    """
    parts: list[str] = []
    if caption and caption.strip():
        parts.append(caption.strip())
    if transcript and transcript.strip():
        parts.append(transcript.strip())
    return "\n\n".join(parts)
