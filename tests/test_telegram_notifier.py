"""Tests for ``src.telegram_notifier.PipelineTelegramNotifier``."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aiogram.exceptions import TelegramNetworkError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.telegram_notifier import (
    PipelineTelegramNotifier,
    _STEP1_NEW_POST_MESSAGE_DELAY_SEC,
    build_apify_run_alert_text,
    build_deepseek_batch_all_failed_alert_text,
    build_nexara_batch_all_failed_alert_text,
    build_sherlock_batch_all_failed_alert_text,
    build_step1_date_filter_section_lines,
    build_step1_new_post_message,
    build_step1_pipeline_summary_telegram_text,
    build_step5_sherlock_summary_telegram_text,
    build_step2_human_confirm_body,
    build_step2_scored_post_message,
    is_apify_run_failure_status,
    step1_display_content_type,
)


def test_build_step1_date_filter_realtors_vs_client() -> None:
    rel = build_step1_date_filter_section_lines(
        discovery_mode="realtors",
        posts_max_age_days=14,
        age_dropped_client=5,
        age_kept_missing_ts=1,
    )
    rel_s = "\n".join(rel)
    assert "onlyPostsNewerThan" in rel_s
    assert "Client-side" in rel_s
    assert "Dropped — older than window: 5" in rel_s
    assert "missing or unparseable" in rel_s

    ht = build_step1_date_filter_section_lines(
        discovery_mode="hashtags",
        posts_max_age_days=7,
        age_dropped_client=12,
        age_kept_missing_ts=3,
    )
    s = "\n".join(ht)
    assert "Dropped — older than window: 12" in s
    assert "missing or unparseable" in s


def test_is_apify_run_failure_status() -> None:
    assert is_apify_run_failure_status("FAILED")
    assert is_apify_run_failure_status("TIMED-OUT")
    assert is_apify_run_failure_status("timed_out")
    assert is_apify_run_failure_status("ABORTED")
    assert not is_apify_run_failure_status("SUCCEEDED")
    assert not is_apify_run_failure_status("RUNNING")
    assert not is_apify_run_failure_status(None)


def test_build_apify_run_alert_text() -> None:
    text = build_apify_run_alert_text(
        step="Step 1",
        actor_id="apify/instagram-post-scraper",
        run_id="abc123",
        status="FAILED",
    )
    assert "Apify run failed" in text
    assert "Step: Step 1" in text
    assert "Status: FAILED" in text
    assert "Run: abc123" in text


@patch("src.telegram_notifier.Bot")
def test_maybe_notify_apify_run_failure_sends_to_alert_chat(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier(
        "fake-token",
        -42,
        alert_chat_id=-777,
        enabled=False,
    )
    n.maybe_notify_apify_run_failure(
        {"id": "run1", "status": "FAILED"},
        actor_id="apify/instagram-post-scraper",
        step="Step 1",
    )

    mock_bot.send_message.assert_awaited_once()
    assert mock_bot.send_message.await_args.kwargs["chat_id"] == -777
    assert "Apify run failed" in mock_bot.send_message.await_args.kwargs["text"]


@patch("src.telegram_notifier.Bot")
def test_maybe_notify_apify_run_failure_skips_succeeded(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier(
        "fake-token",
        -42,
        alert_chat_id=-777,
        enabled=True,
    )
    n.maybe_notify_apify_run_failure(
        {"id": "run1", "status": "SUCCEEDED"},
        actor_id="apify/instagram-post-scraper",
        step="Step 1",
    )
    mock_bot.send_message.assert_not_awaited()


def test_build_deepseek_batch_all_failed_alert_text() -> None:
    text = build_deepseek_batch_all_failed_alert_text(deepseek_calls=12)
    assert "DeepSeek API error" in text
    assert "Step: Step 2" in text
    assert "All 12 DeepSeek" in text


@patch("src.telegram_notifier.Bot")
def test_maybe_notify_deepseek_batch_all_failed_sends_when_all_failed(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier(
        "fake-token",
        -42,
        alert_chat_id=-777,
        enabled=False,
    )
    n.maybe_notify_deepseek_batch_all_failed(
        deepseek_calls=12,
        deepseek_succeeded=0,
    )

    mock_bot.send_message.assert_awaited_once()
    assert mock_bot.send_message.await_args.kwargs["chat_id"] == -777


def test_build_deepseek_batch_all_failed_alert_text_step5() -> None:
    text = build_deepseek_batch_all_failed_alert_text(
        deepseek_calls=3,
        step="Step 5",
        call_kind="usermatch call(s)",
        outcome_label="usermatch picks",
    )
    assert "Step: Step 5" in text
    assert "usermatch call(s)" in text


@patch("src.telegram_notifier.Bot")
def test_maybe_notify_deepseek_batch_all_failed_skips_partial_success(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier(
        "fake-token",
        -42,
        alert_chat_id=-777,
        enabled=False,
    )
    n.maybe_notify_deepseek_batch_all_failed(
        deepseek_calls=100,
        deepseek_succeeded=1,
    )
    mock_bot.send_message.assert_not_awaited()


def test_build_nexara_batch_all_failed_alert_text() -> None:
    text = build_nexara_batch_all_failed_alert_text(transcription_attempts=10)
    assert "Nexara API error" in text
    assert "Step: Step 2" in text
    assert "All 10 transcription" in text


@patch("src.telegram_notifier.Bot")
def test_maybe_notify_nexara_batch_all_failed_sends_when_all_failed(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier(
        "fake-token",
        -42,
        alert_chat_id=-777,
        enabled=False,
    )
    n.maybe_notify_nexara_batch_all_failed(
        transcription_attempts=10,
        transcribed_count=0,
    )

    mock_bot.send_message.assert_awaited_once()
    assert mock_bot.send_message.await_args.kwargs["chat_id"] == -777


@patch("src.telegram_notifier.Bot")
def test_maybe_notify_nexara_batch_all_failed_skips_partial_success(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier(
        "fake-token",
        -42,
        alert_chat_id=-777,
        enabled=False,
    )
    n.maybe_notify_nexara_batch_all_failed(
        transcription_attempts=100,
        transcribed_count=1,
    )
    mock_bot.send_message.assert_not_awaited()


def test_build_sherlock_batch_all_failed_alert_text() -> None:
    text = build_sherlock_batch_all_failed_alert_text(leads_processed=5)
    assert "Sherlock API error" in text
    assert "Step: Step 5" in text
    assert "All 5 lead(s)" in text


@patch("src.telegram_notifier.Bot")
def test_maybe_notify_sherlock_batch_all_failed_sends_when_every_lead_errored(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier(
        "fake-token",
        -42,
        alert_chat_id=-777,
        enabled=False,
    )
    n.maybe_notify_sherlock_batch_all_failed(
        leads_processed=5,
        error_count=5,
    )

    mock_bot.send_message.assert_awaited_once()
    assert mock_bot.send_message.await_args.kwargs["chat_id"] == -777
    assert "All 5 lead(s)" in mock_bot.send_message.await_args.kwargs["text"]


@patch("src.telegram_notifier.Bot")
def test_maybe_notify_sherlock_batch_all_failed_skips_partial_errors(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier(
        "fake-token",
        -42,
        alert_chat_id=-777,
        enabled=False,
    )
    n.maybe_notify_sherlock_batch_all_failed(
        leads_processed=5,
        error_count=4,
    )
    mock_bot.send_message.assert_not_awaited()


def test_maybe_notify_apify_run_failure_no_alert_chat() -> None:
    n = PipelineTelegramNotifier("fake-token", -42, enabled=True)
    with patch("src.telegram_notifier.asyncio.run") as mock_run:
        n.maybe_notify_apify_run_failure(
            {"id": "run1", "status": "FAILED"},
            actor_id="apify/instagram-post-scraper",
            step="Step 1",
        )
    mock_run.assert_not_called()


def test_build_step1_pipeline_summary_telegram_multiline() -> None:
    text = build_step1_pipeline_summary_telegram_text(
        new_posts=0,
        source_count=2,
        discovery_mode="hashtags",
        min_comments=10,
        fetched_total=50,
        updated_posts=1,
        with_video_count=3,
        skipped_no_video_url=2,
        step1_skip_low_comments=40,
        step1_skip_no_shortcode=0,
        step1_existing_unchanged=5,
        cost_usd=0.1234,
        posts_max_age_days=7,
        age_dropped_client=9,
        age_kept_missing_ts=1,
    )
    assert "\n" in text
    assert "Date filter" in text
    assert "posts_max_age_days = 7" in text
    assert "Dropped — older than window: 9" in text
    assert "Gate breakdown" in text


def test_build_step5_sherlock_summary_telegram_multiline() -> None:
    text = build_step5_sherlock_summary_telegram_text(
        pulled=42,
        batch_limit=1000,
        counters={
            "found_nick": 3,
            "found_photo": 4,
            "no_match": 31,
            "no_face_photo": 2,
            "error": 2,
        },
        step5_deepseek_calls=2,
        step5_deepseek_api_ok=1,
    )
    assert "\n" in text
    assert "Step 5" in text
    assert "Sherlock summary" in text
    assert "Leads pulled from DB (this run): 42 (batch limit 1000)" in text
    assert "Contact found (total): 7" in text
    assert "Found via nick: 3" in text
    assert "Found via photo: 4" in text
    assert "Not found: 33" in text
    assert "no_match: 31" in text
    assert "Errors: 2" in text
    assert "DeepSeek usermatch: 1/2 API ok" in text


def test_step1_display_content_type_known() -> None:
    assert step1_display_content_type({"type": "Sidecar"}) == "Sidecar"
    assert step1_display_content_type({"type": "Video"}) == "Video"


def test_build_step1_new_post_message_shape() -> None:
    msg = build_step1_new_post_message(
        {
            "shortCode": "AbCdEfGh",
            "url": "https://www.instagram.com/p/AbCdEfGh/",
            "timestamp": "2026-03-10T14:00:00.000Z",
            "ownerUsername": "seller_spb",
            "hashtags": ["спб", "новостройка"],
            "commentsCount": 40,
            "likesCount": 200,
            "type": "Video",
            "locationName": "Санкт-Петербург",
            "locationId": "12345",
        }
    )
    assert "https://www.instagram.com/p/AbCdEfGh/" in msg
    assert "2026-03-10 14:00:00 UTC" in msg
    assert "seller_spb" in msg
    assert "#спб" in msg and "#новостройка" in msg
    assert "commentsCount" in msg and "40" in msg
    assert "likesCount" in msg and "200" in msg
    assert "type" in msg and "Video" in msg
    assert "locationName" in msg
    assert "Санкт-Петербург" in msg
    assert "locationId" in msg and "12345" in msg


def test_build_step1_new_post_message_omits_geo_when_absent() -> None:
    msg = build_step1_new_post_message(
        {
            "shortCode": "Xx",
            "ownerUsername": "u",
            "hashtags": [],
            "commentsCount": 1,
            "likesCount": 2,
            "type": "Image",
        }
    )
    assert "locationName" not in msg
    assert "locationId" not in msg


def test_build_step2_scored_post_message_shape() -> None:
    msg = build_step2_scored_post_message(
        "https://www.instagram.com/p/XX/",
        {
            "is_real_estate": True,
            "has_call_to_action": True,
            "call_to_action_type": "comment",
        },
        "relevant",
        "Caption here\n\nTranscript here",
    )
    assert "Step 2: scored post" in msg
    assert "https://www.instagram.com/p/XX/" in msg
    assert "is_real_estate: True" in msg
    assert "Pipeline relevance: relevant" in msg
    assert "Caption here" in msg and "Transcript here" in msg


def test_build_step2_human_confirm_body_shape() -> None:
    ig_url = "https://www.instagram.com/p/ABC123/"
    body = build_step2_human_confirm_body(
        index=1,
        total=3,
        post_url=ig_url,
        combined_text="Line one\n\nLine two",
        location="Москва",
    )
    assert body.startswith("[1/3]")
    assert "ПОДТВЕРДИТЕ, ЧТО ПОСТ ЦЕЛЕВОЙ" in body
    headline_pos = body.index("ПОДТВЕРДИТЕ, ЧТО ПОСТ ЦЕЛЕВОЙ")
    link_pos = body.index(ig_url)
    assert link_pos > headline_pos
    assert body[headline_pos:link_pos].strip() == "ПОДТВЕРДИТЕ, ЧТО ПОСТ ЦЕЛЕВОЙ"
    loc_pos = body.index("Локация - Москва")
    assert loc_pos > link_pos
    assert "Line one" in body and "Line two" in body


def test_build_step2_human_confirm_body_missing_location() -> None:
    body = build_step2_human_confirm_body(
        index=1,
        total=1,
        post_url="https://www.instagram.com/p/X/",
        combined_text="text",
    )
    assert "Локация - отсутствует" in body


def test_inline_confirm_token_and_chat_uses_result_chat_id() -> None:
    n = PipelineTelegramNotifier("fake-token", -42, result_chat_id=-999, enabled=True)
    assert n.inline_confirm_token_and_chat() == ("fake-token", -999)


def test_inline_confirm_token_and_chat_falls_back_to_report() -> None:
    n = PipelineTelegramNotifier("fake-token", -42, enabled=True)
    assert n.inline_confirm_token_and_chat() == ("fake-token", -42)


def test_inline_confirm_token_and_chat_disabled() -> None:
    n = PipelineTelegramNotifier("fake-token", -42, enabled=False)
    assert n.inline_confirm_token_and_chat() is None


@patch("src.telegram_notifier.Bot")
def test_notify_sherlock_lead_sends_summary_to_result_chat_on_hit(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -42, result_chat_id=-999, enabled=True)
    n.notify_sherlock_lead(
        {"username": "someuser"},
        {"status": "found_nick", "telegram_username": "SomeTG"},
        cfg=None,
    )
    assert mock_bot.send_message.await_count == 2
    calls = mock_bot.send_message.await_args_list
    assert calls[0].kwargs["chat_id"] == -42
    assert calls[1].kwargs["chat_id"] == -999


@patch("src.telegram_notifier.Bot")
def test_notify_sherlock_lead_no_match_skips_result_chat(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -42, result_chat_id=-999, enabled=True)
    n.notify_sherlock_lead(
        {"username": "someuser"},
        {"status": "no_match"},
        cfg=None,
    )
    assert mock_bot.send_message.await_count == 1
    assert mock_bot.send_message.await_args.kwargs["chat_id"] == -42


@patch("src.telegram_notifier.Bot")
def test_notify_sherlock_lead_both_messages_report_without_result_chat(
    mock_bot_class: MagicMock,
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -42, enabled=True)
    n.notify_sherlock_lead(
        {"username": "u"},
        {"status": "no_match"},
        cfg=None,
    )
    assert mock_bot.send_message.await_count == 1
    assert mock_bot.send_message.await_args.kwargs["chat_id"] == -42


@patch("src.telegram_notifier.Bot")
def test_notify_step2_scored_post_sends(mock_bot_class: MagicMock) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -42, enabled=True)
    n.notify_step2_scored_post(
        post_url="https://www.instagram.com/p/Zz/",
        raw_score={"error": "timeout"},
        resolved_relevance="unknown",
        combined_text="only caption",
    )

    mock_bot.send_message.assert_awaited_once()
    text = mock_bot.send_message.await_args.kwargs["text"]
    assert "https://www.instagram.com/p/Zz/" in text
    assert "error: timeout" in text
    assert "only caption" in text


def test_notifier_skips_when_no_token() -> None:
    n = PipelineTelegramNotifier(None, -123, enabled=True)
    with patch("src.telegram_notifier.asyncio.run") as mock_run:
        n.notify_step1(1, 2)
        n.notify_step2_scored_post(
            post_url="https://www.instagram.com/p/AbC/",
            raw_score={"is_real_estate": True},
            resolved_relevance="relevant",
            combined_text="hello",
        )
    mock_run.assert_not_called()


def test_notifier_skips_when_no_chat_id() -> None:
    n = PipelineTelegramNotifier("tok", None, enabled=True)
    with patch("src.telegram_notifier.asyncio.run") as mock_run:
        n.notify_step1(1, 2)
    mock_run.assert_not_called()


@patch("src.telegram_notifier.time.sleep")
@patch("src.telegram_notifier.Bot")
def test_notify_step1_new_posts_sleeps_between(
    mock_bot_class: MagicMock, mock_sleep: MagicMock
) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -42, enabled=True)
    n.notify_step1_new_posts(
        [
            {"shortCode": "A", "url": "https://www.instagram.com/p/A/", "type": "Image"},
            {"shortCode": "B", "url": "https://www.instagram.com/p/B/", "type": "Image"},
        ]
    )

    assert mock_bot.send_message.await_count == 2
    mock_sleep.assert_called_once_with(_STEP1_NEW_POST_MESSAGE_DELAY_SEC)


@patch("src.telegram_notifier.Bot")
def test_notifier_sends_via_bot(mock_bot_class: MagicMock) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -42, enabled=True)
    n.notify_step1(7, 3)

    mock_bot.send_message.assert_awaited_once()
    kwargs = mock_bot.send_message.await_args.kwargs
    assert kwargs["chat_id"] == -42
    text = kwargs["text"]
    assert "Step 1" in text
    assert "New post(s) saved: 7" in text
    assert "Searched 3 active realtor" in text


def test_build_step1_new_post_message_cookie_extras() -> None:
    msg = build_step1_new_post_message(
        {
            "shortCode": "Zz",
            "url": "https://www.instagram.com/p/Zz/",
            "timestamp": "2026-04-01T10:00:00.000Z",
            "ownerUsername": "owner",
            "hashtags": [],
            "commentsCount": 10,
            "likesCount": 3,
            "type": "Image",
            "searchKeyword": "недвижимость",
            "cookieMediaType": "Photo",
            "captionPreview": "Short caption preview…",
            "cookieMentions": ["@friend"],
        }
    )
    assert "searchKeyword" in msg
    assert "недвижимость" in msg
    assert "cookieMediaType" in msg
    assert "captionPreview" in msg
    assert "@friend" in msg


@patch("src.telegram_notifier.Bot")
def test_notifier_step1_cookie_keywords_wording(mock_bot_class: MagicMock) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -42, enabled=True)
    n.notify_step1(3, 5, discovery_mode="cookie_keywords")

    kwargs = mock_bot.send_message.await_args.kwargs
    text = kwargs["text"]
    assert "New post(s) saved: 3" in text
    assert "5 keyword" in text
    assert "hashtag" not in text.lower()
    assert "realtor" not in text.lower()


@patch("src.telegram_notifier.Bot")
def test_notifier_step1_hashtags_wording(mock_bot_class: MagicMock) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -42, enabled=True)
    n.notify_step1(2, 8, discovery_mode="hashtags")

    kwargs = mock_bot.send_message.await_args.kwargs
    text = kwargs["text"]
    assert "New post(s) saved: 2" in text
    assert "8 hashtag" in text
    assert "realtor" not in text.lower()


def test_notifier_step3_skips_when_no_token() -> None:
    n = PipelineTelegramNotifier(None, -123, enabled=True)
    with patch("src.telegram_notifier.asyncio.run") as mock_run:
        n.notify_step3(10)
    mock_run.assert_not_called()


@patch("src.telegram_notifier.Bot")
def test_notifier_step3_sends_via_bot(mock_bot_class: MagicMock) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -1, enabled=True)
    n.notify_step3(12)

    mock_bot.send_message.assert_awaited_once()
    text = mock_bot.send_message.await_args.kwargs["text"]
    assert "Step 3" in text
    assert "12 new commenter" in text


def test_notifier_step4_skips_when_no_token() -> None:
    n = PipelineTelegramNotifier(None, -123, enabled=True)
    with patch("src.telegram_notifier.asyncio.run") as mock_run:
        n.notify_step4(
            profiles_queued=10,
            single_face_avatar=3,
            face_leader_resolved=2,
            without_suitable_photo=5,
            contacts_from_bio=4,
        )
    mock_run.assert_not_called()


@patch("src.telegram_notifier.Bot")
def test_notifier_step4_sends_via_bot(mock_bot_class: MagicMock) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -1, enabled=True)
    n.notify_step4(
        profiles_queued=50,
        single_face_avatar=20,
        face_leader_resolved=5,
        without_suitable_photo=25,
        contacts_from_bio=10,
    )

    mock_bot.send_message.assert_awaited_once()
    text = mock_bot.send_message.await_args.kwargs["text"]
    assert "Step 4" in text
    assert "50 profile(s) queued" in text
    assert "20 single-face avatar" in text
    assert "5 face photo(s) via face_leader" in text
    assert "25 lead(s) without suitable photo" in text
    assert "10 lead(s) with bio/contact" in text


def test_notifier_step5_summary_skips_when_no_token() -> None:
    n = PipelineTelegramNotifier(None, -123, enabled=True)
    with patch("src.telegram_notifier.asyncio.run") as mock_run:
        n.notify_step5_sherlock_summary(
            pulled=10,
            batch_limit=1000,
            counters={"found_nick": 1, "found_photo": 0, "no_match": 9},
        )
    mock_run.assert_not_called()


@patch("src.telegram_notifier.Bot")
def test_notifier_step5_summary_sends_via_bot(mock_bot_class: MagicMock) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("fake-token", -1, enabled=True)
    n.notify_step5_sherlock_summary(
        pulled=5,
        batch_limit=100,
        counters={
            "found_nick": 1,
            "found_photo": 2,
            "no_match": 2,
            "no_face_photo": 0,
            "error": 0,
        },
    )

    mock_bot.send_message.assert_awaited_once()
    text = mock_bot.send_message.await_args.kwargs["text"]
    assert "Step 5" in text
    assert "Sherlock summary" in text
    assert "Leads pulled from DB (this run): 5 (batch limit 100)" in text
    assert "Found via nick: 1" in text
    assert "Found via photo: 2" in text


def test_notifier_step5_summary_skips_zero_pulled() -> None:
    n = PipelineTelegramNotifier("fake-token", -1, enabled=True)
    with patch("src.telegram_notifier.asyncio.run") as mock_run:
        n.notify_step5_sherlock_summary(
            pulled=0,
            batch_limit=1000,
            counters={},
        )
    mock_run.assert_not_called()


def test_send_sync_retries_on_network_error() -> None:
    """First two asyncio.run failures retry; third uses real asyncio.run + mocked Bot."""
    n = PipelineTelegramNotifier("fake-token", -42, enabled=True)
    attempts: list[int] = []
    real_run = asyncio.run

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    bot_cm = MagicMock()
    bot_cm.__aenter__ = AsyncMock(return_value=mock_bot)
    bot_cm.__aexit__ = AsyncMock(return_value=False)

    def fake_run(coro):
        attempts.append(1)
        if len(attempts) < 3:
            coro.close()
            raise TelegramNetworkError(
                method=MagicMock(),
                message="Server disconnected",
            )
        return real_run(coro)

    with patch("src.telegram_notifier.Bot", return_value=bot_cm):
        with patch("src.telegram_notifier.asyncio.run", side_effect=fake_run):
            n._send_sync("ping")

    assert len(attempts) == 3
    mock_bot.send_message.assert_awaited_once()
