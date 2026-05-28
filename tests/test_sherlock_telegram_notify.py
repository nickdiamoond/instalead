"""Tests for Step 5 per-lead Sherlock Telegram payloads."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.telegram_notifier import (
    PipelineTelegramNotifier,
    build_sherlock_face_photo_caption,
    build_sherlock_lead_notification_text,
    build_sherlock_lead_result_summary_text,
    truncate_for_telegram,
)


def test_build_nick_hit_includes_prefixed_handle() -> None:
    lead = {
        "username": "alice",
        "context_post_url": "https://www.instagram.com/p/ABC/",
        "context_post_shortcode": "ABC",
        "context_comment_pk": "123",
    }
    res = {
        "status": "no_match",
        "nick_hit": True,
        "nick_telegram_username": "AliceTG",
        "nick_search_ran": True,
        "nick_skipped_dot": False,
        "photo_search_ran": True,
        "photo_task": {"status": "completed", "id": "tid", "result": {"results": []}},
        "nick_query": "@alice",
    }
    txt = build_sherlock_lead_notification_text(lead, res)
    assert "Telegram match (nick search): @AliceTG" in txt
    assert "Photo search — full Sherlock task JSON" in txt
    assert "Comment: https://www.instagram.com/p/ABC/c/123/" in txt
    assert "Sherlock contact saved to DB" not in txt


def test_build_photo_search_has_json_when_no_nick_hit() -> None:
    lead = {"username": "bob", "context_post_url": "https://ex/p/zz/"}
    res = {
        "status": "no_match",
        "nick_hit": False,
        "nick_search_ran": True,
        "nick_skipped_dot": False,
        "photo_search_ran": True,
        "photo_task": {"status": "completed", "id": "tid", "result": {"results": []}},
        "nick_query": "@bob",
    }
    txt = build_sherlock_lead_notification_text(lead, res)
    assert "Telegram nick not found for @bob; photo search." in txt
    assert '"status": "completed"' in txt
    assert '"id": "tid"' in txt
    assert "Sherlock contact saved to DB" not in txt


def test_build_nick_saved_normalizes_without_at_prefix() -> None:
    txt = build_sherlock_lead_notification_text(
        {"username": "carol"},
        {
            "status": "no_match",
            "nick_hit": True,
            "nick_telegram_username": "@carolk",
            "nick_search_ran": True,
            "nick_skipped_dot": False,
            "photo_search_ran": True,
            "photo_task": {"id": "t"},
            "nick_query": "@carol",
        },
    )
    assert "Telegram match (nick search): @carolk" in txt


def test_build_nick_hit_and_photo_json_both_present() -> None:
    txt = build_sherlock_lead_notification_text(
        {"username": "dana"},
        {
            "status": "no_match",
            "nick_hit": True,
            "nick_telegram_username": "DanaTG",
            "nick_search_ran": True,
            "nick_skipped_dot": False,
            "photo_search_ran": True,
            "photo_task": {"status": "completed", "id": "photo-1"},
            "nick_query": "@dana",
        },
    )
    assert "Telegram match (nick search): @DanaTG" in txt
    assert "Photo search — full Sherlock task JSON" in txt
    assert '"id": "photo-1"' in txt


def test_truncate_for_long_payload() -> None:
    blob = build_sherlock_lead_notification_text(
        {"username": "d", "context_post_url": "u"},
        {
            "status": "found_photo",
            "phone": "+1",
            "nick_hit": False,
            "nick_search_ran": True,
            "nick_skipped_dot": False,
            "photo_search_ran": True,
            "nick_query": "@d",
            "photo_task": {"x": "y" * 10_000},
        },
    )
    short = truncate_for_telegram(blob)
    assert len(short) <= 4096
    assert "truncated for Telegram limit" in short


def test_build_face_photo_caption_with_percent_and_na() -> None:
    lead = {"username": "alice"}
    assert (
        "@alice\nInsightFace (SCRFD det. confidence): 87.5%"
        == build_sherlock_face_photo_caption(lead, 87.5)
    )
    assert "n/a" in build_sherlock_face_photo_caption(lead, None)


@patch("src.telegram_notifier.Bot")
def test_notify_sherlock_lead_dispatches(mock_bot_class: MagicMock) -> None:
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("tok", -1, enabled=True)
    n.notify_sherlock_lead(
        {"username": "eve"},
        {
            "status": "no_match",
            "nick_hit": True,
            "nick_telegram_username": "EveTG",
            "nick_search_ran": True,
            "nick_skipped_dot": False,
            "photo_search_ran": True,
            "photo_task": {"id": "t"},
            "nick_query": "@eve",
        },
    )
    assert mock_bot.send_message.await_count == 2
    calls = mock_bot.send_message.await_args_list
    assert all(c.kwargs["chat_id"] == -1 for c in calls)


def test_build_sherlock_result_summary_nick_hit() -> None:
    lead = {
        "username": "alice",
        "full_name": "Alice Иванова",
        "context_post_url": "https://www.instagram.com/p/ABC/",
        "context_post_shortcode": "ABC",
        "context_comment_pk": "999",
    }
    res = {
        "status": "found_nick",
        "telegram_username": "AliceTG",
    }
    s = build_sherlock_lead_result_summary_text(lead, res)
    assert s.startswith('Результат по "AliceTG"')
    assert "Профиль: https://www.instagram.com/alice/" in s
    assert "Ник в тг: @AliceTG" in s
    assert "Имя пользователя из био инсты: Alice Иванова" in s
    assert "совпадение: найден по нику" in s


def test_build_sherlock_result_summary_photo_exact() -> None:
    lead = {"username": "bob", "context_post_url": "https://ex/p/1/"}
    res = {
        "status": "found_photo",
        "phone": "+7999",
        "sherlock_link": "https://t.me/bobtg",
        "photo_match_kind": "exact",
        "sherlock_person": "Иван Иванов",
    }
    s = build_sherlock_lead_result_summary_text(lead, res)
    assert 'Результат по "bobtg"' in s
    assert "Профиль: https://www.instagram.com/bob/" in s
    assert "person: Иван" in s
    assert "Телефон: +7999" in s
    assert "совпадение: точное совпадение" in s


def test_build_sherlock_result_summary_no_match() -> None:
    lead = {
        "username": "carol",
        "full_name": "Кэрол Смит",
        "context_post_url": None,
    }
    res = {"status": "no_match"}
    s = build_sherlock_lead_result_summary_text(lead, res)
    assert 'Результат по "carol"' in s
    assert "Профиль: https://www.instagram.com/carol/" in s
    assert "ФИО из Instagram: Кэрол Смит" in s
    assert "совпадение: пользователь не найден" in s


@patch("src.telegram_notifier.Bot")
@patch("src.telegram_notifier.compute_insightface_best_det_percent", return_value=90.0)
def test_notify_sherlock_lead_photo_search_sends_photo_then_text(
    _mock_pct: MagicMock,
    mock_bot_class: MagicMock,
    tmp_path: Path,
) -> None:
    img = tmp_path / "face.jpg"
    img.write_bytes(b"x")

    mock_bot = MagicMock()
    mock_bot.send_photo = AsyncMock()
    mock_bot.send_message = AsyncMock()
    mock_bot_class.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot_class.return_value.__aexit__ = AsyncMock(return_value=False)

    n = PipelineTelegramNotifier("tok", -1, enabled=True)
    n.notify_sherlock_lead(
        {"username": "bob", "face_photo_path": str(img)},
        {
            "status": "no_match",
            "nick_hit": False,
            "nick_search_ran": True,
            "nick_skipped_dot": False,
            "photo_search_ran": True,
            "photo_task": {"id": "t"},
            "nick_query": "@bob",
        },
        cfg={"face_detection": {}},
    )
    mock_bot.send_photo.assert_awaited_once()
    assert mock_bot.send_message.await_count == 1
    pc = mock_bot.send_photo.await_args.kwargs["caption"]
    assert "@bob" in pc and "90.0%" in pc
