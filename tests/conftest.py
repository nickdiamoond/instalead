"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def telegram_notifier_no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telegram sends use 20s delays between retries — zero them in tests."""
    monkeypatch.setattr("src.telegram_notifier.time.sleep", lambda *_a, **_kw: None)
