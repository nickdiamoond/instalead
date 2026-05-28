"""Unit tests for Sherlock /v1/health pre-flight helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.sherlock_client import (
    SherlockClient,
    SherlockError,
    pool_idle_count,
    probe_health_pool_idle,
)


def test_pool_idle_count_parses_int_and_string() -> None:
    assert pool_idle_count({"by_status": {"idle": 2}}) == 2
    assert pool_idle_count({"by_status": {"idle": "3"}}) == 3
    assert pool_idle_count(None) is None
    assert pool_idle_count({"by_status": {}}) is None


def test_probe_health_pool_idle_success() -> None:
    client = MagicMock(spec=SherlockClient)
    client.health.return_value = {
        "status": "ok",
        "pool": {"total": 3, "by_status": {"idle": 2, "busy": 1}},
    }
    body, idle = probe_health_pool_idle(client, max_attempts=3)
    assert body is not None
    assert idle == 2
    assert client.health.call_count == 1


def test_probe_health_pool_idle_zero_idle() -> None:
    client = MagicMock(spec=SherlockClient)
    client.health.return_value = {
        "pool": {"by_status": {"idle": 0, "busy": 3}},
    }
    body, idle = probe_health_pool_idle(client, max_attempts=3)
    assert body is not None
    assert idle == 0


def test_probe_health_pool_idle_retries_then_succeeds() -> None:
    client = MagicMock(spec=SherlockClient)
    client.health.side_effect = [
        requests.Timeout("timed out"),
        {"pool": {"by_status": {"idle": 1}}},
    ]
    body, idle = probe_health_pool_idle(client, max_attempts=3, retry_delay_secs=0)
    assert body is not None
    assert idle == 1
    assert client.health.call_count == 2


def test_probe_health_pool_idle_exhausted() -> None:
    client = MagicMock(spec=SherlockClient)
    client.health.side_effect = SherlockError("health failed: HTTP 503")
    body, idle = probe_health_pool_idle(client, max_attempts=3, retry_delay_secs=0)
    assert body is None
    assert idle == 0
    assert client.health.call_count == 3


@patch("src.sherlock_client.time.sleep")
def test_probe_health_pool_idle_sleeps_between_failed_attempts(
    mock_sleep: MagicMock,
) -> None:
    """60s pause after each failure, but not after the final attempt."""
    client = MagicMock(spec=SherlockClient)
    client.health.side_effect = SherlockError("HTTP 503")
    probe_health_pool_idle(client, max_attempts=3)
    # 3 attempts -> 2 inter-attempt pauses (none after the last).
    assert mock_sleep.call_count == 2
    assert all(c.args[0] == 60.0 for c in mock_sleep.call_args_list)


@patch("src.sherlock_client.time.sleep")
def test_probe_health_pool_idle_no_sleep_on_first_success(
    mock_sleep: MagicMock,
) -> None:
    client = MagicMock(spec=SherlockClient)
    client.health.return_value = {"pool": {"by_status": {"idle": 2}}}
    probe_health_pool_idle(client, max_attempts=3)
    mock_sleep.assert_not_called()
