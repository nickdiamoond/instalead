"""Unit tests for Step 5 per-lead Sherlock resolution (nick log + photo authoritative)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pipeline_lib.constants import (  # noqa: E402
    SH_STATUS_FOUND_PHOTO,
    SH_STATUS_NO_MATCH,
)
from scripts.pipeline_lib.step5_sherlock import (  # noqa: E402
    _resolve_one_lead_via_sherlock,
)


@pytest.fixture
def face_photo(tmp_path: Path) -> Path:
    p = tmp_path / "face.jpg"
    p.write_bytes(b"x")
    return p


def test_nick_hit_still_runs_photo_and_does_not_set_found_nick_status(
    face_photo: Path,
) -> None:
    sherlock = MagicMock()
    sherlock.enqueue_nick.return_value = {"id": "nick-task-1"}
    sherlock.enqueue_photo.return_value = {"id": "photo-task-1"}
    sherlock.wait_for_task.side_effect = [
        {
            "status": "completed",
            "result": {
                "results": [
                    {
                        "profile_url": "https://t.me/alice",
                        "username": "AliceTG",
                        "link": "https://t.me/AliceTG",
                    }
                ]
            },
        },
        {
            "status": "completed",
            "result": {"results": []},
        },
    ]

    lead = {
        "username": "alice",
        "full_name": "Alice",
        "face_photo_path": str(face_photo),
    }
    res = _resolve_one_lead_via_sherlock(
        sherlock,
        lead,
        nick_cfg={},
        photo_cfg={},
        task_cfg={},
        deepseek=None,
        usermatch_prompt="{username} {full_name} {candidates}",
    )

    sherlock.enqueue_photo.assert_called_once()
    assert res["nick_hit"] is True
    assert res["nick_telegram_username"] == "AliceTG"
    assert res["status"] == SH_STATUS_NO_MATCH
    assert res["status"] != "found_nick"
    assert res.get("telegram_username") is None
    assert res["photo_search_ran"] is True


def test_photo_exact_match_after_nick_miss(face_photo: Path) -> None:
    sherlock = MagicMock()
    sherlock.enqueue_nick.return_value = {"id": "nick-task-1"}
    sherlock.enqueue_photo.return_value = {"id": "photo-task-1"}
    sherlock.wait_for_task.side_effect = [
        {"status": "completed", "result": {"results": []}},
        {
            "status": "completed",
            "result": {
                "results": [
                    {
                        "status": "точное совпадение",
                        "phone": "+7999",
                        "link": "https://t.me/bobtg",
                        "person": "Bob",
                    }
                ]
            },
        },
    ]

    res = _resolve_one_lead_via_sherlock(
        sherlock,
        {
            "username": "bob",
            "full_name": "",
            "face_photo_path": str(face_photo),
        },
        nick_cfg={},
        photo_cfg={},
        task_cfg={},
        deepseek=None,
        usermatch_prompt="{username} {full_name} {candidates}",
    )

    assert res["nick_hit"] is False
    assert res["status"] == SH_STATUS_FOUND_PHOTO
    assert res["phone"] == "+7999"
