"""Unit tests for ``src.comment_normalizer.normalize_apidojo_api``.

Covers the contract Step 3 of ``scripts/pipeline.py`` relies on:
    * commenter username + numeric user_id are in the nested ``user`` block
    * media_id is the int decoded from postId (so the existing
      shortcode_to_id fuzzy lookup matches without precision tricks)
    * camelCase fields (fullName, isPrivate, isVerified, profilePicUrl)
      are renamed to snake_case
    * ``message`` is renamed to ``text``
    * ``createdAt`` ISO is converted to a unix int (best-effort)
    * items without a username are dropped (None)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.comment_normalizer import (
    _iso_to_unix,
    _shortcode_to_id,
    normalize_apidojo_api,
)


SAMPLE_ITEM = {
    "inputSource": "https://www.instagram.com/p/DXdv7B1jFDF",
    "postId": "DXdv7B1jFDF",
    "type": "comment",
    "id": "17919079491232885",
    "userId": "6677757292",
    "message": "Great post!",
    "createdAt": "2026-04-26T20:36:33.000Z",
    "likeCount": 3,
    "replyCount": 1,
    "user": {
        "id": "6677757292",
        "username": "alice",
        "fullName": "Alice Wonderland",
        "isVerified": True,
        "isPrivate": False,
        "profilePicUrl": "https://cdn/alice.jpg",
    },
    "isRanked": True,
}


def test_normalize_returns_louisdeconinck_shape():
    out = normalize_apidojo_api(SAMPLE_ITEM)
    assert out is not None
    assert out["pk"] == "17919079491232885"
    assert out["user_id"] == "6677757292"
    assert out["text"] == "Great post!"
    assert out["post_shortcode"] == "DXdv7B1jFDF"
    assert out["_source_actor"] == "apidojo-api"


def test_user_block_is_snake_cased():
    out = normalize_apidojo_api(SAMPLE_ITEM)
    user = out["user"]
    assert user["pk"] == "6677757292"
    assert user["id"] == "6677757292"
    assert user["username"] == "alice"
    assert user["full_name"] == "Alice Wonderland"
    assert user["is_verified"] is True
    assert user["is_private"] is False
    assert user["profile_pic_url"] == "https://cdn/alice.jpg"


def test_media_id_decodes_postid_exactly():
    out = normalize_apidojo_api(SAMPLE_ITEM)
    assert out["media_id"] == _shortcode_to_id("DXdv7B1jFDF")
    assert isinstance(out["media_id"], int)


def test_created_at_iso_converted_to_unix():
    out = normalize_apidojo_api(SAMPLE_ITEM)
    from datetime import datetime, timezone
    expected = int(
        datetime(2026, 4, 26, 20, 36, 33, tzinfo=timezone.utc).timestamp()
    )
    assert out["created_at_utc"] == expected


def test_missing_username_returns_none():
    item = dict(SAMPLE_ITEM)
    item["user"] = {"id": "1", "fullName": "Bob"}
    assert normalize_apidojo_api(item) is None


def test_missing_user_block_returns_none():
    item = dict(SAMPLE_ITEM)
    del item["user"]
    assert normalize_apidojo_api(item) is None


def test_missing_optional_fields_dont_crash():
    item = {
        "id": "1",
        "userId": "2",
        "postId": "ABC",
        "user": {"id": "2", "username": "minimal"},
    }
    out = normalize_apidojo_api(item)
    assert out is not None
    assert out["user"]["username"] == "minimal"
    assert out["user"]["full_name"] is None
    assert out["user"]["is_private"] is None
    assert out["text"] == ""
    assert out["created_at_utc"] is None


def test_iso_to_unix_passes_through_unparseable():
    assert _iso_to_unix("not a date") == "not a date"
    assert _iso_to_unix(None) is None
    assert _iso_to_unix(1745603812) == 1745603812


def test_false_isprivate_preserved():
    """Regression: an explicit False must not be coerced to None."""
    out = normalize_apidojo_api(SAMPLE_ITEM)
    assert out["user"]["is_private"] is False
