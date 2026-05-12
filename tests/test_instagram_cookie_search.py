"""Unit tests for ``src.instagram_cookie_search``."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ig_media_payload import filter_items_within_max_age
from src.instagram_cookie_search import (
    cookies_json_string_for_actor,
    dedupe_keyword_items_by_shortcode,
    normalize_keyword_search_item,
    shortcode_from_post_url,
)


def test_shortcode_from_post_url() -> None:
    assert shortcode_from_post_url("https://www.instagram.com/p/AbCdEf/") == "AbCdEf"
    assert shortcode_from_post_url("https://www.instagram.com/reel/XyZ123/") == "XyZ123"


def test_cookies_json_string_simple_pair() -> None:
    s = cookies_json_string_for_actor("sessionid=abc123def")
    assert "sessionid" in s
    assert "abc123def" in s
    assert s.startswith("[")


def test_normalize_success_post_shape() -> None:
    n = normalize_keyword_search_item(
        {
            "status": "success",
            "post_url": "https://www.instagram.com/p/AbCdXyZ/",
            "username": "seller_spb",
            "comment_count": 12,
            "like_count": 5,
            "caption": "flat for sale",
            "pub_date": "2026-01-15T12:00:00.000Z",
            "search_keyword": "недвижимостьспб",
            "media_urls": [],
            "media_type": "Photo",
        }
    )
    assert n is not None
    assert n["shortCode"] == "AbCdXyZ"
    assert n["commentsCount"] == 12
    assert n["likesCount"] == 5
    assert n["ownerUsername"] == "seller_spb"
    assert n["type"] == "Image"
    assert n["searchKeyword"] == "недвижимостьспб"
    assert n["cookieSearchKeywords"] == ["недвижимостьспб"]


def test_normalize_skips_non_success() -> None:
    assert (
        normalize_keyword_search_item(
            {"status": "No posts found", "search_keyword": "x"}
        )
        is None
    )


def test_normalize_skips_bad_url() -> None:
    assert (
        normalize_keyword_search_item(
            {
                "status": "success",
                "post_url": "https://example.com/nope",
                "username": "u",
                "comment_count": 10,
            }
        )
        is None
    )


def test_normalize_reel_sets_video_url() -> None:
    mp4 = "https://scontent.cdninstagram.com/v/t50.2886-16/12345_n.mp4"
    n = normalize_keyword_search_item(
        {
            "status": "success",
            "post_url": "https://www.instagram.com/reel/ZzReelZ/",
            "username": "u",
            "comment_count": 20,
            "like_count": 1,
            "caption": "r",
            "pub_date": "2026-02-01T10:00:00.000Z",
            "search_keyword": "kw",
            "media_urls": [mp4],
        }
    )
    assert n is not None
    assert n["type"] == "Video"
    assert n.get("productType") == "clips"
    assert n.get("videoUrl") == mp4


def test_dedupe_merges_cookie_search_keywords() -> None:
    base = {
        "status": "success",
        "post_url": "https://www.instagram.com/p/SameCode/",
        "username": "u",
        "comment_count": 15,
        "like_count": 1,
        "pub_date": "2026-03-01T08:00:00.000Z",
        "media_urls": [],
    }
    a = normalize_keyword_search_item({**base, "search_keyword": "alpha"})
    b = normalize_keyword_search_item({**base, "search_keyword": "beta"})
    assert a and b
    out = dedupe_keyword_items_by_shortcode([a, b])
    assert len(out) == 1
    assert out[0]["cookieSearchKeywords"] == ["alpha", "beta"]
    assert "alpha" in out[0]["searchKeyword"] and "beta" in out[0]["searchKeyword"]


def test_filter_items_within_max_age_on_normalized() -> None:
    ref = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    young_iso = (ref - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    old_iso = (ref - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    y = normalize_keyword_search_item(
        {
            "status": "success",
            "post_url": "https://www.instagram.com/p/YoungPost/",
            "username": "u",
            "comment_count": 11,
            "like_count": 0,
            "pub_date": young_iso,
            "search_keyword": "a",
            "media_urls": [],
        }
    )
    o = normalize_keyword_search_item(
        {
            "status": "success",
            "post_url": "https://www.instagram.com/p/OldPostX/",
            "username": "u",
            "comment_count": 11,
            "like_count": 0,
            "pub_date": old_iso,
            "search_keyword": "b",
            "media_urls": [],
        }
    )
    assert y and o
    kept, stats = filter_items_within_max_age([y, o], 14, reference_time=ref)
    assert len(kept) == 1
    assert kept[0]["shortCode"] == "YoungPost"
    assert stats["dropped_too_old"] == 1
