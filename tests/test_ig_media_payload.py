"""Unit tests for ``src.ig_media_payload``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.ig_media_payload import (
    extract_video_url,
    filter_items_within_max_age,
    is_reel_payload,
    is_valid_video_url,
    merge_hashtag_items_by_shortcode,
    post_location_label_from_item,
)


def test_is_reel_payload() -> None:
    assert is_reel_payload({"type": "Video", "productType": "clips"}) is True
    assert is_reel_payload({"type": "Video", "productType": "feed"}) is True
    assert is_reel_payload({"type": "Image"}) is False


def test_post_location_label_from_apify_keys() -> None:
    assert post_location_label_from_item({}) is None
    assert (
        post_location_label_from_item(
            {"locationName": "Санкт-Петербург", "locationId": "99"}
        )
        == "Санкт-Петербург (id 99)"
    )
    assert post_location_label_from_item({"locationName": " SPB "}) == "SPB"


def test_post_location_label_from_nested_location_dict() -> None:
    assert (
        post_location_label_from_item(
            {"location": {"name": "Moscow", "pk": "123"}}
        )
        == "Moscow (id 123)"
    )


@pytest.mark.parametrize(
    "url,ok",
    [
        ("https://scontent.cdninstagram.com/v/abc", True),
        ("https://video.fbcdn.net/x", True),
        ("https://www.instagram.com/x", True),
        ("http://cdninstagram.com/x", False),
        ("https://evil.com/cdninstagram.com", False),
        ("", False),
        (None, False),
    ],
)
def test_is_valid_video_url(url: str | None, ok: bool) -> None:
    assert is_valid_video_url(url) is ok


def test_extract_video_url_nested() -> None:
    assert extract_video_url({"videoUrl": " https://cdninstagram.com/a "}) == (
        "https://cdninstagram.com/a"
    )
    assert extract_video_url({"video": {"url": "https://fbcdn.net/b"}}) == "https://fbcdn.net/b"
    assert extract_video_url({"caption": "x"}) is None


def test_merge_prefers_valid_video_url() -> None:
    posts = [
        {"shortCode": "AAA", "type": "Video", "videoUrl": "https://cdninstagram.com/old"},
    ]
    reels = [
        {"shortCode": "AAA", "type": "Video", "productType": "clips"},
    ]
    out = merge_hashtag_items_by_shortcode(posts, reels)
    assert len(out) == 1
    assert extract_video_url(out[0]) == "https://cdninstagram.com/old"


def test_merge_reel_wins_tie_with_valid_urls() -> None:
    a = {"shortCode": "Z", "type": "Image"}
    b = {"shortCode": "Z", "type": "Video", "productType": "clips", "videoUrl": "https://fbcdn.net/v"}
    out = merge_hashtag_items_by_shortcode([a], [b])
    assert is_reel_payload(out[0])


def test_filter_age_drops_old() -> None:
    ref = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    items = [
        {"shortCode": "a", "timestamp": "2026-05-09T12:00:00.000Z"},
        {"shortCode": "b", "timestamp": "2026-05-01T12:00:00.000Z"},
    ]
    kept, stats = filter_items_within_max_age(items, 7, reference_time=ref)
    assert [x["shortCode"] for x in kept] == ["a"]
    assert stats["dropped_too_old"] == 1


def test_filter_age_disabled() -> None:
    items = [{"shortCode": "b", "timestamp": "2000-01-01T00:00:00.000Z"}]
    kept, stats = filter_items_within_max_age(items, 0, reference_time=datetime.now(timezone.utc))
    assert len(kept) == 1
    assert stats["dropped_too_old"] == 0
