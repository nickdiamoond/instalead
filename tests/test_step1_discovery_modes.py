"""Tests for Step 1 multi discovery_mode parsing."""

from scripts.pipeline_lib.step1_discovery import (
    build_step1_searched_summary,
    format_step1_discovery_modes_label,
    parse_step1_discovery_modes,
)


def test_parse_discovery_mode_string() -> None:
    assert parse_step1_discovery_modes("realtors") == ["realtors"]
    assert parse_step1_discovery_modes("  Hashtags ") == ["hashtags"]


def test_parse_discovery_mode_list_dedupes_order() -> None:
    assert parse_step1_discovery_modes(
        ["realtors", "hashtags", "realtors", "cookie_keywords"]
    ) == ["realtors", "hashtags", "cookie_keywords"]


def test_parse_discovery_mode_none_uses_default() -> None:
    assert parse_step1_discovery_modes(None) == ["realtors"]


def test_format_label_and_searched_summary() -> None:
    assert format_step1_discovery_modes_label(["realtors", "hashtags"]) == (
        "realtors + hashtags"
    )
    assert build_step1_searched_summary({"realtors": 3, "hashtags": 2}) == (
        "Searched 3 realtor(s), 2 hashtag(s)."
    )
