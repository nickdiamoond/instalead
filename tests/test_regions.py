"""Unit tests for src.regions (region catalog helpers)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.regions import (
    parse_active_regions,
    region_cookie_keywords,
    region_hashtags,
    region_realtor_accounts,
    region_result_chat_id,
    region_sources,
    region_sources_for_mode,
)


def _cfg() -> dict:
    return {
        "telegram": {"report_chat_id": -100},
        "pipeline": {"regions": ["moscow", "rostov"]},
        "region_definitions": {
            "moscow": {
                "result_chat_id": -111,
                "realtor_accounts": ["pik", "pik", "  ", "lsr.petersburg"],
                "hashtags": ["квартираспб"],
                "cookie_search_keywords": ["новостройки"],
            },
            "rostov": {
                # no result_chat_id -> region_result_chat_id returns None
                "realtor_accounts": ["msk_rostov"],
                "hashtags": [],
                "cookie_search_keywords": [],
            },
        },
    }


def test_parse_active_regions_list() -> None:
    assert parse_active_regions(_cfg()) == ["moscow", "rostov"]


def test_parse_active_regions_string() -> None:
    cfg = {"pipeline": {"regions": "Moscow"}}
    assert parse_active_regions(cfg) == ["moscow"]


def test_parse_active_regions_dedup_and_strip() -> None:
    cfg = {"pipeline": {"regions": ["moscow", " moscow ", "Rostov", ""]}}
    assert parse_active_regions(cfg) == ["moscow", "rostov"]


def test_parse_active_regions_empty() -> None:
    assert parse_active_regions({}) == []
    assert parse_active_regions({"pipeline": {}}) == []


def test_region_realtor_accounts_cleans_list() -> None:
    # duplicates removed (order preserved), blanks dropped
    assert region_realtor_accounts(_cfg(), "moscow") == ["pik", "lsr.petersburg"]


def test_region_hashtags_and_keywords() -> None:
    cfg = _cfg()
    assert region_hashtags(cfg, "moscow") == ["квартираспб"]
    assert region_cookie_keywords(cfg, "moscow") == ["новостройки"]
    assert region_hashtags(cfg, "rostov") == []
    assert region_cookie_keywords(cfg, "rostov") == []


def test_region_sources_bundle() -> None:
    src = region_sources(_cfg(), "rostov")
    assert src == {
        "realtors": ["msk_rostov"],
        "hashtags": [],
        "cookie_keywords": [],
    }


def test_region_sources_for_mode() -> None:
    cfg = _cfg()
    assert region_sources_for_mode(cfg, "moscow", "realtors") == [
        "pik",
        "lsr.petersburg",
    ]
    assert region_sources_for_mode(cfg, "moscow", "hashtags") == ["квартираспб"]
    assert region_sources_for_mode(cfg, "moscow", "cookie_keywords") == [
        "новостройки"
    ]
    assert region_sources_for_mode(cfg, "moscow", "bogus") == []


def test_region_result_chat_id_uses_region_value() -> None:
    assert region_result_chat_id(_cfg(), "moscow") == -111


def test_region_result_chat_id_missing_returns_none() -> None:
    # rostov has no result_chat_id of its own; no global fallback exists
    assert region_result_chat_id(_cfg(), "rostov") is None


def test_region_result_chat_id_none_region_returns_none() -> None:
    assert region_result_chat_id(_cfg(), None) is None


def test_region_result_chat_id_unknown_region_returns_none() -> None:
    assert region_result_chat_id(_cfg(), "spb") is None


def test_region_result_chat_id_invalid_region_value_returns_none() -> None:
    cfg = {"region_definitions": {"moscow": {"result_chat_id": "not-an-int"}}}
    assert region_result_chat_id(cfg, "moscow") is None


def test_region_result_chat_id_no_definition_returns_none() -> None:
    cfg = {"region_definitions": {"moscow": {}}}
    assert region_result_chat_id(cfg, "moscow") is None
