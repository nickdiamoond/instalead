"""Region catalog helpers.

A *region* (e.g. ``moscow`` / ``rostov``) bundles its own Step 1 discovery
sources (``realtor_accounts`` / ``hashtags`` / ``cookie_search_keywords``) and
the Telegram ``result_chat_id`` that Step 2 human confirmations and Step 5
lead results for that region are routed to.

Config layout (see ``config.yaml``)::

    region_definitions:
      moscow:
        result_chat_id: -100...
        realtor_accounts: [...]
        hashtags: [...]
        cookie_search_keywords: [...]
      rostov:
        ...

    pipeline:
      regions:          # active selector for the run (str or list)
        - moscow
        - rostov

This module lives in ``src/`` so both ``src.telegram_notifier`` and the
pipeline scripts can import it without a circular dependency.
"""

from __future__ import annotations

from typing import Any

# Step 1 discovery mode -> the per-region ``region_definitions`` key that
# supplies that mode's source content.
_MODE_SOURCE_KEY = {
    "realtors": "realtor_accounts",
    "hashtags": "hashtags",
    "cookie_keywords": "cookie_search_keywords",
}


def _region_definitions(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("region_definitions")
    return raw if isinstance(raw, dict) else {}


def parse_active_regions(cfg: dict[str, Any]) -> list[str]:
    """Normalize ``pipeline.regions`` to an ordered, unique list of names.

    Accepts a string or a list (mirrors ``parse_step1_discovery_modes``).
    Names are lower-cased and stripped; empties and duplicates are dropped
    with first-seen order preserved. Returns ``[]`` when nothing is
    configured (caller decides how to handle "no regions").
    """
    raw = (cfg.get("pipeline") or {}).get("regions")
    if raw is None:
        items: list[object] = []
    elif isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [raw]

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = str(item).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _string_list(raw: object) -> list[str]:
    """Stripped, de-duplicated (order-preserving) list of non-empty strings."""
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if s:
            out.append(s)
    return list(dict.fromkeys(out))


def region_realtor_accounts(cfg: dict[str, Any], region: str) -> list[str]:
    """Instagram usernames for ``discovery_mode=realtors`` in ``region``."""
    defn = _region_definitions(cfg).get(region) or {}
    return _string_list(defn.get("realtor_accounts"))


def region_hashtags(cfg: dict[str, Any], region: str) -> list[str]:
    """Hashtags for ``discovery_mode=hashtags`` in ``region``."""
    defn = _region_definitions(cfg).get(region) or {}
    return _string_list(defn.get("hashtags"))


def region_cookie_keywords(cfg: dict[str, Any], region: str) -> list[str]:
    """Keyword-search terms for ``discovery_mode=cookie_keywords`` in ``region``."""
    defn = _region_definitions(cfg).get(region) or {}
    return _string_list(defn.get("cookie_search_keywords"))


def region_sources(cfg: dict[str, Any], region: str) -> dict[str, list[str]]:
    """All three source lists for ``region`` keyed by discovery mode."""
    return {
        "realtors": region_realtor_accounts(cfg, region),
        "hashtags": region_hashtags(cfg, region),
        "cookie_keywords": region_cookie_keywords(cfg, region),
    }


def region_sources_for_mode(
    cfg: dict[str, Any], region: str, discovery_mode: str
) -> list[str]:
    """Source list for a single ``(region, discovery_mode)`` pair."""
    key = _MODE_SOURCE_KEY.get(discovery_mode)
    if key is None:
        return []
    defn = _region_definitions(cfg).get(region) or {}
    return _string_list(defn.get(key))


def region_result_chat_id(
    cfg: dict[str, Any], region: str | None
) -> int | None:
    """Result chat id for ``region`` from ``region_definitions``.

    Returns ``region_definitions[region].result_chat_id`` when present and
    parseable as an int, else ``None``. There is no global result chat: an
    unknown region, a region without ``result_chat_id``, or ``region=None``
    (legacy rows) all return ``None``, and the caller decides where to route
    (the pipeline falls back to ``telegram.report_chat_id``).
    """
    if region:
        defn = _region_definitions(cfg).get(region)
        if isinstance(defn, dict):
            raw = defn.get("result_chat_id")
            if raw is not None:
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    pass
    return None
