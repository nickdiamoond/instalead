"""Shared helpers for Instagram Apify media payloads (Step 1 discovery + tests).

Used by ``scripts/pipeline.py`` and ``scripts/test_hashtag_step1.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse


def is_reel_payload(item: dict) -> bool:
    """Video / Reel-shaped Apify item (requires a playable video URL for the pipeline)."""
    return item.get("type") == "Video" or item.get("productType") == "clips"


def extract_video_url(item: dict) -> str | None:
    """Best-effort URL from hashtag-scraper / post-scraper shaped dicts."""
    raw = item.get("videoUrl")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    vid = item.get("video")
    if isinstance(vid, dict):
        u = vid.get("url")
        if isinstance(u, str) and u.strip():
            return u.strip()
    return None


def post_location_label_from_item(item: dict) -> str | None:
    """Human-readable geotag label from a Step 1 Apify-shaped post dict, if any.

    ``apify/instagram-hashtag-scraper`` and ``apify/instagram-post-scraper`` use
    ``locationName`` / ``locationId`` when the post has a place tag. Cookie keyword
    search rows are normalized to the same keys in
    ``src.instagram_cookie_search.normalize_keyword_search_item``.
    Many posts have no geotag — then this returns ``None``.
    """
    name_raw = item.get("locationName")
    name_s = name_raw.strip() if isinstance(name_raw, str) else ""
    lid_raw = item.get("locationId")
    lid_s = str(lid_raw).strip() if lid_raw is not None and str(lid_raw).strip() else ""

    if name_s and lid_s:
        return f"{name_s} (id {lid_s})"
    if name_s:
        return name_s
    if lid_s:
        return lid_s

    loc = item.get("location")
    if isinstance(loc, dict):
        n = loc.get("name")
        if isinstance(n, str) and n.strip():
            pk = loc.get("id") if loc.get("id") is not None else loc.get("pk")
            pk_s = str(pk).strip() if pk is not None and str(pk).strip() else ""
            if pk_s:
                return f"{n.strip()} (id {pk_s})"
            return n.strip()
    return None


def is_valid_video_url(url: str | None) -> bool:
    """HTTPS URL accepted for Nexara / CDN-style Instagram video links."""
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    parsed = urlparse(u)
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    allowed_suffixes = (
        "instagram.com",
        "cdninstagram.com",
        "fbcdn.net",
        "fb.watch",
    )
    return any(host == s or host.endswith("." + s) for s in allowed_suffixes)


def parse_item_timestamp_utc(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def filter_items_within_max_age(
    items: list[dict],
    max_age_days: int,
    *,
    reference_time: datetime | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Drop items older than ``max_age_days`` (UTC) using ``timestamp``.

    ``max_age_days <= 0`` disables filtering.
    Items without a parseable timestamp are kept.
    ``reference_time`` fixes "now" for tests (default: real UTC now).
    """
    if max_age_days <= 0:
        n = len(items)
        return list(items), {
            "fetched": n,
            "dropped_too_old": 0,
            "kept_missing_timestamp": 0,
        }

    ref = reference_time or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    else:
        ref = ref.astimezone(timezone.utc)
    cutoff = ref - timedelta(days=max_age_days)
    kept: list[dict] = []
    dropped = 0
    missing_ts = 0
    for p in items:
        ts = parse_item_timestamp_utc(p.get("timestamp"))
        if ts is None:
            missing_ts += 1
            kept.append(p)
            continue
        if ts >= cutoff:
            kept.append(p)
        else:
            dropped += 1
    return kept, {
        "fetched": len(items),
        "dropped_too_old": dropped,
        "kept_missing_timestamp": missing_ts,
    }


def merge_hashtag_items_by_shortcode(posts: list[dict], reels: list[dict]) -> list[dict]:
    """Dedupe posts vs reels runs by ``shortCode``, preferring a row with a valid video URL."""
    merged: dict[str, dict] = {}

    def url_score(d: dict) -> int:
        u = extract_video_url(d)
        return 1 if is_valid_video_url(u) else 0

    def pick(a: dict, b: dict) -> dict:
        sa, sb = url_score(a), url_score(b)
        if sb > sa:
            return b
        if sa > sb:
            return a
        if is_reel_payload(b) and not is_reel_payload(a):
            return b
        if is_reel_payload(a) and not is_reel_payload(b):
            return a
        return b

    for item in posts + reels:
        sc = (item.get("shortCode") or "").strip()
        if not sc:
            continue
        if sc not in merged:
            merged[sc] = item
        else:
            merged[sc] = pick(merged[sc], item)
    return list(merged.values())
