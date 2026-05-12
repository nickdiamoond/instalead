"""Instagram session cookies + keyword-search actor payload normalization.

Used by ``scripts/test_cookie_keyword_search.py`` and Step 1 when
``pipeline.step1.discovery_mode`` is ``cookie_keywords`` (Apify
``crawlerbros/instagram-keyword-search-scraper``).
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.ig_media_payload import is_valid_video_url, parse_item_timestamp_utc

# Ephemeral / layout cookies; ``rur`` values embed backslashes that break JSON when pasted into .env.
_SKIP_BROWSER_COOKIE_NAMES = frozenset({"rur", "wd", "dpr"})
_NAME_FIELD_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')

_SHORTCODE_FROM_URL_RE = re.compile(
    r"instagram\.com/(?:p|reel|reels|tv)/([^/?#]+)", re.IGNORECASE
)


def shortcode_from_post_url(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    m = _SHORTCODE_FROM_URL_RE.search(url)
    return m.group(1) if m else None


def _json_object_end_exclusive(s: str, open_brace: int) -> int | None:
    if open_brace < 0 or open_brace >= len(s) or s[open_brace] != "{":
        return None
    depth = 0
    i = open_brace
    in_str = False
    esc = False
    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _rewrite_cookie_json_array_without_skip_names(raw: str) -> str:
    s = raw.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return s
    kept: list[str] = []
    i = 1
    n = len(s)
    while i < n:
        while i < n and s[i] in " \t\n\r,":
            i += 1
        if i >= n or s[i] == "]":
            break
        if s[i] != "{":
            return s
        start = i
        end = _json_object_end_exclusive(s, start)
        if end is None:
            return s
        blob = s[start:end]
        m = _NAME_FIELD_RE.search(blob)
        nm = (m.group(1) or "").lower() if m else ""
        if nm not in _SKIP_BROWSER_COOKIE_NAMES:
            kept.append(blob)
        i = end
    return "[" + ",".join(kept) + "]"


def _sanitize_browser_cookie_export(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        if not isinstance(name, str):
            continue
        if name.lower() in _SKIP_BROWSER_COOKIE_NAMES:
            continue
        domain = it.get("domain", "")
        if not isinstance(domain, str) or "instagram" not in domain.lower():
            continue
        entry: dict[str, Any] = {
            "name": name,
            "value": it.get("value", "")
            if isinstance(it.get("value"), str)
            else str(it.get("value", "")),
            "domain": domain,
            "path": it.get("path", "/") if isinstance(it.get("path"), str) else "/",
            "secure": bool(it.get("secure", True)),
            "httpOnly": bool(it.get("httpOnly", False)),
        }
        ss = it.get("sameSite")
        if isinstance(ss, str) and ss:
            entry["sameSite"] = ss
        out.append(entry)
    if not out:
        raise ValueError(
            "no Instagram cookies after filtering export (check domain / login export)"
        )
    return out


def cookies_json_string_for_actor(raw: str) -> str:
    """Return a JSON array string suitable for the actor's ``cookies`` input field."""
    s = raw.strip()
    if s.startswith("["):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            s2 = _rewrite_cookie_json_array_without_skip_names(s)
            parsed = json.loads(s2)
        if not isinstance(parsed, list):
            raise ValueError("cookies JSON must be an array of cookie objects")
        cleaned = _sanitize_browser_cookie_export(parsed)
        return json.dumps(cleaned, separators=(",", ":"))

    pairs: list[tuple[str, str]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        name, value = name.strip(), value.strip()
        if name:
            pairs.append((name, value))

    if not pairs and "=" in s:
        name, value = s.split("=", 1)
        pairs.append((name.strip(), value.strip()))

    if not pairs:
        raise ValueError("no name=value cookie pairs found")

    blob = [
        {
            "name": name,
            "value": value,
            "domain": ".instagram.com",
            "path": "/",
            "secure": True,
            "httpOnly": name.lower() == "sessionid",
        }
        for name, value in pairs
    ]
    return json.dumps(blob, separators=(",", ":"))


def _first_valid_video_url(media_urls: list[Any]) -> str | None:
    for u in media_urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if is_valid_video_url(s):
            return s
    return None


def _is_reel_post_url(url: str) -> bool:
    u = url.lower()
    return "/reel/" in u or "/reels/" in u or "/tv/" in u


def _is_reel_media_type(media_type: Any) -> bool:
    if not isinstance(media_type, str):
        return False
    return "reel" in media_type.lower()


def normalize_keyword_search_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Map crawlerbros keyword-search row to hashtag/post-scraper-shaped dict for Step 1.

    Returns ``None`` for non-success rows or missing shortcode.
    Adds Telegram-oriented extras: ``searchKeyword``, ``cookieSearchKeywords``,
    ``cookieMediaType``, ``cookieMediaUrlsPreview``, ``captionPreview``,
    ``cookieMentions``.
    """
    if item.get("status") != "success":
        return None

    post_url = item.get("post_url")
    if not isinstance(post_url, str) or not post_url.strip():
        return None
    post_url = post_url.strip()

    shortcode = shortcode_from_post_url(post_url)
    if not shortcode:
        return None

    sk_raw = item.get("search_keyword")
    search_kw = sk_raw.strip() if isinstance(sk_raw, str) else ""

    comments = item.get("comment_count")
    try:
        comments_count = int(comments) if comments is not None else 0
    except (TypeError, ValueError):
        comments_count = 0

    likes = item.get("like_count")
    try:
        likes_count = int(likes) if likes is not None else 0
    except (TypeError, ValueError):
        likes_count = 0

    username = item.get("username")
    owner = username.strip() if isinstance(username, str) else None

    cap = item.get("caption")
    caption = cap if isinstance(cap, str) else None

    pub = item.get("pub_date")
    if pub is None:
        timestamp: str | int | float | None = None
    elif isinstance(pub, (int, float, str)):
        timestamp = pub
    else:
        timestamp = str(pub)

    media_urls = item.get("media_urls") if isinstance(item.get("media_urls"), list) else []
    video_url = _first_valid_video_url(media_urls)

    media_type = item.get("media_type")
    is_reel = _is_reel_post_url(post_url) or _is_reel_media_type(media_type)

    out: dict[str, Any] = {
        "shortCode": shortcode,
        "url": post_url,
        "commentsCount": comments_count,
        "ownerUsername": owner,
        "likesCount": likes_count,
        "videoViewCount": 0,
        "caption": caption,
        "timestamp": timestamp,
        "hashtags": item.get("hashtags") if isinstance(item.get("hashtags"), list) else [],
    }

    if is_reel:
        out["type"] = "Video"
        out["productType"] = "clips"
        if video_url:
            out["videoUrl"] = video_url
    else:
        out["type"] = "Image"

    loc = item.get("location")
    if isinstance(loc, dict):
        name = loc.get("name")
        if isinstance(name, str) and name.strip():
            out["locationName"] = name.strip()
        lid = loc.get("id") or loc.get("pk")
        if lid is not None and str(lid).strip():
            out["locationId"] = str(lid).strip()

    # Extras for Telegram / debugging (do not rely on these in DB upsert path).
    out["searchKeyword"] = search_kw
    out["cookieSearchKeywords"] = [search_kw] if search_kw else []
    if isinstance(media_type, str) and media_type.strip():
        out["cookieMediaType"] = media_type.strip()
    preview: list[str] = []
    for u in media_urls[:5]:
        if isinstance(u, str) and u.strip():
            preview.append(u.strip()[:200])
    if preview:
        out["cookieMediaUrlsPreview"] = preview
    if caption:
        cap_one = caption.replace("\n", " ").strip()
        out["captionPreview"] = cap_one[:400] + ("…" if len(cap_one) > 400 else "")
    mentions = item.get("mentions")
    if isinstance(mentions, list) and mentions:
        out["cookieMentions"] = [str(m) for m in mentions if str(m).strip()]

    return out


def dedupe_keyword_items_by_shortcode(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per ``shortCode``; merge ``cookieSearchKeywords`` (sorted unique)."""
    merged: dict[str, dict[str, Any]] = {}
    for p in items:
        sc = (p.get("shortCode") or "").strip()
        if not sc:
            continue
        if sc not in merged:
            merged[sc] = p
            continue
        a = merged[sc]
        keys_a = a.get("cookieSearchKeywords")
        keys_b = p.get("cookieSearchKeywords")
        la = list(keys_a) if isinstance(keys_a, list) else []
        lb = list(keys_b) if isinstance(keys_b, list) else []
        combined = sorted({x for x in (la + lb) if isinstance(x, str) and x.strip()})
        a["cookieSearchKeywords"] = combined
        if combined:
            a["searchKeyword"] = ", ".join(combined)
    return list(merged.values())
