"""Smoke test for Apify ``crawlerbros/instagram-keyword-search-scraper``.

Runs one actor call with all keywords from ``config.yaml`` → ``search.cookie_search_keywords``.
Requires ``APIFY_API_TOKEN`` and Instagram cookies in the env var named by
``search.cookie_search.session_cookie_env_var`` (default ``INSTAGRAM_SESSION_COOKIE``).

Cookies may be either a browser JSON export (array of cookie objects) or a simple
``sessionid=...`` / ``a=b; c=d`` header-style string; the latter is wrapped into the JSON
shape the actor expects.

Paste the value into ``.env`` only (no cookie files are read by this script). Cookie-Editor
exports may contain ``rur`` with backslashes that break ``json.loads`` after copy/paste; in
that case the script **rebuilds the top-level JSON array** by walking braces (string-aware)
and dropping objects whose ``name`` is ``rur`` / ``wd`` / ``dpr``, then parses again. After
parse, the same names are filtered again and only ``*.instagram.com`` cookies are kept.

This does **not** touch the daily pipeline; it only prints raw dataset items for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv

from src.config import load_config


def _truncate(text: str | None, max_len: int = 200) -> str:
    if not text:
        return "—"
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _shortcode_from_post_url(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    m = re.search(r"instagram\.com/(?:p|reel|reels|tv)/([^/?#]+)", url, re.IGNORECASE)
    return m.group(1) if m else None


# Ephemeral / layout cookies; ``rur`` values embed backslashes that break JSON when pasted into .env.
_SKIP_BROWSER_COOKIE_NAMES = frozenset({"rur", "wd", "dpr"})
_NAME_FIELD_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _json_object_end_exclusive(s: str, open_brace: int) -> int | None:
    """Index after the closing ``}`` of the object that starts at ``open_brace``, or ``None``."""
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
    """Drop top-level array objects whose ``name`` is in ``_SKIP_BROWSER_COOKIE_NAMES`` without ``json.loads``."""
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


def _sanitize_browser_cookie_export(items: list) -> list[dict]:
    """Keep instagram.com cookies only; drop fragile names; minimal fields for the actor."""
    out: list[dict] = []
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
        entry: dict = {
            "name": name,
            "value": it.get("value", "") if isinstance(it.get("value"), str) else str(it.get("value", "")),
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
        raise ValueError("no Instagram cookies after filtering export (check domain / login export)")
    return out


def _cookies_json_string_for_actor(raw: str) -> str:
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


def _print_post_block(*, row_num: int, success_index: int | None, item: dict) -> None:
    status = item.get("status")
    sk = item.get("search_keyword")
    if status == "No posts found":
        print("  " + "-" * 72)
        print(f"  Row {row_num} (no posts)  |  search keyword: {sk!r}  status: {status!r}")
        return

    post_url = item.get("post_url") if isinstance(item.get("post_url"), str) else None
    code = _shortcode_from_post_url(post_url)
    media_urls = item.get("media_urls") if isinstance(item.get("media_urls"), list) else []

    print("  " + "-" * 72)
    label = f"success #{success_index}" if success_index is not None else f"row {row_num}"
    print(f"  Post ({label})  |  search keyword: {sk!r}")
    print(f"    shortcode (parsed): {code!r}")
    print(f"    post_url:          {post_url!r}")
    print(f"    username:          {item.get('username')!r}")
    print(f"    full_name:         {item.get('full_name')!r}")
    print(f"    profile_url:       {item.get('profile_url')!r}")
    print(f"    pub_date:          {item.get('pub_date')!r}")
    print(f"    caption:           {_truncate(item.get('caption') if isinstance(item.get('caption'), str) else None, 400)}")
    print(f"    media_type:        {item.get('media_type')!r}")
    print(f"    media_count:       {item.get('media_count')!r}")
    print(f"    comment_count:     {item.get('comment_count')!r}")
    print(f"    like_count:        {item.get('like_count')!r}")
    print(f"    likes_hidden:      {item.get('likes_hidden')!r}")
    print(f"    thumbnail_url:     {_truncate(item.get('thumbnail_url') if isinstance(item.get('thumbnail_url'), str) else None, 120)}")
    if media_urls:
        print(f"    media_urls ({len(media_urls)}):")
        for j, u in enumerate(media_urls[:5]):
            print(f"      [{j}] {_truncate(str(u), 120)}")
        if len(media_urls) > 5:
            print(f"      … +{len(media_urls) - 5} more")
    else:
        print("    media_urls:        —")

    mentions = item.get("mentions")
    hashtags = item.get("hashtags")
    if isinstance(mentions, list) and mentions:
        print(f"    mentions:          {mentions!r}")
    if isinstance(hashtags, list) and hashtags:
        print(f"    hashtags:          {hashtags!r}")

    collaborators = item.get("collaborators")
    if isinstance(collaborators, list) and collaborators:
        print(f"    collaborators:     {collaborators!r}")

    loc = item.get("location") if isinstance(item.get("location"), dict) else {}
    if loc:
        print(f"    location:          {loc!r}")

    music = item.get("music") if isinstance(item.get("music"), dict) else {}
    if music:
        print(f"    music:             {music!r}")

    print(f"    status:            {item.get('status')!r}")
    print(f"    scraped_at:        {item.get('scraped_at')!r}")

    extra_keys = sorted(
        k
        for k in item.keys()
        if k
        not in {
            "search_keyword",
            "post_url",
            "username",
            "full_name",
            "profile_url",
            "pub_date",
            "caption",
            "media_type",
            "media_count",
            "comment_count",
            "like_count",
            "likes_hidden",
            "thumbnail_url",
            "media_urls",
            "mentions",
            "hashtags",
            "collaborators",
            "location",
            "music",
            "status",
            "scraped_at",
        }
    )
    if extra_keys:
        print("    other top-level keys:", ", ".join(extra_keys))
        for ek in extra_keys:
            val = item[ek]
            if isinstance(val, (dict, list)):
                snippet = json.dumps(val, ensure_ascii=False, default=str)[:300]
                if len(snippet) >= 300:
                    snippet += "…"
                print(f"      {ek}: {snippet}")
            else:
                print(f"      {ek}: {val!r}")


def main() -> int:
    load_dotenv()
    cfg = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keyword",
        help="Run a single keyword instead of every entry in search.cookie_search_keywords.",
    )
    args = parser.parse_args()

    actors = (cfg.get("apify") or {}).get("actors") or {}
    actor_id = actors.get("cookie_search_posts", "crawlerbros/instagram-keyword-search-scraper")

    search = cfg.get("search") or {}
    cs_cfg = search.get("cookie_search") or {}
    max_posts = int(cs_cfg.get("size_per_keyword", 5))
    cookie_var = str(cs_cfg.get("session_cookie_env_var", "INSTAGRAM_SESSION_COOKIE"))
    session_name = str(cs_cfg.get("session_name", "instalead_cookie_search"))

    if args.keyword:
        keywords = [args.keyword.strip()]
    else:
        keywords = list(search.get("cookie_search_keywords") or [])

    if not keywords:
        print("ERROR: no keywords — set search.cookie_search_keywords in config.yaml or pass --keyword.")
        return 1

    cookies_raw = (os.environ.get(cookie_var) or "").strip()
    if not cookies_raw:
        print(
            f"ERROR: {cookie_var} is empty or unset. "
            f"Paste Instagram cookies into .env (see .env.example)."
        )
        return 1

    try:
        cookies_payload = _cookies_json_string_for_actor(cookies_raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: could not normalize cookies for the actor: {e}")
        return 1

    token = (cfg.get("apify") or {}).get("token")
    if not token:
        print("ERROR: Apify token missing after load_config().")
        return 1

    client = ApifyClient(token)

    print("=" * 76)
    print("Cookie keyword search (Apify)")
    print(f"  Actor:               {actor_id}")
    print(f"  maxPosts per keyword: {max_posts}")
    print(f"  Keywords ({len(keywords)}): {keywords}")
    print(f"  Cookie env var:      {cookie_var} (value hidden, raw len={len(cookies_raw)})")
    print(f"  sessionName:         {session_name!r}")
    print("=" * 76)

    print()
    print(f">>> Single run with {len(keywords)} keyword(s)")
    run = client.actor(actor_id).call(
        run_input={
            "keywords": keywords,
            "maxPosts": max_posts,
            "cookies": cookies_payload,
            "sessionName": session_name,
        }
    )
    run_id = run.get("id")
    detail = client.run(run_id).get() if run_id else {}
    cost = detail.get("usageTotalUsd")
    status = run.get("status") or detail.get("status")
    print(
        f"    Run id: {run_id!r}  status: {status!r}  "
        f"usageTotalUsd: {cost!r}  dataset: {run.get('defaultDatasetId')!r}"
    )

    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    n = len(items)
    success_n = sum(1 for it in items if isinstance(it, dict) and it.get("status") == "success")
    empty_n = sum(1 for it in items if isinstance(it, dict) and it.get("status") == "No posts found")

    print(f"    Items in dataset: {n}  (status=success: {success_n}, no posts markers: {empty_n})")

    if n == 0:
        print("    (empty dataset — check cookie validity, keywords, or actor limits.)")
        print()
        print("=" * 76)
        print("DONE. Total post rows printed: 0")
        print("=" * 76)
        return 0

    success_i = 0
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            print(f"  Row #{i + 1}: non-dict item: {item!r}")
            continue
        row_num = i + 1
        if item.get("status") == "success":
            success_i += 1
            _print_post_block(row_num=row_num, success_index=success_i, item=item)
        else:
            _print_post_block(row_num=row_num, success_index=None, item=item)

    print()
    print("=" * 76)
    print(f"DONE. Total success posts printed: {success_i} (dataset rows: {n})")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
