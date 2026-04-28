"""Normalize alternative comment-scraper output to the louisdeconinck shape.

Step 3 of :mod:`scripts.pipeline` was written against
``louisdeconinck/instagram-comments-scraper``, whose output mirrors
Instagram's raw snake_case schema (``user.full_name``, ``is_private``,
``created_at_utc``, ``media_id``, ...). When we fall back to a
different scraper that uses a different shape -- camelCase user block,
ISO 8601 timestamps, ``message`` instead of ``text``, etc. -- we run
the items through the matching ``normalize_*`` function below so the
rest of Step 3 doesn't have to branch on actor.

Currently supports:
    * ``apidojo/instagram-comments-scraper-api`` -- camelCase output
      with a clean ``postId`` (= shortcode), see
      https://apify.com/apidojo/instagram-comments-scraper-api/api.

Adding a new normalizer
-----------------------
The output dict must carry every field that Step 3 reads from a
louisdeconinck item: ``pk`` (comment id, used for in-memory dedup),
``media_id`` (used for the ``shortcode_to_id`` fuzzy match against
``processed_posts.shortcode``), ``text``, ``created_at_utc``, and a
nested ``user`` block with ``pk``, ``username``, ``full_name``,
``profile_pic_url``, ``is_private``, ``is_verified``. Anything else
the source actor returns is fine to drop -- downstream code only reads
these keys.
"""

from __future__ import annotations

from datetime import datetime


# Same charset as ``scripts/pipeline.py:CHARSET``. We duplicate it here
# so this module stays free of upward imports (pipeline.py is the
# orchestration script, not a library) -- keep both in sync.
_SHORTCODE_CHARSET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


def _shortcode_to_id(shortcode: str) -> int:
    """Decode an Instagram shortcode into the canonical numeric media id.

    Instagram exposes both representations: the URL-friendly base64ish
    shortcode (``DXdv7B1jFDF``) and the 64-bit numeric media id. The
    pipeline keys posts on shortcode but louisdeconinck-style comment
    items carry ``media_id``. We synthesize the matching int from
    apidojo's clean ``postId`` so the existing fuzzy-match logic in
    Step 3 finds the right post without losing precision (the
    ``+/- 1000`` tolerance on louisdeconinck media ids isn't even
    needed here -- this conversion is exact -- but keeping the same
    output shape means no Step 3 changes).
    """
    mid = 0
    for ch in shortcode:
        mid = mid * 64 + _SHORTCODE_CHARSET.index(ch)
    return mid


def _iso_to_unix(value):
    """Best-effort ISO 8601 -> unix int. Returns the input on failure.

    apidojo-api emits ``"2026-04-26T20:36:33.000Z"``. louisdeconinck
    emits the unix int directly. Step 3 stores the value as TEXT in
    ``lead_post_links.comment_at`` so a parse failure is non-fatal --
    we just preserve whatever we got and let the DB record it as-is.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return value
    text = value.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(text).timestamp())
    except (ValueError, TypeError):
        return value


def normalize_apidojo_api(item: dict) -> dict | None:
    """Convert one ``apidojo-api`` comment item to a louisdeconinck dict.

    Returns ``None`` if the item carries no recognizable username (a
    ranking marker, a deleted comment, a non-comment row produced by
    the actor, ...). Callers should drop ``None`` from the dedup pool.

    Output is shaped to mirror louisdeconinck so Step 3's existing
    dedup / save loop works unchanged. We additionally include a
    private ``_source_actor`` key for logging -- downstream code reads
    only the well-known fields and ignores anything extra.
    """
    user = item.get("user")
    if not isinstance(user, dict) or not user.get("username"):
        return None

    user_id = item.get("userId") or user.get("id")
    user_id_str = str(user_id) if user_id is not None else ""
    post_id = item.get("postId") or ""
    media_id = _shortcode_to_id(post_id) if post_id else None

    return {
        "pk": str(item.get("id") or ""),
        "user_id": user_id_str,
        "media_id": media_id,
        "post_shortcode": post_id,
        "text": item.get("message") or "",
        "created_at_utc": _iso_to_unix(item.get("createdAt")),
        "comment_like_count": item.get("likeCount"),
        "child_comment_count": item.get("replyCount"),
        "user": {
            "pk": user_id_str,
            "id": user_id_str,
            "username": user.get("username"),
            "full_name": user.get("fullName"),
            "is_private": user.get("isPrivate"),
            "is_verified": user.get("isVerified"),
            "profile_pic_url": user.get("profilePicUrl"),
        },
        "_source_actor": "apidojo-api",
    }
