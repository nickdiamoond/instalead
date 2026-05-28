"""One-off Sherlock nick search for a hardcoded Telegram handle.

Mirrors pipeline Step 5 stage 1: ``POST /v1/search/nick`` with
``search_in=telegram``, poll ``GET /v1/tasks/{id}`` until terminal, then
``GET /v1/tasks/{id}/interactions``. Interprets the nick hit the same way as
:func:`pipeline_lib.step5_sherlock._resolve_one_lead_via_sherlock` (first
``result.results[]`` item with ``profile_url``). Prints the resolved Telegram
nick or ``not found`` to stderr; dumps full API payloads as JSON on stdout.

Usage:
    python scripts/test_sherlock_nick_hardcoded.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

import requests

from src.config import load_config
from src.sherlock_client import SherlockError, make_sherlock_client

# Hardcoded Telegram nick (with @ prefix, same as Step 5 ``nick_query``).
HARDCODED_NICK = "@marria_ro"


def _fallback_username(nick: str) -> str:
    return nick.lstrip("@")


def _nick_hit_from_task(task: dict, *, fallback_username: str) -> dict | None:
    """Same gate as Step 5: completed task + first result with ``profile_url``."""
    if (task.get("status") or "").lower() != "completed":
        return None
    results = ((task.get("result") or {}).get("results")) or []
    if isinstance(results, dict):
        results = [results]
    match = next(
        (
            item for item in results
            if isinstance(item, dict) and item.get("profile_url")
        ),
        None,
    )
    if not match:
        return None
    tg_username = match.get("username") or fallback_username
    return {
        "telegram_username": tg_username,
        "sherlock_link": match.get("link") or f"https://t.me/{tg_username}",
        "match": match,
    }


def _format_tg_nick(username: str) -> str:
    if username.startswith("@"):
        return username
    return f"@{username}"


def main() -> int:
    cfg = load_config()
    sh = cfg.get("sherlock") or {}
    nick_cfg = sh.get("nick_search") or {}
    task_cfg = sh.get("task") or {}

    max_pages = int(nick_cfg.get("max_pages", 1))
    max_attempts = int(nick_cfg.get("max_attempts", 3))
    poll_interval = float(task_cfg.get("poll_interval_secs", 3))
    max_wait = float(task_cfg.get("max_wait_secs", 300))
    fallback = _fallback_username(HARDCODED_NICK)

    print(f"Nick search: {HARDCODED_NICK!r}", file=sys.stderr, flush=True)

    sherlock = make_sherlock_client(cfg)
    out: dict = {"nick": HARDCODED_NICK}
    try:
        enq = sherlock.enqueue_nick(
            nick=HARDCODED_NICK,
            search_in="telegram",
            max_pages=max_pages,
            max_attempts=max_attempts,
        )
        out["enqueue"] = enq

        task = sherlock.wait_for_task(
            enq["id"],
            poll_interval=poll_interval,
            max_wait=max_wait,
        )
        out["task"] = task

        hit = _nick_hit_from_task(task, fallback_username=fallback)
        if hit:
            out["nick_hit"] = True
            out["telegram_username"] = hit["telegram_username"]
            out["sherlock_link"] = hit["sherlock_link"]
            print(_format_tg_nick(hit["telegram_username"]), file=sys.stderr, flush=True)
        else:
            out["nick_hit"] = False
            print("not found", file=sys.stderr, flush=True)

        try:
            out["interactions"] = sherlock.get_interactions(enq["id"])
        except SherlockError as exc:
            out["interactions"] = None
            out["interactions_error"] = str(exc)
    except (SherlockError, TimeoutError, requests.RequestException) as exc:
        out["error"] = str(exc)
        print("not found", file=sys.stderr, flush=True)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1
    finally:
        sherlock.close()

    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
