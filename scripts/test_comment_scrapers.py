"""scripts/test_comment_scrapers.py

Side-by-side comparison test of Apify Instagram comment scrapers.

Background
----------
``louisdeconinck/instagram-comments-scraper`` was the cheapest reliable
scraper (~$0.50 / 1K) but recently started returning 0 items per URL
while still status=SUCCEEDED — see ``logs/pipeline_*`` and
``scripts/reset_failed_scans.py``. The official
``apify/instagram-comment-scraper`` works but is ~5x more expensive.
We need to find a cheap working alternative before re-running the
pipeline at scale.

What this script does
---------------------
Picks one Instagram post URL (CLI flag or the highest-comment relevant
post in the DB) and runs it through every actor in ``ACTORS`` below,
capturing for each:

    * dataset items_count
    * real ``usageTotalUsd`` from the Apify run detail
    * duration_ms
    * extractable / unique-username counts via a tolerant extractor
      that knows every known commenter field shape

Writes a single comparison JSON to ``logs/comment_scrapers_<ts>.json``
and prints a side-by-side table to stdout. Does NOT touch ``leads.db``
— purely a comparison harness.

Usage
-----
    # Interactive: pick a post automatically from DB, then prompt for
    # which actor(s) to run. Default cap is 50 comments per actor so a
    # full sweep on a 2k-comment post stays under ~$0.35.
    python scripts/test_comment_scrapers.py

    # Specific post
    python scripts/test_comment_scrapers.py --url https://www.instagram.com/p/DXZm2A1iDn_/
    python scripts/test_comment_scrapers.py --shortcode DXZm2A1iDn_

    # Skip the actor picker by passing a subset (names or 1-based numbers
    # accepted, comma-separated)
    python scripts/test_comment_scrapers.py --actors official,louisdeconinck
    python scripts/test_comment_scrapers.py --actors 1

    # Override the per-actor comment cap (passed into every actor's
    # input as ``resultsLimit``/``maxComments`` -- some actors honor
    # it, some quietly ignore; the report shows the actual count.)
    python scripts/test_comment_scrapers.py --limit 30

    # List configured actors and exit
    python scripts/test_comment_scrapers.py --list

    # Fully unattended: skip both the actor picker (= all actors) and
    # the cost-confirmation prompt
    python scripts/test_comment_scrapers.py --yes

Adding a new actor
------------------
Append an entry to ``ACTORS`` keyed by a short CLI name. The
``build_input`` callable receives ``(post_url, limit)`` and returns
the actor's ``run_input`` dict — different scrapers use different URL
input keys (``directUrls`` vs ``urls`` vs ``startUrls``). The output
extractor is shared, so if the new actor returns a novel item shape,
extend ``_extract_commenter`` rather than adding a per-adapter parser
— that keeps the comparison apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apify_client import ApifyClient
from dotenv import load_dotenv

from src.logger import get_logger, setup_logging

setup_logging()
log = get_logger("test_comment_scrapers")


# ============================================================================
# Adapters
# ============================================================================

@dataclass
class ActorAdapter:
    name: str
    actor_id: str
    notes: str
    build_input: Callable[[str, int], dict]


ACTORS: dict[str, ActorAdapter] = {
    "official": ActorAdapter(
        name="official",
        actor_id="apify/instagram-comment-scraper",
        notes=(
            "Official Apify scraper. Reliable. Item shape: top-level "
            "ownerUsername + nested owner{id, username, full_name, ...}. "
            "Pricing ~$0.0023/comment."
        ),
        build_input=lambda url, limit: {
            "directUrls": [url],
            "resultsLimit": limit,
        },
    ),
    "louisdeconinck": ActorAdapter(
        name="louisdeconinck",
        actor_id="louisdeconinck/instagram-comments-scraper",
        notes=(
            "Was the cheapest reliable scraper (~$0.50/1K). Recently "
            "started returning 0 items per URL with status=SUCCEEDED. "
            "Item shape: nested user{pk, username, full_name, ...} + "
            "top-level text/media_id/created_at_utc."
        ),
        # ``resultsLimit`` and ``maxComments`` are not officially documented
        # on this actor's input schema, but most Apify Instagram scrapers
        # respect at least one of them. We pass both defensively so the
        # actor caps itself instead of pulling all 2k+ comments on a busy
        # post; if it ignores them entirely we'll see that in the report.
        build_input=lambda url, limit: {
            "urls": [url],
            "resultsLimit": limit,
            "maxComments": limit,
        },
    ),
    "apidojo": ActorAdapter(
        name="apidojo",
        actor_id="apidojo/instagram-comments-scraper",
        notes=(
            "Alternative. CLAUDE.md notes severe pagination/dedup "
            "issues historically -- re-verify on a fresh URL. "
            "Input shape: startUrls (flat string array) + maxItems."
        ),
        # apidojo's schema is *different*: ``startUrls`` is a plain string
        # array (not the ``[{url: ...}]`` object form some actors use),
        # and the cap field is ``maxItems`` rather than ``resultsLimit``.
        # See https://apify.com/apidojo/instagram-comments-scraper/api.
        build_input=lambda url, limit: {
            "startUrls": [url],
            "maxItems": limit,
        },
    ),
    "apidojo-api": ActorAdapter(
        name="apidojo-api",
        actor_id="apidojo/instagram-comments-scraper-api",
        notes=(
            "Standby/realtime apidojo actor. Pricing: $0.0075/post "
            "(first 15 comments FREE) + $0.0005/comment. Item shape: "
            "camelCase nested user{fullName, isPrivate, isVerified, "
            "profilePicUrl} + top-level message/createdAt(ISO)/userId/"
            "postId. The postId field is the post shortcode directly, "
            "so media_id fuzzy matching is unnecessary on this output."
        ),
        # Same input shape as plain apidojo (startUrls + maxItems) per
        # https://apify.com/apidojo/instagram-comments-scraper-api/api.
        build_input=lambda url, limit: {
            "startUrls": [url],
            "maxItems": limit,
        },
    ),
}


def _safe_str(*candidates) -> str | None:
    """Return the first non-empty candidate stringified, else None.

    Avoids the ``str(None) == 'None'`` trap that breaks dedup keys.
    """
    for c in candidates:
        if c is None:
            continue
        s = str(c)
        if s and s != "None":
            return s
    return None


def _coalesce(*candidates):
    """Return the first non-None candidate, else None.

    Handy when merging snake_case and camelCase variants of the same
    field (e.g. ``is_private`` from louisdeconinck vs ``isPrivate``
    from apidojo-api) -- ``a or b`` would mis-handle a legitimate
    ``False`` value.
    """
    for c in candidates:
        if c is not None:
            return c
    return None


def _extract_commenter(item: dict) -> dict | None:
    """Tolerantly extract a commenter from a comment item.

    Tries every known field shape used by the actors in ``ACTORS``.
    Returns ``None`` if no recognizable username can be pulled out --
    the caller uses that to flag actors whose output we don't yet
    handle.
    """
    # louisdeconinck (snake_case) and apidojo-api (camelCase) both put
    # the commenter in a nested ``user`` object with ``username``. Merge
    # field reads from both shapes so this single branch covers both.
    user = item.get("user")
    if isinstance(user, dict) and user.get("username"):
        return {
            "username": user.get("username"),
            "user_id": _safe_str(
                user.get("pk"), user.get("id"), item.get("userId")
            ),
            "full_name": _coalesce(
                user.get("full_name"), user.get("fullName")
            ),
            "text": _coalesce(item.get("text"), item.get("message")),
            "timestamp": _coalesce(
                item.get("created_at_utc"),
                item.get("createdAt"),
                item.get("timestamp"),
            ),
            "is_private": _coalesce(
                user.get("is_private"), user.get("isPrivate")
            ),
            "is_verified": _coalesce(
                user.get("is_verified"), user.get("isVerified")
            ),
        }

    owner = item.get("owner") if isinstance(item.get("owner"), dict) else {}
    username = item.get("ownerUsername") or owner.get("username")
    if username:
        return {
            "username": username,
            "user_id": _safe_str(owner.get("id"), owner.get("pk")),
            "full_name": owner.get("full_name"),
            "text": item.get("text"),
            "timestamp": item.get("timestamp"),
            "is_private": owner.get("is_private"),
            "is_verified": owner.get("is_verified"),
        }

    fallback_username = (
        item.get("commenterUsername")
        or item.get("commenter_username")
        or item.get("username")
    )
    if fallback_username:
        return {
            "username": fallback_username,
            "user_id": _safe_str(
                item.get("commenterId"),
                item.get("commenter_id"),
                item.get("userId"),
                item.get("user_id"),
            ),
            "full_name": item.get("commenterFullName") or item.get("fullName"),
            "text": item.get("text") or item.get("commentText"),
            "timestamp": item.get("timestamp"),
            "is_private": None,
            "is_verified": None,
        }
    return None


# ============================================================================
# Post URL resolution
# ============================================================================

def resolve_post_url(args: argparse.Namespace, db_path: str) -> str:
    """Pick a single Instagram post URL to run every actor against.

    Priority: ``--url`` > ``--shortcode`` > DB query (highest-comment
    relevant + cta=comment post, preferring unscanned ones so we know
    the comments are still fresh).
    """
    if args.url:
        return args.url
    if args.shortcode:
        return f"https://www.instagram.com/p/{args.shortcode}/"

    if not Path(db_path).exists():
        raise SystemExit(
            f"DB not found at {db_path}. Pass --url or --shortcode explicitly."
        )

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT post_url, owner_username, comments_count, last_scanned_at "
        "FROM processed_posts "
        "WHERE relevance = 'relevant' AND cta_type = 'comment' "
        "  AND post_url IS NOT NULL AND post_url != '' "
        "ORDER BY (last_scanned_at IS NULL) DESC, comments_count DESC "
        "LIMIT 1"
    ).fetchall()
    con.close()

    if not rows:
        raise SystemExit(
            "No suitable post in DB. Pass --url or --shortcode explicitly."
        )

    row = rows[0]
    log.info(
        "auto_picked_post",
        url=row["post_url"],
        owner=row["owner_username"],
        comments=row["comments_count"],
        last_scanned=row["last_scanned_at"],
    )
    print(f"  Auto-picked from DB: @{row['owner_username']} "
          f"({row['comments_count']} comments, "
          f"scanned={'never' if not row['last_scanned_at'] else row['last_scanned_at']})")
    return row["post_url"]


# ============================================================================
# Per-actor probe
# ============================================================================

def probe_actor(
    client: ApifyClient,
    adapter: ActorAdapter,
    post_url: str,
    limit: int,
    timeout_secs: int,
) -> dict:
    run_input = adapter.build_input(post_url, limit)
    run_input.setdefault("proxy", {"useApifyProxy": True})

    log.info(
        "probe_start",
        actor=adapter.actor_id,
        input_keys=list(run_input.keys()),
        timeout_secs=timeout_secs,
        max_items=limit,
    )
    started = time.monotonic()

    try:
        # ``max_items`` propagates as ACTOR_MAX_PAID_DATASET_ITEMS to the
        # actor process and truncates the dataset / aborts the run early
        # -- a hard SDK-level cap that catches actors which silently
        # ignore our ``resultsLimit``/``maxItems`` input field.
        # ``timeout_secs`` aborts a run that's still going past the
        # budget (we've seen apify/instagram-comment-scraper hang for
        # 10+ min on busy posts; this gives back control instead of
        # waiting forever).
        run = client.actor(adapter.actor_id).call(
            run_input=run_input,
            max_items=limit,
            timeout_secs=timeout_secs,
        )
    except Exception as exc:
        elapsed = round((time.monotonic() - started) * 1000)
        log.error("probe_failed", actor=adapter.actor_id, error=str(exc))
        return {
            "name": adapter.name,
            "actor_id": adapter.actor_id,
            "input_params": run_input,
            "run_id": None,
            "status": "ERROR",
            "items_count": 0,
            "extractable_count": 0,
            "unhandled_items_count": 0,
            "unique_usernames": 0,
            "cost_usd": 0.0,
            "cost_per_unique": None,
            "duration_ms": elapsed,
            "samples": [],
            "unhandled_sample": None,
            "error": str(exc),
        }

    run_id = run["id"]
    status = run["status"]
    detail = client.run(run_id).get() or {}
    cost = detail.get("usageTotalUsd") or 0.0
    duration_ms = (
        detail.get("stats", {}).get("durationMillis")
        or round((time.monotonic() - started) * 1000)
    )

    items: list[dict] = []
    dataset_id = run.get("defaultDatasetId")
    if dataset_id:
        items = list(client.dataset(dataset_id).iterate_items())

    extractable: list[dict] = []
    unhandled_sample: dict | None = None
    unhandled_count = 0
    seen: set[str] = set()
    for it in items:
        c = _extract_commenter(it)
        if c is None:
            unhandled_count += 1
            if unhandled_sample is None:
                unhandled_sample = it
            continue
        extractable.append(c)
        key = c.get("user_id") or c.get("username") or ""
        if key:
            seen.add(key)

    log.info(
        "probe_done",
        actor=adapter.actor_id,
        run_id=run_id,
        status=status,
        items=len(items),
        extractable=len(extractable),
        unique=len(seen),
        cost=cost,
        duration_s=round(duration_ms / 1000, 1),
    )

    return {
        "name": adapter.name,
        "actor_id": adapter.actor_id,
        "input_params": run_input,
        "run_id": run_id,
        "status": status,
        "items_count": len(items),
        "extractable_count": len(extractable),
        "unhandled_items_count": unhandled_count,
        "unique_usernames": len(seen),
        "cost_usd": round(cost, 6),
        "cost_per_unique": (
            round(cost / len(seen), 6) if cost and seen else None
        ),
        "duration_ms": duration_ms,
        "samples": extractable[:5],
        "unhandled_sample": unhandled_sample,
        "error": None,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Side-by-side comparison of Apify Instagram comment scrapers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", help="Specific post URL to test")
    p.add_argument("--shortcode", help="Specific post shortcode (e.g. DXZm2A1iDn_)")
    p.add_argument(
        "--actors",
        default="",
        help="Comma-separated subset (default: all). See --list for names.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max comments per actor -- cost cap (default: 50)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help=(
            "Per-actor run timeout in seconds (default: 300). "
            "Aborts hung runs (e.g. apify/instagram-comment-scraper "
            "occasionally hangs 10+ min on busy posts)."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List configured actors and exit",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the cost-confirmation prompt",
    )
    p.add_argument(
        "--db",
        default="data/leads.db",
        help="Path to leads.db (used to auto-pick a post URL)",
    )
    return p.parse_args()


def list_actors() -> None:
    print(f"\nConfigured actors ({len(ACTORS)}):\n")
    name_w = max(len(a.name) for a in ACTORS.values())
    actor_w = max(len(a.actor_id) for a in ACTORS.values())
    for a in ACTORS.values():
        print(f"  {a.name:<{name_w}}  {a.actor_id:<{actor_w}}")
        for line in _wrap_notes(a.notes, indent=name_w + 4):
            print(line)
        print()


def _wrap_notes(text: str, *, indent: int, width: int = 90) -> list[str]:
    pad = " " * indent
    out: list[str] = []
    line = pad
    for word in text.split():
        if len(line) + len(word) + 1 > width and line.strip():
            out.append(line)
            line = pad
        line += (" " if line != pad else "") + word
    if line.strip():
        out.append(line)
    return out


def _resolve_actors_from_arg(raw: str) -> list[ActorAdapter]:
    """Parse a comma-separated string of actor names or 1-based numbers.

    Used for both the ``--actors`` CLI flag and the interactive picker so
    they accept the same syntax. Raises ``SystemExit`` on unknown tokens
    so a typo aborts before we burn money on a partial selection.
    """
    selected: list[ActorAdapter] = []
    seen: set[str] = set()
    actors_list = list(ACTORS.values())
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            continue
        if tok.isdigit():
            idx = int(tok)
            if not (1 <= idx <= len(actors_list)):
                raise SystemExit(
                    f"Invalid actor number: {tok} "
                    f"(valid range 1..{len(actors_list)})"
                )
            adapter = actors_list[idx - 1]
        elif tok in ACTORS:
            adapter = ACTORS[tok]
        else:
            raise SystemExit(
                f"Unknown actor: {tok!r}. Run with --list to see options."
            )
        if adapter.name not in seen:
            selected.append(adapter)
            seen.add(adapter.name)
    return selected


def _prompt_actor_choice() -> list[ActorAdapter]:
    """Show a numbered menu and read the user's pick from stdin.

    Empty input picks every actor (matches ``--actors`` not being set
    in non-interactive mode). Accepts numbers, names, or a comma-mix:
    ``1,3`` or ``official,apidojo`` or ``1,apidojo``.
    """
    actors_list = list(ACTORS.values())
    print()
    print("  Available actors:")
    name_w = max(len(a.name) for a in actors_list)
    for i, a in enumerate(actors_list, 1):
        print(f"    {i}. {a.name:<{name_w}}  {a.actor_id}")
    print()
    try:
        raw = input(
            "  Pick actor(s) by number or name (comma-separated, "
            "empty = all): "
        ).strip()
    except EOFError:
        raw = ""
    if not raw:
        return actors_list
    return _resolve_actors_from_arg(raw)


def main() -> int:
    args = parse_args()

    if args.list:
        list_actors()
        return 0

    load_dotenv()
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        raise SystemExit("APIFY_API_TOKEN not set in env (.env)")

    print()
    print("=" * 70)
    print("  Comment-scraper comparison")
    print("=" * 70)

    post_url = resolve_post_url(args, args.db)

    # Actor selection: explicit --actors wins. Otherwise prompt the
    # operator interactively (one actor at a time is usually what we
    # want when iterating on a broken scraper). ``--yes`` suppresses
    # both this prompt and the cost prompt -- pure unattended path =
    # all actors.
    if args.actors:
        selected = _resolve_actors_from_arg(args.actors)
    elif args.yes:
        selected = list(ACTORS.values())
    else:
        selected = _prompt_actor_choice()

    if not selected:
        print("No actors selected.", file=sys.stderr)
        return 2

    print(f"  Post URL:          {post_url}")
    print(f"  Limit per actor:   {args.limit}")
    print(f"  Timeout per actor: {args.timeout}s")
    print(f"  Actors ({len(selected)}):")
    for a in selected:
        print(f"    - {a.name:<16} {a.actor_id}")
    upper_bound = round(args.limit * 0.0023 * len(selected), 4)
    print(
        f"  Upper-bound cost:  ~${upper_bound:.4f} "
        f"(worst case: limit * $0.0023 per actor)"
    )
    print("=" * 70)

    if not args.yes:
        prompt = (
            "  Run this actor? (y/n): "
            if len(selected) == 1
            else f"  Run these {len(selected)} actors? (y/n): "
        )
        try:
            confirm = input(prompt).strip().lower()
        except EOFError:
            confirm = ""
        if confirm != "y":
            print("Cancelled.")
            return 0

    client = ApifyClient(token)

    results: list[dict] = []
    for adapter in selected:
        result = probe_actor(
            client, adapter, post_url, args.limit, args.timeout
        )
        results.append(result)
        status = (result["status"] or "").upper()
        if result["error"]:
            marker = "ERROR"
        elif status in {"TIMED-OUT", "TIMEOUT", "ABORTED"}:
            marker = status
        elif result["items_count"] == 0:
            marker = "EMPTY"
        elif result["extractable_count"] == 0:
            marker = "UNREADABLE"
        else:
            marker = "OK"
        print(
            f"  [{marker:>11}] {adapter.actor_id:<50} "
            f"items={result['items_count']:<5} "
            f"unique={result['unique_usernames']:<5} "
            f"cost=${result['cost_usd']:.4f}"
        )

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = log_dir / f"comment_scrapers_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "post_url": post_url,
                "limit_per_actor": args.limit,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print()
    print("=" * 110)
    print(
        f"  {'Actor':<46}  {'Items':>6}  {'Extr':>5}  {'Uniq':>5}  "
        f"{'Cost':>9}  {'$/uniq':>9}  {'Status':<14}"
    )
    print("-" * 110)
    for r in results:
        cost_str = f"${r['cost_usd']:.4f}"
        per_uniq = (
            f"${r['cost_per_unique']:.5f}"
            if r["cost_per_unique"] is not None
            else "-"
        )
        raw_status = (r["status"] or "").upper()
        if r["error"]:
            status_str = "ERROR"
        elif raw_status in {"TIMED-OUT", "TIMEOUT", "ABORTED", "FAILED"}:
            status_str = raw_status
        elif r["items_count"] == 0 and raw_status == "SUCCEEDED":
            status_str = "EMPTY"
        elif r["extractable_count"] == 0 and r["items_count"] > 0:
            status_str = "UNREADABLE"
        else:
            status_str = r["status"]
        print(
            f"  {r['actor_id']:<46}  "
            f"{r['items_count']:>6}  "
            f"{r['extractable_count']:>5}  "
            f"{r['unique_usernames']:>5}  "
            f"{cost_str:>9}  "
            f"{per_uniq:>9}  "
            f"{status_str:<14}"
        )
    print("=" * 110)
    print(f"  Report: {report_path}")

    unhandled = [r for r in results if r["unhandled_items_count"]]
    if unhandled:
        print()
        print("  NOTE: some actors returned items in an unrecognized shape — "
              "extend _extract_commenter() to handle them:")
        for r in unhandled:
            print(
                f"    - {r['actor_id']}: "
                f"{r['unhandled_items_count']} unhandled item(s). "
                f"Sample keys: {sorted((r['unhandled_sample'] or {}).keys())[:8]}"
            )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
