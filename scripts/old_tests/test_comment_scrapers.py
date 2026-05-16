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

    # Multiple posts (test batching behavior)
    python scripts/test_comment_scrapers.py --urls https://www.instagram.com/p/A/,https://www.instagram.com/p/B/
    python scripts/test_comment_scrapers.py --from-db 17    # take top-N from the same queue Step 3 uses

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

Reproducing the pipeline failure mode
-------------------------------------
``louisdeconinck`` works here but returns 0 items in
``scripts/pipeline.py`` Step 3. To narrow down which difference is
responsible, the script can drop test-only safety rails one at a time:

    --mode batch | per-url      one Apify call per URL list, or one per URL
    --mimic-pipeline            shortcut: --mode batch + drop every safety rail
    --no-proxy                  do NOT inject ``proxy: useApifyProxy``
    --no-input-limit            do NOT pass ``resultsLimit``/``maxComments``/``maxItems`` in run_input
    --no-sdk-cap                do NOT pass ``max_items`` to ``.call()`` (SDK-level dataset cap)
    --no-timeout                do NOT pass ``timeout_secs`` to ``.call()``

Example reproduction:

    # Pull 17 unscanned posts and run louisdeconinck the way the pipeline does
    python scripts/test_comment_scrapers.py \\
        --actors louisdeconinck --from-db 17 --mimic-pipeline --yes

If that returns 0, narrow down by re-enabling rails one at a time
(remove ``--mimic-pipeline`` and add ``--mode batch`` plus subset of
``--no-*`` flags) until the actor recovers — that's the offender.

Adding a new actor
------------------
Append an entry to ``ACTORS`` keyed by a short CLI name. The
``build_input`` callable receives ``(urls: list[str], limit: int | None)``
and returns the actor's ``run_input`` dict — different scrapers use
different URL input keys (``directUrls`` vs ``urls`` vs ``startUrls``).
``limit=None`` means "do not include any comment-cap field" — that's how
``--no-input-limit`` works. The output extractor is shared, so if the
new actor returns a novel item shape, extend ``_extract_commenter``
rather than adding a per-adapter parser — that keeps the comparison
apples-to-apples.
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
# Hardcoded overrides (for IDE-run usage)
# ============================================================================
# Passing CLI flags from the Cursor "run" button is awkward, so instead of
# editing the command line, edit the dict below. Each key must match the
# corresponding argparse dest (CLI ``--from-db`` -> key ``from_db``, etc.).
# Anything set here OVERRIDES the CLI value AFTER parsing -- so a stale
# command-line argv from the IDE wrapper can't undo your override.
#
# Leave the dict EMPTY ``{}`` to fall back to CLI flags exclusively.
#
# Recipes -- copy a block into HARDCODE_OVERRIDES below:
#
# 1) Reproduce the pipeline failure (mimic Step 3 exactly):
#       {"actors": "louisdeconinck", "from_db": 3,
#        "mimic_pipeline": True, "yes": True}
#
# 2) Bisect: turn ONLY input_limit back ON
#    (probes whether resultsLimit/maxComments fixes it):
#       {"actors": "louisdeconinck", "from_db": 3, "mode": "batch",
#        "no_sdk_cap": True, "no_timeout": True, "yes": True}
#
# 3) Bisect: turn ONLY SDK max_items back ON:
#       {"actors": "louisdeconinck", "from_db": 3, "mode": "batch",
#        "no_input_limit": True, "no_timeout": True, "yes": True}
#
# 4) Bisect: turn ONLY timeout back ON:
#       {"actors": "louisdeconinck", "from_db": 3, "mode": "batch",
#        "no_input_limit": True, "no_sdk_cap": True, "yes": True}
#
# 5) Bisect: switch from batch to per-url
#    (probes whether the 17-URL batch itself is what breaks things):
#       {"actors": "louisdeconinck", "from_db": 3, "mode": "per-url",
#        "no_input_limit": True, "no_sdk_cap": True,
#        "no_timeout": True, "yes": True}
#
# 6) Healthy baseline (the historical "this works" config):
#       {"actors": "louisdeconinck", "from_db": 3, "yes": True}
HARDCODE_OVERRIDES: dict = {
    # Bisection step 3 -- final control before patching the pipeline.
    # We confirmed:
    #   * Step 1 (per-url + every rail OFF) -> 0 items
    #   * Step 2 (per-url + input_limit ON) -> works
    # So ``input_limit`` is the offender. The remaining question is
    # whether the pipeline can keep its ONE-call-with-N-URLs batch
    # shape (cheaper, fewer cold starts) or has to split into per-URL
    # calls. Test that here: ``mode=batch`` + ``input_limit`` ON,
    # everything else OFF.
    #
    # If this returns >0 items: pipeline fix = add
    # ``resultsLimit``/``maxComments`` to the existing batch call.
    # If this returns 0: the actor mishandles batches even with the
    # cap; pipeline fix = split into per-URL calls in addition to
    # passing the cap.
    "actors": "louisdeconinck",
    "from_db": 3,
    "mode": "batch",
    # input_limit ON (no_input_limit not set)
    "no_sdk_cap": True,
    "no_timeout": True,
    "yes": True,
}


# ============================================================================
# Adapters
# ============================================================================

@dataclass
class ActorAdapter:
    name: str
    actor_id: str
    notes: str
    # ``build_input`` receives a *list* of URLs (the script may run a
    # single URL or a batch -- ``--mode batch`` drives the difference)
    # and an optional ``limit`` that caps comments per actor. ``limit``
    # is ``None`` when ``--no-input-limit`` is set: in that case the
    # adapter must omit every ``resultsLimit`` / ``maxComments`` /
    # ``maxItems`` field so we can probe whether the absence of those
    # keys is what trips the actor up in the pipeline.
    build_input: Callable[[list[str], int | None], dict]


def _opt_cap(field: str, limit: int | None) -> dict:
    """Return ``{field: limit}`` if ``limit`` is set, else ``{}``.

    Helper used by every adapter so ``--no-input-limit`` consistently
    drops the cap field across all scrapers without duplicating the
    ``if limit:`` branch four times.
    """
    return {field: limit} if limit is not None else {}


ACTORS: dict[str, ActorAdapter] = {
    "official": ActorAdapter(
        name="official",
        actor_id="apify/instagram-comment-scraper",
        notes=(
            "Official Apify scraper. Reliable. Item shape: top-level "
            "ownerUsername + nested owner{id, username, full_name, ...}. "
            "Pricing ~$0.0023/comment."
        ),
        build_input=lambda urls, limit: {
            "directUrls": list(urls),
            **_opt_cap("resultsLimit", limit),
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
        build_input=lambda urls, limit: {
            "urls": list(urls),
            **_opt_cap("resultsLimit", limit),
            **_opt_cap("maxComments", limit),
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
        build_input=lambda urls, limit: {
            "startUrls": list(urls),
            **_opt_cap("maxItems", limit),
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
        build_input=lambda urls, limit: {
            "startUrls": list(urls),
            **_opt_cap("maxItems", limit),
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

def resolve_post_urls(args: argparse.Namespace, db_path: str) -> list[str]:
    """Pick one or more Instagram post URLs to run every actor against.

    Priority (first match wins):

    1. ``--urls`` -- explicit comma-separated list (used to repro the
       pipeline's batch-call shape).
    2. ``--from-db N`` -- pull the same posts Step 3 would scan: the
       relevance=relevant + cta=comment + (never scanned OR comments
       grew >= COMMENTS_GROWTH_PCT) queue, capped at N. Mirrors
       ``LeadDB.get_posts_needing_comments`` but inlined here so the
       script keeps its single-file shape and ignores the 5% growth
       threshold (we just want a representative batch).
    3. ``--url`` -- single explicit URL.
    4. ``--shortcode`` -- single explicit shortcode.
    5. Auto-pick: highest-comment relevant + cta=comment post,
       preferring unscanned ones so the comments are still fresh.
    """
    if args.urls:
        urls = [u.strip() for u in args.urls.split(",") if u.strip()]
        if not urls:
            raise SystemExit("--urls was empty after parsing.")
        return urls
    if args.from_db:
        if args.from_db <= 0:
            raise SystemExit("--from-db must be a positive integer")
        if not Path(db_path).exists():
            raise SystemExit(
                f"DB not found at {db_path}. --from-db needs leads.db."
            )
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT post_url, owner_username, comments_count, "
            "       last_scanned_at "
            "FROM processed_posts "
            "WHERE relevance = 'relevant' AND cta_type = 'comment' "
            "  AND post_url IS NOT NULL AND post_url != '' "
            "ORDER BY (last_scanned_at IS NULL) DESC, "
            "         comments_count DESC "
            "LIMIT ?",
            (args.from_db,),
        ).fetchall()
        con.close()
        if not rows:
            raise SystemExit(
                "No suitable posts in DB for --from-db. "
                "Run the pipeline through Step 2 first."
            )
        urls = [r["post_url"] for r in rows]
        log.info(
            "auto_picked_from_db",
            count=len(urls),
            total_comments=sum(r["comments_count"] or 0 for r in rows),
        )
        print(f"  Auto-picked from DB: {len(urls)} post(s), "
              f"~{sum(r['comments_count'] or 0 for r in rows)} comments total")
        return urls
    if args.url:
        return [args.url]
    if args.shortcode:
        return [f"https://www.instagram.com/p/{args.shortcode}/"]

    if not Path(db_path).exists():
        raise SystemExit(
            f"DB not found at {db_path}. Pass --url, --urls, --shortcode, "
            f"or --from-db explicitly."
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
    return [row["post_url"]]


# ============================================================================
# Per-actor probe
# ============================================================================

@dataclass
class ProbeOpts:
    """Per-probe knobs that control which "safety rails" we keep on.

    The script's job is to find which difference between the test and
    ``scripts/pipeline.py`` Step 3 makes ``louisdeconinck`` return 0 in
    the pipeline. Each rail can be independently disabled so a bisect
    is a single ``--no-*`` flag per run, not a code edit.

    Attributes
    ----------
    mode :
        ``"batch"`` -- one ``.call()`` with the full URL list (matches
        the pipeline). ``"per-url"`` -- one ``.call()`` per URL with
        results aggregated. Default ``per-url`` -- it's the historical
        behavior of this script and the only path that has been
        observed to work consistently for ``louisdeconinck``.
    proxy :
        Inject ``proxy: {useApifyProxy: true}`` into the actor input.
        Apify Instagram scrapers without proxy frequently get blocked
        by IG and report ``status=SUCCEEDED`` with 0 items.
    input_limit :
        When set, every adapter passes the value as
        ``resultsLimit`` / ``maxComments`` / ``maxItems`` (whichever
        applies). When ``None``, the cap field is omitted entirely --
        ``--no-input-limit``.
    sdk_max_items :
        SDK-level ``max_items`` arg to ``actor.call()``. Propagates as
        ``ACTOR_MAX_PAID_DATASET_ITEMS``, aborts the run when the
        dataset reaches the cap. ``None`` = no cap (matches pipeline).
    sdk_timeout_secs :
        SDK-level ``timeout_secs`` arg to ``actor.call()``. ``None`` =
        no timeout (matches pipeline). The default test value (300s)
        was added to catch ``apify/instagram-comment-scraper`` hangs.
    """

    mode: str = "per-url"
    proxy: bool = True
    input_limit: int | None = None
    sdk_max_items: int | None = None
    sdk_timeout_secs: int | None = None


def _empty_probe_result(adapter: ActorAdapter) -> dict:
    """Skeleton result used as the accumulator in per-url mode."""
    return {
        "name": adapter.name,
        "actor_id": adapter.actor_id,
        "input_params": [],
        "run_id": [],
        "status": [],
        "items_count": 0,
        "extractable_count": 0,
        "unhandled_items_count": 0,
        "unique_usernames": 0,
        "cost_usd": 0.0,
        "cost_per_unique": None,
        "duration_ms": 0,
        "samples": [],
        "unhandled_sample": None,
        "error": None,
    }


def _single_call(
    client: ApifyClient,
    adapter: ActorAdapter,
    urls: list[str],
    opts: ProbeOpts,
) -> dict:
    """Run the actor exactly once with ``urls`` as the URL batch.

    Returns the per-call slice of the report dict. ``probe_actor``
    aggregates one or more of these depending on ``opts.mode``.
    """
    run_input = adapter.build_input(urls, opts.input_limit)
    if opts.proxy:
        run_input.setdefault("proxy", {"useApifyProxy": True})

    log.info(
        "probe_call_start",
        actor=adapter.actor_id,
        urls=len(urls),
        input_keys=list(run_input.keys()),
        sdk_max_items=opts.sdk_max_items,
        sdk_timeout_secs=opts.sdk_timeout_secs,
    )
    started = time.monotonic()

    # Build kwargs lazily so ``None`` values are *omitted* (instead of
    # passed as ``None``, which the SDK might interpret as "use 0").
    # This is what makes ``--no-sdk-cap`` and ``--no-timeout`` actually
    # match the pipeline's call shape.
    call_kwargs: dict = {"run_input": run_input}
    if opts.sdk_max_items is not None:
        call_kwargs["max_items"] = opts.sdk_max_items
    if opts.sdk_timeout_secs is not None:
        call_kwargs["timeout_secs"] = opts.sdk_timeout_secs

    try:
        run = client.actor(adapter.actor_id).call(**call_kwargs)
    except Exception as exc:
        elapsed = round((time.monotonic() - started) * 1000)
        log.error("probe_call_failed", actor=adapter.actor_id, error=str(exc))
        return {
            "input_params": run_input,
            "run_id": None,
            "status": "ERROR",
            "items": [],
            "cost_usd": 0.0,
            "duration_ms": elapsed,
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

    log.info(
        "probe_call_done",
        actor=adapter.actor_id,
        run_id=run_id,
        status=status,
        urls=len(urls),
        items=len(items),
        cost=cost,
        duration_s=round(duration_ms / 1000, 1),
    )

    return {
        "input_params": run_input,
        "run_id": run_id,
        "status": status,
        "items": items,
        "cost_usd": cost,
        "duration_ms": duration_ms,
        "error": None,
    }


def probe_actor(
    client: ApifyClient,
    adapter: ActorAdapter,
    urls: list[str],
    opts: ProbeOpts,
) -> dict:
    """Run ``adapter`` against ``urls`` and aggregate the result.

    In ``opts.mode == "batch"`` we issue exactly one ``.call()`` with
    every URL stuffed into ``run_input`` -- this matches what
    ``scripts/pipeline.py`` does in Step 3. In ``"per-url"`` mode each
    URL gets its own run; the outputs are concatenated, and per-call
    metadata (run ids, statuses, costs) is preserved as a list so the
    JSON report shows whether *some* URLs succeeded and others didn't.
    """
    aggregated = _empty_probe_result(adapter)

    calls: list[list[str]]
    if opts.mode == "batch":
        calls = [list(urls)]
    elif opts.mode == "per-url":
        calls = [[u] for u in urls]
    else:
        raise SystemExit(f"Unknown probe mode: {opts.mode!r}")

    extractable: list[dict] = []
    unhandled_sample: dict | None = None
    unhandled_count = 0
    seen: set[str] = set()
    errors: list[str] = []

    for url_batch in calls:
        slice_ = _single_call(client, adapter, url_batch, opts)

        # Keep the per-call breakdown so the JSON report shows the
        # success/empty pattern when running per-url -- crucial when
        # only a subset of URLs trips the actor up.
        aggregated["input_params"].append(slice_["input_params"])
        aggregated["run_id"].append(slice_["run_id"])
        aggregated["status"].append(slice_["status"])
        aggregated["cost_usd"] += slice_["cost_usd"]
        aggregated["duration_ms"] += slice_["duration_ms"]
        aggregated["items_count"] += len(slice_["items"])
        if slice_.get("error"):
            errors.append(slice_["error"])

        for it in slice_["items"]:
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

    aggregated["extractable_count"] = len(extractable)
    aggregated["unhandled_items_count"] = unhandled_count
    aggregated["unique_usernames"] = len(seen)
    aggregated["cost_usd"] = round(aggregated["cost_usd"], 6)
    aggregated["cost_per_unique"] = (
        round(aggregated["cost_usd"] / len(seen), 6)
        if aggregated["cost_usd"] and seen
        else None
    )
    aggregated["samples"] = extractable[:5]
    aggregated["unhandled_sample"] = unhandled_sample
    aggregated["error"] = "; ".join(errors) if errors else None

    # Collapse a single-call probe back to flat scalars for readability
    # in the JSON report -- nobody wants to read ``run_id: ["abc"]``
    # when there's only one run.
    if len(calls) == 1:
        aggregated["input_params"] = aggregated["input_params"][0]
        aggregated["run_id"] = aggregated["run_id"][0]
        aggregated["status"] = aggregated["status"][0]

    # Pick a single status for the table heuristics. Per-url mode is
    # graded "FAILED" if *any* call errored, "EMPTY" if *all* succeeded
    # but yielded zero items, and the most common single status (e.g.
    # SUCCEEDED) otherwise so the side-by-side stays glanceable.
    if isinstance(aggregated["status"], list):
        statuses = [s for s in aggregated["status"] if s]
        if errors:
            display_status = "ERROR"
        elif statuses and all(s == statuses[0] for s in statuses):
            display_status = statuses[0]
        elif "SUCCEEDED" in statuses:
            display_status = "MIXED"
        else:
            display_status = statuses[0] if statuses else "UNKNOWN"
        aggregated["display_status"] = display_status
    else:
        aggregated["display_status"] = aggregated["status"]

    log.info(
        "probe_done",
        actor=adapter.actor_id,
        mode=opts.mode,
        urls=len(urls),
        calls=len(calls),
        items=aggregated["items_count"],
        extractable=aggregated["extractable_count"],
        unique=aggregated["unique_usernames"],
        cost=aggregated["cost_usd"],
        duration_s=round(aggregated["duration_ms"] / 1000, 1),
    )

    return aggregated


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Side-by-side comparison of Apify Instagram comment scrapers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", help="Specific post URL to test")
    p.add_argument(
        "--urls",
        default="",
        help=(
            "Comma-separated URL list (use to repro the pipeline's "
            "batch-call shape -- pair with --mode batch)."
        ),
    )
    p.add_argument(
        "--shortcode",
        help="Specific post shortcode (e.g. DXZm2A1iDn_)",
    )
    p.add_argument(
        "--from-db",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Pull top-N posts from the same queue Step 3 uses "
            "(relevance=relevant + cta=comment). Useful for repro: "
            "--from-db 17 --mode batch matches the failure observed "
            "in scripts/pipeline.py."
        ),
    )
    p.add_argument(
        "--actors",
        default="",
        help="Comma-separated subset (default: all). See --list for names.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "Max comments per actor as passed in run_input "
            "(resultsLimit/maxComments/maxItems). Default: 50. "
            "Use --no-input-limit to omit the field entirely."
        ),
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help=(
            "Per-actor run timeout in seconds (default: 300). "
            "Aborts hung runs. Use --no-timeout to omit the SDK arg "
            "entirely (matches scripts/pipeline.py)."
        ),
    )

    # Reproduction tumblers -- each one independently strips a "safety
    # rail" so we can bisect which rail's absence breaks louisdeconinck
    # in the pipeline.
    p.add_argument(
        "--mode",
        choices=("per-url", "batch"),
        default="per-url",
        help=(
            "per-url (default): one .call() per URL. "
            "batch: one .call() with every URL in run_input -- matches "
            "scripts/pipeline.py Step 3."
        ),
    )
    p.add_argument(
        "--mimic-pipeline",
        action="store_true",
        help=(
            "Shortcut for the exact call shape Step 3 uses: "
            "--mode batch, --no-input-limit, --no-sdk-cap, --no-timeout "
            "(proxy stays on -- it was just added to the pipeline). "
            "Run with --from-db N to feed it real URLs."
        ),
    )
    p.add_argument(
        "--no-proxy",
        action="store_true",
        help="Drop the proxy: useApifyProxy field from run_input.",
    )
    p.add_argument(
        "--no-input-limit",
        action="store_true",
        help=(
            "Drop resultsLimit / maxComments / maxItems from run_input "
            "(matches scripts/pipeline.py)."
        ),
    )
    p.add_argument(
        "--no-sdk-cap",
        action="store_true",
        help=(
            "Drop SDK-level max_items=ACTOR_MAX_PAID_DATASET_ITEMS "
            "from .call() (matches scripts/pipeline.py)."
        ),
    )
    p.add_argument(
        "--no-timeout",
        action="store_true",
        help=(
            "Drop SDK-level timeout_secs from .call() "
            "(matches scripts/pipeline.py)."
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
    args = p.parse_args()

    # Apply HARDCODE_OVERRIDES last so they win over whatever the IDE
    # wrapper passed on argv. We validate every key against the parser
    # so a typo (``form_db`` instead of ``from_db``) blows up loudly
    # instead of silently being ignored.
    if HARDCODE_OVERRIDES:
        valid_attrs = set(vars(args).keys())
        applied: list[str] = []
        for key, value in HARDCODE_OVERRIDES.items():
            if key not in valid_attrs:
                raise SystemExit(
                    f"HARDCODE_OVERRIDES key {key!r} does not match any "
                    f"argparse dest. Valid keys: {sorted(valid_attrs)}"
                )
            setattr(args, key, value)
            applied.append(f"{key}={value!r}")
        print()
        print("  HARDCODE_OVERRIDES applied:")
        for line in applied:
            print(f"    - {line}")

    return args


def _opts_from_args(args: argparse.Namespace) -> ProbeOpts:
    """Translate the CLI flags into a ``ProbeOpts``.

    ``--mimic-pipeline`` is treated as a *shorthand*: it sets the
    pipeline-equivalent rails OFF, but explicit ``--no-*`` flags can
    still be combined with it (e.g. ``--mimic-pipeline --no-proxy``
    if you want to roll back the proxy fix as well).
    """
    if args.mimic_pipeline:
        mode = "batch"
        # ``--no-*`` flags AND --mimic-pipeline both go to the same
        # "rail off" state. Independent settings (proxy) stay
        # respectful of an explicit --no-proxy on top.
        no_input_limit = True
        no_sdk_cap = True
        no_timeout = True
    else:
        mode = args.mode
        no_input_limit = args.no_input_limit
        no_sdk_cap = args.no_sdk_cap
        no_timeout = args.no_timeout

    return ProbeOpts(
        mode=mode,
        proxy=not args.no_proxy,
        input_limit=None if no_input_limit else args.limit,
        sdk_max_items=None if no_sdk_cap else args.limit,
        sdk_timeout_secs=None if no_timeout else args.timeout,
    )


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

    urls = resolve_post_urls(args, args.db)
    opts = _opts_from_args(args)

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

    if len(urls) == 1:
        print(f"  Post URL:          {urls[0]}")
    else:
        print(f"  Post URLs ({len(urls)}):")
        for u in urls[:5]:
            print(f"    - {u}")
        if len(urls) > 5:
            print(f"    ... +{len(urls) - 5} more")
    print(f"  Mode:              {opts.mode}")
    if args.mimic_pipeline:
        print("  Mimic pipeline:    ON (rails: input_limit, sdk_cap, timeout OFF)")
    print(f"  proxy injected:    {opts.proxy}")
    print(f"  run_input cap:     "
          f"{opts.input_limit if opts.input_limit is not None else 'OFF'}")
    print(f"  SDK max_items:     "
          f"{opts.sdk_max_items if opts.sdk_max_items is not None else 'OFF'}")
    print(f"  SDK timeout_secs:  "
          f"{opts.sdk_timeout_secs if opts.sdk_timeout_secs is not None else 'OFF'}")
    print(f"  Actors ({len(selected)}):")
    for a in selected:
        print(f"    - {a.name:<16} {a.actor_id}")

    # Cost ceiling guess: per-url makes one call per URL; batch makes
    # one. Each call is bounded by ``input_limit`` if set, else there's
    # no real cap and cost is proportional to comments * $0.0023. Show
    # both numbers so the operator knows what they're consenting to.
    calls_n = len(urls) if opts.mode == "per-url" else 1
    cap_per_call = opts.input_limit if opts.input_limit is not None else 200
    upper_bound = round(cap_per_call * 0.0023 * calls_n * len(selected), 4)
    cap_label = (
        f"{cap_per_call} items/call"
        if opts.input_limit is not None
        else f"~{cap_per_call} items/call (UNCAPPED -- estimate only)"
    )
    print(
        f"  Upper-bound cost:  ~${upper_bound:.4f} "
        f"({calls_n} call(s) per actor x {len(selected)} actor(s) "
        f"x {cap_label})"
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
        result = probe_actor(client, adapter, urls, opts)
        results.append(result)
        status = (result.get("display_status") or "").upper()
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
                "post_urls": urls,
                "opts": {
                    "mode": opts.mode,
                    "proxy": opts.proxy,
                    "input_limit": opts.input_limit,
                    "sdk_max_items": opts.sdk_max_items,
                    "sdk_timeout_secs": opts.sdk_timeout_secs,
                    "mimic_pipeline": args.mimic_pipeline,
                },
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
        raw_status = (r.get("display_status") or "").upper()
        if r["error"]:
            status_str = "ERROR"
        elif raw_status in {"TIMED-OUT", "TIMEOUT", "ABORTED", "FAILED"}:
            status_str = raw_status
        elif r["items_count"] == 0 and raw_status == "SUCCEEDED":
            status_str = "EMPTY"
        elif r["extractable_count"] == 0 and r["items_count"] > 0:
            status_str = "UNREADABLE"
        else:
            status_str = raw_status or "?"
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
