"""Smoke test for the Sherlock REST API.

Sherlock is an external Telegram bot wrapped behind a FastAPI service
that we call to resolve Instagram leads to phone numbers / Telegram
contacts (Module 2 of the project). This script is the very first
sanity check: hit ``GET /v1/health`` via :class:`SherlockClient` and
confirm we can reach the service and that our ``SHERLOCK_API_KEY`` is
recognized.

The ``/v1/health`` endpoint itself is not gated by ``X-API-Key`` in
the OpenAPI schema, but the client still sends the header — that way
an obvious typo / wrong key surfaces here instead of later on a real
/search call.

The endpoint returns the ``HealthResponse`` payload:

    {
      "status":  "...",   # service liveness
      "version": "...",   # service version string
      "db":      "...",   # DB connectivity (e.g. "ok")
      "pool":    {        # optional: account pool summary
        "total":     <int>,
        "by_status": { "<AccountStatus>": <int>, ... }
      }
    }

The ``pool`` block is the practically interesting bit — it tells us how
many Telegram accounts the bot has behind it and how they break down
across the ``AccountStatus`` enum (idle / busy / limited / banned / ...).
A healthy service with usable accounts should report ``pool.total > 0``
and at least one account in ``idle`` (= ready to take a search). The
pipeline's Step 5 sizes its worker pool against that ``idle`` count.

Usage:
    python scripts/check_sherlock_health.py
    python scripts/check_sherlock_health.py --base-url http://other.host:8000
    python scripts/check_sherlock_health.py --timeout 30
    python scripts/check_sherlock_health.py --json     # raw JSON dump only

On success (``Verdict: OK``), prints ``(Ready for job - N accounts)`` where
``N`` is ``pool.by_status.idle``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mirror the photo-test fix: the Sherlock API can include non-ASCII
# (Russian text, emoji) in any field, and the default Windows console
# codepage (cp1251) can't encode them. Reconfigure stdout/stderr to
# UTF-8 with replacement so we never crash mid-print.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


from src.sherlock_client import (
    API_KEY_ENV_VAR,
    DEFAULT_BASE_URL,
    HEALTH_PATH,
    SherlockClient,
    SherlockError,
    pool_idle_count,
)

DEFAULT_TIMEOUT = 15


def _mask_key(key: str) -> str:
    """Render only the first/last 4 chars so the key isn't echoed in full."""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]} (len={len(key)})"


def _print_pool(pool: dict) -> None:
    """Pretty-print the ``PoolSummary`` block: total + per-status counts.

    Sorts statuses by count desc so the most populated buckets surface first.
    """
    total = pool.get("total")
    by_status = pool.get("by_status") or {}
    print(f"  Pool total: {total}")
    if not by_status:
        print("  Pool by_status: (empty)")
        return
    items = sorted(by_status.items(), key=lambda kv: (-int(kv[1] or 0), kv[0]))
    print("  Pool by_status:")
    for status, count in items:
        print(f"    - {status:<22} {count}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe the Sherlock API /v1/health endpoint."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SHERLOCK_API_BASE_URL", DEFAULT_BASE_URL),
        help=(
            "Sherlock API base URL (default: %(default)s). "
            "Can also be set via SHERLOCK_API_BASE_URL."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only the raw JSON response (machine-readable).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip().strip("'\"") or None

    if not args.json:
        print("=" * 60)
        print(f"Base URL:  {args.base_url}")
        print(f"Endpoint:  {HEALTH_PATH}")
        print(f"API key:   {_mask_key(api_key or '')}"
              + ("" if api_key else f"  ({API_KEY_ENV_VAR} not set)"))
        print(f"Timeout:   {args.timeout}s")
        print("=" * 60)

    # Build a client that bypasses make_sherlock_client's
    # mandatory-key check -- /v1/health works without auth and we
    # want users to be able to probe the service even before the
    # SHERLOCK_API_KEY env var exists.
    client = SherlockClient(
        base_url=args.base_url,
        api_key=api_key,
        http_timeout=args.timeout,
    )

    t0 = time.monotonic()
    try:
        body = client.health()
    except requests.Timeout:
        print(f"ERROR: request timed out after {args.timeout}s.")
        return 2
    except requests.ConnectionError as exc:
        print(f"ERROR: cannot reach {args.base_url}: {exc}")
        return 2
    except SherlockError as exc:
        # Wraps non-2xx and JSON-decode errors. The message already
        # includes the HTTP status + truncated body for debugging.
        if not args.json:
            print(f"ERROR: {exc}")
        return 1
    except requests.RequestException as exc:
        print(f"ERROR: request failed: {exc}")
        return 2
    finally:
        client.close()

    elapsed_ms = (time.monotonic() - t0) * 1000

    if args.json:
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    print(f"HTTP status: 200  ({elapsed_ms:.0f} ms)")
    print("-" * 60)

    print(f"  status:  {body.get('status')!r}")
    print(f"  version: {body.get('version')!r}")
    print(f"  db:      {body.get('db')!r}")
    pool = body.get("pool")
    if pool:
        _print_pool(pool)
    else:
        print("  pool:    (not reported)")

    print("=" * 60)
    if (body.get("status") or "").lower() in {"ok", "healthy", "up"}:
        verdict = "OK"
    else:
        verdict = "REACHABLE (status field not 'ok' — inspect manually)"
    print(f"Verdict: {verdict}")
    if verdict == "OK":
        idle = pool_idle_count(pool if isinstance(pool, dict) else None)
        if idle is not None:
            noun = "account" if idle == 1 else "accounts"
            print(f"(Ready for job - {idle} {noun})")
        else:
            print("(Ready for job - unknown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
