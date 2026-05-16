"""Batch nick-search harness mirroring pipeline Step 5 stage 1 only.

Loads up to ``DB_SAMPLE_LIMIT`` Instagram usernames from ``lead_accounts``
using the **same filters** as :meth:`LeadDB.get_leads_for_sherlock`
(Step 5 pool: SQL predicates + disk check on ``face_photo_path``).
Opens SQLite in ``mode=ro`` — no writes.

Runs ``POST /v1/search/nick`` + polls ``GET /v1/tasks/{id}`` per nick and
prints the full final ``TaskOut`` JSON from Sherlock.

Usage:
    python scripts/test_sherlock_step5_nick_only.py
    python scripts/test_sherlock_step5_nick_only.py --workers 4
    python scripts/test_sherlock_step5_nick_only.py --db data/leads.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import lead_disk_photo_usable
from src.sherlock_client import SherlockClient, SherlockError, make_sherlock_client

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

# Hardcoded: how many lead usernames to pull from the DB (Step 5 pool).
DB_SAMPLE_LIMIT = 50
SH_STATUS_FOUND_NICK = "found_nick"
SH_STATUS_NO_MATCH = "no_match"
SH_STATUS_ERROR = "error"

# Same selection as LeadDB.get_leads_for_sherlock (scripts/pipeline Step 5).
_STEP5_CANDIDATE_BATCH_SQL = """
SELECT username, face_photo_path
FROM lead_accounts
WHERE profile_fetched = 1
  AND phone IS NULL
  AND telegram_username IS NULL
  AND sherlock_processed_at IS NULL
  AND COALESCE(is_private, 0) = 0
  AND face_photo_path IS NOT NULL
LIMIT ? OFFSET ?
"""


def _load_cfg(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_db_path(cfg: dict, root: Path, override: Path | None) -> Path:
    if override is not None:
        p = override
    else:
        rel = (cfg.get("db") or {}).get("path") or "data/leads.db"
        p = Path(rel)
    if not p.is_absolute():
        p = root / p
    return p


def _fetch_usernames_readonly(
    db_path: Path, want: int, repo_root: Path
) -> list[str]:
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    uri_path = db_path.resolve().as_posix()
    uri = f"file:{uri_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    batch = max(256, min(want * 4, 4000))
    offset = 0
    out: list[str] = []
    try:
        while len(out) < want:
            rows = conn.execute(
                _STEP5_CANDIDATE_BATCH_SQL.strip(),
                (batch, offset),
            ).fetchall()
            if not rows:
                break
            offset += len(rows)
            for r in rows:
                fph = r["face_photo_path"]
                if not lead_disk_photo_usable(
                    None if fph is None else str(fph),
                    base_dir=repo_root,
                ):
                    continue
                name = r["username"]
                if name:
                    out.append(str(name))
                if len(out) >= want:
                    break
        return out
    finally:
        conn.close()


def _nick_params(cfg: dict) -> tuple[int, int, float, float]:
    sh = cfg.get("sherlock") or {}
    nick_cfg = sh.get("nick_search") or {}
    task_cfg = sh.get("task") or {}
    max_pages = int(nick_cfg.get("max_pages", 1))
    max_attempts = int(nick_cfg.get("max_attempts", 3))
    poll_interval = float(task_cfg.get("poll_interval_secs", 3))
    max_wait = float(task_cfg.get("max_wait_secs", 300))
    return max_pages, max_attempts, poll_interval, max_wait


def _run_one_nick(
    client: SherlockClient,
    username: str,
    *,
    max_pages: int,
    max_attempts: int,
    poll_interval: float,
    max_wait: float,
) -> tuple[str, dict | None, str | None]:
    try:
        query_nick = f"@{username}"
        enq = client.enqueue_nick(
            nick=query_nick,
            search_in="telegram",
            max_pages=max_pages,
            max_attempts=max_attempts,
        )
        task_id = enq["id"]
        task = client.wait_for_task(
            task_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
        )
        return username, task, None
    except (SherlockError, TimeoutError, OSError) as exc:
        return username, None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return username, None, f"{type(exc).__name__}: {exc}"


def _interpret_nick_outcome(
    queried_username: str,
    task: dict | None,
    client_error: str | None,
) -> dict:
    """Mirror Step 5's human-readable status for nick stage only."""
    out = {
        "status": SH_STATUS_ERROR,
        "telegram_username": None,
        "sherlock_link": None,
        "error": None,
    }
    if client_error:
        out["error"] = client_error
        return out
    if not task:
        out["error"] = "empty task payload"
        return out

    task_status = (task.get("status") or "").lower()
    if task_status == "completed":
        results = ((task.get("result") or {}).get("results")) or []
        if results:
            first = results[0] or {}
            tg_username = first.get("username") or queried_username
            tg_link = first.get("link") or f"https://t.me/{tg_username}"
            out.update({
                "status": SH_STATUS_FOUND_NICK,
                "telegram_username": tg_username,
                "sherlock_link": tg_link,
            })
            return out
        out["status"] = SH_STATUS_NO_MATCH
        return out

    out["status"] = SH_STATUS_NO_MATCH
    return out


def _has_profile_url(task: dict) -> bool:
    """True when Sherlock payload includes result->results->profile_url."""
    result = task.get("result") or {}
    results = result.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and item.get("profile_url"):
                return True
        return False
    if isinstance(results, dict):
        return bool(results.get("profile_url"))
    return False


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Sherlock nick-search batch test (Step 5 stage 1 only). "
            "Read-only DB; no pipeline writes."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config.yaml",
        help="Path to config.yaml (default: repo root).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to leads.db (default: db.path from config.yaml).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel Sherlock clients (default: 1 sequential).",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    db_path = _resolve_db_path(cfg, root, args.db)

    try:
        usernames = _fetch_usernames_readonly(
            db_path, DB_SAMPLE_LIMIT, root
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr, flush=True)
        return 1

    max_pages, max_attempts, poll_interval, max_wait = _nick_params(cfg)

    print(
        "\n=== Usernames under test (Step 5 pool from DB, read-only) ===",
        flush=True,
    )
    print(f"  DB: {db_path}", flush=True)
    print(f"  Limit (hardcoded): {DB_SAMPLE_LIMIT}", flush=True)
    print(f"  Fetched: {len(usernames)}", flush=True)
    for i, uname in enumerate(usernames, start=1):
        print(f"  {i:2}. {uname}", flush=True)
    print(f"=== Total: {len(usernames)} ===\n", flush=True)

    if not usernames:
        print("No candidates match Step 5 selector — nothing to query.", flush=True)
        return 0

    skipped_bad_nicks = 0
    eligible_usernames: list[str] = []
    for username in usernames:
        if "." in username:
            skipped_bad_nicks += 1
            print(
                f"  SKIP (nick not suitable): @{username} contains '.'",
                flush=True,
            )
            continue
        eligible_usernames.append(username)

    if not eligible_usernames:
        print("No suitable usernames after nick validation — nothing to query.", flush=True)
        return 0

    workers = max(1, int(args.workers))
    print_lock = threading.Lock()
    counters = {
        SH_STATUS_FOUND_NICK: 0,
        SH_STATUS_NO_MATCH: 0,
        SH_STATUS_ERROR: 0,
    }
    progress_done = 0

    def dump_block(username: str, task: dict | None, err: str | None) -> None:
        nonlocal progress_done
        interpreted = _interpret_nick_outcome(username, task, err)
        tag = interpreted["status"]
        counters[tag] = counters.get(tag, 0) + 1
        progress_done += 1
        detail_bits: list[str] = []
        if interpreted.get("telegram_username"):
            detail_bits.append(f"tg=@{interpreted['telegram_username']}")
        if interpreted.get("sherlock_link"):
            detail_bits.append(f"link={interpreted['sherlock_link']}")
        if interpreted.get("error") and interpreted["status"] == SH_STATUS_ERROR:
            detail_bits.append(f"err={interpreted['error'][:80]}")
        detail = "  " + " ".join(detail_bits) if detail_bits else ""
        with print_lock:
            print(
                f"  [{progress_done:>4}/{len(eligible_usernames)}] "
                f"@{username:<25} -> {tag}{detail}",
                flush=True,
            )
            print(f"----- nick={username} -----", flush=True)
            if task is not None:
                if _has_profile_url(task):
                    print(json.dumps(task, indent=2, ensure_ascii=False), flush=True)
                else:
                    print("аккаунт не найден", flush=True)
            else:
                print(
                    json.dumps(
                        {"nick": username, "client_error": err},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            print(flush=True)

    clients = [make_sherlock_client(cfg) for _ in range(workers)]
    try:
        if workers == 1:
            client = clients[0]
            for username in eligible_usernames:
                u, task, err = _run_one_nick(
                    client,
                    username,
                    max_pages=max_pages,
                    max_attempts=max_attempts,
                    poll_interval=poll_interval,
                    max_wait=max_wait,
                )
                dump_block(u, task, err)
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="sherlock_nick"
            ) as pool:
                futures = {}
                for idx, username in enumerate(eligible_usernames):
                    c = clients[idx % workers]
                    fut = pool.submit(
                        _run_one_nick,
                        c,
                        username,
                        max_pages=max_pages,
                        max_attempts=max_attempts,
                        poll_interval=poll_interval,
                        max_wait=max_wait,
                    )
                    futures[fut] = username
                for fut in as_completed(futures):
                    u, task, err = fut.result()
                    dump_block(u, task, err)
    finally:
        for c in clients:
            c.close()

    print()
    print(f"  DONE: {len(eligible_usernames)} processed")
    if skipped_bad_nicks:
        print(f"    skipped_bad_nicks   {skipped_bad_nicks}")
    for label in (SH_STATUS_FOUND_NICK, SH_STATUS_NO_MATCH, SH_STATUS_ERROR):
        print(f"    {label:<18} {counters.get(label, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
