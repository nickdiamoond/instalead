"""Batch nick-search harness aligned with pipeline Step 5 (nick stage only).

Loads up to ``DB_SAMPLE_LIMIT`` rows from ``lead_accounts`` with the same
WHERE clause as :meth:`LeadDB.get_leads_for_sherlock`. Opens SQLite with
``mode=ro`` (no writes).

Orchestration mirrors :func:`scripts.pipeline._step_5_resolve_contacts_via_sherlock`
for the parts that apply to nick search:

  * one shared :class:`~src.sherlock_client.SherlockClient` across the thread
    pool (same as the pipeline — do not give each worker its own session);
  * worker count from ``--workers``, else ``sherlock.concurrency.workers`` in
    config, else ``GET /v1/health`` → ``pool.by_status.idle`` (fallback 3);
  * same ``enqueue_nick`` / ``wait_for_task`` arguments as
    :func:`scripts.pipeline._resolve_one_lead_via_sherlock` stage 1.

Photo fallback is intentionally omitted; the script dumps the final ``TaskOut``
JSON from Sherlock for each nick.

**Why it looked “stuck”:** each nick can take ~30–300s with no output unless
we log polls — this script prints enqueue + per-poll status lines (like the
pipeline’s per-lead progress, but more verbose).

Usage:
    python scripts/test_sherlock_step5_nick_only.py
    python scripts/test_sherlock_step5_nick_only.py --workers 1
    python scripts/test_sherlock_step5_nick_only.py --db data/leads.db
    python scripts/test_sherlock_step5_nick_only.py --no-poll-log   # JSON blocks only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sherlock_client import SherlockClient, SherlockError, make_sherlock_client

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

# Hardcoded: how many lead usernames to pull from the DB (Step 5 pool).
DB_SAMPLE_LIMIT = 50

# Same selection as LeadDB.get_leads_for_sherlock (scripts/pipeline Step 5).
_STEP5_CANDIDATE_SQL = """
SELECT username, user_id, full_name, face_photo_path
FROM lead_accounts
WHERE profile_fetched = 1
  AND phone IS NULL
  AND telegram_username IS NULL
  AND sherlock_processed_at IS NULL
  AND COALESCE(is_private, 0) = 0
LIMIT ?
"""

PollPrinter = Callable[..., None]


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


def _fetch_leads_readonly(db_path: Path, limit: int) -> list[dict]:
    """Return Step 5-shaped rows (read-only), same columns as get_leads_for_sherlock."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    uri_path = db_path.resolve().as_posix()
    uri = f"file:{uri_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(_STEP5_CANDIDATE_SQL.strip(), (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _resolve_workers(
    sherlock: SherlockClient,
    cfg: dict,
    workers_override: int | None,
) -> tuple[int, str]:
    """Match pipeline: CLI > config > /v1/health pool.idle."""
    sh_cfg = cfg.get("sherlock") or {}
    conc_cfg = sh_cfg.get("concurrency") or {}
    if workers_override is not None:
        return max(1, int(workers_override)), "--workers"
    if conc_cfg.get("workers") is not None:
        return max(1, int(conc_cfg["workers"])), "config.yaml sherlock.concurrency.workers"
    return max(1, sherlock.get_pool_idle(fallback=3)), "/v1/health pool.idle"


def _format_eta(seconds: float) -> str:
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    return f"{seconds:.0f}s"


def _run_one_nick(
    client: SherlockClient,
    lead: dict,
    *,
    nick_cfg: dict,
    task_cfg: dict,
    on_poll: PollPrinter | None,
) -> tuple[str, dict | None, str | None]:
    """Nick stage only — same API calls as pipeline._resolve_one_lead_via_sherlock."""
    username = lead["username"]
    poll_interval = float(task_cfg.get("poll_interval_secs", 3))
    max_wait = float(task_cfg.get("max_wait_secs", 300))
    max_pages = int(nick_cfg.get("max_pages", 1))
    max_attempts = int(nick_cfg.get("max_attempts", 3))
    try:
        enq = client.enqueue_nick(
            nick=username,
            search_in="telegram",
            max_pages=max_pages,
            max_attempts=max_attempts,
        )
        task_id = enq["id"]
        if on_poll is not None:
            on_poll(
                "enqueued",
                username,
                task_id=task_id,
            )
        task = client.wait_for_task(
            task_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            on_poll=(
                lambda pc, el, tk: on_poll("poll", username, poll_count=pc, elapsed=el, task=tk)
                if on_poll is not None
                else None
            ),
        )
        return username, task, None
    except (SherlockError, TimeoutError) as exc:
        return username, None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return username, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Sherlock nick-search batch test (Step 5 nick stage only). "
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
        default=None,
        help="Thread pool size (default: same rules as pipeline Step 5).",
    )
    parser.add_argument(
        "--no-poll-log",
        action="store_true",
        help="Disable enqueue/poll progress lines (only username banner + JSON).",
    )
    args = parser.parse_args()

    # Ensure .env is loaded when the IDE cwd is not the repo root (pipeline
    # often runs from repo root; this script may not).
    load_dotenv(root / ".env")
    load_dotenv()

    cfg = _load_cfg(args.config)
    db_path = _resolve_db_path(cfg, root, args.db)

    try:
        leads = _fetch_leads_readonly(db_path, DB_SAMPLE_LIMIT)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr, flush=True)
        return 1

    sh_cfg = cfg.get("sherlock") or {}
    nick_cfg = sh_cfg.get("nick_search") or {}
    task_cfg = sh_cfg.get("task") or {}

    print(
        "\n=== Usernames under test (Step 5 pool from DB, read-only) ===",
        flush=True,
    )
    print(f"  DB: {db_path}", flush=True)
    print(f"  Limit (hardcoded): {DB_SAMPLE_LIMIT}", flush=True)
    print(f"  Fetched: {len(leads)}", flush=True)
    for i, lead in enumerate(leads, start=1):
        print(f"  {i:2}. {lead['username']}", flush=True)
    print(f"=== Total: {len(leads)} ===\n", flush=True)

    if not leads:
        print("No candidates match Step 5 selector — nothing to query.", flush=True)
        return 0

    print("Initializing Sherlock client (same factory as pipeline)...", flush=True)
    try:
        sherlock = make_sherlock_client(cfg)
    except EnvironmentError as exc:
        print(f"Cannot build Sherlock client: {exc}", file=sys.stderr, flush=True)
        return 1

    workers, workers_source = _resolve_workers(sherlock, cfg, args.workers)
    n = len(leads)
    nick_eta_s = 30
    best_eta = (n * nick_eta_s) / max(workers, 1)

    print(f"  Workers:         {workers}  (from {workers_source})", flush=True)
    print(
        f"  Rough ETA:       ~{_format_eta(best_eta)} wall-clock "
        f"if ~{nick_eta_s}s per nick (actual time varies).",
        flush=True,
    )
    print(
        "  Poll log:        "
        + ("off (--no-poll-log)" if args.no_poll_log else "on (stderr)"),
        flush=True,
    )
    print(flush=True)

    print_lock = threading.Lock()

    def poll_printer(kind: str, username: str, **kw: object) -> None:
        if args.no_poll_log:
            return
        with print_lock:
            if kind == "enqueued":
                tid = kw.get("task_id", "?")
                print(
                    f"  [{username}] enqueued task_id={tid!r}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "poll":
                pc = kw.get("poll_count")
                el = kw.get("elapsed")
                task = kw.get("task") or {}
                st = task.get("status")
                print(
                    f"  [{username}] poll #{pc} {float(el):.1f}s "
                    f"task.status={st!r}",
                    file=sys.stderr,
                    flush=True,
                )

    def dump_block(username: str, task: dict | None, err: str | None) -> None:
        with print_lock:
            print(f"----- nick={username} -----", flush=True)
            if task is not None:
                print(json.dumps(task, indent=2, ensure_ascii=False), flush=True)
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

    on_poll: PollPrinter | None = poll_printer if not args.no_poll_log else None

    try:
        if workers == 1:
            for lead in leads:
                u, task, err = _run_one_nick(
                    sherlock,
                    lead,
                    nick_cfg=nick_cfg,
                    task_cfg=task_cfg,
                    on_poll=on_poll,
                )
                dump_block(u, task, err)
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="sherlock_nick"
            ) as pool:
                futures = {
                    pool.submit(
                        _run_one_nick,
                        sherlock,
                        lead,
                        nick_cfg=nick_cfg,
                        task_cfg=task_cfg,
                        on_poll=on_poll,
                    ): lead
                    for lead in leads
                }
                for fut in as_completed(futures):
                    u, task, err = fut.result()
                    dump_block(u, task, err)
    finally:
        sherlock.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
