"""End-to-end test for Sherlock's photo-search scenario.

Sends a portrait photo to ``POST /v1/search/photo`` via
:class:`SherlockClient`, then polls ``GET /v1/tasks/{task_id}`` until
the task reaches a terminal status (``completed`` / ``failed`` /
``cancelled`` / ``timeout``) and dumps the final ``TaskOut`` payload —
including ``result`` on success or ``error_code`` / ``error_message``
on failure.

Background: Sherlock proxies the request to a real Telegram bot, which
walks paginated results inside Telegram. End-to-end the call typically
takes tens of seconds (Telegram-side I/O dominates), so we poll the task
endpoint with a configurable interval rather than blocking on a single
HTTP request. ``--max-wait`` caps total time spent waiting (defaults to
5 minutes — anything longer almost certainly indicates the bot is stuck
or the pool is saturated).

Before submitting, the script also runs the project's own SCRFD face
detector on the photo locally (det_size matches our pipeline's avatar
calibration by default) and prints per-face det_score along with the
configured ``min_det_score``. This gives you an early sanity check on
whether the photo is something Sherlock will reasonably match: if our
own detector struggles to find a face above threshold, Sherlock's
upstream Telegram bot will likely struggle too.

Usage:
    python scripts/test_sherlock_photo.py                    # uses man.jpg from repo root
    python scripts/test_sherlock_photo.py path/to/face.jpg
    python scripts/test_sherlock_photo.py --max-pages 5      # stop the bot earlier
    python scripts/test_sherlock_photo.py --poll-interval 5 --max-wait 600
    python scripts/test_sherlock_photo.py --interactions     # also dump TG event log
    python scripts/test_sherlock_photo.py --face-kind post   # use 640px det_size (for feed photos)
    python scripts/test_sherlock_photo.py --face-kind off    # skip local face detection
    python scripts/test_sherlock_photo.py --json             # final TaskOut as raw JSON

This is a dev / smoke test — nothing is written to the project DB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.face_embedder import make_face_embedder
from src.sherlock_client import (
    API_KEY_ENV_VAR,
    DEFAULT_BASE_URL,
    SherlockClient,
    SherlockError,
    make_sherlock_client,
)

# Sherlock returns Russian text and TG interactions occasionally contain
# emoji. The default Windows console codepage is cp1251, which can't
# encode either - print() will raise UnicodeEncodeError mid-output and
# kill the script. Reconfigure to UTF-8 with replacement so a stray
# unprintable char becomes '?' instead of a hard crash. Python 3.7+.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


DEFAULT_PHOTO = "man.jpg"
DEFAULT_HTTP_TIMEOUT = 30      # per-request HTTP timeout
DEFAULT_POLL_INTERVAL = 3.0    # seconds between /tasks/{id} polls
DEFAULT_MAX_WAIT = 600         # overall budget for one task (5 min)
DEFAULT_MAX_PAGES = 20         # Sherlock-side knob (also the API default)


def _mask_key(key: str) -> str:
    """Render only the first/last 4 chars so the key isn't echoed in full."""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]} (len={len(key)})"


def _print_result(result: dict | None) -> None:
    """Pretty-print the ``result`` block. Sherlock returns a free-form
    object (per OpenAPI it's just ``additionalProperties: true``), so we
    don't make assumptions about its shape — JSON-dump and let the user
    eyeball it."""
    if not result:
        print("  result:  (empty)")
        return
    print("  result:")
    formatted = json.dumps(result, indent=2, ensure_ascii=False)
    for line in formatted.splitlines():
        print("    " + line)


def describe_local_face_detection(
    photo_path: Path,
    cfg: dict,
    kind: str,
) -> None:
    """Run SCRFD locally on the photo and print per-face det_score along
    with the configured ``min_det_score`` threshold.

    Sherlock matches by face, so a low local det_score is a useful early
    warning that we're sending a photo Sherlock will likely reject (or
    match poorly on). We mirror ``test_face_detector.py``'s technique:
    run the detector once with threshold=0 to see *every* SCRFD candidate
    (raw), then highlight which ones our config would keep (``KEEP``) vs
    drop (``drop``).

    ``kind="avatar"`` (default for this script) uses det_size=320 — the
    native Instagram avatar size; "face fills frame" selfies stay in
    SCRFD's anchor sweet spot. ``kind="post"`` uses 640 for feed-style
    images with smaller / multiple faces.
    """
    fd = cfg.get("face_detection") or {}
    min_score = float(fd.get("min_det_score", 0.6))
    det_size = int(
        fd.get("avatar_det_size" if kind == "avatar" else "post_det_size",
               320 if kind == "avatar" else 640)
    )

    print(f"  kind:           {kind!r}  (det_size={det_size}x{det_size})")
    print(f"  min_det_score:  {min_score}  (from config.yaml face_detection)")

    embedder = make_face_embedder(cfg, kind=kind)
    # Force the threshold to 0 so SCRFD's raw output is visible — the
    # configured ``min_score`` is applied manually in the printout below.
    embedder.min_det_score = 0.0

    t0 = time.monotonic()
    faces = embedder.embed_faces(photo_path)
    elapsed_ms = (time.monotonic() - t0) * 1000

    embedder.close()

    print(f"  raw detections: {len(faces)}  ({elapsed_ms:.0f} ms incl. cold load)")
    if not faces:
        print("  -> NO face detected at any score. Sherlock will likely "
              "reject the photo or return zero matches.")
        return

    kept = [f for f in faces if f.det_score >= min_score]
    print(f"  above threshold: {len(kept)} / {len(faces)}")
    print("  per-face:")
    for i, f in enumerate(sorted(faces, key=lambda x: -x.det_score), 1):
        x1, y1, x2, y2 = f.bbox
        w, h = x2 - x1, y2 - y1
        verdict = "KEEP" if f.det_score >= min_score else "drop"
        print(
            f"    #{i}  det_score={f.det_score:.3f}  "
            f"bbox=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  "
            f"size={w:.0f}x{h:.0f}  [{verdict}]"
        )

    if len(kept) == 0:
        print("  -> face(s) found but all BELOW min_det_score "
              f"({min_score}). For an avatar this usually means a side "
              "view / occlusion / very small face — Sherlock may still "
              "match but quality is uncertain.")
    elif len(kept) == 1:
        print("  -> exactly 1 face above threshold = ideal Sherlock input.")
    else:
        print(f"  -> {len(kept)} faces above threshold. Sherlock will "
              "match against the largest/most-confident one.")


def _print_interactions(interactions: list[dict]) -> None:
    """One-line-per-event TG interaction log: timestamp, direction, kind,
    truncated text. Useful for diagnosing where the bot got stuck."""
    if not interactions:
        print("  (no interactions recorded)")
        return
    print(f"  ({len(interactions)} TG events)")
    for ev in interactions:
        at = ev.get("at", "")
        direction = (ev.get("direction") or "?")[:3]
        kind = ev.get("kind") or "?"
        text = (ev.get("text") or "").replace("\n", " ").strip()
        if len(text) > 100:
            text = text[:97] + "..."
        print(f"    {at}  {direction:<3}  {kind:<14}  {text}")


def _make_progress_callback():
    """Build an ``on_poll`` callback for SherlockClient.wait_for_task
    that prints a single-line ``\\r``-refreshed status indicator. Kept
    out of :class:`SherlockClient` because the pipeline (multi-worker)
    doesn't want its output interleaved across threads."""
    def on_poll(poll_count: int, elapsed_s: float, task: dict) -> None:
        status = (task.get("status") or "").lower()
        attempts = task.get("attempts")
        max_attempts = task.get("max_attempts")
        account_id = task.get("account_id")
        sys.stdout.write(
            f"\r  [poll #{poll_count:>3}]  t+{elapsed_s:>5.1f}s  "
            f"status={status:<10}  attempt={attempts}/{max_attempts}  "
            f"account_id={account_id}    "
        )
        sys.stdout.flush()
    return on_poll


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit a photo to Sherlock /v1/search/photo and wait for the result.",
    )
    parser.add_argument(
        "photo",
        nargs="?",
        default=DEFAULT_PHOTO,
        help=f"Path to the JPEG/PNG portrait (default: {DEFAULT_PHOTO}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SHERLOCK_API_BASE_URL", DEFAULT_BASE_URL),
        help="Sherlock API base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=(
            "Sherlock-side: max pages of TG results to walk (1..100, default %(default)s). "
            "Lower = faster + cheaper, but you may miss matches on the tail."
        ),
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Queue priority (-100..100, default %(default)s).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Sherlock-side max retries on transient failures (1..10, default %(default)s).",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=DEFAULT_HTTP_TIMEOUT,
        help="Per-request HTTP timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds between /tasks/{id} polls (default: %(default)s).",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=DEFAULT_MAX_WAIT,
        help="Total seconds to wait for the task to finish (default: %(default)s).",
    )
    parser.add_argument(
        "--face-kind",
        choices=["avatar", "post", "off"],
        default="avatar",
        help=(
            "Run local SCRFD on the photo before submitting and show "
            "per-face det_score + the configured min_det_score. "
            "'avatar' (default) uses det_size=320 — the native Instagram "
            "avatar size; 'post' uses 640 (feed photos); 'off' skips the "
            "local detection pass entirely (saves ~2-3s of model load)."
        ),
    )
    parser.add_argument(
        "--interactions",
        action="store_true",
        help="After completion, also fetch /v1/tasks/{id}/interactions and print the TG event log.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only the final TaskOut as raw JSON (machine-readable).",
    )
    args = parser.parse_args()

    photo_path = Path(args.photo)
    if not photo_path.exists():
        print(f"ERROR: file not found: {photo_path.resolve()}")
        return 2
    if not photo_path.is_file():
        print(f"ERROR: not a regular file: {photo_path.resolve()}")
        return 2

    # Building the client validates that SHERLOCK_API_KEY is set --
    # photo search is gated by APIKeyHeader and must have a key. The
    # helper also reads cfg["sherlock"] for poll/timeout defaults so
    # we stay aligned with the pipeline's settings.
    cfg: dict | None = None
    try:
        cfg = load_config()
    except Exception as exc:
        # Config might fail (e.g. missing APIFY_API_TOKEN). Fall back
        # to direct construction so the test still works on minimal
        # setups -- only SHERLOCK_API_KEY is strictly required here.
        print(f"  (config load skipped: {exc})")

    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip().strip("'\"")
    if not api_key:
        print(f"ERROR: {API_KEY_ENV_VAR} is not set in .env")
        return 2

    client: SherlockClient
    if cfg is not None and cfg.get("sherlock"):
        # Use the factory so config defaults flow through.
        try:
            client = make_sherlock_client(cfg, api_key=api_key)
            client.base_url = args.base_url.rstrip("/")  # CLI override
            client.http_timeout = args.http_timeout
        except EnvironmentError as exc:
            print(f"ERROR: {exc}")
            return 2
    else:
        client = SherlockClient(
            base_url=args.base_url,
            api_key=api_key,
            http_timeout=args.http_timeout,
        )

    if not args.json:
        size_kb = photo_path.stat().st_size / 1024
        print("=" * 70)
        print(f"Base URL:      {args.base_url}")
        print(f"Photo:         {photo_path.resolve()}  ({size_kb:.1f} KB)")
        print(f"API key:       {_mask_key(api_key)}")
        print(f"max_pages:     {args.max_pages}")
        print(f"priority:      {args.priority}")
        print(f"max_attempts:  {args.max_attempts}")
        print(f"poll_interval: {args.poll_interval}s   max_wait: {args.max_wait}s")
        print("=" * 70)

        if args.face_kind != "off":
            print("Step 0: local SCRFD face detection on the photo ...")
            try:
                # Reuse the loaded cfg if we have it; otherwise reload
                # just for face detection (lightweight).
                fd_cfg = cfg if cfg is not None else load_config()
                describe_local_face_detection(
                    photo_path=photo_path,
                    cfg=fd_cfg,
                    kind=args.face_kind,
                )
            except Exception as exc:
                # Local face detection is informational only — never let
                # an InsightFace / config hiccup block the actual API test.
                print(f"  (face detection skipped: {exc})")
            print("-" * 70)

        print("Step 1: POST /v1/search/photo ...")

    try:
        enq = client.enqueue_photo(
            photo_path,
            max_pages=args.max_pages,
            priority=args.priority,
            max_attempts=args.max_attempts,
        )
    except (requests.RequestException, SherlockError) as exc:
        print(f"ERROR: enqueue failed: {exc}")
        client.close()
        return 1

    task_id = enq.get("id")
    if not task_id:
        print(f"ERROR: enqueue response missing 'id': {enq}")
        client.close()
        return 1

    if not args.json:
        print(f"  -> task_id:   {task_id}")
        print(f"     scenario:  {enq.get('scenario')!r}")
        print(f"     status:    {enq.get('status')!r}")
        print(f"     priority:  {enq.get('priority')}")
        print(f"     created:   {enq.get('created_at')}")
        print()
        print("Step 2: polling task state until terminal status ...")

    on_poll = _make_progress_callback() if not args.json else None

    t_poll_start = time.monotonic()
    try:
        task = client.wait_for_task(
            task_id,
            poll_interval=args.poll_interval,
            max_wait=args.max_wait,
            on_poll=on_poll,
        )
    except TimeoutError as exc:
        if not args.json:
            sys.stdout.write("\n")
        print(f"ERROR: {exc}")
        client.close()
        return 1
    except (requests.RequestException, SherlockError) as exc:
        if not args.json:
            sys.stdout.write("\n")
        print(f"ERROR: polling failed: {exc}")
        client.close()
        return 1
    elapsed = time.monotonic() - t_poll_start

    if not args.json:
        sys.stdout.write("\n")

    if args.json:
        print(json.dumps(task, indent=2, ensure_ascii=False))
        client.close()
        return 0 if (task.get("status") or "").lower() == "completed" else 1

    final_status = (task.get("status") or "").lower()
    print()
    print("=" * 70)
    print(f"Step 3: task finished in {elapsed:.1f}s with status={final_status!r}")
    print("-" * 70)
    print(f"  account_id:    {task.get('account_id')}")
    print(f"  attempts:      {task.get('attempts')}/{task.get('max_attempts')}")
    print(f"  started_at:    {task.get('started_at')}")
    print(f"  finished_at:   {task.get('finished_at')}")

    if final_status == "completed":
        _print_result(task.get("result"))
    else:
        err_code = task.get("error_code")
        err_msg = task.get("error_message")
        print(f"  error_code:    {err_code!r}")
        print(f"  error_message: {err_msg!r}")
        # Result may still be partial on failure - dump if present.
        if task.get("result"):
            _print_result(task.get("result"))

    if args.interactions:
        print("-" * 70)
        print("TG interactions:")
        try:
            interactions = client.get_interactions(task_id)
            _print_interactions(interactions)
        except (requests.RequestException, SherlockError) as exc:
            print(f"  (failed to fetch interactions: {exc})")

    client.close()
    print("=" * 70)
    return 0 if final_status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
