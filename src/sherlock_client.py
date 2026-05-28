"""Reusable client for the Sherlock REST API.

Sherlock is a FastAPI wrapper around the external Telegram bot
(``http://94.131.9.237:8000`` by default) that the pipeline's Step 5
talks to in order to resolve Instagram leads to phone numbers /
Telegram contacts. The bot drives a real Telegram client behind the
scenes, so end-to-end calls are slow (tens of seconds to a few
minutes) -- every endpoint that *does* the work is async: you POST to
``/v1/search/...`` to enqueue, then poll ``/v1/tasks/{id}`` until the
``status`` field reaches a terminal state.

This module centralizes the request shape, the API-key header, the
polling loop, and the terminal-status set so callers (the pipeline's
Step 5 + the dev test scripts under ``scripts/test_sherlock_*.py``)
don't drift apart on field names / endpoint paths over time.

The client is intentionally synchronous: each call blocks the calling
thread on a single ``requests`` session. The pipeline parallelises by
spawning multiple workers (one per Sherlock account in the pool) and
giving each worker its own client instance -- ``requests`` is
thread-safe for separate ``Session`` objects, and the API itself
serialises tasks server-side via the account pool.

Typical use:

    from src.sherlock_client import make_sherlock_client

    sherlock = make_sherlock_client(cfg)
    workers = sherlock.get_pool_idle(fallback=3)
    enq = sherlock.enqueue_nick("raizzep", search_in="telegram")
    task = sherlock.wait_for_task(enq["id"])
    if task["status"] == "completed":
        results = (task.get("result") or {}).get("results") or []
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import requests
from dotenv import load_dotenv

from src.logger import get_logger

log = get_logger("sherlock_client")


DEFAULT_BASE_URL = "http://94.131.9.237:8000"
API_KEY_HEADER = "X-API-Key"
API_KEY_ENV_VAR = "SHERLOCK_API_KEY"

# TaskStatus enum values that are *final* (per OpenAPI). Polling stops
# as soon as the server reports any of these. Anything else
# (``pending`` / ``assigned`` / ``running``) means keep waiting.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "timeout"})

# Status returned per-result inside ``result.results`` for photo search
# when Sherlock decided the candidate is *not* the same person. Used as
# the gate in Step 5: we save the first result only if its ``status``
# is anything other than this exact Cyrillic string. Stored as a
# constant so the spec never drifts into a typo / wrong dash.
PHOTO_RESULT_STATUS_NO_MATCH = "не совпадение"

# Endpoint paths.
HEALTH_PATH = "/v1/health"
NICK_PATH = "/v1/search/nick"
PHOTO_PATH = "/v1/search/photo"
TASK_PATH = "/v1/tasks/{task_id}"
INTERACTIONS_PATH = "/v1/tasks/{task_id}/interactions"

# Defaults for the polling loop. Sized to match what Sherlock observed
# in our smoke tests: photo search lands at ~135 s (max_pages=20),
# nick search at ~30 s with max_pages=1. ``max_wait=300`` covers the
# slowest expected photo search with comfortable headroom; bumping it
# higher mostly buys patience for a saturated pool.
DEFAULT_POLL_INTERVAL = 3.0
DEFAULT_MAX_WAIT = 300.0
DEFAULT_HTTP_TIMEOUT = 90.0
DEFAULT_HEALTH_PROBE_MAX_ATTEMPTS = 3

# Image content types Sherlock advertises (JPEG / PNG). Anything else
# is sent as ``application/octet-stream`` -- the server still accepts
# it but we stop pretending we know what it is.
CONTENT_TYPE_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


# Callback signature for ``wait_for_task``: ``(poll_count, elapsed_s, task)``.
# The task dict is the latest TaskOut payload. Callbacks return nothing;
# they're for progress UI only.
PollCallback = Callable[[int, float, dict], None]


class SherlockError(RuntimeError):
    """Raised on any non-2xx response or invalid JSON from Sherlock.

    Wraps the underlying ``requests`` failure so callers don't have to
    catch both ``requests.RequestException`` and our own JSON decode
    errors -- a single ``except SherlockError`` suffices.
    """


def _content_type_for(path: Path) -> str:
    return CONTENT_TYPE_BY_EXT.get(path.suffix.lower(), "application/octet-stream")


class SherlockClient:
    """Thin synchronous client around the Sherlock REST API.

    One instance is safe to share across calls on a single thread.
    For parallel workers (the pipeline's Step 5), construct one
    instance per worker -- each carries its own ``requests.Session``.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        *,
        http_timeout: float = DEFAULT_HTTP_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_wait: float = DEFAULT_MAX_WAIT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.http_timeout = http_timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        # Reusing a Session keeps the underlying TCP connection warm,
        # which matters when one worker fires off a few requests per
        # task (enqueue + N polls).
        self._session = requests.Session()

    # --- low-level helpers ----------------------------------------

    def _headers(self, *, with_key: bool = True) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if with_key and self.api_key:
            h[API_KEY_HEADER] = self.api_key
        return h

    def _url(self, path: str, **kwargs) -> str:
        return self.base_url + (path.format(**kwargs) if kwargs else path)

    def _decode(self, resp: requests.Response) -> dict | list:
        """Parse JSON or raise ``SherlockError`` with the response text
        attached for debugging. We don't ``raise_for_status()`` first
        because the API encodes 422 validation errors as JSON too and
        callers may want to inspect them; the *caller's* check on HTTP
        code drives whether to treat it as an error."""
        try:
            return resp.json()
        except ValueError as exc:
            raise SherlockError(
                f"non-JSON response: HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            ) from exc

    # --- public API -----------------------------------------------

    def health(self) -> dict:
        """``GET /v1/health``. The endpoint is not gated by ``X-API-Key``
        per the OpenAPI schema, but we still send the header so an
        invalid key surfaces here on the first call instead of later
        on a real /search."""
        resp = self._session.get(
            self._url(HEALTH_PATH),
            headers=self._headers(),
            timeout=self.http_timeout,
        )
        if resp.status_code != 200:
            raise SherlockError(
                f"health failed: HTTP {resp.status_code}: {resp.text[:500]}"
            )
        body = self._decode(resp)
        if not isinstance(body, dict):
            raise SherlockError(f"health returned non-object: {body!r}")
        return body

    def get_pool_idle(self, fallback: int = 3) -> int:
        """How many accounts are currently in the ``idle`` bucket.

        This is the practical concurrency cap for Sherlock -- one task
        consumes one account, and pending tasks queue server-side
        ordered by ``priority``. The pipeline's Step 5 sizes its
        worker pool against this number.

        Falls back to ``fallback`` (3 = pool size we've observed) if
        the health call fails or the response is malformed -- the
        pipeline should still run with a reasonable default rather
        than blowing up because the server is briefly unreachable.
        """
        try:
            body = self.health()
        except (SherlockError, requests.RequestException) as exc:
            log.warning("sherlock_pool_idle_fallback", error=str(exc),
                        fallback=fallback)
            return fallback
        pool = body.get("pool")
        idle = pool_idle_count(pool if isinstance(pool, dict) else None)
        if idle is None or idle < 1:
            log.warning("sherlock_pool_idle_unusable", pool=pool,
                        fallback=fallback)
            return fallback
        return idle

    def enqueue_nick(
        self,
        nick: str,
        *,
        search_in: str = "telegram",
        max_pages: int = 1,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> dict:
        """``POST /v1/search/nick`` -> returns the ``TaskEnqueueResponse``
        body (most importantly ``id``)."""
        payload = {
            "nick": nick,
            "search_in": search_in,
            "max_pages": max_pages,
            "priority": priority,
            "max_attempts": max_attempts,
        }
        resp = self._session.post(
            self._url(NICK_PATH),
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=self.http_timeout,
        )
        if resp.status_code != 200:
            raise SherlockError(
                f"enqueue_nick failed: HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )
        body = self._decode(resp)
        if not isinstance(body, dict) or "id" not in body:
            raise SherlockError(f"enqueue_nick: unexpected body: {body!r}")
        return body

    def enqueue_photo(
        self,
        photo_path: Path | str,
        *,
        max_pages: int = 20,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> dict:
        """``POST /v1/search/photo`` (multipart/form-data) -> returns
        the ``TaskEnqueueResponse`` body. Reads the file off disk on
        every call -- callers concerned about throughput should keep
        the file pre-existent (we don't touch its lifetime)."""
        path = Path(photo_path)
        if not path.exists():
            raise SherlockError(f"enqueue_photo: file not found: {path}")

        with path.open("rb") as fh:
            files = {
                "photo": (path.name, fh, _content_type_for(path)),
            }
            # Starlette/FastAPI parses non-file form fields as plain
            # strings, so we send ints stringified.
            data = {
                "max_pages": str(max_pages),
                "priority": str(priority),
                "max_attempts": str(max_attempts),
            }
            resp = self._session.post(
                self._url(PHOTO_PATH),
                headers=self._headers(),
                files=files,
                data=data,
                timeout=self.http_timeout,
            )
        if resp.status_code != 200:
            raise SherlockError(
                f"enqueue_photo failed: HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )
        body = self._decode(resp)
        if not isinstance(body, dict) or "id" not in body:
            raise SherlockError(f"enqueue_photo: unexpected body: {body!r}")
        return body

    def get_task(self, task_id: str) -> dict:
        """``GET /v1/tasks/{task_id}`` -> ``TaskOut`` body."""
        resp = self._session.get(
            self._url(TASK_PATH, task_id=task_id),
            headers=self._headers(),
            timeout=self.http_timeout,
        )
        if resp.status_code != 200:
            raise SherlockError(
                f"get_task failed: HTTP {resp.status_code}: {resp.text[:500]}"
            )
        body = self._decode(resp)
        if not isinstance(body, dict):
            raise SherlockError(f"get_task: unexpected body: {body!r}")
        return body

    def wait_for_task(
        self,
        task_id: str,
        *,
        poll_interval: float | None = None,
        max_wait: float | None = None,
        on_poll: PollCallback | None = None,
    ) -> dict:
        """Poll ``/v1/tasks/{task_id}`` until ``status`` becomes terminal.

        Returns the final ``TaskOut`` payload. Raises ``TimeoutError``
        if ``max_wait`` elapses without a terminal status.

        ``on_poll`` (optional) receives ``(poll_count, elapsed_s, task)``
        after every successful poll -- used by the dev test scripts to
        render a single-line ``\\r`` progress indicator. The pipeline
        leaves it ``None`` (multiple workers would interleave anyway).
        """
        interval = poll_interval if poll_interval is not None else self.poll_interval
        budget = max_wait if max_wait is not None else self.max_wait

        start = time.monotonic()
        deadline = start + budget
        poll_count = 0

        while True:
            task = self.get_task(task_id)
            poll_count += 1
            elapsed = time.monotonic() - start
            if on_poll is not None:
                on_poll(poll_count, elapsed, task)
            status = (task.get("status") or "").lower()
            if status in TERMINAL_STATUSES:
                return task
            if time.monotonic() + interval > deadline:
                raise TimeoutError(
                    f"task {task_id} did not reach terminal status within "
                    f"{budget:.0f}s (last status: {status!r})"
                )
            time.sleep(interval)

    def get_interactions(self, task_id: str) -> list[dict]:
        """``GET /v1/tasks/{task_id}/interactions`` -> full TG event log."""
        resp = self._session.get(
            self._url(INTERACTIONS_PATH, task_id=task_id),
            headers=self._headers(),
            timeout=self.http_timeout,
        )
        if resp.status_code != 200:
            raise SherlockError(
                f"get_interactions failed: HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )
        body = self._decode(resp)
        if not isinstance(body, list):
            raise SherlockError(
                f"get_interactions: expected list, got {type(body).__name__}"
            )
        return body

    def close(self) -> None:
        self._session.close()


def make_sherlock_client(
    cfg: dict | None = None,
    *,
    api_key: str | None = None,
) -> SherlockClient:
    """Build a SherlockClient from ``config.yaml`` + ``.env``.

    Reads:
      * ``cfg["sherlock"]["base_url"]`` (default: :data:`DEFAULT_BASE_URL`)
      * ``cfg["sherlock"]["api_key_env_var"]`` (default: ``SHERLOCK_API_KEY``)
      * ``cfg["sherlock"]["task"]["poll_interval_secs"]`` (default: 3)
      * ``cfg["sherlock"]["task"]["max_wait_secs"]`` (default: 300)

    The API key is read from the environment variable named in config
    (after :func:`dotenv.load_dotenv`), unless an explicit ``api_key``
    is passed in (mostly for tests). A missing key raises immediately
    -- every search endpoint requires it, so it's pointless to fail
    later on the first POST.
    """
    sh = (cfg or {}).get("sherlock") or {}
    base_url = sh.get("base_url") or DEFAULT_BASE_URL
    env_var = sh.get("api_key_env_var") or API_KEY_ENV_VAR
    task_cfg = sh.get("task") or {}
    poll_interval = float(task_cfg.get("poll_interval_secs", DEFAULT_POLL_INTERVAL))
    max_wait = float(task_cfg.get("max_wait_secs", DEFAULT_MAX_WAIT))

    if api_key is None:
        load_dotenv()
        api_key = os.environ.get(env_var, "").strip().strip("'\"") or None
    if not api_key:
        raise EnvironmentError(
            f"Sherlock API key is missing: set {env_var} in .env "
            f"(or pass api_key= explicitly)."
        )

    return SherlockClient(
        base_url=base_url,
        api_key=api_key,
        poll_interval=poll_interval,
        max_wait=max_wait,
    )


def pool_idle_count(pool: dict | None) -> int | None:
    """Return ``pool.by_status.idle`` as int, or None if missing/unparseable."""
    if not pool:
        return None
    idle = (pool.get("by_status") or {}).get("idle")
    if isinstance(idle, int):
        return idle
    if idle is not None:
        try:
            return int(idle)
        except (TypeError, ValueError):
            pass
    return None


def probe_health_pool_idle(
    client: SherlockClient,
    *,
    max_attempts: int = DEFAULT_HEALTH_PROBE_MAX_ATTEMPTS,
) -> tuple[dict | None, int]:
    """Probe ``GET /v1/health`` up to *max_attempts* times.

    Returns ``(body, idle)`` when HTTP succeeds (``idle`` is 0 if the field is
    absent). Returns ``(None, 0)`` when every attempt raises — callers should
    treat ``body is None`` as API unavailable.
    """
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            body = client.health()
            pool = body.get("pool")
            idle = pool_idle_count(pool if isinstance(pool, dict) else None)
            return body, idle if idle is not None else 0
        except (SherlockError, requests.RequestException) as exc:
            last_error = str(exc)
            log.warning(
                "sherlock_health_probe_failed",
                attempt=attempt,
                max_attempts=max_attempts,
                error=last_error,
            )
    log.error(
        "sherlock_health_probe_exhausted",
        attempts=max_attempts,
        last_error=last_error,
    )
    return None, 0
