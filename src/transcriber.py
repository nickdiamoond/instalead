"""Nexara API wrapper for audio/video transcription.

Used by the pipeline's Step 2 to recover relevance signal for video posts
whose caption is missing or whose caption analysis returned ``unknown``:
the audio track of the video itself is transcribed and re-fed into the
``RELEVANCE_PROMPT`` flow.

Instagram CDN video URLs are signed and expire within ~1-2 days, so the
download + transcribe pass must happen in the same pipeline run as the
post-scraper fetch that produced the URL.
"""

from __future__ import annotations

import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import requests

from src.logger import get_logger
from src.pipeline_logger import PipelineLogger

log = get_logger("transcriber")

NEXARA_ENDPOINT = "https://api.nexara.ru/api/v1/audio/transcriptions"
DEFAULT_HTTP_TIMEOUT = 30
DEFAULT_API_TIMEOUT = 180

# Same UA as avatar_downloader — IG CDN otherwise returns 403 on
# unauthenticated bot-flavored requests.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class NexaraTranscriber:
    """Thin wrapper around Nexara's ``/audio/transcriptions`` endpoint.

    If ``api_key`` is empty/None, the wrapper degrades gracefully:
    :py:meth:`transcribe` returns ``None`` immediately and logs a warning
    once. This keeps the pipeline runnable without a Nexara key — it just
    skips the video-fallback branch and falls back to the legacy
    ``relevance="unknown"`` behavior.
    """

    def __init__(
        self,
        api_key: str | None,
        *,
        http_timeout: int = DEFAULT_HTTP_TIMEOUT,
        api_timeout: int = DEFAULT_API_TIMEOUT,
        pipeline: PipelineLogger | None = None,
    ) -> None:
        self.api_key = api_key or None
        self.http_timeout = http_timeout
        self.api_timeout = api_timeout
        self.pipeline = pipeline
        self._warned_no_key = False

        if not self.api_key:
            log.warning(
                "nexara_no_api_key",
                msg="NEXARA_API_KEY is not set — transcription disabled",
            )
            self._warned_no_key = True

    def transcribe(self, video_url: str) -> str | None:
        """Download a video URL and return its transcribed text.

        Returns ``None`` on any failure (no key, download error, Nexara
        non-2xx, empty body). Errors are logged but never raised — the
        caller should treat ``None`` as "fall back to whatever you'd do
        without a transcript".
        """
        if not self.api_key:
            if not self._warned_no_key:
                log.warning("nexara_no_api_key_skip")
                self._warned_no_key = True
            return None
        if not video_url:
            return None

        tmp_path = self._download_video(video_url)
        if tmp_path is None:
            self._log_pipeline(status="DOWNLOAD_FAILED", error="ig_download")
            return None

        try:
            return self._post_to_nexara(tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError as e:
                log.warning("nexara_tmp_cleanup_failed", error=str(e))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _download_video(self, url: str) -> Path | None:
        """Stream the IG video URL into a NamedTemporaryFile (.mp4).

        Returns the closed temp file path on success, ``None`` on any
        HTTP / network error.
        """
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            tmp = tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False, prefix="ig_video_"
            )
            tmp_path = Path(tmp.name)
            try:
                with urllib.request.urlopen(req, timeout=self.http_timeout) as resp:
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
            finally:
                tmp.close()

            if tmp_path.stat().st_size == 0:
                log.warning("nexara_video_empty", url=url[:80])
                tmp_path.unlink(missing_ok=True)
                return None
            return tmp_path
        except urllib.error.HTTPError as e:
            log.warning("nexara_video_http_error", url=url[:80], status=e.code)
            return None
        except Exception as e:
            log.warning("nexara_video_download_error", url=url[:80], error=str(e))
            return None

    def _post_to_nexara(self, video_path: Path) -> str | None:
        """POST the local video file to Nexara, return plain-text body."""
        run_id = uuid.uuid4().hex[:12]
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with video_path.open("rb") as fh:
                files = {"file": (video_path.name, fh, "video/mp4")}
                data = {"response_format": "text"}
                resp = requests.post(
                    NEXARA_ENDPOINT,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self.api_timeout,
                )
        except requests.RequestException as e:
            log.warning("nexara_request_error", error=str(e))
            self._log_pipeline(
                status="REQUEST_ERROR", error=str(e)[:200], run_id=run_id
            )
            return None

        if resp.status_code != 200:
            body = resp.text[:300] if resp.text else ""
            log.warning(
                "nexara_non_200", status=resp.status_code, body=body
            )
            self._log_pipeline(
                status=f"HTTP_{resp.status_code}",
                error=body,
                run_id=run_id,
            )
            return None

        text = (resp.text or "").strip()
        if not text:
            log.warning("nexara_empty_body")
            self._log_pipeline(status="EMPTY_BODY", run_id=run_id)
            return None

        log.info(
            "nexara_transcribed",
            chars=len(text),
            preview=text[:80],
        )
        self._log_pipeline(
            status="OK",
            run_id=run_id,
            transcript_chars=len(text),
            file_bytes=video_path.stat().st_size,
        )
        return text

    def _log_pipeline(
        self,
        *,
        status: str,
        run_id: str | None = None,
        error: str | None = None,
        transcript_chars: int | None = None,
        file_bytes: int | None = None,
    ) -> None:
        if self.pipeline is None:
            return
        extra: dict = {}
        if transcript_chars is not None:
            extra["transcript_chars"] = transcript_chars
        if file_bytes is not None:
            extra["file_bytes"] = file_bytes
        self.pipeline.log_run(
            actor_id="nexara/transcriptions",
            run_id=run_id or uuid.uuid4().hex[:12],
            status=status,
            input_params={"endpoint": NEXARA_ENDPOINT},
            items_count=1 if status == "OK" else 0,
            cost_usd=None,
            duration_ms=None,
            error=error,
            extra=extra or None,
        )
