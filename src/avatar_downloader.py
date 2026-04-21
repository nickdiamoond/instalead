"""Download Instagram avatars to local disk for face detection.

Instagram CDN URLs are signed and expire within 1-2 days, so avatars
must be downloaded right after the profile scrape. Files are stored at
`data/avatars/<user_id>.jpg` (or `<username>.jpg` if user_id is missing).
"""

from __future__ import annotations

from pathlib import Path

import urllib.request
import urllib.error

from src.logger import get_logger

log = get_logger("avatar_downloader")

AVATARS_DIR = Path("data/avatars")
DEFAULT_TIMEOUT = 10
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _safe_stem(value: str) -> str:
    """Make a filesystem-safe filename stem."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)[:120]


def download_avatar(
    url: str,
    user_id: str | None,
    username: str | None = None,
    *,
    avatars_dir: Path = AVATARS_DIR,
    timeout: int = DEFAULT_TIMEOUT,
) -> str | None:
    """Download an avatar image. Idempotent.

    Returns the relative path on success, None on failure.
    """
    if not url:
        return None

    identifier = user_id or username
    if not identifier:
        log.warning("avatar_no_identifier", url=url[:80])
        return None

    avatars_dir.mkdir(parents=True, exist_ok=True)
    dest = avatars_dir / f"{_safe_stem(str(identifier))}.jpg"

    if dest.exists() and dest.stat().st_size > 0:
        return str(dest).replace("\\", "/")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            log.warning("avatar_empty", identifier=identifier)
            return None
        dest.write_bytes(data)
        return str(dest).replace("\\", "/")
    except urllib.error.HTTPError as e:
        log.warning("avatar_http_error", identifier=identifier, status=e.code)
        return None
    except Exception as e:
        log.warning("avatar_download_error", identifier=identifier, error=str(e))
        return None
