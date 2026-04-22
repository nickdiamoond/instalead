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
LEAD_PHOTOS_DIR = Path("data/lead_photos")
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


def _download_one(url: str, dest: Path, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Download a single URL to ``dest``. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            return False
        dest.write_bytes(data)
        return True
    except urllib.error.HTTPError as e:
        log.warning("lead_photo_http_error", url=url[:80], status=e.code)
        return False
    except Exception as e:
        log.warning("lead_photo_download_error", url=url[:80], error=str(e))
        return False


def download_post_photos(
    urls: list[str],
    user_id: str,
    *,
    dest_root: Path = LEAD_PHOTOS_DIR,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[Path]:
    """Download a batch of post photo URLs to ``data/lead_photos/<user_id>/``.

    Ignores empty/None URLs. Returns the list of successfully downloaded
    local paths (in input order). Failures are logged but do not abort
    the batch — Instagram CDN URLs occasionally 403 individually.
    """
    if not urls or not user_id:
        return []

    dest_dir = dest_root / _safe_stem(str(user_id))
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for idx, url in enumerate(urls):
        if not url:
            continue
        dest = dest_dir / f"{idx:02d}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            saved.append(dest)
            continue
        if _download_one(url, dest, timeout=timeout):
            saved.append(dest)
    return saved


def cleanup_lead_photos(
    user_id: str,
    *,
    keep: Path | None = None,
    dest_root: Path = LEAD_PHOTOS_DIR,
) -> tuple[int, int]:
    """Remove a lead's post-photo folder, optionally sparing one file.

    If ``keep`` is provided and lives inside the lead's folder, it is
    preserved (the rest of the folder is wiped). If ``keep`` is None,
    the whole per-lead folder is removed.

    Returns (deleted, failed).
    """
    if not user_id:
        return 0, 0
    lead_dir = dest_root / _safe_stem(str(user_id))
    if not lead_dir.exists():
        return 0, 0

    keep_resolved = keep.resolve() if keep else None
    deleted = 0
    failed = 0
    for f in lead_dir.iterdir():
        if not f.is_file():
            continue
        try:
            if keep_resolved is not None and f.resolve() == keep_resolved:
                continue
            f.unlink()
            deleted += 1
        except OSError:
            failed += 1

    # If nothing is being kept and the directory is now empty, remove it
    # too so data/lead_photos/ doesn't accumulate stale folders.
    if keep_resolved is None:
        try:
            lead_dir.rmdir()
        except OSError:
            pass
    return deleted, failed
