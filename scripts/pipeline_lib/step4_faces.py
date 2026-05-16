import os
from pathlib import Path

from src.db import LeadDB

from scripts.pipeline_lib.logging import log


def face_bbox_percent_of_image(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float]:
    """BBox vs full raster: ``(area_percent, width_percent, height_percent)``."""
    x1, y1, x2, y2 = bbox
    bw = max(0.0, float(x2 - x1))
    bh = max(0.0, float(y2 - y1))
    iw = float(image_width)
    ih = float(image_height)
    if iw <= 0.0 or ih <= 0.0:
        return (0.0, 0.0, 0.0)
    area_pct = 100.0 * (bw * bh) / (iw * ih)
    w_pct = 100.0 * bw / iw
    h_pct = 100.0 * bh / ih
    return (area_pct, w_pct, h_pct)


def _same_disk_face_file(path_a: str, path_b: str) -> bool:
    """True if both strings refer to the same on-disk file."""
    try:
        return Path(path_a).resolve() == Path(path_b).resolve()
    except OSError:
        return os.path.normcase(os.path.abspath(path_a)) == os.path.normcase(
            os.path.abspath(path_b)
        )


def _reconcile_step4_ephemeral_avatar(
    db: LeadDB,
    log,
    *,
    username: str,
    downloaded_avatar_path: str,
    final_face_path: str | None,
) -> None:
    """Drop avatar file if it is not the canonical ``face_photo_path`` target.

    When the avatar *is* the canonical photo, keep the file and ``avatar_path``
    until Step 6 post-Sherlock cleanup.
    """
    if final_face_path is not None and _same_disk_face_file(
        downloaded_avatar_path, final_face_path
    ):
        return
    p = Path(downloaded_avatar_path)
    try:
        p.unlink(missing_ok=True)
    except OSError as e:
        log.warning(
            "step4_avatar_unlink_failed",
            username=username,
            path=str(p),
            error=str(e),
        )
    db.clear_lead_avatar_path(username)
    reason = "no_canonical_face" if final_face_path is None else "canonical_elsewhere"
    log.info("step4_avatar_disk_released", username=username, reason=reason)


def _pick_post_images(
    latest_posts: list[dict] | None,
    limit: int,
    *,
    skip_videos: bool = True,
) -> list[str]:
    """Pick at most one representative image URL from each of the first
    ``limit`` posts in ``latestPosts``.

    We intentionally take one image per post (not every carousel slide)
    so that clustering counts *distinct post appearances* — if the same
    person posts a 10-slide carousel of themselves, it shouldn't drown
    out four separate posts showing someone else.

    Preference per post:
      1. ``images[0]`` — carousel cover / first slide (always a photo).
      2. ``displayUrl`` — the single photo of a photo post.
      3. Otherwise skip (videos, empties).
    """
    if not latest_posts:
        return []

    urls: list[str] = []
    for post in latest_posts[:limit]:
        images = post.get("images") or []
        if images and images[0]:
            urls.append(images[0])
            continue
        display_url = post.get("displayUrl")
        video_url = post.get("videoUrl")
        if not display_url:
            continue
        if skip_videos and video_url:
            continue
        urls.append(display_url)
    return urls
