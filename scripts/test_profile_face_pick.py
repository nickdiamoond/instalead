"""Standalone test: same face-photo flow as pipeline Step 4 for one profile URL.

Fetches the profile via ``apify/instagram-profile-scraper``, downloads the
avatar, runs the avatar embedder; promotes the avatar only when there is
exactly one face and its bbox covers at least ``face_detection.min_avatar_face_area_pct``
percent of the image area (same gate as ``scripts/pipeline.py``). Otherwise
probes the last N posts (one image per post), downloads them, and runs
:func:`src.face_leader.resolve_face_leader` with the post embedder —
mirroring ``scripts/pipeline.py`` (without DB writes).

The winning image is copied to ``facetest/profile_face_winner/`` so temp
files under ``data/avatars`` and ``data/lead_photos`` stay aligned with the
production layout while you still get a stable artifact for inspection.

Edit ``INSTAGRAM_PROFILE_URL`` below, then run::

    python scripts/test_profile_face_pick.py
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Hardcoded test target — replace with any public Instagram profile URL.
# ---------------------------------------------------------------------------
INSTAGRAM_PROFILE_URL = "https://www.instagram.com/mmsh_14/"

# Output directory (under project root, separate from production DB paths).
OUTPUT_REL = Path("facetest") / "profile_face_winner"

_RESERVED_USER_SEGMENTS = frozenset(
    {
        "p",
        "reel",
        "reels",
        "stories",
        "explore",
        "accounts",
        "direct",
        "tv",
        "legal",
        "about",
    }
)


def _username_from_instagram_url(url: str) -> str:
    m = re.search(r"instagram\.com/([^/?#]+)", url.strip(), re.I)
    if not m:
        raise ValueError(f"Could not parse Instagram username from URL: {url!r}")
    user = m.group(1).strip().lstrip("@")
    if not user or user.lower() in _RESERVED_USER_SEGMENTS:
        raise ValueError(
            f"URL segment {user!r} is not a profile username — use e.g. "
            "https://www.instagram.com/<username>/"
        )
    return user


def _pick_post_images(
    latest_posts: list[dict] | None,
    limit: int,
    *,
    skip_videos: bool = True,
) -> list[str]:
    """Same selection rules as ``scripts.pipeline._pick_post_images``."""
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


# Same default as ``scripts/pipeline.DEFAULT_MIN_AVATAR_FACE_AREA_PCT`` —
# overridden by ``face_detection.min_avatar_face_area_pct`` in config.yaml.
DEFAULT_MIN_AVATAR_FACE_AREA_PCT = 2.0


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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from apify_client import ApifyClient
    from dotenv import load_dotenv

    from src.avatar_downloader import (
        cleanup_lead_photos,
        download_avatar,
        download_post_photos,
    )
    from src.config import load_config
    from src.face_embedder import make_face_embedder
    from src.face_leader import resolve_face_leader
    from src.logger import get_logger, setup_logging

    setup_logging()
    log = get_logger("test_profile_face_pick")

    load_dotenv()
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        print("APIFY_API_TOKEN is not set.", file=sys.stderr)
        return 1

    username = _username_from_instagram_url(INSTAGRAM_PROFILE_URL)
    cfg = load_config()
    fb_cfg = cfg.get("face_fallback") or {}
    fb_limit = int(fb_cfg.get("latest_posts_limit", 5))
    fb_min_cluster = int(fb_cfg.get("min_cluster_size", 2))
    fb_threshold = float(fb_cfg.get("cluster_threshold", 0.5))
    fb_skip_videos = bool(fb_cfg.get("skip_videos", True))
    fb_keep_photos = bool(fb_cfg.get("keep_photos", False))

    fd_cfg = cfg.get("face_detection") or {}
    min_avatar_face_area_pct = float(
        fd_cfg.get("min_avatar_face_area_pct", DEFAULT_MIN_AVATAR_FACE_AREA_PCT)
    )

    avatar_embedder = make_face_embedder(cfg, kind="avatar")
    post_embedder = make_face_embedder(cfg, kind="post")

    out_dir = repo_root / OUTPUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ApifyClient(token)
    log.info("apify_profile_fetch", username=username)
    run = client.actor("apify/instagram-profile-scraper").call(
        run_input={"usernames": [username]},
    )
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    if not items:
        print(f"No dataset items returned for @{username}.", file=sys.stderr)
        return 1

    p = items[0]
    if p.get("username") and p["username"].lower() != username.lower():
        log.warning(
            "username_mismatch",
            requested=username,
            returned=p.get("username"),
        )

    if p.get("private"):
        print(f"Profile @{username} is private — cannot download media.", file=sys.stderr)
        return 1

    avatar_url = p.get("profilePicUrlHD") or p.get("profilePicUrl")
    uid = p.get("id") or p.get("pk")
    uid_str = str(uid) if uid else None

    avatar_path_str = download_avatar(
        avatar_url,
        user_id=uid_str,
        username=username,
    )
    if not avatar_path_str:
        print("Avatar download failed (missing URL or HTTP error).", file=sys.stderr)
        return 1

    avatar_path = Path(avatar_path_str)
    avatar_faces = avatar_embedder.embed_faces(avatar_path)
    faces_count = len(avatar_faces)
    winner_path: Path | None = None
    source = "avatar"

    avatar_area_ok = False
    if faces_count == 1:
        img_bgr = cv2.imread(str(avatar_path))
        if img_bgr is not None:
            ih, iw = img_bgr.shape[:2]
            area_pct, _, _ = face_bbox_percent_of_image(
                avatar_faces[0].bbox, iw, ih
            )
            avatar_area_ok = area_pct >= min_avatar_face_area_pct

    if faces_count == 1 and avatar_area_ok:
        winner_path = avatar_path
    elif uid_str:
        post_urls = _pick_post_images(
            p.get("latestPosts"),
            limit=fb_limit,
            skip_videos=fb_skip_videos,
        )
        local_paths = download_post_photos(post_urls, user_id=uid_str)
        result = resolve_face_leader(
            local_paths,
            post_embedder,
            min_cluster_size=fb_min_cluster,
            cluster_threshold=fb_threshold,
        )
        if result:
            winner_path = result.photo_path
            source = "post_fallback"
        if not fb_keep_photos:
            cleanup_lead_photos(uid_str, keep=(result.photo_path if result else None))
    else:
        print(
            "Avatar has no single face and profile has no numeric id — "
            "cannot run post fallback.",
            file=sys.stderr,
        )

    if winner_path is None or not winner_path.is_file():
        hint = ""
        if faces_count == 1 and not avatar_area_ok:
            hint = " (one face on avatar but below min_avatar_face_area_pct — post fallback did not yield a winner)"
        elif faces_count != 1 and uid_str:
            hint = " (post fallback did not yield a winner)"
        print(
            f"No suitable single-face winner (avatar faces={faces_count}){hint}.",
            file=sys.stderr,
        )
        return 1

    dest = out_dir / f"{username}_face_winner.jpg"
    shutil.copy2(winner_path, dest)
    print(f"Winner ({source}): {winner_path}")
    print(f"Copied to: {dest}")
    log.info(
        "test_profile_face_pick_done",
        username=username,
        source=source,
        avatar_faces=faces_count,
        dest=str(dest),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
