"""Delete avatars without confident faces and sync DB paths.

Scans ``data/avatars`` and runs SCRFD detection with the same calibration
as the main pipeline for avatars:
  - threshold: ``face_detection.min_det_score`` from ``config.yaml``
  - canvas: ``face_detection.avatar_det_size``
  - minimum face bbox area vs full image: ``face_detection.min_avatar_face_area_pct``

Deletes every image that has zero faces above the threshold, or exactly
one face whose bounding box covers less than ``min_avatar_face_area_pct``
percent of the raster (same rule as Step 4 / ``scripts/pipeline.py``).

After deletion the script calls ``LeadDB.null_missing_photo_paths()`` to
clear stale ``avatar_path`` / ``face_photo_path`` references in SQLite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import LeadDB
from src.face_embedder import make_face_embedder

# Matches ``scripts/pipeline.DEFAULT_MIN_AVATAR_FACE_AREA_PCT`` when YAML omits it.
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


def _load_face_cfg(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def _iter_avatar_files(avatars_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(
        p for p in avatars_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prune avatars with no confident face, or a single face smaller "
            "than min_avatar_face_area_pct; sync SQLite photo paths."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--db-path",
        default="data/leads.db",
        help="Path to SQLite DB (default: data/leads.db)",
    )
    parser.add_argument(
        "--avatars-dir",
        default="data/avatars",
        help="Avatar directory to scan (default: data/avatars)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be deleted and DB-updated.",
    )
    args = parser.parse_args()

    cfg = _load_face_cfg(Path(args.config))
    fd_cfg = cfg.get("face_detection") or {}
    min_score = float(fd_cfg.get("min_det_score", 0.6))
    avatar_det_size = int(fd_cfg.get("avatar_det_size", 320))
    min_avatar_face_area_pct = float(
        fd_cfg.get("min_avatar_face_area_pct", DEFAULT_MIN_AVATAR_FACE_AREA_PCT)
    )
    avatars_dir = Path(args.avatars_dir)

    if not avatars_dir.exists():
        print(f"Avatar directory not found: {avatars_dir.resolve()}")
        return

    files = _iter_avatar_files(avatars_dir)
    if not files:
        print(f"No avatar files found in {avatars_dir.resolve()}")
        return

    print(
        f"Scanning {len(files)} avatars "
        f"(min_det_score={min_score}, avatar_det_size={avatar_det_size}, "
        f"min_avatar_face_area_pct={min_avatar_face_area_pct})"
    )

    embedder = make_face_embedder(cfg, kind="avatar")
    to_delete: list[Path] = []
    kept = 0
    total = len(files)
    for idx, path in enumerate(files, start=1):
        embs = embedder.embed_faces(path)
        faces_count = len(embs)
        if faces_count < 1:
            print(f"[{idx}/{total}] {path.name} -> faces_count=0 DELETE")
            to_delete.append(path)
            continue
        if faces_count > 1:
            print(f"[{idx}/{total}] {path.name} -> faces_count={faces_count} KEEP")
            kept += 1
            continue

        # Exactly one face: same area gate as pipeline Step 4 for avatars.
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(
                f"[{idx}/{total}] {path.name} -> faces_count=1 "
                f"(imread failed, cannot measure area) KEEP"
            )
            kept += 1
            continue
        ih, iw = img_bgr.shape[:2]
        area_pct, _, _ = face_bbox_percent_of_image(embs[0].bbox, iw, ih)
        if area_pct < min_avatar_face_area_pct:
            print(
                f"[{idx}/{total}] {path.name} -> faces_count=1 "
                f"area={area_pct:.1f}% < {min_avatar_face_area_pct}% DELETE"
            )
            to_delete.append(path)
        else:
            print(
                f"[{idx}/{total}] {path.name} -> faces_count=1 "
                f"area={area_pct:.1f}% KEEP"
            )
            kept += 1
    embedder.close()

    if not to_delete:
        print(
            "Nothing to delete: every avatar has at least one confident face "
            "and (if single-face) bbox area >= min_avatar_face_area_pct."
        )
        return

    print(f"Will delete {len(to_delete)} avatar(s), keep {kept}.")
    if args.dry_run:
        for p in to_delete:
            print(f"[dry-run] delete {p.as_posix()}")
        print("[dry-run] DB sync skipped.")
        return

    deleted = 0
    failed = 0
    for p in to_delete:
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            failed += 1
            print(f"[warn] failed to delete {p.as_posix()}: {e}")

    db = LeadDB(args.db_path)
    sync_stats = db.null_missing_photo_paths()

    print(f"Deleted: {deleted}, failed: {failed}, kept: {kept}")
    print(
        "DB sync: "
        f"leads_changed={sync_stats['leads_changed']}, "
        f"avatar_path_nulled={sync_stats['avatar_path_nulled']}, "
        f"face_photo_path_nulled={sync_stats['face_photo_path_nulled']}"
    )


if __name__ == "__main__":
    main()
