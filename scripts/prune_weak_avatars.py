"""Delete avatars without confident faces and sync DB paths.

Scans ``data/avatars`` and runs SCRFD detection with the same calibration
as the main pipeline for avatars:
  - threshold: ``face_detection.min_det_score`` from ``config.yaml``
  - canvas: ``face_detection.avatar_det_size``

Every image with zero faces above the configured threshold is deleted.
After deletion the script calls ``LeadDB.null_missing_photo_paths()`` to
clear stale ``avatar_path`` / ``face_photo_path`` references in SQLite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import LeadDB
from src.face_embedder import make_face_embedder


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
            "Prune avatars with no faces above configured threshold and "
            "sync SQLite photo paths."
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
        f"(min_det_score={min_score}, avatar_det_size={avatar_det_size})"
    )

    embedder = make_face_embedder(cfg, kind="avatar")
    to_delete: list[Path] = []
    kept = 0
    total = len(files)
    for idx, path in enumerate(files, start=1):
        faces_count = embedder.count_faces(path)
        print(
            f"[{idx}/{total}] {path.name} -> faces_count={faces_count}"
        )
        if faces_count < 1:
            to_delete.append(path)
        else:
            kept += 1
    embedder.close()

    if not to_delete:
        print("Nothing to delete: every avatar has at least one confident face.")
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
