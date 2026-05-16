"""Small-scale smoke test for InsightFace + greedy clustering.

Reads every image from the ``facetest/`` folder (jpg/jpeg/png/webp, recursive),
runs ArcFace embedding extraction on each, clusters the resulting faces by
cosine similarity, and prints:

- per-image face count + detection time
- per-cluster grouping (how many photos belong to the same person)
- timing breakdown (model load, average embed time, total)

Usage:
    python scripts/test_face_matcher.py
    python scripts/test_face_matcher.py --threshold 0.45
    python scripts/test_face_matcher.py --model buffalo_l
    python scripts/test_face_matcher.py --dir some/other/folder

This is a dev test — nothing is written to the DB.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_embedder import FaceEmbedder
from src.face_matcher import FaceInstance, cluster_faces

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.2f} s"


def _collect_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster faces in a small folder (dev smoke test)."
    )
    parser.add_argument(
        "--dir",
        default="facetest",
        help="Folder with images (default: facetest/ in project root).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine-similarity threshold for greedy clustering (default 0.5).",
    )
    parser.add_argument(
        "--model",
        default="buffalo_s",
        help="InsightFace model bundle (buffalo_s/m/l). Default: buffalo_s.",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Detection input size (square). Default: 640.",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    images = _collect_images(root)

    print(f"Folder:     {root.resolve()}")
    print(f"Model:      {args.model}")
    print(f"Threshold:  {args.threshold}")
    print(f"Images:     {len(images)}")
    print("=" * 60)

    if not images:
        print(f"ERROR: no images found in {root.resolve()}")
        print("Put some jpg/png files into facetest/ and re-run.")
        sys.exit(1)

    embedder = FaceEmbedder(model_name=args.model, det_size=(args.det_size, args.det_size))

    print("Loading InsightFace model (first run may download ~100 MB)...")
    t0 = time.perf_counter()
    embedder._ensure_loaded()
    load_time = time.perf_counter() - t0
    print(f"Model load:         {_fmt_s(load_time)}")
    print("=" * 60)

    instances: list[FaceInstance] = []
    per_image_times: list[float] = []
    total_embed_t0 = time.perf_counter()

    for idx, path in enumerate(images):
        t0 = time.perf_counter()
        faces = embedder.embed_faces(path)
        dt = time.perf_counter() - t0
        per_image_times.append(dt)

        rel = path.relative_to(root) if path.is_relative_to(root) else path
        print(f"[{idx + 1:>3}/{len(images)}] {str(rel):<40} "
              f"faces={len(faces)}  {_fmt_ms(dt)}")

        for f in faces:
            instances.append(
                FaceInstance(
                    embedding=f.embedding,
                    source_idx=idx,
                    image_path=path,
                    det_score=f.det_score,
                )
            )

    total_embed_time = time.perf_counter() - total_embed_t0

    print("=" * 60)
    t0 = time.perf_counter()
    clusters = cluster_faces(instances, threshold=args.threshold)
    cluster_time = time.perf_counter() - t0

    total_faces = len(instances)
    singletons = sum(1 for c in clusters if c.size == 1)
    multi = [c for c in clusters if c.size > 1]

    print("Clustering results")
    print("-" * 60)
    print(f"Total faces detected:  {total_faces}")
    print(f"Total clusters:        {len(clusters)}")
    print(f"Unique singletons:     {singletons}  (faces seen only once)")
    print(f"Multi-face clusters:   {len(multi)}")
    print()

    if multi:
        print("People appearing on multiple photos:")
        for ci, c in enumerate(multi, start=1):
            distinct_imgs = len(c.distinct_sources())
            counts: dict[Path, int] = {}
            for m in c.members:
                if m.image_path is None:
                    continue
                counts[m.image_path] = counts.get(m.image_path, 0) + 1
            print(f"  person #{ci}: {c.size} faces across "
                  f"{distinct_imgs} image(s)")
            for img_path in sorted(counts, key=str):
                rel = (img_path.relative_to(root)
                       if img_path.is_relative_to(root) else img_path)
                n = counts[img_path]
                suffix = f"  (x{n})" if n > 1 else ""
                print(f"      - {rel}{suffix}")
    else:
        print("No person was found on more than one photo.")

    print()
    print("=" * 60)
    print("Timing summary")
    print("-" * 60)
    print(f"Model load:         {_fmt_s(load_time)}")
    if per_image_times:
        avg = sum(per_image_times) / len(per_image_times)
        print(f"Embed per image:    avg {_fmt_ms(avg)}  "
              f"(min {_fmt_ms(min(per_image_times))}, "
              f"max {_fmt_ms(max(per_image_times))})")
    print(f"Embed total:        {_fmt_s(total_embed_time)}")
    print(f"Clustering:         {_fmt_ms(cluster_time)}")
    print(f"Total (incl. load): "
          f"{_fmt_s(load_time + total_embed_time + cluster_time)}")
    print("=" * 60)

    embedder.close()


if __name__ == "__main__":
    main()
