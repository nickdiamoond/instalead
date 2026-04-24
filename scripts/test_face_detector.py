"""Quick smoke test: run SCRFD (InsightFace) face detection on a local image.

SCRFD is the detector bundled with InsightFace's ``buffalo_s`` pack
(``models/buffalo_s/det_500m.onnx``, already vendored in the repo). We
rely on ``FaceEmbedder`` which handles the ``min_det_score`` filter
internally (default 0.7, override via ``--min-score`` or
``face_detection.min_det_score`` in ``config.yaml``).

The script prints raw detections (everything SCRFD returned before our
threshold) AND filtered detections (after ``min_det_score``), so you can
eyeball how strict the current threshold is on your image.

Usage:
    python scripts/test_face_detector.py                 # defaults to man.jpg
    python scripts/test_face_detector.py path/to.jpg     # any image
    python scripts/test_face_detector.py man.jpg --min-score 0.8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.face_embedder import FaceEmbedder


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Count faces on an image (SCRFD)")
    parser.add_argument(
        "image",
        nargs="?",
        default="man.jpg",
        help="Path to image (default: man.jpg in project root).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=(
            "Override FaceEmbedder.min_det_score for this run. "
            "If omitted, uses config.yaml face_detection.min_det_score "
            "(falls back to 0.7)."
        ),
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: file not found: {image_path.resolve()}")
        sys.exit(1)

    cfg = load_config()
    fd_cfg = cfg.get("face_detection") or {}
    cfg_score = float(fd_cfg.get("min_det_score", 0.7))
    min_score = float(args.min_score) if args.min_score is not None else cfg_score

    size_kb = image_path.stat().st_size / 1024
    print(f"Image:         {image_path.resolve()}")
    print(f"Size:          {size_kb:.1f} KB")
    print(f"min_det_score: {min_score}"
          + (" (from config)" if args.min_score is None else " (from --min-score)"))
    print("=" * 60)
    print("Loading SCRFD (InsightFace buffalo_s)...")

    # Primary embedder — applies the configured threshold.
    embedder = FaceEmbedder(min_det_score=min_score)

    t0 = time.perf_counter()
    embedder._ensure_loaded()
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = embedder.embed_faces(image_path)
    detect_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    faces = embedder.embed_faces(image_path)
    detect_warm = time.perf_counter() - t0

    # Second pass with threshold=0 to see every raw SCRFD detection
    # (including ones we just filtered out). Re-uses the loaded model.
    embedder.min_det_score = 0.0
    raw_faces = embedder.embed_faces(image_path)
    embedder.min_det_score = min_score

    embedder.close()

    print("=" * 60)
    print(f"Model load time:     {_fmt_ms(load_time)}")
    print(f"Detection (cold):    {_fmt_ms(detect_cold)}   # det + ArcFace")
    print(f"Detection (warm):    {_fmt_ms(detect_warm)}   # det + ArcFace")
    print("=" * 60)
    print(f"Raw detections (all scores):       {len(raw_faces)}")
    print(f"After filter (score >= {min_score}):  {len(faces)}")

    if raw_faces:
        print("-" * 60)
        print("Per-face details (sorted by det_score desc):")
        for i, f in enumerate(sorted(raw_faces, key=lambda x: -x.det_score), 1):
            x1, y1, x2, y2 = f.bbox
            w, h = x2 - x1, y2 - y1
            kept = "KEEP" if f.det_score >= min_score else "drop"
            print(
                f"  #{i}  score={f.det_score:.3f}  "
                f"bbox=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  "
                f"size={w:.0f}x{h:.0f}  [{kept}]"
            )

    print("=" * 60)
    n = len(faces)
    if n == 0:
        print("Verdict: no face (logo / scenery / blurred / default avatar)")
    elif n == 1:
        print("Verdict: single face - valid Sherlock-bot candidate")
    else:
        print(f"Verdict: {n} faces - group photo, skip for Sherlock")


if __name__ == "__main__":
    main()
