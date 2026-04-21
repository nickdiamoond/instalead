"""Quick smoke test: run MediaPipe face detection on a local image.

Usage:
    python scripts/test_face_detector.py               # defaults to man.jpg
    python scripts/test_face_detector.py path/to.jpg   # any image
    python scripts/test_face_detector.py man.jpg --confidence 0.3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_detector import FaceDetector


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Count faces on an image")
    parser.add_argument(
        "image",
        nargs="?",
        default="man.jpg",
        help="Path to image (default: man.jpg in project root).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence, 0.0-1.0 (default: 0.5).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: file not found: {image_path.resolve()}")
        sys.exit(1)

    size_kb = image_path.stat().st_size / 1024
    print(f"Image:       {image_path.resolve()}")
    print(f"Size:        {size_kb:.1f} KB")
    print(f"Confidence:  {args.confidence}")
    print(f"{'='*50}")
    print("Loading MediaPipe model...")

    detector = FaceDetector(min_confidence=args.confidence)

    t0 = time.perf_counter()
    detector._ensure_loaded()
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    faces = detector.count_faces(image_path)
    detect_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = detector.count_faces(image_path)
    detect_time_warm = time.perf_counter() - t0

    detector.close()

    print(f"{'='*50}")
    print(f"Model load time:    {_fmt_ms(load_time)}")
    print(f"Detection (cold):   {_fmt_ms(detect_time)}")
    print(f"Detection (warm):   {_fmt_ms(detect_time_warm)}")
    print(f"Total:              {_fmt_ms(load_time + detect_time)}")
    print(f"{'='*50}")
    print(f"Faces detected: {faces}")

    if faces == 0:
        print("Verdict: no face (logo / scenery / blurred / default avatar)")
    elif faces == 1:
        print("Verdict: single face — valid Sherlock-bot candidate")
    else:
        print(f"Verdict: {faces} faces — group photo, skip for Sherlock")


if __name__ == "__main__":
    main()
