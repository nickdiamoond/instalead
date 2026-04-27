"""Quick smoke test: run SCRFD (InsightFace) face detection on a local image.

SCRFD is the detector bundled with InsightFace's ``buffalo_s`` pack
(``models/buffalo_s/det_500m.onnx``, already vendored in the repo). We
rely on ``FaceEmbedder`` which handles the ``min_det_score`` filter
internally (default 0.7, override via ``--min-score`` or
``face_detection.min_det_score`` in ``config.yaml``).

The script prints raw detections (everything SCRFD returned before our
threshold) AND filtered detections (after ``min_det_score``), so you can
eyeball how strict the current threshold is on your image.

For diagnosing recall problems (SCRFD missing "obvious" faces), use
``--sweep`` to try the same image under several ``det_size`` values.
Background: SCRFD has fixed anchor scales — at ``det_size=640`` a
face filling a 320x320 Instagram avatar gets upscaled to ~560 px on
the canvas, bigger than the largest anchor (256 px), and falls out.
Lowering ``det_size`` to the native input size rescues huge faces;
raising it helps small / distant faces. ``--model buffalo_l`` swaps
in the larger detector pack (``det_10g.onnx``) for broader recall —
auto-downloads on first use into ``models/buffalo_l/`` (~300 MB).

This script defaults to ``--det-size 320`` (matches Instagram avatar
size) since that's the most common diagnostic target. Pass
``--det-size 640`` to inspect what the main pipeline currently does
on a given image.

Usage:
    python scripts/test_face_detector.py                       # defaults to man.jpg, det_size=320
    python scripts/test_face_detector.py path/to.jpg           # any image
    python scripts/test_face_detector.py man.jpg --min-score 0.8
    python scripts/test_face_detector.py face.jpg --det-size 640    # match pipeline
    python scripts/test_face_detector.py face.jpg --sweep      # try multiple det_size
    python scripts/test_face_detector.py face.jpg --model buffalo_l --sweep
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


# det_size values to try in --sweep mode. Covers the full range of
# typical use: 320 = native Instagram thumbnail (best for huge
# face-fills-frame avatars), 640 = SCRFD default (current pipeline),
# 1024-1600 = upsampling to recover tiny / distant faces in posts.
SWEEP_DET_SIZES = [320, 384, 480, 640, 1024, 1600]


def run_single(
    image_path: Path,
    min_score: float,
    model_name: str,
    det_size: int,
    explicit_min_score: bool,
) -> None:
    """Original single-config detection: print raw + filtered counts
    and per-face bbox / score details. Used when ``--sweep`` is off."""
    print(f"min_det_score: {min_score}"
          + (" (from --min-score)" if explicit_min_score else " (from config)"))
    print(f"model:         {model_name}")
    print(f"det_size:      {det_size}x{det_size}")
    print("=" * 60)
    print(f"Loading SCRFD (InsightFace {model_name})...")

    embedder = FaceEmbedder(
        model_name=model_name,
        det_size=(det_size, det_size),
        min_det_score=min_score,
    )

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


def run_sweep(
    image_path: Path,
    min_score: float,
    model_name: str,
) -> None:
    """Run the same image through several det_size values and print
    a comparison table. Loads the model once and re-prepares with each
    new det_size — much faster than spinning up FaceEmbedder per size,
    since the ONNX session stays warm.
    """
    print(f"min_det_score: {min_score}")
    print(f"model:         {model_name}")
    print(f"det_sizes:     {SWEEP_DET_SIZES}")
    print("=" * 78)
    print(f"Loading SCRFD ({model_name})...")
    embedder = FaceEmbedder(model_name=model_name, min_det_score=min_score)
    t0 = time.perf_counter()
    embedder._ensure_loaded()
    print(f"Model loaded in {_fmt_ms(time.perf_counter() - t0)}\n")

    print(f"{'det_size':<10} {'raw':>5} {'kept':>5}  "
          f"{'top scores (raw, desc)':<35} {'ms':>7}")
    print("-" * 78)

    rows: list[dict] = []
    for size in SWEEP_DET_SIZES:
        # Re-prepare the loaded model with the new det_size. ONNX session
        # is already in memory; this just resets the input shape and
        # rebuilds the anchor grid. Touches an internal API but cheap and
        # the test script is the right place for it.
        assert embedder._app is not None
        embedder._app.prepare(ctx_id=-1, det_size=(size, size))

        # Get raw detections (no threshold) by temporarily zeroing it.
        embedder.min_det_score = 0.0
        t = time.perf_counter()
        raw = embedder.embed_faces(image_path)
        elapsed_ms = (time.perf_counter() - t) * 1000
        embedder.min_det_score = min_score

        kept = [f for f in raw if f.det_score >= min_score]
        top = sorted([f.det_score for f in raw], reverse=True)[:5]
        scores_str = ", ".join(f"{s:.2f}" for s in top) if top else "-"

        print(f"{size:<10} {len(raw):>5} {len(kept):>5}  "
              f"{scores_str:<35} {elapsed_ms:>6.1f}")

        rows.append({
            "det_size": size,
            "raw": len(raw),
            "kept": len(kept),
            "top_scores": top,
            "ms": elapsed_ms,
        })

    embedder.close()

    print("-" * 78)
    print("\nHow to read this:")
    print("  raw  = SCRFD detections at any score (threshold=0)")
    print(f"  kept = detections >= min_det_score ({min_score})")
    print("  If 'raw' is 0 across ALL sizes, the model genuinely doesn't")
    print(f"  see a face here. Try --model buffalo_l for broader recall.")
    print("  If 'raw' >= 1 but 'kept' = 0, the face is found but with low")
    print("  confidence — lower min_det_score in config.yaml.")

    # Sanity-check signal: was there any det_size that actually saw the face?
    best = max(rows, key=lambda r: r["kept"])
    if best["kept"] == 0 and any(r["raw"] > 0 for r in rows):
        weak = max(rows, key=lambda r: r["raw"])
        top_raw = weak["top_scores"][0] if weak["top_scores"] else 0.0
        print(f"\nBest raw signal: det_size={weak['det_size']} "
              f"with top score {top_raw:.2f} (below {min_score}).")
    elif best["kept"] > 0:
        print(f"\nBest config: det_size={best['det_size']}  "
              f"({best['kept']} face(s) kept)")


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
    parser.add_argument(
        "--model",
        choices=["buffalo_s", "buffalo_l"],
        default="buffalo_s",
        help="InsightFace model pack. buffalo_s (default, ~16 MB det) is "
             "what the pipeline uses. buffalo_l (~300 MB total, "
             "auto-downloaded on first use) has stronger recall on hard "
             "cases — profiles, occlusions, extreme face sizes.",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=320,
        help="SCRFD input canvas size (square). Default 320 matches the "
             "native size of Instagram avatars (no upscale; huge 'face "
             "fills frame' avatars stay in SCRFD's anchor sweet spot). "
             "Pass 640 to match the main pipeline; 1024-1600 for tiny / "
             "distant faces in post photos. Ignored in --sweep mode.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=f"Run the same image through det_size values "
             f"{SWEEP_DET_SIZES} and print a comparison table. Use to "
             "diagnose why an 'obvious' face is missed at the default 640.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: file not found: {image_path.resolve()}")
        sys.exit(1)

    cfg = load_config()
    fd_cfg = cfg.get("face_detection") or {}
    cfg_score = float(fd_cfg.get("min_det_score", 0.6))
    min_score = float(args.min_score) if args.min_score is not None else cfg_score

    size_kb = image_path.stat().st_size / 1024
    print(f"Image:         {image_path.resolve()}")
    print(f"Size:          {size_kb:.1f} KB")

    if args.sweep:
        run_sweep(
            image_path=image_path,
            min_score=min_score,
            model_name=args.model,
        )
    else:
        run_single(
            image_path=image_path,
            min_score=min_score,
            model_name=args.model,
            det_size=args.det_size,
            explicit_min_score=args.min_score is not None,
        )


if __name__ == "__main__":
    main()
