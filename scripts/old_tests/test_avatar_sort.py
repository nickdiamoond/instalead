"""Test: run SCRFD locally on images in a folder and bin them into
``0/``, ``1/``, ``2/``, ... subfolders by detected face count.

Designed as a follow-up to ``scripts/test_avatar_batch.py``. That
script downloads avatars into ``data/avatars_review/<username>.jpg``
and writes ``faces_count`` to the DB. This one then physically
sorts those same files into ``data/avatars_review/0/``,
``data/avatars_review/1/``, ``data/avatars_review/2/``, ... so you
can open each bucket in Explorer / a file browser and visually
confirm SCRFD's verdict — much faster than scrolling a JSON log.

No Apify, no DB, no network — purely local. Safe to re-run: only
files in the source ROOT are processed (already-bucketed subfolders
are left alone). Pass ``--reset`` to flatten the buckets back into
the root before sorting again, e.g. after tweaking ``min_det_score``.

By default the script ``moves`` files (sorting in place is the whole
point); pass ``--copy`` to keep originals in the root.

Detector tuning — important: this script is AVATAR-ONLY, and avatars
from ``download_avatar`` come back as 320x320 thumbnails. SCRFD's
default ``det_size=640`` upscales them and a face that fills the whole
avatar lands at ~560 px on the canvas — bigger than SCRFD's largest
anchor (256 px). Counter-intuitively, large faces drop out at high
``det_size``. We therefore default to ``--det-size 320`` here (native
size, no upscale, faces stay in the anchor sweet spot) — this is
PLAN A from our diagnostic. Override with ``--det-size 640`` to match
the pipeline's behavior, or run ``--reset --det-size N`` to re-sort
with a different value and compare distributions.

Usage:
    python scripts/test_avatar_sort.py                       # sort with det_size=320 (avatars-tuned)
    python scripts/test_avatar_sort.py --det-size 640        # match main pipeline behavior
    python scripts/test_avatar_sort.py --source path/to/dir  # custom folder
    python scripts/test_avatar_sort.py --copy                # keep originals
    python scripts/test_avatar_sort.py --min-score 0.6       # override threshold
    python scripts/test_avatar_sort.py --reset               # flatten then sort
    python scripts/test_avatar_sort.py --limit 20 --yes      # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.face_embedder import FaceEmbedder
from src.logger import get_logger, setup_logging

setup_logging()
log = get_logger("test_avatar_sort")

LOGS_DIR = Path("logs")
DEFAULT_SOURCE = Path("data/avatars_review")
# Mirrors what download_avatar / Instagram's CDN tend to produce. We
# intentionally don't touch other extensions so stray files (like
# accidental .DS_Store, .txt notes the user might drop in) stay put.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Thresholds to explore in the post-run "what-if" sweep. Stepping
# only UP from 0.50 — we've established that lower than 0.50 is
# noise (InsightFace's own SCRFD default is 0.5; we don't override
# it, so we never see detections below that anyway). The point of
# the sweep is to decide whether RAISING the configured threshold
# (currently 0.5) drops any real faces.
SWEEP_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
# Bin width for the score histogram.
HIST_BIN = 0.05
# Top-N lowest-scoring detections to surface as "files at risk if
# threshold is raised" — small enough to skim manually, big enough
# to expose a pattern.
BORDERLINE_TOP_N = 30


def verdict(faces: int) -> str:
    if faces == 0:
        return "no face"
    if faces == 1:
        return "single face"
    return f"{faces} faces"


def faces_at(scores: list[float], threshold: float) -> int:
    """How many detections survive ``threshold``."""
    return sum(1 for s in scores if s >= threshold)


def is_bucket_dir(p: Path) -> bool:
    """A subdir we own — purely numeric name like '0', '1', '12'."""
    return p.is_dir() and p.name.isdigit()


def list_root_images(source: Path) -> list[Path]:
    """Image files directly under ``source`` — does NOT recurse into
    bucket subfolders, so re-running the script doesn't keep bouncing
    already-sorted files between buckets.
    """
    return sorted(
        f for f in source.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


def flatten_buckets(source: Path) -> tuple[int, int]:
    """Move every image out of bucket subfolders back into ``source``.

    Used by ``--reset`` to undo a previous sort run before re-sorting
    with a different threshold. Returns ``(moved, overwrites)``.
    Filename collisions (same name already in root) are resolved by
    overwriting — the bucketed copy is treated as the more recent /
    authoritative version, since the root copy could be a stale leftover.
    """
    moved = 0
    overwrites = 0
    for sub in source.iterdir():
        if not is_bucket_dir(sub):
            continue
        for f in sub.iterdir():
            if not (f.is_file() and f.suffix.lower() in IMAGE_EXTS):
                continue
            target = source / f.name
            if target.exists():
                overwrites += 1
                target.unlink()
            shutil.move(str(f), str(target))
            moved += 1
        try:
            sub.rmdir()
        except OSError:
            # Bucket dir not empty (user dropped something else in there) —
            # leave it alone, no harm done.
            pass
    return moved, overwrites


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locally bin review-folder images into N-faces buckets"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Folder to sort. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into buckets instead of moving them. "
             "Default is to MOVE — sorting in place is the whole point.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Before sorting, flatten existing 0/, 1/, ... subfolders "
             "back into the source root. Useful when re-sorting with a "
             "different --min-score.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Override SCRFD min_det_score (default: from config.yaml's "
             "face_detection.min_det_score).",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=None,
        help="SCRFD input canvas size (square). Default: from config.yaml's "
             "face_detection.avatar_det_size (currently 320 — matches the "
             "native size of downloaded Instagram avatars: no upscale, big "
             "'face fills frame' avatars land in SCRFD's anchor sweet spot). "
             "Override with 640 to match the post-photo path of the main "
             "pipeline, or 1024+ if you suspect tiny-face misses.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on images to process (default: 0 = all).",
    )
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt.")
    args = parser.parse_args()

    source: Path = args.source
    if not source.exists():
        print(f"ERROR: source folder does not exist: {source}")
        sys.exit(1)
    if not source.is_dir():
        print(f"ERROR: source is not a directory: {source}")
        sys.exit(1)

    cfg = load_config()
    fd_cfg = cfg.get("face_detection") or {}
    cfg_score = float(fd_cfg.get("min_det_score", 0.6))
    cfg_det_size = int(fd_cfg.get("avatar_det_size", 320))
    min_score = float(args.min_score) if args.min_score is not None else cfg_score
    det_size = int(args.det_size) if args.det_size is not None else cfg_det_size

    flatten_moved = 0
    flatten_overwrites = 0
    if args.reset:
        flatten_moved, flatten_overwrites = flatten_buckets(source)

    images = list_root_images(source)
    if args.limit > 0:
        images = images[: args.limit]

    print(f"\n{'='*60}")
    print("Local face-count sort")
    print(f"{'='*60}")
    print(f"Source:                 {source}")
    print(f"Min det score:          {min_score}")
    print(f"Det size:               {det_size}x{det_size}"
          + (" (avatar-tuned, native 320x320 — no upscale)" if det_size == 320
             else " (matches post-photo pipeline)" if det_size == 640
             else ""))
    print(f"Action:                 {'COPY' if args.copy else 'MOVE'}")
    if args.reset:
        msg = f"{flatten_moved} files flattened to root"
        if flatten_overwrites:
            msg += f" ({flatten_overwrites} overwrites)"
        print(f"Reset:                  {msg}")
    else:
        print("Reset:                  skipped (pass --reset to re-bucket)")
    cap_note = "" if args.limit == 0 else f" (capped at --limit {args.limit})"
    print(f"Images in root:         {len(images)}{cap_note}")
    print(f"{'='*60}")
    if not images:
        print("Nothing to sort.")
        return
    if not args.yes:
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    print(f"\nLoading SCRFD model "
          f"(min_det_score={min_score}, det_size={det_size})...")
    # We deliberately initialize with min_det_score=0 so embed_faces
    # returns every detection SCRFD itself didn't drop. The configured
    # ``min_score`` is applied post-hoc per file (sets the bucket) and
    # the captured per-detection scores feed the threshold sweep below.
    embedder = FaceEmbedder(
        min_det_score=0.0,
        det_size=(det_size, det_size),
    )
    t0 = time.perf_counter()
    embedder._ensure_loaded()
    print(f"Model loaded in {(time.perf_counter() - t0) * 1000:.1f} ms\n")

    print(f"{'#':>4}  {'file':<42} {'faces':>5}  {'ms':>6}  bucket  "
          f"{'top scores'}")
    print("-" * 90)

    face_counts: Counter[int] = Counter()
    detect_times: list[float] = []
    moved_count = 0
    overwritten = 0
    per_file: list[dict] = []
    # Flat list of every detection's score across every file. Drives
    # the histogram + summary stats at the end.
    all_scores: list[float] = []

    total_start = time.perf_counter()
    for idx, img_path in enumerate(images, 1):
        t = time.perf_counter()
        # embed_faces with min_det_score=0 returns every detection SCRFD
        # produced. Missing / unreadable / SCRFD-crash cases return [],
        # i.e. the file ends up in the "0" bucket — the user spots
        # genuinely corrupt files on visual review.
        raw = embedder.embed_faces(img_path)
        elapsed_ms = (time.perf_counter() - t) * 1000
        detect_times.append(elapsed_ms)

        # Sort high-to-low so the printed "top scores" cell shows the
        # most confident detections first; all_scores keeps full data
        # for stats.
        scores = sorted((float(f.det_score) for f in raw), reverse=True)
        all_scores.extend(scores)
        faces = faces_at(scores, min_score)
        face_counts[faces] += 1

        bucket_name = str(faces)
        bucket_dir = source / bucket_name
        bucket_dir.mkdir(exist_ok=True)
        target = bucket_dir / img_path.name
        if target.exists():
            overwritten += 1
            target.unlink()

        if args.copy:
            shutil.copy2(str(img_path), str(target))
        else:
            shutil.move(str(img_path), str(target))
        moved_count += 1

        # Truncate from the LEFT — usernames are usually distinctive at
        # the end, while a long shared prefix would just push the
        # interesting part out of view.
        display_name = (img_path.name if len(img_path.name) <= 42
                        else "..." + img_path.name[-39:])
        # Show up to 3 scores; "-" if the image had none above SCRFD's
        # internal threshold (i.e. likely a logo / blank / 0-bucket case).
        scores_str = (", ".join(f"{s:.2f}" for s in scores[:3])
                      if scores else "-")
        print(f"{idx:>4}  {display_name:<42} {faces:>5}  "
              f"{elapsed_ms:>6.1f}  {bucket_name:>6}  {scores_str}")

        per_file.append({
            "filename": img_path.name,
            "faces": faces,
            "bucket": bucket_name,
            "detect_ms": round(elapsed_ms, 2),
            "det_scores": [round(s, 4) for s in scores],
        })

    embedder.close()
    total_time = time.perf_counter() - total_start

    print("-" * 78)
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Source:                 {source}")
    print(f"Files processed:        {len(images)}")
    print(f"  {('copied' if args.copy else 'moved'):<22}{moved_count}")
    print(f"  overwrote in bucket:  {overwritten}")
    print("\nFace count distribution:")
    for count in sorted(face_counts):
        print(f"  {count} faces: {face_counts[count]:>4}  "
              f"({verdict(count)})")
    if detect_times:
        print("\nDetection timing:")
        print(f"  avg:     {sum(detect_times) / len(detect_times):.1f} ms")
        print(f"  min/max: {min(detect_times):.1f} / "
              f"{max(detect_times):.1f} ms")
    print(f"\nTotal time:             {total_time:.1f}s")
    print(f"\nBuckets created under:  {source}")
    for count in sorted(face_counts):
        n = face_counts[count]
        path_str = str(source / str(count))
        print(f"  {path_str:<42}  {n} files  ({verdict(count)})")

    # ----- det_score statistics & threshold sweep --------------------
    score_stats: dict | None = None
    histogram: list[dict] = []
    sweep_dist: dict[str, dict[str, int]] = {}
    borderline_payload: list[dict] = []

    if all_scores:
        sorted_scores = sorted(all_scores)
        n = len(sorted_scores)

        def pct(p: float) -> float:
            # Linear-interp percentile; small enough sample that exact
            # method doesn't matter, this is just for human reading.
            if n == 1:
                return sorted_scores[0]
            k = (n - 1) * p
            lo = int(k)
            hi = min(lo + 1, n - 1)
            frac = k - lo
            return sorted_scores[lo] * (1 - frac) + sorted_scores[hi] * frac

        score_stats = {
            "count": n,
            "mean": round(sum(sorted_scores) / n, 4),
            "min": round(sorted_scores[0], 4),
            "max": round(sorted_scores[-1], 4),
            "p10": round(pct(0.10), 4),
            "p25": round(pct(0.25), 4),
            "p50": round(pct(0.50), 4),
            "p75": round(pct(0.75), 4),
            "p90": round(pct(0.90), 4),
        }

        # Histogram in 5% bins from 0.50 to 1.00 — anything <0.50 can't
        # appear since SCRFD's internal default threshold gates it.
        hist_lo = 0.50
        hist_hi = 1.00
        bin_count = round((hist_hi - hist_lo) / HIST_BIN)
        bins = [0] * bin_count
        for s in sorted_scores:
            if s < hist_lo:
                continue
            idx = min(int((s - hist_lo) / HIST_BIN), bin_count - 1)
            bins[idx] += 1
        max_bin = max(bins) if bins else 0
        bar_unit = max(max_bin // 50, 1)  # roughly 50 cols at the peak

        print(f"\n{'='*60}")
        print("Det score statistics (across all detected faces)")
        print(f"{'='*60}")
        print(f"Total detections:       {n}")
        print(f"Mean / median:          {score_stats['mean']:.3f} / "
              f"{score_stats['p50']:.3f}")
        print(f"Min / max:              {score_stats['min']:.3f} / "
              f"{score_stats['max']:.3f}")
        print(f"Percentiles (p10/25/50/75/90):  "
              f"{score_stats['p10']:.3f} / {score_stats['p25']:.3f} / "
              f"{score_stats['p50']:.3f} / {score_stats['p75']:.3f} / "
              f"{score_stats['p90']:.3f}")

        print("\nScore histogram (bin width 0.05):")
        for i, count in enumerate(bins):
            lo = hist_lo + i * HIST_BIN
            hi = lo + HIST_BIN
            bracket = "]" if i == bin_count - 1 else ")"  # close last bin
            pctv = (count / n) * 100 if n else 0.0
            bar = "#" * (count // bar_unit) if count else ""
            histogram.append({
                "lo": round(lo, 2),
                "hi": round(hi, 2),
                "count": count,
                "pct": round(pctv, 2),
            })
            print(f"  [{lo:.2f}, {hi:.2f}{bracket}  {count:>5}  "
                  f"({pctv:>5.1f}%)  {bar}")

        # Threshold sweep: how does the per-file face-count distribution
        # shift if we raise the threshold from min_score to each value
        # in SWEEP_THRESHOLDS. Counts are derived from the captured
        # per-file det_scores — no second model pass.
        print(f"\nThreshold sweep (raising min_det_score; "
              f"current = {min_score}):")
        observed_counts = sorted(face_counts.keys() | {0, 1, 2})
        head_counts = [c for c in observed_counts if c <= 3]
        plus_label = "3+"
        header = (f"  {'thr':<6} "
                  + "  ".join(f"{c}f" for c in head_counts)
                  + f"  {plus_label}")
        print(header)
        for t in SWEEP_THRESHOLDS:
            dist: Counter[int] = Counter()
            for f in per_file:
                dist[faces_at(f["det_scores"], t)] += 1
            row_parts = []
            for c in head_counts:
                row_parts.append(f"{dist.get(c, 0):>3}")
            three_plus = sum(v for k, v in dist.items() if k > max(head_counts))
            row_parts.append(f"{three_plus:>3}")
            mark = " <-- current" if abs(t - min_score) < 1e-9 else ""
            print(f"  {t:<6.2f} " + "  ".join(row_parts) + mark)
            sweep_dist[f"{t:.2f}"] = {str(k): v for k, v in sorted(dist.items())}

        # Borderline files: detections with the lowest scores. If raising
        # the threshold would lose a real face, that file's score is
        # somewhere just above the current threshold.
        all_with_min: list[tuple[float, str]] = []
        for f in per_file:
            if f["det_scores"]:
                # Lowest non-trivial detection per file (or only one).
                # Using the file's MAX is wrong here — we want to know
                # which detection is closest to falling below a raised
                # threshold. So take the smallest score that's still
                # being counted, i.e. the one above min_score with the
                # tightest margin.
                kept = [s for s in f["det_scores"] if s >= min_score]
                if kept:
                    all_with_min.append((min(kept), f["filename"]))
        all_with_min.sort()
        borderline = all_with_min[:BORDERLINE_TOP_N]

        if borderline:
            print(f"\nBorderline files (lowest {len(borderline)} kept "
                  f"det_scores — first to drop if threshold rises):")
            for score, fname in borderline:
                disp = (fname if len(fname) <= 50
                        else "..." + fname[-47:])
                print(f"  {score:.3f}  {disp}")
                borderline_payload.append({
                    "min_kept_score": round(score, 4),
                    "filename": fname,
                })

    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = LOGS_DIR / f"test_avatar_sort_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "min_det_score": min_score,
        "det_size": det_size,
        "action": "copy" if args.copy else "move",
        "reset": args.reset,
        "reset_flattened": flatten_moved,
        "reset_overwrites": flatten_overwrites,
        "limit": args.limit,
        "files_processed": len(images),
        "moved_or_copied": moved_count,
        "overwritten_in_bucket": overwritten,
        "face_counts": {str(k): v for k, v in sorted(face_counts.items())},
        "detect_timing_ms": {
            "avg": (round(sum(detect_times) / len(detect_times), 2)
                    if detect_times else None),
            "min": round(min(detect_times), 2) if detect_times else None,
            "max": round(max(detect_times), 2) if detect_times else None,
        },
        "time_s": round(total_time, 2),
        "score_stats": score_stats,
        "score_histogram": histogram,
        "threshold_sweep": sweep_dist,
        "borderline_files": borderline_payload,
        "files": per_file,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\nJSON log: {json_path}")


if __name__ == "__main__":
    main()
