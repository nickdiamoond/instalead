# models/

Project-local cache for ML model weights. The folder is intentionally
**not** under `data/` (which is gitignored), so its contents can be
committed and deployed together with the code — no need to re-download
weights on a fresh machine (e.g. when shipping to an Ubuntu server).

## Layout

```
models/
├── buffalo_s/                # InsightFace ArcFace bundle (trimmed)
│   ├── det_500m.onnx         # ~2.5 MB   face detector
│   └── w600k_mbf.onnx        # ~13.6 MB  ArcFace recognition (512-d)
└── mediapipe/
    └── blaze_face_short_range.tflite   # ~230 KB — face counting
```

**Total footprint: ~16 MB** — small enough to commit to git directly
without LFS.

### What about the other InsightFace files?

The upstream `buffalo_s.zip` ships 5 ONNX files, but we only enable two
modules in [`src/face_embedder.py`](../src/face_embedder.py) via
`allowed_modules=["detection", "recognition"]`:

| File | Module | Kept? |
|---|---|---|
| `det_500m.onnx` | detection | ✅ |
| `w600k_mbf.onnx` | recognition (512-d embed) | ✅ |
| `2d106det.onnx` | 2D landmarks | removed (not used) |
| `1k3d68.onnx` | 3D landmarks (~140 MB!) | removed (not used) |
| `genderage.onnx` | gender/age | removed (not used) |

InsightFace scans the folder for `*.onnx` files and only loads those
matching `allowed_modules`, so removing the extras is safe. If a future
feature needs, say, landmarks, delete the folder and let the next run
re-download everything.

## How this folder gets populated

Nothing manual — both [`src/face_embedder.py`](../src/face_embedder.py)
and [`src/face_detector.py`](../src/face_detector.py) point here and
download missing files on first use. Once downloaded, commit them:

```bash
git add models/
git commit -m "chore: vendor face models"
```

**Idempotency:** InsightFace skips download if `models/buffalo_s/`
already exists, so the trimmed 2-file setup is stable — `ensure_loaded`
won't try to restore the deleted ONNX files.

## Upgrading / swapping the InsightFace bundle

- `FaceEmbedder(model_name="buffalo_m")` → downloads `models/buffalo_m/`
  (~105 MB, stronger recognition) alongside `buffalo_s/`.
- `FaceEmbedder(model_name="buffalo_l")` → `models/buffalo_l/` (~330 MB).
- All bundles can coexist; pick whichever gives the best
  accuracy/speed trade-off.

For bundles larger than ~50 MB, prefer Git LFS:

```bash
git lfs install
git lfs track "models/**/*.onnx"
git lfs track "models/**/*.tflite"
git add .gitattributes models/
git commit -m "chore: vendor face models via LFS"
```

## Cross-platform notes

Paths are built via `pathlib.Path` from the project root, so the same
tree works on Windows and Ubuntu. The only platform-specific step is
installing the `insightface` wheel itself — see
[CLAUDE.md](../CLAUDE.md) § Development Commands.
