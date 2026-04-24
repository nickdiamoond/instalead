"""InsightFace wrapper for SCRFD detection + ArcFace embedding.

Wraps ``insightface.app.FaceAnalysis`` and exposes two entry points:

* :py:meth:`FaceEmbedder.count_faces` — just the face count on an image
  (used by the pipeline to decide whether an avatar / post photo has
  exactly one recognizable face).
* :py:meth:`FaceEmbedder.embed_faces` — per-face records with
  L2-normalized 512-d embeddings suitable for cosine-similarity matching.

Both methods apply a post-hoc ``min_det_score`` filter on top of SCRFD's
internal gate: SCRFD's default threshold is ~0.5, which is permissive
enough to pick up low-confidence faces from image backgrounds. We
override with ``0.7`` by default — strict enough to reject background
faces on Instagram post photos while keeping real profile / side-view
detections (observed scores 0.78–0.85 on mixed test data).

The model bundle (default ``buffalo_s``, ~155 MB) is auto-downloaded on
first use into ``<project>/models/<name>/`` — kept inside the repo so the
weights travel with the code (no per-user ``~/.insightface/`` cache,
Ubuntu deploys don't re-download). Detection + embed runs on CPU via
``onnxruntime``.

Usage:
    embedder = FaceEmbedder()
    n = embedder.count_faces(Path("data/avatars/123.jpg"))
    faces = embedder.embed_faces(Path("data/avatars/123.jpg"))
    for f in faces:
        print(f.embedding.shape, f.bbox, f.det_score)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    import numpy as np

log = get_logger("face_embedder")

# Project-local models dir. InsightFace downloads to ``<root>/models/<name>/``,
# so we point ``root`` at the project root and the actual ONNX files end up
# at ``<project>/models/<name>/``. This way models travel with the repo
# (can be git-committed or tracked via Git LFS) instead of being pulled into
# ``~/.insightface/`` on every fresh machine.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_ROOT = PROJECT_ROOT


@dataclass
class FaceEmb:
    """One detected face with its ArcFace embedding."""

    embedding: "np.ndarray"       # shape (512,), float32, L2-normalized
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    det_score: float              # detection confidence 0..1


class FaceEmbedder:
    """Thin wrapper around InsightFace FaceAnalysis (CPU-only).

    The underlying model is created lazily on first call so that simply
    importing this module stays cheap.
    """

    def __init__(
        self,
        model_name: str = "buffalo_s",
        det_size: tuple[int, int] = (640, 640),
        min_det_score: float = 0.7,
        models_root: Path | str | None = None,
    ) -> None:
        self.model_name = model_name
        self.det_size = det_size
        # Post-hoc filter on SCRFD det_score. Anything below this is treated
        # as "not a real face" and dropped before downstream consumers see it.
        # See module docstring for the rationale behind the 0.7 default.
        self.min_det_score = min_det_score
        # ``models_root`` is the directory that *contains* the ``models/``
        # subfolder where InsightFace keeps its ONNX bundles. Defaults to the
        # project root so weights live at ``<project>/models/<name>/``.
        self.models_root = Path(models_root) if models_root else DEFAULT_MODELS_ROOT
        self._app = None

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise RuntimeError(
                "insightface is not installed. Run: pip install insightface onnxruntime"
            ) from e

        self.models_root.mkdir(parents=True, exist_ok=True)
        log.info(
            "face_embedder_loading",
            model=self.model_name,
            models_root=str(self.models_root),
        )
        app = FaceAnalysis(
            name=self.model_name,
            root=str(self.models_root),
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=self.det_size)
        self._app = app
        log.info("face_embedder_loaded", model=self.model_name, det_size=self.det_size)

    def embed_faces(self, image_path: str | Path) -> list[FaceEmb]:
        """Detect + embed every face on the image above ``min_det_score``.

        Returns ``[]`` on any failure (missing file, unreadable image,
        InsightFace crash). Faces below the score threshold are silently
        dropped, so callers can treat ``len(result)`` as the "real" face
        count without a second filter.
        """
        import numpy as np

        path = Path(image_path)
        if not path.exists() or path.stat().st_size == 0:
            log.warning("face_embed_missing_image", path=str(path))
            return []

        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("opencv-python is required for FaceEmbedder") from e

        img = cv2.imread(str(path))
        if img is None:
            log.warning("face_embed_unreadable", path=str(path))
            return []

        try:
            self._ensure_loaded()
            assert self._app is not None
            faces = self._app.get(img)
        except Exception as e:
            log.warning("face_embed_error", path=str(path), error=str(e))
            return []

        out: list[FaceEmb] = []
        for f in faces:
            det_score = float(getattr(f, "det_score", 0.0))
            if det_score < self.min_det_score:
                continue
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
                if emb is None:
                    continue
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    emb = emb / norm
            emb = np.asarray(emb, dtype=np.float32)
            bbox = tuple(float(v) for v in getattr(f, "bbox", (0, 0, 0, 0)))
            if len(bbox) != 4:
                bbox = (0.0, 0.0, 0.0, 0.0)
            out.append(
                FaceEmb(
                    embedding=emb,
                    bbox=bbox,  # type: ignore[arg-type]
                    det_score=det_score,
                )
            )
        return out

    def count_faces(self, image_path: str | Path) -> int:
        """Return the number of faces above ``min_det_score``.

        Thin wrapper around :py:meth:`embed_faces` — currently runs the
        full detection + ArcFace pass and just discards the embeddings.
        The extra ArcFace cost is ~30-50 ms per detected face on CPU,
        negligible for typical Instagram avatars (0-1 faces). If this
        ever shows up in a profile we can split out a detection-only
        ``FaceAnalysis`` instance.
        """
        return len(self.embed_faces(image_path))

    def close(self) -> None:
        self._app = None
