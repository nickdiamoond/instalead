"""Local face detection wrapper around MediaPipe Tasks API.

Uses BlazeFace short-range model, optimized for selfie-like images
within ~2 meters — a good fit for Instagram avatars (typically 320x320
or 1080x1080 portraits).

The MediaPipe Tasks API replaced the legacy `mp.solutions` API in
mediapipe >= 0.10.20. Model is a ~230KB .tflite file auto-downloaded
on first use to `data/models/blaze_face_short_range.tflite`.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from src.logger import get_logger

log = get_logger("face_detector")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

# Project-local models dir so the weights travel with the repo
# (and Ubuntu deployments don't need to re-download them).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
MODEL_PATH = MODEL_DIR / "blaze_face_short_range.tflite"

# Legacy location (pre-``models/`` migration): kept for silent migration
# of existing checkouts that already downloaded the weights locally.
_LEGACY_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "blaze_face_short_range.tflite"


def _ensure_model() -> Path:
    """Download the BlazeFace short-range model if missing. Returns path."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _LEGACY_MODEL_PATH.exists() and _LEGACY_MODEL_PATH.stat().st_size > 0:
        shutil.copy(_LEGACY_MODEL_PATH, MODEL_PATH)
        log.info("face_model_migrated",
                 src=str(_LEGACY_MODEL_PATH), dst=str(MODEL_PATH))
        return MODEL_PATH
    log.info("face_model_downloading", url=MODEL_URL, dst=str(MODEL_PATH))
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info("face_model_downloaded", path=str(MODEL_PATH),
             size=MODEL_PATH.stat().st_size)
    return MODEL_PATH


class FaceDetector:
    """Counts faces on local images using MediaPipe Tasks FaceDetector."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
        self._detector = None

    def _ensure_loaded(self) -> None:
        if self._detector is not None:
            return
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_model()
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=self.min_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        log.info("face_detector_loaded", min_confidence=self.min_confidence)

    def count_faces(self, image_path: str | Path) -> int:
        """Return number of detected faces. Returns 0 on any failure."""
        path = Path(image_path)
        if not path.exists() or path.stat().st_size == 0:
            log.warning("face_detect_missing_image", path=str(path))
            return 0

        try:
            import mediapipe as mp

            self._ensure_loaded()
            assert self._detector is not None
            image = mp.Image.create_from_file(str(path))
            result = self._detector.detect(image)
            detections = result.detections or []
            return len(detections)
        except Exception as e:
            log.warning("face_detect_error", path=str(path), error=str(e))
            return 0

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
