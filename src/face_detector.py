"""Local face detection wrapper around MediaPipe.

Uses MediaPipe Face Detection short-range model (model_selection=0),
optimized for selfie-like images within ~2 meters — a good fit for
Instagram avatars (typically 320x320 or 1080x1080 portraits).

MediaPipe and OpenCV are imported lazily so the rest of the codebase
(and tests) does not pay the startup cost unless face detection is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    import mediapipe as mp  # noqa: F401

log = get_logger("face_detector")


class FaceDetector:
    """Counts faces on local images using MediaPipe Face Detection."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
        self._detector = None

    def _ensure_loaded(self) -> None:
        if self._detector is not None:
            return
        import mediapipe as mp

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=self.min_confidence,
        )
        log.info("face_detector_loaded", model=0, min_confidence=self.min_confidence)

    def count_faces(self, image_path: str | Path) -> int:
        """Return number of detected faces. Returns 0 on any failure."""
        path = Path(image_path)
        if not path.exists() or path.stat().st_size == 0:
            log.warning("face_detect_missing_image", path=str(path))
            return 0

        try:
            import cv2

            image = cv2.imread(str(path))
            if image is None:
                log.warning("face_detect_unreadable", path=str(path))
                return 0

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._ensure_loaded()
            assert self._detector is not None
            result = self._detector.process(rgb)
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
