"""Face-leader resolution from a batch of candidate photos.

Given a handful of already-downloaded local images (typically the last N
posts of an Instagram lead whose avatar did not resolve to exactly one
face), this module:

1. Keeps only photos with exactly one face (MediaPipe count).
2. Extracts an ArcFace 512-d embedding for each kept photo.
3. Greedy-clusters the embeddings by cosine similarity.
4. Returns the largest cluster's representative photo iff the cluster
   covers at least ``min_cluster_size`` photos. Otherwise ``None``.

The module is pure glue: it reuses ``FaceDetector``, ``FaceEmbedder`` and
``cluster_faces``. No new ML logic. The embeddings are used internally
for clustering and discarded afterwards — downstream only needs the one
chosen photo path (it gets forwarded to the external Sherlock bot).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.face_matcher import FaceInstance, cluster_faces
from src.logger import get_logger

if TYPE_CHECKING:
    from src.face_detector import FaceDetector
    from src.face_embedder import FaceEmbedder

log = get_logger("face_leader")


@dataclass
class LeaderResult:
    """Winner of the leader-resolution step.

    Only ``photo_path`` is needed by the pipeline; the remaining fields
    exist for dev/observability in the test script's JSON log.
    """

    photo_path: Path
    det_score: float
    cluster_size: int             # photos in the leader cluster
    photos_tried: int             # total photos we attempted
    photos_single_face: int       # photos that passed the single-face filter


def resolve_face_leader(
    photo_paths: list[Path],
    face_detector: "FaceDetector",
    face_embedder: "FaceEmbedder",
    *,
    min_cluster_size: int,
    cluster_threshold: float = 0.5,
) -> LeaderResult | None:
    """Pick the dominant face across ``photo_paths``.

    Returns ``None`` if no cluster covers at least ``min_cluster_size``
    distinct photos — i.e. "no unambiguous leader, skip this lead".
    """
    photos_tried = len(photo_paths)
    if photos_tried == 0:
        log.info("face_leader_no_photos")
        return None

    # Step 1: keep only single-face photos (MediaPipe, cheap).
    single_face_paths: list[Path] = []
    for p in photo_paths:
        try:
            n = face_detector.count_faces(p)
        except Exception as e:
            log.warning("face_leader_detect_error", path=str(p), error=str(e))
            continue
        if n == 1:
            single_face_paths.append(p)

    photos_single_face = len(single_face_paths)
    if photos_single_face == 0:
        log.info(
            "face_leader_no_single_face",
            tried=photos_tried,
        )
        return None

    # Step 2: ArcFace embed each kept photo. One FaceInstance per photo.
    instances: list[FaceInstance] = []
    for idx, p in enumerate(single_face_paths):
        embs = face_embedder.embed_faces(p)
        if not embs:
            continue
        # We already filtered to single-face photos, but the ArcFace
        # detector may disagree with MediaPipe on edge cases — always
        # take the highest-score detection to be safe.
        best = max(embs, key=lambda e: e.det_score)
        instances.append(
            FaceInstance(
                embedding=best.embedding,
                source_idx=idx,
                image_path=p,
                det_score=best.det_score,
            )
        )

    if not instances:
        log.info(
            "face_leader_no_embeddings",
            tried=photos_tried,
            single_face=photos_single_face,
        )
        return None

    # Step 3: greedy cluster by cosine similarity. Largest first.
    clusters = cluster_faces(instances, threshold=cluster_threshold)
    leader = clusters[0]

    if leader.size < min_cluster_size:
        log.info(
            "face_leader_below_threshold",
            tried=photos_tried,
            single_face=photos_single_face,
            largest_cluster=leader.size,
            min_required=min_cluster_size,
        )
        return None

    # Step 4: pick the best-scoring member as the representative photo.
    rep = max(leader.members, key=lambda m: m.det_score)
    assert rep.image_path is not None  # we always set image_path above

    log.info(
        "face_leader_resolved",
        tried=photos_tried,
        single_face=photos_single_face,
        cluster_size=leader.size,
        det_score=rep.det_score,
        photo=str(rep.image_path),
    )

    return LeaderResult(
        photo_path=rep.image_path,
        det_score=rep.det_score,
        cluster_size=leader.size,
        photos_tried=photos_tried,
        photos_single_face=photos_single_face,
    )
