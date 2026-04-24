"""Face-leader resolution from a batch of candidate photos.

Given a handful of already-downloaded local images (typically the last N
posts of an Instagram lead whose avatar did not resolve to exactly one
face), this module:

1. Runs SCRFD + ArcFace on each photo in a single pass. Photos that
   don't have exactly one high-confidence face (``FaceEmbedder``'s
   ``min_det_score`` gate) are discarded.
2. Greedy-clusters the remaining embeddings by cosine similarity.
3. Returns the largest cluster's representative photo iff the cluster
   covers at least ``min_cluster_size`` photos. Otherwise ``None``.

The module is pure glue: it reuses ``FaceEmbedder`` and ``cluster_faces``.
The embeddings are used internally for clustering and discarded
afterwards — downstream only needs the one chosen photo path (it gets
forwarded to the external Sherlock bot).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.face_matcher import FaceInstance, cluster_faces
from src.logger import get_logger

if TYPE_CHECKING:
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

    # Single SCRFD + ArcFace pass per photo: detection and embedding come
    # from the same call. We keep only photos with exactly one face above
    # the embedder's ``min_det_score`` threshold.
    instances: list[FaceInstance] = []
    photos_single_face = 0
    for idx, p in enumerate(photo_paths):
        try:
            embs = face_embedder.embed_faces(p)
        except Exception as e:
            log.warning("face_leader_embed_error", path=str(p), error=str(e))
            continue
        if len(embs) != 1:
            continue
        photos_single_face += 1
        emb = embs[0]
        instances.append(
            FaceInstance(
                embedding=emb.embedding,
                source_idx=idx,
                image_path=p,
                det_score=emb.det_score,
            )
        )

    if not instances:
        log.info(
            "face_leader_no_single_face",
            tried=photos_tried,
        )
        return None

    # Greedy cluster by cosine similarity. Largest first.
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

    # Pick the best-scoring member as the representative photo.
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
