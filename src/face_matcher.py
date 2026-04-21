"""Pure-Python greedy clustering of ArcFace embeddings.

No ML dependencies — only numpy. Works on L2-normalized embeddings, so
cosine similarity reduces to a dot product.

Typical flow:

    instances = [FaceInstance(embedding=emb, source_idx=i, image_path=p) ...]
    clusters = cluster_faces(instances, threshold=0.5)
    dominant = find_dominant_face(per_post_instances, threshold=0.5, min_posts=3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import numpy as np


@dataclass
class FaceInstance:
    """A single detected face tagged with its source (post/image index).

    ``source_idx`` is typically the post index when clustering posts, or a
    plain image index when clustering a flat set of photos. It is used by
    ``find_dominant_face`` to count *distinct* sources a cluster covers.
    """

    embedding: "np.ndarray"          # 512-d L2-normalized
    source_idx: int                  # e.g. post index or image index
    image_path: Path | None = None
    det_score: float = 0.0


@dataclass
class FaceCluster:
    """Running cluster with a centroid kept as a re-normalized mean."""

    centroid: "np.ndarray"
    members: list[FaceInstance] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    def distinct_sources(self) -> set[int]:
        return {m.source_idx for m in self.members}


@dataclass
class DominantFace:
    """Result of ``find_dominant_face`` — a cluster that covers enough sources."""

    cluster: FaceCluster
    post_coverage: int               # number of distinct source_idx in cluster
    total_faces: int                 # total face instances in cluster


def _cos_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity for two L2-normalized vectors (== dot product)."""
    import numpy as np

    return float(np.dot(a, b))


def _update_centroid(cluster: FaceCluster, new_emb: "np.ndarray") -> None:
    """Running mean, re-normalized to unit length."""
    import numpy as np

    n = len(cluster.members)
    combined = (cluster.centroid * n + new_emb) / (n + 1)
    norm = float(np.linalg.norm(combined))
    if norm > 0:
        combined = combined / norm
    cluster.centroid = combined.astype(np.float32)


def cluster_faces(
    instances: Iterable[FaceInstance],
    threshold: float = 0.5,
) -> list[FaceCluster]:
    """Greedy online clustering by cosine similarity.

    For each incoming instance, pick the cluster with the highest centroid
    similarity; join it if similarity > threshold, otherwise start a new
    cluster. Centroids are running means re-normalized to unit length.

    Returns clusters sorted by ``size`` descending (largest first).
    """
    import numpy as np

    clusters: list[FaceCluster] = []

    for inst in instances:
        emb = np.asarray(inst.embedding, dtype=np.float32)

        best: FaceCluster | None = None
        best_sim = -1.0
        for c in clusters:
            sim = _cos_sim(c.centroid, emb)
            if sim > best_sim:
                best_sim = sim
                best = c

        if best is not None and best_sim > threshold:
            best.members.append(inst)
            _update_centroid(best, emb)
        else:
            clusters.append(FaceCluster(centroid=emb.copy(), members=[inst]))

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def find_dominant_face(
    posts_faces: list[list[FaceInstance]],
    threshold: float = 0.5,
    min_posts: int = 3,
) -> DominantFace | None:
    """Flatten per-post faces, cluster, pick the best cluster.

    A cluster qualifies only if it covers at least ``min_posts`` distinct
    ``source_idx`` values. Among qualifying clusters the one with the
    highest post coverage wins (ties broken by total face count).
    """
    flat: list[FaceInstance] = []
    for post_idx, faces in enumerate(posts_faces):
        for f in faces:
            f.source_idx = post_idx
            flat.append(f)

    if not flat:
        return None

    clusters = cluster_faces(flat, threshold=threshold)

    best: DominantFace | None = None
    for c in clusters:
        coverage = len(c.distinct_sources())
        if coverage < min_posts:
            continue
        candidate = DominantFace(
            cluster=c,
            post_coverage=coverage,
            total_faces=c.size,
        )
        if best is None or (
            candidate.post_coverage > best.post_coverage
            or (
                candidate.post_coverage == best.post_coverage
                and candidate.total_faces > best.total_faces
            )
        ):
            best = candidate

    return best
