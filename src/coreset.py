"""Coreset subsampling for the memory bank.

`greedy_coreset` implements (approximate) k-center greedy selection, as used by
PatchCore: starting from one point, repeatedly add the point that is farthest
from the already-selected set. This yields a small subset that *covers* the
feature space far better than uniform random sampling, so the memory bank can be
shrunk aggressively with little accuracy loss.

The selection order is nested: the first k indices are themselves a valid
k-sized coreset, which makes size/accuracy trade-off sweeps cheap.
"""
from __future__ import annotations

import numpy as np
from sklearn.random_projection import SparseRandomProjection


def greedy_coreset(
    features: np.ndarray,
    n_samples: int,
    seed: int = 0,
    projection_dim: int = 64,
) -> np.ndarray:
    """Return indices of a k-center-greedy coreset (nested by construction)."""
    n = len(features)
    if n_samples >= n:
        return np.arange(n)

    feats = features
    if projection_dim and features.shape[1] > projection_dim:
        # Johnson–Lindenstrauss projection: preserves distances, speeds things up.
        feats = SparseRandomProjection(
            n_components=projection_dim, random_state=seed
        ).fit_transform(features)
    feats = np.ascontiguousarray(feats, dtype=np.float32)

    rng = np.random.default_rng(seed)
    selected = np.empty(n_samples, dtype=np.int64)
    start = int(rng.integers(n))
    selected[0] = start
    min_dist = np.linalg.norm(feats - feats[start], axis=1)

    for i in range(1, n_samples):
        idx = int(np.argmax(min_dist))
        selected[i] = idx
        np.minimum(min_dist, np.linalg.norm(feats - feats[idx], axis=1), out=min_dist)

    return selected
