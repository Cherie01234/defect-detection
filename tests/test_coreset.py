"""Light tests for greedy coreset selection (no torch / CI-safe)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.coreset import greedy_coreset  # noqa: E402


def test_returns_requested_unique_subset():
    feats = np.random.default_rng(0).normal(size=(500, 32))
    idx = greedy_coreset(feats, 50, seed=0)
    assert len(idx) == 50
    assert len(set(idx.tolist())) == 50          # unique
    assert idx.min() >= 0 and idx.max() < 500     # valid indices


def test_returns_all_when_oversized():
    feats = np.random.default_rng(0).normal(size=(20, 8))
    idx = greedy_coreset(feats, 100, seed=0)
    assert sorted(idx.tolist()) == list(range(20))


def test_greedy_covers_clusters_better_than_first_points():
    # Two far-apart clusters; a size-2 greedy coreset should hit both.
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0, scale=0.05, size=(100, 4))
    b = rng.normal(loc=10, scale=0.05, size=(100, 4))
    feats = np.concatenate([a, b], axis=0)
    idx = greedy_coreset(feats, 2, seed=0, projection_dim=0)
    picked = feats[idx]
    # The two picked points should be far apart (one per cluster).
    assert np.linalg.norm(picked[0] - picked[1]) > 5
