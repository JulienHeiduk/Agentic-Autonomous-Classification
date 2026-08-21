"""The frozen fold split.

    StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)
    over train.csv in ORIGINAL FILE ROW ORDER -- never sorted, never reindexed.

Every public OOF library on this competition aligns to this split, positionally (most ship
.npy with no ids). Regenerating it with different arguments silently invalidates every
stored OOF array and every cross-experiment comparison, so it is hashed and asserted.
"""
import hashlib

import numpy as np
from sklearn.model_selection import StratifiedKFold

from harness import config as C

_FOLD_NPY = C.STATE / "folds.npy"
_FOLD_SHA = C.STATE / "folds.sha256"


def _build(y) -> np.ndarray:
    fold_id = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.FOLD_SEED)
    for k, (_, iva) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_id[iva] = k
    assert (fold_id >= 0).all()
    return fold_id


def get(y) -> np.ndarray:
    """Return the frozen fold_id array, creating and pinning it on first call."""
    fold_id = _build(y)
    sha = hashlib.sha256(fold_id.tobytes()).hexdigest()
    if _FOLD_SHA.exists():
        pinned = _FOLD_SHA.read_text().strip()
        assert sha == pinned, (
            f"FOLD SPLIT CHANGED: {sha[:16]} != pinned {pinned[:16]}. Every stored OOF "
            f"array is now misaligned. Refusing to continue."
        )
    else:
        np.save(_FOLD_NPY, fold_id)
        _FOLD_SHA.write_text(sha)
    return fold_id


def splits(y):
    """Yield (train_idx, valid_idx) for the frozen split."""
    fold_id = get(y)
    idx = np.arange(len(y))
    for k in range(C.N_FOLDS):
        m = fold_id == k
        yield idx[~m], idx[m]


def splits_seeded(y, seed: int):
    """A DIFFERENT partition, for CONFIRM-tier noise-floor measurement only.

    Never use these for stored OOF arrays -- they do not align with the public pool.
    """
    skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
