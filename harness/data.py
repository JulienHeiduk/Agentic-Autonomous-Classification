"""Data loading. Original file row order is load-bearing -- never sort, never reindex."""
import numpy as np
import pandas as pd

from harness import config as C

_cache = {}


def load():
    """Return (train, test, y). train/test keep the competition's original row order."""
    if "d" not in _cache:
        train = pd.read_csv(C.DATA / "train.csv")
        test = pd.read_csv(C.DATA / "test.csv")
        assert len(train) == C.N_TRAIN, f"train rows {len(train)} != {C.N_TRAIN}"
        assert len(test) == C.N_TEST, f"test rows {len(test)} != {C.N_TEST}"
        assert list(train[C.ID]) == sorted(train[C.ID]), "train id order changed"
        y = train[C.TARGET].to_numpy(np.int8)
        _cache["d"] = (train, test, y)
    return _cache["d"]
