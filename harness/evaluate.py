"""Scoring, paired comparison and the promotion rule (SPEC 5.3).

The LLM never touches this file.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from harness import config as C


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def _auc_fast(y, p) -> float:
    """Mann-Whitney AUC. ~2x faster than sklearn on 691k rows inside bootstrap loops."""
    r = rankdata(p)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def paired_bootstrap(y, p_new, p_ref, n_boot: int = 200, seed: int = 0):
    """P(new > ref) and the delta distribution, resampling the SAME rows for both.

    Pairing cancels the row-sampling noise that dominates an unpaired comparison, which is
    why a +0.0002 delta is detectable at all on this dataset.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() in (0, len(yb)):
            deltas[b] = 0.0
            continue
        deltas[b] = _auc_fast(yb, p_new[idx]) - _auc_fast(yb, p_ref[idx])
    return float((deltas > 0).mean()), float(deltas.mean()), float(deltas.std(ddof=1))


def noise_floor(scores) -> float:
    """Std of repeated-CV MEANS across partition seeds -- the bar every idea must clear.

    The per-fold spread inside one run is NOT this; it badly overstates how much the mean
    moves under a different partition (SPEC 1.4).
    """
    s = np.asarray(scores, dtype=float)
    if len(s) < 2:
        return C.NOISE_FLOOR_PRIOR
    return float(s.std(ddof=1))


def promote(y, p_new, p_ref, confirm_scores=None, floor=None):
    """Apply SPEC 5.3. Returns (status, detail dict).

    status in {promoted, real-but-invisible, rejected}
    """
    floor = floor if floor is not None else C.NOISE_FLOOR_PRIOR
    a_new, a_ref = auc(y, p_new), auc(y, p_ref)
    p_imp, d_mean, d_std = paired_bootstrap(y, p_new, p_ref)
    delta = a_new - a_ref

    sign_consistent = True
    if confirm_scores is not None and len(confirm_scores) >= 2:
        arr = np.asarray(confirm_scores, dtype=float)
        sign_consistent = bool((arr > 0).all() or (arr < 0).all())

    detail = dict(
        auc_new=a_new, auc_ref=a_ref, delta=delta, p_improve=p_imp,
        delta_boot_mean=d_mean, delta_boot_std=d_std,
        floor=floor, floors=delta / floor if floor else float("nan"),
        sign_consistent=sign_consistent,
    )

    if delta <= 0:
        return "rejected", detail
    if p_imp <= C.MIN_P_IMPROVE:
        return "rejected", detail
    if delta < C.MIN_DELTA_FLOORS * floor:
        return "real-but-invisible", detail
    if not sign_consistent:
        return "real-but-invisible", detail
    return "promoted", detail
