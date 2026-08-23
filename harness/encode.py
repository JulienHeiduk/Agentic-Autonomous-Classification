"""Out-of-fold target encoding, computed by the harness and handed to plugins as columns.

Target encoding is the highest-value technique on this competition -- tier0's `lgb_te` and
`xgb_te` are the two best experiments in the ledger -- but a plugin cannot build it, because
make_features never receives y. That is deliberate: a plugin computing its own target
statistics would fit them on the same rows it is scored on, and report a CV that does not
survive the leaderboard.

The consequence was that the orchestrator proposed target encoding in roughly 8 of every 10
strategies (measured across gpt-oss:20b and qwen3.5:9b alike) and every one of them was
unimplementable. Forbidding it harder does not work -- it is the correct answer to a dataset
whose lookup keys shift the target rate by 0.22 between adjacent values.

So the harness builds it, using the SAME frozen split the model is scored on, and passes the
result in as ordinary numeric columns. The plugin still never sees y, and the encoding a row
receives is never fitted on that row.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _stats(values: pd.Series, y_part: np.ndarray, prior: float, smoothing: float):
    """Smoothed mean target per category: (sum + prior*m) / (count + m).

    Smoothing is what stops a category seen twice from being trusted like one seen ten
    thousand times; with `notifications_per_day` as a high-cardinality key, plenty of
    categories are rare.
    """
    df = pd.DataFrame({"v": values.to_numpy(), "y": np.asarray(y_part, dtype=float)})
    g = df.groupby("v", dropna=True)["y"].agg(["sum", "count"])
    return (g["sum"] + prior * smoothing) / (g["count"] + smoothing)


def oof_target_encode(train: pd.DataFrame, test: pd.DataFrame, y, splits,
                      cols, smoothing: float = 10.0):
    """Return ({col: train_values}, {col: test_values}) as `te_<col>` numeric columns.

    Train rows are encoded strictly out-of-fold against `splits`; test rows use statistics
    fitted on all of train, which is standard and leak-free because test has no labels.
    Unseen and missing categories fall back to the global prior.
    """
    y = np.asarray(y)
    prior = float(y.mean())
    out_tr, out_te = {}, {}

    for c in cols:
        if c not in train.columns:
            continue
        col_tr = train[c]
        enc = np.full(len(train), np.nan, dtype=float)
        for itr, iva in splits:
            stats = _stats(col_tr.iloc[itr], y[itr], prior, smoothing)
            enc[iva] = col_tr.iloc[iva].map(stats).to_numpy(dtype=float)
        out_tr[f"te_{c}"] = np.where(np.isnan(enc), prior, enc)

        full = _stats(col_tr, y, prior, smoothing)
        enc_te = test[c].map(full).to_numpy(dtype=float) if c in test.columns else None
        if enc_te is None:
            out_te[f"te_{c}"] = np.full(len(test), prior)
        else:
            out_te[f"te_{c}"] = np.where(np.isnan(enc_te), prior, enc_te)

    return out_tr, out_te


def fold_target_encode(train: pd.DataFrame, test: pd.DataFrame, y, itr, iva, cols,
                       smoothing: float = 10.0, inner_folds: int = 5, seed: int = 42):
    """Encodings for ONE outer fold, fitted using that fold's training rows only.

    This must be recomputed per outer fold, and encoding once up front is WRONG even when
    that single pass is itself out-of-fold. Measured, on loop06: a single-pass OOF encoding
    scored CV 0.962796 and LB 0.93729, a -0.0267 residual against the +0.0001 every earlier
    iteration produced.

    The reason: with one pass, a row in fold j is encoded from "everything except fold j",
    which includes fold k. Training the fold-k model on the complement of k therefore feeds
    it features built from fold k's labels. The validation rows are clean, but the training
    rows carry the answer, and the model learns to trust the encoding more than it should.

    Here, for outer fold k: training rows are encoded by an inner out-of-fold split within
    the training rows alone, while the held-out rows and the test set are encoded from all
    the training rows. Nothing fitted on fold k ever reaches the fold-k model.
    """
    y = np.asarray(y)
    itr = np.asarray(itr)
    iva = np.asarray(iva)
    prior = float(y[itr].mean())
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    out_tr, out_te = {}, {}

    for c in cols:
        if c not in train.columns:
            continue
        col = train[c]
        sub, y_sub = col.iloc[itr], y[itr]
        enc = np.full(len(train), np.nan, dtype=float)

        # Held-out and test rows are encoded by ONE inner-fold mapping each, not by a fit
        # over all the training rows. Every row then carries an encoding built from the same
        # number of samples, which is what makes the columns comparable across frames.
        #
        # The standard construction (test gets the full fit) leaves training rows noisier
        # than test rows, and that difference is itself a distribution shift: it measured
        # PSI 0.0537 against a 0.05 gate. Any feature a plugin DERIVES from these columns
        # inherits it -- np.log1p(te_app_opens_per_day) scored 0.0537 too and was rejected,
        # costing ten repair attempts on a plugin that was not wrong. Matching the sample
        # size drops it to 0.0181, while a genuine train/test leak still measures 0.064+.
        #
        # Any inner mapping is safe for the held-out rows: all of them are fitted on subsets
        # of itr, which excludes iva entirely.
        inner_stats = []
        for jtr, jva in inner.split(np.zeros(len(itr)), y_sub):
            st = _stats(sub.iloc[jtr], y_sub[jtr], prior, smoothing)
            enc[itr[jva]] = sub.iloc[jva].map(st).to_numpy(dtype=float)
            inner_stats.append(st)

        rng = np.random.default_rng(seed)

        def _matched(values):
            which = rng.integers(0, len(inner_stats), len(values))
            out = np.empty(len(values), dtype=float)
            for k, st in enumerate(inner_stats):
                m = which == k
                if m.any():
                    v = values[m].map(st).to_numpy(dtype=float)
                    out[m] = np.where(np.isnan(v), prior, v)
            return out

        enc[iva] = _matched(col.iloc[iva])
        out_tr[f"te_{c}"] = np.where(np.isnan(enc), prior, enc)
        out_te[f"te_{c}"] = (_matched(test[c]) if c in test.columns
                             else np.full(len(test), prior))

    return out_tr, out_te


def population_stability(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    """PSI between a train column and its test counterpart, binned on train quantiles.

    train and test here are iid draws from the same generator, so a real feature should
    score ~0. A large value means the plugin built the two frames DIFFERENTLY -- typically
    by fitting a mapping inside make_features, which is called once per frame, so
    `value_counts()` or a fitted encoder silently produces a different mapping for each.

    That is invisible to cross-validation, which only ever sees the train mapping, and it
    is what separated loop06's CV of 0.9628 from its leaderboard score of 0.9373.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    eps = 1e-6
    pa = np.clip(pa, eps, None)
    pb = np.clip(pb, eps, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def train_frequency(train: pd.DataFrame, test: pd.DataFrame, cols):
    """`freq_<col>` fitted on train and applied to BOTH frames.

    Plugins kept writing `df[c].value_counts(normalize=True)` inside make_features, which is
    called once per frame and therefore builds a different mapping for train and for test.
    Supplying the correct version removes the reason to hand-roll a broken one.
    """
    out_tr, out_te = {}, {}
    for c in cols:
        if c not in train.columns:
            continue
        f = train[c].value_counts(normalize=True)
        out_tr[f"freq_{c}"] = train[c].map(f).fillna(0.0).to_numpy(dtype=float)
        if c in test.columns:
            out_te[f"freq_{c}"] = test[c].map(f).fillna(0.0).to_numpy(dtype=float)
        else:
            out_te[f"freq_{c}"] = np.zeros(len(test))
    return out_tr, out_te
