"""Runs one generated plugin inside its own process.

The harness owns the fold loop, the scoring and the artifacts. The plugin only supplies a
feature builder and a model factory, so it cannot accidentally score itself, reuse the
target, or reshape the fold split.

Invoked by harness/sandbox.py -- not meant to be run by hand.
Prints a single JSON line on stdout as its last output.
"""
import argparse
import importlib.util
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd

from harness import config as C
from harness import encode, folds, ledger
from harness.data import load
from harness.evaluate import auc


def load_plugin(path):
    spec = importlib.util.spec_from_file_location("plugin_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("make_features", "make_model"):
        if not hasattr(mod, fn):
            raise AttributeError(f"plugin does not define {fn}()")
    return mod


def _fit_predict(model, Xa, ya, Xb, Xt):
    """Fit and return (valid_proba, test_proba). Tolerates estimators without predict_proba."""
    model.fit(Xa, ya)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xb)[:, 1], model.predict_proba(Xt)[:, 1]
    return model.decision_function(Xb), model.decision_function(Xt)


def run(plugin_path, exp_id, n_rows=None, seed=42, partition_seed=None):
    t0 = time.time()
    train, test, y = load()
    if n_rows:                       # screening / preflight mode
        train = train.iloc[:n_rows].reset_index(drop=True)
        test = test.iloc[:n_rows].reset_index(drop=True)
        y = y[:n_rows]

    mod = load_plugin(plugin_path)

    # Splits are resolved BEFORE features are built, because the harness's out-of-fold
    # target encoding must use the same partition the model is scored on -- encoding a row
    # with statistics that saw that row is exactly the leak the contract exists to prevent.
    if partition_seed is not None:
        splits = folds.splits_seeded(y, partition_seed)
    elif n_rows:
        splits = folds.splits_seeded(y, C.FOLD_SEED)
    else:
        splits = list(folds.splits(y))

    def _features_for(tr_df, te_df, check):
        Xa, Xb = mod.make_features(tr_df.copy(), te_df.copy())
        Xa = pd.DataFrame(Xa).reset_index(drop=True)
        Xb = pd.DataFrame(Xb).reset_index(drop=True)
        if check:
            if len(Xa) != len(train):
                raise ValueError(f"make_features returned {len(Xa)} train rows, "
                                 f"expected {len(train)}")
            if len(Xb) != len(test):
                raise ValueError(f"make_features returned {len(Xb)} test rows, "
                                 f"expected {len(test)}")
            if list(Xa.columns) != list(Xb.columns):
                raise ValueError("make_features: train and test columns differ")
            if C.TARGET in Xa.columns:
                raise ValueError(f"make_features leaked '{C.TARGET}' into the feature frame")
        return Xa, Xb

    # NOTE: a partition_seed run deliberately does NOT use the frozen split, so its OOF is
    # not stackable and is never stored. It measures the noise floor only (SPEC 5.2).
    #
    # Features are rebuilt PER FOLD because the target encoding is fitted per fold. Encoding
    # once up front and reusing it leaks fold k's labels into fold k's training rows -- that
    # bug scored CV 0.9628 against LB 0.9373. Rebuilding costs one make_features call per
    # fold, which is seconds against a fit measured in minutes.
    oof = np.zeros(len(y))
    pred = np.zeros(len(test))
    per_fold = []
    feat_s = 0.0
    n_features = 0
    for k, (itr, iva) in enumerate(splits):
        t_feat = time.time()
        te_tr, te_te = encode.fold_target_encode(
            train, test, y, itr, iva, C.TE_COLUMNS,
            smoothing=C.TE_SMOOTHING, seed=C.FOLD_SEED)
        fq_tr, fq_te = encode.train_frequency(train, test, C.FREQ_COLUMNS)
        Xtr, Xte = _features_for(train.assign(**te_tr, **fq_tr),
                                 test.assign(**te_te, **fq_te), check=(k == 0))
        if k == 0:
            _check_consistency(Xtr, Xte)
        feat_s += time.time() - t_feat
        n_features = int(Xtr.shape[1])

        m = mod.make_model(seed)
        pv, pt = _fit_predict(m, Xtr.iloc[itr], y[itr], Xtr.iloc[iva], Xte)
        oof[iva] = pv
        pred += np.asarray(pt) / len(splits)
        per_fold.append(auc(y[iva], pv))
        print(f"  fold{k} auc={per_fold[-1]:.6f} ({time.time()-t0:.0f}s)", flush=True)

    a = auc(y, oof)
    if not n_rows and partition_seed is None:
        ledger.save_arrays(exp_id, oof, pred)

    return dict(
        ok=True, exp_id=exp_id, cv_auc=a, cv_std=float(np.std(per_fold)),
        per_fold=per_fold, n_features=n_features,
        feature_seconds=round(feat_s, 1), runtime_s=round(time.time() - t0, 1),
    )



def _check_consistency(Xtr, Xte):
    """Fail a plugin whose two frames were not built the same way.

    make_features is called once for train and once for test, so any mapping FITTED inside
    it -- value_counts, a fitted encoder, a per-frame normalisation -- silently differs
    between the frames. Cross-validation cannot see this, because CV only ever uses the
    train mapping; the leaderboard sees it immediately.

    This is not hypothetical: loop06 scored CV 0.962796 and LB 0.937290 on exactly this
    mistake, a -0.0267 residual where every honest iteration produced +0.0005.
    """
    from harness import encode as _e
    # Columns the harness builds itself are exempt. Out-of-fold target encoding leaves an
    # irreducible train/test difference -- training rows carry inner-fold noise that test
    # rows do not -- which measures in the same PSI range as the bug this gate catches, so
    # PSI cannot tell them apart. These are correct by construction; the gate is here for
    # what the PLUGIN builds.
    # Exemption is by PREFIX, not exact name: a feature derived from a supplied column
    # inherits its distribution difference, so `te_app_opens_per_day_squared` trips the gate
    # for exactly the reason `te_app_opens_per_day` does. `te_` and `freq_` are therefore
    # reserved namespaces -- prompts.CONTRACT tells plugins not to name their own features
    # with them.
    reserved = ("te_", "freq_")
    bad, warn = [], []
    for c in Xtr.columns:
        if isinstance(c, str) and c.startswith(reserved):
            continue
        try:
            psi = _e.population_stability(Xtr[c].to_numpy(), Xte[c].to_numpy())
        except Exception:
            continue
        if psi > C.PSI_FAIL:
            bad.append((c, psi))
        elif psi > C.PSI_WARN:
            warn.append((c, psi))
    for c, psi in warn:
        print(f"  WARNING: '{c}' train/test PSI {psi:.4f} -- check it is built identically "
              f"for both frames", flush=True)
    if bad:
        worst = ", ".join(f"'{c}' (PSI {p:.3f})" for c, p in sorted(bad, key=lambda t: -t[1]))
        raise ValueError(
            f"train/test feature mismatch: {worst}. These columns have different "
            f"distributions in X_train and X_test, which means they were FITTED separately "
            f"inside make_features -- e.g. `df[c].value_counts()` computed per frame. Any "
            f"statistic must be computed on train and applied to BOTH frames. The harness "
            f"already supplies freq_<col> and te_<col> built correctly; use those instead of "
            f"deriving your own."
        )


def focused_error(plugin_path: str) -> str:
    """Build an error message centred on the PLUGIN's own code.

    A raw traceback here is ~90% pandas/sklearn internals, and the one line that actually
    needs changing is buried in the middle. A 7B repairer given that will rewrite
    something at random. Showing it its own failing line, with source, is the difference
    between a repair loop that converges and one that does not.
    """
    exc_type, exc, tb = sys.exc_info()
    frames = traceback.extract_tb(tb)
    mine = [f for f in frames if f.filename == str(plugin_path)]

    parts = [f"{exc_type.__name__}: {exc}", ""]
    if mine:
        last = mine[-1]
        parts += [
            "THE FAILING LINE IN YOUR FILE:",
            f"  line {last.lineno}, in {last.name}():",
            f"      {last.line}",
            "",
        ]
        if len(mine) > 1:
            parts.append("call chain inside your file:")
            for f in mine:
                parts.append(f"  line {f.lineno} in {f.name}(): {f.line}")
            parts.append("")
    else:
        parts += ["(the error was raised outside your file -- most likely your returned "
                  "frames violate the contract: wrong row count, mismatched columns, or a "
                  "non-numeric dtype the model cannot accept)", ""]

    tail = traceback.format_exception(exc_type, exc, tb)[-4:]
    parts += ["raw traceback tail:", *[t.rstrip() for t in tail]]
    return "\n".join(parts)[:3500]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plugin")
    p.add_argument("exp_id")
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--partition-seed", type=int, default=None)
    a = p.parse_args()
    try:
        out = run(a.plugin, a.exp_id, a.rows, partition_seed=a.partition_seed)
    except Exception:
        out = dict(ok=False, exp_id=a.exp_id, error=focused_error(a.plugin))
    print("###RESULT###" + json.dumps(out))


if __name__ == "__main__":
    main()
