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
from harness import folds, ledger
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

    t_feat = time.time()
    Xtr, Xte = mod.make_features(train.copy(), test.copy())
    Xtr = pd.DataFrame(Xtr).reset_index(drop=True)
    Xte = pd.DataFrame(Xte).reset_index(drop=True)
    if len(Xtr) != len(train):
        raise ValueError(f"make_features returned {len(Xtr)} train rows, expected {len(train)}")
    if len(Xte) != len(test):
        raise ValueError(f"make_features returned {len(Xte)} test rows, expected {len(test)}")
    if list(Xtr.columns) != list(Xte.columns):
        raise ValueError("make_features: train and test columns differ")
    if C.TARGET in Xtr.columns:
        raise ValueError(f"make_features leaked '{C.TARGET}' into the feature frame")
    feat_s = time.time() - t_feat

    # A partition_seed run deliberately does NOT use the frozen split, so its OOF is not
    # stackable and is never stored. It exists only to measure the noise floor (SPEC 5.2).
    if partition_seed is not None:
        splits = folds.splits_seeded(y, partition_seed)
    elif n_rows:
        splits = folds.splits_seeded(y, C.FOLD_SEED)
    else:
        splits = list(folds.splits(y))

    oof = np.zeros(len(y))
    pred = np.zeros(len(Xte))
    per_fold = []
    for k, (itr, iva) in enumerate(splits):
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
        per_fold=per_fold, n_features=int(Xtr.shape[1]),
        feature_seconds=round(feat_s, 1), runtime_s=round(time.time() - t0, 1),
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
