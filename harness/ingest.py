"""Ingest publicly shared OOF libraries as blend members (SPEC 2.5).

Competitors on this competition publish out-of-fold and test prediction pairs as CC0
datasets, explicitly for stacking. Because the fold split is frozen and shared, their arrays
are positionally row-compatible with ours, so they drop straight into harness/blend.py with
no retraining and no LLM involved.

This matters because diversity is the binding constraint. Our own 21 members correlate at
0.98-0.998 -- near copies -- which is why greedy selection takes three and stops. One public
library of 7 members cross-correlates with ours at 0.9479 and moved the stack by +0.000085
on its own. SPEC 2.5 records that the top of this leaderboard is substantially assembled
this way; the reference notebook reached LB 0.97113 from a 177-member pool.

Layouts differ per publisher, so discovery is by convention rather than a fixed schema:

    oof_<name>.npy  + test_<name>.npy         paired arrays        (golem, xgb/lgb/cat)
    bandoof_<name>.npy + bandtest_<name>.npy  same, prefixed       (fm-lattice)
    <anything>.csv with an id column          test-only, skipped   (no OOF -> unusable)

EVERY member is checked against the frozen split before it is accepted: an array of the
wrong length, or one whose recomputed AUC is no better than chance, is misaligned rather
than merely weak, and a misaligned member poisons a stack silently.

    PYTHONPATH=. .venv/bin/python -m harness.ingest
"""
import argparse
import json

import numpy as np

from harness import config as C
from harness.data import load
from harness.evaluate import auc

STORE = C.EXT_OOF / "_members"
INDEX = C.EXT_OOF / "index.json"

# An aligned member scores well above chance on our y. A shuffled or differently-ordered
# array lands at ~0.5, so this separates "misaligned" from "weak but real" cleanly.
MIN_AUC = 0.55


def _csv_pairs(d):
    """(name, oof_csv, test_csv) for publishers who ship CSVs instead of arrays.

    Convention seen in the wild: `<n>_oof_predictions.csv` beside `<n>_submission.csv`,
    each an id column plus one probability column. The ids differ between the two files
    (train ids in one, test ids in the other), so alignment is by SORTED id rather than by
    file order -- a publisher who wrote rows in a different order would otherwise be
    silently misaligned, which is the one failure that poisons a stack without complaining.
    """
    out = []
    for f in sorted(d.rglob("*_oof_predictions.csv")):
        t = f.with_name(f.name.replace("_oof_predictions.csv", "_submission.csv"))
        if t.exists():
            out.append((f"{d.name}:{f.name.split('_oof')[0]}", f, t))
    return out


def _load_csv(path, n_expected):
    """The probability column of a two-column id/prediction CSV, ordered by id."""
    import pandas as pd
    df = pd.read_csv(path)
    idc = C.ID if C.ID in df.columns else df.columns[0]
    val = [c for c in df.columns if c != idc]
    if len(val) != 1 or len(df) != n_expected:
        return None
    return df.sort_values(idc)[val[0]].to_numpy(dtype=np.float64)


def _matrix_pairs(d):
    """Publishers who ship ONE 2D array of members rather than one file each.

    `oof.npy` of shape (n_train, k) beside `test.npy` of shape (n_test, k), with an
    optional members.csv naming the columns. Each column is a separate member.
    """
    o, t = d / "oof.npy", d / "test.npy"
    if not (o.exists() and t.exists()):
        return []
    try:
        a = np.load(o, mmap_mode="r")
    except Exception:
        return []
    if a.ndim != 2:
        return []
    names = None
    mc = d / "members.csv"
    if mc.exists():
        try:
            import pandas as pd
            df = pd.read_csv(mc)
            col = next((c for c in df.columns if c.lower() in ("id", "member", "name")), None)
            if col is not None and len(df) == a.shape[1]:
                names = [str(v) for v in df[col]]
        except Exception:
            names = None
    return [(f"{d.name}:{names[i] if names else i}", (o, i), (t, i))
            for i in range(a.shape[1])]


def _parquet_pairs(d):
    """A wide parquet of member predictions, one column per member."""
    o = next(iter(d.glob("oof_predictions.parquet")), None)
    t = next(iter(d.glob("test_predictions.parquet")), None)
    if not (o and t):
        return []
    try:
        import pyarrow.parquet as pq
        cols = [c for c in pq.ParquetFile(o).schema.names if c != C.ID]
        tcols = set(pq.ParquetFile(t).schema.names)
    except Exception:
        return []
    return [(f"{d.name}:{c}", (o, c), (t, c)) for c in cols if c in tcols]


def _pairs(d):
    """(name, oof_path, test_path) for every OOF/test array pair, at any depth."""
    out = []
    for f in sorted(d.rglob("*.npy")):
        stem = f.name[:-4]
        for pre, tpre in (("oof_", "test_"), ("bandoof_", "bandtest_")):
            if stem.startswith(pre):
                t = f.with_name(tpre + stem[len(pre):] + ".npy")
                if t.exists():
                    out.append((f"{d.name}:{stem[len(pre):]}", f, t))
                break
    return out + _csv_pairs(d) + _matrix_pairs(d) + _parquet_pairs(d)


def discover():
    """Every OOF/test pair across every downloaded library."""
    found = []
    for d in sorted(C.EXT_OOF.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        found.extend(_pairs(d))
    return found


def ingest(verbose: bool = True):
    """Validate every discovered pair against the frozen split and store what survives."""
    _, _, y = load()
    STORE.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], []

    for name, oof_p, test_p in discover():
        try:
            if isinstance(oof_p, tuple):
                # (path, column) -- a slice of a 2D array or a parquet column.
                (op, oc), (tp, tc) = oof_p, test_p
                if str(op).endswith(".parquet"):
                    import pandas as pd
                    oof = pd.read_parquet(op, columns=[oc])[oc].to_numpy(dtype=np.float64)
                    pred = pd.read_parquet(tp, columns=[tc])[tc].to_numpy(dtype=np.float64)
                else:
                    oof = np.asarray(np.load(op, mmap_mode="r")[:, oc], dtype=np.float64)
                    pred = np.asarray(np.load(tp, mmap_mode="r")[:, tc], dtype=np.float64)
            elif oof_p.suffix == ".csv":
                oof = _load_csv(oof_p, len(y))
                pred = _load_csv(test_p, C.N_TEST)
                if oof is None or pred is None:
                    rejected.append((name, "csv shape/columns unusable"))
                    continue
            else:
                oof = np.load(oof_p).astype(np.float64).ravel()
                pred = np.load(test_p).astype(np.float64).ravel()
        except Exception as e:
            rejected.append((name, f"unreadable: {type(e).__name__}"))
            continue
        if len(oof) != len(y) or len(pred) != C.N_TEST:
            rejected.append((name, f"shape {len(oof)}/{len(pred)} "
                                   f"!= {len(y)}/{C.N_TEST}"))
            continue
        if not (np.isfinite(oof).all() and np.isfinite(pred).all()):
            rejected.append((name, "non-finite values"))
            continue
        if np.std(oof) < 1e-9:
            rejected.append((name, "constant"))
            continue
        a = auc(y, oof)
        # Below chance is not automatically wrong -- SPEC 2.5 notes deliberate signed
        # correctors exist -- but at this scale a near-0.5 score means the rows are in a
        # different order, which is the one failure that would silently poison the stack.
        if a < MIN_AUC:
            rejected.append((name, f"AUC {a:.4f} -- misaligned, not merely weak"))
            continue
        safe = name.replace(":", "__").replace("/", "_")
        np.save(STORE / f"{safe}.oof.npy", oof.astype(np.float32))
        np.save(STORE / f"{safe}.pred.npy", pred.astype(np.float32))
        kept.append({"name": name, "file": safe, "auc": float(a)})

    INDEX.write_text(json.dumps(kept, indent=2))
    if verbose:
        print(f"discovered {len(kept) + len(rejected)} pairs, kept {len(kept)}")
        for k in sorted(kept, key=lambda k: -k["auc"]):
            print(f"  {k['name']:<44} AUC {k['auc']:.6f}")
        for n, why in rejected:
            print(f"  REJECTED {n:<35} {why}")
    return kept


def members():
    """Ingested public members as (name, oof, pred), for harness.blend."""
    if not INDEX.exists():
        return []
    out = []
    for k in json.loads(INDEX.read_text()):
        o = STORE / f"{k['file']}.oof.npy"
        p = STORE / f"{k['file']}.pred.npy"
        if o.exists() and p.exists():
            out.append((k["name"], np.load(o).astype(np.float64),
                        np.load(p).astype(np.float64)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    ingest(verbose=not a.quiet)


if __name__ == "__main__":
    main()
