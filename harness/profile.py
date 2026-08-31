"""Derive everything competition-specific from the CSVs in data/.

Until now the system knew this competition by heart: `TARGET = "addicted_label"` and
`N_TRAIN = 691369` in config, 2,286 characters of prose in prompts.TASK naming the nine
numeric and three categorical columns, an EXAMPLE_PLUGIN written against those columns. Drop
in another dataset and the orchestrator would propose features for columns that do not
exist.

Everything here is read off the data instead. The harness half -- folds, encoding, the
consistency gate, the blender, submission -- was already dataset-agnostic; this replaces the
part that was not.

What it deliberately does NOT try to infer is domain knowledge. The "MEASURED FACTS" block in
the old TASK (the generator invariant, the decimal-digit fingerprint) came from a human
reading public notebooks, and no profiler recovers that from a CSV. It becomes an optional
`data/NOTES.md` that is appended when present, rather than something pretended into existence.

    from harness.profile import profile
    p = profile()          # cached; reads data/train.csv and data/test.csv
"""
import json

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# An integer-valued column with more distinct values than this is treated as a
# high-cardinality lookup key -- worth target/frequency encoding rather than using raw.
# Below it, a small integer column is an ordered quantity and the engines handle it fine.
#
# 50 rather than something lower because the separation is usually stark: on this dataset
# the two genuine lookup keys have 166 and 231 levels while `age` has 18. A threshold that
# swept `age` in would change the injected feature set without evidence that it helps.
HIGH_CARD_MIN = 50
# Above this fraction of distinct values it is closer to an identifier than a feature.
HIGH_CARD_MAX_FRAC = 0.20
# Object/string columns with at most this many levels get the integer-code recipe.
LOW_CARD_MAX = 30

_cache = {}


def _read(path, nrows=None):
    return pd.read_csv(path, nrows=nrows)


def profile(sample: int = 200_000, force: bool = False) -> dict:
    """Everything the prompts and the harness need, read off data/train.csv and test.csv.

    Sampled for the column statistics because a full read of a large train.csv costs seconds
    on every process start; row COUNTS are exact, taken from the files themselves.
    """
    if "p" in _cache and not force:
        return _cache["p"]

    train_path, test_path = DATA / "train.csv", DATA / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(f"expected {train_path} and {test_path}")

    head_tr = _read(train_path, nrows=sample)
    head_te = _read(test_path, nrows=sample)

    # The target is the column train has and test does not. If several qualify, the last one
    # wins -- Kaggle convention puts it at the end -- and the choice is reported so a wrong
    # guess is visible rather than silent.
    extra = [c for c in head_tr.columns if c not in set(head_te.columns)]
    if not extra:
        raise SystemExit("no column in train.csv is absent from test.csv -- cannot infer the "
                         "target; set TARGET in harness/config.py by hand")
    target = extra[-1]

    # The id is a column present in both, unique in the sample, and not the target.
    id_col = None
    for c in head_te.columns:
        if c in head_tr.columns and head_te[c].is_unique and head_tr[c].is_unique:
            id_col = c
            break

    n_train = _count_rows(train_path)
    n_test = _count_rows(test_path)

    feats = [c for c in head_tr.columns if c not in (target, id_col)]
    numeric, categorical, high_card = [], [], []
    stats = {}
    for c in feats:
        s = head_tr[c]
        nunique = int(s.nunique(dropna=True))
        missing = float(s.isna().mean())
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
            integral = bool(np.allclose(s.dropna() % 1, 0)) if s.notna().any() else False
            if (integral and HIGH_CARD_MIN <= nunique <= max(2, int(HIGH_CARD_MAX_FRAC * len(s)))):
                high_card.append(c)
            stats[c] = {"kind": "numeric", "nunique": nunique, "missing": missing,
                        "min": float(s.min()) if s.notna().any() else None,
                        "max": float(s.max()) if s.notna().any() else None,
                        "integral": integral}
        else:
            if nunique <= LOW_CARD_MAX:
                categorical.append(c)
                levels = sorted(str(v) for v in s.dropna().unique())
            else:
                # Too many levels for the integer-code recipe; treat as high-cardinality and
                # let the harness encode it.
                high_card.append(c)
                levels = []
            stats[c] = {"kind": "categorical", "nunique": nunique, "missing": missing,
                        "levels": levels}

    y = head_tr[target]
    p = {
        "target": target,
        "id": id_col,
        "n_train": n_train,
        "n_test": n_test,
        "n_classes": int(y.nunique(dropna=True)),
        "positive_rate": float(y.mean()) if y.nunique() == 2 else None,
        "numeric": numeric,
        "categorical": categorical,
        "high_cardinality": high_card,
        "stats": stats,
        "notes": _notes(),
    }
    _cache["p"] = p
    return p


def _count_rows(path) -> int:
    """Exact row count without loading the frame."""
    with open(path, "rb") as f:
        return max(sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b"")) - 1, 0)


def _notes() -> str:
    """Optional human domain knowledge, appended to TASK when present.

    This is the part a profiler cannot produce. The measured facts that made this system
    work on S6E8 -- the generator invariant, the decimal-digit fingerprint, that the lookup
    keys shift the target rate by 0.22 -- came from someone reading public notebooks. Putting
    them in a file keeps them available without pretending they were inferred.
    """
    p = DATA / "NOTES.md"
    return p.read_text().strip() if p.exists() else ""


def summary(p: dict = None) -> str:
    """One-screen description of the dataset, for a human or a prompt."""
    p = p or profile()
    out = [f"target {p['target']!r} ({p['n_classes']} classes"
           + (f", positive rate {p['positive_rate']:.4f}" if p["positive_rate"] is not None else "")
           + ")",
           f"id {p['id']!r}   rows: train {p['n_train']:,}  test {p['n_test']:,}",
           f"numeric ({len(p['numeric'])}): {', '.join(p['numeric']) or '-'}",
           f"categorical ({len(p['categorical'])}): {', '.join(p['categorical']) or '-'}",
           f"high-cardinality keys ({len(p['high_cardinality'])}): "
           f"{', '.join(p['high_cardinality']) or '-'}"]
    miss = {c: s["missing"] for c, s in p["stats"].items() if s["missing"] > 0}
    if miss:
        lo, hi = min(miss.values()), max(miss.values())
        out.append(f"missing values in {len(miss)}/{len(p['stats'])} columns, {lo:.1%}-{hi:.1%}")
    if p["notes"]:
        out.append(f"data/NOTES.md present ({len(p['notes'])} chars) -- appended to TASK")
    return "\n".join(out)


def main():
    p = profile()
    print(summary(p))
    print()
    print(json.dumps({k: v for k, v in p.items() if k not in ("stats", "notes")}, indent=2))


if __name__ == "__main__":
    main()
