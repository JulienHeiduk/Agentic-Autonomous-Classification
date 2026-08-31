"""Render the data profile as the prompt material that used to be hand-written.

`prompts.TASK` was 2,286 characters naming this competition's nine numeric and three
categorical columns, and `EXAMPLE_PLUGIN` was written against those exact names. Both are
generated here from `harness.profile`, so a different classification dataset needs no edit.

What is NOT generated is domain knowledge. The old TASK carried a "MEASURED FACTS" block --
the generator invariant, the decimal-digit fingerprint, that the lookup keys swing the target
rate by 0.22 -- which came from a human reading public notebooks. No profiler recovers that
from a CSV, so it lives in `data/NOTES.md` and is appended verbatim when present. Losing it
costs real signal; pretending it was inferred would cost more.
"""
from harness import profile as _profile


def task(p: dict = None) -> str:
    """The dataset briefing every prompt starts with."""
    p = p or _profile.profile()
    tgt, idc = p["target"], p["id"]
    lines = [
        f"COMPETITION: predict `{tgt}` from the columns below.",
        "METRIC: ROC AUC (ranking only -- calibration does not matter).",
        f"DATA: train.csv {p['n_train']:,} rows; test.csv {p['n_test']:,} rows."
        + (f" Positive rate {p['positive_rate']:.4f}." if p["positive_rate"] is not None else ""),
        "",
        "COLUMNS",
    ]
    if p["numeric"]:
        lines.append(f"  numeric ({len(p['numeric'])}): " + _wrap(p["numeric"]))
    if p["categorical"]:
        cats = []
        for c in p["categorical"]:
            lv = p["stats"][c]["levels"]
            cats.append(f"{c} {{{','.join(lv[:6])}{'...' if len(lv) > 6 else ''}}}")
        lines.append(f"  categorical ({len(p['categorical'])}): " + _wrap(cats))
    lines += [
        f"  target:          {tgt}   -- present in train only",
        f"  id:              {idc}   -- present in both, NOT a feature" if idc else "",
        "",
    ]

    miss = {c: s["missing"] for c, s in p["stats"].items() if s["missing"] > 0}
    if miss:
        lo, hi = min(miss.values()), max(miss.values())
        lines.append(f"MISSING VALUES: {len(miss)} of {len(p['stats'])} columns carry them, "
                     f"{lo:.1%} to {hi:.1%}.")
        lines.append("")

    if p["high_cardinality"]:
        lines += [
            "PROVIDED FOR YOU -- the harness has ALREADY encoded these high-cardinality keys",
            "and put the results in `train` and `test` before you see them:",
            "",
        ]
        for c in p["high_cardinality"]:
            n = p["stats"][c]["nunique"]
            lines.append(f"     te_{c}   freq_{c}      ({n} distinct values)")
        lines += [
            "",
            "  te_ is out-of-fold target encoding, computed against the split you are scored",
            "  on and leak-free. freq_ is frequency, fitted on train and applied to both",
            "  frames. Use them like any other numeric column. Do NOT rebuild either.",
            "",
        ]

    if p["notes"]:
        lines += ["MEASURED FACTS ABOUT THIS DATASET (verified by hand, use them):",
                  p["notes"], ""]
    return "\n".join(l for l in lines if l is not None)


def example_plugin(p: dict = None) -> str:
    """A complete, correct plugin for THIS dataset, as the coder's reference.

    Generated rather than written, because a reference naming columns that do not exist is
    worse than none: the coder follows the example over the instructions every time.
    """
    p = p or _profile.profile()
    tgt, idc = p["target"], p["id"]
    nums = p["numeric"]
    levels = {c: p["stats"][c]["levels"] for c in p["categorical"]}
    drop = [x for x in (idc, tgt) if x]

    levels_src = "\n".join(
        f'          {c!r}: {lv!r},' if i else f'LEVELS = {{{c!r}: {lv!r},'
        for i, (c, lv) in enumerate(levels.items())
    )
    levels_src = (levels_src.rstrip(",") + "}") if levels else "LEVELS = {}"

    first_num = nums[0] if nums else None
    extra = (f'    X["{first_num}_sq"] = X["{first_num}"] ** 2\n'
             if first_num else "")

    return f'''```python
import numpy as np
import pandas as pd
import lightgbm as lgb

NUMS = {nums!r}
{levels_src}


def _fe(df):
    X = df.drop(columns={drop!r}, errors="ignore").copy()
    for c in NUMS:
        X[f"na_{{c}}"] = X[c].isna().astype(np.int8)
{extra}    for c, lv in LEVELS.items():
        X[c] = pd.Categorical(X[c], categories=lv).codes.astype(np.int8)
    return X


def make_features(train, test):
    X_train = _fe(train)
    X_test = _fe(test)[X_train.columns]
    return X_train, X_test


def make_model(seed):
    return lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        min_child_samples=100, random_state=seed, n_jobs=-1, verbose=-1)
```'''


def categorical_recipe(p: dict = None) -> str:
    """The contract's categorical block, with this dataset's actual levels."""
    p = p or _profile.profile()
    if not p["categorical"]:
        return "There are no low-cardinality categorical columns in this dataset."
    levels = {c: p["stats"][c]["levels"] for c in p["categorical"]}
    body = "\n".join(f'              {c!r}: {lv!r},' for c, lv in levels.items())
    return (
        "    LEVELS = {\n" + body + "\n    }\n\n"
        "    for c, lv in LEVELS.items():\n"
        "        X[c] = pd.Categorical(X[c], categories=lv).codes.astype(np.int8)   # NaN -> -1"
    )


def _wrap(items, width=76, indent=" " * 19):
    out, line = [], ""
    for it in items:
        if line and len(line) + len(it) + 2 > width:
            out.append(line + ",")          # keep the separator across the wrap
            line = it
        else:
            line = f"{line}, {it}" if line else it
    if line:
        out.append(line)
    return ("\n" + indent).join(out)
