"""Stack the stored OOF predictions into a new member (SPEC 3 `blend` tier).

Every full-data run already saves its out-of-fold and test predictions, positionally
aligned to the frozen split, and until now nothing read them back. This turns that pile
into a member: quarantine, transform, a nested stacker, greedy member selection.

No LLM is involved and none is wanted. The technique is already decided by measurement --
`logit_or_rankgauss_stack` is marked done in the playbook, "a linear stacker that can
subtract, rank-gauss for a large pool and logits for a small one" -- and what remains is a
handful of discrete choices that greedy search covers exactly. A model proposing subsets
would add a code path that can fail and no capability that this cannot reach.

Measured on the first nine members (all GBM, pairwise OOF correlation 0.98-0.998):

    best single member   0.965142
    simple average       0.965088   <- WORSE than the best single
    nested logit stack   0.965388   (+0.000246, 8.6x the noise floor)

The average losing to the best member is the whole argument for a stacker: averaging
correlated members of unequal strength dilutes the strongest one, while a fitted linear
stacker can down-weight and subtract.

Run:  PYTHONPATH=. .venv/bin/python -m harness.blend
"""
import argparse
import hashlib
import json

import numpy as np
from scipy.stats import ks_2samp, norm
from sklearn.linear_model import LogisticRegression

from harness import config as C
from harness import folds, ledger
from harness.data import load
from harness.evaluate import auc

LOGIT_CLIP = 30.0        # SPEC 2: stack on clip(log(p/(1-p)), +-30), not probabilities
RANKGAUSS_ABOVE = 25     # members above this count -> rank-gauss instead of logits
KS_DRIFT_MAX = 0.10      # OOF vs test-prediction drift that disqualifies a member
STACK_C = 1.0


def candidates():
    """Members eligible to enter a stack, strongest first.

    Only full-data runs on the frozen split have stored arrays at all -- screening and
    noise-floor runs never call save_arrays -- so eligibility is mostly self-enforcing.
    Rejected members are still allowed in: a member rejected for not beating the best
    single model can still carry weight in a stack, which is the entire point of blending.
    What is NOT allowed in is a member whose own number was untrustworthy, and the
    consistency gate has already stopped those from being recorded.
    """
    with ledger.conn() as c:
        rows = c.execute(
            "SELECT exp_id, cv_auc, json_extract(spec_json, '$.model_family') "
            "FROM experiments WHERE cv_auc IS NOT NULL AND family != 'confirm' "
            "ORDER BY cv_auc DESC"
        ).fetchall()
    out = []
    for exp_id, cv, fam in rows:
        if not (C.OOF_DIR / f"{exp_id}.npy").exists():
            continue
        if not (C.PRED_DIR / f"{exp_id}.npy").exists():
            continue
        out.append({"exp_id": exp_id, "cv": cv, "model_family": fam})
    return out


def quarantine(members, y):
    """Drop members that would corrupt the stack (SPEC 2.5).

    Three checks, each for a failure that is silent rather than loud:

    * duplicates -- an identical OOF array entering twice doubles that member's weight
      without anyone choosing to;
    * degenerate -- a constant or non-finite member contributes nothing and can make the
      stacker's design matrix singular;
    * drift -- a member whose test predictions are distributed unlike its OOF predictions
      is the loop06 failure in member form: its CV will not transfer to the leaderboard,
      and stacking it launders that straight into the blend.
    """
    kept, dropped, seen = [], [], {}
    for m in members:
        oof, pred = ledger.load_arrays(m["exp_id"])
        if len(oof) != len(y) or len(pred) != C.N_TEST:
            dropped.append((m["exp_id"], f"wrong length {len(oof)}/{len(pred)}"))
            continue
        if not np.isfinite(oof).all() or not np.isfinite(pred).all():
            dropped.append((m["exp_id"], "non-finite values"))
            continue
        if np.std(oof) < 1e-9:
            dropped.append((m["exp_id"], "constant OOF"))
            continue
        h = hashlib.sha256(np.round(oof, 8).tobytes()).hexdigest()
        if h in seen:
            dropped.append((m["exp_id"], f"duplicate of {seen[h]}"))
            continue
        ks = ks_2samp(oof, pred).statistic
        if ks > KS_DRIFT_MAX:
            dropped.append((m["exp_id"], f"OOF/test drift KS={ks:.3f}"))
            continue
        seen[h] = m["exp_id"]
        m = dict(m, oof=oof, pred=pred, ks=ks)
        kept.append(m)
    return kept, dropped


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.clip(np.log(p / (1 - p)), -LOGIT_CLIP, LOGIT_CLIP)


def _rankgauss(p):
    r = np.argsort(np.argsort(np.asarray(p, dtype=float)))
    return norm.ppf((r + 0.5) / len(r))


def transform(arrs, n_members):
    """Logits for a small pool, rank-gauss for a large one.

    `rank_stack_small_pool` is REJECTED in the playbook -- plain rank hurt on a small
    all-GBM library -- but the same entry notes it flips for a large heterogeneous pool,
    so the choice is made by pool size rather than fixed.
    """
    f = _logit if n_members <= RANKGAUSS_ABOVE else _rankgauss
    return np.column_stack([f(a) for a in arrs])


def nested_stack(X, y, splits, C_reg=STACK_C):
    """Out-of-fold stacker predictions on the frozen split.

    Fitting the stacker on every OOF row and scoring it on those same rows is the loop06
    mistake relocated: the number looks excellent and does not survive the leaderboard. For
    each fold the stacker sees only the other folds' rows, so the score it reports is one
    it could actually reproduce out of sample.
    """
    oof = np.zeros(len(y))
    for itr, iva in splits:
        m = LogisticRegression(C=C_reg, max_iter=2000)
        m.fit(X[itr], y[itr])
        oof[iva] = m.predict_proba(X[iva])[:, 1]
    return oof


def greedy_select(members, y, splits, floor):
    """Forward selection: add the member that most improves the nested stack.

    Stops when the best available addition is worth less than the measured noise floor,
    so the pool grows only while the gain is distinguishable from run-to-run variance.
    """
    chosen, best_auc, history = [], None, []
    remaining = list(range(len(members)))
    while remaining:
        scored = []
        for j in remaining:
            idx = chosen + [j]
            X = transform([members[i]["oof"] for i in idx], len(idx))
            a = auc(y, nested_stack(X, y, splits))
            scored.append((a, j))
        scored.sort(reverse=True)
        gain_auc, j = scored[0]
        if best_auc is not None and gain_auc - best_auc < floor:
            break
        chosen.append(j)
        remaining.remove(j)
        history.append((members[j]["exp_id"], gain_auc,
                        gain_auc - best_auc if best_auc is not None else None))
        best_auc = gain_auc
    return chosen, best_auc, history


def build(exp_id: str = None, verbose: bool = True):
    _, _, y = load()
    splits = list(folds.splits(y))
    floor = _floor()

    members = candidates()
    if len(members) < 2:
        raise SystemExit(f"need at least 2 members with stored OOF, found {len(members)}")
    kept, dropped = quarantine(members, y)
    if verbose:
        print(f"candidates {len(members)}, kept {len(kept)}")
        for eid, why in dropped:
            print(f"  dropped {eid}: {why}")
    if len(kept) < 2:
        raise SystemExit(f"only {len(kept)} members survived quarantine")

    best_single = max(kept, key=lambda m: auc(y, m["oof"]))
    best_single_auc = auc(y, best_single["oof"])

    chosen, stack_auc, history = greedy_select(kept, y, splits, floor)
    if verbose:
        print(f"\nbest single member : {best_single['exp_id']} {best_single_auc:.6f}")
        print("greedy forward selection:")
        for eid, a, d in history:
            print(f"  + {eid:<16} stack {a:.6f}" + (f"  ({d:+.6f})" if d else ""))

    idx = chosen
    Xtr = transform([kept[i]["oof"] for i in idx], len(idx))
    oof = nested_stack(Xtr, y, splits)
    # The submitted prediction comes from a stacker fitted on ALL the OOF rows. That is
    # correct and is not the leak above: it never sees the test labels, and the score it is
    # judged on is the nested one computed a line earlier.
    final = LogisticRegression(C=STACK_C, max_iter=2000).fit(Xtr, y)
    Xte = transform([kept[i]["pred"] for i in idx], len(idx))
    pred = final.predict_proba(Xte)[:, 1]

    delta = stack_auc - best_single_auc
    if verbose:
        print(f"\nstack of {len(idx)}: OOF {stack_auc:.6f}   vs best single "
              f"{best_single_auc:.6f}   delta {delta:+.6f} "
              f"({delta / floor:.1f}x floor {floor:.6f})")
        print("weights:", {kept[i]["exp_id"]: round(float(w), 3)
                           for i, w in zip(idx, final.coef_[0])})

    if delta < C.MIN_DELTA_FLOORS * floor:
        if verbose:
            print("\nNOT RECORDED: the stack does not clear the promotion rule.")
        return None

    exp_id = exp_id or f"blend_{len(idx)}m_{int(stack_auc * 1e6)}"
    ledger.save_arrays(exp_id, oof, pred)
    ledger.record(
        exp_id, family="blend", tier="full", status="promoted", cv_auc=float(stack_auc),
        delta_vs_best=float(delta),
        hypothesis=f"logit stack of {len(idx)} members: "
                   + ", ".join(kept[i]["exp_id"] for i in idx),
        spec={"members": [kept[i]["exp_id"] for i in idx],
              "weights": [float(w) for w in final.coef_[0]],
              "transform": "logit" if len(idx) <= RANKGAUSS_ABOVE else "rankgauss",
              "model_family": "stacker"},
    )
    if verbose:
        print(f"\nrecorded {exp_id} -- submit with:"
              f"\n  PYTHONPATH=. .venv/bin/python -m harness.submit {exp_id}")
    return exp_id


def _floor():
    p = C.STATE / "noise_floor.txt"
    if p.exists():
        try:
            return float(p.read_text().strip())
        except ValueError:
            pass
    return C.NOISE_FLOOR_PRIOR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    build(exp_id=a.exp_id, verbose=not a.quiet)


if __name__ == "__main__":
    main()


def member_frames(exclude_families=("blend", "stack", "confirm")):
    """The OOF matrix as (train, test, y) frames, for a second-level plugin to train on.

    This is what makes a stacking ORCHESTRATOR possible rather than a fixed stacker. The
    plugin contract does not change at all -- make_features still receives a train frame
    with the target and a test frame without it, and make_model still returns one estimator.
    What changes is the columns: instead of age and sleep_hours they are one column per
    member, holding that member's predicted probability. A second-level model therefore
    trains on other models' output, never on the raw data.

    Members that are themselves stacks are excluded. Stacking a stack re-uses the same rows
    twice and inflates the result without adding information.
    """
    import pandas as pd
    _, test_raw, y = load()
    with ledger.conn() as c:
        rows = c.execute(
            "SELECT exp_id, family FROM experiments WHERE cv_auc IS NOT NULL ORDER BY cv_auc DESC"
        ).fetchall()
    keep = [e for e, fam in rows
            if fam not in exclude_families
            and (C.OOF_DIR / f"{e}.npy").exists() and (C.PRED_DIR / f"{e}.npy").exists()]
    if len(keep) < 2:
        raise SystemExit(f"stacking needs >=2 members with stored OOF, found {len(keep)}")

    tr, te = {}, {}
    for e in keep:
        oof, pred = ledger.load_arrays(e)
        if len(oof) != len(y) or len(pred) != C.N_TEST:
            continue
        tr[e] = oof
        te[e] = pred
    train = pd.DataFrame(tr)
    train[C.ID] = np.arange(len(y))
    train[C.TARGET] = y
    test = pd.DataFrame(te)
    test[C.ID] = test_raw[C.ID].to_numpy()
    return train, test, y, list(tr)


def member_names(exclude_families=("blend", "stack", "confirm")):
    """The column names a stacking plugin will receive, without loading any arrays.

    A stacking plugin has to reference its columns by name, and the names are experiment
    ids -- `xgb_te`, `loop09_69453` -- which nothing in a static prompt could predict. One
    generated plugin guessed `pred_1` and died with a KeyError. Cheap enough to call while
    building the coder's prompt.
    """
    with ledger.conn() as c:
        rows = c.execute(
            "SELECT exp_id, family, cv_auc FROM experiments WHERE cv_auc IS NOT NULL "
            "ORDER BY cv_auc DESC"
        ).fetchall()
    return [(e, cv) for e, fam, cv in rows
            if fam not in exclude_families
            and (C.OOF_DIR / f"{e}.npy").exists() and (C.PRED_DIR / f"{e}.npy").exists()]
