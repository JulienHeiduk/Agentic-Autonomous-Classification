"""CONFIRM tier: measure the noise floor (SPEC 5.2).

The spread between folds inside one 5-fold run is NOT the uncertainty -- it badly
overstates how much the mean moves under a different partition. The bar every idea has to
clear is the std of the repeated 5-fold MEANS across partition seeds.

Until this has run, the promotion rule falls back to the prior in config.py and the word
"promoted" in the ledger means less than it should.

These runs deliberately do not use the frozen split, so their OOF is never stored.

Run:  PYTHONPATH=. .venv/bin/python -m harness.confirm            # best plugin so far
      PYTHONPATH=. .venv/bin/python -m harness.confirm loop02_16592
"""
import argparse
import time

import numpy as np

from harness import config as C
from harness import ledger, sandbox
from harness.evaluate import noise_floor

PLUGINS = C.ROOT / "plugins"


def best_plugin() -> str:
    with ledger.conn() as c:
        rows = c.execute(
            "SELECT exp_id FROM experiments WHERE family='loop' AND cv_auc IS NOT NULL "
            "ORDER BY cv_auc DESC"
        ).fetchall()
    for (eid,) in rows:
        if (PLUGINS / f"{eid}.py").exists():
            return eid
    raise SystemExit("no loop plugin with a stored .py found -- run loop.py first")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp_id", nargs="?")
    p.add_argument("--seeds", default="42,2024,7")
    p.add_argument("--timeout", type=int, default=2400)
    a = p.parse_args()

    exp_id = a.exp_id or best_plugin()
    path = PLUGINS / f"{exp_id}.py"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    code = path.read_text()
    seeds = [int(s) for s in a.seeds.split(",")]

    print(f"measuring the noise floor on {exp_id} over partition seeds {seeds}")
    scores = []
    for sd in seeds:
        t0 = time.time()
        ok, res, _ = sandbox.execute(f"confirm_{exp_id}_s{sd}", code,
                                     timeout=a.timeout, partition_seed=sd,
                                     plugin_path=path)
        if not ok:
            raise SystemExit(f"seed {sd} failed:\n{res.get('error','')[:1500]}")
        scores.append(res["cv_auc"])
        print(f"  partition seed {sd:<5d} OOF {res['cv_auc']:.6f}  ({time.time()-t0:.0f}s)")

    arr = np.array(scores)
    fl = noise_floor(scores)
    print(f"\n  mean {arr.mean():.6f}  +/- {arr.std(ddof=1):.6f}")
    print(f"  NOISE FLOOR = {fl:.6f}")
    print(f"  promotion now requires delta >= {C.MIN_DELTA_FLOORS * fl:+.6f} "
          f"AND P(improve) > {C.MIN_P_IMPROVE}")

    (C.STATE / "noise_floor.txt").write_text(f"{fl:.8f}\n")
    ledger.record(f"confirm_{exp_id}", family="confirm", tier="confirm",
                  hypothesis=f"noise floor of {exp_id} over partition seeds {seeds}",
                  what_it_lets_the_model_ask="sets the bar every subsequent delta must clear",
                  cv_auc=float(arr.mean()), cv_std=float(fl), status="promoted",
                  spec={"seeds": seeds, "scores": scores, "source_exp": exp_id})
    print(f"\n  written to {(C.STATE / 'noise_floor.txt').relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
