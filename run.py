"""One command: explore every family, blend the members, submit the blend.

    PYTHONPATH=. .venv/bin/python run.py --iterations 5

This is the autonomous path. Each family explores on its own without submitting -- a family
run produces members, not submissions -- and only the blend at the end goes to Kaggle, which
is both the strongest artefact and the one worth spending quota on.

Families run in SEPARATE PROCESSES. A family that fails outright (a model that will not load,
a contract the coder cannot satisfy) then costs that family and not the run: the remaining
families still produce members and the blend still happens with whatever exists.

Blending is deferred to the end rather than run per family. Greedy selection is O(n^2)
stacker fits, and a member from the third family cannot be weighed against the first until
it exists.
"""
import argparse
import subprocess
import sys
import time

from agents import families
from harness import blend, ledger


def _rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def run_family(family: str, iterations: int, rows: int, repairs: int) -> bool:
    """One family's exploration, in its own process. True if it exited cleanly."""
    cmd = [sys.executable, "-u", "loop.py",
           "--family", family,
           "--iterations", str(iterations),
           "--repairs", str(repairs),
           "--no-submit",      # members now, one submission at the end
           "--no-blend"]       # blended once at the end, over every family
    if rows:
        cmd += ["--rows", str(rows)]
    _rule(f"FAMILY: {family}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=".", env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin:/usr/local/bin",
                                          "HOME": str(__import__("pathlib").Path.home())})
    print(f"\n[{family}] exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)
    return r.returncode == 0


def member_count() -> int:
    with ledger.conn() as c:
        return c.execute(
            "SELECT COUNT(*) FROM experiments WHERE cv_auc IS NOT NULL "
            "AND family NOT IN ('blend', 'confirm')"
        ).fetchone()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=5,
                   help="iterations PER FAMILY")
    p.add_argument("--families", nargs="+", default=["gbm", "linear", "nn"],
                   choices=families.names())
    p.add_argument("--repairs", type=int, default=10)
    p.add_argument("--rows", type=int, default=None,
                   help="screening only; members from a subsample are not stackable")
    p.add_argument("--no-submit", action="store_true",
                   help="build the blend but do not send it to Kaggle")
    a = p.parse_args()

    _rule("AUTONOMOUS RUN")
    print(f"  families   : {', '.join(a.families)}")
    print(f"  iterations : {a.iterations} each ({a.iterations * len(a.families)} total)")
    print(f"  members now: {member_count()}")

    ok = {}
    for fam in a.families:
        ok[fam] = run_family(fam, a.iterations, a.rows, a.repairs)

    _rule("BLEND")
    if a.rows:
        print("  skipped: --rows members are not on the frozen split and store no arrays")
        return
    before = member_count()
    print(f"  members available: {before}")
    try:
        exp_id = blend.build()
    except SystemExit as e:
        print(f"  skipped: {e}")
        exp_id = None
    except Exception as e:
        print(f"  blend failed: {type(e).__name__}: {e}")
        exp_id = None

    _rule("SUBMIT")
    if exp_id is None:
        print("  nothing to submit: the blend did not beat the best single member.")
        print("  submit the best member by hand if you want to spend the quota:")
        print("    PYTHONPATH=. .venv/bin/python -m harness.submit <exp_id>")
    elif a.no_submit:
        print(f"  built {exp_id}, not submitted (--no-submit). Send it with:")
        print(f"    PYTHONPATH=. .venv/bin/python -m harness.submit {exp_id}")
    else:
        from harness import submit as sub
        try:
            lb = sub.submit(exp_id, message=f"autonomous run: {exp_id}")
            print(f"  submitted {exp_id} -> LB {lb}")
        except Exception as e:
            print(f"  submission failed: {type(e).__name__}: {e}")
            print(f"  the member is recorded; retry with "
                  f"`python -m harness.submit {exp_id}`")

    _rule("SUMMARY")
    for fam, good in ok.items():
        print(f"  {fam:<8} {'ok' if good else 'FAILED'}")
    print(f"  members: {before} -> {member_count()}")


if __name__ == "__main__":
    main()
