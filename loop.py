"""The recursive loop.

One iteration =

    orchestrator (local LLM)  reads every previous iteration's feedback -> picks a strategy
    coder        (local LLM)  writes a plugin implementing it
    sandbox                   runs it on the frozen 5-fold split -> OOF AUC
    repairer     (local LLM)  fixes and re-runs on failure (up to --repairs times)
    kaggle                    submits, waits for the public score
    critic       (local LLM)  writes a verdict

The verdict, the CV, the leaderboard score and the strategy all go into the ledger, and
build_context() feeds them to the NEXT iteration's orchestrator. That is the recursion:
each loop searches with knowledge of what the previous loop measured.

Usage:
    PYTHONPATH=. .venv/bin/python loop.py --iterations 2
    PYTHONPATH=. .venv/bin/python loop.py --iterations 2 --no-submit
    PYTHONPATH=. .venv/bin/python loop.py --iterations 2 --rows 120000   # fast screening
"""
import argparse
import json
import re
import time
import traceback

from agents import coder, families, ollama, orchestrator
from harness import config as C
from harness import blend, evaluate, ledger, report, sandbox
from harness.data import load


PREFLIGHT_ROWS = 8000
# 8,000 rows is seconds of work for any sane configuration, so a long ceiling here buys
# nothing and costs a great deal: a plugin that picks 6000 CatBoost iterations at depth 10
# blows any budget, and each repair attempt pays it again: three attempts against a 600s
# ceiling spent 30 minutes to learn what 120s learns, and --repairs now defaults to 10.
PREFLIGHT_TIMEOUT = 120


def _rule(txt=""):
    print(f"\n{'='*78}\n{txt}\n{'='*78}" if txt else "=" * 78, flush=True)


def best_loop_run(family: str = None):
    """(exp_id, cv) of the best experiment THIS FAMILY has produced, or (None, 0.0).

    Excludes the reference pipeline: the loop's progress is measured against its own
    trajectory, so iteration 2 is judged on whether it beat iteration 1.

    Scoped by family, because judging across families is not a comparison -- it is a
    guarantee of failure. A linear member measured against a GBM's 0.965142 is rejected
    whatever it scores; being rejected it never becomes an inheritable base; and a family
    with no base never improves. The family would be permanently stuck at iteration one.
    """
    with ledger.conn() as c:
        rows = c.execute(
            "SELECT exp_id, cv_auc, spec_json FROM experiments WHERE family='loop' "
            "AND cv_auc IS NOT NULL ORDER BY cv_auc DESC"
        ).fetchall()
    for exp_id, cv, spec_json in rows:
        if orchestrator.belongs_to(spec_json, family):
            return exp_id, cv
    return (None, 0.0)


RESET_AFTER = 3   # repairs on one file before abandoning it for the known-good base


def error_signature(err: str) -> str:
    """Collapse an error to what makes it the SAME failure, for cycle detection.

    Numbers, quoted values and bracketed lists change between attempts while the failure
    does not -- 9 duplicate columns then 13 duplicate columns is one problem, not two.
    """
    first = (err or "").strip().splitlines()[0] if (err or "").strip() else ""
    first = re.sub(r"\[.*?\]", "[]", first)
    first = re.sub(r"'[^']*'", "'x'", first)
    first = re.sub(r"\d+", "N", first)
    return first[:120]


def inherit_base(family: str = None):
    """The best plugin THIS FAMILY has produced, as source for the next iteration to mutate.

    This is the code-level half of the recursion. build_context() already carries strategy
    and results forward; without this the coder rebuilt every implementation from a prose
    bullet list and a static example, so a 0.965 file was re-derived from memory each time
    and consecutive plugins were only 41-71% alike. Improvements were being re-invented
    rather than kept.

    Only promoted/baseline runs qualify. A rejected member is rejected because its number
    could not be trusted -- loop06 was rejected for a train/test leak -- and inheriting its
    code would carry that forward. Screening runs are excluded because their CV is a
    subsample number, not a measurement.

    The candidate is re-checked against the CURRENT static rules before being offered: the
    contract has tightened since older plugins were written, and handing the coder a base
    that no longer passes would fail the iteration before it starts.
    """
    # Filtered by family, because a base from another family is worse than none: handing a
    # linear run the best CatBoost plugin tells the coder to mutate a tree while the strategy
    # says use a linear model, and the mutation prompt asks it to keep everything the strategy
    # did not mention. Each family climbs its own hill; a family with no promoted member yet
    # gets the reference example, which is the correct cold start.
    allowed = set(families.get(family)["model_family"]) if family else None
    with ledger.conn() as c:
        rows = c.execute(
            "SELECT exp_id, cv_auc, actual_lb, json_extract(spec_json, '$.model_family') "
            "FROM experiments WHERE family='loop' "
            "AND cv_auc IS NOT NULL AND status IN ('promoted', 'baseline') "
            "ORDER BY cv_auc DESC"
        ).fetchall()
    for exp_id, cv, lb, mf in rows:
        if allowed is not None and mf not in allowed:
            continue
        path = sandbox.PLUGINS / f"{exp_id}.py"
        if not path.exists():
            continue
        code = path.read_text()
        if sandbox.static_check(code):
            continue
        return {"exp_id": exp_id, "cv": cv, "lb": lb, "code": code}
    return None


def best_cv():
    return best_loop_run()[1]


def measured_noise_floor():
    """The floor from `harness.confirm`, or the config prior if it has never been run."""
    p = C.STATE / "noise_floor.txt"
    if p.exists():
        try:
            return float(p.read_text().strip()), "measured"
        except ValueError:
            pass
    return C.NOISE_FLOOR_PRIOR, "PRIOR -- run `python -m harness.confirm` to measure it"


def judge(exp_id, cv, prev_id, prev_cv, screening=False):
    """Apply the SPEC 5.3 promotion rule instead of a bare `delta > 0`.

    Returns (status, detail). The first loop run has nothing to compare against, so it is
    recorded as `baseline`, not `promoted` -- calling it an improvement would be a claim
    about a comparison that was never made. Screening runs are not on the frozen split and
    store no arrays, so they are never judged at all.
    """
    floor, src = measured_noise_floor()
    if screening:
        return "screening", {"floor": floor, "floor_source": src}
    if prev_id is None:
        return "baseline", {"floor": floor, "floor_source": src}

    oof_new, _ = ledger.load_arrays(exp_id)
    oof_ref, _ = ledger.load_arrays(prev_id)
    _, _, y = load()
    status, detail = evaluate.promote(y, oof_new, oof_ref, floor=floor)
    detail["floor_source"] = src
    detail["reference"] = prev_id
    return status, detail


def iteration(n: int, args) -> dict:
    exp_id = f"loop{n:02d}_{int(time.time()) % 100000}"
    _rule(f"ITERATION {n}   ({exp_id})")

    # ---- 1. plan, with every previous result as context -------------------------
    print("[orchestrator] reading feedback from previous iterations...", flush=True)
    # Must pass the family. Without it this printed the WHOLE loop history while
    # propose() correctly sent the family's own -- so the log showed ten CatBoost
    # iterations to a linear run that was in fact starting cold. A display that
    # disagrees with the prompt is worse than no display.
    ctx = orchestrator.build_context(args.family)
    lines = ctx.splitlines()
    shown = lines[:40]
    print("\n".join("    | " + l for l in shown), flush=True)
    if len(lines) > len(shown):
        # Say so. Silently showing less than the model receives hides exactly the kind of
        # bad row you would want to catch by eye -- a screening CV replayed as if it were
        # comparable sat unseen below this cut for six iterations.
        print(f"    | ... {len(lines) - len(shown)} more lines sent to the model but not "
              f"printed ({len(ctx)} chars total)", flush=True)
    spec = orchestrator.propose(family=args.family)
    spec["orchestrator"] = args.family
    print(f"\n[strategy] {spec['strategy_name']}")
    print(f"  hypothesis : {spec['hypothesis']}")
    print(f"  asks       : {spec['what_it_lets_the_model_ask']}")
    print(f"  features   : {'; '.join(spec['feature_engineering'][:8])}")
    print(f"  model      : {spec['model_family']} / {spec['key_hyperparameters']}")
    print(f"  differs by : {spec['differs_from_previous']}")
    print(f"  expects    : CV {spec['expected_cv_auc']}", flush=True)

    ledger.record(
        exp_id, family="loop", tier="full", status="running",
        hypothesis=spec["hypothesis"],
        what_it_lets_the_model_ask=spec["what_it_lets_the_model_ask"],
        playbook_ref=spec.get("playbook_ref") or "other", spec=spec,
    )

    # ---- 2. write the code ------------------------------------------------------
    # A stacking run trains on the member matrix, so --rows subsampling and the
    # target-encoding injection do not apply to it.
    is_stack = args.family == "stack"
    base = None if args.no_inherit else inherit_base(args.family)
    if base:
        lb_txt = f", LB {base['lb']:.5f}" if base.get("lb") else ""
        print(f"\n[coder] writing plugin (mutating {base['exp_id']}, "
              f"CV {base['cv']:.6f}{lb_txt})...", flush=True)
    else:
        print("\n[coder] writing plugin (from the reference example)...", flush=True)
    code = coder.write_plugin(spec, base=base, family=args.family)
    print(f"    {len(code.splitlines())} lines", flush=True)

    # ---- 3. preflight on a tiny subsample, repairing until it runs ---------------
    # A broken plugin should cost ~15s to discover, not a full 691k-row training run.
    ok, result, attempts, seen_errors, sigs = False, {}, 0, [], []
    for attempt in range(args.repairs + 1):
        attempts = attempt
        label = "preflight" if attempt == 0 else f"preflight after repair {attempt}"
        print(f"\n[sandbox] {label} ({PREFLIGHT_ROWS:,} rows)...", flush=True)
        ok, result, _ = sandbox.execute(exp_id, code, timeout=PREFLIGHT_TIMEOUT,
                                        rows=PREFLIGHT_ROWS, stack=is_stack)
        if ok:
            print(f"    preflight OK (auc {result['cv_auc']:.4f}, "
                  f"{result['n_features']} features)", flush=True)
            break
        err = str(result.get("error", ""))
        print("    FAILED:\n" + "\n".join("      " + l for l in err.splitlines()[:18]),
              flush=True)
        if attempt == args.repairs:
            break

        # Escape a rabbit hole instead of digging. Repairing edits whatever is in front of
        # it, so a structurally wrong plugin gets patched and each patch inherits the flaw.
        # Two signs it is not converging: the same error twice, or several attempts spent.
        # Either way, throw the file away and re-implement the strategy minimally on the
        # plugin that already scores.
        sig = error_signature(err)
        stuck = sigs.count(sig) >= 1 or attempt + 1 >= RESET_AFTER
        sigs.append(sig)
        if stuck and base:
            print(f"[repairer] not converging ({sig[:60]}) -- restarting from "
                  f"{base['exp_id']} with a minimal version", flush=True)
            code = coder.restart_from_base(spec, base, seen_errors + [err])
            sigs = []                      # a fresh line of attack gets a fresh budget
        else:
            print(f"[repairer] attempt {attempt+1}/{args.repairs}"
                  f" (temp {min(0.6, 0.15 * attempt):.2f})...", flush=True)
            code = coder.repair(code, err, spec, previous_errors=seen_errors,
                                attempt=attempt, family=args.family)
        seen_errors.append(err)

    # ---- 3b. the real run on the frozen split -----------------------------------
    if ok and not args.rows:
        print("\n[sandbox] full run on the frozen 5-fold split...", flush=True)
        ok, result, _ = sandbox.execute(exp_id, code, timeout=args.timeout,
                                        stack=is_stack)
        if not ok:
            err = str(result.get("error", ""))
            print("    FAILED:\n" + "\n".join("      " + l for l in err.splitlines()[:18]),
                  flush=True)

    prev_id, prev_best = best_loop_run(args.family)

    if not ok:
        verdict = coder.critique(spec, result, prev_best)
        ledger.record(exp_id, status="failed", verdict=verdict,
                      repair_attempts=attempts,
                      runtime_s=result.get("runtime_s"))
        print(f"\n[verdict] {verdict}", flush=True)
        return {"exp_id": exp_id, "ok": False, "verdict": verdict}

    cv = result["cv_auc"]
    delta = cv - prev_best
    print(f"\n[result] CV AUC {cv:.6f}   (previous best {prev_best:.6f}, "
          f"delta {delta:+.6f})   {result['runtime_s']:.0f}s", flush=True)

    # ---- 3c. the promotion rule (SPEC 5.3), not a bare delta > 0 ----------------
    status, det = judge(exp_id, cv, prev_id, prev_best, screening=bool(args.rows))
    if status == "screening":
        print("[judge]  SCREENING -- not on the frozen split, not judged, not stackable")
    elif prev_id:
        print(f"[judge]  {status.upper()}  vs {prev_id}: delta {det['delta']:+.6f} "
              f"= {det['floors']:+.1f}x floor, P(improve) {det['p_improve']:.2f}")
        print(f"         floor {det['floor']:.6f} [{det['floor_source']}], "
              f"threshold {C.MIN_DELTA_FLOORS * det['floor']:+.6f}")
    else:
        print(f"[judge]  BASELINE (nothing to compare against yet)")

    ledger.record(exp_id, status=status, cv_auc=cv, cv_std=result["cv_std"],
                  delta_vs_best=delta, p_improve=det.get("p_improve"),
                  runtime_s=result["runtime_s"], repair_attempts=attempts)

    # ---- 4. submit and get real leaderboard feedback ----------------------------
    lb = None
    if args.submit and not args.rows:
        from harness import submit as sub
        print("\n[kaggle] submitting...", flush=True)
        try:
            lb = sub.submit(exp_id, message=f"{spec['strategy_name']} cv={cv:.6f}")
        except Exception as e:
            print(f"    submission failed: {e}", flush=True)
    elif args.rows:
        print("\n[kaggle] skipped (screening mode -- OOF is not on the frozen split)")
    else:
        print("\n[kaggle] skipped (--no-submit)")

    # ---- 5. verdict -> becomes the next iteration's context --------------------
    print("\n[critic] writing verdict...", flush=True)
    verdict = coder.critique(spec, result, prev_best)
    ledger.record(exp_id, verdict=verdict, actual_lb=lb)
    print(f"[verdict] {verdict}", flush=True)

    return {"exp_id": exp_id, "ok": True, "cv": cv, "lb": lb,
            "delta": delta, "status": status, "verdict": verdict}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--repairs", type=int, default=10)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--rows", type=int, default=None,
                   help="train on the first N rows only (fast screening; disables submit)")
    p.add_argument("--family", default=families.DEFAULT, choices=families.names(),
                   help="which model family this run explores; each has its own "
                        "estimator set and its own contract (see agents/families.py)")
    p.add_argument("--no-inherit", action="store_true",
                   help="write each plugin from the reference example instead of "
                        "mutating the best one so far (wider exploration)")
    p.add_argument("--no-blend", action="store_true",
                   help="skip the stacking pass at the end of the run")
    p.add_argument("--no-submit", dest="submit", action="store_false")
    p.set_defaults(submit=True)
    args = p.parse_args()

    if not ollama.alive():
        raise SystemExit("ollama is not reachable at localhost:11434 -- start it first")

    _rule("AGENTIC AUTONOMOUS CLASSIFICATION")
    print(f"  orchestrator : {ollama.ORCHESTRATOR}  [{args.family} family]")
    print(f"  coder        : {ollama.CODER}")
    print(f"  iterations   : {args.iterations}")
    print(f"  submit       : {args.submit and not args.rows}")
    print(f"  best CV now  : {best_loop_run(args.family)[1]:.6f} ({args.family} family)")

    # continue the numbering from what the ledger already holds, so a second invocation
    # produces iteration 3, not another iteration 1
    done = len(orchestrator.history())
    results = []
    for i in range(done + 1, done + args.iterations + 1):
        try:
            results.append(iteration(i, args))
        except Exception:
            traceback.print_exc()
            results.append({"ok": False, "error": "loop-level exception"})
        if (C.STATE / "PAUSE").exists():
            print("\nPAUSE file present -- stopping.")
            break

    _rule("SUMMARY")
    for i, r in enumerate(results, 1):
        if r.get("ok"):
            lb = f"{r['lb']:.5f}" if r.get("lb") else "—"
            print(f"  {i}. {r['exp_id']:<18s} CV {r['cv']:.6f}  LB {lb}  "
                  f"delta {r['delta']:+.6f}  [{r.get('status','?')}]")
        else:
            print(f"  {i}. {r.get('exp_id','?'):<18s} FAILED")
    fl, src = measured_noise_floor()
    print(f"\n  best CV overall: {best_cv():.6f}")
    print(f"  noise floor    : {fl:.6f} [{src}]")

    # Blend AFTER the iterations, not during: greedy selection is O(n^2) stacker fits, and
    # a run's members are only worth re-stacking once they all exist. Without this the loop
    # produced members and never combined them -- the blender was a manual step nothing
    # invoked, so every new member sat unused until someone remembered.
    if not args.no_blend:
        _rule("BLEND")
        try:
            blend.build()
        except SystemExit as e:
            print(f"  skipped: {e}")
        except Exception as e:
            # A failed blend must not lose the iterations that just succeeded.
            print(f"  blend failed: {type(e).__name__}: {e}")

    report.main()


if __name__ == "__main__":
    main()
