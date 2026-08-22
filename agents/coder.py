"""Coder and repairer: turn a strategy into a runnable plugin, and fix it when it breaks."""
import json

from agents import ollama, prompts


def write_plugin(spec: dict, model: str = None, base: dict = None) -> str:
    """Turn a strategy into a plugin, mutating the best plugin so far when there is one.

    `base` is what makes the recursion reach the code. Given one, the model edits a file
    that is known to run and known to score, instead of rebuilding the same feature
    engineering from a bullet list every iteration and losing pieces of it in the process.
    Without one -- first iteration, or --no-inherit -- it falls back to the static example.
    """
    strategy = json.dumps(spec, indent=2)
    if base:
        lb = f" and LB {base['lb']:.5f}" if base.get("lb") else ""
        anchor = (
            f"YOU ARE MODIFYING AN EXISTING FILE, NOT WRITING ONE FROM SCRATCH.\n\n"
            f"Below is `{base['exp_id']}`, the best plugin this loop has produced -- "
            f"CV {base['cv']:.6f}{lb}. It runs, and it already satisfies every rule above.\n\n"
            f"Apply the STRATEGY to this file using the SMALLEST set of changes that "
            f"implements it. Keep everything the strategy does not ask you to change: the "
            f"feature engineering already here is what earns the current score, and "
            f"rewriting it from memory is how a good score gets lost. Return the COMPLETE "
            f"modified file.\n\n"
            f"```python\n{base['code']}\n```\n"
        )
    else:
        anchor = (
            f"Here is a COMPLETE, CORRECT example of the file format. Follow its structure "
            f"exactly; change the feature engineering and the model to match the strategy "
            f"above:\n\n{prompts.EXAMPLE_PLUGIN}\n"
        )
    user = (
        f"{prompts.TASK}\n\n"
        f"STRATEGY TO IMPLEMENT:\n{strategy}\n\n"
        f"{prompts.CONTRACT}\n\n"
        f"{anchor}"
    )
    txt = ollama.chat(
        model or ollama.CODER, prompts.CODER_SYSTEM, user,
        temperature=0.1, num_predict=3000,
    )
    return ollama.extract_code(txt)


def repair(code: str, error: str, spec: dict, model: str = None,
           previous_errors: list = None, attempt: int = 0) -> str:
    """Fix a failing plugin. `attempt` raises the temperature on successive tries.

    The first attempt stays deterministic: a precise error deserves a precise edit. After
    that, determinism is the enemy -- at temperature 0.0 the same file plus the same
    traceback reproduces the same output, so every retry is a copy of the one before it.
    Measured: three-attempt runs returning 553/553/553 and 577/577/577 tokens, byte for
    byte. Attempts that cannot differ cannot converge.
    """
    history = ""
    if previous_errors:
        # Without this a small model ping-pongs: it fixes error A by reintroducing error B,
        # then fixes B by reintroducing A. Showing it the loop is what breaks the loop.
        history = (
            "\n=== YOU HAVE ALREADY TRIED AND FAILED WITH THESE ERRORS ===\n"
            + "\n".join(f"  attempt {i+1}: {e.splitlines()[0]}"
                        for i, e in enumerate(previous_errors))
            + "\n\nDo NOT reintroduce an earlier error while fixing the current one. If the "
              "errors are alternating, the two constraints conflict -- change approach "
              "entirely and use the simplest thing that satisfies both.\n"
        )
    user = (
        f"This file failed. Fix it.\n\n"
        f"STRATEGY IT IMPLEMENTS: {spec.get('hypothesis','')}\n\n"
        f"=== FILE ===\n```python\n{code}\n```\n\n"
        f"=== CURRENT ERROR ===\n{error}\n"
        f"{history}\n"
        f"{prompts.CONTRACT}\n"
        f"Return the complete corrected file."
    )
    txt = ollama.chat(
        model or ollama.CODER, prompts.REPAIR_SYSTEM, user,
        temperature=min(0.6, 0.15 * attempt), num_predict=3000,
    )
    return ollama.extract_code(txt)


def critique(spec: dict, result: dict, best_cv: float, model: str = None) -> str:
    """One-paragraph verdict written back into the ledger, and read by the NEXT iteration."""
    if not result.get("ok"):
        return f"FAILED: {str(result.get('error',''))[:300]}"
    cv = result["cv_auc"]
    delta = cv - best_cv if best_cv else 0.0
    user = (
        f"An experiment just finished.\n\n"
        f"STRATEGY: {json.dumps(spec, indent=2)}\n\n"
        f"RESULT: CV AUC {cv:.6f} across folds {['%.5f'%f for f in result['per_fold']]}\n"
        f"        {result['n_features']} features, {result['runtime_s']:.0f}s\n"
        f"        previous best CV was {best_cv:.6f} -> delta {delta:+.6f}\n"
        f"        the model expected {spec.get('expected_cv_auc')}\n\n"
        f"{prompts.STRATEGY_CONSTRAINTS}\n\n"
        f"In 3 sentences: did it work, what is the most likely reason, and what should the "
        f"NEXT iteration try differently? Be concrete and blunt. Plain text, no markdown. "
        f"Your suggestion is read by the next orchestrator, so it must obey the "
        f"constraints above -- never suggest target encoding or an out-of-fold scheme."
    )
    return ollama.chat(
        model or ollama.ORCHESTRATOR, "You analyse ML experiment results tersely.",
        # 400 left almost nothing after reasoning -- measured 391 of 400 used.
        user, temperature=0.3, num_predict=1200,
    ).strip()


def restart_from_base(spec: dict, base: dict, errors: list, model: str = None) -> str:
    """Abandon a failing file and re-implement the strategy minimally on the known-good base.

    Repairing edits whatever is in front of it, so a plugin that went wrong structurally --
    a polynomial block duplicating every column, an ensemble that is not one estimator --
    gets patched rather than reconsidered, and each patch inherits the flaw. One run spent
    ten attempts inside that loop and produced no measurement.

    This is the way out: throw the broken file away, go back to the plugin that actually
    scored, and ask for the SMALLEST version of the strategy instead of the fullest one. A
    modest working iteration beats an ambitious one that never runs.
    """
    tried = "\n".join(f"  attempt {i+1}: {e.splitlines()[0][:160]}"
                      for i, e in enumerate(errors))
    user = (
        f"{prompts.TASK}\n\n"
        f"A plugin implementing the strategy below has now failed {len(errors)} times:\n"
        f"{tried}\n\n"
        f"STOP REPAIRING IT. The approach itself is going wrong, not one line of it.\n\n"
        f"STRATEGY:\n{json.dumps(spec, indent=2)}\n\n"
        f"{prompts.CONTRACT}\n\n"
        f"Here is `{base['exp_id']}`, which RUNS and scores CV {base['cv']:.6f}:\n\n"
        f"```python\n{base['code']}\n```\n\n"
        f"Start from THIS file and add the SIMPLEST possible version of the strategy -- the "
        f"one or two features that carry its core idea, nothing more. Drop polynomial "
        f"expansions, drop ensembles, drop anything the errors above touched. A small "
        f"working change is worth more than the full idea that never runs.\n\n"
        f"YOU MUST STILL CHANGE SOMETHING. Returning this file unmodified wastes the "
        f"iteration: the harness has already measured it at CV {base['cv']:.6f}, so an "
        f"identical plugin produces no new information. Add at least one new feature, or "
        f"one concrete change to make_model, that comes from the strategy. Say nothing "
        f"else -- return the complete file."
    )
    txt = ollama.chat(
        model or ollama.CODER, prompts.CODER_SYSTEM, user,
        temperature=0.3, num_predict=3000,
    )
    return ollama.extract_code(txt)
