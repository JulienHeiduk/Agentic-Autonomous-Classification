"""Coder and repairer: turn a strategy into a runnable plugin, and fix it when it breaks."""
import json

from agents import ollama, prompts


def write_plugin(spec: dict, model: str = None) -> str:
    strategy = json.dumps(spec, indent=2)
    user = (
        f"{prompts.TASK}\n\n"
        f"STRATEGY TO IMPLEMENT:\n{strategy}\n\n"
        f"{prompts.CONTRACT}\n\n"
        f"Here is a COMPLETE, CORRECT example of the file format. Follow its structure "
        f"exactly; change the feature engineering and the model to match the strategy "
        f"above:\n\n{prompts.EXAMPLE_PLUGIN}\n"
    )
    txt = ollama.chat(
        model or ollama.CODER, prompts.CODER_SYSTEM, user,
        temperature=0.1, num_predict=3000,
    )
    return ollama.extract_code(txt)


def repair(code: str, error: str, spec: dict, model: str = None,
           previous_errors: list = None) -> str:
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
        temperature=0.0, num_predict=3000,
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
        f"In 3 sentences: did it work, what is the most likely reason, and what should the "
        f"NEXT iteration try differently? Be concrete and blunt. Plain text, no markdown."
    )
    return ollama.chat(
        model or ollama.ORCHESTRATOR, "You analyse ML experiment results tersely.",
        # 400 left almost nothing after reasoning -- measured 391 of 400 used.
        user, temperature=0.3, num_predict=1200,
    ).strip()
