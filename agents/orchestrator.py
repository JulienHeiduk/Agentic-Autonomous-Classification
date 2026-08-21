"""Orchestrator: reads the feedback from every previous iteration and picks the next
strategy.

The whole point of the loop lives in build_context(): iteration N's prompt contains
iteration N-1's strategy, its measured CV, its leaderboard score and its verdict. The
model is explicitly told what has already been tried so it searches somewhere new.
"""
from agents import ollama, prompts
from harness import ledger


def history():
    """Every completed loop iteration, oldest first."""
    with ledger.conn() as c:
        return c.execute(
            "SELECT e.exp_id, e.hypothesis, e.cv_auc, e.actual_lb, e.status, e.verdict, "
            "e.spec_json, e.runtime_s "
            "FROM experiments e WHERE e.family='loop' ORDER BY e.ts"
        ).fetchall()


def build_context() -> str:
    """The feedback block handed to the next iteration."""
    rows = history()
    if not rows:
        return (
            "ITERATION 1 of this loop. Nothing has been tried yet.\n\n"
            "Reference points measured on this exact fold split:\n"
            "  - a plain LightGBM on the raw columns scores CV 0.96320\n"
            "  - the public leaderboard leader is at 0.97140\n"
            "Start with a solid, low-risk strategy that you are confident will run.\n"
        )

    out = [f"PREVIOUS ITERATIONS ({len(rows)} completed). Read these carefully.\n"]
    best = max((r[2] for r in rows if r[2]), default=None)
    for i, (eid, hyp, cv, lb, status, verdict, spec, rt) in enumerate(rows, 1):
        out.append(f"--- iteration {i}: {eid} ---")
        out.append(f"  strategy   : {hyp}")
        if spec:
            import json
            try:
                s = json.loads(spec)
                if s.get("feature_engineering"):
                    out.append(f"  features   : {'; '.join(s['feature_engineering'][:8])}")
                if s.get("model_family"):
                    out.append(f"  model      : {s['model_family']} / {s.get('key_hyperparameters','')}")
            except Exception:
                pass
        out.append(f"  status     : {status}")
        out.append(f"  CV AUC     : {cv:.6f}" if cv else "  CV AUC     : did not run")
        out.append(f"  Kaggle LB  : {lb:.5f}" if lb else "  Kaggle LB  : not submitted")
        if rt:
            out.append(f"  runtime    : {rt:.0f}s")
        if verdict:
            out.append(f"  VERDICT    : {verdict}")
        out.append("")

    if best:
        out.append(f"BEST CV SO FAR: {best:.6f}")
    out.append(
        "\nYour job now: propose a strategy that is MEANINGFULLY DIFFERENT from every "
        "iteration above and that you expect to beat the best CV. Say plainly in "
        "`differs_from_previous` what you are changing and why the previous result makes "
        "you believe it will help. Do not repeat a strategy that already failed."
    )
    return "\n".join(out)


def propose(model: str = None) -> dict:
    ctx = build_context()
    user = f"{prompts.TASK}\n\n{ctx}\n\nPropose the next strategy as JSON."
    spec = ollama.chat_json(
        model or ollama.ORCHESTRATOR,
        prompts.ORCH_SYSTEM,
        user,
        prompts.STRATEGY_SCHEMA,
        temperature=0.6,      # some spread, or every iteration proposes the same thing
        # Headroom, not a fit: a reasoning orchestrator spends most of this thinking
        # before it emits a token of JSON, and 1200 was already marginal at 3 iterations
        # of history. chat() doubles this on an empty response.
        num_predict=4096,
    )
    return spec
