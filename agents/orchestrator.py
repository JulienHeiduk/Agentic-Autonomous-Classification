"""Orchestrator: reads the feedback from every previous iteration and picks the next
strategy.

The whole point of the loop lives in build_context(): iteration N's prompt contains
iteration N-1's strategy, its measured CV, its leaderboard score and its verdict. The
model is explicitly told what has already been tried so it searches somewhere new.
"""
import json
import re
import unicodedata

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


# Per-field caps for a full-detail entry. Capping how MANY iterations are replayed is not
# enough on its own: the model writes 800-char hypotheses and 1,400-char feature lists, so
# five entries swung between 10k and 15k chars purely on how verbose it felt that round.
# Capping both count and size makes the block bounded rather than merely smaller.
CLIP_HYPOTHESIS = 450
CLIP_FEATURES = 600
CLIP_HYPERPARAMS = 170
CLIP_VERDICT = 450


def _clip(text, limit: int) -> str:
    """Shorten to `limit`, keeping the head AND the tail.

    Tail-truncation would be actively harmful for verdicts: the critic is asked for "what
    should the NEXT iteration try differently", so its recommendation is the last sentence.
    Cutting the end keeps the complaint and discards the instruction.
    """
    t = str(text or "").strip()
    if len(t) <= limit:
        return t
    head = limit * 2 // 3
    tail = limit - head - 5
    return f"{t[:head].rstrip()} [...] {t[-tail:].lstrip()}"


FULL_DETAIL = 4   # most recent iterations replayed in full; older ones are one-liners


def _is_comparable(status, cv):
    """Only full-data runs on the frozen split can be compared to each other."""
    return cv is not None and status not in ("screening", "failed", "running")


def build_context() -> str:
    """The feedback block handed to the next iteration.

    Two things this deliberately does NOT do, both learned the hard way:

    It does not present a `screening` CV as if it were comparable. Those runs used --rows
    on a subsample and are not on the frozen split. loop04 scored 0.9338 that way and its
    verdict recorded a failure -- for CatBoost with ratio features, which is the family
    loop10 later used to reach the best leaderboard score in the ledger. Replaying that
    number unqualified argues against the approach that works.

    It does not replay every iteration in full. The block grew ~600 tokens per iteration
    and was already 38% of num_ctx at ten of them; left alone it crowds out the reasoning
    and the answer, which is the truncation that returns empty content with no error.
    """
    rows = history()
    if not rows:
        return (
            "ITERATION 1 of this loop. Nothing has been tried yet.\n\n"
            "Reference points measured on this exact fold split:\n"
            "  - a plain LightGBM on the raw columns scores CV 0.96320\n"
            "  - the public leaderboard leader is at 0.97140\n"
            "Start with a solid, low-risk strategy that you are confident will run.\n"
        )

    comparable = [(i, r) for i, r in enumerate(rows, 1) if _is_comparable(r[4], r[2])]
    best_i = max(comparable, key=lambda t: t[1][2])[0] if comparable else None
    detail_from = max(1, len(rows) - FULL_DETAIL + 1)

    out = [f"PREVIOUS ITERATIONS ({len(rows)} completed). Read these carefully.\n"]
    summarised = 0
    for i, (eid, hyp, cv, lb, status, verdict, spec, rt) in enumerate(rows, 1):
        full = (i >= detail_from) or (i == best_i)
        if not full:
            summarised += 1
            cv_s = f"CV {cv:.6f}" if _is_comparable(status, cv) else "no comparable CV"
            out.append(f"--- iteration {i}: {eid} --- {status}, {cv_s} "
                       f"-- {scrub(str(hyp or ''))[:110]}")
            continue

        out.append(f"--- iteration {i}: {eid} ---"
                   + ("   <-- BEST SO FAR" if i == best_i else ""))
        out.append(f"  strategy   : {_clip(scrub(hyp), CLIP_HYPOTHESIS)}")
        if spec:
            try:
                sp = json.loads(spec)
                if sp.get("feature_engineering"):
                    out.append("  features   : " + _clip(
                        scrub('; '.join(sp['feature_engineering'][:8])), CLIP_FEATURES))
                if sp.get("model_family"):
                    out.append(f"  model      : {sp['model_family']} / " + _clip(
                        scrub(str(sp.get('key_hyperparameters', ''))), CLIP_HYPERPARAMS))
            except Exception:
                pass
        out.append(f"  status     : {status}")

        if status == "screening":
            out.append(f"  CV AUC     : {cv:.6f}  <-- SUBSAMPLE RUN (--rows), NOT on the "
                       f"frozen split. NOT comparable to any other CV here. Treat this "
                       f"iteration as UNMEASURED: it is not evidence for or against the "
                       f"strategy." if cv else "  CV AUC     : did not run")
        elif cv:
            out.append(f"  CV AUC     : {cv:.6f}")
        else:
            out.append("  CV AUC     : did not run")

        out.append(f"  Kaggle LB  : {lb:.5f}" if lb else "  Kaggle LB  : not submitted")
        if rt:
            out.append(f"  runtime    : {rt:.0f}s")
        if verdict:
            if status == "screening":
                out.append(f"  VERDICT    : [written against a subsample score -- ignore any "
                           f"judgement it makes about whether the strategy works] "
                           f"{_clip(scrub(verdict), CLIP_VERDICT)}")
            else:
                out.append(f"  VERDICT    : {_clip(scrub(verdict), CLIP_VERDICT)}")
        out.append("")

    if summarised:
        out.insert(1, f"({summarised} older iterations condensed to one line each; the "
                      f"best result and the {FULL_DETAIL} most recent are shown in full.)\n")

    if comparable:
        b = max(r[1][2] for r in comparable)
        out.append(f"BEST CV SO FAR: {b:.6f}  (full-data runs on the frozen split only)")
    out.append(
        "\nYour job now: propose a strategy that is MEANINGFULLY DIFFERENT from every "
        "iteration above and that you expect to beat the best CV. Say plainly in "
        "`differs_from_previous` what you are changing and why the previous result makes "
        "you believe it will help. Do not repeat a strategy that already failed."
    )
    return "\n".join(out)


# Techniques that are still structurally impossible in a plugin.
#
# Target encoding used to be on this list and no longer is: the harness now computes it
# out-of-fold and injects te_<col> columns before make_features runs (harness/encode.py).
# That change was forced by measurement -- both gpt-oss:20b and qwen3.5:9b proposed target
# encoding in roughly 8 of every 10 strategies, and no amount of prohibition moved the rate,
# because it is the correct technique for a dataset whose lookup keys swing the target rate
# by 0.22. Banning the right answer was the wrong fix.
#
# What remains impossible is stacking over out-of-fold PREDICTIONS: the plugin returns one
# estimator and never sees another model's OOF matrix.
_DASH = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
FORBIDDEN = re.compile(
    r"oof[ _-]?stack|stack(ing|ed)?[ _-]+(the[ _-]+)?oof|meta[ _-]?learner"
    r"|pseudo[ _-]?label", re.I)


def scrub(text: str) -> str:
    """Redact impossible techniques from history before it is replayed.

    The critic's verdicts are shown to the orchestrator every iteration, and one of them
    recommends "hashed integers or target-encoded values". The orchestrator followed that
    in 6/6 samples even with an explicit prohibition in ORCH_SYSTEM -- precedent from its
    own history outweighs an instruction. Removing the advice is what actually works;
    telling it louder does not.
    """
    if not text:
        return text
    return FORBIDDEN.sub("[removed: impossible here -- the feature step never sees y]", text)


def forbidden_techniques(spec: dict) -> list:
    """Impossible techniques named anywhere in a proposed strategy.

    Normalises Unicode dashes first: models write "target-encoded" with U+2011 often
    enough that an ASCII-only pattern silently reports a clean result.
    """
    blob = unicodedata.normalize("NFKC", json.dumps(spec, ensure_ascii=False))
    return sorted({m.group(0).lower() for m in FORBIDDEN.finditer(blob.translate(_DASH))})


def propose(model: str = None, tries: int = 4) -> dict:
    """Propose the next strategy, rejecting any that cannot physically be implemented.

    Rejection is cheap (~30s of generation) and the alternative is not: an impossible
    strategy costs the coder's attempt plus every repair attempt plus a preflight run,
    and still ends the iteration with no measurement in the ledger.
    """
    ctx = build_context()
    # The constraints come BEFORE the history so they are not buried under it.
    base = (f"{prompts.TASK}\n\n{prompts.STRATEGY_CONSTRAINTS}\n\n{ctx}\n\n"
            f"Propose the next strategy as JSON.")
    bad = []
    for attempt in range(tries):
        # Resample rather than appending a correction to the prompt: mutating the prompt
        # mid-retry pushed gpt-oss into emitting one unterminated string, which then fails
        # as invalid JSON at every budget. temperature=0.6 already gives enough spread.
        spec = ollama.chat_json(
            model or ollama.ORCHESTRATOR,
            prompts.ORCH_SYSTEM,
            base,
            prompts.STRATEGY_SCHEMA,
            temperature=0.6,  # some spread, or every iteration proposes the same thing
            # Headroom, not a fit: a reasoning orchestrator spends most of this thinking
            # before it emits a token of JSON, and 1200 was already marginal at 3
            # iterations of history. chat() doubles this on empty or truncated output.
            num_predict=4096,
        )
        bad = forbidden_techniques(spec)
        if not bad:
            return spec
        print(f"    [rejected {attempt+1}/{tries}: proposes {', '.join(bad)} -- "
              f"make_features never sees the target]", flush=True)
    raise ollama.OllamaError(
        f"orchestrator proposed an impossible technique {tries} times running "
        f"(last: {', '.join(bad)}). The ledger history is likely steering it -- check "
        f"whether old verdicts recommend target encoding."
    )
