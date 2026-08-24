"""Orchestrator: reads the feedback from every previous iteration and picks the next
strategy.

The whole point of the loop lives in build_context(): iteration N's prompt contains
iteration N-1's strategy, its measured CV, its leaderboard score and its verdict. The
model is explicitly told what has already been tried so it searches somewhere new.
"""
import difflib
import json
import re
import unicodedata

from agents import families, ollama, prompts
from harness import ledger


def belongs_to(spec_json, family: str) -> bool:
    """Is this experiment part of `family`?

    Newer rows carry `orchestrator` in their spec; older ones predate the family split, so
    fall back to whether the estimator they chose is in that family's enum.
    """
    if not family:
        return True
    try:
        sp = json.loads(spec_json or "{}")
    except Exception:
        return False
    if sp.get("orchestrator"):
        return sp["orchestrator"] == family
    return sp.get("model_family") in set(families.get(family)["model_family"])


def history(family: str = None):
    """Completed loop iterations, oldest first, restricted to one family.

    Family-scoped because a linear run shown ten CatBoost strategies imitates them -- the
    same precedent-beats-instruction effect measured all through this project -- and because
    "BEST CV SO FAR" taken across families is a number the linear family cannot reach, which
    makes every honest linear result read as a failure.
    """
    with ledger.conn() as c:
        rows = c.execute(
            "SELECT e.exp_id, e.hypothesis, e.cv_auc, e.actual_lb, e.status, e.verdict, "
            "e.spec_json, e.runtime_s "
            "FROM experiments e WHERE e.family='loop' ORDER BY e.ts"
        ).fetchall()
    return [r for r in rows if belongs_to(r[6], family)]


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


def build_context(family: str = None) -> str:
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
    rows = history(family)
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


def forbidden_techniques(spec: dict, family: str = None) -> list:
    """Impossible techniques named anywhere in a proposed strategy.

    Normalises Unicode dashes first: models write "target-encoded" with U+2011 often
    enough that an ASCII-only pattern silently reports a clean result.
    """
    if family == "stack":
        # A stacking run exists precisely to build a meta-learner over other models' OOF
        # predictions. This guard was written when that was impossible in a plugin; for the
        # stack family it is the goal, and leaving it on rejected the orchestrator's own
        # correct proposals.
        return []
    blob = unicodedata.normalize("NFKC", json.dumps(spec, ensure_ascii=False))
    return sorted({m.group(0).lower() for m in FORBIDDEN.finditer(blob.translate(_DASH))})


BACKLOG_QUEUED = 6      # ideas offered per prompt
BACKLOG_REJECTED = 8    # dead ends listed so they are not rediscovered


def backlog_block() -> str:
    """The curated search space, which the orchestrator could not previously see.

    harness/seed.py writes these from SPEC 3.3 -- priority-ordered, each with a measured
    reason -- and build_context() never read the table. The consequence was measurable:
    nine of ten iterations proposed the same idea (treating the lookup keys as categorical)
    while `public_oof_pool`, the highest-priority entry in the playbook, was never proposed
    once. A searcher with no menu re-samples whatever is already in front of it.

    The rejected entries matter as much as the queued ones: SPEC 3.3 exists so the loop does
    not spend iterations rediscovering dead ends someone already measured.
    """
    with ledger.conn() as c:
        queued = c.execute(
            "SELECT idea_id, family, priority, expected_gain, rationale FROM backlog "
            "WHERE status='queued' ORDER BY priority DESC LIMIT ?", (BACKLOG_QUEUED,)
        ).fetchall()
        rejected = c.execute(
            "SELECT idea_id, rationale FROM backlog WHERE status='rejected' "
            "ORDER BY priority DESC LIMIT ?", (BACKLOG_REJECTED,)
        ).fetchall()
    if not queued and not rejected:
        return ""

    out = ["", "=== CURATED IDEA LIST (the search space, priority-ordered) ===",
           "These were written from measured community results, not guessed. Prefer one of",
           "them over inventing a variation of what the last iteration already did.", ""]
    for idea, fam, pri, gain, why in queued:
        g = f", expected {gain:+.5f}" if gain is not None else ""
        out.append(f"  [{pri:>3}] {idea} ({fam}{g})")
        out.append(f"        {_clip(str(why or ''), 200)}")
    if rejected:
        out += ["", "ALREADY MEASURED AND REJECTED -- do not propose these:"]
        for idea, why in rejected:
            out.append(f"  x {idea}: {_clip(str(why or ''), 150)}")
    out.append("")
    return "\n".join(out)


NOVELTY_MAX = 0.5       # Jaccard overlap of content words above which it is a repeat
STOPWORDS = frozenset("""the a an and or of to in on for with as by is are be will can
that this it its from at into than then so we our using use used already more most very
model models feature features column columns key keys data set sets run runs iteration
should would could may might because since while when where which what how also both each
new better best improve improves improving performance signal""".split())


def _tokens(spec) -> set:
    """Content words of a strategy, for overlap comparison."""
    return {w for w in _fingerprint(spec).split() if len(w) > 2 and w not in STOPWORDS}


def _fingerprint(spec_or_row) -> str:
    """The comparable content of a strategy: what it builds and what it fits."""
    if isinstance(spec_or_row, dict):
        parts = [str(spec_or_row.get("hypothesis", "")),
                 "; ".join(spec_or_row.get("feature_engineering", []) or []),
                 str(spec_or_row.get("model_family", ""))]
    else:
        parts = [str(spec_or_row or "")]
    t = " ".join(parts).lower()
    return re.sub(r"[^a-z0-9 ]+", " ", t)


def too_similar(spec: dict, family: str = None):
    """(exp_id, ratio) of a past iteration this strategy repeats, or None.

    The prompt has asked for a MEANINGFULLY DIFFERENT strategy from the start, and nine of
    ten iterations still proposed treating the lookup keys as categorical -- the CV spread
    across all of them was 0.0018, which is sampling noise around one idea rather than a
    search. Instructions lose to the precedent sitting in the context; forbidden_techniques
    showed that a code-level check is what actually holds.
    """
    new = _tokens(spec)
    if not new:
        return None
    worst = None
    for eid, hyp, cv, lb, status, verdict, spec_json, rt in history(family):
        prev = None
        if spec_json:
            try:
                prev = _tokens(json.loads(spec_json))
            except Exception:
                prev = None
        if not prev:
            prev = _tokens({"hypothesis": hyp})
        if not prev:
            continue
        # Jaccard on content words. difflib was tried first and is the wrong tool: it
        # matches character runs, so a reworded version of the same idea scored 0.04-0.06
        # against a 0.62 threshold and sailed through. Overlap of concepts is what "already
        # tried" actually means, and it does not care about sentence length or word order.
        r = len(new & prev) / max(len(new | prev), 1)
        if r >= NOVELTY_MAX and (worst is None or r > worst[1]):
            worst = (eid, r)
    return worst


def _used_refs(family: str = None) -> set:
    """Playbook entries this family has already used. Scoped, because `seed_averaging` tried
    on trees says nothing about `seed_averaging` tried on a linear model."""
    with ledger.conn() as c:
        rows = c.execute(
            "SELECT playbook_ref, spec_json FROM experiments WHERE family='loop' "
            "AND playbook_ref IS NOT NULL").fetchall()
    return {r[0] for r in rows if belongs_to(r[1], family)}


def strategy_schema(family: str = families.DEFAULT) -> dict:
    """STRATEGY_SCHEMA with playbook_ref constrained to the ideas still on the menu.

    A schema enum is enforced by the decoder, so this is the one instruction the model
    cannot paraphrase its way around -- which is what both text-similarity guards failed to
    do. `other` stays available so a genuinely new idea is still possible.
    """
    import copy
    with ledger.conn() as c:
        ids = [r[0] for r in c.execute(
            "SELECT idea_id FROM backlog WHERE status='queued' ORDER BY priority DESC")]
    used = _used_refs(family)
    fresh = [i for i in ids if i not in used]
    schema = copy.deepcopy(prompts.STRATEGY_SCHEMA)
    # `fresh or ids` was wrong: once every entry had been used the fallback re-offered the
    # whole list, so the exclusion silently switched itself off exactly when it was needed
    # -- one run picked no_te_members four times. An exhausted menu should say so, not
    # pretend to be full.
    schema["properties"]["playbook_ref"]["enum"] = fresh + ["other"]
    schema["properties"]["model_family"]["enum"] = families.get(family)["model_family"]
    return schema


def unused_playbook_refs() -> list:
    with ledger.conn() as c:
        ids = [r[0] for r in c.execute(
            "SELECT idea_id FROM backlog WHERE status='queued' ORDER BY priority DESC")]
    used = _used_refs()
    return [i for i in ids if i not in used]


def blend_block() -> str:
    """Tell the orchestrator a blender exists and what that changes about its job.

    Without this the loop optimises solo CV and nothing else, which is the wrong objective
    once members are stacked: nine GBM members correlating at 0.98-0.998 produced a blend
    gain of +0.00025 where the playbook measured +0.0004, purely because they were near
    copies of each other. A member that scores slightly lower but disagrees can be worth
    more to the stack than a marginally better clone.

    It is deliberately NOT an invitation to chase decorrelation for its own sake -- SPEC 3.3
    is explicit that contribution tracks solo OOF and that members below 0.966 contributed
    zero or sign-flipping weight despite being the least correlated. Strong AND different is
    the bar; different alone is not.
    """
    with ledger.conn() as c:
        row = c.execute(
            "SELECT exp_id, cv_auc, actual_lb, spec_json FROM experiments "
            "WHERE family='blend' AND cv_auc IS NOT NULL ORDER BY cv_auc DESC LIMIT 1"
        ).fetchone()
    if not row:
        return ""
    eid, cv, lb, spec_json = row
    try:
        members = json.loads(spec_json or "{}").get("members", [])
    except Exception:
        members = []
    lb_txt = f", public LB {lb:.5f}" if lb else ""
    return (
        "\n=== YOUR MEMBERS ARE STACKED ===\n"
        f"The harness blends stored out-of-fold predictions. The current best blend is "
        f"`{eid}` at CV {cv:.6f}{lb_txt}, built from: {', '.join(members) or 'n/a'}.\n"
        "Your strategy does not have to beat the best single model on its own. A member that "
        "scores a little lower but makes DIFFERENT mistakes can earn more weight in the stack "
        "than one more near-copy of the current best. The existing members agree with each "
        "other far too closely, which is why the blend gains so little.\n"
        "But different is not sufficient: a member below roughly 0.966 solo has historically "
        "contributed nothing at all, whatever its correlation. Aim for strong AND unlike what "
        "is already there.\n"
    )


def propose(model: str = None, tries: int = 4,
            family: str = families.DEFAULT) -> dict:
    """Propose the next strategy, rejecting any that cannot physically be implemented.

    Rejection is cheap (~30s of generation) and the alternative is not: an impossible
    strategy costs the coder's attempt plus every repair attempt plus a preflight run,
    and still ends the iteration with no measurement in the ledger.
    """
    ctx = build_context(family)
    # The constraints come BEFORE the history so they are not buried under it.
    fam = families.get(family)
    base = (f"{prompts.TASK}\n\n{prompts.STRATEGY_CONSTRAINTS}\n\n"
            f"{fam['guidance']}\n"
            f"{blend_block()}"
            f"{backlog_block()}\n{ctx}\n\n"
            f"Propose the next strategy as JSON. It must use a {fam['label']} model.")
    bad = []
    for attempt in range(tries):
        # Resample rather than appending a correction to the prompt: mutating the prompt
        # mid-retry pushed gpt-oss into emitting one unterminated string, which then fails
        # as invalid JSON at every budget. temperature=0.6 already gives enough spread.
        spec = ollama.chat_json(
            model or ollama.ORCHESTRATOR,
            prompts.ORCH_SYSTEM,
            base,
            strategy_schema(family),
            temperature=0.6,  # some spread, or every iteration proposes the same thing
            # Headroom, not a fit: a reasoning orchestrator spends most of this thinking
            # before it emits a token of JSON, and 1200 was already marginal at 3
            # iterations of history. chat() doubles this on empty or truncated output.
            num_predict=4096,
        )
        bad = forbidden_techniques(spec, family)
        if bad:
            print(f"    [rejected {attempt+1}/{tries}: proposes {', '.join(bad)} -- "
                  f"make_features never sees the target]", flush=True)
            continue
        ref = spec.get("playbook_ref")
        if ref and ref != "other" and ref in _used_refs(family) and attempt < tries - 1:
            print(f"    [rejected {attempt+1}/{tries}: playbook_ref '{ref}' already "
                  f"used -- pick an idea that has not been tried]", flush=True)
            continue
        dup = too_similar(spec, family)
        if dup and attempt < tries - 1:
            # Not on the last attempt: a repeat still beats no strategy at all.
            print(f"    [rejected {attempt+1}/{tries}: {dup[1]:.0%} similar to "
                  f"{dup[0]} -- already tried]", flush=True)
            continue
        return spec
    raise ollama.OllamaError(
        f"orchestrator proposed an impossible technique {tries} times running "
        f"(last: {', '.join(bad)}). The ledger history is likely steering it -- check "
        f"whether old verdicts recommend target encoding."
    )
