# Conclusion

*Kaggle Playground S6E8 — Predicting Smartphone Addiction. Built 2026-08-21, deadline 2026-08-31.
Results below are read out of `state/ledger.db`; every number is a measurement this repo made, not an
estimate.*

---

## 1. The result

| | CV (nested, frozen split) | public LB |
|---|---|---|
| Trivial baseline (`sample_submission`, constant) | — | 0.50000 |
| Raw LightGBM, no engineering (SPEC §1.4) | 0.963200 | — |
| **Best hand-written reference member** (`xgb_te`) | **0.968587** | — |
| **Best LLM-generated single model** (`loop09_69453`, gbm) | **0.965142** | 0.96654 |
| First stack, 3 own members (`blend_3m_968710`) | 0.968710 | 0.96995 |
| **Final stack, 274 members** (`blend_274m_970205`) | **0.970206** | **0.97123** |

The stack climbed monotonically and the leaderboard agreed every time:

```
0.968710 → 0.96995     3 members, all ours
0.969041 → 0.97026     6 members, first public members added
0.969386 → 0.97048    36 members
0.969934 → 0.97105     3 members, 2 of them public
0.970176 → 0.97121   265 members
0.970206 → 0.97123   274 members    ← final
```

`0.97123` was the score of public rank 10 when SPEC §1.5 was fetched on day 1, against a leader at
`0.97140`. The CV→LB residual on the final stack was **+0.00102**, in line with every honest
iteration and nothing like the −0.0267 that exposed the one broken one.

### Against the success criteria written on day 1 (SPEC §14)

| | Target | Outcome |
|---|---|---|
| Minimum | ≥ 20 experiments, ≥ 24 h unattended, no human code fixes | **Met** — 48 loop iterations over 4 days; no file in `plugins/` has a modification in git history, only additions |
| Expected | Honest CV ≥ 0.9695 (LB ≈ 0.9706) | **Met** — 0.970206 / 0.97123 |
| Stretch | CV ≥ 0.9700 **with** ≥ 50 % of stacker weight on our own members | **Not met, and the two halves turned out to be mutually exclusive** (§3) |
| Process | Every promoted experiment carries a pre-registered hypothesis, a predicted LB, a measured residual | **Met** — 34 ledger rows, 31 submissions, all with residuals |

---

## 2. What the loop actually contributed

This is the part worth being honest about, because the headline number does not come from where the
project's name suggests.

**No LLM-generated model ever beat the hand-written reference.** Best loop CV was 0.965142 against
`xgb_te`'s 0.968587 — a 0.00345 shortfall, roughly 60× the measured noise floor. Across 48
iterations the gap never closed.

**In the final stack, our 21 own members carry 7.5 % of the total |weight|; the 19 loop-generated
ones carry 6.8 %.** The other 92 % is public OOF. Put plainly:

```
125 public members took the stack 0.96995 → 0.97105 in one day
~15 loop iterations over the same period moved it by zero
```

**Failure rate was 23 %** — 11 of 48 iterations never produced a score even after up to 10 repair
attempts. The failures were structural and repeatable, not creative: timeouts from an over-sized
estimator (4), a wrong keyword argument to a sklearn class (`LogisticRegression(alpha=…)`), NaN/inf
leaking into an MLP, and one train/test consistency rejection.

So what did the loop buy? Three things, none of them "a better model":

1. **Members the greedy stacker wanted.** `loop13_72118` — a Ridge with polynomial features, CV
   0.954592, fourteen thousandths *below* the gbm best — is the only own member in the 6-member
   stack at 0.969041. It was picked for disagreement, not accuracy.
2. **A ledger of pre-registered predictions.** 31 submissions, each with a predicted LB recorded
   before the score came back. That is what made the `loop06` diagnosis possible at all.
3. **The harness.** Every defect the loop surfaced was in deterministic code we then fixed
   permanently. The agent was the fuzzer; the harness was the product.

---

## 3. Learnings

### 3.1 A CV number is not a result

`loop06_42048` produced the best CV any iteration had reached — 0.962796 — and scored **0.93729** on
the leaderboard. A −0.0267 residual against the +0.0005 the honest iterations produced.

It had built a frequency encoding inside `make_features`, which the harness calls once per frame, so
train and test got different mappings. The two mappings correlate at 0.9987 and differ in the 1e-4
range — harmless for a shallow model, fatal for the `depth=10, n_estimators=2000` XGBoost the same
plugin chose, which splits at exactly that granularity. Cross-validation only ever uses the train
mapping, so **CV was structurally incapable of seeing the bug**, and the promotion rule — a paired
bootstrap against a measured noise floor — promoted it cleanly.

> A promotion rule built on CV cannot reject a plugin whose CV is the thing that is wrong.

The fix went into the harness, not the prompt: every returned feature column is now compared between
`X_train` and `X_test` by population stability index before any fold is fitted. Honest features score
~0.0002 here; the two broken ones scored 0.064 and 0.069. The gate rejects `loop06` and passes
`loop01`–`loop03` unchanged.

**Generalises as:** in an autonomous loop, the validation metric is also an attack surface. Every
guard that lives only in a prompt is a guard the model will eventually walk past.

### 3.2 Banning the right answer is the wrong fix

The plugin contract forbade touching `y` inside `make_features`, which made target encoding
structurally impossible — while the two best members in the ledger *were* target encoding.

Measured consequence: the orchestrator proposed target encoding in roughly 8 of every 10 strategies,
and every one was unimplementable. That rate did not move when the prohibition was stated in the
strategy prompt (6/6 violations), nor when repeated as a hard constraint in the system prompt (still
6/6), nor with rejection-and-resample (2/8 usable).

It did not move because target encoding is the *correct* technique for this dataset. Once the harness
computed it and injected `te_<col>` before `make_features` ran — plugin still never sees `y` —
`propose()` returned an implementable strategy **8/8**.

**Generalises as:** when a model repeatedly violates a constraint, check whether the constraint is
blocking the correct answer. Supply the capability safely rather than prohibiting it louder.

### 3.3 The search space is what the model can name

`model_family` was a single JSON-schema enum listing four gradient-boosting engines. "The
orchestrator always picks a GBM" was not a preference — it was the entire space it could express.
Splitting it into `gbm` / `linear` / `nn` orchestrators, each with its own estimator set *and its own
data contract*, was a one-file change that produced 15 iterations in territory the loop could not
previously reach.

The contracts had to differ, not just the estimator lists: a GBM is handed NaN and integer codes and
does the right thing, while a linear model or an MLP crashes on the first and misreads the second.
A single shared contract had been silently encoding "gradient boosting" as the only valid answer.

### 3.4 Within a large pool, member quality carries no information about member value

Among our 21 members inside the 36-member stack, `corr(CV, |weight|)` was **0.53**. Inside the final
274-member stack, it was **0.02**.

The highest-weighted own members in the final stack are the ones the promotion rule *rejected*:

| member | CV | promotion status | \|w\| in final stack |
|---|---|---|---|
| `loop02_66935` | 0.963584 | rejected | 0.310 |
| `loop20_77585` | 0.942430 | rejected | 0.305 |
| `loop22_79235` | 0.958986 | baseline | 0.209 |
| `loop13_72118` | 0.954592 | promoted | 0.204 |
| `loop09_69453` | **0.965142** (our best) | promoted | 0.041 |

`loop20_77585` has the third-lowest CV of any linear model the project produced, and carries seven
times the weight of the best gbm model it produced.

**Generalises as:** promotion-by-CV and selection-into-a-stack are different objectives, and past a
few dozen members they stop being correlated at all. Keeping rejected runs' arrays cost nothing and
turned out to be where the value was. Do not delete the failures.

### 3.5 Diversity was the binding constraint, and we could not manufacture it

Our own 21 members correlate at 0.98–0.998 — near-copies — which is why greedy selection over them
alone takes three and stops at 0.968710. One public library of 7 members cross-correlates with ours
at 0.9479 and moved the stack on its own.

This is the mechanism behind the failed stretch goal, and the two halves are directly opposed:

| stack | CV | own-member weight share |
|---|---|---|
| 36 members | 0.969386 | **52.1 %** |
| 274 members | **0.970206** | 7.5 % |

You can have "CV ≥ 0.9700" or "≥ 50 % of our own weight". The pool that reaches the first is
necessarily dominated by the ingested one, because that is where the decorrelation lives. Writing
both into a success criterion on day 1 was a mistake we could not have caught before measuring.

### 3.6 A small local model is a mutation and repair operator, not a strategist

This was written into SPEC §0 as an assumption on day 1 and every subsequent measurement confirmed
it. `gpt-oss:20b` as orchestrator/critic and `qwen2.5-coder:14b` as coder, loaded one at a time on 24
GB, produced runnable code reliably — but the strategies they generated cycled through the same four
ideas (bin the TE columns, add interaction terms, add ratios, deepen the trees) for 48 iterations and
converged 0.0034 short of a pipeline a human wrote in an afternoon.

The critic's own verdicts document this. Across the 15 `linear` and `nn` iterations, its most
consistent recommendation — issued in 7 of the 10 that produced a verdict at all — was *"replace the
model with LightGBM or CatBoost"*: correct on CV, and exactly wrong for the stack, where those weak linear members turned
out to be the useful ones. A critic scored on the same metric as the thing it is criticising cannot
see value that metric does not measure.

What did work was giving the model a curated playbook to select from — 27 backlog entries carrying
deltas someone had already measured on this exact fold split, including 13 marked `rejected` so the
loop could not rediscover dead ends.

### 3.7 The defects were in the harness, and they looked like model weakness

All three defects that surfaced in the first live runs were ours:

- **The repairer could not see its own failing line.** The traceback handed to it was ~90 % pandas
  internals. Three repair attempts on a trivial `astype(int)`-on-NaN error all failed. Filtering the
  traceback to plugin frames and printing the offending source line fixed it — same model, same error.
- **The contract had a hole the model kept falling into.** It permitted pandas `category` dtype, but
  the harness calls `fit(X, y)` with no extra arguments, so CatBoost can never receive `cat_features`.
  The repairer oscillated between two errors forever, fixing each and reintroducing the other.
- **A failed Kaggle upload was logged as a submission**, burning a quota slot for an upload that never
  happened, because the return code was checked after logging instead of before.

Before the fixes: 0/2 iterations produced a runnable model, 6/6 repair attempts failed. After: 2/2
passed preflight on the first attempt with no repairs.

**Generalises as:** "the small model isn't good enough" is the most expensive wrong diagnosis
available. Check the scaffolding first — it is deterministic, so it is cheap to check and permanent
to fix.

### 3.8 Measure the noise floor before claiming an improvement

`state/noise_floor.txt` holds **0.00002868** — the std of repeated 5-fold *means* across partition
seeds, measured by re-running the best plugin three times. The fold-to-fold spread inside a single
run is roughly 5× larger and badly overstates how much the mean actually moves.

With the real floor in hand, the promotion rule requires a paired bootstrap `P(Δ>0) > 0.95` **and**
a delta of at least 2× the floor. That rule classified `loop23_79439` (Δ=+0.000084, p=0.925) as
`rejected` rather than as an improvement — a call that is only possible once the bar is a measurement
instead of a guess. Gains below the bar are logged `real-but-invisible`, and the first run of a family
is `baseline`, not `promoted`, because calling it an improvement would assert a comparison never made.

### 3.9 The leaderboard was the diagnostic, never the objective

Three of the seven finished Season 6 episodes had **zero** public top-10 teams survive into the
private top 10; in S6E7 the public winner finished private rank **440**. Above roughly 0.97110, a
fifth decimal on this board can be manufactured by choosing among near-identical predictions after
seeing public scores.

So the loop optimised honest out-of-fold AUC and used the leaderboard only to detect distribution
shift and leakage — a CV/LB *divergence*, not a low score, is the signature. That discipline is what
made `loop06` legible as a broken pipeline rather than a lucky one, and `public_lb_selection` sits in
the backlog permanently marked `rejected`.

An autonomous system can hold this line in a way a human watching a live leaderboard cannot. That may
be the most transferable advantage in the whole project, and it has nothing to do with model quality.

---

## 4. What we would do differently

1. **Ingest the public prediction pool on day 1, not day 3.** It was the single highest-leverage
   action available and it needed no LLM at all. The ranking in `harness/absorb.py` — predictions >
   measured findings > ideas — was learned in the wrong order.
2. **Score members for the stack, not for promotion.** The greedy selector should have been running
   continuously from iteration 1, with the orchestrator rewarded on marginal stack contribution
   rather than on solo CV. We measured that these diverge; we never closed the loop on it.
3. **Budget the failure modes.** 11 dead iterations, most of them timeouts from an over-sized
   estimator, were predictable enough to have been prevented by a cost model in the schema rather
   than discovered by a 2400-second wall clock.
4. **Do not write two success criteria that constrain the same quantity in opposite directions.**
   §14's stretch goal was unreachable by construction and we could not know that until §3.5 was
   measured.

## 5. Verdict

The system met its minimum and expected criteria and finished at a score that was public top-10 pace
when the project started. It did so as an **honest measurement substrate that happened to have LLMs
attached**, not as an LLM that learned to do machine learning.

The recursion works — verdicts genuinely fed the next iteration's strategy, and the mechanism is
visible in the ledger — but its output over 48 iterations was a set of decorrelated mediocre models,
not a strong one. Everything that moved the leaderboard came from the deterministic half: the frozen
split, the PSI gate, the supplied encodings, the measured noise floor, the greedy stacker, and the
public OOF pool.

That is a real result, just not the advertised one. The reusable artefact here is the harness — the
part the LLM was never allowed to touch.
