# Agentic-Autonomous-Classification — System Specification

**Target competition:** Kaggle Playground Series S6E8 — *Predicting Smartphone Addiction*
**Metric:** ROC AUC · **Deadline:** 2026-08-31 23:59 UTC · **Written:** 2026-08-21 (10 days left)

---

## 0. Verdict up front

**Can we run the recursive loop on local Ollama? Yes — with one hard constraint on how the LLM is used.**

Everything was measured on this machine, not assumed (§1). Ollama serves usable code at 39–57 tok/s and
honours JSON-schema-constrained output, which is what makes a 7–9B model reliable enough to sit inside an
unattended loop.

The constraint: **a local 7–9B model cannot invent competition-winning ML strategy, and must never be asked
to.** A loop that says "write me a better pipeline" will spend 10 days generating broken notebooks. The loop
that works treats the LLM as a *bounded mutation and repair operator* over a typed search space, with a
deterministic Python harness owning the data, the CV split, the scoring and the submission. Strategy comes
from a curated playbook (§3) that is already largely written for us by the community.

**One finding reframes the whole goal.** Three of the seven finished Season 6 episodes had **zero** public
top-10 teams survive into the private top 10; in S6E7 the public winner finished private rank **440**
(measured in [`raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits`]). So "win the competition" must
not be operationalised as "maximise public LB". This system optimises **honest out-of-fold AUC** and uses the
leaderboard only as a distribution-shift sanity check. That is a real strategic edge for an autonomous
system: it can be disciplined about it in a way humans watching a live LB are not.

---

## 1. Measured facts (all verified on this machine today)

### 1.1 Data

| | |
|---|---|
| `train.csv` | 691,369 rows × 13 features + `addicted_label` |
| `test.csv` | 296,302 rows |
| Positive rate | **0.7094** (`sample_submission.csv` is this constant — the trivial 0.5 AUC baseline) |
| Numeric features | `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time` |
| Categorical | `gender` (3), `stress_level` (3), `academic_work_impact` (2) |
| Missingness | **every column 4.2 % – 19.4 %**; `social_media_hours` worst at 19.38 % |

Missingness is MCAR here (target rate among missing rows ≈ the global base rate), which matters: it makes the
missing-indicator features weak on their own but does *not* make imputation useless (§3.2).

### 1.2 Hardware & runtime

| | |
|---|---|
| Machine | Apple **M5 Pro**, 16 GPU cores, **24 GB unified memory**, 428 GB free |
| Python | 3.13.0; `.venv/` created via `uv`, has pandas/numpy/sklearn/lightgbm/pyarrow/kaggle |
| Ollama | **v0.32.14, server running** |
| Local models | `gemma4:26b` (17 GB), `gemma4:latest` (9.6 GB), `qwen3.5:9b` (6.6 GB), `qwen2.5:7b` (4.7 GB), `codegemma:7b` (5.0 GB) |
| Kaggle CLI | **2.2.4, authenticated** via `~/.kaggle/access_token` — verified against the live API |

### 1.3 LLM benchmarks (measured, not quoted)

| Model | Throughput | Notes |
|---|---|---|
| `qwen2.5:7b` | **57 tok/s** | produced correct, idiomatic LightGBM + StratifiedKFold + early-stopping code first try |
| `qwen3.5:9b` | **39–44 tok/s** | reasoning model; proposed the missing-mask feature idea unprompted |

**Structured output works and is essential.** `POST /api/chat` with a JSON Schema in `format` returned a valid,
schema-conforming object. **Trap found:** with a reasoning model you must send `"think": false` alongside
`format`, or the thinking budget consumes `num_predict` and `message.content` comes back **empty**. The first
test returned an empty string for exactly this reason. Every constrained call in this system sets
`think: false`.

### 1.4 Baseline and noise floor

Ran LightGBM on raw features, `StratifiedKFold(5, shuffle=True, random_state=42)`:

```
fold0 0.962434   fold1 0.963016   fold2 0.963611   fold3 0.963980   fold4 0.962989
OOF AUC = 0.963200        wall clock 154 s
```

Fast screening config (3-fold, 600 trees, no early stopping): **OOF 0.962290 in 31 s** — only 0.0009 below the
full run. That is a good cheap proxy and defines the two-tier evaluation in §5.2.

Single-OOF bootstrap SE = **0.000158**. But that is the wrong quantity to gate on: the community-standard
measurement here is the std of *repeated 5-fold means across partition seeds*, which is roughly **5e-5** on
this dataset. Paired comparisons on identical folds are tighter still. §5.3 sets the promotion rule from this.

### 1.5 Live leaderboard (fetched today)

```
1  Changye Li        0.97140
2  MILANFX           0.97136
3  Szymon Kłapiński  0.97130
…
10 Mikhail Naumov    0.97123
```

Our naive baseline is **0.963**. The gap to the top is **~0.008 AUC** — large for a Playground episode, and
essentially all of it is *known, published technique*. That is the good news: the loop is not searching blind.

---

## 2. Competitive intelligence — what actually moves the score

Pulled and read four top community kernels via `kaggle kernels pull`. Two are ablation studies of ~60 ideas
each, with measured deltas on the same fold split we use. This is the single highest-value input to the
system and it is why §3's playbook is evidence-backed rather than speculative.

### 2.1 The community fold convention — load-bearing

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, train.addicted_label)
# over train.csv in ORIGINAL FILE ROW ORDER — never sorted, never reindexed
```

Every public OOF library aligns to this split, **positionally** (most ship `.npy` with no ids). Our §1.4
baseline already uses it, so our OOF arrays are row-compatible with **six public libraries totalling 155+
members**. This is a strategic asset, not a footnote: it means our own models can be stacked directly against
the public pool. Freezing this split is a hard invariant of the system (§5.1).

### 2.2 What worked (measured deltas, from `tomasa2/s6e8-what-moved-the-score-and-what-didn-t`)

| Idea | Δ AUC | Note |
|---|---|---|
| **Target + frequency encoding on *every* column, continuous included** | **+0.0023 CV / +0.0017 LB** | the win; `daily_screen_time_hours` has 1,389 levels × ~500 rows each, so a smoothed target mean is well estimated |
| **Imputed columns *alongside* the NaNs** (never replacing) | **+0.0012** | same imputer, opposite sign if you replace — GBMs learn a default split direction that beats a point estimate |
| **CatBoost's own ordered target statistics** (raw string levels as `cat_features`) | **+0.0004** | best single model; beat a hand-rolled encoder that took a day to get right |
| Logit stacking over the library | +0.0004 | stack on `clip(log(p/(1-p)), ±30)`, not probabilities |
| Composition / ratio features | +0.0005 | largely superseded by TE |
| **Transductive frequency encoding** (counts over train+test) | **+0.00032 solo**, +0.000015 stacked | one line; legitimate — only labels are hidden, features are not |
| **Decimal lattice** — `frac(x)` and first decimal digit | +0.0001 | an **8.5-point** swing in addiction rate by first decimal digit: a generator fingerprint TE cannot see |
| Explicit `__missing__` level in encodings | +0.0001 on pandas ≥3.0 | see trap below |
| Lower learning rate (depth 5 @ 0.01) | +0.0002 | |
| A new architecture *above* the ~0.966 solo cliff | +0.00004 | only new family that paid |

### 2.3 What failed (equally important — these are pre-pruned from the backlog)

| Idea | Δ AUC | |
|---|---|---|
| Denoising-autoencoder features (768 cols) | **−0.00071** | worst block tested; 80 % of the damage was *dilution*, not the representation |
| …same, compressed to 32 PCs | −0.00014 | size engineered blocks relative to your real features |
| kNN / ordinal features from the source dataset | −0.00003 | 0.90 AUC standalone and still negative |
| Concatenating the original real dataset as extra rows | −0.00008 | **use it as a diagnostic, not as training data** |
| New architectures *below* the ~0.966 cliff (ResNet, MLP-PLR, FM) | 0 / sign flips | decorrelation did not save them |
| Rank / rank-gauss instead of logits at a *small* stacker | −0.00013 / −0.00002 | but see §2.5 — flips for a large pool |
| Non-linear (GBM) meta-learner | −0.0001 | |
| Nine variations on TE (pairwise, multi-resolution, smoothing sweeps) | −0.0004 … +0.00002 | plain single-feature TE at smoothing 10 is the sweet spot |

Two transferable rules the ablations establish, both encoded into the critic prompt in §4:

- **"Marginal signal is not incremental value."** A feature with 0.90 solo AUC made the model *worse*. The
  question is not *does this correlate with the target* but *what does this let the model ask that it
  currently cannot*.
- **"Use gain to discard the bottom, never to pick the top."** Importance screens rank their own output
  badly: the top-8 features by gain contributed nothing while 20 smaller ones carried the entire gain.

### 2.4 Data-structure facts worth hard-coding

- **Generator invariant:** `daily_screen_time_hours = social + gaming + work + other`, `other ≥ 0`, **zero
  violations** in 691k rows — and violated in 60.7 % of the real source dataset. The structure is manufactured;
  exploit it, don't rationalise it.
- **`notifications_per_day` and `app_opens_per_day` are lookup keys, not quantities.** Adjacent integer values
  differ in target rate by **0.22 on average**, 22× what sampling noise allows. Those two columns alone reach
  AUC 0.83. This is why target encoding pays so much.
- **pandas ≥ 3.0 trap:** `df[c].astype(str)` preserves NA instead of writing `"nan"`, so `groupby` silently
  drops 4–20 % of rows from every level statistic. Nothing errors. Use
  `.astype(object).fillna("__missing__").astype(str)` and **assert `groupby(...).size().sum() == len(df)`**.
  Our venv will be on pandas 3.x — this is a live hazard, not a historical one.

### 2.5 The public OOF pool (the other half of the leaderboard)

From `adarsh1077/s6e8-diversity-beats-strength` (LB 0.97113, 177 members): the top of the leaderboard is
substantially assembled from **shared public prediction files**. Six public libraries publish OOF+test pairs
on the frozen split. Findings:

- **"Disagreement, not accuracy."** Three of the four highest-weighted members were built by *removing* the
  target encoder. A plain logistic regression at 0.9589 outranked eight stronger GBMs.
- Our own models are worth **~1.8×** a public member (leave-one-author-out), not the ~6× a naive
  measure-at-moment-of-addition suggests.
- **Meta-model:** rank-transform each member → normal quantile (rank-gauss) → **L2 logistic, fitted nested**
  on the five frozen folds. Rank-gauss beat logit space by +0.00008 *on a large heterogeneous pool* (the
  opposite of §2.3's small-library result — pool size decides).
- **`StandardScaler` is mandatory**, and a non-converged lbfgs fit **reads higher than the truth**. Assert
  `max(n_iter_) < max_iter`.
- **Quarantine before stacking:** hash-dedupe (a duplicate silently doubles a member's weight — two were
  found in the public pool), KS drift check on OOF↔test rank distributions, and drop degenerate members —
  *except* deliberate signed correctors, which can have solo AUC below 0.5 by design.

### 2.6 The CV→LB relationship

Twelve submissions from one pipeline give the mapping:

```
offset = LB − CV,  ranging 0.00150 → 0.00109 as CV improved
fit:    offset = −0.1056 × CV + 0.1035      mean |error| 0.000013
        constant offset                     mean |error| 0.000108   (8× worse)
```

Mechanism: each test prediction averages 5 fold-models while each OOF prediction comes from 1, so a large
stack — which has already averaged that variance away — has a smaller offset. **CV does not estimate the LB,
but CV *differences* do, and every real CV gain appeared on the LB in the same order.**

This is directly actionable for an autonomous system: **predict the LB before submitting**, then compare. It
turns "is this leaking?" and "is this worth a submission slot?" into falsifiable questions, and a
CV/LB *divergence* — not a low score — is the signature of leakage.

---

## 3. The playbook (the LLM's search space)

The orchestrator does **not** free-associate. It selects and parameterises entries from a versioned
`playbook.yaml`. Seeded from §2, priority-ordered by measured evidence:

### 3.1 Tier 0 — deterministic implementation, no LLM involvement (day 1)

These are known-good and worth ~0.006 AUC. Writing them by hand is faster and safer than supervising a 7B
model writing them.

1. Frozen fold split + OOF/test artifact contract (§5.1).
2. XGB regressor imputation of the 9 numeric columns, fit on train+test, **added as extra columns**.
3. Composition features: `resid`, `leisure`, `social_frac`, `work_frac`, `leisure_frac`, `resid_frac`,
   `wk_ratio`, `week_total`, `awake_screen_frac`, `free_time`, `notif_per_open`, `min_per_open`.
4. Missingness flags for all 12 columns.
5. Nested target encoding + frequency encoding on all 12 columns, smoothing 10, explicit `__missing__` level,
   **transductive counts** for frequency.
6. Decimal lattice: `frac(x)`, `first_decimal(x)` for the 6 fractional columns.
7. CatBoost with raw string levels as `cat_features` (its own ordered target statistics).
8. Nested rank-gauss + StandardScaler + L2 logistic stacker with convergence assertion.

**Expected landing zone: CV ≈ 0.9689, LB ≈ 0.9701.** That is roughly top-quartile before the loop runs a
single iteration.

### 3.2 Tier 1 — the LLM loop's actual search space

| Family | Parameterised by the LLM |
|---|---|
| `feature` | new feature functions over a whitelisted namespace (`np`, `pd`, the 12 base columns); must declare *what the model currently cannot ask* |
| `model` | LGBM / XGB / CatBoost variants — depth, leaves, learning rate, colsample, `dart`, seed-averaging |
| `arch` | new families **only if projected solo OOF ≥ 0.966** (the measured cliff) |
| `hpo` | Optuna TPE within a fixed trial budget on the fast tier |
| `blend` | member selection, stacker family, rank-gauss vs logit — cheap, runs on stored OOF with **no retraining** |
| `data` | ingest/quarantine public OOF libraries (§2.5) |

### 3.3 Pre-pruned — the backlog ships with these already marked `rejected`

Autoencoder feature blocks, kNN-from-source-dataset features, original-dataset row concatenation,
sub-0.966 architectures, non-linear meta-learners, pairwise/multi-resolution TE, smoothing sweeps. Each
carries its measured delta and a citation so the orchestrator can see *why* and does not rediscover them.
**This is worth several days of loop time.**

---

## 4. Agents and model assignment

> **Status: built and running.** `loop.py` implements this section. What follows is the design;
> §4.5 records what the first live runs actually measured, including three defects that only
> appeared once real generated code hit the harness.

Four roles, all served by Ollama. Every call is JSON-schema-constrained with `think: false`.

| Role | Model | Job | Why this model |
|---|---|---|---|
| **Orchestrator** | `qwen3.5:9b` | Read ledger → pick next playbook entry → emit a typed `ExperimentSpec` | reasoning model; benchmarked well on exactly this task |
| **Coder** | `qwen2.5-coder:7b` *(to pull)* | `ExperimentSpec` → a Python module implementing the plugin contract | code-specialised; `codegemma:7b` is the installed fallback |
| **Repairer** | same as Coder | traceback + source → unified diff patch, max 3 attempts | tight, mechanical task — well within 7B ability |
| **Critic** | `qwen3.5:9b` | run result + baseline → verdict, delta interpretation, backlog updates | applies the §2.3 rules as an explicit checklist |

**Memory budget (24 GB is the binding constraint).** `qwen3.5:9b` (6.6 GB) + `qwen2.5-coder:7b` (~4.7 GB)
= 11.3 GB resident, leaving ~12 GB for a training process that needs 4–8 GB on 691k rows with the full TE
feature set. That fits, but only just.

**Recommended: serialise instead.** Set `OLLAMA_MAX_LOADED_MODELS=1` and pass `keep_alive: 0` after each
call. The loop is phase-based anyway (plan → code → *train* → critique), model load costs ~5–15 s, and the
training phase runs 31 s–10 min. The overhead is noise and it removes all OOM risk. Do **not** use
`gemma4:26b` (17 GB) — it cannot co-exist with a training process on this machine.

**Honest limitation.** These models will produce mediocre *ideas*. They are being used for what they are
reliably good at: filling typed templates, writing small feature functions against a fixed contract, reading
tracebacks, and summarising numbers into a ledger. The ideas come from §3, which came from §2.

### 4.5 What the first live runs measured

Three defects surfaced only once generated code actually ran. All three were in **my harness**, not in the
models — which is the useful finding, because each one looked like "the 7B is too weak" until it was fixed.

**1. The repairer could not see its own failing line.** The traceback handed to it was ~90% pandas
internals; the one line that needed changing was buried in the middle. Three repair attempts on a trivial
`astype(int)`-on-NaN error all failed. `harness/plugin_runner.py:focused_error` now filters the traceback to
frames from the plugin file and prints the offending source line. Same model, same error → fixed.

**2. The contract had a hole the model kept falling into.** It permitted pandas `category` dtype, but the
harness calls `model.fit(X, y)` with no arguments, so CatBoost can never receive `cat_features`. The repairer
oscillated: fixing the `.cat` accessor error reintroduced the CatBoost error and vice versa. Fixed by
mandating one encoding that works identically for all three engines
(`pd.Categorical(X[c], categories=LEVELS[c]).codes`) and by stating in the contract that `fit` takes no extra
arguments. A repair-history block now also shows the model the errors it has already seen, so it can
recognise an alternating pair as a conflict rather than fixing them one at a time forever.

**3. A failed Kaggle upload was logged as a submission.** CLI 2.2.4 has no `--wait`/`--poll-interval` (those
exist only in newer builds, which is what the online docs describe). The command errored, the loop printed
"submitted", and the ledger burned a quota slot for an upload that never happened. Now the return code is
checked *before* logging, and the score is polled for separately.

**Effect of the fixes.** Before: 0/2 iterations produced a runnable model, 6/6 repair attempts failed. After:
2/2 iterations passed preflight on the **first** attempt with no repairs needed.

**Preflight.** Each plugin is first run on 8,000 rows before the full 691k-row fit, so a broken plugin costs
~15 s to discover instead of several minutes. Repair cycles are cheap; only code that already runs earns a
full training run.

**First clean 2-iteration run, end to end:**

| # | strategy the orchestrator chose | CV | predicted LB | actual LB | residual |
|---|---|---|---|---|---|
| 1 | `Baseline_Engineering_Missingness` — decimal-digit fingerprint, `resid`, missingness flags; LGBM d6/lr0.05/500 | 0.959642 | 0.96084 | **0.96093** | +0.00009 |
| 2 | `DeepTree_Ordinal_CatHash` — dropped the flags, ordinal-coded the lookup keys; LGBM d8/lr0.02/1000 | 0.959805 | 0.96100 | **0.96105** | +0.00005 |

Iteration 2 was chosen *because of* iteration 1's verdict: the critic wrote "drop the explicit missingness
flags as noise… increase max_depth to 8 while reducing learning_rate to 0.02", and the orchestrator did
exactly that. CV moved +0.000162 and the leaderboard moved +0.00012 in the same direction — small, but the
mechanism is confirmed working on real scores rather than on CV alone.

Both LB predictions landed within 1e-4 of the truth using only the §2.6 prior, before any offset line had
been fitted. Total wall clock for both iterations including submissions: about 6 minutes.

**Where this sits.** 0.9598 is below the hand-written reference pipeline (`tier0`, CV 0.9686) and well below
the public leader (0.97140). That is expected at iteration 2 and is the point of the ledger: the loop's job
over many iterations is to climb toward the techniques in §2, and §3.3's pre-rejected list exists so it does
not waste iterations on the dead ends. The reference pipeline is the bar, not the starting point.

---

## 5. Measurement substrate

This is the part that determines whether the loop improves or drifts. It is deterministic Python; the LLM
never touches it.

### 5.1 Frozen invariants

```python
FOLDS = StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)   # original row order
```

Never regenerated, never reseeded, never sorted. Persisted to `artifacts/folds.npy` and **hash-asserted at
the start of every run**. Every experiment writes:

- `artifacts/oof/{exp_id}.npy` — float32, 691,369 rows, positionally aligned
- `artifacts/pred/{exp_id}.npy` — float32, 296,302 rows

At ~2.8 MB + 1.2 MB per experiment, 500 experiments cost ~2 GB. Storing every OOF array is what makes
blending free: the `blend` family retrains nothing.

### 5.2 Two-tier evaluation

| Tier | Config | Cost | Purpose |
|---|---|---|---|
| **FAST** | 3-fold, 600 trees, fixed | **31 s** | screen; kill anything worse than baseline − 0.0005 |
| **FULL** | frozen 5-fold, early stopping | **154 s+** | promote; only FULL results enter the ledger as candidates |
| **CONFIRM** | FULL × 3 partition seeds | ~8 min | required before any submission; std of means = the noise floor |

Measured proxy fidelity: FAST 0.96229 vs FULL 0.96320. Rank-preservation is *assumed*, so the harness
tracks FAST↔FULL rank correlation over the first 20 experiments and widens the FAST kill threshold if it
degrades.

### 5.3 Promotion rule

> **Wired in.** `loop.py:judge` calls `harness/evaluate.py:promote` on every iteration, comparing the
> new OOF array against the best previous loop run on identical rows. The floor comes from
> `state/noise_floor.txt` (written by `harness.confirm`); until that file exists the loop uses the
> prior below and labels every judgement `PRIOR` so the weaker basis is visible rather than implied.
> The first iteration is recorded as `baseline`, not `promoted` — there was nothing to compare it to.


Baseline noise floor is ~5e-5 (std of repeated-CV means). A candidate is promoted iff:

1. paired bootstrap on identical folds gives `P(Δ > 0) > 0.95`, **and**
2. `Δ ≥ 2 × noise_floor` (≈ +0.0001), **and**
3. the sign is consistent across the 3 CONFIRM partition seeds.

Anything smaller is logged as `real-but-invisible` rather than promoted. Rule 3 exists because the
`diversity-beats-strength` ablations show sign-flipping is the characteristic failure of small deltas.

### 5.4 Leakage detector

For every candidate, the harness predicts LB from CV using the running linear offset fit (§2.6), and after
the submission scores, records the residual. A residual > 3× the historical MAD flags the experiment
`suspect-leak` and quarantines its OOF array from the stack. This catches the exact failure mode a target
encoder produces.

---

## 6. Tracking: roadmap, backlog, ledger

`state/ledger.db` (SQLite, single writer, WAL) is the source of truth.

```sql
experiments(
  exp_id TEXT PK, ts, parent_id, family, playbook_ref,
  hypothesis TEXT,            -- from the orchestrator, BEFORE the run
  what_it_lets_the_model_ask TEXT,   -- §2.3 rule, enforced non-null
  spec_json TEXT, code_sha TEXT, git_commit TEXT,
  tier TEXT,                  -- fast | full | confirm
  cv_auc REAL, cv_std REAL, delta_vs_best REAL, p_improve REAL,
  predicted_lb REAL, actual_lb REAL, offset_residual REAL,
  runtime_s REAL, status TEXT,  -- queued|running|failed|rejected|real-but-invisible|promoted
  verdict TEXT, repair_attempts INT
)
submissions(sub_id, ts, exp_id, filename, message, public_lb, quota_day, selected_final BOOL)
backlog(idea_id, family, priority, status, source, expected_gain, measured_gain, rationale)
```

Auto-generated from the ledger, rewritten every cycle:

- **`ROADMAP.md`** — phase, days remaining, current best CV/LB, distance to target, what's blocked.
- **`BACKLOG.md`** — ranked queue with status (`queued` / `running` / `promoted` / `rejected` /
  `real-but-invisible`), each row carrying its measured delta and source.
- **`reports/daily-YYYY-MM-DD.md`** — submissions spent, CV trajectory, promotions, failures, and the
  refitted CV→LB offset line.
- **Git commit per experiment**, message `exp/{exp_id}: {family} Δ{delta:+.5f}` — full audit trail, and
  `git bisect`-able if the stack ever regresses.

---

## 7. Submission policy

The user states **10/day**; the rules page requires auth to read, so the daemon **reconciles against
`kaggle competitions submissions -v` at every cycle and enforces `min(configured, observed_remaining)`**.
Never submit on a local counter alone. *(Open item: confirm the daily cap and the number of selectable final
submissions — see §11.)*

~10 days × 10 = **~100 slots**. Allocation:

| Phase | Slots | Purpose |
|---|---|---|
| **Calibration** (day 1) | 4 | baseline, Tier-0 stack, and two deliberately different models at similar CV — establishes the CV→LB offset line and confirms LB tracks CV |
| **Steady state** (days 2–9) | ≤6/day | only promoted candidates (§5.3), each with a predicted LB recorded *before* submission |
| **Reserve** | 2/day | held for repair verification and drift checks; expire unused |
| **Final selection** (day 10) | remaining | see below |

**Submission is gated on CONFIRM tier, never FAST.** Every submit uses `--wait` so the daemon blocks until
scoring completes and writes `actual_lb` back to the ledger in the same transaction.

**Final selection is by CV, not public LB.** Given §0 — three of seven S6 episodes had zero public top-10
survivors — the selection rule is: (1) best honest nested-CV stack, (2) most diverse stack within 1 noise
floor of it. A submission that is ahead on public LB but behind on OOF is **not selectable**. This is
written into the daemon as a hard constraint, not a guideline, because it is exactly the decision a system
watching a live leaderboard would get wrong.

---

## 8. Safety rails

The generated code is the only untrusted component. It is contained, not trusted.

**Immutable — the LLM cannot read-write these paths:** `harness/`, `state/`, the fold split, the scorer, the
submission writer, `playbook.yaml`.

**The plugin contract** — generated modules implement exactly this and nothing else:

```python
def build_features(train: pd.DataFrame, test: pd.DataFrame, ctx: Ctx) -> tuple[pd.DataFrame, pd.DataFrame]: ...
def make_model(fold: int, ctx: Ctx): ...   # returns an sklearn-style estimator
```

The harness calls them. It owns the folds, the fit loop, the OOF assembly and the scoring. A plugin that
touches `y` outside its fold indices, imports outside the allowlist, or opens a socket does not run.

**Execution sandbox:** subprocess, `RLIMIT_AS` 12 GB, 20-minute timeout, network disabled, cwd restricted to
a per-experiment scratch dir, import allowlist (`numpy pandas sklearn lightgbm xgboost catboost scipy`).
On non-zero exit → Repairer (max 3) → `status='failed'` with the traceback preserved.

**A crash is missing data, not a negative result.** The ablation notebook lost its second-best model for
weeks by logging a library error as "this doesn't help." Failed experiments are recorded as `failed`, never
as `rejected`, and are eligible for re-queue.

**Kill switches:** `state/PAUSE` halts after the current experiment; three consecutive failed repairs in one
family disables that family and files a backlog item for human review.

---

## 9. Repo layout

```
harness/          # deterministic, LLM-immutable
  folds.py  data.py  evaluate.py  sandbox.py  submit.py  ledger.py  offset.py
agents/
  orchestrator.py  coder.py  repairer.py  critic.py  ollama.py   # schema-constrained client
tier0/            # §3.1, hand-written
  impute.py  compose.py  encode.py  lattice.py  models.py  stack.py
plugins/          # LLM-generated, one module per experiment
playbook.yaml     # the search space
state/            # ledger.db, folds.npy, PAUSE
artifacts/        # oof/*.npy, pred/*.npy, submissions/*.csv
external_oof/     # quarantined public OOF libraries (§2.5)
reports/          # daily-*.md
loop.py           # the daemon
ROADMAP.md  BACKLOG.md  SPEC.md
```

---

## 10. Delivery plan

10 days left. **Tier 0 must land on day 1** or the loop has nothing to iterate on.

| Day | Deliverable | Gate |
|---|---|---|
| **1 (today)** | Harness + frozen folds + ledger + Tier-0 pipeline; 4 calibration submissions | CV ≥ 0.9685, offset line fitted |
| **2** | Ollama agents + sandbox + repair loop; loop runs 10 experiments unattended | ≥ 8/10 plugins execute without human help |
| **3** | Quota manager, `--wait` submission, daily report, roadmap/backlog generation | fully unattended overnight run |
| **4–5** | Public OOF ingestion + quarantine + nested rank-gauss stacker | pool of 155+ members stacking cleanly |
| **6–9** | Autonomous loop; human reviews `BACKLOG.md` once a day | CV ≥ 0.9695 |
| **10** | Freeze, CONFIRM-tier final runs, select by CV | 2 finals selected, neither chosen on public LB |

**Fallback if the local loop underperforms:** Tier 0 plus the public-OOF stack is a strong submission on its
own and does not depend on any LLM working. The loop is upside, not the foundation. That ordering is
deliberate.

---

## 11. What we need from you

**Already done ✅**

- Kaggle token — `~/.kaggle/access_token`, verified against the live API.
- Ollama running with usable models.
- Competition data in `data/`.

> One note on the token: you pasted it into this session, so it now exists in the conversation transcript.
> If that matters for your setup, rotate it at kaggle.com/settings/api — the file-based flow keeps working
> unchanged.

**Needed from you**

1. **Pull the coder model** (~4.7 GB): `ollama pull qwen2.5-coder:7b`. `codegemma:7b` is the fallback if you'd
   rather not.
2. **Confirm the submission cap** — 10/day is your number; the rules page needs auth to read. Also confirm how
   many final submissions are selectable (typically 1–2). The daemon reconciles against the API regardless,
   but the budget plan in §7 assumes 10.
3. **Two decisions** (§12).

**Needed but I can install** — `xgboost`, `catboost`, `optuna`; and the public OOF library datasets via
`kaggle datasets download` (they are CC0 and explicitly shared for stacking).

**Community examples: yes, and they are load-bearing.** Not as inspiration — as *measured priors*. The four
notebooks already pulled contain ablations of ~120 ideas with deltas on our exact fold split. Section 3.3
pre-rejects the failures, which is worth several days of loop time on its own, and §2.5's public OOF pool is
how the top of the leaderboard is actually built.

---

## 12. Two decisions for you

**(a) How aggressively do we use the public OOF pool?** Stacking 155 shared member arrays is standard, legal,
and how the current top 10 is constructed — but it makes our result substantially a re-blend of other
people's work, and §2.5 measures our own models at only ~1.8× a public member. *Recommendation: ingest it,
but require our own members to carry ≥ 50 % of the stacker weight.* Best private-LB expectation without the
result being wholly derivative.

**(b) Optimise for private LB or public LB?** They diverge sharply here. *Recommendation: private, i.e. honest
OOF only* — §0's base rates make public-LB chasing a negative-expectation strategy above ~0.9711. This is
already written into §7 as a hard constraint; say the word if you want it relaxed.

---

## 13. Risks

| Risk | Mitigation |
|---|---|
| 7B models produce unusable code | Tier 0 is hand-written and independent; plugin contract is narrow; 3-attempt repair; failures are data |
| Loop overfits CV | Frozen folds, 3-seed CONFIRM, sign-consistency rule, LB residual leak detector |
| pandas 3.x `astype(str)` silently drops 4–20 % of encoding rows | Hard assert `groupby().size().sum() == len(df)` in the encoder |
| 24 GB OOM | Serialised model loading, `RLIMIT_AS` cap, no `gemma4:26b` |
| Non-converged lbfgs reads *higher* than truth | Assert `max(n_iter_) < max_iter` in the stacker |
| Duplicate arrays double a member's stack weight | Hash-dedupe at ingestion (two duplicates exist in the public pool) |
| Public/private divergence | Final selection by CV; §7 forbids selecting on public LB |
| Only 10 days | Tier 0 day 1; the loop is upside, not the foundation |

---

## 14. Success criteria

| | Target |
|---|---|
| **Minimum** | Autonomous loop runs ≥ 24 h unattended, ≥ 20 experiments, no human code fixes |
| **Expected** | Honest nested CV ≥ 0.9695 (LB ≈ 0.9706) |
| **Stretch** | CV ≥ 0.9700 with ≥ 50 % of stacker weight on our own members |
| **Process** | Every promoted experiment has a pre-registered hypothesis, a predicted LB, and a measured residual |

The process criterion is the one that generalises past this competition. It is also the thing an autonomous
system can do better than a human: never submit without a falsifiable prediction on record first.
