# Agentic-Autonomous-Classification

A recursive loop in which **local LLMs do the machine learning**. An orchestrator model plans a
strategy, a coder model writes the pipeline, a deterministic harness runs and scores it on a frozen
fold split, the result is submitted to Kaggle, and the measured feedback — CV, leaderboard score,
verdict — becomes the context the *next* iteration plans from.

Target: Kaggle Playground **S6E8 — Predicting Smartphone Addiction** (ROC AUC, deadline 2026-08-31).
Design notes and the competitive intelligence behind the prompts: [`SPEC.md`](SPEC.md).

---

## Run it

```bash
export PYTHONPATH=.
.venv/bin/python loop.py --iterations 2                 # plan → code → run → judge → submit → feed forward
.venv/bin/python loop.py --iterations 2 --no-submit     # no Kaggle calls
.venv/bin/python loop.py --iterations 4 --rows 120000   # fast screening on a subsample

.venv/bin/python -m harness.confirm                     # measure the noise floor (do this once)
.venv/bin/python -m harness.submit --quota              # submissions left today
.venv/bin/python -m harness.report                      # rewrite ROADMAP.md / BACKLOG.md
```

Iteration numbering continues from the ledger, so a second invocation produces iteration 3, not
another iteration 1. `touch state/PAUSE` stops a long run cleanly after the current iteration.

Requires Ollama running locally. Models used (both already installed):

| Role | Model | Measured |
|---|---|---|
| Orchestrator / critic | `gpt-oss:20b` | 53–57 tok/s |
| Coder / repairer | `qwen2.5-coder:14b-instruct` | 29 tok/s |

Loaded one at a time (`keep_alive=0`) — 24 GB of unified memory will not hold two models plus a
training subprocess.

Swapping either model means checking its `think` setting first: the correct value differs per
model, and the wrong one returns HTTP 200 with empty content rather than an error. The measured
matrix and the per-model map are in `agents/ollama.py`.

## A CV number is not a result

`loop06` produced the best CV any iteration had reached — 0.962796 — and scored 0.93729 on the
public leaderboard. A -0.0267 residual, against the +0.0005 the honest iterations produced.

It had built a frequency encoding inside `make_features`, which the harness calls once per
frame, so train and test got different mappings. Cross-validation only ever uses the train
mapping, so CV could not see it. The promotion rule in `harness/config.py` compares CV deltas
to a measured noise floor and promoted it cleanly — a rule built on CV cannot reject a plugin
whose CV is the thing that is wrong.

The harness now rejects that class of bug directly: every feature column is compared between
`X_train` and `X_test` by population stability index before any fold is fitted. Honest features
score ~0.0002 here; the two broken ones scored 0.064 and 0.069. It rejects `loop06` and passes
`loop01`–`loop03` unchanged.

Two encodings are supplied so plugins have no reason to build their own:

| column | how it is built |
|---|---|
| `te_<key>` | target encoding, nested per outer fold, leak-free |
| `freq_<key>` | frequency, fitted on train, applied to both frames |

for `notifications_per_day` and `app_opens_per_day`. Full write-up in [`SPEC.md`](SPEC.md) §0b–0c.

## One iteration

```
orchestrator  reads every previous iteration's strategy + CV + LB + verdict
              → emits a JSON strategy, told explicitly to differ from what already ran
coder         → writes plugins/<exp_id>.py against a two-function contract
sandbox       static-checks it, runs it in a subprocess on the frozen 5-fold split
repairer      on failure: gets the traceback, returns a patched file (up to --repairs times)
kaggle        submits, blocks on --wait, records the public score
critic        writes a verdict: did it work, why, what should the next iteration change
              ↓
        all of it lands in state/ledger.db → becomes the next iteration's context
```

That last arrow is the recursion. `agents/orchestrator.py:build_context()` is where it happens.

## The contract generated code must satisfy

```python
def make_features(train: pd.DataFrame, test: pd.DataFrame): -> (X_train, X_test)
def make_model(seed: int): -> unfitted sklearn-compatible estimator
```

The harness owns the folds, the fit loop, the scoring and the artifacts. A plugin cannot score
itself, reshape the split, or see the target — so the usual ways an autonomous loop fools itself are
unavailable to it. `harness/sandbox.py` additionally rejects the file before it runs if it imports
outside an allowlist, calls `eval`/`exec`/`open`, or is missing either function.

## Layout

| Path | |
|---|---|
| `loop.py` | the driver |
| `agents/` | `ollama.py` (client), `orchestrator.py`, `coder.py`, `prompts.py` |
| `harness/` | **LLM-immutable**: `config`, `data`, `folds`, `evaluate`, `ledger`, `sandbox`, `plugin_runner`, `submit`, `report`, `confirm`, `seed` |
| `plugins/` | generated experiment modules, one per iteration |
| `state/` | `ledger.db` (source of truth), `folds.sha256`, `noise_floor.txt` |
| `artifacts/oof`, `artifacts/pred` | every experiment's predictions, positionally aligned |

Reference members `xgb_te` (CV 0.9686) and `lgb_te` (0.9682) remain in the ledger as the bar the
loop is climbing toward. The hand-written pipeline that produced them has been removed — it was off
the loop's execution path, and the arrays are what matter.

## Three things that make the measurements trustworthy

**The fold split is frozen and hashed.**
`StratifiedKFold(5, shuffle=True, random_state=42)` over `train.csv` in original row order — the
convention every public OOF library on this competition aligns to. `harness/folds.py` refuses to run
if it ever changes, because a silently different split invalidates every stored array.

**Nothing is promoted on a single number.** `loop.py:judge` calls `harness/evaluate.py:promote`, which
requires a paired bootstrap `P(Δ>0) > 0.95` **and** a delta of at least 2× the measured noise floor.
Smaller-but-real gains are logged `real-but-invisible`; the first run is `baseline`, not `promoted`,
because calling it an improvement would assert a comparison that was never made.

**The noise floor is measured, not assumed.**

```bash
PYTHONPATH=. .venv/bin/python -m harness.confirm     # re-runs the best plugin over 3 partition seeds
```

The fold-to-fold spread inside one run is *not* the uncertainty — it badly overstates how much the
mean moves under a different partition. The bar is the std of the repeated 5-fold **means**, written
to `state/noise_floor.txt`. Until it has run, the loop falls back to the prior in `config.py` and says
so on every judgement.

## Traps encoded into the prompts

Every column here is 4–19% missing, and that breaks generated code in ways a small model does not
anticipate. `agents/prompts.py` states these as rules because each one cost real repair cycles:

- `.astype(int)` on a column containing NaN raises `IntCastingNaNError`. Keep it float.
- `LabelEncoder` handles neither NaN nor a test value unseen in train.
- The harness calls `fit(X, y)` with no extra arguments, so CatBoost can never receive
  `cat_features` — which makes pandas `category` dtype unusable. One encoding is mandated instead:
  `pd.Categorical(X[c], categories=LEVELS[c]).codes`, which works identically for every engine.

Separately, on pandas ≥ 3.0 (this repo runs 3.0.5) `df[c].astype(str)` preserves NA instead of
writing `"nan"`, so a `groupby` silently drops every missing row from a level statistic. Nothing
errors. Any plugin that builds target encodings needs
`.astype(object).fillna("__missing__").astype(str)` — worth a leak probe against a shuffled target
before trusting such a plugin's CV.

## The leaderboard is not the objective

Three of the seven finished Season 6 episodes had **zero** public top-10 teams survive into the
private top 10; in S6E7 the public winner finished private rank **440**. The loop optimises honest
out-of-fold AUC and uses the leaderboard to detect distribution shift and leakage — a CV/LB
*divergence*, not a low score, is the signature of a leak. See `SPEC.md` §7.
