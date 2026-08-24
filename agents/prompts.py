"""Shared prompt material.

Generated from the data where it can be: TASK, EXAMPLE_PLUGIN and the categorical recipe all
come from harness.profile via agents.describe, so this module names no column of any
particular dataset.

Everything a small model needs in order to be useful here is stated explicitly. A 7-9B
model will not infer the plugin contract, the column list, or the fact that the harness
owns cross-validation -- so all three are spelled out every time.
"""

# TASK and EXAMPLE_PLUGIN are GENERATED from the data, not written here. The previous
# versions named this competition's columns directly, so pointing the loop at another
# dataset produced strategies for columns that did not exist. agents/describe.py renders
# both from harness.profile; data/NOTES.md is appended to TASK when present, which is where
# domain knowledge a profiler cannot infer belongs.
from agents import describe
from harness import profile as _profile

TASK = describe.task()

_CONTRACT_TMPL = '''\
PLUGIN CONTRACT -- your file must define exactly these two functions and nothing else at
module level except imports and constants:

    def make_features(train: pd.DataFrame, test: pd.DataFrame):
        """train has the target column; test does not. Return (X_train, X_test)."""
        return X_train, X_test

    def make_model(seed: int):
        """Return an UNFITTED sklearn-compatible estimator with .fit(X, y) and
        .predict_proba(X)."""
        return model

RULES -- violating any of these fails the run:
  * The harness owns cross-validation. DO NOT write a fold loop, DO NOT compute AUC,
    DO NOT touch the target inside make_features. You never see y.
  * `train` and `test` ALREADY CONTAIN `te_<key>` (out-of-fold target encoding) and
    `freq_<key>` (train-fitted frequency) for each high-cardinality key named in the task.
    Keep them in X -- they are there by default if you only drop the id and target columns.
    Never rebuild them.
  * te_<key> and freq_<key> are FLOAT columns, not categories. NEVER name them in
    cat_features / categorical_feature -- LightGBM fails with "Could not find
    categorical_feature te_<key> in data file". The only categorical
    columns are the low-cardinality ones listed in the task above, and the recipe below
    turns those into integer codes, so you do not need cat_features at all.
  * NEVER FIT A MAPPING INSIDE make_features. It is called ONCE for train and ONCE for test,
    so anything fitted from the frame it is given -- `value_counts()`, a fitted encoder, a
    per-frame mean or normalisation -- produces a DIFFERENT mapping for each frame. The run
    is rejected if it does, because cross-validation cannot detect it and the leaderboard
    can: one plugin scored CV 0.9628 and LB 0.9373 on exactly this.
        WRONG:  freq = df[c].value_counts(normalize=True); X[c] = df[c].map(freq)
        RIGHT:  use the provided freq_<c> column.
  * `te_` and `freq_` are RESERVED prefixes belonging to the harness. Use those columns, and
    derive from them freely, but do not name a feature of your own with either prefix.
  * Drop the id and target columns (named in the task above) from X_train. X_train and
    X_test must have IDENTICAL column names in the same order.
  * X_train and X_test must keep EXACTLY the row counts given in the task. Never drop rows.
  * The harness calls `model.fit(X, y)` with NO extra arguments. Your estimator must work
    that way. You CANNOT pass cat_features, eval_set, early_stopping or sample weights at
    fit time -- put everything in the constructor, and set a fixed n_estimators.
  * Allowed imports ONLY: numpy, pandas, sklearn, lightgbm, xgboost, catboost, scipy,
    math, itertools, collections, warnings, functools, re.
    No os, sys, pathlib, open(), file I/O, or network.
  * Return the model UNFITTED. make_model is called once per fold.
  * USE EVERY CORE, WHERE THE ESTIMATOR SUPPORTS IT. This machine has 15:
        LightGBM / XGBoost / RandomForest  ->  n_jobs=-1
        CatBoost                           ->  thread_count=-1
        MLPClassifier, RidgeClassifier, most LogisticRegression solvers
                                           ->  NEITHER. They take no n_jobs argument and
                                               passing one raises TypeError. Omit it.
    A library default, or a hardcoded number like n_jobs=8, leaves most of the machine
    idle and makes every fold slower for no benefit. No LLM is resident while your plugin
    runs, so the whole machine is yours.
  * KEEP THE MODEL CHEAP ENOUGH TO SMOKE-TEST. Before the real run, the harness fits your
    plugin on 8,000 rows with a 120s budget. A sane configuration finishes that in seconds.
    n_estimators/iterations in the low thousands with depth 10, or an ensemble of several
    such models, blows it -- and a timeout gives the repairer no traceback to work from.
    Prefer n_estimators <= 1500 and depth <= 8 unless the strategy explicitly needs more.

NaN RULES -- EVERY COLUMN HAS 4%-19% MISSING VALUES. These are the errors that actually
happen; read them before writing a line:
  * `.astype(int)` / `.astype(np.int8)` on a column containing NaN RAISES
    IntCastingNaNError. Keep such columns float, or `.fillna(-1)` FIRST and then cast.
    e.g.  X["d1"] = (np.floor(X[c] * 10) % 10)              # float, has NaN -- fine
    NOT   X["d1"] = (np.floor(X[c] * 10) % 10).astype(int)  # RAISES
  * Comparisons and string ops on a column with NaN raise TypeError. Guard with .fillna().
  * LightGBM, XGBoost and CatBoost all handle NaN natively. Leaving NaN in a numeric
    column is the SAFE default -- you do not need to impute anything.

COLUMN ALIGNMENT -- the single most common way a plugin fails. `make_features` is called
once per frame, so any feature built for one frame and not the other, or built in a
different order, ends the run with
    ValueError: make_features: train and test columns differ

Use exactly this shape. One function builds both frames, and the last line forces test to
train's columns and order:

    def _fe(df):
        X = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").copy()
        ...                                   # EVERY feature is added in here
        return X

    def make_features(train, test):
        X_train = _fe(train)
        X_test = _fe(test)[X_train.columns]   # <-- this line is what keeps them aligned
        return X_train, X_test

  * Put every feature inside _fe(). Never add one to X_train or X_test afterwards.
  * Never make a column conditional on the frame -- no `if c in df.columns`, no `if
    target in df`. A conditional column exists in one frame and not the other.
  * Do not reorder or select columns after that last line.

CATEGORICAL COLUMNS -- USE EXACTLY THIS, ALWAYS. Do not use pandas `category` dtype and do
not use LabelEncoder. Integer codes work identically for every engine, handle NaN, and
guarantee train and test agree:

%%CATEGORICAL_RECIPE%%

  Anything else -- `category` dtype without cat_features, LabelEncoder, get_dummies with
  differing columns between train and test -- fails. This recipe does not.

Reply with ONE fenced ```python block containing the complete file. No prose.
'''

STRATEGY_CONSTRAINTS = """\
WHAT A STRATEGY MAY ASSUME -- the plugin contract, stated in strategy terms.

A coder turns your strategy into exactly two functions, make_features(train, test) and
make_model(seed). The harness owns everything else: the fold split, the fitting, the
scoring, the submission. That rules some techniques IN and others OUT. Proposing an OUT
technique wastes the entire iteration -- no coder can implement it, at any size, and the
run fails instead of producing a measurement.

OUT -- never propose these:
  * Computing target encoding YOURSELF, or any other target statistic, inside the plugin.
    Not because it is a bad idea -- it is a good one -- but because the harness has already
    done it for you (see IN below) and the feature step is never given y.
  * A fold loop, OOF stacking over other models' predictions, a meta-learner, or computing
    AUC inside the plugin. The harness owns the 5-fold split and the scoring.
  * Early stopping, eval_set, sample weights, or cat_features passed at fit time. The
    harness calls model.fit(X, y) with no extra arguments, so everything must live in the
    estimator's constructor with a fixed n_estimators.
  * Pseudo-labelling or anything requiring the test labels.

IN -- all of these work:
  * TARGET ENCODING and FREQUENCY ENCODING, already built and supplied: `te_<key>` and
    `freq_<key>` for every high-cardinality key are in the data before the plugin runs. Propose
    strategies that USE them -- interactions, binning, combining with the raw keys. Do not
    propose creating them, and never propose a feature fitted separately on train and test:
    that produces a CV that does not survive the leaderboard, and the run is rejected.
  * Any feature computed from the feature columns alone: arithmetic, ratios, residuals,
    digit extraction, binning, value counts, missingness flags, interactions.
  * Any single estimator from sklearn, lightgbm, xgboost or catboost, configured entirely
    in its constructor.
  * An ensemble, but ONLY as one sklearn-compatible object -- e.g.
    VotingClassifier(estimators=[...], voting="soft"). Never a tuple or list of models:
    the harness calls .fit on whatever make_model returns. If you propose an ensemble,
    name VotingClassifier explicitly so the coder builds it correctly.
  * Preprocessing, as long as the whole thing is returned as a single sklearn Pipeline.

PARALLELISM -- the machine has 15 cores and nothing else runs while a plugin trains. Any
estimator you name should use all of them (n_jobs=-1, or thread_count=-1 for CatBoost).
Do not put a hardcoded thread count like thread_count=8 in `key_hyperparameters`.

BUDGET -- every plugin is smoke-tested on 8,000 rows with a 120s budget before the real run.
Keep `key_hyperparameters` within that: n_estimators/iterations in the hundreds to ~1500 and
depth <= 8 is ample here. Proposing 6000 iterations at depth 10 does not buy accuracy on this
dataset; it spends the iteration on a timeout that produces no measurement at all.
"""


ORCH_SYSTEM = """\
You are the orchestrator of an autonomous Kaggle loop. You choose ONE strategy per \
iteration, then a coder implements it and the harness measures it on a frozen 5-fold split.

You are judged on whether each iteration is MEANINGFULLY DIFFERENT from the last and \
whether it improves out-of-fold AUC. Repeating the previous iteration with a tweaked \
learning rate is a wasted iteration.

Target and frequency encoding of the high-cardinality keys is ALREADY DONE for you: the \
`te_<key>` and `freq_<key>` columns named in the task arrive in the data, computed \
out-of-fold by the harness. Build on them rather than proposing to create them, and never \
propose computing target statistics inside the plugin -- the feature step is not given the \
target column.

Respond with JSON only."""

CODER_SYSTEM = """\
You are a machine-learning engineer. You write one Python file implementing a strategy \
given to you. The file must run correctly the first time -- there is no human to fix it.

Write plain, defensive code. Prefer a small number of features that certainly work over a \
large number that might not. Reply with one fenced python block and nothing else."""

REPAIR_SYSTEM = """\
You fix a Python file that failed. You are given the file and the exact traceback.

Change as little as possible: fix the error, keep the strategy intact. Reply with one \
fenced python block containing the COMPLETE corrected file, nothing else."""

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy_name": {"type": "string"},
        "hypothesis": {"type": "string"},
        "what_it_lets_the_model_ask": {"type": "string"},
        "feature_engineering": {"type": "array", "items": {"type": "string"}},
        "model_family": {
            "type": "string",
            "enum": ["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"],
        },
        "key_hyperparameters": {"type": "string"},
        "differs_from_previous": {"type": "string"},
        "expected_cv_auc": {"type": "number"},
        # Which curated idea this implements. The enum is injected at call time from the
        # queued backlog (see orchestrator.strategy_schema), so the model must PICK from
        # the playbook rather than describe whatever the context already shows. Free-text
        # novelty checks were tried first and paraphrases walked straight through them.
        "playbook_ref": {"type": "string"},
    },
    "required": [
        "strategy_name", "hypothesis", "what_it_lets_the_model_ask",
        "feature_engineering", "model_family", "key_hyperparameters",
        "differs_from_previous", "expected_cv_auc", "playbook_ref",
    ],
}

EXAMPLE_PLUGIN = describe.example_plugin()


_P = _profile.profile()
CONTRACT = (_CONTRACT_TMPL
            .replace("%%CATEGORICAL_RECIPE%%", describe.categorical_recipe(_P))
            .replace("ID_COL", repr(_P["id"]))
            .replace("TARGET_COL", repr(_P["target"])))
