"""Model families the loop can explore, one orchestrator each.

A family is not just a list of estimators -- it is a different contract. A GBM is handed
NaN and high-cardinality integer codes and does the right thing; a linear model crashes on
the first and cannot use the second. Splitting the orchestrator by family lets each one be
told the truth about its own estimators instead of averaging the advice into something that
suits neither.

The point is diversity that a blender can use later. The playbook is blunt about the terms:
`arch_below_cliff` records that ResNet / MLP-PLR / factorization machines all scored solo
OOF below 0.966 and contributed **zero or sign-flipping** weight despite being the least
correlated members -- "contribution tracks solo OOF, not decorrelation; there is a visible
cliff at 0.966". So a family earns its place by being strong on its own, not by being
different. A linear family is worth exploring because a well-built linear model on encoded
features can clear that bar; it is not worth exploring merely because it disagrees.
"""

FAMILIES = {
    "gbm": {
        "label": "gradient-boosted trees",
        "model_family": ["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"],
        "guidance": """\
YOUR FAMILY: gradient-boosted trees (LightGBM, XGBoost, CatBoost, HistGradientBoosting).

  * Leave NaN in place. All three engines split on missingness natively, and imputing
    instead of adding an imputed column alongside measured WORSE on this data.
  * Feed the categorical columns as integer codes (the pd.Categorical recipe). No one-hot,
    no scaling -- trees are invariant to monotone transforms of a feature.
  * The high-cardinality lookup keys are usable raw; the te_/freq_ columns are there too.
  * Depth and the tree count are the levers that matter, but THE ARGUMENT NAMES DIFFER:
        LightGBM / XGBoost         n_estimators, max_depth, n_jobs=-1
        CatBoost                   iterations,   depth,     thread_count=-1
        HistGradientBoosting       max_iter,     max_depth, and NO n_jobs at all
    Passing n_estimators to HistGradientBoostingClassifier raises TypeError. Check the
    argument names for the engine you actually chose. Keep the values inside the
    smoke-test budget.""",
    },
    "linear": {
        "label": "linear and kernel-free convex models",
        "model_family": ["logistic_regression", "sgd_logistic", "ridge_classifier",
                         "elasticnet_logistic"],
        "guidance": """\
YOUR FAMILY: linear models (LogisticRegression, SGDClassifier(loss='log_loss'),
RidgeClassifier, ElasticNet-penalised logistic). These have REQUIREMENTS a tree does not:

  * NaN IS FATAL. Every one of these raises on a missing value. You MUST impute -- and per
    the measured result on this dataset, ADD the imputed column next to an isna() flag
    rather than silently replacing the original.
  * SCALE EVERYTHING. Coefficients are not scale-invariant and the penalty is applied in
    the scaled space. Put StandardScaler (or QuantileTransformer) in the Pipeline.
  * Raw integer codes are MEANINGLESS to a linear model -- code 3 is not "more" than code
    1. One-hot the three low-cardinality categoricals. For the high-cardinality lookup
    keys use the supplied te_/freq_ columns, which are already continuous and monotone in
    the target; do NOT one-hot 231 levels.
  * A linear model cannot form interactions by itself. If the strategy needs one, build it
    as an explicit product column.
  * Return the WHOLE thing as one sklearn Pipeline -- imputer, scaler, estimator -- because
    make_model must hand back a single estimator.

WHERE FITTING IS ALLOWED -- this is the rule that breaks plugins in this family:
make_features is called ONCE FOR TRAIN AND ONCE FOR TEST, so a scaler, imputer or encoder
fitted in there is fitted TWICE, on different data, producing two different mappings. The
run is rejected for it (one attempt scored PSI 1.265 on a single column that way).

    WRONG:  def _fe(df): X[NUMS] = StandardScaler().fit_transform(X[NUMS])
    RIGHT:  make_features does arithmetic only -- ratios, products, logs, flags.
            EVERY fitted step goes in the Pipeline returned by make_model, which the
            harness fits once per fold on that fold's training rows:

            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale",  StandardScaler()),
                      ("clf",    <your estimator>)])

To clear the 0.966 solo bar this family needs real feature work, not just a different
penalty. Ratios, products, spline or quantile expansions of the numeric columns are where
its headroom is.""",
        "extra_imports": ("sklearn",),
    },
    "nn": {
        "label": "neural network (sklearn MLP)",
        "model_family": ["mlp_shallow", "mlp_deep", "mlp_wide", "mlp_regularised"],
        "guidance": """\
YOUR FAMILY: neural networks. Only sklearn.neural_network.MLPClassifier is available --
there is no torch, tensorflow or keras in this environment, so TabNet, FT-Transformer,
MLP-PLR and anything else requiring them CANNOT be built. Every model_family value below
is an MLPClassifier configuration, not a different library:

    mlp_shallow      hidden_layer_sizes=(64,) or (128,)
    mlp_deep         (128, 64) or (128, 64, 32)
    mlp_wide         (512,) or (256, 256)
    mlp_regularised  any of the above with a larger alpha and early_stopping=True

REQUIREMENTS, all of which an MLP fails without:
  * NaN IS FATAL -- impute, and add the isna() flag alongside rather than replacing.
  * SCALING IS NOT OPTIONAL. An unscaled MLP will not converge at all; this matters more
    here than for a linear model. StandardScaler or QuantileTransformer in the Pipeline.
  * One-hot the three low-cardinality categoricals. Integer codes are meaningless to a
    dense layer. For the high-cardinality lookup keys use the supplied te_/freq_ columns
    -- never one-hot 231 levels into a dense net.
  * Return imputer + scaler + MLPClassifier as ONE sklearn Pipeline.
  * MLPClassifier takes NO n_jobs argument. Do not pass one.

WHERE FITTING IS ALLOWED -- this is the rule that breaks plugins in this family:
make_features is called ONCE FOR TRAIN AND ONCE FOR TEST, so a scaler, imputer or encoder
fitted in there is fitted TWICE, on different data, producing two different mappings. The
run is rejected for it (one attempt scored PSI 1.265 on a single column that way).

    WRONG:  def _fe(df): X[NUMS] = StandardScaler().fit_transform(X[NUMS])
    RIGHT:  make_features does arithmetic only -- ratios, products, logs, flags.
            EVERY fitted step goes in the Pipeline returned by make_model, which the
            harness fits once per fold on that fold's training rows:

            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale",  StandardScaler()),
                      ("clf",    <your estimator>)])

BUDGET -- an MLP is far slower to fit than a tree, and the plugin is smoke-tested on 8,000
rows in 120s. Set early_stopping=True (it uses an internal validation split, which is legal
because it is a constructor argument, not something passed at fit time) and keep max_iter
in the low hundreds. A wide net with max_iter=1000 will time out.

BE AWARE OF THE MEASURED PRIOR: the playbook entry `arch_below_cliff` records ResNet,
MLP-PLR and factorization machines all scoring solo OOF below 0.966 on this competition and
contributing ZERO or sign-flipping weight to a blend, despite being the least correlated
members. Being different is not enough here -- contribution tracks solo strength. An MLP
earns its place only by getting genuinely strong, which on tabular data means the feature
work matters more than the architecture.""",
    },
    "stack": {
        "label": "second-level meta-learner over other models' predictions",
        "model_family": ["logistic_meta", "ridge_meta", "lightgbm_meta", "mlp_meta",
                         "extratrees_meta"],
        "guidance": """\
YOUR FAMILY: STACKING. This is a SECOND-LEVEL model and the data is not what you are used to.

The train and test frames you receive contain ONE COLUMN PER MEMBER, and each column holds
that member's predicted probability for the row. There is no `age`, no `sleep_hours`, no
`te_notifications_per_day` -- the raw columns are not present and cannot be used. Your model
learns from other models' output.

The train frame's values are OUT-OF-FOLD predictions: row i's value came from a member that
never trained on row i. The test frame holds those members' test predictions. Both are
aligned to the frozen split, so the harness fold loop works unchanged.

WHAT IS WORTH BUILDING AT THIS LEVEL:
  * clip(log(p/(1-p)), +-30) -- logits, not probabilities. A linear meta-learner cannot
    combine probabilities sensibly; in logit space it can add and subtract them.
  * rank-gauss per column, which wins over logits once the pool is large and heterogeneous.
    Import it correctly -- `from scipy.stats import rankdata, norm`, NOT from
    sklearn.preprocessing, and import scipy submodules explicitly rather than writing
    `scipy.stats...` after a bare `import scipy`.
  * row statistics ACROSS members: mean, std, min, max, max-min. Disagreement between
    members is itself a signal -- but check it survives: the train values are out-of-fold
    while the test values come from fully-trained members, so test predictions are
    systematically more confident and a spread statistic can shift between the frames. One
    attempt was rejected for a `row_range` column at PSI 0.106. Per-column transforms are
    safer than row spreads.
  * pairwise differences between the strongest members.
  * NOT the raw columns. They are not in the frame.

MEASURED PRIOR, and it is unfavourable to elaborate models here:
  * `logit_or_rankgauss_stack` is DONE and worth about +0.0004 -- a plain linear stacker
    that can subtract. That is the number to beat.
  * `gbm_meta_learner` -- a non-linear meta-learner over the OOF matrix -- is REJECTED in
    the playbook. A tree on 11 highly-correlated probability columns overfits the fold
    structure fast.
  * The current members correlate 0.98-0.998 with each other, so there is very little for a
    complex model to find. Prefer a simple estimator with strong regularisation, and put
    your effort into the transforms and the row statistics rather than the architecture.

Everything else is unchanged: two functions, one estimator, no fold loop, no scoring.""",
    },
}

DEFAULT = "gbm"


def get(name: str) -> dict:
    if name not in FAMILIES:
        raise KeyError(f"unknown family {name!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[name]


def names() -> list:
    return sorted(FAMILIES)
