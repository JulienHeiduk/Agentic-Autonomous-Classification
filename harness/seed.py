"""Seed the backlog from measured community ablations (SPEC 3.3).

Every row carries a MEASURED delta and a source. The rejected entries matter as much as
the queued ones: they stop the loop rediscovering ~15 dead ends, which is worth several
days of iteration on a 10-day competition.

Sources:
  T = tomasa2/s6e8-what-moved-the-score-and-what-didn-t          (~60 ideas, ablated)
  A = adarsh1077/s6e8-diversity-beats-strength                   (177-member stack)
  R = raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits

Run:  PYTHONPATH=. .venv/bin/python -m harness.seed
"""
from harness import ledger

# (idea_id, family, priority, status, source, measured_gain, rationale)
SEED = [
    # ---------- done in Tier 0 ----------
    ("te_all_columns", "feature", 100, "done", "T", 0.0023,
     "Target+frequency encoding on every column incl. continuous. The single biggest lever."),
    ("impute_alongside", "feature", 95, "done", "T", 0.0012,
     "XGB-imputed columns ADDED next to the NaN-bearing originals. Replacing them flips the sign."),
    ("cat_ordered_ts", "model", 90, "done", "T", 0.0004,
     "CatBoost's own ordered target statistics from raw string levels. Beat a hand-rolled encoder."),
    ("transductive_freq", "feature", 88, "done", "T", 0.00032,
     "Frequency counts over train+test. Only labels are hidden; features are not. One line."),
    ("decimal_lattice", "feature", 85, "done", "T", 0.0001,
     "frac(x) and first decimal digit. 8.5-point target-rate swing by digit; TE cannot see it."),
    ("logit_or_rankgauss_stack", "blend", 84, "done", "T/A", 0.0004,
     "Linear stacker that can subtract. Rank-gauss for a large pool, logits for a small one."),
    ("explicit_missing_level", "feature", 83, "done", "T", 0.0001,
     "__missing__ as its own encoding level. Mandatory on pandas>=3.0 (this repo runs 3.0.5)."),
    ("low_lr_deep", "hpo", 70, "done", "T", 0.0002,
     "depth 5 @ lr 0.01 with more rounds."),

    # ---------- queued ----------
    ("public_oof_pool", "data", 98, "queued", "A", None,
     "Ingest 6 public OOF libraries (155+ members) on the frozen split. Quarantine first: "
     "hash-dedupe, KS drift, degenerate check. Our own members measured at ~1.8x a public one."),
    ("no_te_members", "model", 92, "queued", "A", None,
     "Deliberately train members WITHOUT the target encoder. Three of the four "
     "highest-weighted members in the best public stack were built by removing it. "
     "Disagreement, not accuracy."),
    ("seed_averaging", "model", 60, "queued", "T", 0.0002,
     "3-seed fold averaging as a single model. +0.0002 solo but only +0.000013 inside an "
     "existing stack -- a dead lever if we are shipping a stack."),
    ("trig_lookup_cols", "feature", 58, "queued", "T", 0.00017,
     "sin/cos on notifications_per_day and app_opens_per_day. No periodicity exists; it "
     "works because a split on sin(2*pi*x/20) selects a UNION of disjoint intervals, which "
     "is cheap against a lookup that jumps 0.22 between neighbours. +0.00002 stacked."),
    ("ten_encoding_folds", "hpo", 55, "queued", "T", 0.0001,
     "10 inner encoding folds instead of 5."),
    ("arch_above_cliff", "arch", 50, "queued", "T", 0.00004,
     "New architecture ONLY if projected solo OOF >= 0.966. Contribution tracks solo OOF, "
     "not decorrelation; there is a visible cliff at 0.966."),

    # ---------- pre-rejected: measured dead ends ----------
    ("autoencoder_768", "feature", 0, "rejected", "T", -0.00071,
     "Denoising-autoencoder features. Worst block tested. 80% of the damage was DILUTION "
     "(colsample means most features offered at a split were autoencoder columns)."),
    ("autoencoder_32pc", "feature", 0, "rejected", "T", -0.00014,
     "Same, compressed to 32 PCs. Still negative. Size engineered blocks relative to real features."),
    ("knn_source_dataset", "feature", 0, "rejected", "T", -0.00003,
     "kNN/ordinal features from the original 7.5k-row seed dataset. 0.90 AUC standalone and "
     "STILL negative -- marginal signal is not incremental value."),
    ("concat_original_data", "data", 0, "rejected", "T", -0.00008,
     "Concatenating the real source dataset as extra training rows. Use it as a DIAGNOSTIC: "
     "the generator's accounting identity holds in 100% of competition rows and is violated "
     "in 60.7% of the real data."),
    ("arch_below_cliff", "arch", 0, "rejected", "T", 0.0,
     "ResNet / MLP-PLR / factorization machine, all solo < 0.966. Zero or sign-flipping "
     "contribution despite being the LEAST correlated members. Decorrelation did not save them."),
    ("gbm_meta_learner", "blend", 0, "rejected", "T", -0.0001,
     "Non-linear meta-learner over the OOF matrix."),
    ("pairwise_te", "feature", 0, "rejected", "T", -0.00040,
     "Pairwise target encoding, 36 pairs x 32 bins."),
    ("multires_te", "feature", 0, "rejected", "T", -0.00033,
     "Multi-resolution TE (raw + 300/100/30 bins)."),
    ("te_smoothing_sweep", "hpo", 0, "rejected", "T", -0.00030,
     "Smoothing 50/200 instead of 10. Plain single-feature TE at smoothing 10 is the sweet spot."),
    ("second_decimal", "feature", 0, "rejected", "T", 0.0,
     "Second decimal digit. A conditional control said it had 1.5x the spread of the first, "
     "but each band spanned ~56 values so the second digit nearly DETERMINED the exact value "
     "that TE already encodes. The control was measuring the thing it was holding fixed."),
    ("rank_stack_small_pool", "blend", 0, "rejected", "T", -0.00013,
     "Plain rank instead of logits at the stacker, on a SMALL all-GBM library. Note this "
     "flips for a large heterogeneous pool, where rank-gauss wins by +0.00008."),
    ("importance_screen_top", "feature", 0, "rejected", "T", 0.0,
     "Picking features by importance gain. The top-8 by gain contributed NOTHING while 20 "
     "smaller ones carried the entire effect. Use gain to discard the bottom, never to pick the top."),
    ("public_lb_selection", "blend", 0, "rejected", "R", None,
     "Selecting blend weights on public LB. 3 of 7 finished Season 6 episodes had ZERO "
     "public top-10 teams survive into the private top 10; S6E7's public winner finished "
     "private rank 440. Forbidden by SPEC 7."),
]


def main():
    for row in SEED:
        idea_id, family, priority, status, source, gain, rationale = row
        ledger.add_backlog(idea_id, family, priority, status, source, rationale,
                           measured_gain=gain)
    with ledger.conn() as c:
        n = c.execute("SELECT status, COUNT(*) FROM backlog GROUP BY status").fetchall()
    print("backlog seeded:", dict(n))


if __name__ == "__main__":
    main()
