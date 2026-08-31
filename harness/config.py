"""Frozen constants. Nothing here changes once the first experiment has run."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATE = ROOT / "state"
ART = ROOT / "artifacts"
OOF_DIR = ART / "oof"
PRED_DIR = ART / "pred"
SUB_DIR = ART / "submissions"
EXT_OOF = ROOT / "external_oof"
REPORTS = ROOT / "reports"

for _d in (STATE, OOF_DIR, PRED_DIR, SUB_DIR, EXT_OOF, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# Everything below is READ OFF data/ by harness.profile rather than written here, so a
# different classification dataset needs no edit. The values are asserted against what the
# files actually contain at load time (harness/data.py), so a wrong inference fails loudly
# rather than corrupting a run.
#
# COMPETITION is the exception: nothing in a CSV names the Kaggle competition, so it stays a
# setting. Override any of these with an env var if the inference is wrong for your data.
from harness import profile as _profile

_P = _profile.profile()

COMPETITION = os.environ.get("COMPETITION", "playground-series-s6e8")
TARGET = os.environ.get("TARGET", _P["target"])
ID = os.environ.get("ID", _P["id"])

N_TRAIN = _P["n_train"]
N_TEST = _P["n_test"]

# --- FROZEN. The community convention; every public OOF library aligns to it. ---
N_FOLDS = 5
FOLD_SEED = 42

N_TRAIN = 691369
N_TEST = 296302

# Promotion rule (SPEC 5.3). The floor is measured by `python -m harness.confirm`, which
# writes state/noise_floor.txt; this value is only the fallback until it has been run.
# Columns the harness target-encodes and frequency-encodes out-of-fold, injected as
# te_<col> / freq_<col> before make_features is called. Derived: integer-valued columns with
# enough distinct values to behave as lookup keys rather than quantities. See
# harness/profile.py for the thresholds.
TE_COLUMNS = tuple(_P["high_cardinality"])
TE_SMOOTHING = 10.0
FREQ_COLUMNS = tuple(_P["high_cardinality"])

# Train/test consistency gate. train and test are iid draws here, so an honestly-built
# feature scores ~0.0002. loop06's per-frame frequency encoding scored 0.10 and 0.016, and
# turned a CV of 0.9628 into a leaderboard score of 0.9373. See harness/encode.py.
PSI_FAIL = 0.05
PSI_WARN = 0.01

NOISE_FLOOR_PRIOR = 5e-5
MIN_DELTA_FLOORS = 2.0
MIN_P_IMPROVE = 0.95
