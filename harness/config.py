"""Frozen constants. Nothing here changes once the first experiment has run."""
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

COMPETITION = "playground-series-s6e8"
TARGET = "addicted_label"
ID = "id"

# --- FROZEN. The community convention; every public OOF library aligns to it. ---
N_FOLDS = 5
FOLD_SEED = 42

N_TRAIN = 691369
N_TEST = 296302

# Promotion rule (SPEC 5.3). The floor is measured by `python -m harness.confirm`, which
# writes state/noise_floor.txt; this value is only the fallback until it has been run.
# Columns the harness target-encodes out-of-fold and injects as `te_<col>` before
# make_features is called. These are the two documented high-cardinality lookup keys --
# the ones whose adjacent integer values shift the target rate by 0.22. See harness/encode.py.
TE_COLUMNS = ("notifications_per_day", "app_opens_per_day")
TE_SMOOTHING = 10.0

# Columns the harness frequency-encodes (fitted on train, applied to both frames).
FREQ_COLUMNS = ("notifications_per_day", "app_opens_per_day")

# Train/test consistency gate. train and test are iid draws here, so an honestly-built
# feature scores ~0.0002. loop06's per-frame frequency encoding scored 0.10 and 0.016, and
# turned a CV of 0.9628 into a leaderboard score of 0.9373. See harness/encode.py.
PSI_FAIL = 0.05
PSI_WARN = 0.01

NOISE_FLOOR_PRIOR = 5e-5
MIN_DELTA_FLOORS = 2.0
MIN_P_IMPROVE = 0.95
