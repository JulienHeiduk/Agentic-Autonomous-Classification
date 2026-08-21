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
NOISE_FLOOR_PRIOR = 5e-5
MIN_DELTA_FLOORS = 2.0
MIN_P_IMPROVE = 0.95
