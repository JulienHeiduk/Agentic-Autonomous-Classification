"""SQLite ledger -- the single source of truth for experiments and submissions (SPEC 6)."""
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from harness import config as C

DB = C.STATE / "ledger.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
  exp_id TEXT PRIMARY KEY,
  ts TEXT, parent_id TEXT, family TEXT, playbook_ref TEXT,
  hypothesis TEXT,
  what_it_lets_the_model_ask TEXT,
  spec_json TEXT, code_sha TEXT, git_commit TEXT,
  tier TEXT,
  cv_auc REAL, cv_std REAL, delta_vs_best REAL, p_improve REAL,
  predicted_lb REAL, actual_lb REAL, offset_residual REAL,
  runtime_s REAL, status TEXT, verdict TEXT, repair_attempts INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS submissions (
  sub_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, exp_id TEXT, filename TEXT, message TEXT,
  public_lb REAL, predicted_lb REAL, quota_day TEXT,
  selected_final INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS backlog (
  idea_id TEXT PRIMARY KEY,
  family TEXT, priority INTEGER, status TEXT, source TEXT,
  expected_gain REAL, measured_gain REAL, rationale TEXT
);
CREATE INDEX IF NOT EXISTS ix_exp_status ON experiments(status);
CREATE INDEX IF NOT EXISTS ix_sub_day ON submissions(quota_day);
"""


def conn():
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    return c


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record(exp_id: str, **kw):
    """Upsert an experiment row."""
    kw.setdefault("ts", now())
    if "spec" in kw:
        kw["spec_json"] = json.dumps(kw.pop("spec"), default=str)
    cols = ", ".join(kw)
    ph = ", ".join("?" for _ in kw)
    upd = ", ".join(f"{k}=excluded.{k}" for k in kw)
    with conn() as c:
        c.execute(
            f"INSERT INTO experiments (exp_id, {cols}) VALUES (?, {ph}) "
            f"ON CONFLICT(exp_id) DO UPDATE SET {upd}",
            [exp_id, *kw.values()],
        )


def save_arrays(exp_id: str, oof: np.ndarray, pred: np.ndarray):
    """Store OOF + test predictions. Positionally aligned to the frozen split."""
    assert len(oof) == C.N_TRAIN and len(pred) == C.N_TEST
    assert np.isfinite(oof).all() and np.isfinite(pred).all(), f"{exp_id}: non-finite"
    np.save(C.OOF_DIR / f"{exp_id}.npy", oof.astype(np.float32))
    np.save(C.PRED_DIR / f"{exp_id}.npy", pred.astype(np.float32))


def load_arrays(exp_id: str):
    return (np.load(C.OOF_DIR / f"{exp_id}.npy"),
            np.load(C.PRED_DIR / f"{exp_id}.npy"))




def log_submission(exp_id, filename, message, predicted_lb=None):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO submissions (ts, exp_id, filename, message, predicted_lb, quota_day) "
            "VALUES (?,?,?,?,?,?)",
            (now(), exp_id, str(filename), message, predicted_lb, today()),
        )
        return cur.lastrowid


def submissions_today() -> int:
    with conn() as c:
        return c.execute(
            "SELECT COUNT(*) FROM submissions WHERE quota_day=?", (today(),)
        ).fetchone()[0]


def add_backlog(idea_id, family, priority, status, source, rationale,
                expected_gain=None, measured_gain=None):
    with conn() as c:
        c.execute(
            "INSERT INTO backlog VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(idea_id) DO UPDATE SET "
            "priority=excluded.priority, status=excluded.status, "
            "measured_gain=excluded.measured_gain, rationale=excluded.rationale",
            (idea_id, family, priority, status, source, expected_gain, measured_gain, rationale),
        )
