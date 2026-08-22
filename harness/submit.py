"""Submission: quota manager, LB prediction, and the actual submit (SPEC 7).

Two hard rules live here, both enforced in code rather than documented as guidance:

  1. NEVER submit on a local counter alone. The remaining quota is reconciled against
     `kaggle competitions submissions` every time.
  2. Every submission records a PREDICTED LB before it is sent. That turns "is this
     leaking?" and "is this worth a slot?" into falsifiable questions -- and a CV/LB
     divergence, not a low score, is the signature of leakage.
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from harness import config as C
from harness import ledger

KAGGLE = str(C.ROOT / ".venv" / "bin" / "kaggle")
DAILY_CAP = 10          # user-stated; reconciled against the API every call


def _run(args, timeout=900):
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout)


def remote_submissions() -> pd.DataFrame:
    r = _run(["competitions", "submissions", C.COMPETITION, "-v"])
    if r.returncode != 0 or "," not in r.stdout:
        return pd.DataFrame()
    from io import StringIO
    lines = [l for l in r.stdout.splitlines() if l.count(",") >= 3]
    try:
        return pd.read_csv(StringIO("\n".join(lines)))
    except Exception:
        return pd.DataFrame()


def quota_remaining() -> int:
    """min(configured cap, what the API says we have left today)."""
    local_used = ledger.submissions_today()
    df = remote_submissions()
    remote_used = 0
    if not df.empty:
        col = next((c for c in df.columns if "date" in c.lower()), None)
        if col:
            d = pd.to_datetime(df[col], errors="coerce", utc=True)
            remote_used = int((d.dt.strftime("%Y-%m-%d") == ledger.today()).sum())
    used = max(local_used, remote_used)
    return max(0, DAILY_CAP - used), used, local_used, remote_used


def _score_column(df):
    for c in df.columns:
        if "public" in c.lower().replace("_", "") or c.lower() in ("score", "publicscore"):
            return c
    return None


def poll_score(timeout: int = 900, interval: int = 20):
    """Poll for the newest submission's public score.

    Kaggle scores asynchronously and this CLI cannot block on it, so the loop polls the
    submissions listing until the top row has a numeric score.
    """
    waited = 0
    while waited < timeout:
        df = remote_submissions()
        if not df.empty:
            col = _score_column(df)
            if col:
                v = pd.to_numeric(df[col], errors="coerce")
                if len(v) and pd.notna(v.iloc[0]):
                    return float(v.iloc[0])
        time.sleep(interval)
        waited += interval
        print(f"    waiting for score... {waited}s", flush=True)
    print("    timed out waiting for score -- run `--reconcile` later")
    return None


def offset_fit():
    """Fit LB = CV + offset(CV). The offset is NOT constant -- it shrinks as the model
    improves, because each test prediction averages N fold-models while each OOF
    prediction comes from 1, so a large stack has already averaged that variance away.
    A linear fit was 8x more accurate than a constant on 12 reference submissions."""
    with ledger.conn() as c:
        rows = c.execute(
            # Rejected experiments are excluded: a member is rejected precisely when its
            # CV did not mean what it claimed, so including it fits the line to a point
            # whose x-coordinate is wrong. loop06 (CV 0.9628 -> LB 0.9373) dragged the
            # prediction for loop10 down to 0.92239 against an actual 0.96621.
            "SELECT e.cv_auc, s.public_lb FROM submissions s JOIN experiments e "
            "ON e.exp_id=s.exp_id WHERE s.public_lb IS NOT NULL AND e.cv_auc IS NOT NULL "
            "AND e.status != 'rejected'"
        ).fetchall()
    if len(rows) < 3:
        return None
    cv = np.array([r[0] for r in rows])
    lb = np.array([r[1] for r in rows])
    a, b = np.polyfit(cv, lb - cv, 1)      # offset = a*cv + b
    return a, b, len(rows)


def predict_lb(cv_auc: float):
    f = offset_fit()
    if f is None:
        return cv_auc + 0.00120, "prior (needs >=3 scored submissions to fit)"
    a, b, n = f
    return cv_auc + a * cv_auc + b, f"linear fit on {n} submissions"


def write_submission(exp_id: str, path=None) -> str:
    _, pred = ledger.load_arrays(exp_id)
    test = pd.read_csv(C.DATA / "test.csv", usecols=[C.ID])
    assert len(test) == len(pred) == C.N_TEST
    sub = pd.DataFrame({C.ID: test[C.ID].to_numpy(), C.TARGET: pred})
    assert sub[C.ID].is_unique and np.isfinite(sub[C.TARGET]).all()
    path = path or C.SUB_DIR / f"{exp_id}.csv"
    sub.to_csv(path, index=False)
    return str(path)


def submit(exp_id: str, message: str = None, dry_run: bool = False):
    remaining, used, local, remote = quota_remaining()
    print(f"quota: {used}/{DAILY_CAP} used today (local {local}, remote {remote}) "
          f"-> {remaining} left")
    if remaining <= 0:
        print("REFUSING: daily quota exhausted")
        return None

    with ledger.conn() as c:
        row = c.execute("SELECT cv_auc, status FROM experiments WHERE exp_id=?",
                        (exp_id,)).fetchone()
    assert row, f"unknown experiment {exp_id}"
    cv, status = row
    assert status not in ("failed", "suspect-leak"), f"{exp_id} is {status}"

    plb, how = predict_lb(cv)
    path = write_submission(exp_id)
    message = message or f"{exp_id} cv={cv:.6f} predicted_lb={plb:.5f}"
    print(f"  CV {cv:.6f} -> predicted LB {plb:.5f}  [{how}]")
    print(f"  file {path}")

    if dry_run:
        print("  DRY RUN - not submitted")
        return None

    # CLI 2.2.4 has no --wait / --poll-interval (those exist only in newer builds), so the
    # upload is fire-and-forget and the score is polled for separately.
    r = _run(["competitions", "submit", C.COMPETITION, "-f", path, "-m", message])
    out = (r.stdout or "") + (r.stderr or "")
    print("  " + out.strip().splitlines()[-1] if out.strip() else "  (no output)")
    if r.returncode != 0:
        # Do NOT log it: a failed upload must not consume the local quota count.
        print(f"  SUBMIT FAILED (exit {r.returncode}) -- not logged, quota not consumed")
        return None

    sub_id = ledger.log_submission(exp_id, path, message, predicted_lb=plb)
    score = poll_score(timeout=900)

    if score is not None:
        resid = score - plb
        with ledger.conn() as c:
            c.execute("UPDATE submissions SET public_lb=? WHERE sub_id=?", (score, sub_id))
        ledger.record(exp_id, actual_lb=score, predicted_lb=plb, offset_residual=resid)
        print(f"  LB {score:.5f}   predicted {plb:.5f}   residual {resid:+.5f}")
        if abs(resid) > 0.001:
            print("  WARNING: large CV/LB divergence -- possible leakage. Investigate "
                  "before stacking this member.")
    else:
        print("  submitted; score not parsed yet -- run `reconcile` later")
    return score


def reconcile():
    """Backfill public_lb for submissions whose score was not captured at submit time."""
    df = remote_submissions()
    if df.empty:
        print("no remote submissions")
        return
    print(df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp_id", nargs="?")
    p.add_argument("-m", "--message")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quota", action="store_true")
    p.add_argument("--reconcile", action="store_true")
    a = p.parse_args()
    if a.quota:
        print(quota_remaining())
    elif a.reconcile:
        reconcile()
    elif a.exp_id:
        submit(a.exp_id, a.message, a.dry_run)
    else:
        p.print_help()
        sys.exit(1)
