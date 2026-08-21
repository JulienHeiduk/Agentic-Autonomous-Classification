"""Regenerate ROADMAP.md, BACKLOG.md and the daily report from the ledger (SPEC 6).

These files are derived artifacts -- never hand-edit them.

Run:  PYTHONPATH=. .venv/bin/python -m harness.report
"""
from datetime import date, datetime, timezone

from harness import config as C
from harness import ledger

DEADLINE = date(2026, 8, 31)
TARGET_CV = 0.9695


def _q(sql, args=()):
    with ledger.conn() as c:
        return c.execute(sql, args).fetchall()


def roadmap() -> str:
    days = (DEADLINE - datetime.now(timezone.utc).date()).days
    best = _q("SELECT exp_id, cv_auc, actual_lb FROM experiments "
              "WHERE cv_auc IS NOT NULL ORDER BY cv_auc DESC LIMIT 1")
    n_exp = _q("SELECT COUNT(*) FROM experiments")[0][0]
    by_status = dict(_q("SELECT status, COUNT(*) FROM experiments GROUP BY status"))
    subs = _q("SELECT COUNT(*), MAX(public_lb) FROM submissions")[0]
    used_today = ledger.submissions_today()

    b = best[0] if best else ("-", 0.0, None)
    lb_txt = f"{b[2]:.5f}" if b[2] else "not submitted"
    gap = TARGET_CV - (b[1] or 0)

    lines = [
        "# Roadmap", "",
        "*Generated from `state/ledger.db` — do not hand-edit.*", "",
        f"**{days} days to deadline** (2026-08-31 23:59 UTC)", "",
        "| | |", "|---|---|",
        f"| Best CV | **{b[1]:.6f}** (`{b[0]}`) |",
        f"| Its LB | {lb_txt} |",
        f"| Target CV | {TARGET_CV:.4f} — gap **{gap:+.6f}** |",
        f"| Experiments | {n_exp} ({', '.join(f'{k}: {v}' for k, v in sorted(by_status.items()))}) |",
        f"| Submissions | {subs[0] or 0} total, best LB {subs[1] or 0:.5f}, {used_today} used today |",
        "",
        "## Leaderboard reference (fetched 2026-08-21)", "",
        "| | AUC |", "|---|---|",
        "| Public #1 | 0.97140 |", "| Public #10 | 0.97123 |",
        "| Naive LightGBM baseline | 0.96320 (CV) |",
        "",
        "## Phase", "",
    ]

    phases = [
        ("Tier 0 — deterministic pipeline", n_exp >= 6),
        ("Calibration submissions (fit the CV→LB offset line)", (subs[0] or 0) >= 4),
        ("Ollama agent loop", by_status.get("promoted", 0) > 8),
        ("Public OOF pool ingestion", len(list(C.EXT_OOF.glob("*"))) > 0),
        ("Final selection by CV", days <= 1),
    ]
    for name, done in phases:
        lines.append(f"- [{'x' if done else ' '}] {name}")
    return "\n".join(lines) + "\n"


def backlog() -> str:
    rows = _q("SELECT idea_id, family, priority, status, source, measured_gain, rationale "
              "FROM backlog ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 "
              "WHEN 'done' THEN 2 ELSE 3 END, priority DESC")
    out = ["# Backlog", "",
           "*Generated from `state/ledger.db` — do not hand-edit.*", "",
           "`measured_gain` is a delta someone actually measured on this competition's "
           "frozen fold split, not an estimate. Rejected rows are pre-pruned dead ends: "
           "they exist so the loop does not rediscover them.", ""]
    cur = None
    for idea, fam, pri, status, src, gain, why in rows:
        if status != cur:
            cur = status
            out += ["", f"## {status}", "",
                    "| idea | family | Δ measured | source | rationale |",
                    "|---|---|---|---|---|"]
        g = f"{gain:+.5f}" if gain is not None else "—"
        out.append(f"| `{idea}` | {fam} | {g} | {src} | {why} |")
    return "\n".join(out) + "\n"


def daily() -> str:
    d = ledger.today()
    subs = _q("SELECT exp_id, public_lb, predicted_lb, message FROM submissions "
              "WHERE quota_day=? ORDER BY sub_id", (d,))
    exps = _q("SELECT exp_id, family, cv_auc, status, runtime_s FROM experiments "
              "WHERE substr(ts,1,10)=? ORDER BY cv_auc DESC", (d,))
    from harness.submit import offset_fit
    f = offset_fit()

    out = [f"# Daily report — {d}", "", f"## Experiments ({len(exps)})", "",
           "| exp | family | CV | status | s |", "|---|---|---|---|---|"]
    for e, fam, cv, st, rt in exps:
        cv_s = f"{cv:.6f}" if cv is not None else "—"
        out.append(f"| `{e}` | {fam} | {cv_s} | {st} | {rt or 0:.0f} |")
    out += ["", f"## Submissions ({len(subs)}/{10})", ""]
    if subs:
        out += ["| exp | predicted LB | actual LB | residual |", "|---|---|---|---|"]
        for e, lb, plb, _ in subs:
            r = f"{lb - plb:+.5f}" if (lb and plb) else "—"
            out.append(f"| `{e}` | {plb or 0:.5f} | {lb or 0:.5f} | {r} |")
    else:
        out.append("None.")
    out += ["", "## CV→LB offset", ""]
    out.append(f"`offset = {f[0]:+.4f} × CV + {f[1]:+.4f}` (fitted on {f[2]} submissions)"
               if f else "Not yet fitted — needs ≥3 scored submissions.")
    return "\n".join(out) + "\n"


def main():
    (C.ROOT / "ROADMAP.md").write_text(roadmap())
    (C.ROOT / "BACKLOG.md").write_text(backlog())
    p = C.REPORTS / f"daily-{ledger.today()}.md"
    p.write_text(daily())
    print(f"wrote ROADMAP.md, BACKLOG.md, {p.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
