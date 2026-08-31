"""Turn a public Kaggle kernel into something the pipeline can actually use.

There are three ways external work enters this system, and they are worth ranking honestly,
because the ranking is measured rather than assumed:

  1. ITS PREDICTIONS (harness/ingest.py). If the author published an OOF/test pair, ingest
     it. This is the only channel that has moved the leaderboard here: 125 public members
     took the stack from 0.96995 to 0.97105 in a day, while ~15 loop iterations moved it by
     zero. Always check for a companion dataset before reading the code.

  2. ITS MEASURED FINDINGS (data/NOTES.md). Free text appended verbatim to the prompt TASK.
     For facts a profiler cannot infer -- a generator invariant, a fingerprint, "these two
     columns are lookup keys". This is what carried S6E8's real signal.

  3. ITS IDEAS (the backlog table). Structured entries the orchestrator picks from, each
     with a status. `rejected` entries matter as much as `queued` ones: they stop the loop
     rediscovering dead ends.

This module handles 2 and 3, using the local LLM to SUMMARISE a notebook -- reading a
document and extracting claims, which is a task it is good at. It is not asked to invent
strategy, which is the thing measurement showed it cannot do.

    PYTHONPATH=. .venv/bin/python -m harness.absorb <owner>/<kernel-slug>
    PYTHONPATH=. .venv/bin/python -m harness.absorb --file some.ipynb
"""
import argparse
import json
import subprocess

from agents import ollama
from harness import config as C

KERNELS = C.ROOT / "external_kernels"
KAGGLE = str(C.ROOT / ".venv" / "bin" / "kaggle")

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "reproducible": {
            "type": "boolean",
            "description": "true only if the model could be built from numpy/pandas/"
                           "sklearn/lightgbm/xgboost/catboost alone",
        },
        "blocking_dependency": {"type": "string"},
        "reported_score": {"type": "string"},
        "measured_facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Claims about the DATA that a profiler cannot infer: invariants "
                           "between columns, generator fingerprints, which columns behave "
                           "as lookup keys. Only what the notebook states or demonstrates.",
        },
        "techniques": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idea": {"type": "string"},
                    "family": {"type": "string",
                               "enum": ["feature", "model", "hpo", "blend", "data", "arch"]},
                    "worked": {"type": "boolean"},
                    "detail": {"type": "string"},
                },
                "required": ["idea", "family", "worked", "detail"],
            },
        },
    },
    "required": ["reproducible", "blocking_dependency", "reported_score",
                 "measured_facts", "techniques"],
}

SYSTEM = """\
You read a Kaggle notebook and extract what another competitor could reuse. You report only \
what the notebook states or demonstrates -- never what you assume about the problem. If the \
notebook does not report a score, say so rather than guessing one.

Respond with JSON only."""


def pull(ref: str) -> str:
    """Download a kernel's source. Returns the notebook text."""
    KERNELS.mkdir(exist_ok=True)
    r = subprocess.run([KAGGLE, "kernels", "pull", ref, "-p", str(KERNELS)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"kaggle kernels pull failed: {r.stderr.strip()[:300]}")
    slug = ref.split("/")[-1]
    for ext in (".ipynb", ".py", ".r", ".R"):
        p = KERNELS / f"{slug}{ext}"
        if p.exists():
            return read(p)
    raise SystemExit(f"pulled but no source file found for {slug}")


def read(path) -> str:
    """Notebook or script as plain text."""
    if str(path).endswith(".ipynb"):
        nb = json.loads(path.read_text())
        parts = []
        for c in nb.get("cells", []):
            src = "".join(c.get("source", []))
            if c.get("cell_type") == "markdown":
                parts.append(src)
            elif c.get("cell_type") == "code":
                parts.append(src)
        return "\n\n".join(parts)
    return path.read_text()


def imports(text: str) -> set:
    """Top-level modules the source imports. Parsed, not inferred."""
    import re
    return set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", text, re.M))


def reproducible(text: str):
    """(bool, blocker) -- decided by reading the imports, never by asking the model.

    The local model was asked this first and got it wrong on the first notebook tried: it
    reported "uses only numpy, pandas, sklearn and lightgbm" for a file importing torch
    sixteen times. Reproducibility gates everything downstream, so it is answered by the
    same allowlist harness/sandbox.py enforces, not by a summary.
    """
    from harness.sandbox import ALLOWED_IMPORTS
    extra = {"copy", "random", "pathlib", "os", "sys", "time", "json", "gc", "logging",
             "matplotlib", "seaborn", "tqdm", "IPython", "warnings", "__future__"}
    blockers = sorted(imports(text) - ALLOWED_IMPORTS - extra)
    return (not blockers), ", ".join(blockers)


def analyse(text: str, model: str = None) -> dict:
    """Ask the local model what is reusable here."""
    # Truncated from the middle: a notebook's head states the approach and its tail reports
    # the result, and the bulk in between is usually plotting.
    if len(text) > 24000:
        text = text[:16000] + "\n\n[...]\n\n" + text[-8000:]
    user = (
        f"Here is a Kaggle notebook for a tabular binary-classification competition "
        f"(metric ROC AUC).\n\n"
        f"Our constraint: a model must be buildable from numpy, pandas, sklearn, lightgbm, "
        f"xgboost or catboost. Anything requiring torch, tensorflow, keras or another deep "
        f"learning framework is NOT reproducible for us -- say so plainly in that case.\n\n"
        f"=== NOTEBOOK ===\n{text}\n=== END ===\n\n"
        f"Extract what is reusable, as JSON."
    )
    found = ollama.chat_json(model or ollama.ORCHESTRATOR, SYSTEM, user,
                             FINDINGS_SCHEMA, temperature=0.2, num_predict=3000)
    ok, blocker = reproducible(text)
    found["reproducible"], found["blocking_dependency"] = ok, blocker
    return found


def apply(found: dict, ref: str, write_notes: bool = True, write_backlog: bool = True):
    """Route the findings into the two channels the prompts already read."""
    added = {"notes": 0, "backlog": 0}

    if write_notes and found.get("measured_facts"):
        p = C.DATA / "NOTES.md"
        prev = p.read_text() if p.exists() else ""
        block = f"\n<!-- from {ref} -->\n" + "\n".join(
            f"  - {f}" for f in found["measured_facts"]) + "\n"
        if block.strip() not in prev:
            p.write_text(prev + block)
            added["notes"] = len(found["measured_facts"])

    if write_backlog and found.get("techniques"):
        from harness import ledger
        for i, t in enumerate(found["techniques"]):
            idea_id = f"{ref.split('/')[-1][:24]}_{i}"
            ledger.add_backlog(
                idea_id, t.get("family", "feature"),
                # Unmeasured by us: priority below every seeded entry, so a curated idea
                # with a measured delta is always offered first.
                priority=40,
                status="queued" if t.get("worked") else "rejected",
                source=ref,
                rationale=t.get("detail", t.get("idea", ""))[:400],
                expected_gain=None,
            )
            added["backlog"] += 1
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", nargs="?", help="<owner>/<kernel-slug>")
    ap.add_argument("--file", help="a local .ipynb or .py instead of pulling")
    ap.add_argument("--dry-run", action="store_true", help="analyse but write nothing")
    a = ap.parse_args()

    if a.file:
        import pathlib
        ref, text = a.file, read(pathlib.Path(a.file))
    elif a.ref:
        ref, text = a.ref, pull(a.ref)
    else:
        raise SystemExit("give a kernel ref or --file")

    print(f"read {len(text):,} chars from {ref}")
    found = analyse(text)

    print(f"\n  reproducible here : {found['reproducible']}"
          + (f"  ({found['blocking_dependency']})" if not found["reproducible"] else ""))
    print(f"  reported score    : {found['reported_score'] or 'none stated'}")
    print(f"\n  measured facts ({len(found['measured_facts'])}):")
    for f in found["measured_facts"]:
        print(f"    - {f[:110]}")
    print(f"\n  techniques ({len(found['techniques'])}):")
    for t in found["techniques"]:
        print(f"    [{'worked' if t['worked'] else 'did not'}] {t['idea'][:70]}")

    if a.dry_run:
        print("\n  DRY RUN -- nothing written")
        return
    added = apply(found, ref)
    print(f"\n  wrote {added['notes']} facts to data/NOTES.md, "
          f"{added['backlog']} entries to the backlog")
    if not found["reproducible"]:
        print(f"  NOTE: the model itself is not reproducible here. Check whether the author "
              f"published an OOF dataset -- that channel has been worth far more.")


if __name__ == "__main__":
    main()
