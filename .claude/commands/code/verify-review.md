---
description: Verify a CODE_REVIEW_<date>.md against the actual code and production config, then correct wrong findings in place
---

You are auditing a **code review document** — not the code. Someone (a human or an AI reviewer) produced a `CODE_REVIEW_<date>.md` full of findings. Your job is to check whether each finding is **actually true against the current code and the way the system runs in production**, then **rewrite the document in place** so every surviving claim is correct and every wrong one is fixed or retracted — with evidence.

A finding can be wrong in several distinct ways, and you must tell them apart:

- **Stale reference** — the code moved since the review; the claim may still be true but the `file:line` is off.
- **Problem is real** — the described bug/smell genuinely exists in the current code.
- **Problem is misdescribed** — the mechanism is wrong (e.g. "swallows the error" when the code actually re-raises).
- **Production makes it moot** — the code smell is real, but the running configuration already prevents the harm (a default that's never hit because the config always sets the key; a weak gate that's backed by a second, stronger gate; a "silent" failure that's actually loud because a required-access site crashes first). *This is the most common and most valuable correction* — a review that reads code defaults but ignores what prod actually sets will over-state impact.
- **The proposed fix is wrong** — the problem is real but the suggested change is incorrect or would make things worse (e.g. "default to 365 to match the other callers" when production actually runs 90).
- **Severity is miscalibrated** — real, but graded too high or too low for its true production impact.

## Scope resolution

`$ARGUMENTS` names the review file. If empty, find the most recent `CODE_REVIEW_*.md` at the repo root (sort by the date in the filename); if several are plausible, list them and ask which. State the resolved target in one line before starting.

## Steps

1. **Read the whole review file** and parse it into individual findings. Note its structure — severity codes (`C1`/`M8`/…), the `File / Problem / Why it matters / Fix` shape, the Summary, any Top-N list, section anchors. You will **preserve** that structure.

2. **For each finding, verify against the actual code — never from the review's own snippet.**
   - Open every file the finding references and **read the cited region in full plus enough surrounding context** to judge it. Follow imports into the functions it calls when correctness depends on them.
   - Check the concrete claims one at a time: Do the line numbers still point at the cited code? Does the described mechanism match what the code does? Is the data-flow / leakage / default / off-by-one it describes real?

3. **Then check how it actually runs — the step cheap reviews skip.** A finding about a default, a threshold, a gate, or a config-driven window is only as real as the *running configuration*:
   - Find the production config(s): the shipped `configs/*.yaml`, the live k8s ConfigMap(s), and any per-retailer / per-environment overrides that get deep-merged in. Read what they actually set.
   - Ask: does prod **set** the key the finding assumes is defaulted? Is the threshold it calls weak **backed by** another check? Does a required-access (`cfg["k"]`) site elsewhere make the "silent" failure actually crash loudly first? Does the value the proposed fix hard-codes match what prod actually uses?
   - A code-accurate finding whose harm is prevented by the running config is **Overstated**, not Confirmed — and pinning the exact config `file:line` that neutralizes it is the point of this command.

4. **Classify each finding** with a one-word verdict and a one-line evidence pointer (`file:line`):
   - **Confirmed** — real, correctly described, correctly graded, fix is sound.
   - **Overstated** — real in code but mitigated in production (or graded too high). State exactly what mitigates it and where.
   - **Understated** — real and worse / broader than described.
   - **Misdescribed** — the mechanism or the reference is wrong; give the correct one.
   - **Fix-wrong** — problem real, proposed fix incorrect; give the correct fix.
   - **Incorrect** — the described problem does not exist in the current code.
   - **Unverifiable** — depends on runtime data you can't observe; say exactly what evidence is missing. Never assert either way.

5. **Rewrite the review file in place** (edit `CODE_REVIEW_<date>.md` — this is the **one** file you may edit):
   - Add a short **`> Verified <date> — …`** note near the top: what you checked (code + which prod configs), and a tally (`N confirmed, M overstated, K incorrect, …`).
   - Under each finding, insert a **`- **Verification:**`** line: the verdict, the evidence (`file:line`), and — for Overstated / Fix-wrong / Misdescribed / Understated — the correction.
   - **Correct wrong content in place**: fix drifted line numbers/paths, fix a wrong mechanism description, replace a wrong `Fix:` block with the right one, and re-grade severity when it's off (note the change explicitly, e.g. *"Critical → Minor: mitigated by …"*).
   - **Preserve correct findings verbatim** — don't reword good work.
   - Keep the document's format, severity codes, and anchors intact so existing links still resolve. Update the Summary and any Top-N/quick-wins list if a verdict changes a finding's severity or retracts it — the summary must not contradict the verified findings.

6. **Report a short summary in chat**: the verdict tally, which findings you downgraded or retracted and why (one line each), and whether the review's headline conclusion still holds after verification.

## Rules

- **You edit only the review `.md`. Never edit source code, configs, or tests** — this command corrects the *review*, not the codebase.
- **Read before you rule.** Never confirm or reject a finding from its own snippet, or from a single grep line.
- **Always check production config, not just code defaults.** "The default is wrong" is not a live finding if prod always sets the key — say so with the config `file:line` that proves it.
- **Correct the remedy, not just the diagnosis.** A right problem with a wrong fix is still a wrong finding.
- **Don't invent new findings.** If you spot something the review missed, mention it in the chat summary only — this command verifies the existing document; offer `/code:code-review` for fresh findings.
- **Be honest about uncertainty.** Mark Unverifiable rather than guessing, and name the missing evidence.
- **Match severity to real production impact** — the same calibration the review itself should have used.
