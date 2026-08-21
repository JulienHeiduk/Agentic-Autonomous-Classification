---
description: Propose a PR title (and body) in conventional format type(scope): description
---

You are helping the user name a pull request following the same Conventional
Commits format the repo uses for commits (its PR titles become the squash-merge
commit, e.g. `fix(ksp): correctness fixes … (#128)`):

```
<type>(<scope>): <description>
```

**The `(<scope>)` is MANDATORY.** Every PR title you produce MUST include a
parenthesised scope between the type and the colon. NEVER emit the bare
`<type>: <description>` form (`feat: add gradient boosting model` is WRONG —
it must be `feat(ml): add gradient boosting model`). If you cannot confidently
derive a scope, ASK the user for one rather than dropping it.

## Steps

1. Determine the base branch (default `main`; if the current branch clearly
   targets another, use that) and gather context in parallel:
   - `git branch --show-current`
   - `git log --oneline origin/<base>..HEAD` — the commits this PR would contain
   - `git log origin/<base>..HEAD --format='%s' | grep -oE '^[a-z]+' | sort | uniq -c` — type mix
   - `git diff --stat origin/<base>...HEAD | tail -1` — size / areas touched
   - `git log --oneline -10 origin/<base>` — match the repo's title style

   If the branch has no commits ahead of the base, say so and stop. If it isn't
   pushed yet, note that the user must `git push -u origin <branch>` before opening.

2. Determine the PR title parts:

   **Type** (must be one of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
   `test`, `build`, `ci`, `chore`):
   - Pick the ONE type that best represents the PR's primary purpose — not
     necessarily the most frequent commit type. A PR whose headline is a new
     capability is `feat` even if it also carries `test`/`docs`/`chore` commits;
     a bug-fix PR is `fix` even with supporting refactors.
   - If the PR genuinely has no single primary purpose, that's a sign it should
     be split — say so.

   **Scope** — REQUIRED, never omit (derived from the changed files / area):
   - In this repo, prefer the project/area from the changed paths, e.g. a change
     under `src/kamino_nb_studio/<project>/` → scope `<project>`
     (`ksp`, `traffic-forecasting`, `smart-optimization`, …); `db/` → `db`;
     `.github/` → `ci`; `pyproject.toml`/lockfile → `deps`; a README → `readme`.
   - Pick the most specific meaningful scope spanning the PR's changes.
   - You MUST always output a scope. If the PR is cross-cutting, choose the single
     most representative area (or a comma-joined pair like `ksp,db` only when
     genuinely co-equal); if you truly can't decide, ASK the user — do NOT fall
     back to a scope-less `type:` title.

   **Description**:
   - Imperative mood ("add" not "added"), lowercase first letter, no trailing
     period, concise (aim under ~70 chars). GitHub appends ` (#N)` on squash-merge,
     so leave room.

3. Propose the title, and VALIDATE it against the shape — it MUST match
   `^<type>\(<scope>\): <description>$`. If there is no `(scope)` immediately after
   the type, it is invalid: add the scope (or ask). Show it as:

   ```
   type(scope): description
   ```

4. Write the body to a file so the user can open and edit it before anything is
   created. Do NOT paste the whole body into the chat — the file is the copy that
   matters, and a second copy in the transcript only goes stale the moment they
   edit it.

   **Where:** `<repo-root>/.git/PR_BODY.md`, resolving the root with
   `git rev-parse --show-toplevel`. Inside `.git/` on purpose:

   - git never tracks its own directory, so it cannot pollute `git status` or be
     committed by accident — unlike a file at the repo root, which needs a
     `.gitignore` entry to be safe;
   - it persists across sessions, unlike a temp/scratchpad directory, and the
     path stays short enough to read;
   - it is what git itself does (`.git/COMMIT_EDITMSG`, `.git/MERGE_MSG`).

   One file, overwritten per run — say so, since a second branch's body replaces
   the first.

   **Content: what the PR DOES, and nothing else.** A one-line summary, then
   bullet points grouped from the commit subjects, describing the changes.

   Do NOT add:
   - a "Reviewer notes" section, or any other commentary aimed at the reviewer;
   - caveats — what is untested, unverified, or worth watching after merge;
   - off-theme callouts, or suggestions to split a commit into its own PR;
   - measurements, evidence or rationale for a decision.

   Those belong in the code comments and the commit messages, which is where a
   reader will look for them afterwards. A PR description that carries them goes
   stale the moment the branch moves and duplicates what the diff already says.
   Keep it to the changes.

   Anything worth flagging to the user — an off-theme commit, an unverified path
   — say it in the CHAT reply instead, where it is a conversation rather than a
   permanent record.

   **Then print the absolute path on a line of its own**, so it is clickable in
   the terminal and they can open it directly:

   ```
   /Users/you/repo/.git/PR_BODY.md
   ```

5. Offer the command. Do NOT run it unless the user asks — creating a PR is
   outward-facing and their call:

   ```bash
   gh pr create \
     --title "type(scope): description" \
     --body-file "<repo-root>/.git/PR_BODY.md"
   ```

   `--body-file` rather than `--body`: the body is multi-line and usually holds
   backticks and quotes that would need shell escaping, and reading from the file
   at run time means any edit the user makes is picked up without regenerating
   the command. Add `--base <base>` if the target is not the default branch, and
   `--draft` if they want it opened as a draft.

   `--fill` may replace `--body-file` when the commit subjects already read well
   as a body and no reviewer notes are needed.

   If they do ask you to run it, print the PR URL `gh` returns.

## Examples

- `feat(ml): add gradient boosting model`
- `fix(python): handle empty dataframe in preprocessor`
- `docs(readme): add installation steps`
- `refactor(data): extract loader into its own module`
- `test(ml): cover edge cases in cross-validation`
- `chore(deps): bump pandas to 2.2`

## Rules

- ALWAYS include a `(scope)`. A title without a parenthesised scope
  (`feat: …`, `test: …`, `fix: …`) is INVALID and must never be proposed —
  derive the scope or ask for it first.
- Base the title on the ACTUAL commits/diff of the branch, never on assumptions.
- ALWAYS write the body to `.git/PR_BODY.md` and print its absolute path, so the
  user can read and edit it before the PR exists. Never leave the body only in
  the chat.
- Do NOT create the PR without explicit confirmation; propose the title/body first.
- Do NOT add any Claude attribution to the PR title or body.
- If `$ARGUMENTS` is provided, use it as a hint for the description or scope.
