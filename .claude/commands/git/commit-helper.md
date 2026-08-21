---
description: Create a git commit with conventional format type(scope): description
---

You are helping the user create a git commit following the Conventional Commits format:

```
<type>(<scope>): <description>
```

**The `(<scope>)` is MANDATORY.** Every commit message you produce MUST include a
parenthesised scope between the type and the colon. NEVER emit the bare
`<type>: <description>` form (e.g. `feat: add gradient boosting model` is WRONG —
it must be `feat(ml): add gradient boosting model`). If you cannot confidently
derive a scope from the changed files, ASK the user for one rather than dropping it.

## Steps

1. Run these commands in parallel to understand the current state:
   - `git status` (no `-uall` flag)
   - `git diff --staged` and `git diff` to see staged and unstaged changes
   - `git log --oneline -10` to match the repo's commit style

2. If there are no staged changes but there are unstaged changes, ask the user whether to stage specific files. Do NOT use `git add -A` or `git add .` blindly — prefer adding files by name.

3. Analyze the changes and determine:

   **Type** (must be one of):
   - `feat` — a new feature
   - `fix` — a bug fix
   - `docs` — documentation only
   - `style` — formatting, missing semicolons, etc. (no code change)
   - `refactor` — code change that neither fixes a bug nor adds a feature
   - `perf` — performance improvement
   - `test` — adding or fixing tests
   - `build` — build system or dependencies
   - `ci` — CI configuration
   - `chore` — other changes that don't modify src or test files

   **Scope** — REQUIRED, never omit (derived from the changed files / area of the codebase):
   - Examples: `ml`, `python`, `api`, `data`, `notebooks`, `config`, `deps`, `readme`
   - In this repo, prefer the project/area from the changed paths, e.g. a change
     under `src/kamino_nb_studio/<project>/` → scope `<project>`
     (`ksp`, `traffic-forecasting`, `smart-optimization`, …); `db/` → `db`;
     `.github/` → `ci`; `pyproject.toml`/lockfile → `deps`; a README → `readme`.
   - Pick the most specific meaningful scope based on the file paths changed.
   - You MUST always output a scope. If the changes are cross-cutting, choose the
     single most representative area (or a comma-joined pair like `ksp,db` only
     when genuinely co-equal); if you truly can't decide, ASK the user — do NOT
     fall back to a scope-less `type:` message.
   - If changes span multiple unrelated areas, suggest splitting into multiple commits.

   **Description**:
   - Imperative mood ("add" not "added", "fix" not "fixed")
   - Lowercase first letter
   - No trailing period
   - Concise (under ~70 chars)

4. Show the user the proposed commit message in this format and ask for confirmation before committing:

   ```
   type(scope): description
   ```

   Before showing it, VALIDATE the message against this shape — it MUST match
   `^<type>\(<scope>\): <description>$`. If there is no `(scope)` immediately
   after the type (i.e. the message looks like `type: description`), it is invalid:
   stop and add the scope (or ask the user for one). Do the same check again in
   step 5 on the exact string passed to `git commit -m` — never run the commit
   with a scope-less message.

5. After confirmation, create the commit:

   ```bash
   git commit -m "type(scope): description"
   ```

   Do NOT add a `Co-Authored-By` trailer or any Claude attribution.

6. Run `git status` after the commit to verify success.

## Examples

- `feat(ml): add gradient boosting model`
- `fix(python): handle empty dataframe in preprocessor`
- `docs(readme): add installation steps`
- `refactor(data): extract loader into its own module`
- `test(ml): cover edge cases in cross-validation`
- `chore(deps): bump pandas to 2.2`

## Rules

- ALWAYS include a `(scope)`. A message without a parenthesised scope
  (`feat: …`, `test: …`, `fix: …`) is INVALID and must never be committed —
  derive the scope or ask for it first.
- NEVER commit secrets (`.env`, credentials, keys). Warn the user if they are staged.
- NEVER use `--no-verify` to skip hooks.
- NEVER amend an existing commit unless the user explicitly asks.
- If a pre-commit hook fails, fix the underlying issue and create a NEW commit.
- If `$ARGUMENTS` is provided, use it as a hint for the description or scope.
