---
description: Propose a git branch name in <type>/<kebab-description> format
---

You are helping the user name a git branch following the repo's convention:

```
<type>/<kebab-description>
```

Examples from this repo: `refactor/ksp-inference-pipeline`,
`feat/tf-search-per-retailer-topn`, `fix/traffic_forecasting_major_issues`,
`ci/dependabot-ignore-setuptools`.

**The `<type>/` prefix is MANDATORY.** Every branch name you propose MUST start
with one of the allowed types followed by a slash. NEVER propose a bare
description with no type segment (`drift-check-fix` is WRONG — it must be
`fix/drift-check-paths`).

## Steps

1. Gather context in parallel:
   - `git branch --show-current` — are we on `main` (start fresh) or already on a branch?
   - `git status --short` and `git diff --stat` — uncommitted work to characterise
   - `git log --oneline origin/main..HEAD` — any commits not yet on a named branch
   - `git branch -a` — match the existing naming style AND check the proposed name
     isn't already taken (local or remote)

   If `$ARGUMENTS` is provided, treat it as a description of the intended work
   (which may not exist on disk yet — then base the name on the description, not
   on a diff). If there are neither changes nor `$ARGUMENTS`, ASK what the branch
   is for rather than guessing.

2. Determine the parts:

   **Type** (prefix before `/`, must be one of `feat`, `fix`, `docs`, `style`,
   `refactor`, `perf`, `test`, `build`, `ci`, `chore` — same set as the
   commit-helper):
   - Pick the ONE type that best represents the branch's primary purpose. If the
     work spans types, choose the dominant one (a feature branch that also adds
     tests is `feat`).

   **Description** — a short kebab-case slug:
   - lowercase, words separated by hyphens, no spaces, no dates, ≤ ~5 words.
   - Lead with the project/area when the work is scoped to one, matching repo
     names: `ksp`, `tf`/`traffic-forecasting`, `smart-optimization`, `db`,
     `deps` — e.g. `fix/ksp-drift-check-paths`, `feat/tf-search-topn`.
   - Describe the change, not the files touched.

3. Validate the full name `"<type>/<description>"`:
   - It MUST match `^(feat|fix|docs|style|refactor|perf|test|build|ci|chore)/[a-z0-9._-]+$`
     (a legal git ref: no spaces, uppercase, or special chars; no `..`, no
     leading/trailing `-`, no `.lock` suffix). If it has no `<type>/` prefix it is
     invalid — fix it or ASK.
   - It MUST NOT already exist. Check `git branch -a` for `<name>` and
     `remotes/origin/<name>`; if taken, disambiguate (add a distinguishing word).

4. Propose the primary name plus up to 2 alternatives, then offer the command
   (do NOT run it without explicit confirmation):

   ```bash
   git checkout -b <type>/<description>            # branches from current HEAD
   git checkout -b <type>/<description> main       # branch from main instead
   ```
   - If the user is on `main` with uncommitted changes, note that `git checkout -b`
     carries those changes onto the new branch (usually what they want).
   - If they're already on a feature branch, note that HEAD isn't `main` — ask
     whether to branch from `main` instead so the new branch is clean.

## Examples

- `feat/ksp-segment-exclusion-threshold`
- `fix/tf-meta-model-per-series-metric`
- `refactor/ksp-inference-pipeline`
- `test/tf-config-parity`
- `ci/enforce-pr-title`
- `chore/deps-bump-pandas`

## Rules

- ALWAYS start with a `<type>/` prefix. A name without one is INVALID and must
  never be proposed — add the type or ask.
- kebab-case only; the full name must be a git-legal ref.
- Base the name on the ACTUAL changes or the described intent, never invent scope.
- Check for collisions (local + remote) before proposing; disambiguate if taken.
- Do NOT create or switch branches without explicit confirmation — propose first.
- If `$ARGUMENTS` is provided, use it as the description / intent hint.
