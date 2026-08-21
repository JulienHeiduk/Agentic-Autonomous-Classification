---
description: Explore the codebase and write a README for the repo or a specific folder
---

You are helping the user generate a README by first exploring the relevant code, then writing a clear, accurate README based on what actually exists.

## Scope resolution

- If `$ARGUMENTS` is provided, treat it as the target. It may be:
  - A folder path (e.g. `src/ml`, `notebooks/`) → write a README **for that folder only**, scoped to its contents
  - A short description / hint about which area to document → resolve it to the matching folder, ask the user if ambiguous
- If `$ARGUMENTS` is empty → write/refresh the **global README at the repo root** (`README.md`)

State the resolved target in one line before starting (e.g. "Target: `src/ml/README.md`").

## Steps

1. **Explore before writing.** Never write a README from assumptions. Gather the facts:
   - Read `package.json` / `pyproject.toml` / `requirements.txt` / `Cargo.toml` / etc. to learn the language, dependencies, and entry points
   - Use Glob to map the directory structure of the target scope
   - Read the most important files (entry points, main modules, configs, existing docs)
   - For a folder-scoped README, stay inside that folder — do not document the whole repo
   - If an existing README is present, read it first and preserve any human-written sections you cannot verify (don't delete content you can't replace with something better)

2. **Check for an existing README at the target path.** If one exists, plan an update rather than a full rewrite.

3. **Draft the README.** Use this structure as a default, but drop sections that don't apply:

   ```markdown
   # <Project / Folder name>

   <One- or two-sentence description of what this is and why it exists>

   ## Features            (only if there are real features to list)
   ## Installation        (only for the root README, or folders that are independently installable)
   ## Usage               (concrete examples, copy-pastable)
   ## Project structure   (only for the root README — show the top-level layout)
   ## Configuration       (env vars, config files — only if relevant)
   ## Development         (how to run tests, lint, etc. — only if relevant)
   ## License             (only if a LICENSE file exists)
   ```

   Rules for the content:
   - Be **concrete and accurate** — every command, path, and example must match what's actually in the repo
   - Prefer short sentences and code blocks over prose
   - Do NOT invent features, dependencies, or commands that don't exist
   - Do NOT add badges, emojis, or marketing fluff unless the user asks
   - For folder-scoped READMEs, focus on: what this folder contains, how its modules relate, and how to use them from outside
   - Match the tone of any existing docs in the repo

4. **Show the draft to the user before writing the file.** Print the proposed content and ask for confirmation or edits. If the user approves, write it with the Write tool to the resolved path.

5. **After writing**, briefly tell the user the path and suggest running `/git:commit-helper` if they want to commit it.

## Rules

- NEVER fabricate API references, install steps, or example output. If you're unsure, read the code or ask the user.
- NEVER overwrite an existing README without first showing the diff/draft and getting confirmation.
- If the target folder is empty or trivial, tell the user — don't pad a README to look substantial.
- Do not create a README in a folder that already has a perfectly good one unless the user explicitly asks for a rewrite.
