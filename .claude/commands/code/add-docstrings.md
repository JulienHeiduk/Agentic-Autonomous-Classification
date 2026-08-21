---
description: Add or improve docstrings for a single script or all scripts in a folder
---

You are helping the user add high-quality docstrings to Python scripts (or the project's primary language, if different).

## Scope resolution

`$ARGUMENTS` tells you what to document. It may be:

- **A single file** (e.g. `src/ml/train.py`) → document only that file
- **A folder** (e.g. `src/ml/`) → document every script in that folder (non-recursive by default; ask the user if they want recursion when the folder has subfolders)
- **Empty** → ask the user which file or folder to target. Do NOT default to the whole repo.

State the resolved target in one line before starting (e.g. "Target: 3 files in `src/ml/`").

## Steps

1. **List the target files** with Glob (e.g. `src/ml/*.py`). Skip:
   - `__init__.py` files unless they contain real logic
   - Test files (`test_*.py`, `*_test.py`) unless the user explicitly asked
   - Auto-generated files, migrations, vendored code

2. **Read each file fully before editing.** You need to understand what the code actually does — never write a docstring from a function name alone.

3. **Identify what needs docstrings**, in this priority order:
   - Module-level docstring (top of the file)
   - Public classes
   - Public functions and methods
   - Private helpers (`_name`) only if their purpose is non-obvious

   Skip anything that already has a clear, accurate docstring. If an existing docstring is **wrong or outdated**, fix it — mention the change to the user.

4. **Match the project's existing docstring style.** Before writing, check 1–2 already-documented files in the repo to detect the convention:
   - **Google style** (`Args:`, `Returns:`, `Raises:`)
   - **NumPy style** (`Parameters\n----------`)
   - **reST / Sphinx** (`:param x:`, `:returns:`)
   - If no convention is detectable, default to **Google style** for Python.

5. **Write each docstring to be useful, not decorative:**
   - First line: a single imperative sentence summarizing what it does ("Train the model on the given dataset.")
   - Then a blank line and a longer description **only if the behavior isn't obvious**
   - Document parameters, return values, and raised exceptions when they exist
   - Mention important side effects (file I/O, network calls, mutation of inputs)
   - Include a small `Example:` block only when usage is non-trivial
   - Do NOT restate the type annotations as prose ("x (int): an int") — that's noise
   - Do NOT invent behavior. If you're unsure what something does, read more of the code or ask the user

6. **Edit the files** with the Edit tool, one docstring at a time. Do not reformat or refactor surrounding code — docstrings only.

7. **After each file**, give the user a one-line summary: `src/ml/train.py — added 4 docstrings, fixed 1 outdated`.

8. **At the end**, suggest running tests (if any exist) and `/git:commit-helper` to commit the changes.

## Rules

- NEVER change code logic, signatures, imports, or formatting. Docstrings only.
- NEVER add a docstring that just repeats the function name in English ("get_user: gets the user").
- NEVER fabricate parameter descriptions, return values, or exceptions you didn't verify in the code.
- If a function is genuinely too unclear to document accurately, flag it to the user instead of guessing.
- For very large files, work through them in passes and report progress so the user can stop you early if needed.
