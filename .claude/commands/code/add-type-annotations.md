---
description: Add or improve type annotations for a single script or all scripts in a folder
---

You are helping the user add accurate Python type annotations (PEP 484 / PEP 604) to scripts.

## Scope resolution

`$ARGUMENTS` tells you what to type. It may be:

- **A single file** (e.g. `src/ml/train.py`) → annotate only that file
- **A folder** (e.g. `src/ml/`) → annotate every script in that folder (non-recursive by default; ask the user if they want recursion when the folder has subfolders)
- **Empty** → ask the user which file or folder to target. Do NOT default to the whole repo.

State the resolved target in one line before starting (e.g. "Target: 3 files in `src/ml/`").

## Steps

1. **List the target files** with Glob (e.g. `src/ml/*.py`). Skip:
   - `__init__.py` files unless they contain real logic
   - Test files (`test_*.py`, `*_test.py`) unless the user explicitly asked
   - Auto-generated files, migrations, vendored code

2. **Detect the project's Python version and typing conventions** before editing:
   - Check `pyproject.toml` / `setup.py` / `requirements.txt` for `python_requires` or `requires-python`
   - **Python ≥ 3.10** → use modern syntax: `list[int]`, `dict[str, int]`, `X | None`, `X | Y`
   - **Python ≥ 3.9** → use `list[int]`, `dict[str, int]`, but `Optional[X]` / `Union[X, Y]` from `typing`
   - **Python < 3.9** → use `List[int]`, `Dict[str, int]`, `Optional`, `Union` from `typing`
   - If unsure, ask the user or default to modern syntax (3.10+)
   - Also check if the project uses `from __future__ import annotations` — match that convention if present

3. **Read each file fully before editing.** You need to understand actual return values, parameter usages, and any duck-typing patterns. Never annotate from a function name alone.

4. **Identify what needs annotations**, in priority order:
   - Public function and method signatures (parameters + return type)
   - Public class attributes (when their type isn't obvious from `__init__`)
   - Module-level constants when their type adds clarity
   - Private helpers (`_name`) only if their types are non-obvious
   - Local variables only when type inference would genuinely fail (rare — usually skip)

   Skip anything already correctly annotated. If an existing annotation is **wrong**, fix it and mention the change.

5. **Write annotations that are accurate, not just present:**
   - Use the **most specific type that's actually true** — don't slap `Any` on everything
   - For containers, prefer concrete element types (`list[str]`, not `list`)
   - Use `| None` (or `Optional[X]`) when `None` is a real possible value — check the code, don't guess
   - Use `Iterable`, `Sequence`, `Mapping` from `collections.abc` for parameters that only need to be iterated/indexed (more flexible than `list`/`dict`)
   - Use `TypeAlias` or simple aliases for repeated complex types
   - For pandas/numpy/torch projects, use the libraries' own types (`pd.DataFrame`, `np.ndarray`, `torch.Tensor`) — don't reach for `Any`
   - For callbacks, use `Callable[[ArgTypes], ReturnType]`
   - Use `TYPE_CHECKING` guarded imports for types that are only needed at type-check time and would create import cycles
   - Do NOT add `-> None` to `__init__` only to add noise — actually, **do** add it; it's idiomatic and tools expect it

6. **Add necessary imports** at the top of the file (`from typing import ...`, `from collections.abc import ...`). Group them with existing imports — do not reformat the whole import block.

7. **Edit the files** with the Edit tool, one signature at a time. Do not reformat or refactor surrounding code — annotations and the imports they need, only.

8. **After each file**, give the user a one-line summary: `src/ml/train.py — annotated 4 functions, fixed 1 wrong type`.

9. **If the project has a type checker configured** (`mypy`, `pyright`, `ty`, `pyrefly` in `pyproject.toml` or config file), suggest running it after the changes so the user can verify. Do not run it yourself unless asked.

10. **At the end**, suggest `/git:commit-helper` to commit the changes.

## Rules

- NEVER change code logic, control flow, or behavior. Annotations and required imports only.
- NEVER use `Any` as a lazy escape hatch. If you genuinely cannot determine a type, flag it to the user instead.
- NEVER add annotations you didn't verify against the actual code. Read the function body, check what it returns, check what callers pass in.
- NEVER reformat existing code or imports beyond what's needed to add the annotations.
- If a function's behavior is too dynamic to type accurately (e.g. heavy `**kwargs` plumbing), say so and suggest `TypedDict` or `Protocol` rather than forcing a wrong type.
- For very large files, work in passes and report progress so the user can stop you early if needed.
