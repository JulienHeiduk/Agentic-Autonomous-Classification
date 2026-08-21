---
description: Run the full code-quality pipeline (annotations → annotated-types → pydantic → docstrings → review) on a target, with regression checks between steps
---

You are running an end-to-end code-quality workflow on a target file or folder. The pipeline chains five commands in a specific order, and **the critical rule is that no step is allowed to remove or weaken work done by a previous step**. You are responsible for verifying this between each step.

## Scope resolution

`$ARGUMENTS` is the target:

- **A single file** (e.g. `src/ml/train.py`) → run the pipeline on that file
- **A folder** (e.g. `src/ml/`) → run on every Python file in that folder (non-recursive by default; ask about subfolders)
- **Empty** → ask the user which file or folder to target. Do NOT default to the whole repo.

State the resolved target in one line before starting (e.g. "Pipeline target: 3 files in `src/ml/`").

## Pipeline order — and why

1. **`/code:add-type-annotations`** — get baseline type annotations on every signature. Pure addition; touches nothing else.
2. **`/code:annotated-types`** — promote semantic types into reusable `Annotated[..., Field(...)]` aliases in `types.py`. **Refactors** signatures to use the aliases instead of raw types — must preserve every type from step 1 (alias is an upgrade, not a removal).
3. **`/code:add-pydantic`** — convert real boundary data (configs, payloads) into `BaseModel` classes, using the type aliases from step 2 as field types. Must not touch existing annotations or aliases.
4. **`/code:add-docstrings`** — last, because previous steps may have changed signatures, renamed parameters, or introduced new classes. Running docstrings last means writing them once against the final shape.
5. **`/code:code-review`** — read-only final pass. Cannot regress anything.

This order is deliberate. Do not reorder unless the user asks.

## Preflight (before step 1)

1. **Check git working tree is clean.** If there are uncommitted changes:
   - Show them to the user
   - Ask whether to commit/stash them first or include them in the pipeline
   - Never blow them away

2. **Capture a baseline snapshot** of the target with the script below (or an equivalent inline approach). Save the result — you'll re-run it after each step and diff the counts.

   ```python
   # Run via Bash: python -c "..."
   import ast, pathlib, sys
   target = pathlib.Path("<target>")
   files = [target] if target.is_file() else sorted(target.glob("*.py"))
   totals = {"funcs": 0, "typed_params": 0, "typed_returns": 0,
             "docstrings": 0, "field_calls": 0, "annotated_aliases": 0,
             "basemodels": 0}
   for f in files:
       try:
           tree = ast.parse(f.read_text(encoding="utf-8"))
       except Exception:
           continue
       for node in ast.walk(tree):
           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
               totals["funcs"] += 1
               if node.returns is not None:
                   totals["typed_returns"] += 1
               for arg in node.args.args + node.args.kwonlyargs:
                   if arg.annotation is not None and arg.arg != "self":
                       totals["typed_params"] += 1
               if ast.get_docstring(node):
                   totals["docstrings"] += 1
           if isinstance(node, ast.ClassDef):
               if ast.get_docstring(node):
                   totals["docstrings"] += 1
               if any(isinstance(b, ast.Name) and b.id == "BaseModel"
                      or isinstance(b, ast.Attribute) and b.attr == "BaseModel"
                      for b in node.bases):
                   totals["basemodels"] += 1
           if isinstance(node, ast.Call):
               func = node.func
               if (isinstance(func, ast.Name) and func.id == "Field") or \
                  (isinstance(func, ast.Attribute) and func.attr == "Field"):
                   totals["field_calls"] += 1
           if isinstance(node, ast.Subscript):
               val = node.value
               if (isinstance(val, ast.Name) and val.id == "Annotated") or \
                  (isinstance(val, ast.Attribute) and val.attr == "Annotated"):
                   totals["annotated_aliases"] += 1
       if ast.get_docstring(tree):
           totals["docstrings"] += 1
   print(totals)
   ```

   Print the baseline as `BASELINE: {...}` and remember it.

3. **Confirm with the user before starting.** Show them the pipeline order, the target, and the baseline. Ask for go-ahead. They may want to skip a step (e.g. "skip pydantic") — honor that.

## Per-step procedure

For each step in order, do exactly this:

1. **Announce the step.** `### Step N/5: <command name>`

2. **Pre-step snapshot.** Re-run the AST counter against the target. Save as `before_step_N`.

3. **Execute the step's instructions.** Read `.claude/commands/code/<command>.md` and follow its procedure exactly. Do not improvise — the source-of-truth is that file. Run any sub-command rules (skip rules, file selection rules, etc.) as written.

4. **Post-step snapshot.** Re-run the AST counter. Save as `after_step_N`.

5. **Preservation check.** Compare `after_step_N` against `before_step_N` AND against the original `BASELINE`. The following counts must **never decrease** across the pipeline:

   | Metric | Allowed change |
   |---|---|
   | `typed_params` | only increase (or stay equal) |
   | `typed_returns` | only increase (or stay equal) |
   | `docstrings` | only increase (or stay equal) |
   | `field_calls` | only increase (or stay equal) |
   | `annotated_aliases` | only increase (or stay equal) |
   | `basemodels` | only increase (or stay equal) |
   | `funcs` | should stay equal — a decrease means functions were deleted, which no quality step should ever do |

   **Exception for step 2 (annotated-types):** when the refactor replaces `def f(x: int)` with `def f(x: TopK)`, the `typed_params` count stays the same (still annotated). It must not drop. If it drops, the refactor removed an annotation without replacing it.

6. **Git diff inspection.** Run `git diff` against the changes the step just made and look specifically for **removed lines** that contained any of:
   - Triple-quoted strings (`"""` or `'''`) → would indicate a deleted docstring
   - `: ` followed by a type (parameter annotations) → unless the same line is being replaced with a typed alias
   - `-> ` (return annotations)
   - `Field(` → would indicate a deleted Field metadata
   - `Annotated[` → would indicate a deleted type alias usage

   For each removed line of these kinds, verify that the same metric still went up or stayed equal in the AST counts. If both checks agree, you're fine.

7. **On regression** (counts dropped, OR a deletion of a protected pattern with no compensating addition):
   - **STOP the pipeline immediately.**
   - Show the user: which metric dropped, by how much, and the relevant `git diff` hunks.
   - Offer to: (a) revert this step with `git checkout -- <files>`, (b) keep the changes and continue anyway with the user's explicit permission, or (c) abandon the pipeline.
   - Do NOT proceed without an answer.

8. **Checkpoint commit.** If the step passed the preservation check, create a checkpoint commit so each step is a separate, revertible point in history:

   ```bash
   git add <only the files touched by this step>
   git commit -m "wip(quality): step N — <command name>" --no-verify
   ```

   Use a `wip(quality)` prefix so these are easy to spot and squash later. Do **not** use `--no-verify` if the user has pre-commit hooks they want enforced — ask first if you're unsure. (In this repo the secrets hook is configured at the Claude level, so a Claude-side `git commit` will still go through it.)

   Tell the user: `Step N done — checkpoint <short-sha>. Counts: <before> → <after>`.

9. **Move to step N+1.**

## After all steps

1. **Final summary.** Show:
   - The full count delta: `BASELINE → final` for every metric
   - The list of checkpoint commits created
   - A one-line per-file summary of what changed

2. **Suggest squashing the wip checkpoints into a single clean commit** if the user wants a tidy history. Offer the `git reset --soft <baseline>` + re-commit pattern, but **do not run it without confirmation** — destructive operations need approval.

3. **Suggest running the project's tests and type checker** (mypy / pyright / etc.) to verify nothing broke at runtime. Do not run them unless asked.

4. **Suggest `/git:commit-helper`** for the final consolidated commit.

## Rules

- NEVER skip a preservation check between steps. The whole point of this workflow is that each step is provably non-destructive.
- NEVER run two steps in parallel — this pipeline is strictly sequential.
- NEVER modify a step's source-of-truth command file from inside this workflow. If a step's procedure has a bug, fix it in the command file in a separate session.
- NEVER reorder the pipeline. The order is chosen so each step can build on the previous without conflict.
- NEVER continue past a regression without explicit user permission.
- If a step has nothing to do (e.g. `add-pydantic` finds no boundary data worth converting), that's fine — log it as `Step N: no candidates, skipped` and move on.
- Honor `--skip-step <name>` style requests from the user (e.g. "run the pipeline but skip pydantic").
- If the target has zero Python files, stop immediately and tell the user.
