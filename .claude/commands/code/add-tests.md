---
description: Generate tests in tests/ for a specific script or every script in a folder
---

You are helping the user add **real, useful tests** for Python code. Your job is to test what the code actually does, not to inflate a coverage number with assertions that always pass.

Tests are written into the project's `tests/` folder (the one scaffolded by `/project:init-zenml-project`, or whichever convention the project already uses). You do not modify the source code being tested.

## Scope resolution

`$ARGUMENTS` tells you what to test:

- **A single file** (e.g. `src/ml/train.py` or `steps/data_loader.py`) → write tests for that file
- **A folder** (e.g. `src/ml/`, `steps/`) → write tests for every script in that folder (non-recursive by default; ask about subfolders)
- **Empty** → ask the user which file or folder to target. Do NOT default to the whole repo.

State the resolved target in one line before starting (e.g. "Adding tests for 4 files in `steps/`").

## Steps

1. **Detect the test framework and folder layout** before writing anything:
   - Look for `tests/` first (this is what `/project:init-zenml-project` creates). Fall back to `test/`, `src/tests/`, or whatever the project already uses.
   - Check `pyproject.toml` / `setup.cfg` / `tox.ini` for `pytest`, `unittest`, or other configured runners. **Default to pytest** if nothing is specified — it's the most common Python convention.
   - If no test framework is installed, STOP and tell the user. Do NOT `pip install`.
   - Check existing tests in the folder to learn the project's conventions: fixture names, parametrize style, helper modules, naming patterns. Match what's there.

2. **Plan the destination path** for each test file by **mirroring the source layout** under `tests/`:

   | Source | Test file |
   |---|---|
   | `steps/data_loader.py` | `tests/steps/test_data_loader.py` |
   | `pipelines/training_pipeline.py` | `tests/pipelines/test_training_pipeline.py` |
   | `src/ml/utils.py` | `tests/ml/test_utils.py` |

   Create intermediate `__init__.py` files only if the rest of `tests/` uses them. Most modern pytest projects don't need them.

   **If a test file already exists** at the planned path:
   - Read it
   - **Add new test cases** to it instead of overwriting
   - Never delete existing tests
   - Mention to the user which existing tests you found and which new ones you're adding

3. **Read each source file fully before writing tests.** You need to understand:
   - What the function actually does (return values, side effects, exceptions)
   - What inputs are valid and invalid
   - Whether it has side effects (file I/O, network, mutation, randomness)
   - Whether it depends on external state (env vars, files, database, GCS, model files)

   Never write a test from a function name alone.

4. **Identify what's worth testing**, in priority order:
   - **Pure functions** with deterministic inputs/outputs → easiest, highest value
   - **Functions with branching logic** → cover each branch, especially edge cases
   - **Error paths** → verify the right exception is raised on bad input
   - **Boundary conditions** → empty input, single element, max size, off-by-one
   - **Public API surfaces** → if it's exported, it should have at least one test
   - **Bug-prone code** → anything with manual validation, regex, off-by-one, type coercion

   **Skip** (don't waste tests on):
   - Trivial getters/setters with no logic
   - Code that just delegates to a well-tested library
   - `__repr__` / `__str__` unless they're load-bearing
   - Auto-generated code, migrations
   - Logging statements, debug helpers

5. **Write each test to be useful, not decorative:**

   - **One behavior per test.** Don't test five things in one function.
   - **Test names describe the behavior, not the function name.** Good: `test_load_data_raises_when_path_missing`. Bad: `test_load_data_1`.
   - **Use AAA structure** (Arrange / Act / Assert) — keep them separated by blank lines for readability.
   - **Assert on real values**, not just `is not None`. `assert result == expected_value` beats `assert result`.
   - **Use `pytest.parametrize`** when testing the same behavior across multiple inputs — avoids copy-paste.
   - **Use `pytest.raises`** for exception tests, and check the exception message when it matters.
   - **Use fixtures** for shared setup, defined in `conftest.py` if reused across files.
   - **Use `tmp_path`** for any test that touches the filesystem — never write to the real cwd.
   - **For randomness**, seed with a fixed value or use `pytest.MonkeyPatch` to make it deterministic.
   - **For time-dependent code**, freeze time with `freezegun` or by patching `datetime.now`.

6. **For ZenML-specific code** (pipelines and steps), test what's actually testable:

   - **`@step`-decorated functions** are still callable as plain functions (with `.entrypoint(...)` in newer ZenML, or directly as the wrapped function). Test the underlying logic by calling the entrypoint directly with concrete inputs — don't spin up a real ZenML run for unit tests.
   - **`@pipeline`-decorated functions** are wiring code; integration-test them by running the pipeline against a tiny in-memory dataset, OR unit-test the wiring by inspecting that the right steps are connected. Don't do both for the same function.
   - **Materializers** are testable in isolation: serialize then deserialize and assert round-trip equality.
   - **Don't mock ZenML internals** — they change between versions and break tests on every upgrade. Mock the *external* dependencies (GCS, BigQuery, model registry) instead.

7. **Handle external dependencies carefully:**

   - **Filesystem** → use `tmp_path` fixture
   - **HTTP** → use `responses` or `httpx_mock` if available; otherwise patch the client
   - **Cloud storage (GCS, S3)** → use `moto`/`gcsfs` mocks if installed; otherwise patch the SDK calls
   - **Database** → prefer a real test database (SQLite in-memory for SQL); only mock when the cost is unreasonable
   - **ML models** → load a tiny fixture model from `tests/fixtures/` rather than the production checkpoint
   - **Randomness** → seed it
   - **Environment variables** → use `monkeypatch.setenv`

   Never mock the function under test itself.

8. **Add necessary imports** at the top of each test file. Do not import the source modules using fragile relative paths — use the project's package layout.

9. **Show the user the planned test files (paths + a one-line summary of what each will cover) before writing them.** Ask for confirmation. The user may want to drop some, add others, or change scope.

10. **Write the test files** with the Write tool, one at a time. After each file, give the user a one-line summary: `tests/steps/test_data_loader.py — added 6 tests covering happy path, missing-file, empty-file, schema validation`.

11. **After all files are written**, suggest:
    - Running `pytest tests/<scope>` (or the project's standard command) to verify they pass. Do not run it yourself unless asked.
    - Using `/git:commit-helper` to commit the new tests.

## Rules

- NEVER fabricate expected values. Every assertion must be based on what the source code actually does — read it, trace it, verify.
- NEVER write tests that always pass (`assert True`, `assert result is not None` for a function whose return type guarantees non-None, etc.).
- NEVER modify the source code being tested. Tests only.
- NEVER overwrite an existing test file. Append new tests to it, or pick a different name.
- NEVER mock the function under test. Mock its external dependencies, not its internals.
- NEVER use sleeps, network calls, or production credentials in tests.
- NEVER commit secrets or real credentials in fixture files.
- NEVER skip a test with `@pytest.mark.skip` to "make it pass" — if a test can't be written meaningfully, tell the user and don't write it.
- All test names, docstrings, and comments must be written in **English**.
- If a function is genuinely too dynamic, side-effect-heavy, or coupled to external state to test meaningfully, say so to the user instead of writing a fake test.
- For very large folders, work file-by-file and report progress so the user can stop you early.
