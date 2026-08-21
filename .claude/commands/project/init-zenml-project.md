---
description: Scaffold a ZenML ML-project folder structure in the repo root or a target folder
---

You are helping the user initialize a folder architecture for a ZenML-based ML project.

## Scope resolution

`$ARGUMENTS` tells you **where** to scaffold:

- **A folder path** (e.g. `projects/churn-model`) → scaffold the structure **inside that folder**. Create the folder if it doesn't exist.
- **Empty** → scaffold at the repo root (current working directory).

State the resolved target in one line before starting (e.g. "Target: `projects/churn-model/`").

## Steps

1. **Check the target.** Use Bash `ls` (or Glob) to see what's already there:
   - If the folder doesn't exist, plan to create it
   - If it exists and already contains files, list which ZenML folders/files are already present and which would be added — **never silently overwrite existing files**

2. **Detect ZenML context** in the surrounding project:
   - Check `pyproject.toml` / `uv.lock` for `zenml`
   - If ZenML is not declared, mention it to the user and add `zenml` to the proposed `pyproject.toml` of the new project — but do NOT run `uv sync`, `uv add`, or `pip install` yourself
   - Check the Python version (same logic as `/code:add-type-annotations`) so the scaffolded files use compatible syntax

3. **Show the planned structure to the user before creating anything** and ask for confirmation. Present it as a tree:

   ```
   <target>/
   ├── pipelines/
   │   ├── __init__.py
   │   ├── training_pipeline.py
   │   └── inference_pipeline.py
   ├── steps/
   │   ├── __init__.py
   │   ├── data_loader.py
   │   ├── data_preprocessor.py
   │   ├── model_trainer.py
   │   └── model_evaluator.py
   ├── materializers/
   │   └── __init__.py
   ├── configs/
   │   ├── training.yaml
   │   └── inference.yaml
   ├── utils/
   │   └── __init__.py
   ├── data/
   │   ├── raw/.gitkeep
   │   └── processed/.gitkeep
   ├── models/.gitkeep
   ├── notebooks/.gitkeep
   ├── tests/
   │   └── __init__.py
   ├── run.py
   ├── pyproject.toml
   ├── .python-version
   ├── README.md
   ├── CHANGELOG.md
   └── .gitignore
   ```

   Ask the user if they want to add or skip anything (e.g. some users won't want `notebooks/`, others want a `deployment/` folder).

4. **Create the files** with the Write tool, one at a time. Use minimal but functional stubs — runnable boilerplate, not empty files:

   - **`pipelines/training_pipeline.py`** — a `@pipeline`-decorated function wiring the steps in order (load → preprocess → train → evaluate)
   - **`pipelines/inference_pipeline.py`** — a `@pipeline`-decorated function for inference (load model → predict)
   - **`steps/*.py`** — each step is a `@step`-decorated function with proper type annotations on params and return types. Bodies should be `# TODO:` placeholders, not invented logic.
   - **`configs/training.yaml`** — a minimal ZenML config with `settings`, `steps`, and `parameters` sections (commented placeholders)
   - **`run.py`** — entry point that imports and runs the training pipeline, with a `if __name__ == "__main__":` guard
   - **`pyproject.toml`** — minimal PEP 621 manifest managed by **uv**. Include:
     - `[project]` table with `name`, `version = "0.1.0"`, `description`, `requires-python = ">=3.10"` (or whichever version the surrounding repo targets), and a `dependencies` array containing `"zenml"` plus any other obvious deps the user mentions
     - `[build-system]` table only if the user wants the project to be installable as a package — otherwise omit it (uv handles dependency-only projects fine without one)
     - A `[tool.uv]` table only if there's a real reason to set options; otherwise omit
     - Do NOT pin exact versions — let `uv lock` resolve them
   - **`.python-version`** — a single line with the chosen Python version (e.g. `3.12`). uv reads this to select the interpreter.
   - **`.gitignore`** — Python defaults plus ZenML-specific entries (`.zen/`, `mlruns/`, `*.pkl`, `__pycache__/`) and uv-specific entries (`.venv/`)
   - **`README.md`** — a short README explaining the structure, how to install, how to run the pipeline. Match the style of the repo's existing README if there is one.
   - **`CHANGELOG.md`** — follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format with a `## [Unreleased]` section and an initial `## [0.1.0] - YYYY-MM-DD` entry containing an `### Added` bullet for "Initial ZenML project scaffold". Use today's date (resolve relative dates to absolute `YYYY-MM-DD`). Include a top-of-file note that the project follows [Semantic Versioning](https://semver.org/).
   - **`__init__.py`** files — empty (or with module docstrings only)
   - **`.gitkeep`** files — empty, just to make empty folders trackable in git

5. **Use modern, idiomatic ZenML syntax** for the stubs:
   - Functions decorated with `@step` and `@pipeline` (not the deprecated class-based API)
   - Type annotations on every step signature
   - Return types declared (use `Annotated[X, "name"]` for named outputs when there are multiple)
   - Imports: `from zenml import pipeline, step`

6. **Do NOT invent ML logic.** Steps should contain clear `# TODO:` comments and `pass` / `raise NotImplementedError` — the user will fill in the actual model code. Your job is the scaffold, not the model.

7. **After creating all files**, give the user:
   - A summary list of what was created
   - The uv-based bootstrap commands to run from inside the target folder:
     - `uv sync` — create the virtualenv and install dependencies from `pyproject.toml`
     - `uv run python run.py` — run the training pipeline inside the managed venv
     - `uv add <package>` — how to add new dependencies later
   - A suggestion to run `/git:commit-helper` to commit the scaffold

## Rules

- NEVER overwrite an existing file. If a file at the target path already exists, skip it and tell the user.
- NEVER run `pip install`, `zenml init`, `git init`, or any state-changing command. Files only.
- NEVER create files outside the resolved target folder.
- NEVER fabricate ML logic, model architectures, or dataset loading code. Stubs with `TODO`s only.
- If the target folder already looks like a ZenML project (has `pipelines/` and `steps/`), STOP and ask the user whether they want to merge into it or pick a different target.
- Match the existing repo's Python version, dependency manager (pip / poetry / uv), and code style if detectable.
