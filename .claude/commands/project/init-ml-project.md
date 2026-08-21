---
description: Scaffold a kamino_nb_studio ML project under src/kamino_nb_studio/<project-name>/ per the README architecture
---

You are helping the user scaffold a new ML project inside this repo, following the "ML Repository Architecture" section of the top-level `README.md`.

## Resolve the project name

`$ARGUMENTS` is the project name.

- If `$ARGUMENTS` is empty, ask the user: "What is the project name? (snake_case — it becomes a Python package)".
- Validate the name matches `^[a-z][a-z0-9_]*$`. If not, explain why (must be importable as a Python package) and ask again.
- Resolved target: `src/kamino_nb_studio/<project-name>/`.
- If the target folder already exists, STOP and tell the user. Do NOT overwrite.

State the resolved target in one line before creating anything (e.g. "Target: `src/kamino_nb_studio/churn_model/`").

## Source of truth

The folder tree to create is the one in `README.md` under **Canonical project layout**. Re-read that block before scaffolding so the structure stays in sync if the README evolves. Do not invent extra folders or files that aren't in the README.

As of now the layout to scaffold is (mirrors `ksp_audience_targeting` / `traffic_forecasting`):

```
src/kamino_nb_studio/<project-name>/
├── configs/
│   └── config.yaml                     # flat shared config (identifiers, dates)
├── data/.gitkeep                       # local CSVs / models / outputs (gitignored)
├── models/.gitkeep                     # saved model artefacts
├── reports/.gitkeep                    # generated reports / figures
├── scripts/.gitkeep                    # one-off ops shell scripts
├── tests/
│   ├── __init__.py
│   └── test_placeholder.py
├── src/
│   ├── __init__.py
│   ├── transform_pipeline.py           # feature-engineering entrypoint
│   ├── training_pipeline.py            # training entrypoint (fit + save)
│   ├── inference_pipeline.py           # inference entrypoint (load + score)
│   ├── app/.gitkeep                    # optional Streamlit / dashboards
│   ├── steps/                          # per-pipeline orchestration (thin wrappers)
│   │   ├── __init__.py
│   │   ├── transform/__init__.py       # load → preprocess → features → assemble
│   │   ├── training/__init__.py        # split → fit → evaluate → save
│   │   └── inference/__init__.py       # cohort → score → audience
│   ├── functions/                      # reusable stateless primitives
│   │   ├── __init__.py
│   │   ├── transform/__init__.py       # (project subfolders allowed later)
│   │   ├── training/__init__.py
│   │   └── inference/__init__.py
│   ├── query/.gitkeep                  # .sql templates
│   ├── utils/__init__.py               # IO, logging, drift checks
│   └── experiments/.gitkeep            # one-off analyses
├── k8s/.gitkeep                        # deploy.sh + manifests + argo/<env>/ (added at deploy time)
├── Dockerfile
├── README.md
└── pyproject.toml                      # optional/inert — deps live in the ROOT pyproject.toml
```

## Steps

1. **Show the planned tree** to the user and ask for confirmation before writing anything. Include the resolved target path at the top.

2. **Create all files** with the Write tool. Prefer writing many files in parallel in a single message. Use minimal but sensible stubs — not empty files where a stub adds value:

   - `__init__.py` files — empty. Create one in each Python package so the tree is importable as `kamino_nb_studio.<project>.src…`: `src/`, `src/steps/`, `src/steps/{transform,training,inference}/`, `src/functions/`, `src/functions/{transform,training,inference}/`, `src/utils/`, `tests/`.
   - The three pipeline entrypoints — `transform_pipeline.py`, `training_pipeline.py`, `inference_pipeline.py` (note the **singular** `transform_pipeline`) — each gets a one-line module docstring describing its role, then `# TODO: implement`. No invented logic. Leave `steps/` and `functions/` (and their `transform/training/inference` subpackages) as empty packages; real modules land there as the project grows.
   - `tests/test_placeholder.py` — a module docstring and a `# TODO: add tests` placeholder. No pytest boilerplate unless pytest is already in the repo deps.
   - `configs/config.yaml` — a minimal **flat** YAML (the repo does NOT use Hydra): a header comment plus a couple of placeholder identifier keys as commented `# TODO` (e.g. `clickhouse_env`, project/campaign identifiers, date window). Env-split (`config.<env>.yaml`) and `config_eda_*.yaml` are added later if the project needs them.
   - `Dockerfile` — single-line comment `# TODO: project-specific image (build context = repo root; installs via uv sync --group <project>)`. Do not invent a base image.
   - `pyproject.toml` — a minimal, **inert** PEP 621 manifest: `[project]` with `name = "<project-name>"`, `version = "0.1.0"`, `description`, `requires-python` matching the repo root, and empty `dependencies = []`. Add a comment noting the real dependencies live in the **root** `pyproject.toml` under `[dependency-groups].<project-name>`. No `[build-system]` unless asked, no pinned versions.
   - `README.md` — a short project-specific README: title = project name, one line explaining the project lives under `src/kamino_nb_studio/<project-name>/`, and a "Structure" section linking back to the repo-root README for the full architecture.
   - `.gitkeep` files — empty, to keep otherwise-empty folders in git: `data/`, `models/`, `reports/`, `scripts/`, `src/app/`, `src/query/`, `src/experiments/`, `k8s/`. (Real `k8s/` manifests and `query/*.sql` are added later — never fabricate them.)

3. **After writing**, give the user:
   - A summary list of what was created.
   - **Follow-ups the skill cannot do itself** (they touch the repo-root `pyproject.toml`, outside the project folder): register the three console scripts in the root `[project.scripts]` (`<abbr>-features` / `-train` / `-infer` → the pipelines' `:main`, mirroring `ksp-*` / `tf-*`), and add a `[dependency-groups].<project-name>` entry for the project's ML stack. `k8s/` manifests are added at deploy time by copying an existing project's — not scaffolded.
   - A suggestion to run `/git:commit-helper` to commit the scaffold.

## Rules

- NEVER overwrite an existing file. If the target folder already exists, stop and ask.
- NEVER create files outside `src/kamino_nb_studio/<project-name>/`.
- NEVER fabricate ML logic, model hyperparameters, or query SQL. Stubs with `TODO`s only.
- NEVER run package managers, git commands, or anything state-changing outside file creation. Files only.
- Match the repo's Python version (read from the repo-root `pyproject.toml`).
- Keep the layout in sync with the README. If the user asks for an extra folder/file, add it here *and* to the README so they don't drift.
