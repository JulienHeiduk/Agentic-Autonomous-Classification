---
description: Introduce Pydantic models for a single script or all scripts in a folder
---

You are helping the user introduce Pydantic models into Python scripts — replacing loose dicts, dataclasses, or untyped configs with validated `BaseModel` classes where it actually adds value.

## Scope resolution

`$ARGUMENTS` tells you what to convert. It may be:

- **A single file** (e.g. `src/ml/config.py`) → only that file
- **A folder** (e.g. `src/ml/`) → every script in that folder (non-recursive by default; ask if the folder has subfolders)
- **Empty** → ask the user which file or folder to target. Do NOT default to the whole repo.

State the resolved target in one line before starting (e.g. "Target: 3 files in `src/ml/`").

## Steps

1. **Detect the Pydantic version available in the project:**
   - Read `pyproject.toml` / `requirements.txt` / `setup.py` / `uv.lock` / `poetry.lock`
   - **Pydantic v2** → use `from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict`. Use `model_config = ConfigDict(...)`, `model_dump()`, `model_validate()`.
   - **Pydantic v1** → use `from pydantic import BaseModel, Field, validator, root_validator`. Use `class Config:`, `dict()`, `parse_obj()`.
   - **Settings management** → if the project uses settings classes, prefer `pydantic-settings` (v2) or `BaseSettings` (v1). Check whether `pydantic-settings` is installed before using it.
   - **If Pydantic is not installed**, STOP and tell the user. Ask whether they want you to add it to their dependency file (don't run `pip install` yourself).

2. **Read each target file fully** before editing. You need to understand:
   - What data shapes flow through the file (dicts, dataclasses, TypedDicts, raw kwargs)
   - Where validation actually matters (system boundaries: config loading, API request/response, file parsing, CLI args)
   - What's already typed and working — those don't need to become Pydantic models

3. **Identify good candidates for Pydantic models** — be selective, not aggressive:
   - Config / settings loaded from env, YAML, JSON, TOML
   - Request/response payloads (FastAPI, HTTP clients)
   - Parsed data from external sources (CSV rows, API responses, file formats)
   - Existing dataclasses that are validated manually elsewhere
   - Functions that take `**kwargs` or `dict[str, Any]` for structured input

   **Skip** (do NOT pydantic-ify):
   - Internal-only data structures that never cross a boundary
   - Hot loops where validation overhead would matter
   - Things that are already `@dataclass(frozen=True)` and well-typed and never validated — converting adds no value
   - DataFrame rows, numpy arrays, tensors (wrong tool)

   If a file has no good candidates, say so and move on. Don't force it.

4. **Design each model carefully:**
   - Use precise field types — not `Any`
   - Use `Field(...)` for defaults, descriptions, constraints (`ge=`, `le=`, `min_length=`, `pattern=`)
   - Use `Optional[X]` / `X | None` only when `None` is genuinely valid
   - Add validators (`@field_validator` v2 / `@validator` v1) only when there's real validation logic to enforce — don't add empty ones
   - Use nested models for nested structures, not `dict[str, Any]`
   - Set `model_config = ConfigDict(extra="forbid")` (v2) or `class Config: extra = "forbid"` (v1) for strict configs unless the user wants the opposite
   - Use enums (`StrEnum` / `Enum`) for fixed string sets

5. **Update the call sites** that produce or consume the converted data:
   - Replace `dict(...)` construction with `Model(...)`
   - Replace dict access (`d["key"]`) with attribute access (`m.key`)
   - Replace manual validation with the model's built-in validation
   - For serialization, use `model_dump()` (v2) or `dict()` (v1)
   - Keep changes minimal — only touch what's needed for the conversion to work

6. **Add necessary imports** at the top of the file. Group with existing imports — do not reformat the whole import block.

7. **Edit one file at a time** with the Edit tool. After each file, give the user a one-line summary: `src/ml/config.py — added 2 models, replaced 1 dataclass`.

8. **Show the user the proposed changes for the first file before mass-applying** — Pydantic conversions touch behavior, so confirm the approach is right before continuing through a folder.

9. **If the project has tests**, suggest running them after the changes. Do not run them yourself unless asked.

10. **At the end**, suggest `/git:commit-helper` to commit the changes.

## Rules

- NEVER convert code to Pydantic just to add a model — there must be a real benefit (validation, parsing, settings, boundary).
- NEVER change behavior beyond what the conversion requires. No refactoring of unrelated code.
- NEVER use `Any` as a field type unless the data is genuinely unstructured (and even then, prefer `dict` / `JsonValue`).
- NEVER mix Pydantic v1 and v2 idioms in the same project. Pick the version actually installed and stick to it.
- NEVER assume Pydantic is installed — verify in the dependency file first.
- If a conversion would require deeper refactoring than docstring/annotation-style edits (e.g. large API surface changes), STOP and present a plan to the user instead of editing.
- If in doubt about whether something should become a model, ask the user rather than guessing.
