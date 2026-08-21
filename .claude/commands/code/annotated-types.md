---
description: Extract domain types into a types.py module using Annotated + pydantic Field
---

You are helping the user build a vocabulary of **reusable, constrained, self-documenting type aliases** using `Annotated[..., Field(...)]`. The output is a `types.py` module that other code imports from, and (optionally) refactored call sites that use the new aliases.

This is **not** the same as adding bare type annotations. The goal here is to capture domain semantics — patterns, ranges, descriptions, examples — once, and reuse them everywhere.

## Example of the target style

```python
# types.py
from typing import Annotated
from pydantic import Field

BUCode = Annotated[str, Field(
    pattern=r"^[A-Z]{2,4}$",
    description="Code Business Unit Leroy Merlin (ex: FR, ES, BRPL)",
    examples=["FR", "ES", "IT"],
)]

TopK = Annotated[int, Field(
    gt=0, le=5000,
    description="Nombre de candidats à retourner par requête",
    examples=[100, 500],
)]

Threshold = Annotated[float, Field(
    ge=0.0, le=1.0,
    description="Seuil de similarité cosine minimum pour filtrer les résultats",
    examples=[0.7, 0.85],
)]

GCSPath = Annotated[str, Field(
    pattern=r"^gs://",
    description="Chemin GCS complet (gs://bucket/path)",
    examples=["gs://my-bucket/embeddings/"],
)]
```

Every alias carries: a base type, real constraints (`pattern`, `gt`/`ge`/`le`/`lt`, `min_length`, `max_length`), a human description, and concrete examples.

## Scope resolution

`$ARGUMENTS` tells you what to analyze:

- **A single file** (e.g. `src/api/routes.py`) → harvest candidate types from that file only
- **A folder** (e.g. `src/api/`) → harvest from every Python file in that folder (non-recursive by default; ask about subfolders)
- **Empty** → ask the user which file or folder to scan. Do NOT default to the whole repo.

State the resolved target in one line before starting (e.g. "Scanning 4 files in `src/api/` for candidate types").

## Steps

1. **Verify Pydantic is available.** Check `pyproject.toml` / `requirements.txt` / lockfile for `pydantic`. If absent, STOP and tell the user. Do NOT `pip install`.
   - Detect v1 vs v2. The `Annotated[..., Field(...)]` style works on both, but the import paths and some Field kwargs differ slightly. Use the version actually installed.

2. **Locate or plan a `types.py`.** Look for an existing one in a sensible location (project root, `src/`, the target folder, or a sibling `common/` / `schemas/` folder). If multiple exist, ask the user which one to extend. If none exists, propose a path and ask before creating.

3. **Read the target files fully.** You're hunting for **semantic types** — values whose meaning is more specific than their base type. Look for:

   - **Manual validation patterns** repeated across functions:
     ```python
     if not re.match(r"^[A-Z]{2,4}$", bu_code): raise ValueError(...)
     assert 0 < top_k <= 5000
     assert 0.0 <= threshold <= 1.0
     ```
   - **Pydantic Field definitions** that recur in multiple `BaseModel` classes (same constraints, same description) — these are the strongest candidates.
   - **Function parameters** named with strong domain meaning (`bu_code`, `gcs_path`, `top_k`, `threshold`, `customer_id`, `embedding_dim`, `iso_country_code`, `email`, `url`).
   - **String/int parameters with format constraints** documented in docstrings or comments.
   - **Magic literals** that hint at constraints (`gs://`, `https://`, `^[A-Z]+$`, hardcoded min/max).

   For each candidate, gather: base type, every constraint you can find, a clear description (translate from comments/docstrings if needed), and 2–3 realistic example values pulled from the actual code or tests.

4. **Skip these — they're not good candidates:**
   - Internal-only ints/strings with no constraints (`count`, `index`, `name`)
   - Things that already have a `BaseModel` representing them (use the model)
   - Types that appear exactly once and are unlikely to be reused (extracting them adds indirection without payoff)
   - Highly dynamic types (`dict[str, Any]`, untyped JSON blobs)

5. **Show the user a proposed table of new types before writing anything**, in this format:

   ```
   Proposed annotated types (4):

   1. BUCode      = Annotated[str, ...]   pattern=^[A-Z]{2,4}$    used in 6 places
   2. TopK        = Annotated[int, ...]   0 < x <= 5000           used in 3 places
   3. Threshold   = Annotated[float, ...] 0.0 <= x <= 1.0         used in 4 places
   4. GCSPath     = Annotated[str, ...]   pattern=^gs://          used in 2 places

   Target file: src/common/types.py (will be created)
   ```

   Ask the user to confirm, drop, or rename any entries before proceeding. Naming matters — use **PascalCase domain names**, not type-shaped names (`BUCode` not `BUCodeStr`).

6. **Write `types.py`** with the confirmed aliases. Rules for the file:

   - Single import block at the top: `from typing import Annotated` and `from pydantic import Field`
   - One alias per type, separated by blank lines
   - Field arguments in this order: constraints first (`pattern`, `gt`/`ge`/`le`/`lt`, `min_length`, `max_length`), then `description`, then `examples`
   - Descriptions are full sentences in **English**, regardless of the language used elsewhere in the project
   - Examples are real values, not `"foo"` / `"bar"`
   - If extending an existing `types.py`, **append** new aliases — don't reorder or rewrite existing ones. Group new ones logically.
   - If a useful comment header exists in the file, preserve it.

7. **Refactor call sites — only if the user asks.** This is opt-in because changing signatures touches behavior and call sites. If the user says yes:

   - Update function/method signatures and `BaseModel` field declarations to use the new alias instead of the raw type
   - Add `from <module>.types import BUCode, TopK, ...` to each touched file
   - **Remove** the now-redundant manual validation that the alias makes the type system enforce (e.g., delete the `assert 0 < top_k <= 5000` lines, since Pydantic will validate them) — but ONLY where the type is actually validated by Pydantic (BaseModel fields, validated function args). Leave plain function args alone unless they're going through a `@validate_call` decorator or similar.
   - Touch nothing else. No reformatting, no logic changes.

8. **Verify it parses.** After writing, suggest the user run `python -c "import <module>.types"` (or run it yourself if the user asks) to confirm the new file imports cleanly.

9. **At the end**, give the user:
   - A summary: `N types added to <path>, M call sites refactored`
   - A suggestion to run their type checker (mypy/pyright) and tests
   - A suggestion to run `/git:commit-helper` to commit the changes

## Rules

- NEVER invent constraints. Every `pattern`, `gt`, `le`, `min_length` must come from real evidence in the code (validation logic, regex literals, docstrings, comments, or existing Field definitions).
- NEVER invent descriptions. If you can't find a clear meaning in the code or comments, ask the user for one — don't guess.
- NEVER invent examples. Pull them from tests, fixtures, default values, or comments. If none exist, ask.
- NEVER create a one-off alias that's only used in one place and has no clear reuse potential — that's premature abstraction.
- NEVER refactor call sites without explicit user confirmation.
- NEVER `pip install` Pydantic. Verify it's installed first.
- NEVER mix Pydantic v1 and v2 idioms in the same file — match what the project uses.
- If the same constraint set already exists as a `BaseModel` field in the project, prefer reusing/promoting that to a top-level alias rather than creating a parallel one.
- All descriptions must be written in **English**, even if other docstrings or comments in the project are in another language.
