"""Ollama client for the loop.

Two things measured on this machine that the rest of the code depends on:

  * JSON-schema-constrained output works and is what makes a 7-9B model reliable enough to
    sit inside an unattended loop.
  * With a reasoning model you MUST send think=False alongside `format`, or the thinking
    budget consumes num_predict and message.content comes back EMPTY. This is silent.

Models are loaded one at a time (keep_alive=0). This machine has 24 GB of unified memory
and the training subprocess needs 4-8 GB of it, so two resident models plus a fit is not
safe. A model load costs ~5-15 s against training runs of minutes -- the overhead is noise.
"""
import json
import time
import urllib.error
import urllib.request

HOST = "http://localhost:11434"

ORCHESTRATOR = "qwen3.5:9b"     # reasoning model, benchmarked 39-44 tok/s
CODER = "qwen2.5:7b"            # 57 tok/s; wrote correct LGBM+StratifiedKFold code first try


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        raise OllamaError(f"ollama unreachable at {HOST}: {e}") from e


def alive() -> bool:
    try:
        _post("/api/tags", {}, timeout=5)
        return True
    except Exception:
        try:
            with urllib.request.urlopen(HOST + "/api/tags", timeout=5):
                return True
        except Exception:
            return False


def chat(model: str, system: str, user: str, schema: dict = None,
         temperature: float = 0.2, num_predict: int = 4096,
         keep_alive: int = 0, retries: int = 2) -> str:
    """One turn. Returns message content. If `schema` is given the content is valid JSON."""
    payload = {
        "model": model,
        "think": False,          # MANDATORY with `format` -- see module docstring
        "stream": False,
        "keep_alive": keep_alive,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if schema:
        payload["format"] = schema

    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        r = _post("/api/chat", payload)
        content = (r.get("message") or {}).get("content", "").strip()
        tok = r.get("eval_count", 0)
        dur = max(r.get("eval_duration", 1) / 1e9, 1e-9)
        print(f"    [{model} {tok} tok, {tok/dur:.0f} tok/s, {time.time()-t0:.0f}s]")

        if not content:
            last = "empty content (is think=False set?)"
            continue
        if schema:
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                last = f"invalid JSON: {e}"
                continue
        return content
    raise OllamaError(f"{model} failed after {retries+1} attempts: {last}")


def chat_json(model: str, system: str, user: str, schema: dict, **kw) -> dict:
    return json.loads(chat(model, system, user, schema=schema, **kw))


def extract_code(text: str) -> str:
    """Pull the python block out of a model response.

    Code is requested as a fenced block rather than a JSON string field: escaping a
    multi-hundred-line program through JSON is a reliable way to make a 7B model produce
    something that will not parse.
    """
    if "```" in text:
        blocks, cur, inside = [], [], False
        for line in text.splitlines():
            if line.strip().startswith("```"):
                if inside:
                    blocks.append("\n".join(cur))
                    cur, inside = [], False
                else:
                    inside = True
                continue
            if inside:
                cur.append(line)
        if inside and cur:
            blocks.append("\n".join(cur))
        if blocks:
            return max(blocks, key=len).strip()
    return text.strip()
