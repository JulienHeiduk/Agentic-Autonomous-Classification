"""Ollama client for the loop.

Two things measured on this machine that the rest of the code depends on:

  * JSON-schema-constrained output works and is what makes a 7-9B model reliable enough to
    sit inside an unattended loop.
  * The correct `think` setting is PER MODEL, and getting it wrong returns 200 OK with an
    empty message.content -- no error, no warning, tokens billed against num_predict.
    Measured on both paths (`format` given vs plain text):

                     think=False              think key omitted
        qwen3.5:9b   ok on both paths         EMPTY on the plain-text path
        gpt-oss:20b  EMPTY with `format`      ok on both paths

    There is no single value that is correct for both, hence THINK/_think_for below.

Models are loaded one at a time (keep_alive=0). This machine has 24 GB of unified memory
and the training subprocess needs 4-8 GB of it, so two resident models plus a fit is not
safe. A model load costs ~5-15 s against training runs of minutes -- the overhead is noise.
"""
import json
import time
import urllib.error
import urllib.request

HOST = "http://localhost:11434"

ORCHESTRATOR = "gpt-oss:20b"    # 55-56 tok/s vs qwen3.5:9b's 38, schema-complete JSON
CODER = "qwen2.5:7b"            # 47 tok/s; qwen3-coder:30b matched its AUC within noise

# Which models need `think` sent, and with what value. None == omit the key entirely.
# The default is False because that is correct for every Qwen and Gemma tag here; only
# gpt-oss needs the key absent. See the module docstring for the measurements.
THINK = {"gpt-oss": None}
DEFAULT_THINK = False


def _think_for(model: str):
    """Resolve the `think` setting for a model tag, by exact tag then by family."""
    if model in THINK:
        return THINK[model]
    return THINK.get(model.split(":")[0], DEFAULT_THINK)


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
        "stream": False,
        "keep_alive": keep_alive,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    think = _think_for(model)    # per model -- see module docstring
    if think is not None:
        payload["think"] = think
    if schema:
        payload["format"] = schema

    last = None
    npred = num_predict
    for attempt in range(retries + 1):
        payload["options"]["num_predict"] = npred
        t0 = time.time()
        r = _post("/api/chat", payload)
        content = (r.get("message") or {}).get("content", "").strip()
        tok = r.get("eval_count", 0)
        dur = max(r.get("eval_duration", 1) / 1e9, 1e-9)
        print(f"    [{model} {tok} tok, {tok/dur:.0f} tok/s, {time.time()-t0:.0f}s]")

        if not content:
            # A reasoning model that spent the whole budget thinking returns 200 OK with
            # empty content, and will do it again at the same budget -- so retrying the
            # identical request cannot succeed. Reasoning length also grows with the
            # context, and build_context() grows every iteration, so any fixed ceiling is
            # only ever temporarily large enough. Escalate instead of repeating.
            last = (f"empty content at num_predict={npred} -- budget spent reasoning, or "
                    f"wrong `think` for {model} (sent think={think!r}); see THINK")
            npred *= 2
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
