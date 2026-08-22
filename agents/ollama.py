"""Ollama client for the loop.

Two things measured on this machine that the rest of the code depends on:

  * JSON-schema-constrained output works and is what makes a 7-9B model reliable enough to
    sit inside an unattended loop.
  * The correct `think` setting is PER MODEL, and getting it wrong returns 200 OK with an
    empty message.content -- no error, no warning. When that happens the tokens went to
    message.thinking instead and done_reason is 'length': the model reasoned until it ran
    out of room and never opened the final channel. eval_count does NOT include those
    thinking tokens, so the response looks far under budget while it is in fact truncating.

    Measured on the orchestrator's schema-constrained call, n=8 per cell:

                     think=False    think omitted   think="low"
        qwen3.5:9b   8/8            EMPTY (plain)   n/a
        gpt-oss:20b  EMPTY          2/8             8/8

    There is no single value correct for both, hence THINK/_think_for below. Note the
    2/8: this failure is intermittent, so a one- or two-sample check will pass a setting
    that then fails in the loop. Qualify any new model at n>=8 on this exact call.

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
CODER = "qwen2.5-coder:14b-instruct"   # 5/5 clean first drafts vs qwen2.5:7b's 1/5
                                       # on a stacked-ensemble strategy (n=5 each)

# Which models need `think` sent, and with what value. None == omit the key entirely.
# The default is False because that is correct for every Qwen and Gemma tag here; gpt-oss
# takes a reasoning-effort string instead. See the module docstring for the measurements.
#
# gpt-oss MUST be "low" and not omitted: left at its default effort it spends generation on
# the reasoning channel and emits no final answer, measured at 2/8 usable responses on the
# orchestrator's schema-constrained call. "low" measured 8/8. Two samples are not enough to
# qualify a setting here -- the failure is intermittent, so it hides at small n.
THINK = {"gpt-oss": "low"}
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
         keep_alive: int = 0, retries: int = 2, num_ctx: int = 16384) -> str:
    """One turn. Returns message content. If `schema` is given the content is valid JSON."""
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # num_ctx is the whole window: prompt + thinking + answer. Ollama defaults it
        # to 4096, and build_context() grows every iteration -- at ~3300 prompt tokens
        # that left ~800 for the answer and the JSON truncated mid-string, reported as
        # done_reason 'length' at an eval_count far below num_predict. Raising
        # num_predict cannot fix it; num_ctx is the binding limit. Measured on the
        # orchestrator call: 0/4 valid JSON at the default, 4/4 at 16384.
        "options": {"temperature": temperature, "num_predict": num_predict,
                    "num_ctx": num_ctx},
    }
    think = _think_for(model)    # per model -- see module docstring
    if think is not None:
        payload["think"] = think
    if schema:
        payload["format"] = schema

    last = None
    npred = num_predict
    # Escalation helps a genuinely truncated answer and does nothing for a model stuck
    # emitting one runaway string -- which looks identical from here. Cap it so the
    # useless case costs seconds rather than minutes.
    npred_cap = max(num_predict * 4, 8192)
    for attempt in range(retries + 1):
        payload["options"]["num_predict"] = npred
        t0 = time.time()
        r = _post("/api/chat", payload)
        content = (r.get("message") or {}).get("content", "").strip()
        tok = r.get("eval_count", 0)
        dur = max(r.get("eval_duration", 1) / 1e9, 1e-9)
        print(f"    [{model} {tok} tok, {tok/dur:.0f} tok/s, {time.time()-t0:.0f}s]")

        if not content:
            # The generation went to message.thinking and the final channel never opened.
            # The fix is the `think` setting (see THINK), not the budget -- but a wrong
            # setting is not the only way to get here, so give the next attempt more room
            # rather than re-sending an identical request that already failed.
            think_chars = len(((r.get("message") or {}).get("thinking") or ""))
            last = (f"empty content at num_predict={npred} "
                    f"(thinking={think_chars} chars, done_reason="
                    f"{r.get('done_reason')!r}) -- wrong `think` for {model}? "
                    f"sent think={think!r}; see THINK")
            npred = min(npred * 2, npred_cap)
            continue
        if schema:
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                # Almost always truncation, not malformed generation: the schema forces
                # well-formed JSON, so the usual way it fails to parse is running out of
                # budget mid-string. Same escalation as the empty case, same reason --
                # retrying at an identical budget cannot fix a length problem.
                last = f"invalid JSON at num_predict={npred}: {e}"
                npred = min(npred * 2, npred_cap)
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
