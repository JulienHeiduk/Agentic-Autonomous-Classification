"""Execute generated plugin code in a contained subprocess.

Generated code is the only untrusted component in the system. It is contained, not
trusted: separate process, wall-clock timeout, import allowlist, no network.

A crash is missing data, not a negative result -- failures are recorded with their
traceback so the repairer can act on them and the idea can be re-queued, never silently
filed as "this doesn't help".
"""
import ast
import json
import subprocess
import sys

from harness import config as C

PLUGINS = C.ROOT / "plugins"
PLUGINS.mkdir(exist_ok=True)

ALLOWED_IMPORTS = {
    "numpy", "np", "pandas", "pd", "sklearn", "lightgbm", "lgb", "xgboost", "xgb",
    "catboost", "scipy", "math", "itertools", "collections", "warnings", "typing",
    "functools", "dataclasses", "re",
}

BANNED = {
    "os", "sys", "subprocess", "socket", "requests", "urllib", "shutil", "pathlib",
    "pickle", "importlib", "builtins", "ctypes", "multiprocessing", "http", "glob",
}


def static_check(code: str):
    """Reject a plugin before running it. Returns list of problems (empty == ok)."""
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: line {e.lineno}: {e.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in BANNED:
                    problems.append(f"banned import: {a.name}")
                elif root not in ALLOWED_IMPORTS:
                    problems.append(f"import not on allowlist: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED:
                problems.append(f"banned import: from {node.module}")
            elif root and root not in ALLOWED_IMPORTS:
                problems.append(f"import not on allowlist: from {node.module}")
        elif isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, "id", None) or getattr(f, "attr", None)
            if name in ("eval", "exec", "compile", "__import__", "open"):
                problems.append(f"banned call: {name}()")

    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for req in ("make_features", "make_model"):
        if req not in names:
            problems.append(f"missing required function: {req}()")
    return problems


def write_plugin(exp_id: str, code: str):
    path = PLUGINS / f"{exp_id}.py"
    path.write_text(code)
    return path


def execute(exp_id: str, code: str, timeout: int = 2400, rows: int = None,
            partition_seed: int = None, plugin_path=None):
    """Static-check, write, run. Returns (ok, result_dict, stderr_text)."""
    problems = static_check(code)
    if problems:
        return False, {"ok": False, "error": "STATIC CHECK FAILED:\n" + "\n".join(problems)}, ""

    path = plugin_path or write_plugin(exp_id, code)
    cmd = [sys.executable, "-u", "-m", "harness.plugin_runner", str(path), exp_id]
    if rows:
        cmd += ["--rows", str(rows)]
    if partition_seed is not None:
        cmd += ["--partition-seed", str(partition_seed)]

    try:
        p = subprocess.run(
            cmd, cwd=str(C.ROOT), capture_output=True, text=True, timeout=timeout,
            env={"PYTHONPATH": str(C.ROOT), "PATH": "/usr/bin:/bin",
                 "PYTHONUNBUFFERED": "1", "HOME": str(C.ROOT)},
        )
    except subprocess.TimeoutExpired:
        return False, {"ok": False, "error": f"TIMEOUT after {timeout}s"}, ""

    out, err = p.stdout, p.stderr
    for line in out.splitlines():
        if line.startswith("###RESULT###"):
            res = json.loads(line[len("###RESULT###"):])
            return bool(res.get("ok")), res, err

    return False, {"ok": False,
                   "error": f"no result line (exit {p.returncode})\n{err[-3000:]}"}, err
