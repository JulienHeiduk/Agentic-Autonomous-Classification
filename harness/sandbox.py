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

    problems += _contract_problems(tree, names)
    return problems


# The harness owns cross-validation and scoring. A plugin that builds its own fold loop is
# not just redundant -- it leaks the target into make_features and reports a number nobody
# asked for. These are named in prompts.CONTRACT; checking them here is what makes the rule
# real rather than advisory.
CV_AND_METRIC = {
    "KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit",
    "RepeatedKFold", "RepeatedStratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit",
    "cross_val_score", "cross_validate", "cross_val_predict", "train_test_split",
    "roc_auc_score",
}


def _contract_problems(tree, defined):
    """Contract rules that a run cannot report clearly for itself.

    Both of these produced a plugin that PASSED the old static check, ran, and then died
    with a traceback pointing at the wrong line -- so the repairer spent every attempt
    fixing a function that was fine. Catching them here turns a misleading KeyError into a
    precise instruction, before 691k rows are spent finding out.
    """
    problems = []

    for node in tree.body:
        # Module level must be imports and constants. The harness imports the plugin, so
        # anything executable here runs at import time, before it has any data to run on.
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
        local = [c for c in calls
                 if (getattr(c.func, "id", None) or getattr(c.func, "attr", None)) in defined]
        if local:
            called = ", ".join(sorted({(getattr(c.func, "id", None)
                                        or getattr(c.func, "attr", None)) for c in local}))
            problems.append(
                f"module-level code at line {node.lineno}: calls {called}() at import time. "
                f"The file must define make_features and make_model and NOTHING else at "
                f"module level -- no self-test, no example call, no print.")
        elif isinstance(node, ast.Expr) and calls:
            problems.append(
                f"module-level statement at line {node.lineno}: runs at import time. "
                f"Module level is imports and constants only.")

    # The plugin receives `train` WITH the target column, because it has to drop it. That
    # also means it could compute its own target statistics -- fitted on the same rows it is
    # scored on, which inflates CV and does not survive the leaderboard. The harness already
    # supplies leak-free te_ columns (harness/encode.py), so any other read of the target is
    # a mistake worth catching before it produces a number someone believes.
    dropped = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fname == "drop":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant):
                        dropped.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "addicted_label":
            if id(node) not in dropped:
                problems.append(
                    f"target read at line {node.lineno}: the plugin must not use "
                    f"'addicted_label' for anything except dropping it. Target encoding is "
                    f"already provided as te_notifications_per_day / te_app_opens_per_day.")
        elif isinstance(node, ast.Attribute) and node.attr == "addicted_label":
            problems.append(
                f"target read at line {node.lineno}: .addicted_label may not be read. Use "
                f"the provided te_ columns instead.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in CV_AND_METRIC:
                problems.append(
                    f"forbidden call at line {node.lineno}: {name}(). The harness owns "
                    f"cross-validation and scoring -- do not split folds, do not compute "
                    f"AUC, and never read the target inside make_features.")
    return problems



def _timeout_error(timeout: int, rows) -> str:
    """A timeout the repairer can act on.

    `focused_error` cannot help here: nothing raised, so there is no traceback and no
    failing line to point at. A bare "TIMEOUT after 600s" gives a repairer nothing to
    change, so it returns a near-identical file and times out again -- three attempts,
    three full waits, no progress. Naming the cause and the specific knobs is the
    difference between a repair that converges and one that just costs time.
    """
    where = f"{rows:,} rows" if rows else "the full dataset"
    msg = [f"TIMEOUT: the plugin did not finish within {timeout}s on {where}.",
           "",
           "Nothing crashed -- the code is too EXPENSIVE, not wrong. Do not restructure it."]
    if rows:
        msg += [
            f"{rows:,} rows is a smoke test; a sane configuration finishes in seconds.",
            "The cost is almost always the estimator, not the feature engineering.",
            "",
            "Fix it by shrinking the model in make_model(), keeping the strategy intact:",
            "  * cut n_estimators / iterations hard (e.g. 6000 -> 600, 2000 -> 400)",
            "  * cut depth / max_depth (e.g. 10 -> 6)",
            "  * if you built an ensemble of several models, keep ONE of them",
            "  * drop any n_jobs/thread_count above 8",
            "",
            "Change ONLY those numbers. Do not add features, do not change the algorithm.",
        ]
    else:
        msg += ["Reduce n_estimators/iterations and depth so the fit completes in time."]
    return "\n".join(msg)


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
        return False, {"ok": False, "error": _timeout_error(timeout, rows)}, ""

    out, err = p.stdout, p.stderr
    for line in out.splitlines():
        if line.startswith("###RESULT###"):
            res = json.loads(line[len("###RESULT###"):])
            return bool(res.get("ok")), res, err

    return False, {"ok": False,
                   "error": f"no result line (exit {p.returncode})\n{err[-3000:]}"}, err
