#!/usr/bin/env python
"""PreToolUse hook: block git commit if staged changes contain likely secrets.

Reads tool-call JSON from stdin. If the command is a `git commit`, scans the
staged diff (and the unstaged diff too, when `-a`/`--all` is used) for common
secret patterns and `.env` files. Prints a PreToolUse decision JSON to deny
the call when something is found.

Override: prepend `SKIP_SECRETS_CHECK=1 ` to the git commit command.
"""
import json
import re
import subprocess
import sys

PATTERNS = [
    ("AWS Access Key (AKIA…)", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Google API Key (AIza…)", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("Private key block",       re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("GitHub token (gh[pousr]_)", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("Slack token (xox[baprs]-)", re.compile(r"xox[baprs]-[0-9a-zA-Z\-]{10,}")),
    ("JWT (eyJ…)",
        re.compile(r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")),
    ("Generic api_key/secret/password assignment",
        re.compile(
            r"""(?i)(?:api[_-]?key|apikey|secret|password|token)\s*[:=]\s*['"][^'"\s]{12,}['"]"""
        )),
]

ENV_FILE_RE = re.compile(r"(?:^|/)\.env(?:[./]|$)")
ENV_FILE_ALLOW_SUFFIX = re.compile(r"\.(example|sample|template|dist)$")

# Top-level `env/` folder is reserved for local-only credentials
# (.env.production, .env.staging, etc.). Nothing under it should ever
# land in git — even files that don't match the .env* pattern (a stray
# notes.txt, a downloaded service-account JSON, etc.).
ENV_FOLDER_PREFIX_RE = re.compile(r"^env/")


def emit_deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    cmd = (data.get("tool_input") or {}).get("command", "")
    if not cmd:
        sys.exit(0)

    if not re.search(r"(?:^|[^a-zA-Z])git\s+commit\b", cmd):
        sys.exit(0)

    if "SKIP_SECRETS_CHECK=1" in cmd:
        sys.exit(0)

    # Collect diffs to scan: always staged; unstaged too if `-a` / `--all`.
    chunks = []
    try:
        chunks.append(subprocess.check_output(
            ["git", "diff", "--cached", "--diff-filter=ACM"],
            stderr=subprocess.DEVNULL, text=True, errors="replace",
        ))
    except Exception:
        pass

    uses_all = bool(
        re.search(r"git\s+commit\s+(?:\S+\s+)*-(?:[a-zA-Z]*a[a-zA-Z]*)", cmd)
        or re.search(r"git\s+commit\s+(?:\S+\s+)*--all\b", cmd)
    )
    if uses_all:
        try:
            chunks.append(subprocess.check_output(
                ["git", "diff", "--diff-filter=ACM"],
                stderr=subprocess.DEVNULL, text=True, errors="replace",
            ))
        except Exception:
            pass

    diff_text = "\n".join(chunks)

    findings = []

    if diff_text.strip():
        added = "\n".join(
            line for line in diff_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        for name, pattern in PATTERNS:
            if pattern.search(added):
                findings.append(name)

    # .env files being staged
    try:
        staged_files = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"],
            stderr=subprocess.DEVNULL, text=True, errors="replace",
        ).splitlines()
    except Exception:
        staged_files = []

    bad_env = [
        f for f in staged_files
        if ENV_FILE_RE.search(f) and not ENV_FILE_ALLOW_SUFFIX.search(f)
    ]
    if bad_env:
        findings.append(".env file(s) staged: " + ", ".join(bad_env))

    bad_env_folder = [
        f for f in staged_files
        if ENV_FOLDER_PREFIX_RE.match(f) and not ENV_FILE_ALLOW_SUFFIX.search(f)
    ]
    if bad_env_folder:
        findings.append(
            "file(s) under env/ staged: " + ", ".join(bad_env_folder)
        )

    if findings:
        bullets = "\n".join(f"  - {f}" for f in findings)
        emit_deny(
            "Possible secrets detected in this commit:\n"
            f"{bullets}\n\n"
            "Unstage the offending content and try again. If this is a false "
            "positive, prepend SKIP_SECRETS_CHECK=1 to the command."
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
