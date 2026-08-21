#!/usr/bin/env python
"""PreToolUse hook: block direct commits on main/master.

Reads tool-call JSON from stdin, prints a PreToolUse decision JSON if the
command would create a commit while HEAD is on a protected branch.

Override: prepend `ALLOW_COMMIT_MAIN=1 ` to the git commit command.
"""
import json
import shlex
import subprocess
import sys

PROTECTED = ("main", "master")


def emit_deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    cmd = (data.get("tool_input") or {}).get("command", "")
    if not cmd:
        sys.exit(0)

    if "ALLOW_COMMIT_MAIN=1" in cmd:
        sys.exit(0)

    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        sys.exit(0)

    has_commit = any(
        tokens[i] == "git" and tokens[i + 1] == "commit"
        for i in range(len(tokens) - 1)
    )
    if not has_commit:
        sys.exit(0)

    branch = current_branch()
    if branch in PROTECTED:
        emit_deny(
            f"You are on '{branch}' — direct commits to '{branch}' are blocked.\n"
            "Create a feature branch (`git checkout -b <name>`), or prepend "
            "ALLOW_COMMIT_MAIN=1 to override."
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
