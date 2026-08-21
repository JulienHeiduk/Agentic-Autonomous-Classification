#!/usr/bin/env python
"""PreToolUse hook: block `gh pr create`.

Opening pull requests is a manual, human-only step in this repo. Reads
tool-call JSON from stdin and denies any command that runs `gh pr create`.

Override: prepend `ALLOW_GH_PR_CREATE=1 ` to the command.
"""
import json
import shlex
import sys

_MSG = (
    "`gh pr create` is blocked — opening PRs is a manual step here.\n"
    "Create the PR yourself, or prepend ALLOW_GH_PR_CREATE=1 to override."
)


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

    # User-explicit override
    if "ALLOW_GH_PR_CREATE=1" in cmd:
        sys.exit(0)

    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        # Unparseable shell — coarse fallback
        if "gh pr create" in cmd:
            emit_deny(_MSG)
        sys.exit(0)

    # Deny on any `gh pr create` occurrence in the token stream
    for i in range(len(tokens) - 2):
        if (
            tokens[i] == "gh"
            and tokens[i + 1] == "pr"
            and tokens[i + 2] == "create"
        ):
            emit_deny(_MSG)

    sys.exit(0)


if __name__ == "__main__":
    main()
