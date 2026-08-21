#!/usr/bin/env python
"""PreToolUse hook: block `git push` on ANY branch.

Pushing is a manual, human-only step in this repo. Reads tool-call JSON from
stdin and denies any command that runs `git push`, regardless of branch.

Override: prepend `ALLOW_PUSH=1 ` to the git push command.
"""
import json
import shlex
import sys

_MSG = (
    "`git push` is blocked on all branches — pushing is a manual step here.\n"
    "Push it yourself, or prepend ALLOW_PUSH=1 to the command to override."
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
    if "ALLOW_PUSH=1" in cmd:
        sys.exit(0)

    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        # Unparseable shell — coarse fallback
        if "git push" in cmd:
            emit_deny(_MSG)
        sys.exit(0)

    # Deny on any `git push` occurrence in the token stream
    for i in range(len(tokens) - 1):
        if tokens[i] == "git" and tokens[i + 1] == "push":
            emit_deny(_MSG)

    sys.exit(0)


if __name__ == "__main__":
    main()
