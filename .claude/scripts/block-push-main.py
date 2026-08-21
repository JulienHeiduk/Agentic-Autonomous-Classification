#!/usr/bin/env python
"""PreToolUse hook: block direct pushes to main/master.

Reads tool-call JSON from stdin, prints a PreToolUse decision JSON if the
command would push to a protected branch.

Override: prepend `ALLOW_PUSH_MAIN=1 ` to the git push command.
"""
import json
import shlex
import subprocess
import sys

PROTECTED = ("main", "master")
SHELL_BREAK = {"&&", "||", ";", "|", "&"}


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


def positional_args_after_push(tokens: list[str], push_idx: int) -> list[str]:
    """Return non-flag positional args after `git push`, until a shell break."""
    args: list[str] = []
    i = push_idx + 1
    while i < len(tokens):
        t = tokens[i]
        if t in SHELL_BREAK:
            break
        if t.startswith("-"):
            i += 1
            continue
        args.append(t)
        i += 1
    return args


def check_push(tokens: list[str], push_idx: int) -> None:
    """Inspect one `git push` occurrence and emit_deny if it targets main/master."""
    args = positional_args_after_push(tokens, push_idx)

    # Cases:
    #   git push                                   → uses upstream of current branch
    #   git push <remote>                          → uses current branch
    #   git push <remote> <refspec>                → explicit target
    #   git push <remote> <refspec> <refspec>...   → multiple, check all
    if len(args) >= 2:
        for refspec in args[1:]:
            target = refspec.rsplit(":", 1)[-1]  # local:remote → remote side
            if target in PROTECTED:
                emit_deny(
                    f"Direct push to '{target}' is blocked.\n"
                    "Open a pull request, or if you really need to push "
                    "directly, prepend ALLOW_PUSH_MAIN=1 to the command."
                )
        return

    # 0 or 1 positional args → branch is implicit (current HEAD)
    branch = current_branch()
    if branch in PROTECTED:
        emit_deny(
            f"You are on '{branch}' and a bare `git push` would push to it.\n"
            f"Direct push to '{branch}' is blocked. Switch branches, open a PR, "
            "or prepend ALLOW_PUSH_MAIN=1 to override."
        )


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    cmd = (data.get("tool_input") or {}).get("command", "")
    if not cmd:
        sys.exit(0)

    # User-explicit override
    if "ALLOW_PUSH_MAIN=1" in cmd:
        sys.exit(0)

    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        # Unparseable shell — fall through with a coarse string check
        if "git push" not in cmd:
            sys.exit(0)
        # Can't tokenize; be conservative and allow.
        sys.exit(0)

    # Scan for every `git push` occurrence in the token stream
    for i in range(len(tokens) - 1):
        if tokens[i] == "git" and tokens[i + 1] == "push":
            check_push(tokens, i + 1)

    sys.exit(0)


if __name__ == "__main__":
    main()
