#!/usr/bin/env python
"""PreToolUse hook: block branch creation.

Reads tool-call JSON from stdin, prints a PreToolUse decision JSON if the
command would bring a new branch ref into existence. Branch naming is the
human's call — Claude works on whatever branch it is handed.

Blocked:
    git checkout -b / -B <name>
    git switch -c / -C / --create / --force-create <name>
    git branch <name>                (bare positional = create)
    git branch -c / -C / --copy      (copy = new ref)
    git branch -m / -M / --move      (rename = new name appears)
    git worktree add -b / -B <name>

Allowed (read-only or destructive-but-not-creating):
    git branch                       (list)
    git branch -a / -r / -v / --list / --show-current / --merged / ...
    git branch -d / -D <name>        (delete)
    git checkout <existing> / -- <path>
    git switch <existing> / -

Remote branch creation via `git push` is not handled here — block-push.py
already denies every push, so a new remote ref can't be created either way.

Override: prepend `ALLOW_BRANCH_CREATE=1 ` to the command.
"""

import json
import shlex
import sys

OVERRIDE = "ALLOW_BRANCH_CREATE=1"

# Token boundaries between chained commands, so `git status && git checkout -b x`
# is scanned as two invocations rather than one long argument list.
SHELL_OPS = {"&&", "||", ";", "|", "&", "(", ")", "{", "}"}

# `git` global options that consume the following token, so the real
# subcommand isn't mistaken for their value (`git -C /repo checkout -b x`).
GIT_GLOBAL_VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}

# `git branch` options that consume the following token — their value must not
# be mistaken for a new branch name (`git branch --contains HEAD` is a query).
BRANCH_VALUE_OPTS = {
    "--contains",
    "--no-contains",
    "--merged",
    "--no-merged",
    "--points-at",
    "--sort",
    "--format",
    "--color",
    "-u",
    "--set-upstream-to",
}

# Presence of any of these means the invocation lists or deletes; it cannot
# create. Deliberately conservative: flags that *could* accompany a create
# (e.g. -v, -t, -f) are omitted so they still fall through to the check.
BRANCH_READ_ONLY_FLAGS = {
    "-d",
    "-D",
    "--delete",
    "-l",
    "--list",
    "-a",
    "--all",
    "-r",
    "--remotes",
    "--show-current",
    "--merged",
    "--no-merged",
    "--contains",
    "--no-contains",
    "--points-at",
    "--unset-upstream",
    "--edit-description",
}

BRANCH_CREATE_FLAGS = {
    "-c",
    "-C",
    "--copy",
    "--force-copy",
    "-m",
    "-M",
    "--move",
    "--force-move",
}

SWITCH_CREATE_FLAGS = {"-c", "-C", "--create", "--force-create"}


def emit_deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def _short_flag_has(token: str, letters: str) -> bool:
    """True if ``token`` is a short-flag cluster containing one of ``letters``.

    Catches the bundled form (``git checkout -qb name``) as well as plain
    ``-b``. Long options (``--foo``) and bare ``--`` are excluded.
    """
    if not token.startswith("-") or token.startswith("--"):
        return False
    return any(ch in token[1:] for ch in letters)


def tokenize(cmd: str) -> list[str]:
    """Shell-split ``cmd``, keeping operators as their own tokens.

    Plain ``shlex.split`` leaves ``;`` glued to the preceding word, so
    ``git fetch; git switch -c x`` tokenised to ``['git', 'fetch;', ...]``
    and the second invocation was swallowed into the first's arguments.
    ``punctuation_chars`` splits ``; & | ( )`` out while still respecting
    quoting, so a literal ``;`` inside ``-m "a; b"`` stays part of its token.
    """
    lexer = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def git_invocations(tokens: list[str]) -> list[list[str]]:
    """Return the argument list of every `git ...` invocation in the command."""
    out: list[list[str]] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "git":
            args: list[str] = []
            j = i + 1
            # A bare `git` also terminates the previous invocation, so a
            # chained command still splits even if its separator was consumed
            # by something this tokenizer doesn't model.
            while j < len(tokens) and tokens[j] not in SHELL_OPS and tokens[j] != "git":
                args.append(tokens[j])
                j += 1
            out.append(args)
            i = j
        else:
            i += 1
    return out


def strip_global_opts(args: list[str]) -> list[str]:
    """Drop git's own options so args[0] is the subcommand."""
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in GIT_GLOBAL_VALUE_OPTS:
            i += 2
        elif tok.startswith("-"):
            i += 1
        else:
            return args[i:]
    return []


def creates_branch(args: list[str]) -> str | None:
    """Return a reason string if this invocation would create a branch."""
    args = strip_global_opts(args)
    if not args:
        return None
    sub, rest = args[0], args[1:]

    if sub == "checkout" and any(_short_flag_has(t, "bB") for t in rest):
        return "`git checkout -b` creates a new branch"

    if sub == "switch" and any(
        t in SWITCH_CREATE_FLAGS or _short_flag_has(t, "cC") for t in rest
    ):
        return "`git switch -c` creates a new branch"

    if sub == "worktree" and rest[:1] == ["add"]:
        if any(_short_flag_has(t, "bB") for t in rest[1:]):
            return "`git worktree add -b` creates a new branch"

    if sub == "branch":
        for tok in rest:
            base = tok.split("=", 1)[0]
            if tok in BRANCH_CREATE_FLAGS or base in BRANCH_CREATE_FLAGS:
                return f"`git branch {tok}` creates a new branch ref"
        if any(
            tok in BRANCH_READ_ONLY_FLAGS or tok.split("=", 1)[0] in BRANCH_READ_ONLY_FLAGS
            for tok in rest
        ):
            return None
        # A bare positional on `git branch` is a create.
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok in BRANCH_VALUE_OPTS:
                i += 2
                continue
            if tok.startswith("-"):
                i += 1
                continue
            return f"`git branch {tok}` creates a new branch"
        return None

    return None


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    cmd = (data.get("tool_input") or {}).get("command", "")
    if not cmd:
        sys.exit(0)

    if OVERRIDE in cmd:
        sys.exit(0)

    try:
        tokens = tokenize(cmd)
    except ValueError:
        sys.exit(0)

    for args in git_invocations(tokens):
        reason = creates_branch(args)
        if reason:
            emit_deny(
                f"Branch creation is blocked: {reason}.\n"
                "Ask the user to create and check out the branch themselves, "
                "then continue on it. To override, prepend "
                f"{OVERRIDE} to the command."
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
