#!/usr/bin/env python3
"""Set a pull request's body and PROVE it took, or read one back and prove it matches.

The whole tool exists for one failure (#2918): `gh pr edit <N> --body-file <f>` returned rc=1
with a `GraphQL: Projects (classic) is being deprecated ...` error and **did not apply the
edit**, while the author carried on believing it had. On #2910 the corrected body *was* the
deliverable -- four blocking review findings, all prose -- so a silently-unapplied edit would
have meant merging a PR whose stated reasoning had been rejected.

A PR body is where reviewers, the merge step and every later reader take the author's account of
the change, and it fails in the direction nobody checks: `gh pr edit` succeeding is assumed, and
almost nobody greps the body back.

WHAT THIS TOOL ASSERTS, AND WHY IT IS NOT "check the exit code"

The exit code is the thing that lied. So `set` **always reads the body back from GitHub and
compares it against the bytes you meant to publish**, whatever the write reported:

  * write rc=0 but the live body still differs  -> exit 1. This is the reported defect's shape
    and the only reading that would have caught it.
  * write rc!=0 but the live body now matches   -> exit 0, with the write's stderr reported.
    A write that errored on some unrelated sub-query and still applied is a success; refusing it
    would train people to ignore this tool.

The verdict is therefore always a statement about the live body, never about a subprocess.

WHY REST RATHER THAN `gh pr edit`

`gh pr edit` goes through GraphQL, which is where the deprecation error came from; the REST
endpoint applied the same body first try on the same failure. So the write is
`gh api -X PATCH repos/{owner}/{repo}/pulls/<N> --input -`, with the payload as JSON on stdin so
no `-F`/`-f` field parsing can reinterpret a body that happens to look like a number or a bool.

RE-MEASURED 2026-09-11, and the measurement is the reason the read-back is the fix rather than a
version bump: on this machine neither reported symptom reproduces any more -- `gh pr edit
--body-file` applies and exits 0, and trap 240's `gh pr view <N> --comments` exits 0 with output.
The `gh` binary has not changed (2.93.0, released 2026-05-27, i.e. three months BEFORE the
2026-08-23 reproduction), so what changed is server-side and outside this repository's control.
That cuts both ways: the trigger is not something we can pin to a version, and it can return the
same way it left.

COMPARISON IS NEWLINE-INSENSITIVE, DELIBERATELY

GitHub stores what the API sends but normalises to CRLF on some paths (a body last edited in the
web UI, notably). A tool that reported a mismatch for line endings alone would cry wolf on a body
that is byte-for-byte the intended prose, and a gate people learn to ignore protects nothing. So
`\\r\\n` is folded to `\\n` and trailing blank lines are ignored; every other byte must match, and
the report names the first line that differs.

Usage:
    pr_body.py set    <pr> (--body-file F | --body TEXT) [--repo-dir DIR]
    pr_body.py verify <pr> (--body-file F | --body TEXT) [--repo-dir DIR]
    pr_body.py get    <pr> [--out F] [--repo-dir DIR]

Exit status: 0 the live body is what you intended; 1 it is not (say so and stop); 2 the tool
could not reach a verdict (`gh` missing, network, malformed payload). 2 is distinct from 1 for
the same reason `pr_merge_gate.py` distinguishes them: "could not check" must never be readable
as "checked, and it is fine".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


class BodyError(RuntimeError):
    """No verdict could be reached (bad input, `gh` failure, malformed payload)."""


def run(cmd: list[str], cwd: str | None = None, stdin: str | None = None):
    """Run a command, never raising. Returns (rc, stdout, stderr).

    Module-level and injectable so every test arm is hermetic: nothing in this file's test suite
    touches the network or a real pull request.

    UTF-8 is forced rather than left to `text=True`, and that is not tidiness. `text=True` decodes
    with the LOCALE encoding, which on a Windows host is cp1252 -- so `gh`'s UTF-8 output came back
    as mojibake and an em dash in the body read as `\u00e2\u20ac\u201d`. The tool then reported
    BODY MISMATCH on a body that was byte-for-byte correct: a false alarm from the one instrument
    whose job is to say whether the live text is right, and one that would fire on this
    repository's own PR bodies, which use em dashes throughout. Caught by a live smoke test on a
    real PR; the arm below pins it without one.
    """
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True,
                           input=None if stdin is None else stdin.encode("utf-8"))
        return (p.returncode,
                p.stdout.decode("utf-8", "replace"),
                p.stderr.decode("utf-8", "replace"))
    except OSError as exc:
        return 127, "", str(exc)


def normalise(text: str) -> str:
    """Fold the differences that are not differences: CRLF, a lone CR, and trailing blank lines.

    Nothing else is touched. Leading whitespace, interior blank lines and trailing spaces WITHIN
    a line all stay significant -- in Markdown two trailing spaces are a hard line break, so
    stripping them would let the tool bless a body that renders differently from the file.
    """
    if text is None:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def compare(intended: str, live: str) -> tuple[bool, str]:
    """Pure verdict on two bodies. Returns (ok, human explanation).

    The explanation names the FIRST differing line, because the realistic failure is not a body
    that is wholly wrong -- it is one that is wholly STALE, and "line 1 differs" plus the two
    texts is what tells those apart at a glance.
    """
    a, b = normalise(intended), normalise(live)
    if a == b:
        return True, "live body matches the intended body (%d chars)" % len(a)
    if not b:
        return False, "the live body is EMPTY while %d chars were intended" % len(a)
    alines, blines = a.split("\n"), b.split("\n")
    for i, (x, y) in enumerate(zip(alines, blines), start=1):
        if x != y:
            return False, ("line %d differs:\n    intended: %r\n    live:     %r"
                           % (i, x[:120], y[:120]))
    return False, ("the first %d line(s) match but the bodies differ in length "
                   "(intended %d lines, live %d lines)"
                   % (min(len(alines), len(blines)), len(alines), len(blines)))


def _api(pr: int, repo_dir: str, extra: list[str], stdin: str | None = None):
    cmd = ["gh", "api", "repos/{owner}/{repo}/pulls/%d" % pr] + extra
    return run(cmd, cwd=repo_dir, stdin=stdin)


def read_body(pr: int, repo_dir: str = ".") -> str:
    """Fetch the live body. Raises BodyError rather than returning "" on a failed fetch.

    Returning "" would make an unreachable API indistinguishable from a genuinely empty body,
    and the caller's next move on an empty body is to overwrite it.
    """
    rc, out, err = _api(pr, repo_dir, ["--jq", ".body"])
    if rc != 0:
        raise BodyError("could not read PR #%d: gh api exited %d: %s" % (pr, rc, err.strip()[:300]))
    # `--jq .body` appends exactly one newline of its own; a null body prints "null".
    if out.endswith("\n"):
        out = out[:-1]
    if out == "null":
        return ""
    return out


def write_body(pr: int, body: str, repo_dir: str = ".") -> tuple[int, str]:
    """PATCH the body over REST. Returns (rc, stderr) -- the CALLER must still read it back.

    Deliberately does not raise on a non-zero rc. Whether the edit took is a question about the
    live body, and a write that errors while still applying is exactly one of the two cases
    #2918 is about.
    """
    payload = json.dumps({"body": body})
    rc, _, err = _api(pr, repo_dir, ["-X", "PATCH", "--input", "-"], stdin=payload)
    return rc, err


def set_and_verify(pr: int, body: str, repo_dir: str = ".") -> tuple[bool, list[str]]:
    """Write, then read back. The verdict is the read-back, never the write's exit code."""
    notes = []
    rc, err = write_body(pr, body, repo_dir)
    if rc != 0:
        notes.append("NOTE: the write exited %d (%s) -- checking the live body anyway, because "
                     "that exit code is precisely what #2918 showed cannot be trusted in either "
                     "direction" % (rc, err.strip()[:200] or "no stderr"))
    live = read_body(pr, repo_dir)
    ok, why = compare(body, live)
    notes.append(why)
    if ok and rc != 0:
        notes.append("the edit APPLIED despite the non-zero exit code")
    if not ok and rc == 0:
        notes.append("the write reported SUCCESS and the body is still wrong -- this is the "
                     "#2918 failure; retry, or the body stays stale while you believe otherwise")
    return ok, notes


def emit(stream, text: str) -> None:
    """Write one line of REPORT text, degrading characters the stream cannot encode.

    A console codepage narrower than the body is not hypothetical: on a Windows host `sys.stdout`
    is cp1252, and printing a body containing an em dash or a not-equal sign raises
    UnicodeEncodeError. That crash is caught by main()'s catch-all and reported as
    "COULD NOT VERIFY" -- so a tool whose entire job is to tell you whether the live text is right
    would answer "I cannot tell" for the sole reason that it could not PRINT the answer. Found by
    running `get` against a real PR.

    Degrading rather than raising is correct for a report: the verdict and the line number are the
    payload, and a substituted glyph in a quoted line does not change either. `write_body_out`
    below is the opposite case and stays byte-exact.
    """
    enc = getattr(stream, "encoding", None)
    if enc:
        text = text.encode(enc, "replace").decode(enc, "replace")
    stream.write(text + "\n")


def write_body_out(stream, body: str) -> None:
    """Write a body to stdout BYTE-EXACTLY, so `pr_body.py get <N> > file` round-trips.

    The opposite choice from `emit`: this is content, not a report, and a substituted character
    here would silently corrupt whatever the caller does with it next -- including feeding it back
    to `verify`, which would then report a mismatch the body does not have. So it goes out as UTF-8
    bytes through the binary buffer, bypassing the console codepage entirely. Falls back to a plain
    write for streams that have no buffer (a StringIO under test).
    """
    buf = getattr(stream, "buffer", None)
    if buf is None:
        stream.write(body + "\n")
        return
    stream.flush()
    buf.write(body.encode("utf-8") + b"\n")
    buf.flush()


def _read_intended(args) -> str:
    if args.body is not None:
        return args.body
    try:
        with open(args.body_file, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise BodyError("could not read --body-file %s: %s" % (args.body_file, exc))
    except UnicodeDecodeError as exc:
        raise BodyError("--body-file %s is not UTF-8: %s" % (args.body_file, exc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Set or verify a PR body, with the verdict taken from the live body.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_body(p):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--body-file", help="file holding the intended body (UTF-8)")
        g.add_argument("--body", help="the intended body, inline")
        return p

    for name, help_ in (("set", "write the body, then prove the live body matches"),
                        ("verify", "prove the live body matches, without writing")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("pr", type=int)
        p.add_argument("--repo-dir", default=".", help="repository to run gh in")
        with_body(p)

    p = sub.add_parser("get", help="print the live body")
    p.add_argument("pr", type=int)
    p.add_argument("--repo-dir", default=".")
    p.add_argument("--out", help="write to this file instead of stdout")

    args = ap.parse_args(argv)

    try:
        if args.cmd == "get":
            body = read_body(args.pr, args.repo_dir)
            if args.out:
                with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(body)
                print("PR #%d body -> %s (%d chars)" % (args.pr, args.out, len(body)),
                      file=sys.stderr)
            else:
                write_body_out(sys.stdout, body)
            return 0

        intended = _read_intended(args)
        if args.cmd == "set":
            ok, notes = set_and_verify(args.pr, intended, args.repo_dir)
        else:
            ok, why = compare(intended, read_body(args.pr, args.repo_dir))
            notes = [why]
    except BodyError as exc:
        print("PR #%d: COULD NOT VERIFY -- %s" % (args.pr, exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- see pr_merge_gate.py: a crash must not read as "no"
        print("PR #%d: COULD NOT VERIFY -- unexpected %s: %s"
              % (args.pr, type(exc).__name__, exc), file=sys.stderr)
        return 2

    stream = sys.stdout if ok else sys.stderr
    emit(stream, "PR #%d: %s" % (args.pr, "BODY VERIFIED" if ok else "BODY MISMATCH"))
    for n in notes:
        emit(stream, "  " + n)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
