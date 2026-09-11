#!/usr/bin/env python3
"""Self-checking tests for pr_body.py (exit code is truth).

Nothing here touches the network or a real pull request: `pr_body.run` is replaced by a fake that
records the argv it was handed and answers from an in-memory body, so every `gh` outcome -- an
edit that silently does not apply, an edit that errors and applies anyway, an unreachable API --
is reachable on demand instead of by waiting for GitHub to misbehave.

HOW TO TELL A REAL ARM FROM ONE THAT REDDENS NOTHING -- ask "what else could satisfy this?"

This tool's verdict is a boolean, so the trap is the same one `test_pr_merge_gate.py` names: a
tool that answered MISMATCH to everything would satisfy every failure arm here. So the suite
leads with a positive control, and each failure arm asserts on the stated REASON, not merely on
the boolean.

The arm that matters most is `set` with a write that exits **0** while the body does not change.
That is the exact #2918 shape, and it is the one case where every signal a caller normally reads
says success. An implementation that trusted the write's exit code passes every other arm in this
file and fails only that one.
"""

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pr_body  # noqa: E402
from pr_body import BodyError, compare, normalise  # noqa: E402

FAILURES = []


def case(label, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + label)
    if not ok:
        FAILURES.append("%s (got %r, want %r)" % (label, got, want))


class FakeGh:
    """A `gh` that answers from one in-memory body, with each failure mode switchable.

    `applies` is the knob the reported defect turns: a write can report any exit code and
    independently choose whether it actually changed anything. Those two being independent is the
    whole subject -- a fake that tied them together could not express the bug.
    """

    def __init__(self, body="", write_rc=0, applies=True, read_rc=0, write_err="",
                 repo_name="owner/canonical", repo_rc=0):
        self.repo_name = repo_name
        self.repo_rc = repo_rc
        self.body = body
        self.write_rc = write_rc
        self.applies = applies
        self.read_rc = read_rc
        self.write_err = write_err
        self.calls = []

    def __call__(self, cmd, cwd=None, stdin=None):
        self.calls.append((list(cmd), stdin))
        if "repo" in cmd and "view" in cmd:
            return self.repo_rc, self.repo_name + "\n", "" if self.repo_rc == 0 else "no remote"
        # Every other call must already be addressed at the resolved name, never at the
        # placeholders: `{owner}/{repo}` is what 307s on a renamed repository.
        assert not any("{owner}" in a for a in cmd), "endpoint still uses {owner}/{repo}: %r" % (cmd,)
        if "-X" in cmd and "PATCH" in cmd:
            if self.applies:
                self.body = json.loads(stdin)["body"]
            return self.write_rc, "", self.write_err
        if "--jq" in cmd:
            if self.read_rc != 0:
                return self.read_rc, "", "HTTP 404: Not Found"
            return 0, ("null" if self.body is None else self.body) + "\n", ""
        raise AssertionError("unexpected gh invocation: %r" % (cmd,))


class gh:
    """Install a fake `gh` for the duration of a block, and always put the real one back.

    A context manager rather than an assignment because a leaked fake does not fail -- the next
    `drive()` saves it as "the real one" and restores it just as faithfully, so every later arm
    silently runs against the wrong stub while still reporting ok.
    """

    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        self.real = pr_body.run
        pr_body.run = self.fake
        return self.fake

    def __exit__(self, *exc):
        pr_body.run = self.real
        return False


def drive(fake, argv):
    """Run main() under the fake, capturing both streams. Returns (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with gh(fake), redirect_stdout(out), redirect_stderr(err):
        rc = pr_body.main(argv)
    return rc, out.getvalue(), err.getvalue()


BODY = "## Approach\n\nOne paragraph, and a `code span`.\n"

print("\n-- the positive control (without it every failure arm below is void)")
fake = FakeGh(body="stale")
rc, out, err = drive(fake, ["set", "42", "--body", BODY])
case("a write that applies exits 0", rc, 0)
case("and says so on stdout", "BODY VERIFIED" in out, True)
case("the live body really changed", fake.body, BODY)
case("verdict text is not on stderr", "BODY VERIFIED" in err, False)

print("\n-- the reported defect: write says SUCCESS, body does not change (#2918)")
fake = FakeGh(body="a stale body from before the correction", write_rc=0, applies=False)
rc, out, err = drive(fake, ["set", "42", "--body", BODY])
case("exits 1", rc, 1)
case("names it a mismatch", "BODY MISMATCH" in err, True)
case("names the #2918 shape explicitly", "#2918 failure" in err, True)
case("reports the first differing line", "line 1 differs" in err, True)
case("nothing reassuring went to stdout", out, "")

print("\n-- the mirror case: write ERRORS but the edit applied anyway")
# rc=1 with the body correct must PASS. Refusing here would make the tool cry wolf on exactly the
# GraphQL-noise case #2918 describes, and a gate people learn to ignore protects nothing.
fake = FakeGh(body="stale", write_rc=1, applies=True,
              write_err="GraphQL: Projects (classic) is being deprecated.")
rc, out, err = drive(fake, ["set", "42", "--body", BODY])
case("exits 0 because the LIVE body is right", rc, 0)
case("but the non-zero write is reported", "the write exited 1" in out, True)
case("and the outcome is spelled out", "APPLIED despite" in out, True)

print("\n-- the verdict must never be the write's exit code, in either direction")
# Stated as one arm because the two cases above are only interesting together: an implementation
# that returns `rc == 0` passes the first and fails this; one that returns `rc != 0` does the
# reverse. Only reading the live body passes both.
seen = []
for wrc, applies in ((0, False), (1, True)):
    f = FakeGh(body="stale", write_rc=wrc, applies=applies)
    seen.append((wrc, drive(f, ["set", "42", "--body", BODY])[0]))
case("rc=0 + not applied -> 1, rc=1 + applied -> 0", seen, [(0, 1), (1, 0)])

print("\n-- an empty live body is called out, not diffed line by line")
fake = FakeGh(body="", write_rc=0, applies=False)
rc, _, err = drive(fake, ["set", "42", "--body", BODY])
case("exits 1", rc, 1)
case("says the live body is empty", "live body is EMPTY" in err, True)

print("\n-- a body GitHub never had reads as empty, not as the string 'null'")
# `gh api --jq .body` prints the four characters `null` for a PR opened with no body at all.
# Read literally, a later `verify` would be comparing the intended text against the word "null".
with gh(FakeGh(body=None)):
    case("a null body reads as the empty string", pr_body.read_body(42), "")

print("\n-- an unreachable API is COULD NOT VERIFY (2), never a mismatch (1)")
fake = FakeGh(body=BODY, read_rc=1)
rc, out, err = drive(fake, ["verify", "42", "--body", BODY])
case("exits 2", rc, 2)
case("says it could not verify", "COULD NOT VERIFY" in err, True)
case("does NOT claim a mismatch", "BODY MISMATCH" in err, False)

print("\n-- verify never writes")
fake = FakeGh(body=BODY)
rc, out, err = drive(fake, ["verify", "42", "--body", BODY])
case("matching body exits 0", rc, 0)
case("no PATCH was issued", [c for c, _ in fake.calls if "PATCH" in c], [])

print("\n-- the write goes over REST with a JSON payload, not through `gh pr edit`")
# `gh pr edit` is the GraphQL path the deprecation error came from, and `-F body=@file` would let
# gh's field parsing reinterpret the value. Both are pinned here because both are the plausible
# "simplification" of this code.
fake = FakeGh(body="stale")
drive(fake, ["set", "42", "--body", BODY])
patch = [(c, s) for c, s in fake.calls if "PATCH" in c][0]
case("uses gh api", patch[0][:2], ["gh", "api"])
case("targets the REST pulls endpoint at the RESOLVED name, not the placeholders",
     patch[0][2], "repos/owner/canonical/pulls/42")
case("sends the body as JSON on stdin", json.loads(patch[1]), {"body": BODY})
case("never invokes `gh pr edit`",
     any("pr" in c and "edit" in c for c, _ in fake.calls), False)

print("\n-- --body-file is read as UTF-8 and a missing one is COULD NOT VERIFY")
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "body.md")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("a body with an em dash — and a å\n")
    fake = FakeGh(body="stale")
    rc, out, err = drive(fake, ["set", "42", "--body-file", p])
    case("non-ASCII body applies", rc, 0)
    case("and survives the round trip", fake.body, "a body with an em dash — and a å\n")

    rc, out, err = drive(FakeGh(), ["set", "42", "--body-file", os.path.join(d, "nope.md")])
    case("a missing body file exits 2", rc, 2)
    case("and says which file", "nope.md" in err, True)

print("\n-- comparison folds line endings and trailing blank lines, and NOTHING else")
# GitHub normalises a body to CRLF on some paths, notably one last edited in the web UI. Failing
# on that alone would report a mismatch for a body that is byte-for-byte the intended prose.
case("CRLF equals LF", compare("a\r\nb\r\n", "a\nb\n")[0], True)
case("a lone CR equals LF", compare("a\rb", "a\nb")[0], True)
case("trailing blank lines are ignored", compare("a\n\n\n", "a")[0], True)
# ...but a trailing SPACE is a hard line break in Markdown, so it renders differently and must
# not be folded away. This is the arm that stops `normalise` growing a `.strip()`.
case("a trailing space within a line still differs", compare("a  \nb", "a\nb")[0], False)
case("a leading space still differs", compare(" a", "a")[0], False)
case("an interior blank line still differs", compare("a\n\nb", "a\nb")[0], False)
case("normalise leaves interior text alone", normalise("x\r\ny\r\n\r\n"), "x\ny")

print("\n-- a longer body that is a strict PREFIX of the live one is still a mismatch")
# The stale-body failure can also look like this: a correction that appends a section, where
# every line that exists matches. A zip()-based comparison that stopped at the shorter body
# would report a match.
ok, why = compare("one\ntwo", "one\ntwo\nthree")
case("prefix is a mismatch", ok, False)
case("and the reason names the length", "differ in length" in why, True)

print("\n-- read_body raises rather than returning '' when the fetch fails")
# Returning '' would make an unreachable API indistinguishable from an empty body, and the
# caller's next move on an empty body is to overwrite it.
with gh(FakeGh(body=BODY, read_rc=1)):
    try:
        pr_body.read_body(42)
        case("a failed read raises rather than returning a body", False, True)
    except BodyError as exc:
        case("failed read raises BodyError", "could not read PR #42" in str(exc), True)

print("\n-- the REAL run() decodes as UTF-8, whatever the host locale says")
# Hermetic but not faked: this drives a genuine subprocess, because the bug it pins lives in the
# one function every fake replaces. `text=True` decodes with the locale encoding -- cp1252 on a
# Windows host -- so `gh`'s UTF-8 came back as mojibake and a correct body reported BODY MISMATCH.
# It was found by running `verify` against a real PR, which is exactly the check no fake could
# have made; this arm makes it reproducible with no network.
EM_DASH = "guard bypassed \u2014 but a consumer cannot tell"
rc, out, err = pr_body.run(
    [sys.executable, "-c",
     "import sys; sys.stdout.buffer.write(%r)" % (EM_DASH.encode("utf-8"),)])
case("a real subprocess runs", rc, 0)
case("UTF-8 output survives the host locale", out, EM_DASH)
case("and is not mojibake", "\u00e2" in out, False)

# The write direction too: a non-ASCII body must reach the subprocess as UTF-8 bytes.
rc, out, err = pr_body.run(
    [sys.executable, "-c",
     "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
    stdin=EM_DASH)
case("UTF-8 stdin survives the round trip", out, EM_DASH)

print("\n-- reporting a non-ASCII body must not CRASH on a narrow console codepage")
# Found by dogfooding: `get` against a real PR raised UnicodeEncodeError on a cp1252 stdout for a
# body containing U+2260, and main()'s catch-all turned that into "COULD NOT VERIFY" -- the tool
# answering "I cannot tell" purely because it could not print the answer. The fixture is a real
# cp1252 text stream, so the arm reproduces it on every platform rather than only on Windows.
def cp1252_stream():
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict", newline="")


s = cp1252_stream()
try:
    pr_body.emit(s, "line 1 differs: \u2260 and \u2014")   # raised, before the fix
    s.flush()
    got = s.buffer.getvalue().decode("cp1252")
except UnicodeEncodeError as exc:
    # Reported as a named failure rather than allowed to escape: an uncaught traceback is a red
    # suite too, but it stops every later arm and says nothing about what was being asserted.
    got = ""
    case("emit does not raise on a narrow codepage", "raised %s" % exc, "no exception")
case("a report line survives an encoding the body does not fit", got.startswith("line 1 differs:"),
     True)
case("and the unencodable characters are substituted, not dropped", "?" in got, True)

# ...but CONTENT is byte-exact, because `get > file` must round-trip. A substituted character here
# would silently corrupt the body, and feeding it back to `verify` would report a mismatch the PR
# does not have.
s = cp1252_stream()
try:
    pr_body.write_body_out(s, "a body with \u2260 and \u2014")
    written = s.buffer.getvalue()
except UnicodeEncodeError as exc:
    written = b"<raised %s>" % str(exc).encode("ascii", "replace")
case("`get` writes UTF-8 bytes, bypassing the console codepage",
     written, "a body with \u2260 and \u2014\n".encode("utf-8"))

print("\n-- the endpoint is built from the CANONICAL repo name, not from the git remote")
# `gh api` expands {owner}/{repo} from the remote, and a RENAMED repository answers there by
# redirect: GitHub returns HTTP 307 to a PATCH and `gh api` does not follow it, so the write does
# nothing. This tool's first live `set` hit exactly that -- remote `ps5ys`, canonical `prosper` --
# and only the read-back noticed. The FakeGh above asserts no call carries the placeholders, so
# reverting resolve_repo() reddens every arm rather than only this one.
fake = FakeGh(body="stale", repo_name="someone/renamed")
drive(fake, ["set", "42", "--body", BODY])
case("the repo was resolved before any endpoint was built",
     [c for c, _ in fake.calls][0][:3], ["gh", "repo", "view"])
case("and the resolved name is what the endpoint uses",
     [c[2] for c, _ in fake.calls if len(c) > 2 and c[2].startswith("repos/")][0],
     "repos/someone/renamed/pulls/42")

# An unresolvable repository is COULD NOT VERIFY, never a mismatch: without a name there is no
# endpoint to ask, so the tool knows nothing about the live body.
rc, out, err = drive(FakeGh(body=BODY, repo_rc=1), ["verify", "42", "--body", BODY])
case("an unresolvable repo exits 2", rc, 2)
case("and says the repository could not be resolved", "could not resolve the repository" in err,
     True)

print()
if FAILURES:
    print("FAILED (%d):" % len(FAILURES))
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("all pr_body checks passed")
