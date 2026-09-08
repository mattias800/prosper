#!/usr/bin/env python3
"""Pin skip_survey.py's patterns to the emitter's own format strings.

WHY THIS FILE EXISTS, and why the arms grep C++ rather than feed the patterns a handcrafted line.

`skip_survey.py` counts lines the renderer prints. Its worst failure is not a crash: it is a pattern
that silently stops matching, because then every title reports **zero refused shaders** and that is
indistinguishable from good news. The survey's numbers are published on #3464, so a silent zero
would be believed.

A test that builds its own input in the format the pattern expects cannot detect this -- it passes
forever while the emulator prints something else entirely. That is instrument trap 272, measured on
this repository three days ago: `flip_pacing_report.py` required `[ev] GpuFlip t=<seconds>`, the
emitter never wrote a timestamp, and the tool parsed zero flips from every real log in its existence
while its self-test passed, because the self-test composed `f"[ev] GpuFlip t={t:.3f}"` itself.

So the arms below read the PRODUCING SOURCE. They fail if the emitter's format string changes shape,
which is the event that would otherwise silently zero the survey.

    python test_skip_survey.py            # from anywhere in the checkout
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]                       # <root>/prosper/tools/shader_inspect/x.py
sys.path.insert(0, str(HERE.parent))

import skip_survey  # noqa: E402


RENDER_RUNNER = REPO / "prosper" / "tests" / "fixtures" / "render_runner.h"
APP_MAIN = REPO / "prosper" / "frontends" / "prosper-app" / "main.cpp"

failures = []


def check(name, ok, detail=""):
    if ok:
        print("ok   - %s" % name)
    else:
        failures.append(name)
        print("FAIL - %s %s" % (name, detail))


def source(path):
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    if not RENDER_RUNNER.exists():
        print("cannot find %s -- run from inside the checkout" % RENDER_RUNNER, file=sys.stderr)
        return 2

    rr = source(RENDER_RUNNER)
    app = source(APP_MAIN)

    # ---- the emitter still prints what SKIP looks for ------------------------------------------
    # Matched against the format string itself, not against a line this test wrote.
    check("the skip emitter still exists in render_runner.h",
          '"[render] skip draw=%zu fs=%016llx: fragment shader requires subgroup size %u "' in rr,
          "-- the skip line's format string changed shape; SKIP will match nothing and every "
          "title will report zero")
    check("the skip emitter still carries the why= field",
          '"why=%s)' in rr,
          "-- SKIP requires why=; without it the reason histogram silently empties")

    # ---- the ADMIT line, which is the whole of finding B1 ---------------------------------------
    check("the native-width admit emitter still exists",
          '"[render] native-width fragment vote: subgroup %u -> %u "' in rr,
          "-- an allowlisted title's wave-any shaders would become invisible with no admitted count")
    # Every field the ADMIT regex depends on, not just the prefix. The prefix arm above stayed green
    # while the emitter gained an `fs=` field and the regex -- anchored on `(why=...)` -- silently
    # matched nothing, reporting admitted=0 for a whole corpus. A pin that only checks the start of a
    # line cannot see a change at the end of it.
    check("the admit emitter still carries why= and fs= in the order the regex reads them",
          '"(why=0x%x fs=%016llx)' in rr,
          "-- the ADMIT regex reads why= then fs=; reordering or dropping either silently zeroes "
          "the admitted column")

    # ---- the fps emitter -------------------------------------------------------------------------
    check("the fps emitter still prints '<n> fps (<m> frames'",
          'fps (%llu frames' in app or 'fps (%zu frames' in app or 'fps (%u frames' in app,
          "-- FPS will match nothing and every row will read 0 frames")

    # ---- the patterns match a line built FROM the format string, not from the pattern ------------
    # Still not a substitute for the greps above: this proves the pattern parses the shape, while
    # the greps prove the shape is what the emulator emits.
    skip_line = ('[render] skip draw=13 fs=0000000000000016: fragment shader requires subgroup '
                 'size 64 (device range 32..32 required-stages=0x3fff subgroup-stages=0x3fff '
                 'ops=0x7ff required-ops=0x1 control=1 gds=0 fragment-atomics=1 why=0x2 wave-any)')
    m = skip_survey.SKIP.findall(skip_line)
    check("SKIP extracts hash, width and reason from a real skip line",
          m == [("0000000000000016", "64", "0x2", " wave-any")], repr(m))

    # A reason set with no names, which the emitter can produce, must not break the pattern.
    none_line = skip_line.replace("why=0x2 wave-any)", "why=0x0 none)")
    check("SKIP survives a reason set with no named bits",
          len(skip_survey.SKIP.findall(none_line)) == 1)

    # The emitter's REAL shape, fs= included. Written by hand once and left stale is how
    # the previous version of this arm passed while the regex matched nothing.
    admit_line = ("[render] native-width fragment vote: subgroup 64 -> 32 "
                  "(why=0x2 fs=00000000000000a2)")
    check("ADMIT extracts the widths and reason from a real admit line",
          skip_survey.ADMIT.findall(admit_line) == [("64", "32", "0x2")])

    # The two must not overlap: an admit line counted as a refusal would invert the finding it
    # exists to report.
    check("ADMIT does not match a skip line", not skip_survey.ADMIT.findall(skip_line))
    check("SKIP does not match an admit line", not skip_survey.SKIP.findall(admit_line))

    check("FPS extracts the frame count",
          skip_survey.FPS.findall("[app] 22.5 fps (480 frames, gpu-present)") == ["480"])

    # ---- the allowlist must not drift from the renderer's -----------------------------------------
    # It is a copy of a title id that lives in live_renderer.cpp; if that gains a title and this does
    # not, the survey reports a structural zero as a real one.
    live = source(REPO / "prosper" / "frontends" / "shared" / "live" / "live_renderer.cpp")
    # W5 removed the per-title gate, so this now asserts its ABSENCE. Anchored on the assignment
    # rather than on any `title_id ==` in the file: a bare match would redden on an unrelated
    # per-title switch, and the natural repair -- adding that title to NATIVE_VOTE_ALLOWLIST --
    # would make the survey claim wave-any is specially handled for a title where it is not.
    #
    # If a title gate is ever reintroduced, this reddens and the survey's universal caveat has to be
    # narrowed again. That is the point: the tools describe the renderer, not the other way round.
    # Collected from the whole gate REGION rather than from one syntactic form. This arm has been
    # anchored on a single `title_id ==`, then on the assignment expression, and each time the thing
    # moved: to an assignment, then to a `kNativeFragmentVoteTitles` array the assignment merely
    # references. The array version passed an "is not title-scoped" assertion while the gate was
    # fully title-scoped, because no PPSA literal appeared in the assignment any more.
    #
    # So: find the ids wherever they live between the allowlist declaration and the gate, and assert
    # the tools EQUAL the renderer. That is the property that matters -- a survey claiming a title is
    # allowlisted when it is not hides real refusals behind a reassuring note.
    start = live.find("kNativeFragmentVoteTitles")
    end = live.find("native_fragment_vote_width", start if start >= 0 else 0)
    region = live[start:end + 400] if start >= 0 else live
    ids = set(re.findall(r'"([A-Z]{4}\d{5})"', region))
    check("skip_survey's allowlist matches the renderer's, exactly",
          ids == set(skip_survey.NATIVE_VOTE_ALLOWLIST),
          "-- renderer admits %s, survey has %s" % (sorted(ids) or "no title",
                                                    sorted(skip_survey.NATIVE_VOTE_ALLOWLIST)))
    check("the renderer's allowlist is non-empty and was actually found",
          bool(ids),
          "-- found no title ids near the gate; the pin may be looking at the wrong place again")

    # N7: `admitted` is a shader count only because the emitter dedupes on shader identity
    # before printing. Nothing else pins that, so a refactor dropping the guard would turn
    # the column into a draw count with no test noticing -- the exact units confusion the
    # module docstring warns about, arriving through the other door.
    check("the admit emitter still dedupes on shader identity",
          "native_width_logged.insert(shader_key).second" in rr,
          "-- admitted would become a DRAW count while still labelled shaders")

    print("\n%d checks failed" % len(failures) if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
