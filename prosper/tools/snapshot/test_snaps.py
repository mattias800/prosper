#!/usr/bin/env python3
"""Self-test for snaps.py — the human-authored render snapshot store.

Runs with no game dump and no emulator: every arm builds its own BMPs, so this is CI-safe. What it
cannot cover is the replay itself (that needs a title), which is exercised by hand and recorded in
the PR.

The arms are chosen around the ways this tool could fail SILENTLY, because a snapshot system that
reports a confident wrong answer is worse than one that errors:

  - a signature that cannot tell two different scenes apart would pass every regression
  - a signature that calls one scene two different things would fail every run
  - an unanchored snap imported as flip 0 would replay to the boot logo and compare it to a menu
  - accept() writing the signature but not the reference image (or vice versa) would leave the
    store and the images disagreeing, and the next failure would show a misleading "expected"
"""

import json
import shutil
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import snaps  # noqa: E402

failures = 0


def check(ok, what):
    global failures
    print(f"{'ok' if ok else 'FAIL'}: {what}")
    if not ok:
        failures += 1


def write_bmp(path, width, height, pixel):
    """Write a 24-bit bottom-up BMP whose pixel at (x, y) is pixel(x, y) -> (r, g, b)."""
    stride = ((width * 3 + 3) // 4) * 4
    body = bytearray()
    for y in range(height - 1, -1, -1):
        row = bytearray()
        for x in range(width):
            r, g, b = pixel(x, y)
            row += bytes((b, g, r))
        row += bytes(stride - len(row))
        body += row
    header = bytearray(54)
    header[0:2] = b"BM"
    struct.pack_into("<I", header, 2, 54 + len(body))
    struct.pack_into("<I", header, 10, 54)
    struct.pack_into("<I", header, 14, 40)
    struct.pack_into("<i", header, 18, width)
    struct.pack_into("<i", header, 22, height)
    struct.pack_into("<H", header, 26, 1)
    struct.pack_into("<H", header, 28, 24)
    with open(path, "wb") as handle:
        handle.write(bytes(header) + bytes(body))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        black = os.path.join(tmp, "black.bmp")
        gradient = os.path.join(tmp, "gradient.bmp")
        gradient2 = os.path.join(tmp, "gradient2.bmp")
        flat_grey = os.path.join(tmp, "grey.bmp")
        write_bmp(black, 64, 36, lambda x, y: (0, 0, 0))
        write_bmp(gradient, 64, 36, lambda x, y: (x * 4 % 256, y * 7 % 256, (x + y) * 3 % 256))
        # Same scene, slightly perturbed -- the "a correct fix moved some pixels" case.
        write_bmp(gradient2, 64, 36,
                  lambda x, y: (min(255, x * 4 % 256 + 2), y * 7 % 256, (x + y) * 3 % 256))
        write_bmp(flat_grey, 64, 36, lambda x, y: (128, 128, 128))

        # ---- 1. The signature reads a BMP and reports honest content metrics ------------------
        sig_black = snaps.signature_of(black)
        sig_grad = snaps.signature_of(gradient)
        check(sig_black["width"] == 64 and sig_black["height"] == 36, "signature reports dimensions")
        check(len(sig_black["luma16x9"]) == 288, "the luma thumbnail is 16x9 bytes as hex")
        check(sig_black["nonblack_ratio"] == 0.0 and sig_black["distinct_colors"] == 1,
              "an all-black frame reports 0 non-black and one colour")
        check(sig_grad["nonblack_ratio"] > 0.9 and sig_grad["distinct_colors"] > 100,
              "a rich frame reports high non-black and many colours")

        # ---- 2. SSIM separates different scenes and accepts small perturbation ----------------
        # Both directions matter. A signature that cannot tell black from a gradient would pass
        # every regression; one that rejects a 2/255 nudge would fail every honest improvement.
        luma = lambda s: snaps.decode_luma(s["luma16x9"])
        different = snaps.structural_similarity(luma(sig_black), luma(sig_grad))
        same = snaps.structural_similarity(luma(sig_grad), luma(snaps.signature_of(gradient2)))
        identical = snaps.structural_similarity(luma(sig_grad), luma(sig_grad))
        check(identical > 0.999, "a frame is identical to itself")
        check(different < snaps.DEFAULT_MIN_SSIM,
              "a collapse to black scores BELOW the failure threshold")
        check(same >= snaps.DEFAULT_MIN_SSIM,
              "a small perturbation of the same scene stays ABOVE the threshold")

        # ---- 3. Two flat frames of different colours are not confused ------------------------
        # SSIM alone scores these highly, which is exactly why the content metrics sit beside it.
        sig_grey = snaps.signature_of(flat_grey)
        check(sig_grey["distinct_colors"] == 1 and sig_black["distinct_colors"] == 1,
              "two different flat frames both report one colour...")
        check(sig_grey["nonblack_ratio"] == 1.0 and sig_black["nonblack_ratio"] == 0.0,
              "...and are separated by the non-black ratio, which SSIM alone would not do")

        # ---- 4. import: anchors are kept, unanchored snaps are REJECTED ----------------------
        capture = os.path.join(tmp, "capture")
        os.makedirs(capture)
        write_bmp(os.path.join(capture, "a.bmp"), 64, 36,
                  lambda x, y: (x * 4 % 256, y * 7 % 256, (x + y) * 3 % 256))
        write_bmp(os.path.join(capture, "b.bmp"), 64, 36, lambda x, y: (0, 0, 0))
        write_bmp(os.path.join(capture, "c.bmp"), 64, 36, lambda x, y: (10, 10, 10))
        with open(os.path.join(capture, "route.pad"), "w", encoding="utf-8") as handle:
            handle.write("# recorded route\n")
        with open(os.path.join(capture, "snaps.jsonl"), "w", encoding="utf-8") as handle:
            for record in [
                {"index": 0, "verdict": "correct", "pad_flip": 4000, "guest_present": 0,
                 "width": 64, "height": 36, "file": "a.bmp", "title_id": "PPSATEST"},
                {"index": 1, "verdict": "incorrect", "pad_flip": 900, "guest_present": 0,
                 "width": 64, "height": 36, "file": "b.bmp", "title_id": "PPSATEST"},
                {"index": 2, "verdict": "correct", "pad_flip": -1, "guest_present": 0,
                 "width": 64, "height": 36, "file": "c.bmp", "title_id": "PPSATEST"},
            ]:
                handle.write(json.dumps(record) + "\n")

        store = os.path.join(tmp, "store")
        refs = os.path.join(tmp, "refs")
        os.makedirs(store)
        snaps.SNAP_STORE = store
        snaps.REF_HOME = refs
        snaps.REPO_ROOT = tmp

        class Args:
            capture_dir = capture
            name = "unit"
            route = None
            dump = None
            timeout = 60
            min_ssim = snaps.DEFAULT_MIN_SSIM
            savedata = "fresh"
            det_fps = snaps.DEFAULT_DET_FPS
            det_clock = "on"
            flip_window = snaps.DEFAULT_FLIP_WINDOW
            window_samples = snaps.DEFAULT_WINDOW_SAMPLES

        snaps.cmd_import(Args())
        entry = snaps.load_entry("unit")
        check(len(entry["snaps"]) == 2,
              "the unanchored snap is rejected, the two anchored ones are imported")
        check(all(s["pad_flip"] >= 0 for s in entry["snaps"]),
              "no imported snap carries the unanchored sentinel")
        check([s["pad_flip"] for s in entry["snaps"]] == [900, 4000],
              "snaps are stored in replay order, not capture order")
        verdicts = {s["index"]: s["verdict"] for s in entry["snaps"]}
        check(verdicts == {0: "correct", 1: "incorrect"},
              "both verdicts survive the round trip")
        check(os.path.exists(os.path.join(refs, "unit", "0000.bmp")),
              "the reference image is kept locally for later A/B review")

        # ---- 4b. import takes its parameters from the SESSION, not from its own defaults -------
        # --det-fps is a separate argument on `author` and on `import`, each defaulting to 60. Author
        # at 30, import without the flag, and the entry recorded 60 -- so both halves then paced at
        # 60 against a session a person actually played at 30. Nothing failed loudly; the anchors
        # just moved. cmd_author now writes session.json and cmd_import prefers it.
        cap2 = os.path.join(tmp, "cap2")
        os.makedirs(cap2)
        shutil.copyfile(os.path.join(capture, "route.pad"), os.path.join(cap2, "route.pad"))
        write_bmp(os.path.join(cap2, "snap_0000_correct_f900.bmp"), 64, 36,
                  lambda x, y: (x * 4 % 256, y * 7 % 256, 90))
        with open(os.path.join(cap2, "snaps.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "index": 0, "verdict": "correct", "mode": "anchor", "pad_flip": 900,
                "guest_present": 900, "width": 64, "height": 36,
                "file": "snap_0000_correct_f900.bmp", "title_id": "PPSATEST"}) + "\n")
        with open(os.path.join(cap2, "session.json"), "w", encoding="utf-8") as handle:
            json.dump({"det_fps": 30, "det_clock": "off", "savedata": "fresh",
                       "name": "unit2"}, handle)

        class Args2(Args):
            capture_dir = cap2
            name = "unit2"
            det_fps = snaps.DEFAULT_DET_FPS      # the CLI default, NOT what was authored
            det_clock = "on"                     # ditto

        snaps.cmd_import(Args2())
        authored = snaps.load_entry("unit2")
        check(authored.get("det_fps") == 30,
              "import records the rate the session was AUTHORED at, not its own default")
        check(authored.get("det_clock") == "off",
              "...and the clock stance the session was authored under")
        # The whole point is that both halves then agree.
        env2 = snaps.replay_env(authored, tmp, base={})
        check(env2.get("PROSPER_FLIP_PACE_FPS") == "30",
              "so the check paces at the rate the person actually played at")

        # ---- 5. accept: store AND reference image move together -------------------------------
        # If only one moved, the next failure would show an "expected" image that does not
        # correspond to the signature that failed -- misleading exactly when trust matters most.
        review = os.path.join(tmp, "failures")
        os.makedirs(review)
        snaps.HERE = tmp
        actual = os.path.join(review, "unit-0000-actual.bmp")
        write_bmp(actual, 64, 36, lambda x, y: (128, 128, 128))

        class AcceptArgs:
            name = "unit"
            indices = [0]
            verdict = None

        snaps.cmd_accept(AcceptArgs())
        updated = snaps.load_entry("unit")
        promoted = next(s for s in updated["snaps"] if s["index"] == 0)
        check(promoted["distinct_colors"] == 1 and promoted["nonblack_ratio"] == 1.0,
              "accept replaces the stored signature with the actual's")
        ref_sig = snaps.signature_of(os.path.join(refs, "unit", "0000.bmp"))
        check(ref_sig["luma16x9"] == promoted["luma16x9"],
              "accept moves the reference IMAGE too, so store and image cannot disagree")

        # ---- 6a1. The guest clock is pinned, in both halves ---------------------------------
        # Without this the anchors are frame-rate dependent. Measured: authoring windowed at 60.4 fps
        # against a headless check at 77.1 fps made a time-based intro logo burn ~28% more flips, and
        # an anchor drifted clean out of its window. The game was not misbehaving -- it uses deltaTime
        # correctly; prosper was answering "what time is it" from a clock tied to render speed.
        clk = {}
        snaps.apply_deterministic_clock(clk, 60, enabled=True)
        check(clk.get("PROSPER_DET_CLOCK") == "1", "the deterministic clock is enabled")
        check(clk.get("PROSPER_DET_FPS") == "60", "the clock rate is passed through")
        clk2 = {}
        snaps.apply_deterministic_clock(clk2, 30, enabled=True)
        check(clk2.get("PROSPER_DET_FPS") == "30", "a non-default rate is honoured")

        # The clock is NOT universally safe and must be disablable per title. GRIS freezes on its
        # opening FMV with it on -- 42,000 frames in 150 s off against 1,680 on, a 25x collapse.
        off = {}
        snaps.apply_deterministic_clock(off, 60, enabled=False)
        implicit = {}
        snaps.apply_deterministic_clock(implicit, 60)
        check("PROSPER_DET_CLOCK" not in implicit,
              "the helper's own default matches the tool's stance (off), so a new caller cannot "
              "silently pin the clock")
        check("PROSPER_DET_CLOCK" not in off, "the clock can be turned off for a title")
        # And an inherited value must be CLEARED, not merely left unset: an authoring shell that
        # exported it would otherwise leak it into a run whose entry says the clock is off, making
        # the check silently disagree with how the session was authored.
        leaked = {"PROSPER_DET_CLOCK": "1", "PROSPER_DET_FPS": "60"}
        snaps.apply_deterministic_clock(leaked, 60, enabled=False)
        check("PROSPER_DET_CLOCK" not in leaked and "PROSPER_DET_FPS" not in leaked,
              "an inherited clock is CLEARED when the title disables it")
        check(entry.get("det_fps") == snaps.DEFAULT_DET_FPS,
              "the rate is RECORDED in the entry, so the check reproduces how it was authored")

        # ---- 6a1b. The det_clock READER fallback stays "on" -------------------------------
        # The author/import CLI default is "off", but a stored set from before the key existed was
        # necessarily authored with the clock pinned -- cmd_author applied it unconditionally then.
        # Following the new default when READING would replay such a set under conditions it was
        # never recorded under. Measured when this was wrong: all four alexkidd snaps failed, with
        # colour counts collapsing 14548 -> 108, because the flip-anchored ROUTE diverged rather than
        # the anchors merely drifting.
        #
        # These arms CALL snaps.clock_enabled rather than restating `.get(..., "on")`. An earlier
        # version of this block asserted `legacy.get("det_clock", "on") == "on"` -- which is
        # "on" == "on", passes with the bug reinstated, and printed a message claiming otherwise.
        legacy = {"det_fps": 60}                       # no det_clock key, as pre-4bf1c80c sets have
        check(snaps.clock_enabled(legacy) is True,
              "a set with no det_clock key reads as CLOCK ON, not as the new author default")
        check(snaps.clock_enabled({"det_clock": "off"}) is False,
              "...while a set that explicitly says off is still honoured")
        check(snaps.clock_enabled({"det_clock": "on"}) is True, "...and one that says on is too")

        # ---- 6a1b-ii. BOTH halves pace, at the SAME rate -----------------------------------
        # This is the mechanism that lets the clock stay off, and until now nothing tested it at all.
        # Routes are flip-anchored, so flip N is a fixed moment in the game only if flips happen at a
        # fixed RATE; if one side paces and the other does not, presses land at different guest times
        # and the route diverges. Assert against the real environment builder, not a restatement.
        paced = snaps.replay_env({"det_fps": 60, "route": "r.pad", "snaps": [],
                                  "savedata": "none"}, tmp, base={})
        check(paced.get("PROSPER_FLIP_PACE_FPS") == "60",
              "the CHECK paces its flips -- without this a flip-anchored route diverges")
        check(paced.get("PROSPER_DET_CLOCK") == "1",
              "...and a legacy entry with no det_clock key still replays with the clock pinned")

        modern = snaps.replay_env({"det_fps": 60, "det_clock": "off", "route": "r.pad",
                                   "snaps": [], "savedata": "none"}, tmp, base={})
        check(modern.get("PROSPER_FLIP_PACE_FPS") == "60",
              "a clock-off entry paces too -- pacing is what REPLACED the clock, not a companion "
              "to it")
        check("PROSPER_DET_CLOCK" not in modern, "...while its clock stays off")

        # The rate both halves use has to come from one place, or a non-default authoring rate
        # silently replays at 60.
        slow = snaps.replay_env({"det_fps": 30, "det_clock": "off", "route": "r.pad",
                                 "snaps": [], "savedata": "none"}, tmp, base={})
        check(slow.get("PROSPER_FLIP_PACE_FPS") == "30",
              "the pace rate is read from the entry, not hardcoded")
        check(snaps.pace_fps({}) == snaps.DEFAULT_DET_FPS,
              "an entry with no det_fps falls back to the documented default")

        # The AUTHOR half. Everything above this asserts the CHECK. An earlier revision headed this
        # block "BOTH halves pace" while testing one of them, and deleting the author's pacing line
        # left the whole suite green -- every future session would then run unpaced against a check
        # paced at 60, which diverges rather than drifts.
        class AuthorArgs:
            name = "unit-author"
            det_fps = 30
            det_clock = "off"
            savedata = "none"

        aenv = snaps.author_env(AuthorArgs(), tmp, base={})
        check(aenv.get("PROSPER_FLIP_PACE_FPS") == "30",
              "the AUTHORING session paces its flips too")
        check("PROSPER_DET_CLOCK" not in aenv, "...with the clock off when the session says off")
        check(aenv.get("PROSPER_PAD_RECORD", "").endswith("route.pad"),
              "...and records the route the check will replay")

        # The two halves must agree by CONSTRUCTION, not by two literals that happen to match.
        # Same det_fps in, same pace out, whichever side computes it.
        for rate in (30, 60, 120):
            class R:
                name = "x"; det_fps = rate; det_clock = "off"; savedata = "none"
            a = snaps.author_env(R(), tmp, base={}).get("PROSPER_FLIP_PACE_FPS")
            c = snaps.replay_env({"det_fps": rate, "det_clock": "off", "route": "r.pad",
                                  "snaps": [], "savedata": "none"}, tmp,
                                 base={}).get("PROSPER_FLIP_PACE_FPS")
            check(a == c == str(rate),
                  f"author and check derive the SAME pace at det_fps={rate} ({a} vs {c})")

        # ---- 6a1b-iii. What author WRITES is what import READS -------------------------------
        # These were two hand-written literals. Renaming the keys cmd_author records left the suite
        # fully green, because the import test mirrored the format by hand rather than obtaining it
        # from the producer. Bind them to one list and assert the round trip.
        class SessArgs:
            name = "roundtrip"
            det_fps = 30
            det_clock = "off"
            savedata = "fresh"

        record = snaps.session_record(SessArgs())
        check(set(snaps.SESSION_KEYS) <= set(record),
              "the session record carries every key import looks up")
        check(record.get("det_fps") == 30 and record.get("det_clock") == "off"
              and record.get("savedata") == "fresh",
              "...with the values the session was authored under")
        # And import must consume exactly these -- not a superset, not a stale alias.
        import inspect
        import_src = inspect.getsource(snaps.cmd_import)
        check("SESSION_KEYS" in import_src,
              "import reads the shared key list rather than its own copy of the names")

        # ---- 6a1b-iii-b. cmd_author -> cmd_import, for real ----------------------------------
        # The strongest form of the producer/consumer check: actually RUN cmd_author (with the
        # emulator stubbed out) and feed its output directory to cmd_import. Every earlier version
        # of this test mirrored the session format by hand, which is exactly why renaming the keys
        # cmd_author writes stayed green through two review rounds.
        e2e = os.path.join(tmp, "e2e")
        fake_dump = os.path.join(tmp, "PPSATEST-app0")
        os.makedirs(fake_dump)
        fake_app = os.path.join(tmp, "fake-prosper-app")
        with open(fake_app, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\nexit 0\n")
        os.chmod(fake_app, 0o755)

        class AuthorE2E:
            name = "e2e"
            dump = fake_dump
            out = e2e
            append = False
            det_fps = 30            # deliberately NOT the default
            det_clock = "off"
            savedata = "none"

        saved_app, saved_run = snaps.APP, snaps.subprocess.run
        # The session's own snaps.jsonl and route.pad are what a real run would leave behind.
        def fake_run(cmd, env=None, check=False):
            os.makedirs(env["PROSPER_SNAP_DIR"], exist_ok=True)
            write_bmp(os.path.join(env["PROSPER_SNAP_DIR"], "snap_0000_correct_f900.bmp"),
                      64, 36, lambda x, y: (x * 3 % 256, y * 5 % 256, 40))
            with open(os.path.join(env["PROSPER_SNAP_DIR"], "snaps.jsonl"), "w",
                      encoding="utf-8") as h:
                h.write(json.dumps({
                    "index": 0, "verdict": "correct", "mode": "anchor", "pad_flip": 900,
                    "guest_present": 900, "width": 64, "height": 36,
                    "file": "snap_0000_correct_f900.bmp", "title_id": "PPSATEST"}) + "\n")
            open(env["PROSPER_PAD_RECORD"], "w", encoding="utf-8").write("f900:cross\n")
            return None
        try:
            snaps.APP = fake_app
            snaps.subprocess.run = fake_run
            snaps.cmd_author(AuthorE2E())
        finally:
            snaps.APP, snaps.subprocess.run = saved_app, saved_run

        check(os.path.exists(os.path.join(e2e, "session.json")),
              "an authoring session records itself on disk")

        # The reviewer's exact reproduction, end to end: append to this directory WITHOUT repeating
        # --det-fps and confirm the emulator is actually launched at the session's rate, not the
        # CLI default. Asserting on `saw` -- the env handed to the process -- is the whole point:
        # the previous version recorded 30 and ran at 60, and only the env can tell them apart.
        saw = {}

        class AppendE2E(AuthorE2E):
            append = True
            det_fps = snaps.DEFAULT_DET_FPS       # 60 -- the default, NOT what was authored

        def capture_run(cmd, env=None, check_=False, **kw):
            saw.update(env)
            return fake_run(cmd, env=env)
        saved_app, saved_run = snaps.APP, snaps.subprocess.run
        try:
            snaps.APP = fake_app
            snaps.subprocess.run = capture_run
            snaps.cmd_author(AppendE2E())
        finally:
            snaps.APP, snaps.subprocess.run = saved_app, saved_run

        check(saw.get("PROSPER_FLIP_PACE_FPS") == "30",
              "an appending run is PACED at the session's own rate, not the command-line default")
        with open(os.path.join(e2e, "session.json"), "r", encoding="utf-8") as handle:
            after = json.load(handle)
        check(after.get("det_fps") == 30, "...and still records that rate")
        check(saw.get("PROSPER_FLIP_PACE_FPS") == str(after.get("det_fps")),
              "...so the run and its own record agree, which is the whole invariant")

        class ImportE2E(Args):
            capture_dir = e2e
            name = "e2e"
            det_fps = snaps.DEFAULT_DET_FPS      # the CLI default, NOT what was authored
            det_clock = "on"

        snaps.cmd_import(ImportE2E())
        round_tripped = snaps.load_entry("e2e")
        check(round_tripped.get("det_fps") == 30,
              "import reads the rate cmd_author actually wrote -- producer to consumer, no "
              "hand-written mirror in between")
        check(round_tripped.get("det_clock") == "off",
              "...and the clock stance, through the same path")
        check(snaps.replay_env(round_tripped, tmp, base={}).get("PROSPER_FLIP_PACE_FPS") == "30",
              "...so the check ends up paced at the rate the session was authored at")

        # ---- 6a1b-iv. A check does not write its frames into the shared tmpfs ----------------
        # /tmp here is RAM-backed with a per-user quota shared by every concurrent agent, and
        # exhausting it takes the machine's RAM rather than merely failing the write. Scan mode
        # raises the captures per snap from 7 to 34, so this got ~5x heavier than it used to be.
        out = snaps.default_check_out_dir("somegame")
        check(not out.startswith(tempfile.gettempdir() + os.sep) and out != tempfile.gettempdir(),
              "a check's scratch does not land in the shared RAM-backed tmpfs")
        check(out.startswith(os.path.expanduser("~")), "...it lands on real disk under $HOME")
        check("somegame" in out, "...in a per-set directory, so two sets cannot collide")

        # ---- 6a1b-v. --append: the run and its own record must agree -------------------------
        # The first version of this fix resolved the stored session AFTER building the run
        # environment, so `--append --det-fps 60` on a directory authored at 30 ran the emulator
        # paced at 60 while recording 30. Every snap from that run would then be anchored on flips
        # taken at 60 and replayed at 30 -- the divergence this whole mechanism exists to prevent,
        # moved from run 1 to run 2 rather than removed.
        class Fresh:
            name = "s"; det_fps = 60; det_clock = "off"; savedata = "fresh"; append = False

        rec, adopted, conflicts = snaps.reconcile_session(Fresh(), {}, was_passed=lambda k: False)
        check(rec is not None and rec.get("det_fps") == 60 and not adopted and not conflicts,
              "a fresh session records its own parameters")

        # Not explicitly asked for -> adopt the stored value, AND mutate args so the run matches.
        class Defaulted:
            name = "s"; det_fps = 60; det_clock = "off"; savedata = "fresh"; append = True

        d = Defaulted()
        rec2, adopted2, conflicts2 = snaps.reconcile_session(
            d, {"det_fps": 30, "det_clock": "off", "savedata": "fresh"},
            was_passed=lambda k: False)
        check(adopted2 == ["det_fps"] and not conflicts2,
              "appending without the flag adopts the session's own rate")
        check(d.det_fps == 30,
              "...and ARGS is updated, so author_env paces the run to match what is recorded")
        check(rec2.get("det_fps") == 30, "...so the run and its record cannot disagree")

        # Explicitly asked for and contradicting -> a conflict the caller must resolve.
        class Explicit:
            name = "s"; det_fps = 60; det_clock = "off"; savedata = "fresh"; append = True

        e = Explicit()
        rec3, adopted3, conflicts3 = snaps.reconcile_session(
            e, {"det_fps": 30, "det_clock": "off", "savedata": "fresh"},
            was_passed=lambda k: k == "det_fps")
        check(conflicts3 == ["det_fps"] and not adopted3,
              "an explicit flag contradicting the session is a CONFLICT, not a silent override")
        check(e.det_fps == 60, "...and args is left alone, so the caller sees what they asked for")

        # Same settings -> silent, and the record still gets rewritten so a key missing from an
        # older session file can be completed.
        class Same:
            name = "s"; det_fps = 30; det_clock = "off"; savedata = "fresh"; append = True

        rec4, adopted4, conflicts4 = snaps.reconcile_session(
            Same(), {"det_fps": 30}, was_passed=lambda k: False)
        check(not adopted4 and not conflicts4, "appending at the same settings is silent")
        check(rec4 is not None and rec4.get("savedata") == "fresh",
              "...and a key missing from an older session file is completed rather than left out")

        # ---- 6a1b-vi. _flag_was_passed sees argparse ABBREVIATIONS -----------------------------
        # argparse accepts any unambiguous prefix, so `--det-f 30` sets det_fps. An exact-match
        # check missed it, and the stored session would then silently override a flag the person
        # did type. This is the arm that was missing when the fix landed.
        saved_argv = sys.argv
        try:
            for argv, key, want in (
                    (["--det-fps", "30"], "det_fps", True),
                    (["--det-f", "30"], "det_fps", True),      # abbreviation
                    (["--det-fps=30"], "det_fps", True),
                    (["--det-f", "30"], "det_clock", False),   # must not bleed across flags
                    (["--savedata", "none"], "det_fps", False),
                    (["--"], "det_fps", False),                # end-of-options, not a flag
                    ([], "det_fps", False)):
                sys.argv = ["snaps.py", "import", "d"] + argv
                got = snaps._flag_was_passed(key)
                check(got == want,
                      f"_flag_was_passed({key}) is {want} for {argv or 'no flags'}")
        finally:
            sys.argv = saved_argv

        # An inherited pace from the authoring shell must not survive into a run that specifies its
        # own -- same leak the clock arm above guards.
        leaked_pace = snaps.replay_env({"det_fps": 30, "route": "r.pad", "snaps": [],
                                        "savedata": "none"}, tmp,
                                       base={"PROSPER_FLIP_PACE_FPS": "144"})
        check(leaked_pace.get("PROSPER_FLIP_PACE_FPS") == "30",
              "an inherited pace rate is OVERRIDDEN by the entry's own")

        # ---- 6a1c. The scan window never exceeds its configured span ------------------------
        # `step` floors at 1, so a sample count larger than the span used to walk past `forward`.
        tight = snaps.scan_offsets({"scan_forward": 10, "scan_back": 0, "scan_samples": 33})
        check(max(tight) <= 10, "a sample count larger than the span does not overshoot it")
        check(min(tight) >= 0, "...and does not reach behind a zero scan_back")
        zero = snaps.scan_offsets({"scan_forward": 0, "scan_back": 0, "scan_samples": 33})
        check(zero == [0], "a zero span collapses to the anchor alone")
        wide = snaps.scan_offsets({})
        check(max(wide) <= snaps.DEFAULT_SCAN_FORWARD and min(wide) >= -snaps.DEFAULT_SCAN_BACK,
              "the DEFAULTS stay inside their own span (nothing else exercises them)")
        # A negative bound turns the clamp from a limit into a filter that drops EVERY offset,
        # anchor included, and the snap then reports NOT REACHED with nothing pointing at the store.
        # Only reachable from a hand-edited set (neither key has a flag), which is why it needs a
        # test rather than validation at the CLI.
        for broken in ({"scan_forward": 100, "scan_back": -200},
                       {"scan_forward": -10, "scan_back": -10},
                       {"scan_forward": -5}, {"scan_back": -5}):
            got = snaps.scan_offsets(broken)
            check(len(got) > 0 and 0 in got,
                  f"a negative span still yields the anchor rather than an empty sweep: {broken}")

        # ---- 6a2. Both save roots are isolated, not just one -------------------------------
        # A title with a save offers "Continue" ABOVE "New Game", so the same D-pad inputs select a
        # different item -- the route does not merely mismatch, it diverges. prosper has two
        # independent save roots and redirecting only one leaves file-mount titles reading the
        # developer's real saves, which also means authoring silently writes into them.
        sd_root = os.path.join(tmp, "sdroot")
        sd_env = {}
        snaps.apply_fresh_savedata(sd_env, sd_root)
        check("PROSPER_SAVEDATA_DIR" in sd_env and "PROSPER_SAVE0" in sd_env,
              "BOTH save roots are redirected, not just SaveDataMemory")
        check(sd_env["PROSPER_SAVEDATA_DIR"] != sd_env["PROSPER_SAVE0"],
              "the two roots are distinct directories")
        for key in ("PROSPER_SAVEDATA_DIR", "PROSPER_SAVE0"):
            check(os.path.isdir(sd_env[key]), f"{key} exists so the guest can write to it")
            check(os.listdir(sd_env[key]) == [], f"{key} starts EMPTY, which is what 'fresh' means")
            check(os.path.abspath(sd_env[key]).startswith(os.path.abspath(sd_root)),
                  f"{key} is inside the run directory, never the developer's real save path")

        # ---- 6b. The anchor window: samples either side, always including the anchor ----------
        # Required, not defensive. A flip anchor is stable against rendering changes but drifts when
        # the guest's own progress speeds up or slows down -- measured during development, an anchor
        # that landed on the title screen on an idle machine landed on a still-black loading frame
        # while builds were running, and reported a confident FAIL. Comparing only the exact flip
        # makes the suite a measure of system load.
        offsets = snaps.window_offsets(entry)
        check(0 in offsets, "the exact anchor is always sampled")
        check(min(offsets) < 0 < max(offsets), "the window looks both earlier and later")
        check(len(offsets) >= 3, "the window takes several samples")
        narrow = snaps.window_offsets({"flip_window": 0, "window_samples": 7})
        check(narrow == [0], "a zero window degrades to exact-anchor matching")

        candidates = snaps.candidate_flips(entry)
        check(all(f >= 0 for f in candidates),
              "no candidate flip is negative -- a flip before the origin does not exist")
        for snap_entry in entry["snaps"]:
            check(snap_entry["pad_flip"] in candidates,
                  f"the anchor {snap_entry['pad_flip']} is itself always captured")

        # ---- 6c. best_match picks the closest frame in the window, not the exact anchor -------
        # The whole point: the authored frame is somewhere in the span, and which sample lands on it
        # depends on how fast the machine happened to be.
        win_dir = os.path.join(tmp, "win")
        os.makedirs(win_dir, exist_ok=True)
        target = next(s for s in entry["snaps"] if s["index"] == 0)
        anchor = target["pad_flip"]
        off = [o for o in snaps.window_offsets(entry) if o != 0][0]
        # The exact anchor holds a black frame; a windowed sample holds the frame that was approved.
        write_bmp(os.path.join(win_dir, "at.bmp"), 64, 36, lambda x, y: (0, 0, 0))
        write_bmp(os.path.join(win_dir, "off.bmp"), 64, 36,
                  lambda x, y: (x * 4 % 256, y * 7 % 256, (x + y) * 3 % 256))
        windowed = {
            anchor: {"target_flip": anchor, "actual_flip": anchor, "file": "at.bmp"},
            anchor + off: {"target_flip": anchor + off, "actual_flip": anchor + off,
                           "file": "off.bmp"},
        }
        # target currently holds the grey signature from arm 5; restore the gradient it was authored
        # with so the window has something to find.
        target.update(snaps.signature_of(os.path.join(capture, "a.bmp")))
        found = snaps.best_match(target, entry, windowed, win_dir)
        check(found is not None, "best_match finds a frame in the window")
        score, record, _, matched_offset = found
        check(matched_offset == off and record["file"] == "off.bmp",
              "the windowed sample wins over a black frame sitting exactly on the anchor")
        check(score >= snaps.DEFAULT_MIN_SSIM,
              "...and it scores as a pass, where exact-anchor matching would have failed")

        # ---- 6c2. Sampling is DENSE near the anchor, because that is where the match is ------
        # Measured: two identical runs through Blue Prince's intro cutscene reached the same page of
        # the same text at flip 10000 -- so drift across an FMV is small. They scored 0.465 anyway
        # because one was mid-fade, and a few flips of drift during a transition is a large
        # luminance change. Uniform spacing spends its samples far from the anchor, where the match
        # almost never is.
        geo = snaps.window_offsets({"flip_window": 900, "window_samples": 9})
        positives = [o for o in geo if o > 0]
        check(positives == sorted(positives), "offsets are ordered")
        first_gap = positives[0]
        last_gap = positives[-1] - positives[-2]
        check(last_gap > first_gap * 2,
              "spacing widens away from the anchor (dense near, sparse far)")
        check(positives[0] < 900 // (len(positives) + 1),
              "the nearest sample is much closer than uniform spacing would place it")
        check(geo == [-o for o in reversed(geo)], "the window is symmetric about the anchor")
        check(len(set(geo)) == len(geo), "no two samples collapse onto the same offset")

        # ---- 6c3. A scan snap sweeps forward instead of hugging the anchor ------------------
        # This is what replaces the deterministic clock. Drift stops being something to eliminate by
        # lying to the guest about time, and becomes something to search past.
        scan_entry = {"scan_forward": 12000, "scan_back": 900, "scan_samples": 33}
        sc = snaps.window_offsets(scan_entry, {"mode": "scan"})
        an = snaps.window_offsets(scan_entry, {"mode": "anchor"})
        check(max(sc) > max(an) * 5, "a scan reaches far beyond the anchor window")
        check(0 in sc, "a scan still samples the anchor itself")
        check(min(sc) >= -scan_entry["scan_back"],
              "a scan looks only a little way BACK -- it is forward-biased on purpose")
        check(len(sc) > len(an), "a scan takes more samples than a tight window")
        # Uniform, not geometric: a scan is used precisely when the match is NOT near the anchor, so
        # weighting toward the anchor would spend samples in the least likely place.
        pos = sorted(o for o in sc if o > 0)
        gaps = [b - a for a, b in zip(pos, pos[1:])]
        check(max(gaps) - min(gaps) <= 2, "scan sampling is uniform across the span")
        # The span is bounded so a check cannot run for an unbounded time. It is NOT bounded to stop
        # a recurring scene matching the wrong occurrence -- best_match takes the BEST score in the
        # span rather than the first, so a repeat inside the span is resolved by score, not by
        # position. (That reasoning was in an earlier draft and is retracted.)
        check(max(sc) <= scan_entry["scan_forward"] + 1, "a scan is bounded by its forward span")

        # ---- 6d. An edge match is detectable, because that is how "the window is too narrow"
        # is told apart from "the picture changed" ----------------------------------------------
        edge_entry = {"flip_window": 600, "window_samples": 7}
        edge_offsets = snaps.window_offsets(edge_entry)
        check(max(edge_offsets) == 600 and min(edge_offsets) == -600,
              "the window reaches exactly the configured span at its edges")
        check(abs(edge_offsets[0]) >= edge_entry["flip_window"],
              "an outermost sample is recognisable as an edge match")
        check(abs(edge_offsets[len(edge_offsets) // 2]) < edge_entry["flip_window"],
              "a central sample is not mistaken for one")

        # ---- 6. accept --verdict reclassifies a fixed known-bad frame -------------------------
        write_bmp(os.path.join(review, "unit-0001-actual.bmp"), 64, 36,
                  lambda x, y: (x * 4 % 256, y * 7 % 256, (x + y) * 3 % 256))

        class ReclassifyArgs:
            name = "unit"
            indices = [1]
            verdict = "correct"

        snaps.cmd_accept(ReclassifyArgs())
        reclassified = next(s for s in snaps.load_entry("unit")["snaps"] if s["index"] == 1)
        check(reclassified["verdict"] == "correct",
              "a known-bad frame that now renders correctly can be promoted to a guard")

    if failures:
        print(f"== FAIL: {failures} ==")
        return 1
    print("== PASS ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
