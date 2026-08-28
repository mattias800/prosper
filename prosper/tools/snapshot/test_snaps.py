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
