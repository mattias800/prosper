# `tools/screenshot` — headless capture, and the evidence contract around it

This is how prosper produces **progression evidence** without a human at the keyboard: boot a dump,
sample the present layer on a frame or wall-clock cadence, write PNGs, and answer a set of
assertions about what was actually captured. `tools/snapshot` drives it; a bring-up lane runs it by
hand; the trackers and `COMPATIBILITY.md` quote its output.

The folder's real subject is not "write a PNG" — it is **what a captured frame is allowed to be
claimed as**. Almost every non-obvious line here exists because some earlier claim turned out to be
weaker than it read:

- **A PNG is not progress.** `--min-distinct-frames`, `--min-pixel-distinct-frames` and the two
  staleness bounds separate "120 files were written" from "120 advancing frames". A run that samples
  one retained frame 120 times writes 120 identical files and looks identical to a healthy one in a
  directory listing.
- **A published frame is not a composited one.** `CaptureSource::GuestScanout` exists so
  `--require-composited-frame` cannot be satisfied by the exact frame it exists to reject (#2026,
  reverted as #2044).
- **A framerate is not a present rate.** The renderer re-publishes its retained frame when a submit
  produces no present source, so publications-per-second reads full speed for a frozen title. Both
  rates are reported and the distinct one comes first — `src/gpu/present/present_frame_rate.hpp`.
- **A short run is not a failed one, and must not be silently either.** `stop_reason` and the guest's
  own terminal state travel in the summary so a truncated artifact set says why it is short.

## What lives here

| File | Responsibility |
| --- | --- |
| `screenshot.cpp` | The tool: argument parsing, the guest thread, the sampling loop, the assertions, the summary. Owns Vulkan and the dump; everything it can push out to a pure module, it has. |
| `capture_manifest.{hpp,cpp}` | The JSONL manifest and the pure classification behind it — sample/run/summary records, the capture tracker, the run verdict, the early-stop predicate. No Vulkan, no dump: `tests/test_capture_manifest.cpp` exercises all of it. |
| `overlay_text.{hpp,cpp}` | The 5x7 bitmap font and the RGBA text burner behind `--fps-overlay`. Pure; `tests/test_overlay_text.cpp` asserts it on pixels. |
| `README.md` | The user-facing contract: every option, what each assertion means, and what the manifest fields are. |

## The boundary against its siblings

`tools/snapshot` decides *whether a capture is a regression*; this folder decides *what was
captured*. `tools/gpu_replay` answers *why a frame looks wrong*, offline, from a `.prgbundle`.
Nothing here should grow a likeness oracle or a golden image — those belong to `tools/snapshot`,
which owns the reviewed baselines.

## Two rules that are easy to break here

1. **Metrics come from the pristine frame.** The CRC32, colour count, perceptual hashes and 16x9
   luminance signature must be computed before any annotation exists; `--fps-overlay` draws into a
   copy that only the PNG encoder sees. Moving the overlay above those lines would silently change
   what every content assertion in the tool measures — and snapshot guards compare exactly those.
2. **A maximum over samples is not a floor.** Every assertion here is a floor except
   `--max-stale-seconds` and `--max-pixel-stale-seconds`, and a shortened run can only make a floor
   harder to satisfy while making a maximum easier. That asymmetry is why the staleness bounds veto
   the early stop rather than coexisting with it (`early_stop_armed`).
