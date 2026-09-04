# `frontends/shared/diagnostics` — aiming and bounding the frontends' diagnostics

This folder holds the **policy and control** side of frontend diagnostics: the small, testable pieces
that decide *when* a diagnostic is armed, *which* window it covers, and *what it must not disturb*.
It deliberately does not hold the diagnostics themselves — the renderer, compute and capture code
that produces the data lives in the sibling `live/`, `perf/`, `present/` and `rtt/` folders, and in
`tools/`.

The boundary is worth stating because it is what makes this folder useful: a policy here can be
unit-tested with no Vulkan device, no game dump and no GPU, which is precisely what the code that
*implements* a diagnostic cannot be. `frame_dump_policy.hpp` resolves its switch independently of
`getenv` for that reason; `trip_bound_witness.hpp` is split out of `live_compute.cpp` so a test can
assert guest GDS is byte-identical after an instrumented dispatch rather than resting on a reading of
the code.

What belongs here: an environment-switch resolver, a census-window selector, a residency or
ownership policy, a runtime binding to an external debugger. What does not: anything that needs a
device to compile against, and anything a single call site is the only plausible user of.

## The recurring lesson in this folder

Every file here exists because a diagnostic was **aimed wrongly and still produced a confident
answer** — which is worse than one that fails. `diagnostic_window.hpp` exists because a
renderer-callback ordinal has no published rate, so "census at N" lands somewhere different on every
route. `renderdoc_capture.hpp` exists because RenderDoc, pointed at the obvious Vulkan device,
returns a complete and valid capture of prosper's *presentation* device containing zero draws.

So when adding to this folder: make the wrong aim **loud**. Report which axis fired and what the
span actually covered, prefer a switch whose malformed value disables its own trigger over one that
fires at an unintended moment, and say "unavailable, and here is why" at arm time rather than
letting a four-minute route reach its trigger and find nothing there.
