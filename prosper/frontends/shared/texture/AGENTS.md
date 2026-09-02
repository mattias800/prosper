# `frontends/shared/texture` — the policies that decide whether guest bytes must be re-read

Header-only, dependency-free decision logic shared by the two frontend caches that hold guest-backed
resources: the renderer's texture cache (`live/live_renderer.cpp`) and the compute source/result
cache (`live/live_compute.cpp`). Nothing here touches Vulkan, guest memory, or the host write-watch
implementation — it answers questions *about* them, from plain integers, so each rule is unit-testable
without a device. The tests live in `frontends/shared/tests/`.

**The boundary against its siblings.** `src/host/memory/guest_write_watch.*` owns the mechanism (page
protection, alias tracking, the SIGSEGV fault path). This folder owns the *policy*: whether a source is
worth arming, how many proven-unchanged validations it takes, how much may be armed per submit. Cache
identity and materialization belong to the callers; media plane classification lives in
`shared/media/`.

**The one thing a newcomer gets wrong.** `should_promote_write_watch` takes both a size threshold and a
stability threshold and promotes on *either*, and the two call sites pass different sizes. The renderer
passes 8 MiB, so a 1-8 MiB texture arms on first sight — a decision measured, and its opposite rejected,
in `docs/RENDERER_PERFORMANCE_2026_07.md`. The compute path passes **1**, which makes the size exemption
unreachable and leaves the stability ladder as its only route to a watch. That asymmetry is real, mostly
accidental, and unresolved (#3155). So read the *call site's arguments* before concluding what the policy
does for a given path; the header describes a contract only one caller uses in full.

`write_watch_census.hpp` is the instrument for exactly that question, and it exists because the previous
answer was measured from outside the process with an `LD_PRELOAD` `memcmp` interposer whose periodic
tally was read as a run total. Prefer counting a decision where it is taken over counting a proxy for
its consequences; if you add a counter here, make its report carry its own denominator.
