# `frontends/shared/live/submit_renderer/` — the submit callback, in named pieces

`register_live_renderer` (in `../live_renderer.cpp`) installs one callback that turns each decoded
guest submit into GPU work. It grew into a single 12,000-line lambda; #3892 carves it along the
responsibilities it already had. This folder holds what has been carved out so far. The callback
itself still lives in `live_renderer.cpp` and still owns all the state.

- `callback_types.hpp` — the types, aliases and constants the callback used to declare locally
  (`RenderTiming`, `BuiltFrameResources`, the depth-snapshot records, `RC`, `RenderClock`, ...).
  Types only: the callback's statics stay in the callback.
- `guest_reads.hpp` — `safe_span` / `safe_copy` / `safe_equal`, the bounds-checked guest reads.
- `draw_resources.{hpp,cpp}` — `build_draw_frame_resources`: one draw's VS/PS resource tables to
  backend frame resources (the texture- and buffer-reference resolution chain; the hottest code in
  the renderer). Formerly the `build_R` lambda.
- `timing_report.{hpp,cpp}` — `report_render_timing_aggregates`: the `PROSPER_RENDER_TIMING`
  lifetime/window summaries.

**How a piece talks to the callback.** Each extracted function takes a `*Context` struct of
references, one per callback object it touches. The callback builds it once per callback, so the
per-draw path copies nothing. The function binds each reference to the name the old code used,
and that is why the moved bodies are verbatim. Keep it that way: a context of references, built
once, bound by name. No `std::function`, no copies, and no per-draw allocation.

**Two traps when moving more code out.** A lambda inside the callback may name one of its
`static` or `static thread_local` objects without capturing it. That is legal because the object
has static storage. Once the object arrives as a context reference, the lambda can no longer see
it. If the lambda can run later (a stored writeback or a failure cleanup), bind that name with a
function-scope `static` / `static thread_local` reference, as `draw_resources.cpp` does. Do not
edit the capture list. Second, a lambda the callback also uses elsewhere cannot move with the
code. Pass it as a template parameter, as `render_per_target_passes` in `live_renderer.cpp`
does. That keeps the call direct and inlinable.

Where the next pieces go, and what remains in the callback, is tracked on #3892.
