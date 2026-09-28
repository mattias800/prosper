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
  lifetime/window summaries; and `publish_renderer_timing_record`, the F8 capture's per-submit
  `RendererTimingRecord`.
- `backend_draws.{hpp,cpp}` — `build_backend_draws`: one `BackendDraw` per realized draw of a pass
  group (shaders or diagnostic overrides, `build_draw_frame_resources`, contract validation,
  poison mode, indices, per-draw census/skip diagnostics). Formerly `build_bds`. Also `clear_for`.
- `backend_timing.{hpp,cpp}` — `record_backend_timing_stats` (folds one backend call's counters
  into the submit's `RenderTiming`) and the `PROSPER_RTT_TIMING` record formatters.
- `per_target_passes.{hpp,cpp}` — `render_per_target_passes`: the `PROSPER_RTT` per-target pass
  loop (pass grouping, MRT, RTT publication, and choosing the present candidate among its passes).
  The scanout resolution and GPU scanout publish that follow it still live in the callback.
- `resolve_pass.{hpp,cpp}` — `resolve_pass`: one `CB_COLOR_CONTROL.MODE=RESOLVE` pass, a copy of
  colour0 into colour1's retained target. The pass loop's `continue` stays at the call site.
- `pass_diagnostics.{hpp,cpp}` — the pass loop's `PROSPER_*` diagnostic blocks (readback reasons,
  resource and draw-step hashes, pass/RT-group/persistent-target dumps, the pass logs).
- `callback_prelude.{hpp,cpp}` — per-callback setup work: `load_shader_overrides` (the REFVS /
  TESTPS / FS_SPV / SKIP_DRAW overrides, one `ShaderOverrides` value that `BackendDrawContext`
  refers to) and `materialize_dirty_dcc_clears`.
- `callback_diagnostics.{hpp,cpp}` — the callback's own diagnostics: submit-index logs, the
  diagnostic render-window selectors, the SPIR-V dump and the presented-frame dumps.

**Diagnostics keep their gate beside their report.** When a diagnostic block moves out, a gate
that is a `PROSPER_*` test moves with it, so `tools/env/check_diag_gates.py` still sees the gate
next to the report it arms (the scanner is per-file and lexical, and does not follow a gate across
a call). A gate that is a value of the calling code (`rtt_log`, a hash extent) stays at the call
site, so a false gate builds no context and makes no call.

**How a piece talks to the callback.** Each extracted function takes a `*Context` struct of
references, one per callback object it touches. The callback builds it once per callback, so the
per-draw path copies nothing. The function binds each reference to the name the old code used,
and that is why the moved bodies are verbatim. Keep it that way: a context of references, built
once, bound by name. No `std::function`, no copies, and no per-draw allocation.

**Two traps when moving more code out.** A lambda inside the callback may name one of its
`static` or `static thread_local` objects without capturing it. That is legal because the object
has static storage. Once the object arrives as a context reference, the lambda can no longer see
it. If the lambda can run later (a stored writeback or a failure cleanup), bind that name with a
function-scope `static` / `static thread_local` reference, as `draw_resources.cpp`,
`backend_draws.cpp` and `per_target_passes.cpp` do. Do not edit the capture list. Second, a lambda
the callback also uses elsewhere cannot move with the code: turn it into a named function first
(as `build_backend_draws` and `record_backend_timing_stats` were), and leave a one-line forwarder
where the call sites expect the old name.

**When the static-reference binding is safe.** A function-scope `static T& x = ctx.x;` binds
**once, on the first call, forever**. A `static thread_local T& x = ctx.x;` binds once **per
thread**, on that thread's first call. So:

- Use a plain `static` reference only for an object that is the *same object on every call*: one
  of the callback's own process-lifetime statics (`g_rtt`, `skip_draws_env`). Never bind a
  per-call local, a per-submit object or a `thread_local` this way: every later call would reach
  the first call's object, and once that is gone, a dangling one.
- Use `static thread_local` for the callback's `static thread_local` objects. The reference is
  null (unbound) on a thread that has not run the binding function, so a deferred lambda that
  names it must run on the thread that bound it. The code it replaced already required that
  (storage-image writebacks run on the render thread); keep it that way.
- Anything else a deferred lambda needs must be captured by value, or captured by reference to an
  object that outlives the lambda. A `[&]` capture of a context *binding* (`auto& x = ctx.x`)
  reaches the callback's object, not the context struct (a reference captured by reference names
  the object it is bound to, CWG 2011), but never capture `ctx` itself in a
  lambda that can outlive the call: the context struct dies with the callback frame.

Where the next pieces go, and what remains in the callback, is tracked on #3892.
