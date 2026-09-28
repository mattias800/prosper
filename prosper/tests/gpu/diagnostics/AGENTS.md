# tests/gpu/diagnostics — tests for the GPU diagnostic instruments

Covers the instruments in `src/gpu/diagnostics/`: selectors that change what the GPU layer does
when armed (`draw_program_skip`, `watch_list`, `shader_dump_filter`), and censuses that only
observe (`draw_disposition`, `pass_break_census`, `link_list_census`, `compute_tree_watch`).

**What belongs here is a test of the INSTRUMENT, not of the thing it measures.** A census is
exercised by driving its counters directly and asserting on what it reports; a GPU boot is not
needed and would make the test depend on the very behaviour the census exists to describe. Tests
that need a live device or a real draw stream live in `tests/gpu/` beside the subsystem they
exercise, not here.

The property these tests exist to pin is that an instrument cannot quietly lie. That is a
different bar from "it compiles and counts":

- a census must stay SILENT when there is nothing to report, or it gets switched off and then
  reports nothing when it matters;
- it must distinguish "I saw nothing" from "I was never called" — those look identical from
  outside and only the first is a finding;
- it must keep its per-key distribution separable, because the loudest key is usually not the
  actionable one;
- where two independent routes should agree, it must SAY SO when they do not, rather than
  presenting a balanced total that hides a gap.

Each of those has cost this project real time when it was missing, which is why they are asserted
rather than assumed. See `src/gpu/diagnostics/AGENTS.md` for the instruments themselves and
`diag_ratelimit.hpp` for the rate-limiting contract every capped diagnostic here must follow.
