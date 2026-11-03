# Execution-layer tests

The native pixel parameter GPU fixture also checks the saved-mask/SCC0 terminal trailer:
one genuine Input keeps the colour, another must match an independent always-discard module.

`test_native_entry_exec_words_gpu` executes complete groups and the final 56-worker
group on a required native Wave64 device, checking retained HIGH, the new LOW, old inactive
register values and absent-worker output sentinels.

`test_pixel_parameter_producer` drives the actual draw producer/cache with known, absent and
changed PS user prefixes. `test_native_pixel_parameter_gpu` observes real native interpolation
and the reused SGPR lifetime in the colour attachment, with two distinct parameter values.

`test_loop_vcc_data_exit_gpu` executes the shared structured-loop witness on a required native
Wave64 subgroup, checking numeric VCC halves and both nested scalar branch outcomes.

`test_arbitrary_placeholder_exec` runs the real exact64 production compute backend and checks
both initialized low lanes and retained arbitrary high lanes. It keeps the actual device
subgroup-contract gate and rejects CPU-fill substitution.
Runtime-relative bank cases retain provided or genuinely unloaded selected values and reject a
lost provided candidate even when other possible candidates are unloaded.
Aliased destinations also retain this check before their temporary map slot becomes a value.
Wide scalar and implicit SCC selectors observe every lane for both supplied condition values;
genuinely unloaded high words and SCC sources retain their consistent computed selections.

These tests cover the contracts that turn decoded GPU work into executable operations: dependency
ordering, draw realization, index interpretation, and compiled-shader cache identity. The cache tests
compare emitted modules and warm-cache behavior without running a game; they are not visual or
performance oracles. Vulkan execution coverage is separate, including tests under `tests/shared/`.

When changing a cache key, exercise both a reusable identical key and a changed code-generation
input. Diagnostic selectors are code-generation inputs too, even when normal launches leave them off.

`test_index_size_detect` covers the `draw_index_width` fingerprints. `test_ngg_indexed_realize`
covers the shared `draw_index_source` rule and indexed NGG fetch through the executor umbrella.
Both targets link the implementations from `prosper_core`.

`test_allocated_movrels_gpu` checks the declared bank and current-VGPR0 OOB source on a
required native Wave64 subgroup, using real runtime push constants and both cohort halves.
Its LDS boundary arm checks a final-bank address and two-word data without charging reserved
DS fields as reads; each lane owns a distinct LDS cell.

`test_native_fp_mask_participation` observes saved physical mask words after a real native
loop controlled by finite, NaN and mixed floating predicates. The scalar branch is wave-wide;
missing spill and high-word inputs remain refusals. The integer control retains its old route.

The native entry-word fixture also observes a complete numerical overwrite after a dead
READFIRST arithmetic relay; the old publication supplies no later mask bit.

A PC0 loop executes its current one-hot mask after a complete per-iteration overwrite,
independently of an earlier READFIRST publication on the backedge.
