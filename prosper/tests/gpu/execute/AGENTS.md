# Execution-layer tests

`test_arbitrary_placeholder_exec` runs the real exact64 production compute backend and checks
both initialized low lanes and retained arbitrary high lanes. It keeps the actual device
subgroup-contract gate and rejects CPU-fill substitution.

These tests cover the contracts that turn decoded GPU work into executable operations: dependency
ordering, draw realization, index interpretation, and compiled-shader cache identity. The cache tests
compare emitted modules and warm-cache behavior without running a game; they are not visual or
performance oracles. Vulkan execution coverage is separate, including tests under `tests/shared/`.

When changing a cache key, exercise both a reusable identical key and a changed code-generation
input. Diagnostic selectors are code-generation inputs too, even when normal launches leave them off.

`test_index_size_detect` covers the `draw_index_width` fingerprints. `test_ngg_indexed_realize`
covers the shared `draw_index_source` rule and indexed NGG fetch through the executor umbrella.
Both targets link the implementations from `prosper_core`.
