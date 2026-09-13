# Shader-key preparation candidate (#3407)

Status: candidate implemented and independently reviewed; 512 functional tests and nine focused
Vulkan synchronization checks pass, with no validation messages. The validation runner first proved
the layer loads and reports a deliberately introduced synchronization hazard. Game comparisons and
the CPU benchmark have not been collected. This is not an FPS improvement claim or a recommendation to
merge the candidate before measurement.

The retained PR #3622 CPU profiles attribute approximately 1.1–1.2% of sampled leaf period weight
in Sonic Frontiers and 2.8–3.2% in GTA V to `ShaderCompileKeyHash::compute`. These are sample weights,
not elapsed frame time or newly rendered FPS. The same lookup path constructs a temporary resource
key vector on every call. The candidate groups cheaper hashing with reuse of that allocation.

## Implementation boundaries

- Hash each existing semantic field a word at a time, then avalanche the result for bucket
  selection. Full key equality remains unchanged. Shader-code hashes, shader dump names, compiled
  module identities and persisted cache formats are unchanged.
- Rebuild resource semantics on every lookup using reusable vector storage. Each active call owns
  its vector. No guest data, descriptor result or resource-generation decision is memoized.
- Keep at most 64 KiB of idle vector capacity per thread. Active keys and cache entries have their
  own allocations. Larger scratch is discarded. A smaller cache miss restores the existing fresh
  vector allocation behavior before insertion, returning oversized scratch before cache eviction
  or accounting changes. Stored keys do not participate in scratch recycling.
- Freeze each policy once per process. `PROSPER_NO_SHADER_KEY_WORD_HASH=1` restores the original
  byte mixer; `PROSPER_NO_SHADER_KEY_SCRATCH=1` restores allocation on each resource-bearing lookup.
  Both controls are for comparison and bisection. No Vulkan feature or platform-specific memory
  tracking is introduced.

## Reproduction

Build `test_shader_recompile_cache` in a Release configuration. The focused CTest matrix runs the
semantic suite in all four policy combinations:

```sh
ctest --test-dir prosper/build-linux -R '^shader_recompile_cache' --output-on-failure
```

The optional benchmark invokes the real warm shared-module API with synthetic resource contexts:

```sh
prosper/build-linux/test_shader_recompile_cache --benchmark-key-preparation
PROSPER_NO_SHADER_KEY_WORD_HASH=1 prosper/build-linux/test_shader_recompile_cache --benchmark-key-preparation
PROSPER_NO_SHADER_KEY_SCRATCH=1 prosper/build-linux/test_shader_recompile_cache --benchmark-key-preparation
PROSPER_NO_SHADER_KEY_WORD_HASH=1 PROSPER_NO_SHADER_KEY_SCRATCH=1 prosper/build-linux/test_shader_recompile_cache --benchmark-key-preparation
```

Use the same executable and a quiet CPU window; repeat with balanced order. JSON lines report
resource/context/iteration counts, scalar C++ allocation requests, SPIR-V checksums and nanoseconds
per lookup. The shader does not execute these synthetic resources, so this measures API preparation,
not representative shader execution. Compare checksums and exact hit/miss assertions across arms
before comparing times. There are no timing thresholds in the functional tests.

The new functional guards require allocation-free warm resource keys, distinguish resource-only
context changes, return immediately to a large warm key after smaller misses, and check that
storage exceeding the idle limit is not retained. Against original production code, the unchanged
new tests fail six allocation assertions. Removing only miss compaction fails the immediate return
to the large key. To reproduce these negative controls in a disposable checkout, keep the new test
and replace `gpu_executor.cpp` with its version from `ac4dc1c2d735`, then rebuild and run
`test_shader_recompile_cache`. For the second control, retain the candidate and remove its two
`scratch.prepare_for_cache()` calls before rebuilding. Restore the candidate after each control. The existing suite also covers shader changes, diagnostic selectors, compute
configuration, vertex chains, concurrent readers and shared-module lifetime across cache reset.

## Remaining decision

First separate hashing and scratch effects with the CPU benchmark. Then compare the combined
candidate with both controls enabled using the same binary and committed Sonic/GTA routes, fresh
save/cache directories, matching frame captures, CPU profiles and F8 measurements. Keep repeated
presentations distinct from newly rendered frames and retain audio demand observations. Reject
runs contaminated by another build or game. Keep or revise the candidate based on those results;
allocation reduction alone does not establish an improvement on the frame's critical path.
