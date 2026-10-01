# `tests/gpu/recompiler` — translation tests and the legacy compute execution fixture

Most tests here ask a question of `src/gpu/recompiler` alone: does this instruction stream decode,
does it recompile, does the module it produces have the shape the contract promises, and — just as
often — does a stream that must NOT compile still reject, and reject *loudly*. No device is created,
so these run everywhere and are the cheapest place in the project to pin a translator contract.
The historical exception is `test_game_compute.cpp`: it executes synthetic shaders through the live
Vulkan backend, including storage writeback, cache authority, and real mapped-page write watches.
Its existing end-to-end fixtures remain here; new standalone execution tests belong in `execute/`.

**The boundary against its siblings.** `tests/gpu/execute/` runs modules on a real device and
asserts pixels or buffer contents; anything needing a queue belongs there, not here. Two large
recompiler tests also live one level up in `tests/gpu/` for historical reasons —
`test_rdna2_spirv_struct.cpp` (structural module validation) and `test_recompile_coverage.cpp` —
and new work usually belongs in this folder instead.

**Prefer a small binary over another arm in a big one, whenever rejects are the subject.** Most
tests here return at their first `[FAIL]`, which is fine for a suite that should be all-green but
useless the moment you want to know *which* assertions moved: a mutation run reports the first
failure and nothing after it. A reject-contract test is exactly that case, because the way you
establish it is load-bearing at all is to break the mechanism deliberately and read which arms
redden. So a focused reject test gets its own executable and a `CHECK` macro that counts failures
and keeps going (`test_entry_m0_dispatcher.cpp`, `test_vertex_fetch_reject_diagnostic.cpp`).

**A reject arm without controls beside it is not evidence.** An empty result means the stream was
refused; it never says by whom or why, and a recompiler has many reasons to refuse. The arms that
have held up here carry a positive control with the same region shape and one instruction changed,
so a reject that came from a mis-encoded branch or an unrepresentable region cannot pass as the
contract being tested. Where the route matters, say so in the module: assert an opcode only one
emitter produces, or read `last_terminal_reject_reason()` (it needs a non-zero program address —
`record_terminal_reject_reason` early-returns on zero) and assert on the tag.

Register new cases in `prosper/CMakeLists.txt` under `if(TARGET prosper_core)`, which is
platform-independent — the file also has `if(WIN32)`/`if(UNIX)` branches, and a case registered in
the wrong one silently never runs while ctest stays green. Verify by name with `ctest -N`, never by
the total count.

`test_raw_wide_mask_lifetime` separates a fresh emitted Bool mask from a possible surviving
physical high-word dependency of an earlier raw x4 load. Numeric negatives assert the actual raw
load's PC/op in addition to classification and an empty stage module; a later unrelated refusal
does not prove the backing gate worked. Keep per-compile diagnostic addresses unique across stages.
The no-descriptor-patch high-word arm also exercises the outer lifetime shortcut, while the patched
arm reaches derived provenance. The one-arm compare targets the MUST mask meet at a forward join.

## Ruled out

- Running the owned-input chain proof before every cache lookup is not a negligible warm-hit
  cost. A balanced synthetic unowned 1,024-VMOV chain measured about 0.566 ms per current hit
  versus 0.000817 ms with that private guard disabled. This is CPU cache-hit evidence, not title
  FPS. Memoize the code-only proof by both immutable analysis versions; live resource obligations
  and markers still require a check before every lookup. Preserve the uncached measurement.

- Stripping only the v62 marker tail does not form a valid v61 capture after v63 appends its
  owned-wide count/width tail. The first current-base raw backing calibration failed this
  historical-layout control; remove both tails and separately pin the retained v62 marker.
  A future-version refusal fixture must also use a version above current v63 (#3987).
- A successful current capture cannot still be asserted as v62 after the owned-wide v63
  addition. The full Linux suite exposed this stale expectation in both existing indirect-pointer
  capture controls; keep their owned-byte, unused-candidate and round-trip assertions intact.
- A non-CMPX compare's SGPR-pair-shaped destination does not prove both physical words were
  overwritten in an unknown wave mode. A fresh Bool condition can be independent while an ordinary
  numeric high-word read still requires backing.
- A SAVEEXEC fixture that leaves EXEC narrowed cannot export a vertex position. Restore full EXEC
  before that export when testing independent mask transfer; otherwise an unrelated export refusal
  hides whether the mask proof worked.
