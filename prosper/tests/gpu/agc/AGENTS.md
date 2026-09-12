# `tests/gpu/agc` — the AGC front half, tested without a device

Everything here asks a question of `src/gpu/agc` and the predicates it feeds, with no Vulkan device
and no queue: does an eight-dword T# / four-dword V# decode to the fields the rest of the pipeline
reads, does `build_shader_resources` turn a shader's user-data block into the resource table the
recompiler and the backend consume, and — at least as often — does a descriptor that must NOT
materialize get refused, by the right predicate, for a nameable reason. Several files exist because a
predicate was *extracted* from `execute_item` so it could be reached at all (`test_image_identity`,
`test_mip_provenance_identity`, `test_spirv_storage_match`, `test_atomic_image_staging`); each one's
header says which issue moved it and what the unnamed expression used to hide.

**The boundary against its siblings.** `tests/gpu/execute/` puts modules on a real device and asserts
pixels or buffer contents — anything needing a queue belongs there. `tests/gpu/resources/` guards the
`ShaderResource` container and its own helpers. This folder stops where the descriptor stops: the
decoded struct, the predicate that judges it, and the table entry it becomes.

**A reject verdict here is a CONTRACT STRING, not a diagnostic.** `image_descriptor_reject_reason`
returns a `const char*`, and production code keys on the exact text — `gpu_executor.cpp` publishes an
explicit null image only when the reason is `"base-zero"` *and* all eight dwords are zero. So renaming
a verdict, or reordering the predicate so a descriptor reports a different one first, is a behaviour
change that no compiler will catch. Add a new screen at the END unless you mean to change which
verdict an already-refused descriptor reports, and pin the ordering with an arm.

**Know what your fixture actually encodes.** `make_tsharp` (in `test_build_shader_resources.cpp`)
zeroes the whole descriptor and then sets only base/extent/format/tile/type/array, which leaves
WORD3\[11:0\] at zero — so every T# it builds decodes `DST_SEL` as four SQ_SEL_0 constants, *not* as
identity `(R,G,B,A)`. Any arm about channel routing has to set those bits itself; one that does not is
asserting about a constant-zero descriptor while reading as if it tested the ordinary case.

**Two properties make this folder the right place to do mutation work**, and they are worth keeping:
the tests are pure, so a rebuild is seconds rather than a device round trip, and `test_build_shader_
resources.cpp`'s `CHECK` counts failures and keeps going instead of returning at the first one. That
is what lets you disable a predicate, rebuild, and read *which* arms redden — the only way to show a
reject arm was load-bearing rather than passing beside the thing it claims to test.

Register new cases in `prosper/CMakeLists.txt` under `if(TARGET prosper_core)`; a case registered in a
platform branch silently never runs while ctest stays green. Verify by name with `ctest -N`.
