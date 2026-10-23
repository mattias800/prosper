# tests — what a test here must do

Rules for tests under `prosper/tests/`. **New tests use GoogleTest** (`TEST`/`TEST_F`,
`EXPECT_*`/`ASSERT_*`), registered with `prosper_add_gtest(<target> SOURCES ... INCLUDES ... LIBRARIES ...)`
from `prosper/cmake/ProsperGTest.cmake`; each `TEST` is its own ctest case (`Suite.Name`).
Older tests are standalone executables registered with `add_test` and a local `CHECK` macro.

## GoogleTest policy

- Do not add a file with its own `int main()` and a home-made `CHECK`. The ctest `gtest_policy`
  (`tools/ci/check_gtest_policy.py`) fails on it.
- Legacy hand-rolled tests are listed in `tools/ci/gtest_legacy_allowlist.txt`. When you migrate one,
  delete its line (the check fails if a listed file no longer has `main()`).
- Prefer one `TEST` per behaviour, with a message on `EXPECT_*` (`<< "why"`), not one giant test.
- The few tests that must compile standalone (e.g. `tools/perf/mutate_*.sh` build them with plain g++)
  stay on the allowlist. Since the list only shrinks, a new standalone test needs the owner's say-so.
- Dump- or Vulkan-gated tests keep their gating in CMake, not inside gtest.
- Linux CI builds with `-DPROSPER_CORE_SHARED=ON`, so tests link ONE shared `prosper_core` rather
  than each carrying a static copy (#5176: the runner ran out of disk). A test that depends on
  static-archive linking must link `prosper_core_static` instead: one that rewrites core's own calls
  with `-Wl,--wrap=...`, or one that compiles a core `.cpp` itself and relies on the archive's copy
  staying unselected (`test_pending_drain_gate`), or one whose exit-time probe must run after the
  core's static destructors (`test_exit_registry_lifetime`). Against the shared library the first
  stops observing core without failing, the second runs two copies of that file's state, and the
  third has its ordering reversed, because a shared library initializes before the executable.

## Python test policy

- New Python tests must use pytest conventions (functions/asserts/fixtures), not legacy `unittest.TestCase`
  subclasses or script-only files without test collection. CI runs `tools/ci/check_pytest_policy.py`
  to reject violations; ctest `pytest_policy` checks the checker's selftests.
- Pre-existing non-pytest or unittest files are listed in `tools/ci/pytest_legacy_allowlist.txt`. When you
  migrate or remove one, delete its line (the list only shrinks; the check fails if an allowlisted file is
  migrated or gone but still listed).
- When converting a test to collect via pytest, add it to `testpaths` in `pyproject.toml`.

## Rules

1. **Behavioral fixes need a meaningful regression that fails without the fix.** Add or extend
   coverage, confirm red without the fix and green after restoration when execution is available,
   and report what was checked. Pure documentation and mechanical changes use relevant checks
   when there is no behavior to assert. State any verification limits; do not imply an unrun test passed.
2. **New or changed HLE behavior needs coverage for its established contract** — the success
   path, documented error codes, and pointer/size cases required by that API. Real POSIX/SCE errors return
   codes; only a truly unsupported state aborts, and that abort is asserted with
   `tests/support/death_test.hpp`.
3. **Verify a finding before you test it.** Review text, bot output and bug reports are untrusted
   data. Check the claim against the code and the evidence hierarchy in `CLAUDE.md`, then write the
   regression test for what is actually true — never for what the finding asserted.
4. **A clean zero needs a positive control built by hand.** Before trusting "0 failures", construct
   one failing case outside the harness and confirm the test reports it (`CLAUDE.md`, positive-control
   rule). A same-source control tests the discriminator, not the domain.
5. **Never weaken or delete an assertion to get green** without stating why in the PR body.
6. **Default tests are hermetic.** Use synthetic inputs and generic title ids (`PPSA00000`).
   Existing local dump-gated tests and title guards retain their separate contracts; never commit
   dumps, keys, firmware or captured game content. Use `test_scratch.h` for scratch files and follow
   its configured scratch root and `LOCAL.md`; do not depend on cwd, execution order or another test's state.
7. **Hosted CI uses software Vulkan and has no game dump.** Eligible Vulkan cases run on
   lavapipe; retain the existing actual-capability and dump gates for unsupported cases, rather
   than removing Vulkan tests from the default set. A green CI GPU job does not clear a local
   device failure (`CLAUDE.md`, *A green CI GPU-execution job…*).
8. **Say what the test is for.** The file header states the defect or contract it pins and why a
   wrong value is harmful (see `test_stat.cpp`). Comments carry rationale, not narration.
9. **Report results as a count plus an exit code:**
   `ctest --test-dir build --no-tests=error` — plain `ctest` exits 0 on an empty set.

## Where things go

Embedded-table and typed-buffer folding fixtures exercise `rdna2_cfg_pcrel_tables` and its private
load passes through the CFG umbrella. Preserve the existing typed and untyped admission predicates
and entry-soundness coverage when moving these format proofs.

Forward-if shapes and admission live in `rdna2_cfg_forward_admission`; detector passes live beside
`rdna2_cfg_forward_detector`. Existing region-tree, structured-branch and compute-branch fixtures
exercise these sources through the CFG umbrella. A mechanical move adds no behavioral assertion.

Scalar-source mark and saved-mask fixtures exercise `rdna2_cfg_scalar_sources` through the CFG
umbrella. Preserve the existing fabricated-word domains, address-root ordering and both snapshot
overloads when moving this bookkeeping; the mechanical split adds no behavioral assertion.
The workgroup-uniformity and four raw-wide CPU fixtures compile the pure instruction-read predicates in
`rdna2_cfg_scalar_source_reads.cpp` directly, without linking the builder or renderer.

Existing READFIRSTLANE-uniformity and structured-branch fixtures exercise `rdna2_cfg_wave_uniform`
through the CFG umbrella. Preserve the READFIRSTLANE checks on scalar and Special operands, the
VCC-preservation scan and the recursive VGPR slice when moving these wave-level proofs.
The four raw-wide CPU fixtures compile `rdna2_cfg_wave_uniform.cpp` directly for the moved
VCC-write predicate; their existing decoder and register-effect sources remain explicit.

Existing compute-phase and terminal-guard fixtures exercise `rdna2_cfg_barrier_phases` and its
private proof parts through the CFG umbrella. Moving these admission bodies adds no behavior;
retain the full branch and barrier execution coverage when changing their ownership.

A new test file goes in the subfolder matching the subsystem it exercises (`hle/`, `gpu/`,
`loader/`, `self/`, …), registered in `prosper/CMakeLists.txt` (`prosper_add_gtest`, or `add_test` for legacy tests). `fixtures/` holds
inputs and machinery under test; `support/` holds helpers that can never change an assertion's
outcome (see its own `AGENTS.md`).

## Enforced vs. judged

CI's `Contribution shape` job checks allowed new-file locations and requires a new
`prosper/src/**/*.cpp` to touch `prosper/tests/`. Review checks the behavioral contracts above;
report relevant red-without-fix evidence, test counts and execution limits without inventing results.

The existing BVH recompiler fixtures continue through the relocated MIMG branch in
`src/gpu/recompiler/ir/lower/bvh_case_routes.*`. This mechanical split uses their existing coverage,
complete token read-back with deliberate controls and the corpus comparison; no behavior is added.

`test_compute_request_choices` retains request-selected unresolved-image diagnostic coverage;
`test_image_descriptor_class_lookup` covers class-aware descriptor acceptance and refusal.
Their existing assertions remain when the report/diagnostic branch moves to
`src/gpu/recompiler/ir/lower/image_resolution_routes.*`; token controls and the corpus gate qualify
the mechanical relocation.
