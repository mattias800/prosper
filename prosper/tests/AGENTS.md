# tests — what a test here must do

Rules for tests under `prosper/tests/`. Tests are standalone executables registered with `add_test` in
`prosper/CMakeLists.txt` (there is no GoogleTest dependency), reporting through a local `CHECK`
macro or `tests/support/`.

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

A new test file goes in the subfolder matching the subsystem it exercises (`hle/`, `gpu/`,
`loader/`, `self/`, …), registered by `add_test` in `prosper/CMakeLists.txt`. `fixtures/` holds
inputs and machinery under test; `support/` holds helpers that can never change an assertion's
outcome (see its own `AGENTS.md`).

## Enforced vs. judged

CI's `Contribution shape` job checks allowed new-file locations and requires a new
`prosper/src/**/*.cpp` to touch `prosper/tests/`. Review checks the behavioral contracts above;
report relevant red-without-fix evidence, test counts and execution limits without inventing results.
