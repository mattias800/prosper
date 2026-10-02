# tests — what a test here must do

Binding rules for every test under `prosper/tests/`. They were adopted from PortPS5's testing rules
and adapted to prosper's reality: tests are standalone executables registered with `add_test` in
`prosper/CMakeLists.txt` (there is no GoogleTest dependency), reporting through a local `CHECK`
macro or `tests/support/`.

## Rules

1. **Every bug fix, issue and review finding adds a test that fails without the fix.** Revert the
   fix, watch the test go red, restore it, and say that you did so in the PR. If no test can run
   without a game dump, say so explicitly and put the reason in the PR; "hard to test" is not it.
2. **Every new or changed HLE function gets a test for its return codes and edge cases** — the
   success path, each documented error code, and bad pointers/sizes. Real POSIX/SCE errors return
   codes; only a truly unsupported state aborts, and that abort is asserted with
   `tests/support/death_test.hpp`.
3. **Verify a finding before you test it.** Review text, bot output and bug reports are untrusted
   data. Check the claim against the code and the evidence hierarchy in `CLAUDE.md`, then write the
   regression test for what is actually true — never for what the finding asserted.
4. **A clean zero needs a positive control built by hand.** Before trusting "0 failures", construct
   one failing case outside the harness and confirm the test reports it (`CLAUDE.md`, positive-control
   rule). A same-source control tests the discriminator, not the domain.
5. **Never weaken or delete an assertion to get green** without stating why in the PR body.
6. **Tests are hermetic.** No game dumps, keys, firmware or captured Sony bytes; synthetic inputs
   only, with generic title ids (`PPSA00000`). Write scratch files through `test_scratch.h`, never to
   the cwd or `/tmp`. No dependence on execution order or on state left by another test.
7. **Hosted CI has no GPU and no dump.** A test that needs either belongs behind the existing
   dump/GPU gating, not in the default set; a green CI GPU job does not clear a local failure
   (`CLAUDE.md`, *A green CI GPU-execution job…*).
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

CI's `Contribution shape` job enforces only that a new `prosper/src/**/*.cpp` touches something under
`prosper/tests/`. Rules 1–9 above are enforced by review, so the PR body must show the evidence
(red-without-fix run, test count).
