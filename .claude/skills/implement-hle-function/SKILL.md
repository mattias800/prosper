---
name: implement-hle-function
description: Implement or fix a reimplemented Sony library function (HLE) in prosper, test-first. Use when a guest calls an unregistered, stubbed, or wrongly answering libkernel/libc/libSce* function.
---

# Implement an HLE function

Preserve task authorization, resource coordination and `LOCAL.md`; this skill grants no additional
permission to execute workloads, publish, push or merge.

1. **Name it.** Resolve the NID to a name and owning library: `grep -rn '<NID>' ../PS5-3.20_Libs/`. Names and NIDs only; no bodies.
2. **Re-verify it is still broken on current `main`**, and check `gh issue list` for a claim (`CLAUDE.md`, claiming rule).
3. **Read the owning folder's `AGENTS.md`** under `prosper/src/hle/<area>/` and copy how neighbouring exports are registered (e.g. the `R("name", fn)` tables). Read the title's `## Ruled out` section before forming a hypothesis.
4. **Establish behaviour from evidence, in order:** live trace of the real guest, guest disassembly, published contracts, firmware symbol data. A single secondary implementation is a hypothesis only; never copy another emulator's code or tests. Mark uncertain code `CONFIDENCE: HIGH/MED/LOW`.
5. **Add or extend a regression** in `prosper/tests/` (rules: `prosper/tests/AGENTS.md`): success path, established SCE/POSIX error codes, and pointer/size cases required by the API contract. Reuse existing coverage where it constrains the change. Run the focused test when authorized and confirm it is red without the fix.
6. **Implement.** Return the code a console returns. Only a truly unsupported state aborts, loudly. Never a silent `return 0` stub. Entitlement and add-content answers derive from local inventory and are never unconditionally "owned"; the same question must answer identically through every library that exposes it.
7. **Prove the test constrains the fix:** revert the fix, see red, restore, see green.
8. **Verify relevant cases:** `ctest --test-dir <BUILD_DIR> --no-tests=error` with the focused selection; quote the count and exit code.
9. **Record:** a falsified hypothesis goes in the relevant `## Ruled out`; record deferred bugs through the authorized issue workflow; the PR follows `.github/pull_request_template.md`.
