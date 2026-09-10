<p align="center">
  <img src="assets/prosper-logo.png" alt="prosper" width="200">
</p>

# prosper

prosper is a user-space PS5 (Prospero) → PC compatibility layer for Linux and Windows. The PS5 is
x86-64, so guest code runs natively and there is no CPU emulation. The work is loading Sony's
SELF/ELF modules and linking them into one address space, reimplementing the PS5 system libraries
on top of host facilities, translating the ABI where the guest's conventions differ, and translating
AGC GPU command streams to Vulkan, which includes an RDNA2 → SPIR-V shader recompiler.

It exists to test a method: whether a repository can be structured — charter, tooling, tests,
evidence rules — so that AI agents do reverse-engineering work of this kind over months, with a
human who sets direction and reviews results but does not write code. Preservation work has this
shape, and there is more of it than there are people doing it.

This is not a product. There is no support, no release schedule, and no claim that any title works
beyond the specific route documented for it.

## How the work is organised

- [`CLAUDE.md`](CLAUDE.md) is the project charter: scope, build and run recipes, evidence rules, and
  the recorded mistakes that have cost the most time. It is read before starting work and assumes no
  memory of previous sessions.
- Progress is defined by a six-rung ladder per title. A game loop running over a black screen is not
  gameplay, and a build that compiles is not progress.
- Verification is programmatic ([`docs/VERIFICATION.md`](prosper/docs/VERIFICATION.md)): Vulkan
  tests that assert numeric and pixel results, a `spirv-val` gate on every SPIR-V emitter, and
  golden-image guards over real boots.
- Falsified hypotheses are recorded in a `## Ruled out` section in the relevant status document.
  Measurements that turned out to come from the apparatus rather than the subject are recorded as
  numbered instrument traps.
- Questions that existing tools cannot answer are answered by writing a tool, under
  [`tools/`](prosper/tools/AGENTS.md).
- GitHub issues hold work that spans sessions. Concurrent agents claim an issue before starting and
  work in separate git worktrees. Non-obvious changes get an independent review before merge.

Most of the costly failures so far have not been in generated code, but in bookkeeping and
inference. A retracted figure left standing in the charter kept being used for a day after it was
withdrawn. A review finding with a plausible `file:line` citation was repeated across many sessions
without anyone opening the file. A positive control passed on a case it could not have detected.
Each of those is now a rule in `CLAUDE.md`.

## Scope

No game files, keys or Sony code are in this repository. You supply your own legally obtained dump.
prosper holds no console keys, performs no decryption, and operates only on module segments that are
already unencrypted. It also refuses to link the third-party replacements for Sony libraries that
some dumps carry, since loading one would perform the circumvention by proxy. Entitlement and
add-content queries are answered from the content present on disk rather than approved
unconditionally. Sony's library interfaces are reimplemented from published symbol and NID data,
clean-room. The project is independent and not affiliated with Sony Interactive Entertainment.

## Where things are

- [`BUILDING.md`](BUILDING.md) — dependencies, build and test commands, launching a dump.
- [`prosper/README.md`](prosper/README.md) — the source tree and its architecture documents.
- [`CLAUDE.md`](CLAUDE.md) — the charter.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — what makes an external contribution reviewable.
- [`COMPATIBILITY.md`](COMPATIBILITY.md) — per-title results.

No `LICENSE` file exists yet, so no license is granted by default. Open an issue to ask.
