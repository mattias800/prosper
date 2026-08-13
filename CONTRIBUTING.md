# Contributing to prosper

prosper is a PS5→PC compatibility layer — the same class of project as Wine, Proton, RPCS3 or DXVK.
External contributions are welcome. This page exists so a good one is not rejected for a mechanical
reason, and so you do not spend hours on something that cannot be reviewed.

Read `CLAUDE.md` first if you plan to touch anything beyond a small fix. It is the project charter
and it is binding.

## The one idea behind all of the rules

**A contribution is worth what a reviewer can check.** Not what it claims, not how much of it there
is — what can be independently confirmed. Everything below follows from that, and it is why a
200-line change with a failing-without-the-fix test lands easily while a 4,000-line one that compiles
in isolation does not land at all.

This cuts both ways, and the good direction is the important one: a small change carrying its own
evidence needs very little from us, and usually merges quickly.

## Enforced by CI

The `Contribution shape` job fails on two things. Both are mechanical, both are fast, and both exist
because they cost real review time before they were automated.

**1. Every new file lives under `prosper/`.**

The build collects exactly one set of sources:

```cmake
file(GLOB_RECURSE PROSPER_SRC CONFIGURE_DEPENDS src/*.cpp)   # relative to prosper/
target_include_directories(prosper_core PUBLIC src)          # include root is prosper/src
```

A file added at the repository root — `core/thing.hpp`, `plugins/thing.hpp` — is globbed by nothing,
on no include path, and referenced by nothing. It compiles nowhere. **Every CI job then reports
success, because each one built prosper and prosper does not contain your file.** Green means the
code was absent, not that it was correct.

Diagnostics belong in `prosper/src/diagnostics/`. Tests belong in `prosper/tests/`.

**2. A new `.cpp` under `prosper/src/` requires touching something under `prosper/tests/`.**

Because of the glob, a new source file needs no build-system change — so it can arrive fully compiled
and completely unexercised, where "it compiles" is the only claim anyone can check. Modifying an
existing file does not trip this; only adding a new one does.

## Judged by review

**One change per pull request, and prefer the smallest version that works.** A 1,200-line header
cannot be usefully reviewed: the reader cannot hold it, so the findings you get back will be shallow
exactly when the change most needs deep ones. If you have a subsystem in mind, land its core first
and build on it.

**Wire it to something.** An observer that nothing calls cannot be shown to observe anything, and a
facility with no caller hides its own defects — an incompatible signature, a broken disabled-path
stub, a state that can never be reached. One real call site surfaces more than any amount of
self-contained code.

**Rebase before you submit.** This repository moves fast, with several agents and a maintainer landing
work daily. Check that what you are adding does not already exist: a class re-declared in a namespace
that already has one is a redefinition, not an addition.

**A test should fail without your change.** If it passes either way, it documents the code rather than
constraining it. Say in the PR body what you did to confirm this — reverting the fix and watching the
test go red is the whole check.

**Claims need the command that reproduces them.** Numbers are welcome and useful; unverifiable numbers
are worse than none, because they cost a reviewer the time to discover they cannot be checked. If you
report a measurement, give the exact invocation and attach the artefact. `-fsyntax-only` proves a
header parses; it is not a build, and should not be described as one.

Be careful with clean zeros in particular — "0 failures", "0 divergences", "100% match". Before
reporting one, construct a failing case by hand and confirm your harness reports it. Until that arm
exists, *zero* and *cannot detect* are the same number.

## What you cannot verify, and that is fine

You will not have a PS5 game dump, so you cannot run the boot tests or the snapshot guards. That is
expected and it is not a barrier — say what you could not check rather than implying you did. Plenty
of valuable work needs no dump at all: host-side tooling, the loader and SELF/ELF paths, tests,
diagnostics, build and packaging, documentation.

## What will be rejected

- Entitlement or add-content APIs made to return "owned" unconditionally. prosper answers those from
  local inventory. See `CLAUDE.md` — this one is not negotiable and is not a style preference.
- Code copied from other emulators. Behaviour must be re-derived in prosper's own architecture.
- Shims that fake output to make something appear to work.
- Anything containing Sony code, firmware or keys.

## A worked example

PR #2495 is the template, and it is worth reading before you open anything substantial. It arrived as
6,428 lines compiled by a glob with no call sites and no tests, and was rejected. It came back as 821
lines: opt-in behind a CMake option, zero-cost stubs when disabled, seven real call sites in
`boot_program.cpp`, three tests. It merged.

The subject did not change. The evidence did.

## Practical notes

- Build: `cmake -S prosper -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug && cmake --build build`
- Test: `ctest --test-dir build --no-tests=error` — quote the **count** alongside the exit code.
  Plain `ctest` exits 0 when it finds no tests at all, so "nothing ran" and "everything passed" look
  identical without that flag.
- Do not put absolute paths from your machine in commits, PR text or committed files. Use
  `<REPO_ROOT>`, `<DUMP_ROOT>`, `<WORKTREE>`.
- Open an issue first for anything large. It is much cheaper to hear "that subsystem already exists"
  before you write it than after.
