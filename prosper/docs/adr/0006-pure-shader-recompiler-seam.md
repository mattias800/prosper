---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0006: Make the shader recompiler a pure function behind one entry point

## Context

`src/gpu/recompiler/` exposes a family of entry points (`recompile_vertex`, `recompile_fragment`,
`recompile_compute`, `recompile_fragment_packet`, `recompile_vertex_chain`, plus several
`*_for_test` variants in `rdna2_to_spirv.hpp`), and its behaviour depends on more than its inputs.
Measured on 61557f29 with `grep`:

- 11 files under `src/gpu/recompiler/` (subfolders included) read environment switches, counting
  files with a live, non-comment `getenv`, `PROSPER_ENV_ON`/`PROSPER_ENV_VALUE`, `env_u64_or*`,
  `env_cache` or `env_submit` use (a raw grep gives 12, the twelfth a comment in
  `rdna2_to_spirv.hpp`). So the same bytes can recompile differently depending on the shell.
- `raster_quad_collector.hpp` and `rdna2_raw_x2_lifetime.cpp` include `gpu/execute/`, and two files
  include `gpu/state/` facts -- the recompiler reaches up into the code that calls it.
- 11 lines include `gpu/diagnostics/diagnostic_selectors.hpp`.

The cache is already outside it (`recompile_graphics_shader_cached_shared` in
`gpu/execute/gpu_execute.hpp`), which is the right placement. AnyPS5, the sibling PS5 project compared in the
architecture review, exposes one `Recompile(request) -> result` entry point and documents the
recompiler as isolated and a pure function of its input; its own CMake target still links its cache
and runtime, so that is a direction to take, not a finished example to copy.

## Decision

Spec `GPU-1`. The recompiler is a pure function: shader bytes, the register state it reads, and a
target description (subgroup size, enabled features, float-controls support) in; SPIR-V and a
binding layout out. It reads no switch, owns no cache or device, and includes nothing from
`gpu/execute`, `gpu/state`, the backend or a frontend. Diagnostic selectors that change the output
become fields of the request, so a recorded request reproduces its result. Caching and logging wrap
it from outside. Migrate one entry point at a time; each migration is checked by recompiling a
capsule corpus before and after and comparing the SPIR-V byte for byte.

## Consequences

A recompile becomes reproducible from a recorded request, which is what ADR 0005's replay and any
future offline shader regression need, and per-instruction tests stop depending on process state.
Once it is pure, `gpu/recompiler` can be its own CMake target (ADR 0008) with an include set the
build enforces. Switches that today change recompiler output are guest-behaviour selectors under
`CFG-1` and each needs its issue.

## Alternatives considered

- Leave the entry points and only add a facade: rejected, a facade over impure code reproduces
  nothing.
- Move the cache inside the recompiler as AnyPS5 does: rejected, the cache's key is a
  policy of the executor, and keeping it outside is what makes the function pure.

## Approval

Requires the project owner's acceptance; the migration touches the recompiler's public header,
which every graphics lane includes.
