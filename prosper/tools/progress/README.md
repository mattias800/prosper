# HLE progress census

Static coverage report per `src/hle/<area>/`, counted per **registered Sony NID**: how many get a
real handler and how many only a placeholder. No game dump, no build, no GPU, so it runs on any host.

## What the numbers mean

- **total**: distinct NIDs that prosper's source registers a handler for on the chosen
  `--platform`. This is the registered surface, not the PS5 firmware surface: an import nobody
  registers falls to the dispatcher's unimplemented-import path and is invisible here.
- **done**: a NID that at least one registration installs through a non-placeholder API
  (`register_fn`, `register_typed`, `register_guest_abi`, ...). It says a real handler is installed,
  not that the handler is correct; a body that only returns `ENOSYS` still counts as done.
- **todo** (printed as *placeholder-only*): a NID whose every registration is
  `Hle::register_placeholder`, the overridable tracing thunk (`src/hle/dispatch/dispatch.hpp`).
  A real registration in any file beats a placeholder: `register_placeholder` records a shadow when
  it overwrites a real entry, and ctest `test_hle_no_shadow` fails on any shadow at boot.

The registrations come from the parser in `tools/re/hle_handler_map.py`, which finds direct
`Hle::register_*` calls, file-local wrapper macros and lambdas, and evaluates `#if` arms per
platform. This tool adds two resolvers for NIDs that are not literals: a constant table
(`kUlt[kIdxInitialize].nid`) and a `register_placeholder` fold over a `std::index_sequence`, which
counts one NID per pack element.

## Usage

```bash
python3 prosper/tools/progress/progress.py --root . --platform linux
python3 prosper/tools/progress/progress.py --root . --platform linux --output ~/progress/head
python3 prosper/tools/progress/progress.py --compare ~/progress/base/progress.json ~/progress/head/progress.json
```

`--root` is the checkout root (the directory containing `prosper/`). `--platform` is required for a
census, because registrations differ per platform. `--output` is a directory that receives
`progress.json`. `--compare` prints a markdown delta keyed by NID and writes nothing.

Exit status: 0 when the census is complete, 3 when a registration site was unclaimed or a NID
expression stayed unresolved (both are listed in the header, and every count is then a lower
bound), and 2 when the scan was refused (no `dispatch.hpp`, or no sources).

## Cross-check

The header prints `NIDs from literal sites`, which must equal `distinct NIDs registered` from
`python3 prosper/tools/re/hle_handler_map.py --platform linux`. The census total is that number
plus the NIDs the two resolvers add. The self-test `test_real_tree_agrees_with_hle_handler_map` pins
that identity on the real tree.

Runtime `if` guards are not evaluated, the same as in `hle_handler_map.py`. The census therefore
includes registrations that only run under a diagnostic switch. Today that is two AGC tracer NIDs
behind `PROSPER_AGC_REG_TRACE`, which a compiled `hle_registry_dump` does not list.

## What this does not cover

- **Unregistered NIDs** (imports that fall to `prosper_on_unimpl` at runtime). Use `nid_census`
  with game modules.
- **Live traffic** (which handlers a title actually calls). Use `hle_calls`.
- **Whether a handler is correct.** A real registration whose body is a stub counts as done.
- **Shader ISA coverage** and badges. Deferred.
