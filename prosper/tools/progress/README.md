# HLE progress census

Static coverage report per `hle/<area>/`: implemented handlers vs placeholder stubs.
No game dump, no build, no GPU — pure source scan, so it runs on any host.

## What done/todo mean

- **done**: an `HLE(name)` body that is a real implementation — it does not call
  `prosper_on_unimpl` and is not only reachable through `register_placeholder`.
- **todo**: an `HLE(name)` whose body calls `prosper_on_unimpl`, or whose handler is
  only ever installed via `Hle::register_placeholder` (a diagnostic/tracing thunk a
  real handler is expected to replace, see `prosper/src/hle/dispatch/dispatch.hpp`).
  A handler implemented in one file but placeholder-registered in another counts as
  done — the real registration wins, same as the runtime override rule.
- **total**: done + todo over handlers *declared in the tree*. This is not the PS5
  firmware surface. An import no title's module declares is invisible here by design.

## Usage

```bash
python3 prosper/tools/progress/progress.py --root . --output /tmp/progress
python3 prosper/tools/progress/progress.py --compare /tmp/base.json /tmp/head.json
```

`--root` is the checkout root (the directory containing `prosper/`). `--output` is a
directory receiving `progress.json`. Without `--output`, prints the text summary to
stdout and writes nothing. `--compare` prints a markdown delta and writes nothing.

## Known v1 gaps

- An `HLE()` body that only returns `ENOSYS`/zero without calling
  `prosper_on_unimpl` counts as done. `s_psml_unimplemented`-shaped stubs are the
  known instance. A follow-up classifies single-return bodies against their
  contract; v1 deliberately does not guess.
- `register_fn` macro wrappers (`R("name", fn)`) are not parsed for handler names;
  classification keys on `HLE()` bodies and `register_placeholder`, so a file
  using only macros contributes its bodies but no placeholder rows.

## What this does not cover

- **Unregistered NIDs** (imports that fall to `prosper_on_unimpl` at runtime).
  Static-declared coverage cannot see them. Use `nid_census` with game modules.
- **Live traffic** (which handlers a title actually calls). Use `hle_calls`.
- **Shader ISA coverage**. Deferred: prosper has no single opcode enum to scan the
  way a decoder table allows. A follow-up maps the ISA list against
  `rdna2_decode` + `alu_support` + `cfg_support`.
- **Badges/SVG treemap**. Deferred until the numbers are trusted; JSON is the gate.

## Reading a clean zero

`total=0` for a group means "no handlers declared there", never "complete". Before
quoting `todo=0`, add one hand-built placeholder fixture outside the tree and confirm
this reports it — the self-test `test_placeholder_detected` is that control.
