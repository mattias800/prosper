# `tools/env/` — the PROSPER_* switch surface

Everything here is about the ~850 `PROSPER_*` environment variables prosper reads: whether a switch
can safely be cached, and whether a diagnostic can print a field it never measured. All three
scripts are gates that run in CI; nothing here is part of the shipped runtime and nothing here
changes behaviour. `diag_gate_baseline.txt` is one gate's classified findings, not a fourth tool.

The boundary against its siblings: `tools/doctor/` asks whether the *machine* can profile,
`tools/perf/` reads a capture prosper produced, and `tools/getenv_probe/` measures which switches
are actually hot. This folder is about the switch surface itself.

## The three gates (all run under ctest)

- **`check_cached_env.py`** — refuses to cache a name something arms at runtime. A cached read is
  sampled once at first use, so a test that arms a diagnostic and then asserts on it does not fail
  when the read is cached: it goes **vacuous** and keeps printing `[ok]`. Run this before converting
  any `getenv` to `PROSPER_ENV_ON` / `PROSPER_ENV_VALUE`. The macros, and the reasoning behind the
  whole scheme, live in `src/diagnostics/env_cache.hpp`.
- **`check_diag_gates.py`** + `diag_gate_baseline.txt` — finds a diagnostic whose *producer* and
  whose *printer* are armed by different switches, so arming the one whose name matches your
  question yields a zero that reads as data. The baseline is a classification, not a suppression
  list: a row has to carry a mechanism at a `file:line` the next reader can open.
- **`check_env_numeric_arms.py`** — every knob parsed through `diagnostics/env_numeric.hpp` has a
  matching arm in `tests/diagnostics/test_env_numeric_sites.cpp`, and vice versa. Run it from
  `prosper/`, not the checkout root.

## The measurement that comes first

The gates say whether a name MAY be cached. What they cannot say is which names a real route
actually evaluates millions of times — and that question lives one directory over, in
**`tools/getenv_probe/`**, whose `PROSPER_GETENV_PROBE_NAMES` mode counts `getenv` calls by name.
It is a measurement shim rather than a gate, which is why it lives with the other profiling tools
and not here.

## The rule this folder exists to enforce

Caching a switch is a semantic change, and the direction it fails in is silence rather than noise.
So the order is always: **census first** (which name is hot), **gate second** (may that name be
cached at all), conversion third. Picking sites by reading the source instead measures nothing and
has, historically, converted names no route ever reached.
