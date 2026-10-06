# `tools/env/` — the PROSPER_* switch surface

Everything here is about the ~850 `PROSPER_*` environment variables prosper reads: whether a switch
can safely be cached, and whether a diagnostic can print a field it never measured. All three
scripts are gates that run in CI; nothing here is part of the shipped runtime and nothing here
changes behaviour. `diag_gate_baseline.txt` is one gate's classified findings, not a fourth tool.

The boundary against its siblings: `tools/doctor/` asks whether the *machine* can profile,
`tools/perf/` reads a capture prosper produced, and `tools/getenv_probe/` measures which switches
are actually hot. This folder is about the switch surface itself.

## The gates (the first three run under ctest; the registry runs in the Docs CI job)

- **`check_cached_env.py`** — refuses to cache a name something arms at runtime. A cached read is
  sampled once at first use, so a test that arms a diagnostic and then asserts on it does not fail
  when the read is cached: it goes **vacuous** and keeps printing `[ok]`. Run this before converting
  any `getenv` to `PROSPER_ENV_ON` / `PROSPER_ENV_VALUE`. The macros, and the reasoning behind the
  whole scheme, live in `src/diagnostics/env_cache.hpp`. Its only exception list, `CACHED_ONCE_PINS`,
  is for a test whose assertion is that the arm is NOT observed (#3892's registration-statics
  harness). Each entry is one file and one name with a reason, and a stale entry fails the gate.
- **`check_diag_gates.py`** + `diag_gate_baseline.txt` — finds a diagnostic whose *producer* and
  whose *printer* are armed by different switches, so arming the one whose name matches your
  question yields a zero that reads as data. The baseline is a classification, not a suppression
  list: a row has to carry a mechanism at a `file:line` the next reader can open. Its narrow cached
  reference-getter bridge preserves config provenance through an identified singleton receiver;
  cross-file positive, bridge-off and adversarial controls run with `--selftest`.
- **`check_switch_registry.py`** + `switch_registry.txt` -- the list of every `PROSPER_*` name the
  shipping code mentions, each with the charter's class: `host-capability`, `diagnostic`, or
  `selector` (with the `#issue` that will settle its default and delete it). Names that predate the
  registry are `unclassified` and grandfathered; that set may only shrink, and a new switch must be
  classified in the PR that adds it. Classify one by editing its row; `--update` adds new names and
  drops removed ones. It checks that a class exists, never that it is true. Runs in the `Docs` CI
  job; tests are `test_check_switch_registry.py` (pytest).
- **`member_fact_domains.py`** + `MEMBER_FACT_DOMAINS.md` — explicit TEST_LOCAL source-role
  declarations partition both member declaration and initializer facts without excluding files.
  Every header and uncertain/refused source remains shared; test TUs retain the original all-file
  member table. Literal CMake/include safeguards do not infer complete build reachability.
  `test_member_fact_domains.py` runs its admission, refusal and visibility controls inside the
  existing diagnostic gate selftests and full scans.
- **`ref_output_associations.py`** + `REFERENCE_OUTPUTS.md` — explicit private-function/context
  contracts retain selected lexical producer/report associations across reference outputs.
  Exact callee aliases and caller object/local/call identities are required; admitted effects
  conjoin caller and callee clauses, while unsupported or escaping contracts visibly refuse.
  This does not infer general C++ effects. `test_ref_output_associations.py` calibrates bridge-off,
  gate changes, no-write paths, unrelated fields and refusal/visibility inside the same gate.
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

## Ruled out

- **A byte-preserving function split does not preserve a local producer/report association.**
  The #3892 image extraction initially retained the complete source by reconstruction, but the
  unmodified scanner fell from 122 findings/65 keys to 114 findings/57 keys. Eight defaulted
  caller scalars still printed their original values; their guarded writes moved into context
  reference aliases in the callee. Four other report keys changed only pathname. A declared,
  per-object/per-call output association is needed before treating those eight classifications
  as preserved; deleting their stale baseline rows would hide unchanged diagnostic gates.

- **A source-preserving config relocation does not preserve a lexical scanner's visibility by
  itself.** On #3892's process-owner candidate `df8439ff5543`, the unmodified diagnostic scanner
  produced 2 new and 12 stale findings after cached values became reference-getter calls. The
  producer/printer gates had not changed; their provenance disappeared at the accessor boundary.
  A receiver-aware bridge restores all 122 findings on both main and that candidate with the
  same 65 classified baseline keys. Removing the stale rows would conceal the unchanged gates.
- **A reference getter is not a global scalar predicate.** The mutable-reference negative in the
  getter controls initially still found a gate because the old `bool|int` return matcher accepted
  `int&`. That joined the method's local config to every same-named identifier, bypassing receiver
  checks. Scalar predicates now exclude pointer/reference returns; cached const-reference values
  use the identified receiver, and unknown or shadowed receivers contribute no alias (#3892).
  Independent review then caught alternate CV spellings (`int const&`, `bool const*`) bypassing
  an immediate pointer/reference lookahead. Both cross-file negatives reproduced that leakage;
  matching the complete plain scalar return/name prefix rejects them while preserving the tree's
  existing findings.
