# Fragment compile replay

Record once with `PROSPER_FRAGMENT_COMPILE_CASE_DIR=<PRIVATE_CASE_DIR>` before the shader's producing
compile. Each `.prfc` is one bounded, versioned compiler attempt, including full immutable code
(also trailing retained data), full resource/interface metadata, selected width/PC-relative target,
pinned trip settings, actual consumed semantic choices, immutable linked compiler identity, and
expected complete SOURCE or refusal. It is not a frame capture.
The actual producing key's complete guest FLOAT_MODE byte and availability are retained separately
from wave width and host float-control publication. Unknown stays unknown; replay never infers it.
Schema 3 appends the actual pinned host float-transport profile after the unchanged schema-2
owned-marker tail. An explicitly producing `unknown` profile is a retained input and remains
replayable; schema-1/2 absence is instead inspectable
`INCOMPLETE fragment-transport-config-unavailable` and cannot enter the compiler. Neither profile
nor SOURCE grants feature permission to a different executing Vulkan device.

Schema 4 appends independently observed guest IEEE_MODE/DX10_CLAMP availability and values after
the schema-3 host profile. An explicitly producing Unknown flag state is retained, not inferred
from FLOAT_MODE or transport. Genuine schema-1/2/3 files remain inspectable; absent launch flags
make replay INCOMPLETE (`fragment-float-flags-unavailable`, unless an earlier reason is retained).
Frame capture v67 carries the same independent flag state for realized and failed draws.
Both schema4 and v67 also retain the exact observed RSRC1_PS word with separate availability.
It is producing evidence only, never a fallback for missing MODE/IEEE/DX10 or implicit FP16_OVFL
authority, and is not passed as an unused numerical compiler argument. The actual key includes
the word so a warm case cannot report another launch's otherwise unconsumed register bits.
These flags do not describe FP16_OVFL, do not authorize host float features, and do not establish
general guest arithmetic, half packing, or NaN-payload fidelity.
This retention groundwork does not change emitted arithmetic. The numerical obligations in
#4086/#4101/#4059/#4096 remain open; legacy operations are not renamed as semantic support.

```text
fragment_compile_replay --inspect-only <CASE.prfc>
fragment_compile_replay --baseline <CASE.prfc> --output <BASELINE.spv>
fragment_compile_replay --candidate <CASE.prfc> --output <CANDIDATE.spv>
spirv-val --target-env vulkan1.1 <CANDIDATE.spv>
```

Baseline requires the same linked source/configuration identity and compares every output word,
or reproduces the recorded refusal diagnostic. Candidate uses the captured inputs and semantic
read transcript with the new compiler; a new uncaptured read refuses rather than using today's
environment. Exit 3 means a complete candidate still refused; no empty SPIR-V file is installed.
Exit 2 means invalid/incomplete input or replay divergence. `--compiler-identity` reports the
immutable identity built into this tool, not a digest of files changed after it was linked.

Producing cases are retained in shader-cache entries and included in the existing byte budget.
Warm exports use only that retained context. Entries compiled before recording was armed write
explicit `INCOMPLETE producing-context-not-retained` records; they never claim current environment
or resource metadata as old producer inputs. The record distinguishes lookup and producing addresses;
compact filenames retain the lookup address and only a content-hash hint.
Both byte-exact case identity and destination participate in bounded deduplication.

Ordinary image/DCC pixel content is not a compiler input and is not captured. Its pointer presence,
allocation membership, exact pointer aliases and range metadata remain typed opaque facts, not dummy readable
pointers. Any byte request without an owned validated span fails. Special live-revalidated carriers
are explicitly incomplete in schema 1. No logical guest address is dereferenced during replay.
Codec validation and independent producing baseline must pass before a record is complete. An
actually refused compiler attempt can have complete inputs, even with no semantic reads.

Unowned ordinary buffer backing is also opaque when the invocation needs metadata only. Owned
buffer allocations are retained within the aggregate budget; any actual byte consumer of opaque
backing refuses. This is not permission to reinterpret a guest address or invent readable storage.

Schema 2 appends each resource's owned raw-snapshot byte marker after the schema-1 payload. Both
valid and malformed markers survive the codec so candidate replay retains the producing compiler's
admission or refusal inputs. Schema-1 files remain readable with the historical zero marker.
Code-derived draw capture requirements remain independently derived from the restored shader.
Raw-load SOURCE admission uses the recorded backing-presence fact and exact resource shape;
readable bytes still require the checked resource gateway. Frame replay and draw uploads retain
their separate backing-ownership checks.

Schema 5 appends a separate exact-PC parent/child snapshot marker tail after the unchanged
schema-4 launch context and schema-3 producing profile. Schemas 1 through 4 remain readable with zero nested markers; they cannot supply new
nested admission. Valid and malformed markers are retained exactly so baseline/candidate replay
preserves SOURCE or refusal. The complete original shader independently supplies the one-hop,
width, pointer-selection and no-writer proof. Recorded presence supports SOURCE metadata checks;
actual byte reads still require checked owned storage. These compiler cases do not authenticate
live physical allocation origins or producer completion.

The linked identity covers dirty/untracked project sources, actual GCC/Clang dependency headers,
compiler/front-end bytes and resolved compile commands/configuration. Unsupported dependency
syntax or unavailable compile commands produce an unknown identity, not a complete case. This is
a compiler-input fingerprint, not a claim to fingerprint every external runtime library.
The opt-in native-Clang Windows profile also requires the actual secondary GNU production argv
manifest, its compiler/front-end bytes and transitive headers. Its build-local manifest is never
exported; missing/malformed producer context makes the linked identity unknown. Test-only producer
commands are not claimed as production compiler inputs.

Replay is a debugging aid, not a guard: it reproduces a captured compile offline so a recompiler
change can be iterated on a frozen input. Compiler code should keep routing byte reads through
`compiler_resource_data` and semantic choices through `compiler_choice`, because a read that bypasses
them is invisible to the capture and makes a replay diverge from the live compile. A divergence is
then the thing to investigate, not a reason to distrust the tool. (Until 2026-10-02 a hash inventory
of all `prosper/src` files, `reader_policy.json`, gated replay and a ctest on every source edit; it
was removed because it could not detect a new reader, only that some file changed.)

These files contain private shader bytes. Do not commit or publish them. SOURCE reproduction does
not establish buffer immutability, Wave64 admission, draw execution, or pixel correctness.
