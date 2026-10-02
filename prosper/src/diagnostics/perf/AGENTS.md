# Performance observation

The always-on frame ledger, pure alarm rules and window engine live here. The ledger accumulates
coarse costs and event counters; the rules interpret one window without I/O; the engine handles
sustain, logs and summaries. These observers must not change guest or renderer behavior.

Reuse existing counters at window close before adding event hooks. Rule tests construct both a
failing population and an independent healthy control; an absent signal is NO DATA, not quiet.

`rtt-colorless-publication` observes only slot-0 CPU pass-readback publications in the per-target
loop. It counts an actual publication without a colour writer; legitimate extent aliases and
prevented depth-only candidates are quiet. No evaluated candidates or violations means NO DATA.
Other RTT writers (compute snapshots, resolve copies, materialization) do not use this hook.
See the parent `AGENTS.md` for each rule's meaning and the evidence behind its thresholds.

`unsupported-wave64-shaders` (#3992) is platform-independent. It observes known guest Wave64
fragment/compute translation failures and refused backend subgroup contracts; native Wave64
availability is not an alarm-enable gate. The unit is repeated refused shader uses, not unique
programs or dropped draws. `wave64_refusal.hpp` separately announces bounded refusal identities
by default, with stage/program/site and available host sizes/reason bits. The failure-only identity
inventory takes a lock but reads no guest memory; accepted uses only add a coarse observation
counter. Inventory saturation and unidentified uses remain explicit. No known-Wave64 observations
is NO DATA, not evidence that every shader is supported. Observation counts cover realization and
backend boundaries and must not be called distinct shader counts.

Fragment subgroup refusals additionally retain the actual lowering verdict and, for an unproved
vote, the first failed vote's SOURCE SPIR-V dword offset/result/predicate IDs and defining opcode.
This is not a guest instruction PC or proof of a varying/undefined predicate. Not-attempted and
unavailable vote metadata remain explicit; the existing bounded announcement inventory is reused.

Unplumbed vertex guest wave width is deliberately not inferred from the translator's Wave64
default; its failures remain covered by `dropped-draws`. The known-width scope does not mean all
Wave64 gaps have been implemented or detected. Capture re-realization suppression applies to
both observations and refusals; deliberate compute selectors decline before the backend hook.

`unverified-fragment-f32-arithmetic` (#4062) announces the source-confirmed remaining #4059
lowering gap, not a measured GPU failure. It counts direct/cached fragment compiler requests
that actually reached ordinary guest F32 ADD/MUL emission, including a later refusal. Cold
requests transfer immutable bounded site/mode provenance to the cache; warm requests replay it
with their current lookup address. It is not a draw or execution counter, complete arithmetic
inventory, unique-shader census or permission to infer unknown launch mode as mode0. Fixed site
prefix truncation and announcement-inventory saturation are explicit; repeated request accounting
continues. No observed requests means NO DATA; requests without emitted ADD/MUL are quiet within
this limited inventory. F9 re-realization suppression applies; ordinary offline direct compiles
still announce before flip windows. Shader words, compile keys and numeric/admission policy do not
depend on these observations. Completing #4059 is separate from announcing it.

`texture-direct-validation` is an observer without a new alarm threshold. Its seven counters cover
only actual retained-pixel/complete-encoded-prefix comparison invocations at the live image site;
`memcmp` argument extents include the first differing chunk but do not measure physical traffic.
Completed-window JSONL/exit reporting excludes boot and trailing partial data. Native mapped-source
checks establish helper/ledger/window behavior; live cold/changed/warm, scratch and write-watch
checks separately constrain the caller's publication scope and unchanged pixels/cache policy.
