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

Unplumbed vertex guest wave width is deliberately not inferred from the translator's Wave64
default; its failures remain covered by `dropped-draws`. The known-width scope does not mean all
Wave64 gaps have been implemented or detected. Capture re-realization suppression applies to
both observations and refusals; deliberate compute selectors decline before the backend hook.
