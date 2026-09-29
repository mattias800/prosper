# Performance observation

The always-on frame ledger, pure alarm rules and window engine live here. The ledger accumulates
coarse costs and event counters; the rules interpret one window without I/O; the engine handles
sustain, logs and summaries. These observers must not change guest or renderer behavior.

Reuse existing counters at window close before adding event hooks. Rule tests construct both a
failing population and an independent healthy control; an absent signal is NO DATA, not quiet.
See the parent `AGENTS.md` for each rule's meaning and the evidence behind its thresholds.
