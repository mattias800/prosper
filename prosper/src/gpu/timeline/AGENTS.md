# `timeline` — frame timelines and bundle selection

`gpu_timeline` — the frame's operation sequence, and the signatures used to select an operation
across runs.

**Addresses and operation ordinals are run-local.** A historical capture hash is not a current
renderer oracle, and neither is a target extent or a raw draw count on its own. Re-derive the selector
with `gpu_timeline --signatures` / `--select` before recording a new bundle; a selector that silently
matches nothing produces an empty result that reads exactly like a negative finding.
