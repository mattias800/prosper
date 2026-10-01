# Diagnostic controls

These fixtures check observer/report contracts: signal validity, counters, summaries, optional
state and the distinction between an observed zero and absent coverage. Tests of a pure sampler
or formatter do not certify its production call route. Keep coarse event populations and the
actual report path in controls, with discriminating mutations for missing or substituted signals.

## Ruled out

- Waiting past the deferred timeout after initial packet folding does not by itself age a barrier.
  The first actual deferred recheck records its blocked timestamp. The first packet-route calibration
  left the readable unsupported barrier queued after a 1100 ms sleep taken before that recheck.
  Arm the recheck before measuring the existing timeout; keep local labels alive until it drains.
- Disjoint host-stack addresses do not prove disjoint guest physical topology. The first
  packet-route calibration's retained-effect positive instead took the existing fail-closed
  unknown-topology rejection. Use real kernel direct-memory registration and assert the production
  topology result before claiming that a retained-effect overlay acceptance was reached.
- These multi-mode fixtures do not satisfy the diagnostic member-domain contract's single
  registered-test form. Their invalid TEST_LOCAL annotations were refused and retained shared
  member facts. Keep them unannotated in the shared domain, with every ordinary and disabled mode
  intact, rather than weakening that contract or claiming test isolation that was never admitted.
