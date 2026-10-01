# `tests/hle/kernel` — guest kernel contracts

Regressions for the kernel HLE's guest ABI, error encodings, clocks, and synchronization behavior.
Exercise entry points through the HLE registry when registration and guest-visible results are part
of the contract. Keep acquisition, timeout, cancellation, and object-lifetime assertions here with
bounded waits so a synchronization failure can report a test failure.

Host primitive and platform implementation tests belong under `tests/host`; GPU command execution
and rendered-output tests belong under `tests/gpu`. These tests may supply GPU timing observations
to check kernel accounting, but do not establish graphics execution or visual correctness.
