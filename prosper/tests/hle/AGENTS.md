# Guest-facing HLE contract tests

These fixtures invoke registered guest APIs and check return values, output writes, and observable
state changes. Subdirectories group service families; shared fixtures live in `tests/support/`.
Host implementation helpers belong to `tests/host/`, and GPU execution belongs to `tests/gpu/`.

Memory tests use real mappings and preserve sentinel bytes across refused operations. Explicit
addresses, search hints, and automatically chosen addresses have different contracts; test the
requested behavior and its ownership preconditions instead of treating them as interchangeable.

Existing TLS-module, DTV-recycle and exit-registry fixtures exercise `kernel_tls.cpp` through
the registered `__tls_get_addr` entry and public publication/purge functions. The original
destruction canary and immortal-registry checks retain their lifetime contracts.
