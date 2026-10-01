# Linked build identity

Checks and build-time generators for immutable revision/source provenance. A running binary must
report what was linked, not whichever checkout is inspected later. Compiler dependency identity is
conservative and fail-closed on unsupported command/dependency syntax. Public reports contain only
digests; external dependency paths belong solely to the local build-time calculation.
