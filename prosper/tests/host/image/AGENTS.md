# Host image fixtures

These tests cover the platform substrate that maps modules, publishes import tables, and enters
synthetic guest code. Format parsing belongs in `tests/self` or `tests/loader`; guest-facing HLE
semantics belong in `tests/hle`.

Use owned mappings and synthetic handlers for publication and recovery checks. A refused mapping
or emission must preserve the occupant's bytes and leave an earlier valid table usable. Any syscall
interception must match the fixture's owned request and forward unrelated requests unchanged.

`test_hle_stack_args` owns the generated import-stub ABI and completion-lifetime witness.
Its assembly-visible storage has one global C definition; the remaining fixture state stays
private, and both ordinary and guest-FS entry paths retain the same assertions.
