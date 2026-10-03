# Host image fixtures

These tests cover the platform substrate that maps modules, publishes import tables, and enters
synthetic guest code. Format parsing belongs in `tests/self` or `tests/loader`; guest-facing HLE
semantics belong in `tests/hle`.

Use owned mappings and synthetic handlers for publication and recovery checks. A refused mapping
or emission must preserve the occupant's bytes and leave an earlier valid table usable. Any syscall
interception must match the fixture's owned request and forward unrelated requests unchanged.
