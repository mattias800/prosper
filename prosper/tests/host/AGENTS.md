# Host substrate tests

These fixtures exercise the host facilities that carry guest execution: image mappings, TLS,
platform lifecycle, fault handling, and memory observation. Subdirectories follow those ownership
boundaries; `memory/` tests mapped-page lifetimes and caches, while guest-facing memory API tests
belong to `tests/hle/`. The historical `test_dmem.cpp` also drives real memory HLE mappings and
physical aliases, so its placement and content-preservation checks must exercise both host arms.

Use actual mapped pages when testing mapping ownership. A refused overlap must preserve the
occupant's bytes; an address alone does not prove that no mapping was replaced.
