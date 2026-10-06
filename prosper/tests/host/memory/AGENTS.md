# Host guest-memory tests

These fixtures exercise the host's mapping/readability caches, direct-memory alias registries,
write-watch fault handling, and memory search. They model actual mapped pages and mapping lifetimes;
guest-facing allocation/API behavior belongs to HLE tests, and GPU cache publication belongs to
the live compute execution fixtures. Linux write-watch tests need a real alternate-stack fault
handler so direct stores test dirty tracking rather than an explicit-notification substitute.

`test_windows_guest_window` is the odd one out: its subject is the test PROCESS's own address
space — where the host's memory sits relative to the guest's range on Windows (#4426) — so it is
linked the way a guest-hosting executable is (`prosper_hosts_a_guest`), and each of its cases needs
a fresh process. ctest gives it one; do not add a case there that assumes a pristine address space
after another has run.
