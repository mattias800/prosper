# tools/memcpy_probe

One `LD_PRELOAD` instrument that answers "which code is doing all this copying?" when `perf` cannot
— which on this machine is always, for libc's AVX-512 block routines: they push no frame pointer,
and DWARF unwinding resolves none of their samples. See `README.md` for the measurements behind
that claim and for the two ways this tool itself lies (summed-across-threads cycles, and tail-call
return addresses).

It belongs here rather than in `src/` because it must be *loaded into* the process rather than
linked into it, the same reason `tools/getenv_probe` lives beside it. The two are siblings: that one
counts environment reads, this one attributes block copies and comparisons. Both ship a positive
control, and for the same reason — an empty report and a broken shim look identical.
