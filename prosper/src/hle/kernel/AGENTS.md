# `src/hle/kernel` — libkernel entry points and guest API contracts

Guest thread, scheduling, synchronization, clock and miscellaneous libkernel handlers live here.
These entry points own argument layouts, return values and FreeBSD/SCE error encodings. Shared
synchronization machinery belongs in `../sync`, memory mapping in `../memory`, and filesystem
operations in `../fs`; host execution, TLS and platform mechanisms belong under `src/host`.

Queries that describe the same guest resource must agree across their API surfaces. Register
behavior through the final built-in HLE table and exercise that registry in regressions so later
library registration cannot silently replace a working handler with a no-op.

`apr_submission` owns the eager read-batch completion state and opaque kernel submit handles.
The filesystem layer reports real read outcomes; memory owns command-buffer rewind/destruction;
equeue event binding tags remain a distinct completion payload in the event queue layer.
