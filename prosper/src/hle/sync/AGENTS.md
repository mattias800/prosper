# `src/hle/sync` — the machinery the guest synchronisation entry points share

The `scePthread*` / `sceKernelWaitOnAddress` **handlers** live next door in `src/hle/kernel`. What
lives here is everything more than one of those bodies needs, and everything a *second* consumer
outside the kernel HLE needs — most importantly the GPU command processor, which waits on and wakes
guest completion labels through the same primitives a guest thread does. That is the folder's whole
reason to exist: a waker and the blocking primitive it must pair with cannot be allowed to drift
apart, so there is exactly one implementation of each and both callers reach for it here.

| file | what it is for |
| --- | --- |
| `sync_futex.*` | guest-address futex wait/wake, and the **interruptible** pthread condition/mutex wrappers. On POSIX these are thin; on Windows they are a `WaitOnAddress` implementation, because a guest thread parked in a native wait has to be woken for cooperative guest-exception delivery. |
| `sync_retire.*` | quarantine-then-reclaim for a destroyed guest sync object. A guest may destroy a mutex/cond/semaphore with a thread parked inside it, so the storage is retired rather than freed (#2042, #2176). |
| `pthread_slot.hpp` | the guest handle scheme: a guest "mutex" is a POINTER CELL the init handler writes a host object into, plus FreeBSD's static-initialiser sentinels. Read this before touching anything that takes a guest handle. |
| `hle_fiber.cpp`, `hle_ult.cpp` | Sony's cooperative fiber and ULT libraries — separate schedulers, not part of the pthread family above. |

Two boundaries worth stating because getting them wrong is easy:

- **Host-platform *sleeping* is not here.** `src/host/platform/precise_sleep.*` owns "block until a
  deadline without inheriting the Win32 timer period", and it also owns the **pure arithmetic** of
  this family's timed waits — deadline conversion, poll cadence, saturation. Arithmetic goes there so
  a fake-clock test reaches it on every platform; the code that decides *what to wait on* stays here.
- **Contracts and error encodings are not here either.** Which errno a guest sees, and whether a
  spelling reports the bare FreeBSD number or the encoded `SCE_KERNEL_ERROR_*`, is a property of the
  entry point in `src/hle/kernel`, not of the wait. Several handlers deliberately share one body here
  and answer differently.

The Windows halves of these files carry more history than the POSIX ones and it is not decoration:
each `#ifdef _WIN32` branch exists because a native primitive was measured misbehaving. Read the
comment before replacing one with the obvious call.
