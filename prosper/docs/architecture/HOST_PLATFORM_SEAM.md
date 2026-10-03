# The host-platform seam

prosper runs the guest's x86-64 code natively on Linux (the primary host) and on Windows. Code that
reimplements Sony's libraries keeps needing the same few operating-system services: reserve and
protect memory, wait on an address, start a thread, read a clock, open a file. Today most of those
calls sit inline in the HLE code, inside `#ifdef _WIN32` / `#elif defined(__linux__)` arms. This
document says where they should go instead, and how the move happens without a flag day.

Related: `docs/architecture/ARCHITECTURE_TARGET_TREE.md` (the whole target tree and the layer order),
`src/host/platform/AGENTS.md` (the folder this seam grows into), and
`tools/ci/check_arch_ratchet.py` (the `platform-ifdef` and `layer-include` rules that freeze the
current state). Umbrella issue: see `ARCHITECTURE_TARGET_TREE.md` § Tracking.

## Scope

**In scope**: the operating-system services the HLE layer, the loader and the GPU translation
layer need from the host.

- virtual memory: reserve, commit, protect, release, and **aliased views** (two addresses that map
  the same physical pages, which the direct-memory APIs need);
- write tracking: being told which guest pages were written since a point in time;
- futex-style wait/wake on an address, with a timeout;
- threads, thread naming and affinity, and **host TLS** (the host thread's own storage, as opposed
  to the guest's TLS block);
- fibers (cooperative user-mode context switches);
- fault handling: installing the process's handler for access violations and traps;
- monotonic and high-resolution clocks, and precise sleeps;
- files and paths: open/read/stat/enumerate a directory, and mapping the guest's path spelling onto
  the host's.

**Out of scope**:

- **the guest ABI bridge** — moving arguments between System V and Microsoft x64 is
  `src/host/abi`'s job and is about the *guest's* calling convention, not about OS services;
- **SDL, Vulkan, audio and video devices** — those are frontend or backend concerns, reached
  through the frontend's own interfaces, never from here;
- guest-visible semantics (what `sceKernelWaitEqueue` means). The seam answers "wait on this
  address on this OS"; the HLE layer still decides what the guest asked for.

## Current state (measured on main 6499c4d7 plus this branch)

Counted by `check_arch_ratchet.py`'s `platform-ifdef` rule: one per preprocessor conditional that
tests `_WIN32`, `_WIN64`, `__linux__`, `__APPLE__`, `__MINGW32__` or `_MSC_VER`.

| layer | directives | files |
| --- | ---: | ---: |
| `src/hle` | 326 | 31 |
| `src/gpu` | 29 | 13 |
| `src/loader`, `src/self` | 0 | 0 |
| `src/host` (exempt, for comparison) | 66 | 15 |

The largest HLE files, and the OS calls they make inline (counted by call-site name in each file):

| file | directives | what the arms call |
| --- | ---: | --- |
| `src/hle/fs/hle_file.cpp` | 70 | `stat`, `open`, `mmap`/`munmap`, `opendir`/`readdir` vs `FindFirstFileA` |
| `src/hle/kernel/hle_kernel.cpp` | 60 | `pthread_create`/`join`/`setname_np`, mutexes, `clock_gettime`, `VirtualAlloc`, `sigaction` |
| `src/hle/video/avplayer.cpp` | 26 | file `open`, Windows headers (`<direct.h>`, `_mkdir`), host-specific branches |
| `src/hle/sync/hle_fiber.cpp` | 24 | `SwitchToFiber` on Windows, a raw `syscall(SYS_gettid)` on Linux, host thread identity |
| `src/hle/sync/sync_futex.cpp` | 23 | `WaitOnAddress`/`WakeByAddressAll` vs pthread mutex+condvar, `clock_gettime`, `Sleep` |
| `src/hle/libc/hle_libc.cpp` | 17 | host C runtime differences |
| `src/hle/memory/hle_kernel_mem.cpp` | 16 | `mmap`/`mprotect`/`munmap` vs `VirtualAlloc`/`VirtualProtect`, aliased views (`memfd_create` / `CreateFileMapping`) |
| `src/hle/input/ime.cpp` | 11 | host text input |
| `src/hle/kernel/hle_kernel_time.cpp` | 10 | `nanosleep`, condvars, `clock_gettime` |
| `src/hle/np/np.cpp` | 10 | host networking/user differences |

(`hle_kernel_mem.cpp` reads low on directives because its platform code is in a few very large
arms; #3503 measured about half of the file inactive on any one host.)

What `src/host/platform/` holds today: `lifecycle.*` (cooperative stop/pause signals),
`gpu_submit_gate.*` (counted region with a bounded drain), `precise_sleep.*` (a precise sleep plus
the pure deadline arithmetic), `posix_shim.hpp` (POSIX names Windows lacks), `raw_syscall.hpp` and
`immortal.hpp`. Sibling folders already own part of the seam's scope: `src/host/memory/` (guest
memory query/copy and `guest_write_watch.*`), `src/host/tls/` (`guest_tls.cpp`, `fs_emu.hpp`),
`src/host/fault/` (fault context and trap arbitration) and `src/host/image/` (per-OS image mapping,
including the Linux `fault_handler` in `exec_image_linux.cpp`).

The dependency arrow is inverted in six places: `src/host` includes `src/hle/dispatch/` from
`host/image/boot_program.cpp`, `exec_image.hpp`, `exec_image_linux.cpp`, `exec_image_win.cpp`,
`runtime_module_load.cpp` and `host/tls/guest_tls.cpp` (11 include lines). Those are guest-semantics
code living in the host folder, which is the subject of § Guest vs host.

## Decision

1. **One interface per service, under `src/host/platform/`, with one backend per OS** (for example
   `futex.hpp` + `futex_linux.cpp` + `futex_win.cpp`), chosen at build time by which file the build
   compiles. No runtime dispatch, no virtual interface: there is exactly one host per binary.
2. **New code calls the seam.** A new `#if _WIN32` in `src/hle`, `src/loader`, `src/self` or
   `src/gpu` fails the `platform-ifdef` ratchet.
3. **Existing call sites migrate when touched.** Moving a call behind the seam is a
   behaviour-neutral change and is committed on its own, separate from whatever behaviour change
   motivated touching the file — the same rule the charter applies to file moves.
4. **The ratchet enforces the direction**: `platform-ifdef` counts may only fall, and
   `layer-include` stops a new `host` → `hle` edge.

prosper's loader maps the guest's ELF images directly on both hosts; nothing in this design changes
that or introduces a translation step.

## Target design

Derived from what the HLE code calls today (table above). `CONFIDENCE: MED` for the exact
signatures — they are a starting proposal to be settled by the first migration of each service.

| service | proposed interface (`src/host/platform/`) | Linux backend | Windows backend |
| --- | --- | --- | --- |
| virtual memory | `vm.hpp`: `reserve`, `commit`, `protect`, `release`, `map_alias(view, offset, size, addr)` | `mmap`/`mprotect`/`munmap`, `memfd_create` for alias backing | `VirtualAlloc(2)`/`VirtualProtect`/`VirtualFree`, `CreateFileMappingW` + `MapViewOfFile3`/`MapViewOfFileEx` (all used by `hle_kernel_mem.cpp` today) |
| write tracking | stays `src/host/memory/guest_write_watch.hpp`; the OS part moves under it as backends | `mprotect` + fault | `mprotect`-equivalent via `VirtualProtect` + vectored handler (`CONFIDENCE: LOW` that this is what ships today; verify before migrating) |
| futex | `futex.hpp`: `wait(addr, expected, timeout)`, `wake_one(addr)`, `wake_all(addr)` | `futex(2)` | `WaitOnAddress` / `WakeByAddressSingle` / `WakeByAddressAll` |
| threads + host TLS | `thread.hpp`: `spawn`, `join`, `set_name`, `set_affinity`, `current_id`, stack bounds | `pthread_*`, `pthread_getattr_np` | `CreateThread`/`_beginthreadex`, `SetThreadDescription`, `GetCurrentThreadStackLimits` |
| fibers | `fiber.hpp`: `create(stack, entry)`, `switch_to(from, to)`, `destroy` | whatever `hle_fiber.cpp` does today on Linux (verify before migrating) | `SwitchToFiber` (used today) |
| fault handling | `fault.hpp`: `install(handler)`; the handler receives a host-neutral fault context (`src/host/fault/fault_context.hpp` already exists) | `sigaction(SIGSEGV/SIGBUS/SIGTRAP)` | `AddVectoredExceptionHandler` |
| clocks + sleep | `clock.hpp`: `monotonic_ns`, `process_cpu_ns`; `precise_sleep.hpp` already exists | `clock_gettime`, `nanosleep` | `QueryPerformanceCounter`, waitable timers |
| files + paths | `file.hpp`: `open`, `read_at`, `stat`, `list_dir`, `map_readonly`; path translation stays in `src/hle/fs` | POSIX | `CreateFileW`, `FindFirstFileW`, wide-path conversion |

## Guest vs host

`src/host/` currently mixes two different things:

- **host services** — "how does this OS do X": what this document is about;
- **guest semantics on the host** — the System V ↔ Microsoft x64 bridge (`host/abi`), the guest's
  TLS block (`host/tls`), interpreting a guest fault (`host/fault`), guest memory query/copy/watch
  (`host/memory`), mapping and running guest images (`host/image`).

The proposal (not done, `CONFIDENCE: MED`) is a new `src/guest/` for the second group, leaving
`src/host/` as the OS seam. `guest` sits above `host` and may include `hle/dispatch` legitimately,
which removes most of today's `host` → `hle` edges by relocation rather than by redesign. The
moves, their order and their tools are in `ARCHITECTURE_TARGET_TREE.md`.

## Migration order

1. **`src/hle/sync/sync_futex.cpp` and `src/hle/sync/hle_fiber.cpp` first.** They are small, their
   OS surface is narrow (wait/wake; context switch), they have the most direct cross-title effect
   (every threaded title waits on futexes), and a both-host test is cheap to write: a wait that
   times out, a wake that releases exactly one or all waiters, and a fiber ping-pong.
   **The fiber hazard to design for**: *Uncharted*'s Naughty Dog job system starts fibers on the
   main thread and resumes them on worker threads (CLAUDE.md, `docs/games/UNCHARTED_STATUS.md`). The
   fiber interface must not assume the host thread that switched *into* a fiber is the one that
   switches *out*, and must not tie host TLS to a fiber. A test must resume a fiber on a different
   thread than the one that created it.
2. `src/hle/kernel/hle_kernel_time.cpp` (clocks, sleeps) — mostly mechanical once `clock.hpp`
   exists.
3. `src/hle/memory/hle_kernel_mem.cpp`, together with #3503's per-platform split: virtual memory
   and aliased views.
4. `src/hle/kernel/hle_kernel.cpp` (threads), then `src/hle/fs/hle_file.cpp` (files).
5. The remainder file by file, as each is touched (`avplayer.cpp`, `hle_libc.cpp`, `ime.cpp`,
   `np.cpp`, and the GPU capture serializers).

## Verification

- **Ratchet**: `platform-ifdef` (counts may only fall) and `layer-include` (no new edge against
  `LAYER_ORDER`). Both run in delta mode in CI, so a migration that removes directives shows up as a
  `notice` and can be recorded with `--update`.
- **Tests on both hosts**: each new interface gets a test under `tests/host/platform/` that runs on
  Linux and on the Windows CI jobs. `test_precise_sleep.cpp` is the pattern: the arithmetic is
  platform-neutral and driven by a fake clock; the OS call is a thin backend.
- **CI limits**: the Linux CI GPU jobs run on lavapipe (CLAUDE.md, "A green CI GPU-execution job
  cannot clear a failure on this machine's driver"). That does not matter for the seam itself, which
  has no GPU code, but a migration of a GPU-capture file must still be run locally on RADV.
- A move-only commit must leave behaviour identical; `nm` A/B of the touched object is a cheap check.

## Open questions

- Does write tracking on Windows use `GetWriteWatch` anywhere today, or only page protection?
  (`git grep` finds no `GetWriteWatch` in `src/`; confirm before designing the backend.)
- Fibers: should the interface expose a host-TLS swap hook, or should guest TLS (in `src/guest`)
  own that entirely?
- macOS/Rosetta: `__APPLE__` arms exist; is macOS a third backend or a variant of the POSIX one?
- Where does path translation (guest `/app0/...` → host path) end and the file backend begin?

## Ruled out

- **A relinker that rewrites guest imports before loading.** Not adopted: prosper already loads
  the guest's ELF images natively on both Linux and Windows and resolves imports in its own loader
  (`src/loader`), so a separate relinking step would add a second linker without removing anything.
  The seam is about OS services, not about how guest code is loaded.
