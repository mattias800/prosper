# Blasphemous 2 (macOS): host-global corruption in the equeue cluster (#707)

Last updated: 2026-07-17

## Symptom

On **macOS (x86-64 under Rosetta)**, a routed Blasphemous 2 (`PPSA13579`) run corrupts prosper's own
host global `g_eq_mx` (a `std::mutex`) at ~render frame 120 (guest ~2400), during asset/FMOD load: its
pthread `__sig` word is zeroed, so a later `pthread_mutex_lock` returns `EINVAL`, `std::mutex::lock`
throws, and the process terminates in `vblank_pump`. ~50–80% of routed runs reproduce (high variance).

**Latent on Linux**: 3/3 taint-mode routed runs to sustained gameplay show the watched bytes are never
written. (Linux glibc also tolerates a zeroed mutex, so even a write would not crash there.)

## What is confirmed (this session)

The corruptor is an **in-place zeroing of a set of `boot_trace`'s own `__DATA` globals** — the equeue
mutex cluster (`g_eq_mx`, `g_apr_mx`, `g_det_clock`) plus the diagnostic decoys interleaved with them.

1. **In-place store, not a remap.** `mach_vm_region` of the victim page is byte-identical (base, size,
   prot) before and after the zeroing — the mapping is intact; only the bytes change. (`mmap(MAP_FIXED)`
   / `vm_allocate` would change the region.)
2. **Image-relative, not window-relative.** The zeroed addresses are fixed offsets into `boot_trace`'s
   loaded image and move *with* it: det was at `0x102b97000` / `0x104d79000` (4 GiB + ASLR slide) with a
   normal build, and at `0x247f000` (~36 MB) with a low-loaded build — always `image_base + 0x178000`.
   **Relocating the whole executable does NOT help — the write follows the image.** (This *disproves*
   the earlier "guest GPU-VA window collision" hypothesis: it is not that the image sits inside
   `[4 GiB, 64 GiB)`; the write targets the image directly.)
3. **Bypasses current `mprotect`.** With the lo-decoy page `mprotect(PROT_READ)` from boot (never-locked,
   no co-located writer, no relift), the page's bytes are still zeroed and **no SIGSEGV/SIGBUS reaches
   our fault handler**. A normal CPU store to an RO page must fault (proven by the pageguard `selftest`),
   so this is not an ordinary store.
4. **Honors `max_protection`, and lowering it blocks the write SILENTLY.** With the lo-decoy page's
   *current* protection set to `PROT_READ` (`mprotect`), lo is still zeroed (`lo_hits=24`); with its
   *maximum* protection lowered to `VM_PROT_READ` (`mach_vm_protect(set_maximum=TRUE)`), lo is NOT zeroed
   (`lo_hits=0`) across runs while the max-RW real victims still are. The block produces **no fault** in
   our handler — so the write is a VM-system write that returns `KERN_PROTECTION_FAILURE` and is ignored,
   not a CPU store and not a fault-delivering Rosetta store. prosper's only two `mach_vm` writers
   (`mach_vm_write` via `process_vm_writev`, `mach_vm_read_overwrite` via `process_vm_readv`) are guarded
   and never fire, so the remaining writer is `mach_vm`-family from a dylib / the Rosetta runtime.
5. **Disjoint / page-granular.** It zeroes specific pages (lo-decoy page, det's page, the eq/apr page)
   and provably skips an 11 KiB never-locked ballast array placed between them — so it is targeted, not
   one contiguous sweep.
6. **Inlined / not a libc call.** `DYLD_INTERPOSE` of `memset`/`bzero`/`memcpy`/`memmove` and of
   `mmap`/`munmap` all catch nothing while the zeroing fires.

Taken together (2)+(3)+(4): the write reaches `boot_trace`'s own `__DATA` **through the VM system**
honoring `max_protection` while ignoring current protection. On macOS that signature matches either a
`mach_vm`-family write or a **Rosetta-translated guest store** (Rosetta re-raises a translated store to
a page whose *max* allows write after the current-prot fault — which is exactly why `mprotect` guards
are blind). The macOS-only + image-relative behavior is consistent with the guest being handed a host
pointer into (or adjacent to) `boot_trace`'s `__DATA` and writing through it, landing on the equeue
cluster in macOS's `__DATA` layout but on benign memory in Linux's.

## Mechanisms RULED OUT (do not re-investigate without new evidence)

- Ordinary CPU store — page-guards (det / ballast / lo) never fault; `mprotect(RO)` is bypassed.
- prosper `mmap(MAP_FIXED)` — guarded at `map_at`/`map_phys_at`/`apr_write_guest_dst`/the fault-handler
  lazy-backing and interposed process-wide; never fires on a corruption run.
- `process_vm_writev` → `mach_vm_write` — guarded at the choke point; never fires.
- `process_vm_readv` → `mach_vm_read_overwrite` (writes its destination) — probed; never fires.
- `madvise(MADV_FREE)` / remap / `vm_copy` — no such calls in prosper source.
- Guest GPU-VA window collision — disproved by (2): relocating the image out of `[4 GiB, 64 GiB)`
  does not stop the corruption.

## Why the RIP is still unnamed

The write evades every fault-based catch (bypasses `mprotect`; Rosetta handles its own guest-store
faults) and every symbol interpose (inlined / dylib-internal), and the ~50–80% reproduction with a
variable subset of pages defeats single-run A/Bs. Hardware watchpoints do not exist under Rosetta.

## Recommended next step

Find **what host pointer prosper hands the guest during asset/FMOD load** that resolves to (or adjacent
to) `boot_trace`'s `__DATA` — the guest writing through it is the working hypothesis. Concretely:
instrument every HLE return / record field / callback argument published to the guest in the APR/Ampr,
FMOD, and IoStore paths, and flag any value inside `prosper_fixed_map_hits_host_image(...)`. Once the
handed pointer is found, the fix is to back that buffer with real guest/heap memory (outside the image)
instead of a host global, so guest writes cannot reach the equeue cluster.

## Candidate fix (implemented, NOT yet verified)

Since the corruptor cannot be attributed on this platform, the fix mirrors what already keeps Linux
alive: keep the crash-critical equeue mutex **out of the corrupted `__DATA` range**. On macOS `g_eq_mx`
is now a trivial `.bss` forwarder to a heap `std::mutex` that is allocated once at static init (never
`new`/`malloc` on a lock path — the `%fs` SIGSEGV handler also takes this lock, and malloc there
deadlocks). `.bss` is not touched by the corruptor (round-2 evidence), so the mutex sig survives and
`vblank_pump` no longer hits `EINVAL`. Linux is unchanged (`std::mutex`, already `.bss`).

**Status: compiles cleanly; UNVERIFIED.** During this investigation an earlier (buggy) variant of this
mitigation deadlocked and left the machine's Rosetta runtime wedged (uninterruptible `U`-state
processes), so no x86-64 binary can run until the host is **rebooted**. After a reboot, verify by
running the reproduce recipe below and confirming `g_eq_mx` is no longer reported by SIGWATCH and the
route reaches sustained gameplay. **Caveat:** the corruptor zeroes a *region*, so if it also reaches
`g_eqs`/the reg vectors (extent unconfirmed), those must be relocated the same way — watch for the crash
merely moving rather than disappearing.

The only path to a true *root* fix (identifying the writer) is a platform where hardware watchpoints
work: a real x86-64 Mac or a Linux/x86 VM. A write-watchpoint on `g_eq_mx` there names the writer in one
run.

## Diagnostics on this branch (env-gated, `PROSPER_EQ_*`)

- `PROSPER_EQ_SIGWATCH=1|repair|taint` — 200 µs poller over the cluster sig words; `taint` pre-fills
  `0xA5` (defeats zeros-over-zeros); prints the victim's `mach_vm_region` at first hit (store-vs-remap).
- `PROSPER_EQ_PAGEGUARD=lo|ballast|det|det-step|selftest` — `mprotect`/`mach_vm_protect` guard on a page
  in the zeroed set; `lo` lowers *max* protection to prove the write honors `max_protection`.
- Host-image write/map guards — `prosper_fixed_map_hits_host_image()` +
  `prosper_report_host_image_clobber()` refuse/log any prosper `MAP_FIXED` or `mach_vm_write` that would
  land on the running executable's own image (`exec_image_linux.cpp`, `hle_kernel_mem.cpp`,
  `hle_file.cpp`, `posix_shim.hpp`). Correct defensive invariants; inert for this specific writer.

## Reproduce

```
cd prosper/build-mac-app   # x86_64 build, MoltenVK via -DPROSPER_MACOS_MOLTENVK
PROSPER_EQ_SIGWATCH=1 PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_RENDER=1 PROSPER_RENDER_EVERY=20 \
PROSPER_PAD_SCRIPT=@scripts/blasphemous2/reach-first-gameplay.pad \
  ./boot_trace <path>/PPSA13579-app0
```
Look for `[eq-sigwatch] HIT g_eq_mx ... 0x..->0x00` around render frame ~120.
