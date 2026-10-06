---
kind: status
status: current
---

# Assassin's Creed Black Flag Resynced (`PPSA28183`) — status

Tracker: [#4131](https://github.com/mattias800/prosper/issues/4131). This document is the technical
record the tracker points at; the tracker holds the rung.

**Rung 0 — nothing renders** (the guest now runs and flips, with every draw dropped; see
§ Progress 2026-10-02 (later); the paragraphs below describe the original stall). First measured 2026-10-02 on Windows 11 (MinGW build, NVIDIA RTX 4070
SUPER) at main `deff140b8d4a`, then again with the thread-handle fix of #4129 applied.

The guest boots in about 1.6 to 2.0 s and `prosper-app` opens its window, but no guest frame is
presented, so the window stays black. **A black window here is not a slow load.** On unmodified main
the guest's primary thread was already dead 1.6 s into the run while the app sat idle for over an
hour (see § Ruled out).

## Facts about the dump

- `titleId=PPSA28183`, `contentVersion=01.006.038`, `masterVersion=01.00`, `sdkVersion=0x1200000000000000`,
  `applicationDrmType=upgradable`. The AGC register-defaults line reports SDK version 9, which
  disagrees with that `sdkVersion`; not investigated.
- Content is `DataPS5_*.forge` archives, which points at Ubisoft's Anvil engine (inferred from the
  file names; no engine string was read from the binary).
- The dump carries third-party stand-ins for Sony libraries (a `fakelib/` folder with
  `libSceAgc`, `libSceAgcDriver`, `libSceAmpr`, `libScePlayGo` and `libScePsml`, an
  `ampr_emu.index`, and an `eboot.decoded.elf`). prosper's `module_path_policy` is the guard that
  rejects such locations; whether it fired on this dump was not checked in these runs. Disassembly
  for this document was taken from the SELF `eboot.bin` flattened with
  `tools/il2cpp/prx_to_elf.py`, not from the pre-decoded ELF.

## Reproduction route

The diagnostic route below uses a build with open PR #4129 applied;
`PROSPER_EXIT_ON_GUEST_END` is not available on unmodified main.

```powershell
$env:PROSPER_RENDER='1'; $env:PROSPER_GUEST_ARGS='-force-gfx-direct'
$env:PROSPER_BOOTPHASE='1'; $env:PROSPER_EXIT_ON_GUEST_END='1'
.\prosper\build-mingw-app\prosper-app.exe --dump "<DUMP_ROOT>\PPSA28183-app0"
```

- `PROSPER_BOOTPHASE=1` prints the seven boot phases; the last one names where a stalled boot is.
- `PROSPER_EXIT_ON_GUEST_END=1` (added with the dead-guest report) quits when the guest entry thread
  ends instead of idling on a black window. It currently exits 0 even after a crash.
- **Vulkan on this host.** The NVIDIA ICD was not registered in the Windows registry, so the first
  launch died with `Installed Vulkan doesn't implement the VK_KHR_surface extension` and
  `vulkaninfo` reported `Found no drivers!`. Setting `VK_DRIVER_FILES` to the driver's `nv-vk64.json`
  in the DriverStore makes both work. This is a host setup problem, not a prosper defect.
- There is no input route yet: nothing renders, so there is nothing to navigate.

**Best checked-in screenshot: none.**

## Progress 2026-10-02 (later): the guest now runs; every draw is dropped

With #4129, #4146 and #4166 on main plus the open PRs #4174 (APR completion filter -25), #4189
(fixed direct-memory remap on Windows) and #4194 (empty MultiDcb segments), and the opt-in init
deferral prototype (not for merge; branch `proto/defer-autolink-init` on the contributor fork), the
title no longer stalls or aborts. A 60 s run loads `libmemorywrapper_f.prx` and `libaegir_f.prx` from the
eboot, creates the engine threads, reaches AGC initialisation, registers three 1080p scanout
buffers and flips about 40 to 50 times with audio playing. The window is still black.

Why it is black (`PROSPER_DBG=1`, one 40 s run, NVIDIA host, 32-lane subgroups): only 15 draws are seen
and none is issued. Thirteen fragment programs are refused by the recompiler and two by the backend
(`subgroup-contract`, `unproved-vote`). Eighteen compute programs are refused too, so their outputs
are never written. Of those, eleven stop at an image instruction (`MIMG`) with
`mode=unresolved-operand`; one example reads `[mimg-unresolved] pc=38 op=0x00 srsrc=s20 srt_tag=NONE
key_res=null ud_alias=NONE written=1`, i.e. the T# in `s20..s27` was written by the shader itself and
no SRT-tag, fetch-pc or user-data-alias route resolves it. Others stop at `s_cselect_b64` with `vcc`
as destination (`pc=20`), an `SMEM` load, and one `s_branch` the structured emitter cannot place.
This is the same recompiler/resource-binding frontier recorded for other titles (`docs/gpu/RECOMPILER_REMAINING.md`,
`docs/gpu/RESOURCE_BINDING.md`); it is not specific to this title. Not yet established: whether the host's
32-lane subgroups (against the guest's wave64) are what blocks the two `subgroup-contract` fragment
draws, since the compute refusals print `host-subgroups=unavailable`. (That field never described the
host on a `…/recompile` refusal: it had no host fields to print. Since #4530 those lines print
`not-consulted`; only the two `subgroup-contract` lines carry the host range.)

## Current frontier

1. **Fixed by #4129 (open): the guest dereferenced a Windows thread handle.** At `eboot+0x161ed91`
   the guest does `mov rax,[handle]; mov ecx,[rax]` on the value `scePthreadCreate` wrote. prosper
   wrote the host `pthread_t`, which on MinGW is a small integer index, so the guest read address 3
   and the primary thread died (`ACCESS-VIOLATION addr=0x3 rax=0x3`). With the fix the guest passes
   that instruction and the `SystemLogger` thread runs. `CONFIDENCE: MED` on the handle layout (first
   dword is the thread id): inferred from this one disassembly, not checked against a PS5.
2. **Next: the observed fault is the title's abort path after a failed archive read.** The process
   exit code `0xC0000005` comes from a deliberate guest write to address 2. The cause of the failed
   archive read remains unresolved. Under gdb (Python
   script that lets prosper's expected `%fs:0` TLS-emulation faults and the init-function fault pass),
   the first other fault is at `eboot+0x5bcf9e4`, `mov DWORD PTR ds:0x2,0x0` (a write to address 2),
   reached straight after `call eboot+0x80` with `esi=0x100`. That is a `snprintf`-style call into a
   256-byte buffer, and the format string at `eboot+0xab1bd80` is
   `fseek error while reading the fat. Seek pos: %llu bigfile: %s`. So the title is reporting that an
   `fseek` on one of its `DataPS5_*.forge` "bigfile" archives failed while reading the archive's file
   allocation table, then crashes on purpose. The `%llu` argument was `0x524550534f525000`, which is
   not a plausible file offset and reads as the ASCII bytes of the string "PROSPER" shifted by one;
   whether that is prosper leaving an output uninitialised (for example an unimplemented `ftell` or
   `fgetpos`) is **not** established. Evidence: `gdb` run 2026-10-02 at main `deff140b8d4a` plus
   #4129; disassembly of the SELF `eboot.bin` flattened with `tools/il2cpp/prx_to_elf.py`. Still
   open: which libc or kernel call failed, with what offset and whence, and on which archive.
   Locating the final fault in guest code does not rule out an earlier HLE error or unfilled output
   as the cause of the failed seek; the stubs listed below remain untested.

Unexplained and not yet shown to matter:

- The init-function fault (`init fn 0x5d4000020 faulted ... addr=0x0 rip=0x5d4234ab0`, #4139) is
  **not benign and is now explained, but not fixed**. `0x5d4000000` is `libaegir_f.prx`; `0x5d4000020`
  is its module entry (a crt start that walks `DT_INIT_ARRAY`, so the rest of that array is skipped
  after the fault). The faulting code is a static constructor at image `+0x234a70` that calls the
  title's pooled allocator (`+0xb6f70`, which falls to `+0x1e90`). On a first allocation that
  allocator reserves a 1 GiB arena through the import `WLABcNu8BnU`, which resolves to
  `libmemorywrapper_f.prx` (`0x5d0000000`). That export is a thin dispatcher: when the module's callback
  table (`+0xc018`) is zero it returns 0, the allocator returns NULL, and the constructor stores through
  it (`mov %rax,(%rax)` with `rax = 0`). The table is only filled by `libmemorywrapper_f`'s own init
  export (`d1C59AHrOPI`, `+0xd0`), which no linked module imports, so the title has to call it itself.
  prosper links both root-level PRXs as boot-time dependencies and runs every init before the eboot
  starts, in ascending name order (`libaegir_f` before `libmemorywrapper_f`), so the guest has not yet
  had a chance. The eboot carries the strings `/app0/libmemorywrapper_f.prx` and `libaegir_f.prx`,
  which suggests the title loads both itself with `sceKernelLoadStartModule` (initialising the wrapper
  first) and that deferring auto-linked modules' init until that call would be the faithful fix. Not
  verified: no run got far enough to observe a `sceKernelLoadStartModule` call, because both an
  unmodified and an aegir-init-skipped run die later at `__stack_chk_fail` during `sceUltInitialize`.
  That change would alter init timing for every title that auto-links a module, so it needs a
  cross-title census before landing. Evidence: gdb and disassembly of the SELF modules flattened with
  `tools/il2cpp/prx_to_elf.py`, 2026-10-02, main `4a2ea88d` plus #4129 and #4137.
- The `__stack_chk_fail` that ended the "Loading Thread" right after `sceUltInitialize` is explained
  and fixed on main by #4146 (filed as #4138): `sceKernelAprSubmitCommandBufferAndGetResult`
  (`ASoW5WE-UPo`) takes `(cb, ring, result*, uint32_t* id)`, the title passes a 4-byte stack int
  for the id (`eboot+0x24511b0`) directly under its canary, and prosper stored an 8-byte token
  there, zeroing the canary's low dword. Evidence: the stack copy of the canary read
  `0x5245505300000000` against the expected `0x524550534F525000`.
- With that fix and the deferral prototype applied the process no longer dies: about 25 guest threads
  (`TaskThread00..11`, `IdleThread00..02`, `Loading Thread`, `SaveGameThread`) all sit in
  `sceKernelWaitCond`/timed waits and the guest main thread waits on a condition variable that a
  worker had already broadcast before it began waiting. `kqueue` and `kevent` (`libScePosix`)
  still return 0 through the unimplemented stub, so a descriptor of 0 comes back from `kqueue`;
  whether the stall is an engine file-completion path waiting on them is a hypothesis, not a result.
- Where the stall sits, from guest disassembly (eboot offsets): the title does load the memory
  wrapper itself. `eboot+0x3d136b0` calls `sceKernelLoadStartModule("/app0/libmemorywrapper_f.prx")`
  and then `sceKernelDlsym(..., "AE_MemoryWrapper_Init")`, which is the wrapper's init export
  (`d1C59AHrOPI`) that fills the dispatch table the aegir constructor needs. That function has no
  direct callers (only reachable through an indirect call) and is never reached in a run: no
  `sceKernelLoadStartModule` call is seen in 80 s. The guest main thread is parked in a scheduler
  loop (`eboot+0x21adff0`, called from `+0x21adf60`) waiting for a flag that a worker sets after
  starting (`+0x5deaa34`), and the twelve `TaskThread`s and three `IdleThread`s are idle. So the
  open question is what the engine's startup waits on before it reaches the wrapper load, not the
  wrapper itself. `kqueue`/`kevent` are called once each by the eboot (`+0x334ebba`, `+0x334ebd1`)
  to register one read filter on ident 0 and nothing waits on them, so they are unlikely to be
  the blocker; no implementation is in flight elsewhere (#4153 lists them out of scope).
- `scePthreadAttrGetstack` is unimplemented and returns 0 without filling its outputs. The
  `SystemLogger` thread called it right before the crash; that is a suspicion, not a result.
- Unimplemented calls returning 0, names from `ps5rs/data/nids.csv`: `kqueue`, `kevent`
  (`libScePosix`), `sceKernelGetOperationMode`, `sceNpAppLauncherInitialize`,
  `sceCoredumpRegisterCoredumpHandler`, `sceNpSessionSignalingInitialize`, `sceKeyboardInit`.
  Whether any is required to reach a frame is not known. Per the project's entitlement rule, answers
  must come from local inventory, never a blanket "owned".

## Ruled out

- **"The black window is a slow load or a hang in a long boot."** Falsified: the primary guest thread
  ended with `ACCESS-VIOLATION addr=0x3 rip=eboot+0x161ed91` at about 1.6 s while the process stayed up
  for 72 minutes with 32 s of CPU and no further log lines; `BOOT_COMPLETE` is reached at 1.6 to 2.0 s.
  Evidence: `PROSPER_BOOTPHASE=1` run logs, 2026-10-02. Fix: #4129.
- **"The renderer or present path is at fault for the black window."** Not supported: the window opened
  and the renderer registered; no guest flip was ever issued because the guest was dead. The one
  `VK_KHR_surface` failure seen came from the host Vulkan ICD not being registered (§ Reproduction), not
  from prosper.
- **"The host crashes in an HLE stub after `sceKeyboardInit`."** Falsified for the first host-visible
  death: the exit code `0xC0000005` is the title's own abort path (write to address 2 after formatting
  `fseek error while reading the fat ...`), reached from `eboot+0x5bcf9e4`. This identifies the final
  faulting instruction; it does not rule out an earlier HLE error as the cause of the failed seek.
- **"The boot or link phase is what stalls."** Falsified by the phase log: all seven phases complete
  (`PROCESS_START` through `BOOT_COMPLETE`) in under 2.1 s.
