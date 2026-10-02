# Assassin's Creed Black Flag Resynced (`PPSA28183`) — status

Tracker: [#4131](https://github.com/mattias800/prosper/issues/4131). This document is the technical
record the tracker points at; the tracker holds the rung.

**Rung 0 — nothing renders.** First measured 2026-10-02 on Windows 11 (MinGW build, NVIDIA RTX 4070
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

## Current frontier

1. **Fixed by #4129 (open): the guest dereferenced a Windows thread handle.** At `eboot+0x161ed91`
   the guest does `mov rax,[handle]; mov ecx,[rax]` on the value `scePthreadCreate` wrote. prosper
   wrote the host `pthread_t`, which on MinGW is a small integer index, so the guest read address 3
   and the primary thread died (`ACCESS-VIOLATION addr=0x3 rax=0x3`). With the fix the guest passes
   that instruction and the `SystemLogger` thread runs. `CONFIDENCE: MED` on the handle layout (first
   dword is the thread id): inferred from this one disassembly, not checked against a PS5.
2. **Next: the host process dies with exit code `0xC0000005`** shortly after `sceKeyboardInit`. The
   faulting address and the guest context have not been captured yet.

Unexplained and not yet shown to matter:

- An init function (`0x5d4000020`, fault `rip=0x5d4234ab0`, host code) takes an access violation at
  address 0 and boot continues (`continuing`).
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
- **"The boot or link phase is what stalls."** Falsified by the phase log: all seven phases complete
  (`PROCESS_START` through `BOOT_COMPLETE`) in under 2.1 s.
