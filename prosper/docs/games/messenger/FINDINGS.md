# Findings — reconnaissance of `PPSA24651` (The Messenger, PS5)

Everything below was extracted from the game dump with our own tool
(`tools/self_dump`). These facts define the entire scope of the project.

## The good news that makes this possible

| Fact | Consequence |
|------|-------------|
| CPU is **x86-64** (ELF `e_machine=0x3E`) | **No CPU emulation / JIT.** Guest instructions run natively on the host. This is the single biggest reason the project is tractable — it is Wine-shaped, not RPCS3/Dolphin-shaped. |
| OS ABI is **FreeBSD** (`e_ident[7]=0x09`) | Orbis/Prospero is a FreeBSD 9-ish derivative. Guest syscalls & libc semantics follow FreeBSD, which we HLE. |
| SELF segments are **unencrypted** here (entropy ≈ 6.0 bits/byte, plaintext machine code) | We can load & inspect the real code **without console keys**. A retail-encrypted title (entropy ≈ 8.0) would be a hard stop. |
| Guest uses the **System V AMD64 ABI** | Matches Linux host exactly. On a **Windows host** the calling convention is Microsoft x64, so every guest↔host call boundary needs an ABI trampoline. (See ARCHITECTURE.) |

## Container format

- File wrapper: **SELF** (magic `4F 15 3D 1D`), header 0x560, 12 segment descriptors.
- Inner ELF at offset **0x1A0** (`0x20 + 12×0x20`).
- `eboot.bin` = `ET_SCE_DYNEXEC` (main dynamic executable). `*.prx` = `ET_SCE_DYNAMIC` (shared lib).
- **No `PT_SCE_DYNLIBDATA`** program header (unlike PS4). Dynamic linking uses
  mostly **standard ELF tags** (`DT_STRTAB=5`, `DT_SYMTAB=6`, `DT_NEEDED=1`) as
  **virtual addresses**, plus Sony tags (`DT_SCE_NEEDED_MODULE=0x61000045`,
  `DT_SCE_EXPORT_LIB=0x61000049`, `DT_SCE_SYMTABSZ=0x6100003f`, …).
- Segment data is addressed by **virtual address through `PT_LOAD` segments** —
  the loader must build a VA→file map (the "info/digest" SELF segments with
  flag bit `0x800` clear are skipped; data segments have `0x800` set).
- Symbols are **NID-encoded**: `"<11-char-base64-SHA1>#<libId>#<modId>"`, ids in
  Sony base64. `libId` (0-based) indexes the `DT_SCE_EXPORT_LIB` records → library name.

## The engine

- **Unity 2022.3.32f1**, **IL2CPP** backend (C# → C++ → native `Il2cppUserAssemblies.prx`, 43 MB).
- Graphics API: **AGC** (`libSceAgc` + `libSceAgcDriver`) — the modern low-level PS5
  GPU API Unity 2022 targets. *Not* GNM. This is the hardest translation target.
- Third-party: Rewired (input), Newtonsoft.Json, Unity Services / Purchasing / Analytics.

## Modules shipped in the dump (game's own code — we load these as-is)

| Module | Exports | Imports | Role |
|--------|--------:|--------:|------|
| `eboot.bin` | 1 | 612 | Unity PS5 player (the executable) |
| `libc.prx` | 2921 | 117 | Game ships its own libc |
| `Il2cppUserAssemblies.prx` | 296 | 292 | **The game's C# logic**, compiled |
| `libSceNpCppWebApi.prx` | 42826 | 95 | NP web API C++ wrapper |
| `PSN.prx` | 7 | 2437 | Unity PSN plugin |
| `SaveData.prx` | 28 | 55 | Unity save-data plugin |
| `PS5Util.prx` | 7 | 10 | Unity PS5 utility plugin |

## System modules we must implement (HLE) — the real work-list

These are imported but **not shipped** (they live in the console OS). Union across
all shipped modules, by total imported function count. This is the backlog.

**Core / OS (needed just to start executing):**
- `libkernel` (289) — threads, memory, sync, timers, events, fs. *The heart of the layer.*
- `libc` / `libScePosix` (92) / `libSceLibcInternal` (23) — mostly thin thunks to host libc.
- `libkernel_unity`, `libkernel_sync_on_address` — futex-like primitives.

**Graphics (needed for a frame):**
- `libSceAgc` (68) + `libSceAgcDriver` (9) — command buffers, pipeline state. → Vulkan.
- `libSceVideoOut` (17) — swapchain/flip. → host window + present.
- `libSceGnmDriver`? not present — confirms AGC-only.
- **+ a shader recompiler**: RDNA2/GCN GPU ISA → SPIR-V (not an import; implied by AGC).

**Input / Audio:**
- `libScePad` (17) — DualSense. → SDL gamepad.
- `libSceAudioOut(2)` (20) — PCM out. → host audio (SDL/miniaudio).
- `libSceAjm` (15) — hardware audio decode (ATRAC9/AAC/MP3). `snd0.at9` is ATRAC9.
- `libSceAvPlayer` (20) — video cutscene playback.

**System services:**
- `libSceSystemService` (20), `libSceUserService` (14), `libSceSysmodule` (10),
  `libSceAppContent` (10), `libSceSaveData_native` (24), `libSceMsgDialog`/dialogs,
  `libSceIme`(Dialog) (10), `libSceRtc` (3), `libSceRandom` (1).

**Online (single-player game → mostly stubbable initially):**
- `libSceNp*` family (`Manager`, `WebApi2`, `Trophy2`, `UniversalDataSystem`,
  `EntitlementAccess`, `Commerce`, `Auth`, `SessionSignaling`, `GameIntent`, …),
  `libSceHttp(2)` (31), `libSceSsl` (13), `libSceNet(Ctl)` (49), `libSceJson2` (64).

### Rough sizing to first milestones
- **Boots & traps at first unimplemented call:** loader + stub table only (~0 real impls).
- **Reaches Unity engine init:** ~libkernel + libc/posix (~250–300 fns, most trivial).
- **First rendered frame:** + VideoOut + AGC (77) + **shader recompiler** (the multi-month item).
- **Playable:** + Pad + AudioOut + Ajm + SaveData + Np stubs.
