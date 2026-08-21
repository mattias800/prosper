# Sniper Ghost Warrior Contracts 2 (`PPSA03130`) — status and evidence

**CryEngine — the project's first title on this engine family.** Tracked on
[#2867](https://github.com/mattias800/prosper/issues/2867).

## Where it stands (2026-08-21)

**Rung 0 on a default launch, and the wall is named and proven.** The boot deadlocks **81 ms in**,
inside `run_guest_inits()` — before `run_entry`, before any asset is read, before the renderer is
ever asked for a frame. The blocking frame is the `module_start` of
**`sce_module/libSceNpCppWebApi.prx`**, a module **this title never imports**: prosper preloads it
because the file exists, under a rule added for *Sonic Origins*.

### The measurement

`tools/screenshot`, default flags, live renderer, on the real dump. `PROSPER_BOOTPHASE=1` (new,
`src/diagnostics/boot_phase_log.cpp`) makes the phases visible:

```text
[bootphase] +0.0ms  PROCESS_START
[bootphase] +53.7ms LINKING
[bootphase] +55.1ms HLE_REGISTERED
[bootphase] +80.6ms MODULES_MAPPED
[bootphase] +81.0ms STUBS_INSTALLED
[bootphase] +81.1ms GUEST_INITS_RUNNING     <- last line; BOOT_COMPLETE never arrives
```

**It is stuck, not slow, and not I/O starved.** Sampled every 10 s for the life of the run, the
process shows `cpu+0.00s`, `disk-read+0.00MB`, `reads+0` and three threads all parked in
`futex_do_wait` — and two of those three are RADV's own queue threads, so **the guest has created
zero threads**. A starved-but-alive boot would still be reading; this one issues no read syscalls
at all.

`tools/guest_bt` on the live process names the frame:

```text
#4  prosper::interruptible_cond_wait (... kind=GuestWaitKind::ConditionSequence)   sync_futex.cpp:343
#6  prosper::k_cond_wait (a0=0x5c011a5e8, a1=0x5c011a5e0)                          hle_kernel.cpp:1181
#7  0x5c0086a85   libc.prx
...
#24 0x4d024f15b   libSceNpCppWebApi.prx
#27 0x4d0000010   libSceNpCppWebApi.prx   <- base 0x4d0000000 + init_va 0x10 = module_start
#34 prosper::run_guest_inits (fns = 2 entries)                              exec_image_linux.cpp:3698
#35 prosper::boot_program                                                    boot_program.cpp:332
```

Between the module's `module_start` and the wait, the stack passes through `libc.prx` and back into
**eboot** code (`0x410000d70`, `0x4111b5d9c` …) twice — consistent with an allocation being routed
into CryEngine's own overridden allocator, which the eboot's `module_start` has not initialised yet
(init runs in reverse link order, so the eboot's is last). `CONFIDENCE: MED` on that mechanism; the
module identity above is `CONFIDENCE: HIGH`.

### The A/B — one file, decisive

Two runs of the same binary with the same flags. The only difference is a dump tree in which
`sce_module/libSceNpCppWebApi.prx` is absent; every other byte is the same file.

| arm | `sce_module/libSceNpCppWebApi.prx` | last boot phase | result |
| --- | --- | --- | --- |
| **A** | present (the real dump) | `GUEST_INITS_RUNNING` +81.1 ms | deadlocked; 0 CPU, 0 bytes read in 221 s |
| **B** | absent | **`BOOT_COMPLETE` +70.4 ms** | guest runs: 18 threads, real HLE traffic, assets streaming |

**No linked module imports it.** The string `libSceNpCppWebApi` appears **zero** times in
`eboot.bin`; the title's own import list names `libSceNpWebApi2`, which is a different library. The
only reason the module is in this boot is `boot_program.cpp`'s hard-coded
`{ d + "/sce_module/libSceNpCppWebApi.prx", BOOT_NPCPPWEBAPI }`, whose comment records that it was
added because *Sonic Origins* ships and **statically imports** it. So the preload is right for that
title and wrong here, and there is no environment switch that turns it off —
`PROSPER_NO_PSN` covers `PSN*.prx`/`SaveData.prx` only, and `PROSPER_NO_PLUGIN_AUTOLINK` covers only
auto-discovered `Media/Plugins` entries.

**The fix**: an input may opt in with `LinkInput::only_if_imported`, and is then preloaded only when
some other linked module names its library in its own import table. *Sonic Origins* keeps the module
(it imports it); this title no longer loads it. The policy is a pure function in
`src/loader/support_modules.hpp` so both directions are testable without a dump —
`tests/loader/test_support_modules.cpp`.

The rule is deliberately **narrow, and must not be generalised** to the optional preloads beside it.
The Unity FMOD/Wwise/PSN plugins are preloaded *precisely because* nothing imports them statically:
they are reached at runtime through `sceKernelDlsym` P/Invoke, appear in no import table, and this
test would drop every one of them. The test has an explicit arm for that (`RUNTIME PLUGINS`), and
the mutation arm that makes the filter drop every candidate reddens only the `KEEP` check — which is
what makes the keep direction load-bearing rather than decorative.

### Behind the wall (arm B, so not a claim about the default route)

With the module absent the guest boots and runs. In the first 158 s it read 208 MB and used 2.6 s of
CPU across 18 threads, and had **not** presented a frame. Three imports fall to the return-0 default:
`pthread_attr_getschedparam` (`qlk9pSLsUmM`), `pthread_setprio` (`a2P9wYGeZvc`) and
**`scePlayGoGetOptionalChunk`** (`g4AZyxpSAlA`) — the last is the one to watch, since a 0 with an
unwritten out-parameter is the false-success shape and this is a chunk-installed query.
`sceAjmInitialize` reports an unrecognised config revision `0x300000000` and accepts it.

Read the asset rate as a **lower bound on the box, not on prosper**: these runs were taken while the
machine was 70-90 % I/O-stalled by an unrelated archive extraction, and a direct-I/O control on the
same disk at the same time delivered 38 MB/s against arm B's ~1.3 MB/s.



## What the dump is, before any boot

CryEngine 5.x, laid out the way that engine always is: `engine/` and `gamesdk/` at the dump root,
each holding ordinary **ZIP** archives with a `.pak` extension (`PK\x03\x04`). 15 GB total, of which
**10.6 GB is six texture/video paks**; the startup working set is far smaller — `engine.pak` 14 MB,
`gamedata.pak` 33 MB, `scripts.pak` 6 MB, `shadercachestartup.pak` 776 KB, `shadercache.pak` 65 MB.

`engine.pak:Config/prospero.cfg` is the PS5 profile and states what the renderer is asked for:

```
sys_spec=7            r_width=3840   r_height=2160     ; native 4K
r_texturesstreaming=0                                  ; streaming OFF
r_texturesStreamPoolSize=800
s_FileCacheManagerSize=293216                          ; ~286 MB of Wwise cache
e_StreamCgfPoolSize=300
```

So this title is a **4K, streaming-off** workload with a large up-front audio cache, on a project
whose other titles mostly are not. Expect the startup read to be big in absolute terms even though
it is nowhere near the 15 GB the dump size suggests.

Audio middleware is Wwise. There are seven guest modules besides the eboot, all in `sce_module/`:
`libc.prx`, `libSceFace.prx`, `libSceFaceTracker.prx`, `libSceJobManager.prx`,
`libSceNpCppWebApi.prx`, `libScePfs.prx`.

Seven `gamesdk/levels/` directories: `datacenter`, `fortress`, `longshot_map1..3`, `temple`,
`shooting_range_middle_east`. `engine.pak:Config/levelrotation.xml` names
`singleplayer/forest` — which is **not** one of them, so do not use it as a progression oracle
without checking it against a live load.

### `fakelib/`, `ampr_emu.index` and `dlc_emu.ini` are the release group's, not the title's

The dump root carries a `_DUPLEX_/duplex.nfo`, a `fakelib/` directory holding four `.sprx` stubs
(`libSceAmpr`, `libSceAppContent`, `libSceGameUpdate`, `libSceNpEntitlementAccess`), and an
`ampr_emu.index` (magic `AMPRIDX3`). **These are packaging artifacts of the release, not files the
title loads**, and reading them as a statement about the title's imports is wrong — see
*Ruled out* below, where that exact inference is falsified.

`dlc_emu.ini` is the one of the four prosper already consumes: `hle_addcontent.cpp:689` reads it as
the local add-content inventory. It declares **21 `[PSAC]` records**, each an `INSTALLED`
content id mounted at `/app0/dlcN`, and the matching `dlcN/dlc.xml` files carry the human labels
(`dlcSkullBones`, `dlcWildPlains`, …). prosper derives the entitlement label from the content id's
last 16 characters, which here are numeric (`UP4321-PPSA03130_00-1839829012691776` → `1839829012691776`).

**This is the local-inventory case the charter describes, at more scale than any other tracked
title, and both failure directions are defects.** Answering "owned" unconditionally would make
prosper perform the circumvention itself. Under-reporting the 21 that ARE declared is equally a
defect and fails in the direction that looks safe — if this title's UI ever says content is
missing, check prosper's inventory answer before believing the title.

### SDK 4.00 — below the post-submit-visibility gate

`param.json` declares `sdkVersion 0x0400000000000000`, i.e. **SDK 4.00**. prosper arms its
post-submit completion-visibility contract only for SDK ≥ 13, so this title is in the same
pre-13 class as *ArcRunner* (SDK 10) and *Crisis Core* (SDK 10). If an early death appears that
looks like the submit race of #2219, `PROSPER_POST_SUBMIT_VISIBILITY=1` is the known lever — but a
frame that needs a non-default switch is not default-route evidence, exactly as tracker #1817
holds for *ArcRunner*.

## Ruled out

One line per hypothesis that has been killed, so nobody re-derives it at full cost.

| Hypothesis | Why it is dead |
| --- | --- |
| **This title hits the `libSceAmpr` AMM wall, so it should coordinate with the Yakuza Kiwami lane.** | The string `Ampr` appears **zero** times in `eboot.bin` and in all seven `sce_module/*.prx`. Module imports are recorded by name in the dynamic section, so a library this title imported would have its name in those bytes. The `fakelib/libSceAmpr.sprx` and `ampr_emu.index` that suggested it are DUPLEX packaging shipped on their releases regardless of what a title imports — the same four `fakelib` stubs appear whether or not the game uses them. **No coordination needed; this title imports none of `libSceAmpr`.** Measured 2026-08-21. |
| **`tools/screenshot --timeout` bounds a run, so a run that overran its `--timeout` says something about the title.** | It does not, and cannot, during boot. The deadline is checked at the top of the sampling loop, and that loop is only reached **after `boot_program()` returns** (`tools/screenshot/screenshot.cpp`: `register_live_renderer` → `boot_program(...)` → the `while (saved < count)` loop). `boot_program()` ends by calling `run_guest_inits()`, i.e. guest code, so a title can be inside it indefinitely with the tool's own limit inoperative. A run stopped there has been killed from outside and **its silence is not evidence** (instrument traps 197, 213). |
| **The boot phases `boot_program()` records are observable somewhere, so "where did boot stop?" just needs the right switch.** | They were recorded and **unreachable in every build the project shipped**. `PROSPER_DIAGNOSTICS` is off by default and `CMakeLists.txt` then excluded the whole of `src/diagnostics/` from `prosper_core`; on top of that, `DiagnosticContext::enable()` was never called anywhere in the tree and nothing ever subscribed to the event bus. Three independent reasons, any one of them sufficient. Fixed here by `PROSPER_BOOTPHASE=1` (`src/diagnostics/boot_phase_log.cpp`), which is a **runtime** switch precisely because a title that hangs is one nobody had already instrumented for. |
