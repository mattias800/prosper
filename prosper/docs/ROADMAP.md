# Roadmap — `prosper`

Honest framing: a full PS5→PC layer that runs a commercial Unity title to a
playable state is a **multi-year, reference-implementation-scale effort** (cf.
shadPS4 for PS4). It is *not* infeasible for this title — the code is unencrypted
and x86-64 — but the GPU translation + shader recompiler alone are enormous.

We build it as a sequence of **independently verifiable milestones**. Each one
produces something concrete you can run and check.

## Verification philosophy — agentic-first (non-negotiable)

This project is built to be completed by **AI agents with no human in the loop**.
Therefore **no milestone is "done" on human observation.** Every `Verify:` step
below is a **programmatic, headless, self-checking** gate:
- automated tests + assertions (exit code = truth),
- structured, greppable logs (per-module call tracing behind flags),
- deterministic golden snapshots (memory maps, call traces, **framebuffer pixel
  hashes**) checked by CI, not by eye,
- debugging tooling shipped alongside each feature (map dumpers, packet loggers,
  disassembly helpers, headless render + hash compare).
"A window opens" is never a criterion; "test asserts swapchain created, N frames
presented, framebuffer CRC == golden" is. Each change adds the check that proves it.

---

## Current status (2026-07-12) - at a glance

- **Messenger reaches and renders the first level:** intro, title, menus, save list, dialogue, player,
  background, foreground tree/terrain, water, and structures render at native 1920×1080. The causal fixes
  are the grading-LUT producer/target extent (#528), Vulkan front-face translation (#534), and independent
  depth/stencil aspect validity in the persistent guest-surface cache (#541). The full-resolution route was
  user-confirmed against the hardware reference; #299, #300, #522, #530, and #540 are closed.
- **The investigation tools are operational:** #514 provides versioned live GPU capture/replay with exact
  live/replay hashes and per-draw/resource isolation; #515 provides reflected SPIR-V/runtime descriptor
  validation in live and offline replay paths. Producer provenance, normal screenshot capture, and the local
  content-metric snapshot guard complete the current first-bad-contract workflow.
- **Dead Cells cross-title milestone:** deterministic routing now reaches gameplay. HUD and partial composition
  render, but the world is mostly white (#566). Version-4 captures seed temporal render targets (#568) and isolate
  the first bad composition at draw 18; its missing 642x362 input has no prior color-target writer. The corrected
  dispatch thread/local/group contract (#580), `sceAgcCbSetShRegistersDirect`, and compute direct type-1 V#
  decoding (#574) now execute a valid registered fill program against real guest buffers before submit completion
  (#576). Range provenance proved one submit orders a 642x362 consumer, dispatch 5's fill, then another consumer.
  Graphics spans and compute now execute by that retained PM4 order (#584). Per-consumer hashes and target-prefix
  metrics proved the later overbright screenshot was a warmup artifact: a skipped temporal RTT fell back to an
  all-`0xFF` compute backing, then the bad copy persisted. Preserving the 642x362 producers renders real geometry.
  The faithful playable closure subsequently isolated a second lifetime bug: one detached Vulkan depth image
  accumulated across 883 submits even though a supported compute kernel filled its 32 KiB HTILE allocation
  with `0xfffffff0` before every scene pass. Timeline-v5 depth backing hashes/writer provenance found the exact
  writer, and guest GPU writes now invalidate overlapping persistent DS cache entries (#611). A routed live A/B
  restores the layers rejected without that boundary.
  The four remaining fragment failures were uniform VCCZ-exit light loops; narrowly proved vertex/fragment
  structurization now restores them while varying and compute-wave forms still reject (#615). Capture v7 now
  records bounded raw stages, exact rejection opcode/PC, decoded state, and resource/descriptor summaries for
  every observed realization failure, so the next unsupported shader can be reduced offline (#618).
- **Immediate milestone:** resolve exact depth/stencil checkpointing (#569). Keep extending the workflow across
  Unity and Unreal targets rather than returning to long unstructured live traces. Timeline v6 closes the stale
  Dead Cells checkpoint problem with offline target-span discovery and a two-route semantic selector (#594), and
  the splash guard uses a measured run-level content threshold (#596/#573).

## Historical status (2026-07-05) - retained for the milestone log

*(The milestone log below is a historical, append-only record. Claims in it are not current status.)*

- ✅ **M0–M3 done:** loader (SELF/ELF → relocatable image → multi-module link → NID binding), host
  execution (mmap + import-trap dispatch), and enough libkernel/libc/services that the game boots
  **through** IL2CPP init (GC + thread pool, stop-the-world suspension solved) into Unity's `GfxDevice`.
- 🟢 **M4/M5 graphics — substantially built:**
  - **AGC command frontend complete** — `sceAgcCreateShader` + `sceAgcDriverSubmitDcb`; zero
    unimplemented `libSceAgc` calls in the boot; a real Dcb decodes → `GpuState` (test-locked).
  - **AGC→Vulkan pipeline** — `pm4_decode → command_processor → render_state/resolve_pipeline_state
    → vk_translate → real VkGraphicsPipeline`, with topology/blend/depth/write-mask pixel-verified,
    plus the `gpu_resources` contract (Buffer/Texture/RT/Depth/ShaderProgram/Pipeline).
  - **RDNA2→SPIR-V recompiler** — full ALU (SOP1/2/K/C, VOP1/2/C/3: arith/convert/compare/select/
    bitfield/pack, `mul_hi` via OpU/SMulExtended, 64-bit mask ops `s_wqm`/`s_cselect_b64`, `v_mad_f32`
    from gfx10.1); **divergent control flow** (EXEC predication, `saveexec`/restore, forward
    `s_cbranch_execz`, SCC); the full **resource-binding memory path** — `SMEM` constant buffers,
    `MUBUF` vertex-fetch + format load/store, `MIMG` sampled textures (`image_sample`/`_l`/`_lz`,
    `image_load`) AND compute **storage images** (`image_load`/`image_store`, 1D/2D/3D + 1D/2D_ARRAY,
    NSA split-address coords), `LDS`+`s_barrier`, `VINTRP`; and trivial (all-DWORD) `SDWA`. **~86%
    instruction coverage; 30 of 41 real game shaders fully recompile in-context** — every bring-up-
    critical class (position/blit/clear, textured, interpolated, image-copy, integer-divide). All
    strictly `spirv-val`-validated and, where a harness exists, execution-differential-tested on real
    Vulkan (`recompile_coverage`/`shader_histo`/`test_rdna2_to_spirv`). Remaining 11 shaders need deep
    features only: **structured control-flow reconstruction** (uniform branches + loops → SPIR-V
    if/else + `OpPhi`), **NGG** primitive-shader preambles (`s_sendmsg`/`exp prim`), and cube/MSAA
    image dims — none needed for the initial frame.
  - **`GpuState → recompiled shaders → VkPipeline → frame`** spine proven end-to-end (pixel-verified).
- 🚧 **Live boot blocker (root-caused; needs an interactive/reference-backed session):** the game boots
  through IL2CPP into Unity's `GfxDevice` bring-up and faults **before any draw**, dereferencing a null
  GPU companion `[pipeline+0x140]` for category-{5,9,15} pipelines during a GC/deferred-release drain.
  The CPU-side GfxDevice object graph comes up **systematically null**. Root-cause work **corrected three
  successive misattributions** (the "completion-event-driven residency pass", the `0xd58710`
  reflection-predicate, and the `k3GhuSNmBLU`/type-store probe designs — all disproven with evidence);
  the remaining question is which absent/stubbed AGC-GPU-resource call should build that companion.
  Autonomous probe-cycles return corrections, not a fix, at high cost — this needs a focused, likely
  reference-backed debugging session. The recompiler + pipeline are staged and verified: the moment this
  wall falls and real draws flow through `run_command_buffer`, the spine above renders them. Full brief:
  `docs/GFXDEVICE_BRINGUP_PROBLEM.md`.
- 🔁 **Multi-agent + CI:** developed by parallel agents (recompiler/back-half, AGC/host front-half,
  infra/review) over a branch-protected `master`; GitHub Actions builds + `ctest` on Linux + Windows.

---

### M0 — Recon & tooling ✅ (done)
- [x] Identify format (unencrypted SELF, x86-64, FreeBSD ABI).
- [x] `self_dump`: SELF/ELF parser, VA mapping, dynamic tags, NID import extraction.
- [x] Full HLE work-list: every imported library + function count (see FINDINGS.md).

### M1 — Loader  🟡 (core done & tested; deps/TLS/real-mmap pending)
Turn the file into a resident, relocated guest image in host memory.
- [x] Parse SELF→ELF; build VA→file map; collect segments/dynamic/symbols/relocs/imports
      (`src/self/module.{hpp,cpp}`).
- [x] Build flat image at a chosen guest base; per-segment protection records.
- [x] Apply relocations (`R_X86_64_RELATIVE`/`GLOB_DAT`/`JUMP_SLOT`). *(TLS relocs deferred to M3.)*
- [x] Bind imports to stub slots (mechanism for M2's trap stubs).
- [x] Locate entry; assert it lands in an executable segment with real code.
- [ ] Real host backing: `mmap`(`MAP_FIXED`)/`VirtualAlloc` behind the image (M2, Linux).
- [ ] Load dependent `*.prx`; build module graph; unify address space.
- [ ] TLS templates, `sce_process_param` wiring.
- **Verify (programmatic, GREEN):** `tests/test_module.cpp` — 24/24 checks against the
  real `eboot.bin`: identity, segment/import counts (612 imports / 35 libs), reloc
  counts, and spot-checked relocation results (JUMP_SLOT→stub, RELATIVE→base+addend)
  read back from the built image. Runs hermetically via `ctest` (static binary).

### M2 — First execution  🟢 (core done; readable names + real dispatch next)
- [x] Real host backing: `mmap(MAP_FIXED_NOREPLACE)` maps the relocated image at
      its guest base as executable; import stub region mapped `PROT_NONE`
      (`src/host/exec_image_linux.cpp`).
- [x] Import trap: `SIGSEGV` handler identifies the faulting stub → `lib::NID`.
- [x] Minimal bootstrap: SysV initial stack + `argc/argv` block; jump to entry.
- [x] **Guest executes**: `eboot` crt runs and traps at its **first Sony call**
      (`libc::bzQExy189ZI`). *This is the "it's alive" moment.* ✅
- [x] NID hash (SHA1 + 16-byte salt + Sony base64) + name↔NID DB; traps now print
      readable names (`src/hle/nid.{hpp,cpp}`). Validated: `memcpy`→`Q3VBxCXhUHs`,
      105/159 dictionary names resolve real eboot imports.
- [x] Real HLE **dispatch**: per-import executable stubs — implemented imports
      tail-jump to a C handler (args intact); unimplemented ones log + return 0 so the
      boot advances (`src/hle/dispatch.*`, stubs in `exec_image_linux.cpp`).
- [x] First HLE module: `src/hle/hle_libc.cpp` — libc thunks (mem/str/heap) + CRT
      no-ops, registered by NID. Boot trace now shows the real startup call order.
- [x] Stack alignment fixed (Sony crt wants entry rsp ≡ 8 mod 16); boot clears
      C++ static init.
- [x] NID name DB integrated: `known_names.txt` (idc/ps4libdoc, 42k names) auto-loaded
      into `NidDb` → 476/612 imports named in traces. (Fetch via `tools/fetch_niddb.sh`.)
- [x] libc: mem/str/heap (+`memalign`), `std::call_once` (`_Execute_once`), CRT no-ops.
- [x] libkernel: pthread mutex/cond (+attrs), `scePthreadSelf`; virtual/direct **memory**
      (`sceKernelReserveVirtualRange`, `MapNamedFlexibleMemory`, `AllocateDirectMemory`,
      `MapDirectMemory`, `Munmap`, `Mprotect`, `VirtualQuery`) backed by host mmap.
- [x] Memory model: track mappings; honor reservation alignment; `VirtualQuery` reports
      real regions/holes; `DirectMemoryQuery` + direct-memory tracking.
- [x] Time/clock sources (advancing) — broke a wait-for-time spin.
- [x] **Real multithreading**: `scePthreadCreate` runs the guest thread entry on a host
      pthread (ABI matches); join/detach/attrs; C11 `_Mtx_*`/`_Cnd_*`.
- [x] Assorted stubs: `sceSysmoduleLoadModule`, `sceKernelLoadStartModule`, UUID, sched hints.
- [x] Thread-safe fault handling (per-thread recovery point); crash-proof forked boot test.
- **Boot now runs deep**: crt → C++/`call_once` init → pthread/mutex init → heap →
  virtual+direct memory setup → time → **spawns worker threads** → reaches game
  `Il2cppUserAssemblies` imports, `libScePosix` file I/O, and locale/ctype init
  (~17 distinct unimplemented calls deep).
- [x] **Multi-module dynamic linker** (`src/loader/linker.cpp`): loads the main exe +
      dependent PRX (esp. **`Il2cppUserAssemblies` — the game's compiled C#**) into one
      address space, builds a global export table, resolves imports to real cross-module
      targets or dedup'd HLE stub slots. (3 modules, 914 imports, 11 cross-module.)
- [x] C++ runtime: `operator new`/`delete` (all variants) → host heap.
- [x] stdio: `printf`/`puts`/`snprintf`/`v*` (va_list forwarded) → host.
- [x] File I/O (`src/hle/hle_file.cpp`): stdio `FILE*` + POSIX fd + `sceKernelOpen/…`,
      with **`/app0` → dump-dir path translation** (real asset loading).
- **Now**: the game's own IL2CPP code executes; boot advances through crt → C++/threads →
  memory → stdio → file I/O → locale/ctype init (`_Getpctype`, verified bound + called) → PS5
  services → **graphics device init**, where it terminates in `GfxDevicePS5SharedData::CreateWorkload`
  on a null `std::ctype` facet table (`eboot+0x3b5ea6`). See `docs/GRAPHICS.md` for the verified
  chain — this is a graphics-path facet-construction gap, not the boot-time locale issue once assumed.
- [x] Dependent-module `init_array` (C++ global ctors) now run before entry — the key
      unlock that let IL2CPP's runtime initialize.
- [x] locale/ctype tables; `sync_on_address` futex (Linux `futex(2)`); C++ new/delete; stdio.
- [x] Fault backtrace (rbp-chain walk) for debugging deep crashes.
- **The game's `main()` runs and prints output** (`"Argument Count = 1 … /app0/eboot.bin"`),
  then initializes PS5 services: user (accessibility), NP/online state, **controller
  (`scePadOpen`)**, mouse, AppContent, CommonDialog.
- [x] PS5 system services (`src/hle/hle_service.cpp`): user (initial user, name, accessibility),
      NP/online (signed-out state, account), pad/mouse (open→handle, zeroed state), app content,
      dialogs — openers return handles, queries zero their output + report sane state.
- [x] pthread TLS **keys** (`pthread_key_create`/`get`/`setspecific`) → host pthread keys (IL2CPP TLS).
- [x] `boot_trace` debug tool (`tools/boot_trace/`): links all modules, boots, prints the
      unimplemented-call trace + register state + module-classified rbp backtrace on fault.
- **Now**: boots deep into the **IL2CPP runtime** (backtrace: main→eboot→Il2cpp+0x107752→…→0x13ade0).
- [x] Root-caused & fixed the IL2CPP-init stack smash: `fstat` was memcpy'ing 144-byte Linux
      `struct stat` into the guest's **0x78-byte FreeBSD `SceKernelStat`** buffer, smashing a
      canary. Now translated to the correct FreeBSD layout (`hle_file.cpp:to_sce_stat`).
- **Now: IL2CPP loads the game's C# metadata.** The guest opens
  `/app0/Media/Metadata/global-metadata.dat`, `/dev/urandom`, dev-log paths, and reaches
  **IL2CPP internal-call / type resolution**, initializing event flags, semaphores, an
  exception handler, thread-stack queries, `setjmp`, `dlsym`.
- [x] GC thread-stack queries (`scePthreadAttrGet`/`Getstackaddr`/`Getstacksize`) via
      `pthread_getattr_np` (real bounds for IL2CPP's GC root scanning).
- [x] **Historical 2025 frontier — graphics required.** IL2CPP aborted during internal-call resolution
      on `UnityEngine.GL::Internal_SetRTSimple_Injected` (SetRenderTarget) at `Il2cpp+0x110ea`.
      The rendering icalls are registered when the engine's **graphics device initializes**, so
      this established that the boot genuinely needed the GPU path. M4 now runs through gameplay;
      see the status summary in `README.md` and current GitHub issues for the active frontier.
### M3.5 — Threading correctness (in progress)
- [x] **GC "unknown thread" fixed**: `pthread_equal` was unregistered (POSIX name) so the GC's
      thread-table search never matched. Root cause found via gdb (empty thread table → the
      compare import always returned 0). Registered `pthread_equal`.
- [x] `scePthreadSelf`/`pthread_self` return the real, unique host thread id (was a shared
      static TCB — broke every per-thread lookup).
- [x] Registered the **POSIX `pthread_*` aliases** (create/join/mutex/cond/attr/keys) — the
      game uses both Sony and POSIX names; guest libc is FreeBSD-style (pointer pthread types).
- [x] Real **event flags + semaphores** (mutex+condvar), **`sync_on_address` futex** (Linux futex).
- [x] Real **per-thread stack tracking** (main + workers we spawn) — replaced the crash-prone
      `pthread_getattr_np` so the GC gets accurate stack bounds. (Correctness, not a fake.)
- **Now: the engine fully initializes** — spawns a **14-thread job-system pool** (all correctly
  parked on the futex awaiting work). The main thread is parked on a `sync_on_address` address.
- [x] Confirmed `sceKernelWaitOnAddress`/`WakeByAddress` ABI (`wait(addr,expected,size,timeout)`,
      `wake(addr,count)`) — my `FUTEX_WAIT(addr,expected)` matches; verified via `PROSPER_SYNCLOG`.
- [x] Instrumentation: `PROSPER_SYNCLOG` logs every wait/wake (tid, addr, value) + every
      `pthread_create` (entry, arg, name). Invaluable for concurrency diagnosis.
- **Diagnosed the deadlock precisely (via synclog + gdb):** the runtime creates 13 eboot
  `AssetGarbageCollectorHelper` threads (all handshake + park correctly) and **one special IL2CPP
  thread** (wrapper `Il2cpp+0x170570`: store tid → `pthread_detach` → run body). The main thread
  finishes 13 handshakes, then waits on a semaphore during `il2cpp_init`. That special thread is
  the **GC thread** — it ran its body and is now blocked in `pthread_cond_wait` deep in the Boehm
  GC core (`Il2cpp+0x4870/0x514e/0x4a6f/0x7ed3`). So: **GC-thread startup handshake deadlock** —
  main waits for the GC thread to reach ready; the GC thread waits on a condvar for a trigger.
- [x] Instrumented cond vars + sceKernel semaphores. Revealed the GC's cooperative
      thread-suspension: **`SuspendSemaphore`/`ResumeSemaphore`** (init=0). PS5 can't use
      signal-based suspension, so Boehm/IL2CPP GC suspends threads via these semaphores.
- **Deadlock structure (precise):** main, in `il2cpp_init`, blocks in `0x1280c0` acquiring a
  runtime lock `L = *(r14+0x90)` (`call 0x1e2390` @0x1285fc) — a **mutex held by another thread**.
  The GC thread is parked on `SuspendSemaphore` (needs a resume). So a thread holds `L` across a
  GC stop-the-world suspend, and main wants `L` → classic stop-the-world lock-ordering deadlock.
- [x] **SOLVED — GC stop-the-world deadlock.** It was the IL2CPP GC's thread suspension:
      `sceKernelInstallExceptionHandler`(type 0x1e) + `sceKernelRaiseException`(thread,0x1e) were
      stubbed→0, so threads were never interrupted/ack'd and the collector waited forever on
      `SuspendSemaphore`. Implemented real async exception delivery (`pthread_sigqueue` RT signal →
      SA_SIGINFO handler synthesises a FreeBSD mcontext → runs the guest handler on the target
      thread) + real `setjmp`/`longjmp` (GC root-register flush). Boot now runs far past GC init.
- [x] **SOLVED — GC stack scanning** (two more fixes after the deadlock): exception context sp is
      at offset **0xf8** (set context[0xf8]=rsp; fixes "GC_push_all_stacks: sp not set!"); and
      **scePthreadAttrGet(thread, attr)** takes the attr in **arg1** not arg0 (k_attr_get was reading
      arg0 → "Bad stack base in GC_register_my_thread"). Boot now runs deep into il2cpp_init; the
      guest main() prints its args. (The earlier "marshaling" abort read was a wrong-rodata-offset
      artifact — the real messages were GC errors.)
- [x] **IL2CPP init COMPLETES.** The rgctx null-deref was bypassed by making `sceKernelDlsym` honest
      (return ESRCH for "scriptingGetMem" instead of fake success, keeping the caller's fallback) —
      the game then takes the correct path and finishes `il2cpp_init`.
- [x] **Runtime startup runs.** Auto-dismiss `sceMsgDialog`, hide splash, real `sceKernelUsleep`.
- [x] **Reaches the GPU pipeline.** Headless libSceAgc/libSceVideoOut + event queues (hle_graphics.cpp,
      hle_kernel_time.cpp): the game opens the display, drives its flip/render loop, and enters the
      GPU memory/command path. Guest `main()` prints; boots through the entire non-graphics runtime.
- [ ] **FRONTIER — the graphics stack (M4/M5).** Fault at `eboot+0x3a938c` reading a table at a GPU
      virtual address `0x100000000`. Needs the real GPU work: unified GPU-memory model, **libSceAgc →
      Vulkan** command-buffer translation, an **RDNA2 shader recompiler**, and **libSceVideoOut** window/
      swapchain. Recommended start: real libSceVideoOut (window + Vulkan swapchain), then AGC cmd decode.

### M4 — Window + VideoOut + graphics-device init
- [ ] `libSceVideoOut` → open an SDL3 window + Vulkan swapchain (headless offscreen for tests).
- [ ] Enough `libSceAgc`/`AgcDriver` + video-out for the engine's graphics device to init and
      register its rendering internal calls (unblocks the IL2CPP abort).
### M5 — AGC → Vulkan + shader recompiler → first frame
- **Verify (programmatic, GREEN):** `tests/test_trap_linux.cpp` (map + identify 4
  representative imports) and `tests/test_boot_linux.cpp` (jump into real entry,
  assert it reaches an import trap). Both headless, exit-code = truth.

### M3 — Kernel + libc/posix (reach Unity init)
- [ ] Memory: flexible/direct memory, `mmap`/`VirtualAlloc`, protections.
- [ ] Threads, mutex/cond/sema/event-flags, `sync_on_address`.
- [ ] File I/O with `/app0` → dump dir; timers, clocks, `sceKernelRtc`.
- [ ] `libc`/`libScePosix`/`libSceLibcInternal` — thunk to host libc where safe.
- **Verify:** guest advances through Unity/IL2CPP runtime init, loads assets,
  and calls into `libSceVideoOut`/`libSceAgc` (i.e., it *wants to draw*).

### M4 — Window + VideoOut + AGC capture
- [ ] `libSceVideoOut` → SDL3 window + Vulkan swapchain; flip/vsync.
- [ ] `libSceAgc`/`AgcDriver` → capture command-buffer submissions; log/parse packets.
- **Verify (programmatic):** headless run asserts swapchain created + ≥N flips
  presented; per-frame GPU command stream dumped to file and asserted to contain
  the expected packet opcodes. No human viewing.

### M5 — GPU translation + shader recompiler (the big one)
- [ ] Command/packet → Vulkan pipeline + draw translation; resource/descriptor mgmt.
- [ ] Shader recompiler: GPU ISA decode → IR → SPIR-V; resource binding model.
- **Verify (programmatic):** headless render to offscreen target; **framebuffer
  pixel-hash matches a golden snapshot** (captured once, then regression-gated).
  Shader recompiler has unit tests: ISA fixture → expected SPIR-V/behavior.

### M6 — Input + Audio
- [ ] `libScePad` → SDL gamepad. `libSceAudioOut(2)` → host audio.
- [ ] `libSceAjm` (ATRAC9/AAC decode); `libSceAvPlayer` (cutscenes) via ffmpeg.
- **Verify (programmatic):** inject **synthetic pad input** → assert guest state
  transitions in the call trace; assert audio-out ringbuffer receives non-silent
  PCM (RMS > threshold); `libSceAjm` decode of a fixture matches a reference hash.

### M7 — Services & polish (playable)
- [ ] `libSceSaveData` → host files; `libSceSystemService`/`UserService`.
- [ ] `libSceNp*` stubs (single-player: trophies/presence no-op or local).
- [ ] Dialogs (`MsgDialog`, `Ime`, `CommonDialog`).
- **Verify (programmatic):** a **scripted input sequence** drives boot→menu→gameplay
  with no human; save then reload asserts state round-trips byte-for-byte; the full
  boot call-trace is diffed against a golden trace.

---

## Cross-cutting
- **ABI shim** (Windows host only): SysV⇄MS-x64 trampolines at every call boundary.
- **Tracing**: every HLE call logged behind a flag; per-module enable.
- **Test harness**: golden memory-map & call-trace snapshots to catch regressions.

### RESOLVED (2026-07-04, merged to master) — load the real `sce_module/libc.prx` instead of HLE-ing libc

**DONE.** The real-libc line was validated and merged to master: master now loads real `libc.prx`, with
general-dynamic TLS, procparam, POSIX `_`-wrappers, rwlock/once, and `gettid`-based fault recovery.
eboot's 145 libc imports bind to real Sony code; the boot reaches the same terminal point (the libSceAgc
frontier) with real libc. Original analysis kept below for context.

### (historical) OPEN DECISION — load the real `sce_module/libc.prx` instead of HLE-ing libc? (top correctness lever)

The dump ships the **real Sony `sce_module/libc.prx`** (1.8 MB; 2921 exports, imports only 117: 113
libkernel + 3 libSceLibcInternalExt + 1 libSceSysmodule). We currently **HLE** libc instead of loading
it — eboot imports **145 functions from libc** (+ 37 from libScePosix) that all resolve to our HLE
approximations. Loading the real libc.prx (as we already load PS5Util.prx / Il2cppUserAssemblies.prx)
would replace those ~145 stubs with **real Sony code** — the single biggest stub-reduction +
correctness win available, and independent of the (blocked) libSceAgc graphics frontier. It would also
give real ctype/locale/malloc/stdio behaviour instead of our approximations.

**Why it's a decision, not a quick edit:** this shifts the HLE boundary from *above* libc to *below*
it (HLE the libkernel/syscall layer that libc.prx imports). It is **all-or-nothing** — real libc's
malloc arena, `errno`/TLS, locale, and stdio must all initialize and interoperate together; you can't
mix HLE `malloc` with real `free`. Real libc init also exercises libkernel paths our partial HLE may
not cover yet, so it risks new deadlocks/crashes that need iterative debugging against the boot. Given
the boot currently reaches graphics init, flipping this unilaterally risks regressing a working state.

**FEASIBILITY CONFIRMED (branch `libc-prx-integration`, 2026-07-04).** Adding libc.prx to boot_trace's
module list works: it **loads, links, and inits** — cross-module resolutions jump 11 → **329** (318
more imports bind to real Sony libc instead of HLE). So the mechanism is sound. The boot then regresses
(crashes at `eboot+0x8065ee` via `libc.prx+0x7f42b`) because real libc needs libkernel machinery we stub
to 0. Work-list surfaced by the run:
- **`__tls_get_addr` + ELF TLS — the crux and the hard part.** Real libc is TLS-heavy (errno, locale,
  stdio state). This needs: loader parsing each module's `PT_TLS`, handling `R_X86_64_DTPMOD64/
  DTPOFF64/TPOFF64` relocs, and per-thread TLS blocks. **The deep problem: initial-exec TLS uses `%fs`
  directly (the TCB), which collides with the host glibc's `%fs`.** Resolving that (swap `%fs` at the
  guest/host boundary, or an emulated-TLS scheme) is a Wine-class problem — almost certainly *why*
  HLE-libc was the original choice. This is the real cost of the decision.
- `scePthreadOnce`, `scePthreadRwlock{Init,Destroy,Rd/Wr/Unlock}` — trivial pthread wrappers.
- `_sceKernelSetThreadDtors` / `_sceKernelSetThreadAtexitCount/Report` — thread atexit bookkeeping.

**Recommendation:** the branch is the proven starting point. The gating question is whether to solve
guest `%fs`/TLS (needed for real libc, and eventually for any real system module). If yes, that TLS/`%fs`
work is the prerequisite; the rest (thread wrappers, dropping libc HLE regs) is mechanical, guarded by
`test_boot_linux`. High value, **medium-high risk (the `%fs` problem)**, several sessions.
**libSceNpCppWebApi.prx is also present** (same opportunity, lower priority). libSceAgc/AgcDriver are
NOT in the dump (system firmware) — those must be HLE'd/translated regardless (see `docs/AGC_TRACE.md`).

## Reality checkpoints
- After **M2** we know exactly the call order the game needs — re-prioritize M3+.
- **M5 is the gate.** If the shader recompiler proves intractable for AGC, options
  are: (a) target the simpler subset this 2D-ish game uses, (b) reuse an existing
  RDNA recompiler. The Messenger is 2D/sprite-based → likely a *small* shader set,
  which is a genuinely favorable draw for a first title.
