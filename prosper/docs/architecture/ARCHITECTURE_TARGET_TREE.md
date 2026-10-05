# Target source tree and layer order

> **Status: proposal.** The current-tree measurements and the layer order, which the ratchet
> enforces, are descriptive. The *target* tree is a direction the project owner has not yet approved:
> the new `src/guest/` layer, the move of the shipping Vulkan backend to `frontends/shared/backend/`,
> and the "Planned moves" table. Read them as a plan to discuss, and don't start a move from this
> document alone.
>
> Approval is now tracked per decision in `docs/adr/`: ADR 0003 (the `guest/` layer, moves 4-6
> and 12) and ADR 0004 (the backend move, moves 1-3) are `proposed` until the owner accepts them.
> The binding rules these moves serve are in `docs/spec/`.

What the source tree looks like now, what it should look like, which way dependencies may point,
and the planned moves that get from one to the other. Every move is a move-only PR made with the
refactor tools; every structural cost this document names is frozen by a ratchet rule so it cannot
grow while the moves are pending.

Companion documents: `docs/architecture/HOST_PLATFORM_SEAM.md` (the OS-service seam in detail),
`docs/architecture/REFACTOR_PLAN_2026_09.md` (file and function splits; this document does not repeat its
measurements), `docs/architecture/ARCHITECTURE.md` (what each component does). The checker is
`tools/ci/check_arch_ratchet.py`; its baseline is `tools/ci/arch_ratchet_baseline.txt`.

Measurements below were taken on main 6499c4d7 plus the ratchet branch, with the checker itself
unless stated otherwise. Re-measure rather than quoting them later.

## Current tree (annotated)

```
prosper/
  src/
    self/          SELF/ELF parsing -> host-independent module images. Includes nothing above it.
    loader/        links modules, builds the export table and import slots. Includes self/ and
                   (inversion) hle/dispatch/ for ImportSlot.
    input/         platform-neutral pad state (pad.hpp/.cpp).
    diagnostics/   observer-only instrumentation. Includes (inversion) gpu/ 3x and
                   frontends/shared/present/ 1x.
    host/          MIXED: OS services AND guest-semantics-on-the-host.
      platform/    lifecycle, gpu_submit_gate, precise_sleep, posix_shim, raw_syscall, immortal
      memory/      guest memory query/copy/search, guest_write_watch
      abi/         System V <-> Microsoft x64 bridge, guest varargs
      tls/         guest TLS block, fs emulation; includes hle/dispatch and loader/tls_layout
      fault/       fault context, trap arbitration, guest stack scan
      image/       per-OS image mapping and execution, boot_program; includes hle/dispatch,
                   loader/, self/
      symbols/     IL2CPP symbol lookup
      x86/         instruction decode helpers
    gpu/           AGC/PM4 decode, recompiler (RDNA2 -> SPIR-V), resources, capture, timeline.
                   Includes (inversion) hle/ in 10 files: save_paths, guest_memory_topology,
                   hle_kernel_time, sync_futex. recompiler/gta5/ is a title-named module.
    hle/           the reimplemented Sony libraries; includes gpu/ (23 lines in 13 files), host/,
                   diagnostics/. 326 host-platform #if directives in 31 files.
  frontends/
    prosper-app/   the shipping SDL3 app
    shared/        live renderer (live/), presentation, compute, device, textures; shared by the
                   app, tools/screenshot and the offscreen backend. Includes tests/fixtures/ in
                   7 files (the Vulkan backend lives there) and uses prosper::test:: (343 refs).
    audio_*/ video_*/ pad_*/ dialog_sdl3/   platform device backends
  tests/
    fixtures/      render_runner.h (16,758 lines) -- THE shipping Vulkan backend, header-only, in
                   namespace prosper::test -- plus real test fixtures and harnesses.
    <area>/        unit, boot and Vulkan-execution tests
```

## Target tree (annotated)

```
prosper/
  src/
    self/          unchanged
    loader/        unchanged, except ImportSlot comes from a lower layer (not hle/dispatch)
    input/         unchanged
    diagnostics/   observer-only; nothing from gpu/ or frontends/ (move the shared helpers down)
    host/          ONLY the OS seam (HOST_PLATFORM_SEAM.md)
      platform/    vm, futex, thread, fiber, fault-install, clock, file -- one interface each,
                   one backend file per OS
    guest/         NEW: guest semantics running on the host, split out of src/host/
      abi/         from host/abi
      tls/         from host/tls
      fault/       from host/fault
      memory/      from host/memory (guest memory query/copy/watch)
      image/       from host/image (mapping and running guest images, boot_program)
    gpu/           decode + recompiler only; reaches guest memory and wakeups through guest/ or
                   an interface it owns, never through hle/. No title-named folders.
    hle/           the Sony libraries; reaches gpu/ through a narrow AGC submission interface
                   instead of 23 scattered includes; no host-platform #if.
  frontends/
    shared/
      backend/     NEW: the Vulkan backend (render_runner.h and its retained_depth_* companions),
                   moved out of tests/fixtures/, namespace prosper::gpu::backend
      live/ ...    as today, including backend/ instead of fixtures/
    prosper-app/   unchanged
  tests/
    fixtures/      test-only harnesses and data again
```

Why `frontends/shared/backend/` and not `src/`: `render_runner.h` includes `frontends/shared/`
twelve times, so a move into `src/` would create twelve `src` -> `frontends` edges. Moving it next
to its consumers is behaviour-free; promoting it into `src/` later needs those dependencies cut
first. `REFACTOR_PLAN_2026_09.md` Phase 2 leaves the same choice open. `CONFIDENCE: MED`.

## Layer order (allowed dependency direction)

Enforced by the `layer-include` rule. `LAYER_ORDER` in `check_arch_ratchet.py` is the single
source; this table must change in the same PR as that constant. Lowest first; a layer may include
itself and anything **above it in this table**, never anything below.

| rank | layer | why it sits here (verified against the includes) |
| ---: | --- | --- |
| 0 | `self` | parses images; includes no other layer |
| 1 | `loader` | includes `self`; "host-agnostic" per `src/loader/AGENTS.md` |
| 2 | `input` | pad state; includes no other layer |
| 3 | `diagnostics` | observer-only; included by `gpu` (35 lines), `hle` (19), `host` (3) |
| 4 | `host` | OS services and host execution; included by `hle` (63 lines) and `gpu` (16) |
| 5 | `gpu` | decode/translate; driven by `hle`'s graphics calls (`hle` includes it 23 times) |
| 6 | `hle` | the top of the core |
| -- | `frontends/`, `tests/` | **never** includable from `prosper/src` (`docs/subsystems/FRONTEND_APP.md`: "the core never depends on the frontend") |

Baselined inversions today (22 rows, 29 include lines): `host` -> `hle` (6 files: `host/image/*`,
`host/tls/guest_tls.cpp`), `loader` -> `hle` (`linker.hpp`, `tls_layout.hpp`), `gpu` -> `hle`
(10 files), `diagnostics` -> `gpu` (3 files), `diagnostics` -> `frontends`
(`readback_reason_census.cpp`).

**Target order**, once `src/guest/` exists: `self` < `loader` < `input` < `diagnostics` < `host`
(the seam) < `guest` < `gpu` < `hle`. `guest` may include `hle/dispatch` only if the dispatch
types it needs have not yet moved down; the cleaner end state moves `ImportSlot` and the dispatch
registry's data types into `guest/` or `loader/`. The frontends sit on top and may include
everything; nothing in `src/` includes them.

How this relates to the "pipeline" reading (`self/loader -> guest -> hle -> host/platform, gpu ->
backend`): that describes the *flow of a guest call*, from loading to OS service and GPU
submission. Includes run the other way — the caller includes the callee's header — so `hle` is at
the top of the include order precisely because it calls everything else.

## Planned moves

Each row is one move-only PR (or one commit, if small), made with the tool named, containing
`git mv` plus include rewrites and nothing else. Behaviour changes that the move enables come in
separate PRs afterwards.

| order | move | tool | ratchet rule that freezes the debt meanwhile |
| ---: | --- | --- | --- |
| 1 | `tests/fixtures/render_runner.h` + `retained_depth_{array,cube}_gpu.h` -> `frontends/shared/backend/` | `tools/refactor/move_module.py` | `fixture-include`, `file-size` |
| 2 | namespace `prosper::test` -> `prosper::gpu::backend` for the moved files (mechanical rename, own commit) | sed-free rename via the compiler; verify with `nm` A/B | `test-dep` |
| 3 | split the backend into translation units (`REFACTOR_PLAN_2026_09.md` Phase 2 step 3) | `tools/refactor/map_symbols.py`, `split_file.py` | `file-size` |
| 4 | `src/host/{abi,tls,fault,memory,image}` -> `src/guest/` | `move_module.py` with a new `guest_layout.txt` beside `host_layout.txt` | `layer-include` |
| 5 | add `guest` to `LAYER_ORDER` between `host` and `gpu`; re-key the moved `layer-include` rows | hand edit, reviewed | `layer-include` |
| 6 | move `ImportSlot` (and the TLS layout type) below `hle` so `loader` and `guest` stop including `hle/dispatch` | `promote_internal.py` / `move_module.py` | `layer-include` |
| 7 | host-platform calls in `src/hle` behind `src/host/platform/` interfaces, file by file (`HOST_PLATFORM_SEAM.md` § Migration order) | hand moves, each its own commit | `platform-ifdef` |
| 8 | `gpu` -> `hle` edges: guest memory topology and futex wakeups reached through `guest/` or a gpu-owned interface | `move_module.py` for the types, hand edits for the inversion | `layer-include` |
| 9 | `hle` -> `gpu`: a narrow AGC submission interface | hand design, then moves | `layer-include` (future: an `hle`/`gpu` row once the interface exists) |
| 10 | `src/gpu/recompiler/gta5/` contracts generalized into properties of the shader/data | hand work | `title-dir` (cap 1,832 lines) |
| 11 | `diagnostics` -> `gpu`/`frontends` helpers moved down into `diagnostics/` | `move_module.py` | `layer-include` |
| 12 | extract the APR engine (command buffer model, executor, completion dialects) out of `hle/memory`, `hle/fs` and `hle/kernel` into a platform-neutral component in `src/guest/`, leaving `hle/` the NID handlers; it reaches the OS only through `src/host/platform/` interfaces. Needs moves 4, 5 and 7 first and owner approval of the proposed extraction. `CONFIDENCE: MED`: placement follows the proposed layering; no APR extraction has tested it. | `move_module.py` for the moves, then hand splitting for the shared state | `platform-ifdef`, `layer-include` |

## Bad practices and who owns each

Every structural cost identified in the ratchet work is either a rule in
`check_arch_ratchet.py` or listed here as not mechanically checkable yet, with the reason.

| practice | rule / owner | baseline today |
| --- | --- | --- |
| title ids in conditions | `title-id` | 143 in 9 files |
| title-named directories | `title-dir` | 1 (`src/gpu/recompiler/gta5`, 1,832 lines) |
| scattered `getenv` | `getenv` | 1,111 in 108 files |
| blocking GPU syncs (fence/queue/device wait, `ALL_COMMANDS` barriers) | `blocking-sync` | 54 in 20 rows |
| oversized files | `file-size` (> 5,000 lines) | 9 files |
| shipping code depending on `prosper::test::` | `test-dep` | 343 in 25 files |
| the shipping backend under `tests/` | `fixture-include` (frontends including `fixtures/`) | 13 in 7 files |
| host-platform `#if` outside `src/host` | `platform-ifdef` | 355 in 44 files |
| reverse layer includes | `layer-include` | 29 in 22 rows |
| Vulkan object creation on hot paths | `vk-object` (call-site count of `vkCreateDescriptorPool`, `vkAllocateMemory`, `vkCreateFence`) -- a static proxy only: it cannot tell per-draw from one-time | 22 in 15 rows |
| a platform arm that is a stub or reduced copy of the other arm's logic | `platform-stub`, a **name-based proxy**: it counts handlers defined as `HLE(<name>_stub)` in `src/{hle,loader,self,gpu}`, so a stub with another name is not seen, and a reduced copy that is not a stub is not seen either. Owned by the `src/hle/memory/hle_kernel_mem.cpp` migration in `HOST_PLATFORM_SEAM.md` (move 7 in the table above). | 2 in 1 file: the Windows arm of `hle_kernel_mem.cpp` stubs `sceAmprAprCommandBufferConstructor` and `sceAmprCommandBufferSetBuffer`, and leaves the AMM handlers unregistered (#2384) |
| giant functions | **not checkable here**: needs a parser to find function extents. Owned by the separate clang-tidy PR as `readability-function-size`. | -- |
| per-draw object creation proven at run time | **not checkable statically**; the steady-state performance invariants in #4193 own it | -- |

## Standards

The platform-arm rule in item 2 is current charter policy, with the partial mechanical guards
named there. The guest placement and engine replay requirements in items 1 and 3 describe the
**proposed** target tree and await owner approval; creating the `guest/` layer requires moves 4
and 5 above. This section does not authorize an extraction or claim that those engines or replay
tests already exist.

1. **`hle/` is the API surface; engines live in `guest/`.** A cross-cutting HLE area such as APR
   (`sceAmpr*`, and the file reads and event-queue completions behind it, spread today over `hle/memory`,
   `hle/fs` and `hle/kernel`) would keep only its NID handlers, argument decoding and return codes in
   `hle/`. Its engine (command buffer model, executor, completion dialects) would live once in
   `src/guest/`, because it implements guest-visible coprocessor semantics rather than API decoding.
   This is the placement proposed in move 12, contingent on moves 4, 5 and 7 and owner approval.
2. **Portable logic is written once; a platform arm holds primitives only.** An engine reaches the OS
   through the `src/host/platform/` interfaces (`vm`, `file`, `clock`, ...) and nothing else, so no
   platform arm can be a stub or reduced copy of another arm's logic. Enforced today only in part:
   `platform-ifdef` and `layer-include` catch directives and includes, and `platform-stub` (a
   name-based proxy landed in #4366) catches handlers named `HLE(*_stub)`; a
   reduced copy that is not named as a stub is a review rule (see the stub-arm row in "Bad
   practices").
3. **Each engine ships a replay test that runs on both hosts.** A recorded command-buffer stream from a
   real title is fed to the engine, and the test asserts what the guest can observe (memory contents
   and delivered events). It extends the seam's one-test-per-interface rule from the OS services to
   the engines above them. The aim is to detect an unported platform arm in a test rather than
   discover it title by title. `CONFIDENCE: LOW` on how practical this is for large streams:
   it has not been tried.

## Historical title observations

The earlier Windows APR fault report is preserved in
[`DRAGON_QUEST_STATUS.md`](../games/DRAGON_QUEST_STATUS.md#historical-windows-apr-observation-preserved-2026-10-04)
and [the title tracker](https://github.com/mattias800/prosper/issues/1874#issuecomment-5976105083).
Its tested source, binary and configuration were unpinned; it establishes neither a current-main
failure nor APR as the cause. Removing it from this architecture proposal does not falsify it.

## Tracking

The umbrella issue for this backlog is listed in the PR that adds this document and is linked from
`tools/ci/AGENTS.md`'s ratchet entry; each planned move has its own issue there.

## Ruled out

- **Moving the Vulkan backend straight into `src/`.** It includes `frontends/shared/` twelve
  times, so the move would add twelve `src` -> `frontends` edges that `layer-include` forbids.
  Measured with `grep '#include "shared/' tests/fixtures/render_runner.h`.
