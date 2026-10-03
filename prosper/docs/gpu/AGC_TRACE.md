# libSceAgc call trace — API reconstruction reference (M4 groundwork)

## ⭐ NAMES RESOLVED via shadPS4 (2026-07-04) — no longer guessing

shadPS4's `src/core/aerolib/aerolib.inl` has a full `STUB("<NID>", sceAgc...)` map (same NID format).
Matching our 28 traced NIDs against it gives real names — the game is **building GPU command buffers
(DCBs) and creating shaders**, exactly the AGC command-submission model (which mirrors shadPS4's GNM
PM4 model). Verified map (× = call count from one boot):

| our NID | real name | × | role |
|---|---|---|---|
| `f3dg2CSgRKY` | **sceAgcCreateShader** | 36 | create a shader object from a blob (hottest) |
| `ZvwO9euwYzc` | **sceAgcDcbSetCxRegistersIndirect** | 25 | DCB: set context registers (indirect) |
| `d-6uF9sZDIU` | **sceAgcSetCxRegIndirectPatchAddRegisters** | 25 | patch the indirect Cx-register list (paired) |
| `wr23dPKyWc0` | **sceAgcCbReleaseMem** | 6 | fence / release-memory (EOP write) |
| `+kSrjIVxKFE` | **sceAgcDcbPushMarker** | 5 | DCB: push debug marker (this is "obj"'s call — obj is a **DCB**) |
| `TRO721eVt4g` | **sceAgcDcbResetQueue** | 5 | DCB: reset/init a command-buffer queue |
| `aJf+j5yntiU` | **sceAgcDcbEventWrite** | 5 | DCB: write a GPU event |
| `-KRzWekV120` | **sceAgcCb*/finalize** (not in map) | 3 | (unnamed; near Cb funcs) |
| `H7uZqCoNuWk` | **sceAgcDcbPopMarker** | 3 | DCB: pop debug marker |
| `VmW0Tdpy420` | **sceAgcDcbWaitRegMem** | 3 | DCB: wait on register/memory |
| `MWiElSNE8j8` | **sceAgcDcbWaitUntilSafeForRendering** | 2 | DCB: barrier |
| `hvUfkUIQcOE` | **sceAgcDcbSetUcRegistersIndirect** | 2 | DCB: set user-config registers (indirect) |
| `6lNcCp+fxi4` | **sceAgcSetUcRegIndirectPatchSetAddress** | 1 | patch Uc-register indirect base |
| `vRoArM9zaIk` | **sceAgcSetUcRegIndirectPatchAddRegisters** | 2 | patch Uc-register indirect list |
| `vcmNN+AAXnY` | **sceAgcSetCxRegIndirectPatchSetAddress** | 1 | patch Cx-register indirect base |
| `0fWWK5uG9rQ` | **sceAgcQueueEndOfPipeActionPatchAddress** | 2 | EOP action patch |
| `3KDcnM3lrcU` | **sceAgcWaitRegMemPatchAddress** | 2 | wait-reg-mem patch |
| `57labkp+rSQ` | **sceAgcDcbAcquireMem** | 4 | fence / acquire-memory |
| `LtTouSCZjHM` | **sceAgcCbNop** | 2 | command-buffer NOP |
| `i1jyy49AjXU` | **sceAgcDcbWriteData** | 2 | DCB: write data words |
| `MM4IZSEYytQ` | **sceAgcDriverSetHsOffchipParam** | 1 | tess hull-shader offchip config |
| `XlNp7jzGiPo` | **sceAgcDriverSetTFRing** | 1 | tessellation-factor ring setup |
| `Zw7uUVPulbw` | **sceAgcDriverGetEqContextId** | 2 | event-queue context id |
| `w2rJhmD+dsE` | **sceAgcDriverAddEqEvent** | 2 | add event-queue event |
| `h9z6+0hEydk` | **sceAgcSuspendPoint** | many | TRC R5089 suspend point (the spin-loop msg) |
| `23LRUSvYu1M`, `BfBDZGbti7A`, `V++UgBtQhn0`, `fPSCdQxgpSw` | (not in shadPS4 map) | | device/misc |

**Implications:**
- The faulting "obj" is a **DCB (Draw Command Buffer)** — `sceAgcDcbPushMarker(obj, …)` appends to it.
  The earlier "std::ctype facet"/"generic object" readings were wrong; the ctype-shaped instructions
  are the game's PM4-building code operating on DCB state.
- To progress: implement the **AGC DCB model** — a command buffer the game's `sceAgcDcb*` calls append
  PM4 to (init via `sceAgcDcbResetQueue`, append via SetRegisters/Marker/EventWrite/WriteData, sync via
  AcquireMem/ReleaseMem/WaitRegMem) — plus **sceAgcCreateShader** (parse the RDNA2 shader blob into an
  object). This is directly analogous to shadPS4's `gnmdriver.cpp` (PM4 emitters) + `liverpool.cpp`
  (PM4 decode). Then `submit` → decode DCB PM4 → Vulkan (the big M5 piece; shadPS4's `video_core` is
  the reference, GPL-2.0 — OK, we open-source).
- **shadPS4 reusability (from survey):** near-drop-in = generic Vulkan engine + IR→SPIR-V backend +
  unified-memory/cache model; rewrite for RDNA2 = shader ISA decoder, register/PM4 tables, V#/T#/S#
  bit layouts, tiling. The AGC HLE surface = shadPS4's 325 `sceAgc*` stub NIDs (a ready backlog).

---

## (original reconstruction from arg patterns — superseded by the names above)

Generated from `PROSPER_GFXLOG=1 ./build-linux/boot_trace <dump>` (see `hle_graphics.cpp`'s per-NID
`glog_thunk`s). The game calls **28 distinct libSceAgc/AgcDriver NIDs**, ~148 times total, before the
first null-object fault. libSceAgc is a **PS5 system library not in our dump**, so we cannot
disassemble these functions — this table reconstructs their likely signatures purely from *how the
game calls them* (call count, guest callsites, and argument patterns). **All roles below are
INFERENCES from arg shape, not confirmed** — confirm against the real AGC API before implementing.

Conventions: `obj` = the GfxDevicePS5SharedData workload object (`r14+0x48`) whose null field faults;
`ctx` = a recurring object pointer (`…9c45a8`); heap addresses (`0x7e…`) vary per run; eboot rodata
pointers (`0x401…`/`0x402…`) are stable and usually **name strings**; callsite offsets are stable.

## Call-frequency profile (implement/understand the hot ones first)

| NID           | ×  | callsites (eboot+) | representative args | inferred role (UNCONFIRMED) |
|---------------|----|--------------------|---------------------|------------------------------|
| `f3dg2CSgRKY` | 36 | 14e769c/7772/7795  | a0..a3 = 4 heap ptrs, a5 ptr | buffer/descriptor copy or command build (src/dst/range) |
| `ZvwO9euwYzc` | 25 | 3b601f             | a0=obj a1=8 a4=0x6a | paired w/ d-6uF9 — per-element emit (a1=size 8) |
| `d-6uF9sZDIU` | 25 | 3b6102             | a0=0 a1=1 a2=8 a3=-1 a4=0x6d | paired w/ ZvwO9 — per-element emit/commit |
| `wr23dPKyWc0` | 6  | 3ae3dc             | a0=ctx a1=0x28 a3=1 a5=ptr | per-field setter/descriptor-add on ctx |
| `+kSrjIVxKFE` | 5  | 148e79f/14c7358/14d6495/14dbb35/14dda81 | a0=obj a1,a3=rodata(names) a4=a5=-1 | **create/label object** (a1,a3 = name strings) |
| `TRO721eVt4g` | 5  | 14bbaec/14e6661/14e6c18 | a0=a3=obj a1=0x3ff a4=obj-0x48 a5=code | init object w/ capacity 0x3ff + callback (a5=code ptr) |
| `aJf+j5yntiU` | 5  | 14dd914/3b649d/3b66a4 | a0=obj a1=0x10 a3=1 | reserve/alloc a 0x10 slot on obj |
| `57labkp+rSQ` | 4  | 3b6761/3b69b8      | a0=obj a1=1 a3=0x9000 a4=packed-str | map/alloc 0x9000 region tagged w/ a name |
| `-KRzWekV120` | 3  | 14bbb0f/14c73b9/14e27d7 | a0=ctx a4=0x6d | finalize/commit on ctx |
| `H7uZqCoNuWk` | 3  | 148e7ba/14c7375/14d73e4 | a0=obj a1=heap ptr a4=0x6c | attach sub-object (a1) to obj |
| `VmW0Tdpy420` | 3  | 3a8423/3b691c      | a0=ctx a2=3 a4=2 a5=ptr | set mode/state (enum a2,a4) |
| `MWiElSNE8j8` | 2  | 14bbb03/14c736d    | a0=ctx a1=0x1001 a4=0x7b | set flag/attr 0x1001 |
| `hvUfkUIQcOE` | 2  | 3b62bf             | a0=obj a1=8 a4=0x6d | like ZvwO9 (size 8 emit) |
| `vRoArM9zaIk` | 2  | 3b63a2             | a0=0 a1=1 a2=8 a3=~ a4=0x6d | like d-6uF9 variant |
| `LtTouSCZjHM` | 2  | 3b6895             | a0=heap a1=0xa a2=ptr a5=code(0x4003ae100) | callback-registration (a5=code) |
| `i1jyy49AjXU` | 2  | 3b68c4             | a0=heap a1=4 a4=&a0-4 a5=1 | read/return a 4-byte field into a4 |
| `V++UgBtQhn0` | 2  | 3b6962             | a0=heap a1=tagged-ptr a4=0x68 | store tagged value |
| `0fWWK5uG9rQ` | 2  | 3b6992             | a0=tagged a1=packed-str a4=0x7c | debug/name-string handling |
| `3KDcnM3lrcU` | 2  | 3b6987             | a0=tagged a1=packed-str a4=0x7c | debug/name-string handling (paired w/ 0fWW) |
| `fPSCdQxgpSw` | 2  | 3b697c             | a0=tagged a1=packed-str a4=0x78 | debug/name-string handling |
| `w2rJhmD+dsE` | 2  | 14bb080/14bb34e    | a0=heap a3=0x50 a5=0x50 | alloc/copy 0x50-byte struct (AgcDriver) |
| `Zw7uUVPulbw` | 2  | 14dfb4f            | a0=heap a5=0x401bb2738(code/data) | AgcDriver submit/config |
| `23LRUSvYu1M` | 1  | 3ad4d2             | a0=0x401f53830 a1=0xd a4=4 | device-level init (a0=rodata) |
| `BfBDZGbti7A` | 1  | 3ad504             | a0=0x401f53838 a4=heap | device-level init (paired w/ 23LR) |
| `XlNp7jzGiPo` | 1  | 14ba816            | a0=heap a1=0x3fff8 a4=0x8000 | alloc big region (0x3fff8/0x8000 = sizes) — AgcDriver |
| `MM4IZSEYytQ` | 1  | 14ba8b5            | a1=0x1ff a3=0x30b8…b6bf | map/register region (AgcDriver) |
| `vcmNN+AAXnY` | 1  | 3b60cd             | a0=0 a1=heap a2=0xfffef a4=0x69 | (near d-6uF9 family) |
| `6lNcCp+fxi4` | 1  | 3b636d             | a0=0 a1=heap a2=0xffbef a4=0x69 | (near d-6uF9 family) |

## Cross-cutting observations

- **Trailing `a4` ∈ 0x5a–0x7c** recurs across many *different* NIDs at *different* callsites (0x68,
  0x69, 0x6a, 0x6c, 0x6d, 0x78, 0x7b, 0x7c). Too clustered to be data — likely a **shared trailing
  parameter**: a command/opcode enum, or a `__LINE__`-style debug context the AGC command-builder
  stamps into each packet. Decoding what selects it is high-value (it may key the object layout).
- **`ZvwO9euwYzc` + `d-6uF9sZDIU` (×25 each)** are the dominant pair — almost certainly the inner
  loop of AGC command/descriptor emission. Their objects (`obj` and `0`/global) and `a1=8` (element
  size) fit a "write N 8-byte packets" model.
- **Packed-string args** (`0x3031366233783030` = ASCII, and eboot rodata pointers in `+kSrjIVxKFE`)
  show several functions take **resource/debug names** — consistent with AGC's labeled GPU objects.
- **Code-pointer args** (`a5=0x4014e6700` in `TRO721eVt4g`, `0x4003ae100` in `LtTouSCZjHM`) are guest
  **callbacks** the AGC object is expected to invoke — these must be wired to actually call back into
  the guest once the objects are real.

## Next steps (M4)

1. Map the trailing `a4` enum: breakpoint the callsites, correlate `a4` with the guest's surrounding
   code / the object field it ends up controlling.
2. Identify `obj`'s layout: which AGC call writes `[obj+0x40]` (the field whose null read faults).
   Candidates by arg shape: `TRO721eVt4g` (init w/ capacity), `+kSrjIVxKFE` (create/label),
   `H7uZqCoNuWk` (attach sub-object).
3. Build real objects for the create/init calls (`+kSrjIVxKFE`, `TRO721eVt4g`, `23LRUSvYu1M`/
   `BfBDZGbti7A` device level), then translate the command-emit pair (`ZvwO9euwYzc`/`d-6uF9sZDIU`) +
   `f3dg2CSgRKY` into Vulkan command-buffer building. **Correctness-first: no plausible-looking
   field fakes — implement against the real AGC contract.**
