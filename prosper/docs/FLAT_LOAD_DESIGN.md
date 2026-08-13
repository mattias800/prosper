# General FLAT_LOAD design (#1171) — raw 64-bit-address memory reads in the compute recompiler

**Status:** implemented and regression-covered; retained as a historical design record (2026-07-22).
The raw-address windowed load path described here removed the original GTA V (`PPSA04263`)
`exec_cs_2042d47600` loading-screen blocker. It is not the current explanation for the black 3D world
at gameplay entry; the runtime-selected descriptor lift is complete and current evidence is tracked
by exact failed compute program in #2412/#2481.

## The original problem

A general FLAT load reads from a raw 64-bit guest GPU virtual address held in a VGPR pair. prosper's
recompiler only models FLAT for the compiler's private scratch spill (a per-invocation `SC_Function`
array with no host address — `rdna2_to_spirv.cpp:5206-5241`, `analyze_static_scratch` :137-182). Every
other FLAT/GLOBAL form fails-visible (`ok=false`), so any shader with a real flat load is dropped whole.

The blocking kernel (`shader_inspect exec_cs_2042d47600.bin`) is a linear-buffer → image **decode loop**:

```
pc=0030 v_add_co_u32   v1, vcc, s12, v6      ; v[1:2] = s[12:13] + per-lane offset  (#1170 unblocked this)
pc=0036 v_add_co_u32   v2, vcc, s13, ...     ; high dword of the address
pc=0040 flat_load_ubyte v0, v[1:2]           ; <-- REJECTED: read src byte at a raw 64-bit address
pc=0047-0049 v_mov      v1=v0,v2=v0,v3=v0    ; broadcast byte -> RGBA
pc=0050 image_store     v[0:3], v[4:5], s[0:7]  ; write decoded texel (MIMG already supported)
pc=0052 s_branch pc=3                        ; loop over pixels
```

The address is **`s[12:13]` (a 64-bit base pointer passed as a kernel argument in user-SGPRs) + a
per-lane byte offset**. The base is a *resolvable kernel argument*, not an arbitrary runtime pointer —
this is what makes the design tractable.

## Why not `buffer_device_address` / PhysicalStorageBuffer

The most "general" approach — expose guest memory at its guest VA via `VK_KHR_buffer_device_address` and
dereference the 64-bit pointer directly — is **rejected**:
- The emitted SPIR-V uses the `Logical` addressing model (`rdna2_to_spirv.cpp:1398`), which forbids
  physical-storage pointers. Switching to `PhysicalStorageBuffer64` is a deep, cross-cutting change to
  every emitted module.
- It requires placing host buffers at an exact device address matching the guest VA, which no portable
  Vulkan path guarantees (and llvmpipe, prosper's software renderer, would need it too).
- prosper has **zero** device-address infra today (repo-wide grep: no `bufferDeviceAddress`,
  `PhysicalStorageBuffer`, `VkDeviceAddress`).

## The design: flat-address → windowed SSBO (reuses the existing buffer path)

**Core idea.** Bind the guest allocation that contains the base pointer as an ordinary host-visible
Vulkan SSBO — the *exact same* per-resource create + `memcpy`-from-guest + STORAGE_BUFFER path already
used for MUBUF buffer loads (`live_compute.cpp:838-888`, `ShaderResource{gpu_addr,size,binding}` at
`shader_resources.hpp:99-214`) — and lower the flat load to an indexed read into that SSBO.

Because guest GPU VA == host virtual address (1:1 identity map; `exec_image_linux.cpp:1281`), the base
pointer value *is* a host pointer, and the containing allocation's extent is already discoverable via
`host::guest_readable_mapping_containing(base)` (`guest_memory_map.hpp:141`).

### Recompiler side (`rdna2_to_spirv.cpp`)

1. **`analyze_flat_loads` (new).** For each non-scratch FLAT load (`flat_segment != 1`, VADDR present,
   `saddr == NULL(125)`, `access.valid`), const-fold the VADDR VGPR pair back to
   `base_user_sgpr_pair + residual_offset`. This is the same class of const-fold already implemented for
   descriptor recovery (`resolve_dynamic_fetch`, `add_compute_buffer_resources` at
   `gpu_executor.cpp:2029-2121`), extended to VGPR dataflow tracing a `v_add_co_u32`/`v_add_co_ci_u32`
   pair whose high/low addends are two consecutive user SGPRs. On success record
   `{fetch_pc, base_sgpr_index, bits, sign_extend, dst}`; on failure keep today's fail-visible `ok=false`.
2. **Declare one "flat window" SSBO binding** (`SC_StorageBuffer`, reusing `declare_cbufs` machinery at
   `rdna2_to_spirv.cpp:1326-1355`) assigned by the resource table (below).
3. **Emit the load** as an indexed SSBO read: the shader already computes `flat_address = base + offset`
   in SSA, and `base` (= `s[12:13]`) is already a push-constant (`load_push_constant`, :1371-1378). So
   `byte_offset = low32(flat_address64 - base64)` (which reduces to the residual offset the shader
   already computed), then `SSBO_flat[byte_offset >> 2]` with the existing sub-dword extract used by
   `buffer_load_ubyte/ushort` (`rdna2_to_spirv.cpp:5265-5268`). Read-only: no writeback.

### Executor / resource side (`gpu_executor.cpp` compute discovery + `live_compute.cpp`)

4. **Register a flat-window `ShaderResource`** during compute resource discovery (alongside
   `add_compute_buffer_resources`, `gpu_executor.cpp:2798-2811`): read the traced base pointer from the
   dispatch's user SGPRs — `gpu_addr = user_sgprs[i] | (user_sgprs[i+1] << 32)` (values already read by
   `read_user_sgprs` :2796) — set `size = ` remaining bytes of the containing guest allocation
   (`guest_readable_mapping_containing`, clamped to a max window, e.g. 256 MiB), and assign it a binding
   via `assign_convention_bindings` (:2140-2149).
5. **No new upload code.** `live_compute.cpp:838-888` already creates a host-visible SSBO of `size`,
   `memcpy`s guest bytes from `(uint8_t*)(uintptr_t)gpu_addr`, and binds it as STORAGE_BUFFER, with
   exact-range aliasing (:846-858). The flat window is just another resource in the table.

### What is new vs. reused

| Piece | New? |
|-------|------|
| Guest-memory → host-visible SSBO create + memcpy + bind | **Reused** (`live_compute.cpp:838-888`) |
| `ShaderResource{gpu_addr,size,binding}` + `by_binding` | **Reused** (`shader_resources.hpp`) |
| Push-constant delivery of `s[12:13]` base pointer | **Reused** (`load_push_constant`) |
| Sub-dword (ubyte/ushort) extract from an SSBO dword | **Reused** (MUBUF raw_subword path) |
| Containing-allocation lookup | **Reused** (`guest_readable_mapping_containing`) |
| **Flat-address → base-SGPR-pair + offset const-fold** | **New** (extends `resolve_dynamic_fetch`) |
| **Non-scratch FLAT load emission** using the window binding | **New** (~30 lines in the FLAT case) |
| **Flat-window resource registration** from traced base | **New** (~20 lines in compute discovery) |

## Bounds / correctness

- **Window size.** A raw pointer carries no size (unlike a V#). Bind the containing guest allocation
  (clamped to a max). Out-of-window indices read 0 via SSBO runtime-array robustness or an explicit
  bounds check — a loose match to hardware's bounded-read contract. Confidence: MED; validate the window
  covers the real access range on GTA.
- **Base-tracing fragility.** If the flat address is not a `base_user_sgpr_pair + offset` shape (pointer
  chasing, indirect tables), the const-fold fails and the load stays fail-visible — correct and safe, no
  silent wrong data. Confidence: HIGH that the decode kernel is the simple shape; unknown coverage of
  other titles.
- **FLAT stores** (`0x18/0x1a/0x1c`) are out of scope here (read-only decode). A later store would need
  writeback (the compute path already writes reflected SSBOs back).
- **Non-compute FLAT** (graphics stages) is out of scope; this targets the compute decode path.

## Implementation plan (reviewable pieces)

1. **Base-tracing analysis** — pure, unit-testable in `test_dynfetch_fold` / a new test: assert the
   decode kernel's flat address folds to `{s12, offset}`. No Vulkan.
2. **FLAT load emission** — extend the FLAT case to emit the windowed SSBO read; round-trip via
   `recompile_valu`/`recompile_compute`, spirv-val gated.
3. **Executor flat-window registration** — reuse live_compute; unit-cover the resource shape.
4. **End-to-end** — run GTA, confirm `exec_cs_2042d47600` recompiles + executes and the loading-screen
   collage / legal text render against the user's #1163/#1165 oracles. Snapshot regression + review.

## Empirical validation still to do

- Capture the live dispatch's user SGPRs for `exec_cs_2042d47600` to confirm `s[12:13]` is a valid mapped
  guest pointer in the GPU-memory range (`0x1000000000+`) and see the source allocation's extent.
- Confirm the per-lane offset range stays within the chosen window.

## Stage 4b groundwork — what the SPIR-V emitter needs for an indexed descriptor array (#2412)

Surveyed 2026-08-10 before writing any emitter code, and recorded here because three of the five
findings are things a reader would otherwise conclude the opposite of. Stages 1 and 2 are merged
(#2458, #2460); stage 3 is #2464 and stage 4a is #2463.

**1. `NonUniform` already appears 45 times in `rdna2_to_spirv.cpp`, and NONE of them is the
decoration this needs.** Every hit is `GroupNonUniform*` — `Op_GroupNonUniformAny`,
`Op_GroupNonUniformBallot`, `Cap_GroupNonUniformArithmetic` and friends, i.e. **wave/subgroup
operations**. The descriptor-indexing `NonUniform` is *decoration 5300*, an unrelated concept that
happens to share the word. A `grep -c NonUniform` returning 45 reads exactly like "already
supported" and is the first thing anyone will do. It is not supported: `Dec_NonUniform` does not
exist in the file.

**2. The emitter targets SPIR-V 1.3** (`0x00010300`, the header at the module-assembly site), and the
descriptor-indexing capabilities became core only in **1.5**. So they require
`OpExtension "SPV_EXT_descriptor_indexing"` — either that, or raising the module version, which is a
much larger decision because every consumer and every `spirv-val` target moves with it.

**3. The emitter emits no `OpExtension` at all, and its section list has no slot for one.** Assembly
order is `caps, extimp, mem, entry, exec, debug, deco, types, code`. SPIR-V requires `OpExtension`
**after** `OpCapability` and **before** `OpExtInstImport`, so an `exts` section has to be added
*between the first two* — not appended. Getting that order wrong produces a module that fails
`spirv-val` with a message about layout rather than about descriptors.

**4. The declaration site is `declare_cbufs`'s N-buffer loop.** That loop already walks
`rt->resources` and declares one `OpVariable` per distinct constant/vertex-buffer binding, all sharing
one Block type (`OpTypeRuntimeArray` of u32 inside `OpTypeStruct`). A resource carrying
`table_index_count != 0` (stage 2's field) is the trigger: declare `OpTypeArray %Block %N` (or
`OpTypeRuntimeArray %Block` when the count is 0), a pointer to *that*, and the variable of it.

**5. Every buffer load funnels through ONE accessor**, which is what makes this tractable: eight
`cbuf_var` references in the file, and the read path is a single `binding -> variable` lookup that
falls back to `v_cbuf`. The access chains themselves are built at the call sites as
`{ptr, result, var, uconst(0), elem}`; an indexed binding needs one extra **leading** index —
`{ptr, result, var, desc_index, uconst(0), elem}` — with the index and the resulting pointer
decorated `NonUniform`.

Capability set required, emitted only when such a binding exists so that every module that does not
use one stays byte-identical: `ShaderNonUniform` (5301) and
`StorageBufferArrayNonUniformIndexing` (5308). Successful contracts are bounded fixed arrays, so they
do not request `RuntimeDescriptorArray` or `descriptorBindingPartiallyBound`.

**Why `NonUniform` is unconditional rather than reasoned about per site**, measured on `PPSA04263`:
of 51 compute launches, 29 are `local=64x1x1`, 7 are `8x8x1`, 7 are `32x1x1`, and **5 are
`local=256x1x1` — four waves per workgroup, therefore four EXECs and four distinct scalar indices**.
So the index is wave-uniform but *not* dynamically uniform across the invocation group, and for those
five programs the decoration is load-bearing. 43 of 51 launches being one wave per group is exactly
why the uniform reading looks safe and is wrong; the decoration is per-access, so emit it always.

**Sequencing note.** This work sits on top of #2463, whose reflection contract is unreviewed. Stage 1
changed shape under review (a two-leg fix became three), so the emitter is deliberately not built on
an unreviewed contract.

## Stage 5 groundwork — the five backend sites an indexed descriptor array touches (#2412)

Surveyed 2026-08-10 against `08d42aea`, independently of stage 4b: this half needs only stage 2's
representation (`table_index_count`, merged as 48d978ab) and stage 1's enabled features (merged as
056edd10), not the SPIR-V emitter. Every line number below was read, not recalled.

**1. The descriptor POOL is sized one entry per resource, and this is the site that fails furthest from
its cause.** `tests/render_runner.h` counts demand with `++storage_buffers` / `++storage_images` /
`++sampled_images` — once per `FrameResource`, whatever its arity — and passes those counts as the
`VkDescriptorPoolSize` values. An array of N descriptors consumes **N** pool entries while 1 was
reserved, so allocation fails with `VK_ERROR_OUT_OF_POOL_MEMORY` at a `vkAllocateDescriptorSets` that
has nothing visibly to do with arrays. The counting loop must add `max(1, table_index_count)` per
resource. Recorded first because it is the one a reader will not predict from "declare an array".

**2. The layout binding hardcodes `descriptorCount = 1`.** One site, in the layout-building loop, and it
becomes the array length.

**3. Three write sites hardcode `descriptorCount = 1`** and each supplies a single
`VkDescriptorBufferInfo`. An indexed binding needs N infos, one per table entry, materialised from the
guest table at `base + i * table_entry_stride`. All three are in the same file; missing one yields a
binding whose layout declares N and whose write supplies 1, which is a validation error rather than a
silent wrong result — the good failure direction.

**4. The layout cache key ALREADY includes `descriptorCount`**, alongside binding, type and stage flags.
So arity participates in layout identity and an array layout cannot be wrongly reused for a scalar
binding of the same number. This is a pre-existing correct property, recorded so nobody "fixes" it and
so a reviewer does not have to suspect stale-layout reuse.

**5. `descriptorBindingPartiallyBound` is not required.** The implemented fixed-array buffer path does
not use that feature to hide incomplete input. Every declared slot must have a validated concrete V#;
normalized null V#s are represented explicitly by a small zero source, while unreadable non-null
entries reject the dispatch. This preserves robust zero-read semantics without making a missing table
entry indistinguishable from a deliberate guest null descriptor.

**Materialisation must be lazy or generation-invalidated, and that is measured rather than assumed.**
Stage 0 (`PROSPER_DESCR_COHERENCE`, #2456) found **8.5% of distinct V#-relative descriptor addresses
rewritten during a single routed `PPSA04263` run** — 720,000 observations, 125,563 distinct addresses,
10,630 changed. Eager whole-table pre-materialisation is therefore dead: the table would be stale before
it was used. The same measurement collapses two design holes into one — the per-resource copy is ~67% of
a collapsed frame, so a cache is mandatory rather than optional, **and it needs the same generation
signal invalidation needs**. One mechanism serves both.

A second figure from the same probe bounds the working set: **1,170 distinct descriptor addresses in a
55 s pre-gameplay run against 125,563 in 210 s of gameplay** — a ~100x explosion at the transition, in
the same place the 53 skipped dispatches live. Whatever the cache is, it must survive that step change.

**6. The crux of the implementation is the `VkDescriptorBufferInfo` array's SIZING, and it carries a
pointer-stability constraint that is easy to miss.** Today the backend allocates
`std::vector<VkDescriptorBufferInfo> dbi(R.size())` — exactly one info per resource, sized once — and
each write takes `wr[i].pBufferInfo = &dbi[i]`. Those addresses are handed to Vulkan and must stay valid
until `vkUpdateDescriptorSets`, which they do *only because the vector is never grown after it is sized*.

An indexed binding needs a **contiguous run of N infos**, so the vector must become
`sum over resources of max(1, table_index_count)` entries with a per-resource offset, and
`wr[i].pBufferInfo = &dbi[offset_i]` with `descriptorCount = count_i`. The trap: any implementation that
appends per entry as it materialises (`push_back` in the loop) will **reallocate and silently invalidate
every pointer already stored in `wr`**, producing use-after-free that Vulkan reads as garbage descriptors
rather than as a crash. Size the run table before the loop, or store offsets and resolve the pointers in
a second pass after the vector is final. Same constraint applies to `dii` for image arrays.

**Sequencing consequence.** This makes the backend half a change to the *shared* renderer's descriptor
sizing and indexing semantics, which the charter lists as requiring independent review. Landing
pool-sizing or layout arity *without* the matching write arity is worse than landing neither: a layout
declaring N against a write supplying 1 is a validation error the moment a producer appears, so these
four sites (pool, layout, write, and this run table) must change together.

## Runtime-selected V# arrays — implemented foundation (2026-08-13)

The representation and backend stages above are now one fail-closed path:

1. `ShaderResource` carries the declared arity, guest record stride, selector provenance and a complete
   concrete V# payload. Capture/replay stores one backing reference per entry, and dependency closure
   enumerates the same entries independently.
2. The shared resource contract requires exact arity, agreement between raw and normalized V# fields,
   uniform stride/format/component semantics, representable controls and identity `DST_SEL` on non-null
   entries. Incomplete, unreadable or differently interpreted entries reject the dispatch.
3. The compute user-SGPR selector emitter declares the correct descriptor-indexing capabilities, marks both the
   bounded selector and access-chain pointer `NonUniform`, clamps the host access to descriptor zero,
   and selects architectural zero for an out-of-range guest index. Array stores and atomics remain a
   compile-time rejection. Graphics rejects this selector mode until its shell has an equivalent
   runtime user-data source. This checkpoint admits raw reads only; typed accesses need selected-entry
   default-fill/swizzle semantics, and writes/atomics need writeback authority.
4. Live compute derives pool demand, layout arity, write arity and per-binding offsets from one plan. It
   binds every concrete entry, supplies an explicit zero backing for guest null V#s, and retains each
   entry's own cache identity, lifetime and guest-authority bookkeeping. Arrays are read-only until an
   equivalent writeback contract exists.

The production test executes every concrete/null slot and an index equal to the declared arity through
the Vulkan backend. Same-entry mutations cover stride, format, unrepresented controls and `DST_SEL`.

This checkpoint does **not** claim GTA V's GPU-derived scalar-buffer selector. That producer still needs
a whole-CFG MUST-proof that the multiplying definition reaches the descriptor load and that every
runtime value denotes a snapshotted record start; treating a wrapped off-record byte offset as zero would
not match RDNA2. The capture/backend foundation stays useful while that proof is completed.
