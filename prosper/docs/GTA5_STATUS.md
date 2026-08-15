# Grand Theft Auto V (`PPSA04263`, RAGE) — status

**Rung 3** on the bring-up ladder: routed gameplay entry with real GPU draws. The HUD, radar and
tutorial text render; the 3D world does not.

Tracker: **#1873**. Active frontier: **#2481**. Route: `scripts/gta5/reach-story-mode.pad`
(read its header — the flip timing is measured, not estimated, and the tab navigation needs four R1
presses for a reason).

Historical design note for the descriptor work: `docs/FLAT_LOAD_DESIGN.md`. Do not start from it; the
descriptor-array lift it describes is complete.

## THE CORRUPTING PROGRAM IS `0x413dc3400` (2026-08-15)

Measured with `PROSPER_COMPUTE_TREE_WATCH=0x20f848417c:2063` on a 200 s routed run with the hanging
consumer skipped (`PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700`, so zero device losses, 9,291 frames).
The watch reads the table before and after **every** realized dispatch, so a change is attributed to
the dispatch that made it rather than to an interval.

**321 observations. Every clean -> cyclic transition:**

| program | clean -> cyclic | cyclic -> clean | clean -> clean |
| --- | --- | --- | --- |
| `0x413dc3400` | **37** | 0 | 3 |
| `0x413d88400` | 1 | 37 | 2 |
| `0x413e1c300` | 1 | 1 | 158 |
| `0x413cee500` | 0 | 1 | 79 |

`0x413dc3400` made the table cyclic on **37 of its 40 dispatches**. Nothing else does it
systematically.

**Every observed writer reported `toucher=1`.** No change came from a program whose resource table
does not contain the address, so there is no unknown writer and no out-of-bounds path to chase —
the watch was built to be able to report that case and did not.

### The per-submit chain, identical every submit

```
d15,d16  0x413cee500   Morton keys        -> pairs=0     unpaired=0    oob-roots=0   depth=3
d37      0x413dc3400   topology build     -> cycles=19   pairs=919     unpaired=148  depth=68
d39..47  0x413dc6700   the consumer       (skipped in this run; this is what hangs)
d54      0x413d88400   repurposes the RAM -> pairs=0     unpaired=571
```

`0x413dc3400` writes 2,061 of 2,063 records in one dispatch through six store pcs (bindings/pcs
`23:597, 25:608, 26:620, 27:632, 28:644, 29:656`), turning a table with **no** cycles into one with
19 cycles, 57 cyclic roots and depth 68. The consumer runs two dispatches later and walks exactly
that array. **This is the defect: the builder produces a cyclic parent array, and the hang is the
downstream symptom.**

The same run also shows `0x413d88400` at d54 leaving `pairs=0` — the allocation is **reused scratch**
across phases, not one long-lived structure, which is why a pair count compared across phases moves
by hundreds. Compare pair counts only within the same phase.

### What a correct table looks like — measured, not assumed

Against 91 captured 2,063-dword tables:

| | pairs | unpaired | cycles | max depth |
| --- | --- | --- | --- | --- |
| clean parent tables (`live-a240`, `post-36-3`) | **1030** | **0** | 0 | **11** |
| cyclic captures (85, one resource hash) | 1029 | 1 | 1 | 15 |
| `0x413dc3400` output, live | 919 | 148 | 19 | 68 |

This **confirms Codex's LBVH identification quantitatively**: a full binary tree over 1,032 leaves
has 1,031 internal nodes, and a well-formed table measures 1,030 sibling pairs with zero unpaired
records at depth 11, against log2(1032) = 10.01. Pairs are `(odd, even)`: the odd index is the head
(side bit clear), the even index its mate (side bit set).

In the 85-capture set the anomaly is *fully deterministic* — the same 4-member cycle
`(256, 384, 447, 831)` and the same single unpaired record `300` in every one. All 85 share one
resource hash, so this is one table observed 85 times, i.e. the corruption is **stable** across
submits 7629 -> 9694 rather than re-derived; nothing repairs it. The four cycle members are each in
a *well-formed* sibling pair, so the cycle is not caused by a broken pair — the parent links
themselves are wrong. Slot 300 differs from its mate only in bits[2:0] (`0x942` against `0x940`,
both decoding to parent 296), so it is a flag-bit difference and a separate anomaly from the cycle.

**Ruled out by this measurement:** the "lost update" / two-writer race account of the cycle. Within
one dispatch's pre/post pair there is exactly one writer, and its output is cyclic.

## Falsified for `0x413dc3400` (2026-08-15) — checked, not assumed

- **Barrier uniformity.** All eight `OpControlBarrier`s in the emitted module are in uniform blocks.
  Three sit in each dispatcher's **continue target** (`OpLoopMerge %163 %162` — %162 is the continue
  block, reached by every invocation on every iteration) and five at structured merge targets. The
  design comment in `rdna2_to_spirv.cpp` states this invariant explicitly ("the switch merge is
  reached by every invocation on every iteration"); it holds in the artefact. A barrier inside a
  `switch(pc)` *case* would have been a real Vulkan uniformity violation — it is not what is emitted.
- **Native subgroup / multiwave lowering.** `[subgroup] cs=0x413dc3400 … native=0 … multiwave=0`:
  the program is already lowered through the portable wave model, so there is no native lowering to
  blame. `PROSPER_NO_NATIVE_COMPUTE_MULTIWAVE=1` leaves all nine of its module hashes byte-identical.
- **LDS undersizing.** The module declares 384 dwords = 1,536 bytes = 3 × 512-byte
  `COMPUTE_PGM_RSRC2.LDS_SIZE` granules; Codex's ISA read puts the largest accessed LDS address at
  byte 1,028. Sized correctly and over-provisioned either way.
- **A race.** Two different frames produce broken-pair patterns sharing a 60-character suffix
  exactly, with the same first damaged index. Deterministic given the input.
- **A lane/wave/workgroup boundary effect.** Damage index mod 2/3/4/8/16 is flat.
- **An unknown or out-of-bounds writer.** Every observed change reported `toucher=1`.

## The selected-sbuffer contract declines because the SELECTED V# IS FLOAT DATA (2026-08-15)

`PROSPER_GTA5_SBUFFER_REJECT=1` (added on this branch; the six reasons in
`rdna2_gta5_compute_contracts.cpp` were all behind `PROSPER_DBG`, which desyncs the route) reports
exactly two declines for `0x413ce6000` on a whole run:

```
[gta-selected-sbuffer] reject=consumer-resource
[gta-selected-sbuffer] reject=selected-vsharp
    words=c540fa56:c51e1625:4373fd8a:45de36cc
    base=0x1625c540fa56 stride=1310 records=1131675018 size=4294967295
```

**Those four words are floats**: `0xc540fa56` ≈ **-3087.6**, `0xc51e1625` ≈ **-2529.5**,
`0x4373fd8a` ≈ **244.0**, `0x45de36cc` ≈ **7110.9**. That is a world-space AABB, not a descriptor —
and the derived `base=0x1625c540fa56`, `records=1131675018`, `size=0xffffffff` are what you get from
reinterpreting it. **prosper is correctly refusing to manufacture a descriptor out of it**; the
contract's `selected-vsharp` guard is doing its job.

So the selector resolves to a record that does not contain a V# at the expected offset. The chain to
audit, with what is measured about each link:

| link | measured |
| --- | --- |
| the selector's source records, pc70 | `base=0x20f848e2bc stride=8 records=2064 size=16512` — resolves |
| the descriptor array, pc153 | `base=0x203f249b38 stride=120 records=5 size=600` — resolves |
| `s_buffer_load_dwordx4 s[8:11], s[4:7], s106` | SMEM immediate offset **8**, so the V# is expected at `selector*120 + 8` |
| the selected element | **float AABB data** |

Three candidates, none yet excluded: the selector value is wrong; the V# lives at a different offset
within the 120-byte record than `+8`; or the **source records at pc70 are themselves stale because
their producer also does not run** — which would make this a chain of missing producers rather than
one. That last one is the possibility to test first, because it is the same failure this whole
investigation has already found once.

Note the resource map lifts pc156/pc158 as `base=0x203f2e9b38 size=13360 stride=20 entries=5` —
**stride 20, against the contract's expected 120**. Reconciling those two views is likely the fastest
route in.

## The selector chain at `0x413ce6000` pc149..156 — the exact fix site

Decoded with prosper's own opcode constants (`rdna2_decode.hpp`), not guessed:

```
pc149  s_mulk_i32            s106, 120          ; SOPK 0x10. selector * 120, in VCC_LO as scratch
pc150  s_load_dwordx4        s[4:7], s[0:1], m0 ; the descriptor-ARRAY V#
pc153  s_buffer_load_dwordx4 s[8:11], s[4:7], s106  ; SELECT one V# at selector*120
pc156  buffer_load_dwordx3   v[0:2], v6, s[8:11], 0 ; use it   <-- mode=unresolved-operand
```

**120 is exactly the array's stride.** `PROSPER_DYNTRACE_FAIL` confirms the split:

```
BUF(v4) key=0xa8       use_pc=153  base=0x203f249b38 stride=120 records=5 size=600   RESOLVED
BUF(v4) key=0xffffffff use_pc=156  v4=00000000:00000000:00000000:00000000            UNRESOLVED
```

So the descriptor-array lift **finds the array** at pc153 and **cannot resolve the selected element**
at pc156: `key=0xffffffff` is the sentinel the executor's own comment names as matchable by none of
the three routes (fetch pc, SRT offset, SGPR base).

**Why the selector does not resolve. `CONFIDENCE: MED`.** The selector arrives through **VCC_LO used
as an ordinary scalar register** — GTA V's compiler recycles it, which this file already documents
elsewhere. prosper's VCC-as-scalar recognition is
`is_wave64_vcc_lo_scalar_b32_candidate`, and it covers exactly two shapes: `s_cselect_b32` with
inline operands, and SOP2 B32 logicals. **`s_mulk_i32` is SOPK and is in neither set**, so the write
at pc149 is not recognised as a scalar-scratch definition. That matches the failure exactly, but the
alternative — that the const-fold breaks somewhere else along `s106`'s chain — has not been
separately excluded, so this is a lead and not a conclusion. Verify before building on it.

This is the same underlying difficulty as the execz VCC-half liveness guard cleared earlier today:
**every remaining obstacle in this program comes from the guest recycling VCC as a general scalar
register, and prosper modelling VCC specially in each place independently.**

## THE REMAINING BLOCKER, EXACTLY (2026-08-15)

Every compute reject on a routed run now names its cause without `PROSPER_DBG`. `0x413ce6000` — the
producer whose absence removes the world — is blocked by **one instruction**:

```
0x413ce6000  mode=unresolved-operand pc=156 words=e0382000,80020006 fmt=12 op=0xe
```

`fmt=12` is MUBUF, `op=0xe` is `buffer_load_dwordx3`. `mode=unresolved-operand` means the **lowering
exists and the descriptor does not resolve** — the emitter is fine, the descriptor is the defect.

pc156 is the **runtime-selected buffer array**: `PROSPER_COMPUTE_RESOURCE_MAP` shows it resolving in
the live table as `binding=10 fetch-pc=156 base=0x203f2e9b38 size=13360 stride=20 entries=5`, while
the pre-specialization const-fold trace (`PROSPER_DYNTRACE_FAIL`) shows it as
`use_pc=156 v4=00000000:00000000:00000000:00000000 base=0x0 stride=0 records=0`. Its sibling at
pc158 is the same shape. Those two are the only unresolved uses of the nineteen.

**So the whole "GTA V has no 3D world" chain reduces to one buffer-array descriptor that does not
const-fold at pc156 of `0x413ce6000`.** That is the next thing to implement.

The full reject census, now legible — every one is `mode=unresolved-operand`, i.e. every one is a
descriptor that does not resolve rather than an instruction that is not implemented:

| program | pc | fmt/op |
| --- | --- | --- |
| `0x413ce6000` | 156 | MUBUF `buffer_load_dwordx3` |
| `0x413cf9200` | 5 | MUBUF `buffer_load_dword` |
| `0x413cf9a00` | 11 | MUBUF `buffer_load_dword` |
| `0x413cf9d00` | 70 | FLAT/GLOBAL `op=0xc` |
| `0x413d14100` | 6 | MUBUF `buffer_load_dwordx3` |
| `0x2042f49a00` | 16 | MIMG `op=0x1` |
| `0x2042f4a600` | 7 | SMEM `op=0x4` |
| `0x205b545c00` | 98 | VOP2 `op=0xf` |
| `0x205b54ee00` | 90 | VOP2 `op=0xf` |
| `0x205b5e8600` | 314 | SOP2 `op=0xe` |
| `0x205b654a00` | 1180 | MIMG `op=0xe6` |
| `0x205b657200` | 313 | MIMG `op=0xe6` |
| `0x205b658800` | 82 | SOP1 `op=0x3` |

## REFRAME: this is a RAY-TRACING BVH, and prosper already knows the format (2026-08-15)

The "tag" this investigation has been tracking is the **AMD RDNA2 ray-tracing BVH `NODE_TYPE`**, and
prosper's own recompiler says so. `rdna2_to_spirv.cpp` (search `is_box16`) software-emulates
`IMAGE_BVH_INTERSECT_RAY`, loading 28 dwords of a node and branching on:

```cpp
const uint32_t is_tri0  = b.ucmp(Op_IEqual, node_type, b.uconst(0u));
const uint32_t is_tri1  = b.ucmp(Op_IEqual, node_type, b.uconst(1u));
const uint32_t is_box16 = b.ucmp(Op_IEqual, node_type, b.uconst(4u));   // 64-byte node
const uint32_t is_box32 = b.ucmp(Op_IEqual, node_type, b.uconst(5u));   // 128-byte node
```

So bits[2:0] of a node reference are the node type: 0–3 triangle, **4 box16**, **5 box32**, 6
instance, 7 procedural. Everything this investigation measured now has a name:

| observation | reading |
| --- | --- |
| `0x209cc76000` at 64-byte stride | an array of **64-byte nodes** (box16 or triangle) |
| tags 0 / 2 / 5 / 7 dominating | triangle / triangle2 / box32 / procedural |
| **tag 4 appearing only in broken submits** | **box16 nodes entering the scene** |
| the sibling-paired parent table | BVH topology |
| Morton keys and `0x09249249` | an LBVH build, as Codex identified |
| `0x413dc3400`'s `tag == 2 \|\| tag == 5` predicate | it writes links only for two specific node types |

**`PROSPER_DECODED_BVH` machinery already exists** — `DecodedBvhDescriptor` in
`agc_shader_layout.hpp` carries `box_grow`, `triangle_return_mode`, `box_node_64b`, `sort_enabled`,
and the dynfail dump prints `BVH(bvh4)` descriptors. This is a supported surface, not an unknown one.

### The consumers of the BVH do not compile

`IMAGE_BVH_INTERSECT_RAY` is **MIMG opcode 0xe6** (`rdna2_to_spirv.cpp:14145`). Two programs in the
reject census fail on exactly that instruction:

```
0x205b654a00  mode=unresolved-operand pc=1180 fmt=14 op=0xe6   image_bvh_intersect_ray
0x205b657200  mode=unresolved-operand pc=313  fmt=14 op=0xe6   image_bvh_intersect_ray
```

`mode=unresolved-operand` means **the lowering exists and the BVH descriptor does not resolve**.
prosper has the full software traversal emulation; the shaders that would use it are declined for a
descriptor.

**So there are two independent defects on the ray-tracing path, and the second was invisible until
the reject reasons became readable:**

1. `0x413dc3400` builds a cyclic topology once the scene passes a point — the input-dependent defect
   this document tracks above.
2. **The traversal shaders that consume the BVH never compile at all**, because their BVH descriptor
   does not resolve.

Fixing (1) alone cannot render the world if (2) also holds. **(2) is the better first target**: it is
a descriptor-resolution problem on an instruction prosper already implements, it is named exactly, and
unlike (1) it does not depend on scene state.

## The BVH traversal shader's descriptor is never classified as a BVH (2026-08-15)

`PROSPER_DYNTRACE_FAIL_ADDR=205b654a00`. The shader is a **full-screen ray-tracing pass**:

```
launch groups=240x135x1 threads=1920x1080x1 local=8x8x1 user_sgprs=8
pre-specialization raw const-fold recovered 72 descriptor use(s)
```

**Not one of the 72 is a `BVH(bvh4)`.** The dynfail dump distinguishes three kinds — `TEX/IMG(t8)`,
`BUF(v4)` and `BVH(bvh4)` — and this shader, which executes `image_bvh_intersect_ray` at pc1180,
recovers zero BVH descriptors. Two of its uses are unresolved with a **valid base and zero extent**:

```
BUF(v4) key=0xffffffff use_pc=1032 v4=a1f76200:00000020:00000000:00000000
        base=0x20a1f76200 stride=0 records=0 size=0 required=28
BUF(v4) key=0xffffffff use_pc=1266 v4=a1f76400:00000020:00000000:00000000
        base=0x20a1f76400 stride=0 records=0 size=0 required=124
```

`required=28` is notable: prosper's own BVH emulation loads exactly **28 dwords** per node
(`for (uint32_t k = 0; k < 28; ++k) w[k] = load_node(k);`).

**Reading, `CONFIDENCE: MED`.** These are BVH descriptors being classified and decoded as buffer V#s.
A GFX10 BVH descriptor has its own 4-dword layout — `decode_bvh_descriptor` in
`agc_shader_layout.hpp` reads `type`, `box_grow`, `triangle_return_mode`, `box_node_64b`,
`sort_enabled` from it — and interpreting one as a buffer V# yields `num_records = 0`, hence
`size = 0`, hence an unbounded use, hence `unresolved-operand`. That fits every observation, but the
alternative (the guest genuinely supplies a zero-extent descriptor at this point in the route) is not
excluded, and a `key=0xffffffff` means neither fetch pc, SRT offset nor SGPR base matched — which has
its own possible causes.

**FALSIFIED, by running that check.** Decoding the two words with `decode_bvh_descriptor` gives
`base = 0x20a1f7620000` — the BVH layout shifts its base left by 8, and the result lands far outside
the guest address space, which sits around `0x20xxxxxxxx`. The **buffer** decode gives
`base = 0x20a1f76200`, a perfectly plausible guest address. So these are not misclassified BVH
descriptors, and the reading above is dead. Cost: one four-dword computation, no run.

**What the same words do show.** Dwords 2 and 3 are **entirely zero**, while dwords 0 and 1 carry a
sane base. A real buffer V# has a nonzero dword3 (it carries format and type bits), so an all-zero
upper half is the signature of a **partially recovered descriptor** — the const-fold obtained the
low two dwords and not the high two — rather than of a descriptor the guest genuinely wrote as
zero-extent. `num_records = 0` then follows from dword2 being absent, and `size = 0` from that, and
`unresolved-operand` from that. **That is the next thing to test**, and it is a different defect from
anything this document has chased: not a wrong descriptor, a half-read one.

## THE UNIFYING CAUSE: GTA V recycles VCC_LO as a general scalar register (2026-08-15)

The BVH descriptor at the rejecting instruction is **built in the shader**, and it is built through
VCC_LO. `0x205b654a00` pc1180 is `image_bvh_intersect_ray` with its descriptor in `s[16:19]`:

```
pc1171  s19  = <computed>
pc1174  s106 = s19 & 0x000003ff            ; VCC_LO as scalar scratch
pc1176  s17  = <computed>
pc1177  s19  = s106 | 0x81000000           ; and back out of VCC_LO
pc1179  s_waitcnt
pc1180  image_bvh_intersect_ray  v[0..], v6, s[16:19], s[0..]
```

`(x & 0x3ff) | 0x81000000` is the BVH descriptor's dword3 — its size-high bits and type field. **The
descriptor cannot resolve unless the const-fold tracks a value through VCC_LO.**

That is the same obstacle as everywhere else in this title:

| site | what VCC_LO carries | consequence |
| --- | --- | --- |
| `0x205b654a00` pc1174/1177 | the BVH descriptor's dword3 | `image_bvh_intersect_ray` rejects, the ray-tracing pass never compiles |
| `0x413ce6000` pc149 | `s_mulk_i32 s106, 120`, the descriptor-array selector | `buffer_load_dwordx3` at pc156 rejects |
| `0x413ce6000` pc84/90 | integer scratch inside an execz arm | the structurizer's VCC-half liveness guard rejects (cleared on this branch) |
| GTA V generally | `is_gtav_wave64_vcc_lo_scalar_cselect` exists precisely for this | already a known pattern in the code |

**prosper models VCC specially in each place independently — the liveness proof, the scalar-scratch
recogniser, the descriptor const-fold — and each place has its own, narrower notion of which VCC
writes count as data.** `is_wave64_vcc_lo_scalar_b32_candidate` admits exactly `s_cselect_b32` with
inline operands and SOP2 B32 logicals. `s_mulk_i32` (SOPK) is not in it. Neither is the
`s_and_b32`/`s_or_b32` pair above being tracked *through* to a descriptor.

**This is the frontier.** Not the tree builder, whose lowering is proven correct by eleven perfect
submits; and not a missing producer, which is falsified. A single coherent treatment of "VCC_LO used
as an ordinary scalar register" — one recogniser consulted by the liveness proof, the scalar model
and the const-fold alike — is what the remaining rejects have in common.

`CONFIDENCE: MED` on that being sufficient. It is established that the descriptor passes through
VCC_LO and that prosper's VCC recognisers do not cover these shapes; it is *not* established that
covering them is enough to make either program compile, because neither has been tried.

## `s_mulk_i32` folding: landed, and it did NOT move the reject (2026-08-15)

The const-fold's SOPK case handles **only** `s_movk_i32`; every other SOPK forgets its destination
*and* invalidates SCC. Its own comment notes that `s_movk/s_version/s_cmovk/s_mulk` do not write SCC
— so `s_mulk_i32` was being charged both costs it does not owe, and the comment demands per-opcode
evidence before widening. That evidence exists: `0x413ce6000` pc149 is `s_mulk_i32 s106, 120` where
120 is the descriptor array's exact record stride, feeding the select at pc153 and the rejecting load
at pc156.

Folded it: multiply a known destination, forget an unknown one as before, and stop clobbering SCC.
245/245 ctest green.

**It did not change the reject.** `0x413ce6000` still fails with `mode=unresolved-operand pc=156`.

**Then the probe was run, and it explains why — the fold was aimed at the wrong thing.**

`PROSPER_DYNTRACE_SGPR=106` gives s106's complete fold history in this program:

```
pc=3    KNOWN 0x00000000
pc=56   FORGOTTEN  words=beea376a
pc=84   KNOWN 0x00000000 / 0xfffff7f0
pc=90   KNOWN 0x7f7fffff
pc=116  FORGOTTEN  words=beea3704      <- the break
pc=131  FORGOTTEN  words=beea3704
pc=149  FORGOTTEN  words=b86a0078      <- s_mulk_i32 s106, 120, with s106 ALREADY unknown
```

pc116 and pc131 are `SOP1 op=0x37 s[106:107], s[4:5]`, which prosper classifies as a **B64
data/mask write** and which the RDNA2 encoding makes **`s_andn1_saveexec_b64`**: `exec = ~s4 & exec`,
and **`s106` receives the OLD EXEC MASK**. Both are immediately followed by `s_cbranch_execz`.

**So the descriptor-array selector is derived from a saved EXEC mask.** `s_mulk_i32 s106, 120` is
multiplying a lane mask by the array's record stride. That is a wave-dependent runtime value, not a
constant, and **no const-fold can ever resolve it** — which is precisely why this program has a
dedicated `selected_sbuffer` contract that certifies the selector's complete *domain* from live
source records instead of folding it.

**Conclusion: the `s_mulk_i32` fold does not address this reject and cannot.** The reject is the
contract's `selected-vsharp` decline — record 4 of the outer array holding float AABB data — and the
const-fold was never on that path. The fold change stays for its own reasons; this reject needs the
contract.

The change is kept on its own merits — it is a documented over-conservatism corrected with ISA
backing and it costs nothing — not because it was shown to help.

## The F9 frame grab cannot be used on this title — diagnosed and filed (#2549)

The charter names the F9 grab the fastest loop for a graphical bug, and GTA V — a GPU-driven title
whose world is absent — is exactly the case it exists for. It could not be run at all, and the
message said only "the capture window contained no GPU submits".

Instrumenting the submit hook with three counters and the window's wall-clock duration named the
cause immediately:

```
[grab] submit hook reached=26840, 19206 while inactive, 7634 while not capturing;
       window was open 26 ms for 48 presents
```

**48 presents in 26 ms**, and 12 presents in 7 ms — about **1,700 presents/second** against a 23/s
average. **GTA V flips in bursts**, so a window defined as a present COUNT has an effectively random
wall-clock duration, usually far too short to contain a submit.

Two further refusals were found and opened behind `PROSPER_CAPTURE_ALLOW_UNPROVEN_INDIRECT=1`, which
reports every acceptance as inspection-only: five packed/indirect-pointer provenance validators that
abort the whole bundle when one compute dispatch cannot be exactly proven. Right for a replay bundle,
wrong for a bundle meant to be read — and on this title they fire, so the grab aborted at submit
19358 before the window problem was even reachable.

With both addressed the capture reaches **181 frames / 22,599 submits**, and then hits the byte
budget: 28 frames is 3.4 GB against a 3,072 MB maximum. **There is no setting that both lands on
submits and fits the budget, because one knob controls both.** Filed as **#2549** with a
time-based-window suggestion.

**Consequence for everything above:** there is still no draw-level view of a GTA V frame. Every
conclusion in this document about why the world is absent rests on log statistics, not on the frame's
contents.

## Open, unquantified: 38 of 40 logged `WaitRegMem` waits are on UNMAPPED labels (2026-08-15)

```
[agc] WaitRegMem #3 q=A NOT satisfied at fold time: [0xf58]&0x190 = 0x0, func=5 ref=0xffffffff
      — dependency violated | built@0ms(age=-1ms) pre@build=0x0 LABEL-UNMAPPED
```

Of the 40 logged, **38 carry `LABEL-UNMAPPED`** and 38 are on queue A. The awaited address in that
sample is **`0xf58`** — four-byte aligned, so it passes the call site's gate, and far too small to be
a guest address, which in this title are `0x20xxxxxxxx`.

**That is the same shape as the indirect-dispatch argument truncation** on the previous section
(`0xf8480120` for `0x20f8480120`), which has an in-tree aperture recovery. Whether the same recovery
applies to `WAIT_REG_MEM` labels is untested.

Two things must be checked before treating this as a defect, and neither has been:

- **It may be normal.** `command_processor.cpp` states plainly that an unsatisfied wait is "NORMAL,
  handled state" and that under content-load bursts it "fires thousands of times a minute". The
  barrier model behind `PROSPER_WAIT_DEFER=1` exists precisely because the default folds past them.
- **The count is a LOG CAP, not a measurement.** The diagnostic prints the first 40 and then every
  1024th, so "40" says nothing about the true rate. Quoting it as a frequency would be the rate-limit
  trap the orchestration doc warns about.
- **`wm_addr` may be register space.** PM4 `WAIT_REG_MEM` selects register or memory addressing, and
  this code path reads `wm_addr` as memory unconditionally (`guest_readable(c.wm_addr, 8)`). A
  register-space wait would then always read as unmapped. This area carries substantial prior work
  (#312, #380, #448), so the absence of a `mem_space` check may be deliberate rather than missing —
  it has not been established either way here.

Recorded as an observation with its caveats rather than a lead, so the next reader neither chases it
blind nor loses it.

## What is NOT hiding the world — four eliminations, each with a verified lever (2026-08-15)

Every one of these was measured on a routed run with the lever confirmed to have moved, so each is a
**genuine negative rather than a void arm**. None of them restores the world.

| candidate | lever, verified | outcome |
| --- | --- | --- |
| the compute hang / device loss | consumer skipped → **0 device losses**, 9,363 frames | world still absent |
| truncated indirect-dispatch arguments | aperture recovery fires 24×, unreadable **24 → 0** | frame unchanged |
| storage-image contract dropping a whole batch | per-draw drop reports **"kept 0 of 1 draws"** | the batch *is* one draw; identical either way |
| `CB_COLOR_CONTROL.MODE=0` on 131,072 draws | — | known latching artefact (#1706), not per-draw truth |

### The storage-image rejection, precisely

The failing resource is named exactly, and **the DCC hypothesis for it was wrong**:

```
[render] storage-image contract: set=1 binding=46 portable-uvec4 REJECTED
    guest-texel=4 shape=0
    writable=1  compressed=0  arrayed=0  multisampled=0
    reflected-dim=1 guest-dim=1 depth=1 fmt=4 comps=2
```

The only failing term is **`writable=1`** — `portable_storage_shape` requires
`!writable_storage_image`. Compression, arraying, multisampling and the dimensions are all fine. So
**writable portable-uvec4 storage images are unsupported in the graphics path**, and that is a real
gap worth closing on the charter's own terms.

But its blast radius is one draw, not a frame: `PROSPER_RENDER_DROP_UNPROVEN_DRAW=1` reports
`kept 0 of 1 draws in this batch` every time. The conservative whole-batch abort in `render_runner.h`
looked alarming and costs nothing extra here.

### Still open, and unquantified

Three 4K DCC-compressed **sampled** images remain unsupported (fmt 1/4/9). That is a separate
resource from the storage image above — the storage image is not compressed. Whether the composite
depends on those three has not been established.

### The instrument that would answer this, and why it has not yet

`PROSPER_GRAB_BUNDLE_AFTER_MS` on `prosper-app` is the documented fastest loop for "why does this
frame look wrong". It was tried at 170 s with `PROSPER_CAPTURE_FRAMES=1` and again with 16, and both
report **"the capture window contained no GPU submits"** while the same run shows 271 `[agc]`, 60
`[compute]` and 43 `[render]` lines and reaches 191 s of route. So the capture window and the
submits are not lining up, and **that mismatch is itself the next thing to understand** — without a
bundle there is no draw-level view of the frame, and every conclusion above is from log statistics
rather than from the frame's actual contents.

## The hang is NOT the only blocker — two more, measured (2026-08-15)

Answering "is the compute hang the reason there is no world" directly, by looking at a frame with
the hanging consumer skipped: **zero device losses, 9,363 frames, and the world is still absent.**
Tutorial text and a few light blooms render; no geometry. So the compute chain cannot be the whole
story, and two further blockers are visible in the same run.

### 1. Indirect dispatch arguments arrive with a TRUNCATED address — and the fix is already in-tree, off

```
[agc] indirect dispatch skipped: unreadable arguments at 0xf8480120
```

The real address is `0x20f8480120`; the high byte is gone. `command_processor.cpp` already has an
aperture-recovery path — `(aperture << 32) | (addr & 0xffffffff)` — behind
**`PROSPER_INDIRECT_APERTURE_RECOVERY`, which is opt-in and off by default**. Its own comment records
that with it off, 50 of 64 indirect compute dispatches are skipped as unreadable; with it on, 0 are,
and the probe found the raw low address unmapped and `aperture | low` mapped on 49 of 49.

**Measured here with the lever verified:** recovery fires 24 times
(`0xf8480120 -> 0x20f8480120`, queue 2), unreadable-argument skips go **24 → 0**, device losses stay
at 0. **And the frame is unchanged — still no world.** A genuine negative, not a void arm: the
lever demonstrably moved.

So the truncation is real and worth resolving on its own merits, but it is not what is hiding the
world either.

### 2. Three 4K DCC-compressed sampled images are unsupported

```
[render] DCC-compressed sampled image 3840x2160x1 fmt=1 tile=24 is unsupported; metadata=0/0
[render] DCC-compressed sampled image 3840x2160x1 fmt=4 tile=27 is unsupported; metadata=81920/81920
[render] DCC-compressed sampled image 3840x2160x1 fmt=9 tile=27 is unsupported; metadata=49152/49152
```

`live_renderer.cpp` handles DCC only for the **uniform fast-clear** case
(`gfx10_dcc_fast_clear_rgba8`). When that fails it warns and **falls through to the ordinary
format/detile path, which reads the COMPRESSED base bytes as if uncompressed** — garbage or black.

These are full-screen 4K surfaces, they are **not** renderer-owned RTTs (`!rtt_hit`), and there are
three of them in the formats a scene colour/normal/etc. set would use. **A composite that samples
them cannot produce a world image regardless of what the geometry passes do.**

`CONFIDENCE: MED-HIGH` that this is an independent blocker; `CONFIDENCE: LOW` on it being *the*
remaining one, since nothing yet shows the geometry passes fill those surfaces in the first place.

### Not a lead: `CB_COLOR_CONTROL.MODE=0`

The same run reports 131,072 draws with an unmodeled `CB_COLOR_CONTROL.MODE=0` "still executed as an
ordinary color draw", which looks alarming (MODE 0 is CB_DISABLE). **`render_state.hpp` already
records that prosper's decoded MODE is not per-draw-trustworthy (#1706): a utility sequence's
operation bits stay latched onto later ordinary draws.** So the count measures the latching, not
131,072 draws that should have been suppressed. Checked before chasing.

## Six-reference simulation: the builder is PROVABLY faithful (2026-08-15)

Codex's correction (#2542): `0x413dc3400` reads **six** candidate references per node, not two —
`pc86` loads the first pair, `pc161` follows `(A >> 3) - 4` for a second pair, `pc237` follows
`(B >> 3) - 4` for a third. **A histogram over record dwords 0/1 covers two of six and cannot predict
how many parent slots a dispatch writes**, so `pairs == 1030` is an empirical control for one route
state, not a universal oracle. What survives as a hard oracle is **acyclicity**.

`tools/re/bvh_ref_simulator.py` reproduces all six selections over each captured input and diffs the
expected destination set against the slots the dispatch actually changed:

| dispatch | expected | written | expected-not-written | **written-not-expected** | ref cycles |
| --- | --- | --- | --- | --- | --- |
| s5943 d37 | 2061 | 2061 | 0 | **0** | 0 |
| s7188 d37 | 2061 | 2061 | 0 | **0** | 0 |
| s8842 d37 | 2061 | 2061 | 0 | **0** | 0 |
| s16041 d37 | 2061 | 2061 | 0 | **0** | **117** |
| s17181 d37 | 2061 | 2061 | 0 | **0** | **117** |
| s19002 d37 | 2061 | 2061 | 0 | **0** | **117** |
| s5528 d37 | 1754 | 1457 | 297 | **0** | 0 |
| s9645 d37 | 1788 | 1283 | 505 | **0** | 0 |

**`written but not expected` is ZERO in every frame measured.** The builder never writes a slot its
input does not imply — now established over all six references rather than two.

And the decisive rows are s16041 onward: **`expected = written = 2061`, `missing = 0`, and
`cycles = 117`.** Everything the input implies is written, nothing else is, and the result is cyclic
**because the input is**. There is no discrepancy left for the builder to be responsible for.

(`expected-not-written` is nonzero in other frames because the simulator walks every node while the
dispatch processes only its compacted, depth-parity subset. The informative direction is the other
one, and it is zero everywhere.)

**This closes `0x413dc3400`'s role definitively.** Together with the eleven perfect submits, two
independent methods now say the same thing: the builder is correct and the defect is entirely
upstream, in the node records `0x413ce6000` writes.

## Corrections from Codex (#2542) to earlier sections

- **`0x413cf9000` / `0x413cf9200` are arena aliases, not producers of the 64-byte tag records.**
  `cf9000` is an initializer over an **80-byte** stride; `cf9200` operates **32-byte** records at a
  320-byte V# near `0x209cc7ab00`. Their 117 changes each are a paired initialise/fill of a small
  structure that merely overlaps the watched allocation. **So ranking writers by whole-watch change
  count is misleading, and the +54 cycles this document attributed to `cf9000` should be discounted**
  — its writes are not 64-byte records and decoding them as such is a category error. The genuine
  64-byte-view producers are `cf5400`, `cf6100`, `ce6000` and `d1bf00`. `0x413ce6000`'s **+590** is
  unaffected: it is a real 64-byte-view producer.
- **The `record 4 + 8` arithmetic is exact**, not over-fitted: `pc74` forms `selector = word >> 3`,
  `pc144` `v_readfirstlane_b32 vcc_lo, v7`, `pc149` `s_mulk_i32 vcc_lo, 120`, `pc153`
  `s_buffer_load_dwordx4 … offset:8`. So `4 * 120 + 8 = 488` exactly. **The contract's weakness is
  TEMPORAL, not arithmetic**: `selected_sbuffer_domain()` reads the source and the outer table
  through `complete_resource_bytes()` — the currently CPU-visible mapping — with no command-order
  snapshot coupling either to the bytes the GPU later consumes. Codex has retained evidence of the
  same outer base decoding as **five coherent V#s** at exactly `+8` in one observation and floats in
  another, so the buffer is a reused arena observed at different epochs. **The
  "it holds frustum planes / the contract is over-fitted" framing in this document is therefore
  wrong about the cause** — the layout is right and the epoch is not.
- **The selector is `readfirstlane(source_word >> 3)`**, from the pc70 source records — not from a
  saved EXEC mask, as this document earlier concluded from watching s106 alone. It is still not
  const-foldable (a readfirstlane of live lane data is runtime state), so the conclusion that no fold
  can resolve it stands; the stated *route* to it was wrong.
- **`0x413e1ff00 toucher=0`** — Codex reached the same diagnosis independently: a watch-attribution
  bug, not memory corruption. Already fixed here with `compute_address_window_hits`.

**A systematic caveat this raises over much of the descriptor work above:** every capture in this
document reads guest memory at fold or realization time on the CPU. That is not the execution epoch
the GPU consumes. Codex's recommended discriminator is an ordered GPU-side readback of the pc70
source, the root V# at `s[0:1] + 0xa8`, and all 600 outer bytes, captured **immediately before the
dispatch executes** — a CPU reread is not sufficient.

## ROOT CAUSE, ATTRIBUTED: `0x413ce6000` writes the cyclic child graph (2026-08-15)

Per-dispatch pre/post attribution on the node records at `0x209cc76000`, with the **child-graph cycle
count** as the metric:

```
submit  order   writer         cycles pre -> post
6765    4550    0x413ce6000       0 -> 324    ADDS 324
7180    24812   0x413ce6000     324 -> 398    ADDS 74
7595    26144   0x413ce6000     398 -> 388    removes 10
8835    26556   0x413ce6000     388 -> 402    ADDS 14
15271   16831   0x413ce6000     350 -> 436    ADDS 86
16088   14048   0x413ce6000     436 -> 468    ADDS 32
16088   14546   0x413cf9000     468 -> 469    ADDS 1
```

| program | dispatches that add cycles | total added |
| --- | --- | --- |
| **`0x413ce6000`** | 7 | **590** |
| `0x413cf9000` | 12 | 54 |

**`0x413ce6000` is the program that makes the child graph cyclic**, including the transition from a
completely acyclic graph to 324 cycles in one dispatch. It also *removes* cycles on other dispatches
(−10, −14, −20, −18), which is the signature of an incremental refit that is partly wrong rather than
a rebuild that is wholly wrong.

### The complete chain, every link measured

1. **`0x413ce6000`** writes the 2,063 × 64-byte node records at `0x209cc76000`, and its output
   contains a **cyclic child graph**.
2. **`0x413dc3400`** builds parent links from those records **faithfully** — its lowering is proven
   correct by eleven consecutive perfect submits, and its output tracks its input.
3. The parent table is therefore cyclic.
4. **`0x413dc6700`** walks it with a loop that has no iteration bound and no cycle detector.
5. GPU hang → RADV device loss → live compute disabled process-wide → every later indirect draw
   dropped → **no 3D world**.

**And `0x413ce6000` is exactly the program whose descriptor at pc156 does not resolve**
(`mode=unresolved-operand`, the `selected_sbuffer` contract declining because the buffer it reads
holds frustum planes) and which is declined outright on some dispatches. A program that cannot
resolve one of its descriptors, and is sometimes not run at all, is precisely a program that writes a
partially-correct node array.

**`CONFIDENCE: HIGH`** on the attribution — per-dispatch pre/post, one writer, one metric, and the
0 → 324 transition is unambiguous. **`CONFIDENCE: MED`** that the unresolved descriptor is *why* its
output is wrong; that link is plausible and untested, and the honest alternative is that some other
part of its lowering is at fault.

**This supersedes the earlier "missing producer" framing without contradicting its falsification.**
Absence was correctly ruled out — 29 submits with `0x413ce6000` fully executing still went cyclic.
The cause is its **output**, which the earlier experiments could not distinguish because none of them
measured what it wrote.

## FOUND: the CYCLES ARE IN THE INPUT — the node records' own child graph (2026-08-15)

Two results, and together they close the question.

### 1. The clear hypothesis is FALSIFIED, with the lever verified

`PROSPER_COMPUTE_ZERO_BEFORE=0x413dc3400:0x20f848417c:2063` zeroes the parent array immediately
before every builder dispatch. The clear demonstrably fires. **The later submits are still cyclic** —
`cycles=63..101`, and now with `oob-roots=0`. So the cycles are **not** stale leftovers in the output
array; the builder writes them into a freshly zeroed one.

### 2. The cycles are in the builder's INPUT

Building the child graph directly from the node records — for each node, its two child references
decoded as `index = (ref >> 3) - 4` when the tag is in `{2,5}` — and testing that graph for cycles:

```
s11238 d15,d16,d19..d27   child-graph cycles = 0
s11238 d37                child-graph cycles = 70     <- appears here
s11637 d15,d16 ...        child-graph cycles = 70     (carried into the next submit)
s11637 d37                child-graph cycles = 110
s12036 ...                child-graph cycles = 110
s13220 d37                child-graph cycles = 117
... every later submit    child-graph cycles = 117
```

**The node records at `0x209cc76000` describe a cyclic child graph, and `0x413dc3400` faithfully
reproduces it as a cyclic parent table.** Everything below about the builder now has its explanation:
its lowering is correct, its output tracks its input, and its input is a cyclic graph.

**The cycles appear between dispatch 27 and dispatch 37 of a submit, and they ACCUMULATE**
(70 → 110 → 117), which is the signature of a structure being partially updated rather than rebuilt.
The programs running in that window are `0x413ce3400` at d30 and **`0x413ce6000` at d36 — the bulk
writer of the record array, and the one whose descriptor at pc156 does not resolve.**

**So `0x413ce6000` is back at the centre of this, for a different reason than before.** The earlier
falsification stands and was correct: its *absence* is not the cause, since 29 submits with it fully
executing still went cyclic. The cause is its **output**. A program that is sometimes declined and,
when it does run, has an unresolved descriptor is exactly a program that can write a partially-correct
node array.

### The next measurement, and it is one run

Attribute the cycle appearance to a dispatch the same way the parent table's was: watch
`0x209cc76000` with the tree watch's **pre/post** attribution and the child-graph cycle count as the
metric, rather than the parent-walk metric which is meaningless on a node array. The instrument
exists; only the analysis differs. That names the writer conclusively instead of by elimination
between d30 and d36.

## THE HYPOTHESIS THIS ALL POINTS AT: the parent array is never cleared

If the builder legitimately links only `{2, 5}` children, then **the slots it does not write must
already hold something that terminates the consumer's walk** — the walk is
`while (i != 0) i = bfe(rec[i], 3, 27)`, so a zero terminates and anything else does not.

**It is not cleared.** The tree watch's pre-image at the builder's dispatch is not zero: it is the
previous phase's stride-4 id array (`{0, 0, 0xa9, 1}` repeating). And the slots the builder leaves
alone were measured holding `0x09249249`, `0x12492492`, `0x2db6db6d` — the Morton dilation constant
and multiples — or, sometimes, zero.

**A stale Morton key read as a parent index is exactly a divergent walk.** `0x09249249 >> 3` is
0x1249249, far past 2,063, which terminates by out-of-range; but `0x12492492 >> 3 & 0x7ffffff` is
0x2492492 — also out of range. The values that *do* trap are the ones left over from an earlier
generation of the parent array itself, which are in range by construction.

**This unifies every measurement on this branch:**

- the builder's lowering is correct (eleven perfect submits) ✓
- its output faithfully follows its input (`both` tracks `pairs`) ✓
- the failure mode is "a store that does not execute, leaving a stale slot" — measured directly ✓
- the perfect submits are the ones where nearly every node has two linkable children, so nearly
  every slot gets written and stale values have nowhere to hide ✓
- the transition is at a route position, because that is when the scene starts containing enough
  triangle leaves for unwritten slots to appear ✓
- no producer decline is necessary ✓

**The prediction that would confirm it:** zero the 2,063-record parent array immediately before
`0x413dc3400`'s dispatch and the cyclicity should vanish, with the tree becoming legitimately sparse
(`pairs < 1030`, `cycles = 0`). That is a diagnostic-only experiment — it does not fix anything, since
on hardware something must be doing the clear and the real question is what — but it is decisive, it
needs no ISA knowledge, and it can be built as a `PROSPER_*` switch in one sitting.

**`CONFIDENCE: MED-HIGH`.** Every measurement fits and nothing contradicts it, but it has not been
tested, and the alternative — that the guest's own build does write every slot through a path prosper
declines — is not excluded.

## The sparse tree IS a faithful consequence of its input (2026-08-15)

Testing the right correspondence — for each parent with only one child, the **two child references in
that parent's own 64-byte node record**:

```
submit=10052  lone-child parents=164
   both refs in {2,5} = 0      exactly one = 155      neither = 9
   commonest (tag0,tag1): (5,0) x82, (0,5) x73, (0,0) x6
```

**Never both. 155 of 164 have exactly one linkable reference** — typically one box32 child (type 5)
and one **triangle** child (type 0), which the `tag == 2 || tag == 5` predicate does not link.

Across submits, over all 2,063 node records:

| submit | pairs / unpaired | both refs linkable | exactly one | neither |
| --- | --- | --- | --- | --- |
| 5943 | **1029 / 2** | **1034** | 100 | 929 |
| 7188 | **1029 / 2** | **1034** | 98 | 931 |
| 8842 | **1029 / 2** | **1033** | 98 | 932 |
| 9247 | **1029 / 2** | **1033** | 98 | 932 |
| 10052 | 676 / 74 | 785 | 266 | 1012 |
| 10449 | 616 / 73 | 775 | 271 | 1017 |

**In the perfect submits `both ≈ 1033` and `pairs ≈ 1029` — they track each other.** In the broken
ones `both` falls to ~780 and `exactly one` rises to ~270. The builder's output follows its input.

**So `0x413dc3400` is faithfully building what it is told to build**, and the defect is that the node
records it reads contain roughly 170 fewer linkable child pairs than they do in the frames that come
out right. Combined with the eleven perfect submits, this is now two independent measurements saying
the same thing: **the builder is not the defect; its input is.**

What remains open is whether that input is *wrong* or merely *different* — a scene with more triangle
leaves genuinely has fewer box-to-box links. Distinguishing them needs to know what the guest expects,
which is the question outstanding with Codex, and it cannot be settled from the output alone. The
`0x209cc76000` record array has 23 writers, so "which producer" is not yet a well-posed question
either.

## RETRACTED: that falsification tested the wrong correspondence (2026-08-15)

**The section below asked the wrong question and its conclusion does not follow.** It tested whether
a record's OWN node type predicts whether that record is paired. The predicate does not act on a
record's own type — it acts on the **two child references stored in the PARENT's node record**. The
right test is whether a lone-child parent's two refs differ in linkability, and it says something
quite different (next section). The section is kept for the record; do not cite its conclusion.

## OLD (wrong test): the sparse tree is not "correct but sparse by design"

This reading was flagged as capable of inverting the whole investigation, so it was tested rather
than left standing. If `0x413dc3400`'s `tag == 2 || tag == 5` predicate legitimately skips other node
types, the tree would be *supposed* to be sparse in frames containing them, the 1,030-pair oracle
would be wrong for exactly the frames called broken, and the defect would be the consumer's unbounded
walk over a legitimately-sparse table instead.

Cross-referencing each parent-table record's pairing state against its **node type** in the record
array, from the same dispatch (the aux dump makes this a same-moment comparison):

| submit | unpaired | node types among UNPAIRED | node types among PAIRED |
| --- | --- | --- | --- |
| 10052 | 74 | `{0: 60, 5: 14}` | `{0: 318, 4: 1, 5: 357}` |
| 11238 | 158 | `{0: 37, 5: 121}` | `{0: 414, 5: 518}` |
| 11637 | 200 | `{0: 29, 5: 171}` | `{0: 313, 5: 587}` |
| 12434 | 183 | `{0: 57, 5: 126}` | `{0: 223, 4: 1, 5: 445}` |

**Node type does not determine pairing.** Types 0 and 5 appear on both sides in every sample — a
type-5 record is sometimes paired and sometimes not, and so is a type-0 record. If the predicate
explained the sparseness, unpaired records would be exactly the types outside `{2, 5}`, and they are
not.

**So the tree is genuinely malformed, the 1,030-pair oracle stands, and the consumer is not at
fault.** The eleven perfect submits already showed the lowering is correct; this shows the sparse
output is not a legitimate alternative shape either. Both point at the input.

(The correspondence used here — parent-table index *i* ↔ record-array index *i* — follows from
`0x413dc3400` writing `parent[(ref >> 3) - 4]` where the same reference indexes the node array.)

## CONCLUSION: both rejecting programs build descriptors from LANE MASKS (2026-08-15)

`PROSPER_DYNTRACE_SGPR=106` on `0x205b654a00`, filtered by program identity:

```
pc=1091 s106 <- KNOWN 0x00000005
pc=1092 s106 <- KNOWN 0x00000001
pc=1097 s106 <- FORGOTTEN words=85ea807e     <- SOP2 0x0b, a B64 op reading EXEC_LO
pc=1098 s106 <- FORGOTTEN words=87ea6a00
pc=1101 s106 <- KNOWN 0x00000004
...
pc=1154 s106 <- KNOWN 0x0000ffc8
```

s106 is known on some paths and lost on others, and where it is lost the source is **EXEC** —
pc1097's `0x85ea807e` is a 64-bit scalar op whose `ssrc0` is `EXEC_LO`.

**So both declined programs compute descriptor fields from lane masks:**

| program | descriptor field | source |
| --- | --- | --- |
| `0x205b654a00` | BVH descriptor `s18` = `-1 + VCC_LO` | VCC_LO from **EXEC** at pc1097 |
| `0x413ce6000` | array selector, `s_mulk_i32 s106, 120` | VCC_LO from **`s_andn1_saveexec_b64`** at pc116/131 |

**A constant-folder cannot resolve either, and no widening of it ever will.** EXEC is a runtime
wave state. This is not a gap in the fold's opcode coverage — the `s_mulk_i32`, `s_addk_i32`,
`s_setreg_b32` and `s_waitcnt_vscnt` corrections made on this branch are all real fixes and none of
them could have helped, which is exactly what their verified-lever negatives showed.

**What this means for the fix.** These descriptors need a mechanism that does not fold: either
resolving the descriptor from guest memory at dispatch time (the `selected_sbuffer` contract's
approach — certify the domain, materialise the record), or a lowering that keeps the descriptor
dynamic. Which one is a design question and needs the ISA read of what the guest intends by deriving
a descriptor field from EXEC — plausibly a lane count or an active-mask popcount used as a size.

**`CONFIDENCE: HIGH`** that both fields trace to lane masks — measured per register, per program, with
a program identity that is actually unique. **`CONFIDENCE: LOW`** on what the guest means by it, which
is the open question worth asking.

## The BVH descriptor's unresolved word is `s18`, and it depends on VCC_LO (2026-08-15)

Watching each register of the descriptor `s[16:19]` in `0x205b654a00`, one run each:

| register | state before pc1180 | site |
| --- | --- | --- |
| `s16` | **KNOWN** at pc1163 | |
| `s17` | **KNOWN** at pc1176 | |
| `s18` | **FORGOTTEN** at pc1169 | `words=80126ac1` = `s_add_u32 s18, -1, vcc_lo` |
| `s19` | KNOWN `0x81000000` on some dispatches | `s_or_b32 s19, s106, 0x81000000` |

**`s18` is the word that fails, and it is `-1 + VCC_LO`.**

So the ray-tracing pass's BVH descriptor cannot resolve because **two of its four words are computed
from VCC_LO** — `s18` at pc1169 and `s19` at pc1177 — and prosper does not track a value into VCC_LO
on this path. That is the same obstacle as `0x413ce6000`'s selector and the same one the execz
liveness guard was about, now demonstrated by direct measurement on the exact register rather than
inferred from the ISA.

**This is the sharpest statement of the frontier available:** GTA V's compiler uses VCC_LO as a
general-purpose scalar register, and prosper's descriptor const-fold loses values through it. Two
programs — the ray-tracing pass and the BVH producer — are declined for exactly this, and between
them they are the ray-tracing path.

What is *not* established: **where** s106's value is lost in `0x205b654a00`. The register watch will
say — `PROSPER_DYNTRACE_SGPR=106` filtered to `program=0x205b654a00`, exactly as was done for
`0x413ce6000` — and that is the next measurement. In `0x413ce6000` the answer was
`s_andn1_saveexec_b64`, i.e. a genuine lane mask that no fold can resolve; if the same holds here,
the fix is a contract rather than a fold, and if it does not, it is a fold gap with a named opcode.

## SCC over-invalidation fixed — and it is NOT what gates the BVH descriptor (2026-08-15)

`PROSPER_DYNTRACE_SCC=1` (added here) reports every transition of the fold's tracked SCC with the pc
and words that caused it. On `0x205b654a00` it found **325 conservative SCC losses**, from three SOPK
encodings — and **two of them do not write SCC at all**:

| count | encoding | writes SCC? |
| --- | --- | --- |
| 188 | `s_setreg_b32` (0x13) | **no** — writes a hardware register |
| 43 | `s_waitcnt_vscnt` (0x17, sdst=NULL) | **no** — a wait, register-transparent |
| 94 | `s_addk_i32` (0x0f) | yes, on signed overflow |

Fixed: `s_setreg_b32` and `s_waitcnt_vscnt` no longer touch SCC, and `s_addk_i32` keeps invalidating
it while now folding its **value** when the destination is known.

**Lever verified: conservative SCC invalidations in that program went 325 → 0.**

**And the reject is unchanged.** `0x205b654a00` still fails with `mode=unresolved-operand pc=1180`.
Because the lever demonstrably moved, this is a **genuine negative, not a void arm**: SCC
over-invalidation is not what gates the BVH descriptor.

So of the four registers in the descriptor `s[16:19]`, dword3 (`s19`) was already observed resolving
to `0x81000000` on some dispatches. **The next measurement is `s16`/`s17`/`s18`** — the base and size
words — one watch each.

The change stays on its own merits: two encodings were being charged an SCC write they do not
perform, which is a correctness defect in the fold independent of this title.

## The BVH descriptor resolves only when SCC does (2026-08-15)

`PROSPER_DYNTRACE_SGPR=19` on `0x205b654a00`, now that the watch prints a real program identity:

```
pc=1171 s19 <- FORGOTTEN words=821380c1        pc=1171 s19 <- KNOWN 0x00000000
pc=1177 s19 <- FORGOTTEN words=8813ff6a        pc=1177 s19 <- KNOWN 0x81000000
```

**The descriptor's dword3 resolves on some dispatches and not others**, ending at `0x81000000` when
it does. pc1171 is `s_addc_u32 s19, -1, 0` — **both operands are inline constants**, so the only
input that can make it unknown is **SCC**.

`s_addc_u32` **is** modelled by the fold (`case 0x04`, guarded by `if (scc < 0) { ok = false; }`).
*(I first reported it as unmodelled, from a grep for `kSop2OpcodeAddcU32` that missed the literal
`0x04` at the case label. Corrected here.)* So the BVH descriptor's resolution reduces to: **is SCC
tracked across the instructions before pc1171?**

That makes SCC invalidation the lever, and it is exactly what the `s_mulk_i32` change touched — that
op does not write SCC and the fold was clobbering SCC for it anyway.

**Suggestive but confounded, recorded as such.** Across runs on this branch `0x205b654a00`'s reject
changed from a `compute-struct-reject` to the pc1180 descriptor reject, and the total skip count fell
18 → 15 → 14 → 13. Neither is evidence: the runs differ in more than one variable and the route
reaches different phases. **A clean A/B needs one flag toggled with the artefact hashed first**, which
is not what these runs were.

**The concrete next step** is a census of what invalidates SCC on the path to pc1171 in this program —
`scc = -1` has a handful of sites in the fold, and each is either a real SCC write or a conservative
one like the `s_mulk_i32` case that was corrected.

## The contract reads a BVH NODE-REFERENCE array as a selector array (2026-08-15)

The selector histogram, taken from the same source records the contract's own domain proof walks:

```
[gta-selected-sbuffer]  selectors distinct=2064 (selector:count) 4:1 5:1 6:1 7:1 8:1 9:1 10:1 ...
[gta-selected-sbuffer]  record-4 selector would be 4; outer has 5 records of 120 bytes
```

**All 2,064 source records have a DIFFERENT selector, and they run 4, 5, 6, 7, 8, … — i.e.
`selector = index + 4`.**

That is not a selector. **It is the BVH node-reference encoding**, the same one `0x413dc3400` decodes
at pc618–619 with `v_lshrrev_b32 v36, 3, v69` followed by `v_add_nc_u32 v36, -4, v36` —
`index = (ref >> 3) - 4`. The source array at pc70 (stride 8, 2,064 records) is an array of **node
references**, and `first >> 3` recovers `index + 4`, exactly as observed.

**So `selected_sbuffer_domain`'s proof passes for an accidental reason.** It admits a record when
`selector * 120` is either exactly 480 (record 4) or wholly out of bounds. With `selector = index+4`:
index 0 gives selector 4 → offset 480 → "record 4"; indices 1..2063 give selectors 5..2067 → offsets
600..248,040, every one past the 600-byte buffer → "wholly OOB". **Both arms are satisfied by an
array that is not a selector array at all.** The proof is not wrong about the arithmetic; it is
answering a question about the wrong data.

The contract then reads 16 bytes at `480 + 8` of a 600-byte buffer that holds float data, gets a
non-descriptor, and declines — correctly, at the last possible moment, having been misled four steps
earlier.

**This is the complete account of `0x413ce6000`'s reject**, and it is a contract defect rather than a
missing lowering, a stale buffer, or a wrong selector value:

1. the source array at pc70 holds BVH node references, not table selectors;
2. `selected_sbuffer_domain` reads `first >> 3` as a selector and its two admissible arms are both
   satisfied by that data for arithmetic reasons;
3. the contract therefore reads record 4 of the outer buffer;
4. the outer buffer holds float data in this scene (unit-normal plane equations, measured);
5. `selected_sbuffer_target_descriptor` rejects it;
6. no `selected_sbuffer_soffset` authority is published, so the descriptor for
   `buffer_load_dwordx3` at pc156 never resolves;
7. `mode=unresolved-operand pc=156`, and the program is declined.

**`CONFIDENCE: HIGH`** on steps 1–5, all measured. **`CONFIDENCE: MED`** on 6–7 being the only
remaining link, since the recompiler's descriptor matching has three routes and only the contract one
has been traced.

## The `selected_sbuffer` contract is over-fitted: the "descriptor array" holds FRUSTUM PLANES (2026-08-15)

Dumping **all five** records of the outer array on decline, rather than only the one the contract
reads, settles what one record could not:

```
record=0@+8   words=3e177e0d:3d39304e:3f7ceb16:beec6371
record=1@+128 words=3d044294:bc9a100f:3ce5d234:be5601a6
record=2@+248 words=bdd106eb:3f7e94ad:bccf32ed:beee4863
record=3@+368 words=be6fa5cf:bdb9257a:3f581219:bdf01a64
record=4@+488 words=3f1a1f1a:bdc0f619:3f4afad1:bfa0b479
```

**None of the five is a descriptor. All five are floats, and they are unit normals plus a scalar:**

| record | xyz | ‖xyz‖ | w |
| --- | --- | --- | --- |
| 0 | (0.1479, 0.0452, 0.9879) | **0.999** | −0.461 |
| 2 | (−0.1020, 0.9944, −0.0253) | **1.000** | −0.465 |
| 4 | (0.6021, −0.0942, 0.7929) | **1.001** | −1.256 |

A unit 3-vector with a scalar is a **plane equation**. Five of them at stride 120 is a camera
frustum, not a descriptor table.

**So in the gameplay scene, the buffer the contract reads as a five-entry V# array holds frustum
planes.** The contract was derived from a phase where record 4 did hold a V#, and it does not
generalise. That also explains its companion `reject=consumer-resource`.

This is why the earlier framing — "record 4 is stale, find its producer" — was the wrong question.
The buffer is not stale; it is a **different buffer's worth of data**, correctly written by whoever
owns it. Either `0x413ce6000`'s pc153 does not load a descriptor on this path in this scene, or the
resource the contract binds at `fetch_pc=153` is not the one the guest means here.

**`CONFIDENCE: HIGH` that these are planes and not descriptors** — three of five have unit-length
normals, which floating-point garbage does not do. `CONFIDENCE: MED` on the consequence, because
"the contract binds the wrong resource" and "the shader takes a different path here" both fit and
have not been separated.

## `0x209cc76000` is a SHARED POOL with 23 writers, not a dedicated record array (2026-08-15)

Watching the **whole** 132,032-byte range (`PROSPER_COMPUTE_TREE_WATCH=0x209cc76000:33008`) rather
than its first 8 KB finds **23 distinct writing programs**, not the seven the narrower window showed:

```
0x413e15400 652   0x413e14200 651   0x413e14500 651   0x413cf9200 603   0x413cf9000 569
0x413cdc200 218   0x413cf5400  43   0x413cf6100  43   0x413ce6000  32   0x413e1ff00  31
0x413ced900  10   0x413d1bf00   9   0x413d21600   8   0x413dc3400   6   0x413d87800   5
0x413e16400   5   0x413e13000   3   0x413e14900   2   0x413d21800   1   0x413d21c00   1
0x413d22000   1   0x413d22b00   1   0x413e13200   1
```

**So this is a shared scratch pool that many programs reuse across phases**, exactly like the
traversal table itself. `132032 = 2063 x 64` is how `0x413dc3400` and `0x413ce6000` *view* it, not
proof that the allocation belongs to them.

Two consequences for anything built on the earlier analysis:

- **"The only program binding the full array is `0x413ce6000`" was a statement about the narrow
  window.** Over the full range there are 23 writers, and the three busiest — `0x413e14200`,
  `0x413e14500`, `0x413e15400`, ~650 changes each — were entirely invisible to every census before
  this one. A watch window narrower than the buffer under study reports a writer set that is
  guaranteed incomplete.
- **The tag histograms remain valid** because they were captured by the aux dump *at the builder's
  own dispatch*, which is the only moment that matters. But attributing a tag change to a producer is
  not possible from writer counts alone with 23 of them interleaved; it needs the last-writer-per-
  record, which nothing currently records.

The early phase is visible in the same data and is a different workload: at submit 3864
`0x413ced900` leaves the pool all-zero (`{0: 2063}`) and `0x413d1bf00` fills it progressively over
eight dispatches. Tag distributions there span all eight values, unlike the two-regime pattern at
builder time.

## The builder's INPUT differs between the two regimes (2026-08-15)

`PROSPER_COMPUTE_TREE_WATCH_AUX=0x209cc76000:33008` captures the 2,063 x 64-byte record array
alongside every builder transition, clean and cyclic, from one run. The tags the six store predicates
test are `record.dword0 & 7` and `record.dword1 & 7`.

**Every submit where the builder emits a perfect tree shares one input signature, and no broken
submit has it:**

| submit | pairs / unpaired | tag(dword0) | tag(dword1) |
| --- | --- | --- | --- |
| 5943 | **1029 / 2** | `{0:970, 2:2, 5:1088, 7:3}` | `{0:977, 1:5, 5:1078, 7:3}` |
| 7188 | **1029 / 2** | `{0:972, 2:2, 5:1086, 7:3}` | `{0:977, 1:5, 5:1078, 7:3}` |
| 8016 | **1029 / 2** | `{0:973, 2:2, 5:1085, 7:3}` | `{0:978, 1:5, 5:1077, 7:3}` |
| 8842 | **1029 / 2** | `{0:973, 2:2, 5:1085, 7:3}` | `{0:978, 1:5, 5:1077, 7:3}` |
| 9247 | **1029 / 2** | `{0:973, 2:2, 5:1085, 7:3}` | `{0:978, 1:5, 5:1077, 7:3}` |
| 5528 | 663 / 64 | `{0:1120, 2:3, **4:2**, 5:935, 7:3}` | `{0:1122, **1:15**, 5:923, 7:3}` |
| 6358 | 608 / 51 | `{0:1098, 2:3, **4:2**, 5:957, 7:3}` | `{0:1107, **1:15**, 5:938, 7:3}` |
| 10052 | 676 / 74 | `{0:1133, 2:3, **4:2**, 5:922, 7:3}` | `{0:1134, **1:15**, 5:911, 7:3}` |

Two differences are systematic in the early broken submits: **tag 4 appears in dword0** (exactly 2
records, never present in a perfect submit) and **tag 1 in dword1 jumps from 5 to 15**.

**But neither is the whole story, and the record says so.** The later broken submits (11238 onward)
carry `1:5` and no tag 4 — the perfect signature on those two axes — and are still broken
(`pairs≈861, unpaired≈193`), with tag 5 *higher* than in the perfect submits (1301 against 1085). So
"tag 4 present" and "tag1 == 15" are correlates of the early regime, not the mechanism. Do not
promote them to a rule.

**What is established:** the array the builder reads changes materially with scene state, and the
builder's output tracks it. Combined with the eleven perfect submits, that puts the defect **upstream
of `0x413dc3400`** — in whatever produces `0x209cc76000`.

**Its producers, from `PROSPER_COMPUTE_TREE_WATCH=0x209cc76000`:** `0x413cf9000` (117 changes),
`0x413cf9200` (117), `0x413cf5400` (29), `0x413cf6100` (29), `0x413ce6000` (26), `0x413d1bf00` (2),
and `0x413e1ff00` (3, **`toucher=0`**). `0x413cf9200` has 20 recompile-empty dispatches of 678.

**RETRACTED — `0x413e1ff00` does NOT write bytes it does not bind.** That was my own instrument
producing a phantom. Its binding 7 is `base=0x209cc76080 size=160 stride=32`, i.e. **128 bytes inside
the watched window**. The tree watch detects changes anywhere in the WINDOW but its `toucher` field
asked whether a program binds the window's **first byte** — two different questions, and the
disagreement reads as an out-of-bounds write. Fixed with `compute_address_window_hits`, and the
retraction is recorded rather than quietly dropped because the phantom was reported as a lead before
the resource map contradicted it.

The same run corrects the producer picture in a way that matters: `0x413cf9000` binds only
`0x209cc76000 size=320`, and `0x413cf9200` binds `size=320` plus a 64-byte view at `+0x140`. **They
write the first 320 bytes — five records — not the bulk.** The only program binding the full
132,032-byte array is `0x413ce6000` (bindings 13/14/17). So it *is* the bulk producer of the records,
which the falsification above does not contradict: it executes on 29 of the cyclic submits, so its
running is not the variable.

## THE BUILDER'S LOWERING IS CORRECT — it emits a perfect tree for eleven submits (2026-08-15)

`PROSPER_COMPUTE_DISPATCH_LOG` now carries the launch geometry, so the outcome and the launch can be
read on one line. Across the whole route:

```
submit  threads       local     tree
4802    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0     <- exactly correct
5217    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
5632    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
6047    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
6463    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
6877    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
7292    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
7706    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
8121    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
8951    2063x1x1      256x1x1   pairs=1030 unpaired=0 cycles=0
9360    2063x1x1      256x1x1   pairs=852  unpaired=247 cycles=57  <- and never correct again
...     2063x1x1      256x1x1   cyclic for the remaining 30+ submits
```

**Eleven consecutive submits at pairs=1030, unpaired=0, cycles=0 — the exact clean ground truth — at
an identical launch geometry, from the same compiled module.** A miscompiled shader does not produce
the exactly correct answer eleven times in a row.

**So `0x413dc3400`'s lowering is CORRECT, and the defect is in what it is fed.** Every
lowering-side hypothesis for this program is retired by this one measurement: barrier placement,
LDS sizing, wave model, the compaction, the exec-mask predication, the store addressing. They all
produce a correct tree for eleven submits.

The launch is identical on both sides of the boundary — `threads=2063x1x1 local=256x1x1` throughout —
so an active-record count change is not the trigger either. The transition point is **not a fixed
submit number**: one run breaks at ~6795, another at ~9360, so it tracks route position and scene
content rather than a dispatch ordinal.

**The remaining question is therefore narrow and concrete: what changes in the builder's input at
that boundary?** Its tags come from `0x209cc76000` (2,063 × 64 bytes, binding 4 / fetch-pc 86).
`PROSPER_COMPUTE_TREE_WATCH_AUX=0xADDR:DWORDS` (added here) dumps a second guest range alongside the
watched table, and dumps the builder's clean transitions too so there is a control from the same run
rather than only cyclic samples.

## FALSIFIED: the missing-producer hypothesis is dead (2026-08-15)

`PROSPER_COMPUTE_DISPATCH_LOG` (added here) records **one line per dispatch** with its outcome, which
no existing signal did — every other one is deduped per program, and reading a once-per-program line
as a per-dispatch property is how the root-cause claim two sections down got published and retracted.

One 200 s route, 46 submits carrying a `0x413dc3400` dispatch:

| | tree CYCLIC | tree clean |
| --- | --- | --- |
| `0x413ce6000` had a declined dispatch that submit | 10 | 0 |
| `0x413ce6000` executed on **every** dispatch | **29** | 7 |

**29 submits in which the producer ran on every single dispatch and the builder still produced a
cyclic tree.** A producer decline is therefore **not necessary** for the corruption, and the
"`0x413ce6000` fails to recompile → stale tags → cyclic tree" story is finished. Do not restart it.

Per-dispatch outcomes over the run: `0x413ce6000` 129 executed / 10 recompile-empty; `0x413cf9200`
658 executed / 20 recompile-empty; `0x413dc3400` 53 executed / 0 failures. Those recompile-empty
dispatches are real gaps worth closing on their own merits — an unsupported program is a fatal gap —
but they are **not** the cause of this defect.

### What the same data shows instead: a route-position boundary

The builder's output is clean for the first seven submits and cyclic from submit ~6795 onward,
continuously:

```
3894  clean     6795  CYCLIC     9673  CYCLIC
4307  clean     7211  CYCLIC    10076  CYCLIC
4720  clean     7625  CYCLIC    10482  CYCLIC
5135  clean     8039  CYCLIC    10875  CYCLIC
5550  clean     8457  CYCLIC    11271  CYCLIC
5965  clean     8867  CYCLIC    11674  CYCLIC
6380  clean     9274  CYCLIC    12077  CYCLIC
```

This is a **transition at a point in the route**, not a per-submit coin flip. Whatever changes around
submit 6795 — scene content reaching some size or shape — is the thing to characterise next. The
first clean submit also shows `0x413ce6000` dispatching **94** times against 1 thereafter, so the
early phase is a different workload entirely and the clean result there may not be comparable.

## Causal A/B: forcing the producer off collapses the builder entirely (2026-08-15)

`PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700,0x413ce6000`, lever verified in the log
(`-> 2 program(s) will be declined`), 200 s route, against the baseline arm.

| | baseline | producer forced off |
| --- | --- | --- |
| `0x413dc3400` table-changing dispatches | **40** | **1** |
| `0x413dc3400` clean -> cyclic | 37 | **0** |
| clean -> cyclic, all programs | 40 (37 from the builder) | 46 (all from `0x413d88400`) |

**`0x413dc3400` essentially stops writing a tree.** So its output does depend on `0x413ce6000`
having run — the dependency the resource alias predicted is real and now demonstrated, not inferred.

**The confound, stated because it is not excluded.** A declined compute dispatch clears
`producer_epoch_ok`, and the next `ParserStall` latches `indirect_dependencies_ok` false for the rest
of the submit — the indirect latch this document opens with. Forcing `0x413ce6000` to decline
therefore suppresses later indirect dispatches too, so this arm cannot separate

  (a) `0x413dc3400` runs and reads a stale record array, from
  (b) `0x413dc3400` is never dispatched at all.

Both predict what was measured. Separating them needs a per-dispatch execute/decline record for both
programs in the same run, correlated frame by frame — which is the measurement the retracted
root-cause claim needed and never had.

Also note the dumped post-images differ in *provenance* between the arms: in the forced-off arm the
clean -> cyclic transitions come from `0x413d88400`, not the builder, so their `pairs=0` is not
comparable to the baseline builder's `pairs≈819`. The comparable figure is the dispatch count.

## CORRECTION (2026-08-15): the "never executes" claim above is WRONG

Watching the record array itself — `PROSPER_COMPUTE_TREE_WATCH=0x209cc76000:2063` — refutes the
strongest form of the claim in the section below. **`0x413ce6000` does write it**, 26 times on a
150 s route, `toucher=1`. It is not declined on every dispatch; the `[compute] skip unsupported
program` line that suggested otherwise fires **once per program ever**, so it reports "failed at
least once", never "never ran". That distinction is stated elsewhere in this document and I still
built a causal claim on the wrong side of it.

The array's actual writers on one route:

| program | changes | toucher |
| --- | --- | --- |
| `0x413cf9000` | 117 | 1 |
| `0x413cf9200` | 117 | 1 |
| `0x413cf5400` | 29 | 1 |
| `0x413cf6100` | 29 | 1 |
| `0x413ce6000` | **26** | 1 |
| `0x413d1bf00` | 2 | 1 |
| `0x413e1ff00` | 3 | **0** |

**What survives** from the section below, because it was measured rather than inferred:

- `0x413dc3400` corrupts the parent table on 37 of 40 dispatches. Unchanged.
- `0x413dc3400`'s tag source is `0x209cc76000`, binding 4 / fetch-pc 86 — the resource-map alias is
  exact and stands.
- `0x413ce6000` writes that same array, is *sometimes* declined, and its reject is
  `unresolved-operand pc=156`. Also stands.

**What does not survive:** "`0x413ce6000` never executes, therefore the tags are stale, therefore the
tree is cyclic." The producer runs most of the time, so a missing-producer story cannot be asserted
on this evidence. It remains a *candidate* — a partially-written array would still leave some records
stale — but the step from "sometimes declined" to "these particular tags are stale" is not made, and
the frame-level correlation between a decline and a corrupted tree has not been measured. **Do that
measurement before building on it.**

`0x413e1ff00` changing the array three times with **`toucher=0`** is the case the tree watch was
built to be able to report: a program that changes bytes without binding a range containing them.
That is either an out-of-bounds write or a binding path the resource traversal does not model, and
it is unexamined.

## `0x413ce6000` is a producer of the tags, and is sometimes declined (2026-08-15)

**Read the CORRECTION above first — this section's causal claim was overstated and the
"never executes" premise is refuted. The resource alias it establishes is sound.**

The chain is complete and every link is measured.

**The alias, from `PROSPER_COMPUTE_RESOURCE_MAP`:**

```
0x413dc3400  binding=4  fetch-pc=86   base=0x209cc76000 size=132032 stride=64   (READ)
0x413ce6000  binding=13 fetch-pc=212  base=0x209cc76000 size=132032 stride=64   (STORE)
0x413ce6000  binding=14 fetch-pc=214  base=0x209cc76000 size=132032 stride=64   (STORE)
0x413ce6000  binding=17 fetch-pc=253  base=0x209cc76000 size=132032 stride=64   (STORE)
```

`132032 = 2063 x 64` — one 64-byte record per node, exactly the node count. `fetch-pc=86` is the
load Codex identified as the source of the classified references: `pc86 loads v56/v57 from
s24[compacted_node].dword[0:1]`, and `v36 = v56 & 7`, `v37 = v57 & 7` are the tags the six store
predicates test. Eight more of `0x413dc3400`'s bindings (6, 7, 8, 9, 10, 11, 12, 15) are the same
buffer.

**So `0x413ce6000` writes the record array whose tags `0x413dc3400` reads — and `0x413ce6000` is
declined on every single dispatch.**

### The full causal chain

1. `0x413ce6000` fails to recompile: `compute-struct-reject execz pc=76 scalar write pc=84 leaves
   vcc-half s106 live at merge pc=139`. It is realized 40 times per route, at d36, and executes zero
   times.
2. Its output — the 2,063 x 64-byte node record array at `0x209cc76000` — is therefore never written.
3. `0x413dc3400` at d37 reads its six classified references out of that array.
4. The stale references fail the `tag == 2 || tag == 5` predicate, so the matching child stores never
   fire — which is exactly the measured failure mode: **a store that does not execute, leaving the
   slot holding a stale value**.
5. 220–461 parents end up with one child instead of two → the parent array is cyclic.
6. `0x413dc6700` at d39+ walks it with a loop that has no iteration bound and no cycle detector →
   GPU hang → RADV device loss → live compute disabled process-wide → every later indirect draw
   dropped → **no world**.

**This is the charter's rule in its purest form: an unsupported program is a fatal gap, not an
acceptable skip.** One declined compute program, six dispatches upstream, removes the world.

It also explains why `0x413ce3400` gets its tree right. Codex's ISA read: it computes both child
indices directly from the Morton range/split and stores both links unconditionally, with no tag
predicate anywhere in its 391 instructions. It does not depend on `0x209cc76000`, so a missing
producer cannot reach it.

**Do not "fix" this by removing `0x413dc3400`'s predicate.** The guest declines to dereference
non-materialised references on purpose; making the stores unconditional would turn unmaterialised
references into indices and change guest semantics. The fix is upstream: make `0x413ce6000` compile.

### Status of that fix

`PROSPER_VCC_SCALAR_DATA_MERGE=1` (this branch) widens the execz VCC-half proof from "not read" to
"not read as a lane mask" and clears the first reject; `0x413ce6000` then fails at a *second* path
that logs nothing under any variable. Making that second reject legible is the immediate next step —
it is one of the five programs in the `reason=unrecorded` set.

Note on a red herring: `[compute] program 0x413ce6000 is a proven no-op (only proven no-backing
resources)` does appear, but only for an instance with **zero** backed resources (dispatch 968). At
d36, where it matters, it has **17** backed resources including the three stores above.

## THE SECOND TABLE IS BUILT PERFECTLY — by a different program (2026-08-15)

`PROSPER_COMPUTE_TREE_WATCH=0x20f848a240:2063`, same route. The two addresses are **not** a
ping-ponged copy of one structure: they have disjoint writer sets.

| address | writers | result |
| --- | --- | --- |
| `0x20f848417c` | `0x413dc3400`, `0x413d88400`, `0x413e1c300`, `0x413cee500` | **37 of 40 dispatches leave it cyclic** |
| `0x20f848a240` | `0x413ce3400`, `0x413cf5400`, `0x413d88400`, `0x413cdc200` | **clean; 3 of 133 transitions cyclic, each repaired immediately** |

And the decisive line: **`0x413ce3400` produces a PERFECT tree at `0x20f848a240`, every single time.**

```
program=0x413ce3400 changed=2061 pre{... pairs=0 unpaired=0}
                              post{cycles=0 cyclic-roots=0 oob-roots=0 max-depth=11 pairs=1030 unpaired=0}
```

`pairs=1030, unpaired=0, cycles=0, max-depth=11` — exactly the clean ground truth, on all 41 of its
dispatches. So **prosper can already build this structure correctly.** The defect is specific to
`0x413dc3400`, and there is now a working reference program to differentially compare against.

The two are different programs, not two instances of one:

| | `0x413ce3400` (correct) | `0x413dc3400` (broken) |
| --- | --- | --- |
| instructions | 391 | 765 |
| DS (LDS) ops | **0** | 5 (`ds_write_b32` x2, `ds_add_rtn_u32`, `ds_read_b32` x2) |
| `s_barrier` | **0** | 2 |
| CFG dispatchers | — | 3 (one per barrier-free phase) |
| MUBUF | 29 | 37 |

**The distinguishing features of the failing one are exactly the workgroup-cooperative machinery**:
the LDS stream compaction Codex decoded at pc47..74 (`ds_add_rtn_u32` hands each qualifying lane a
ticket; the two barriers bracket counter initialisation and list publication), and the barrier-phased
CFG dispatcher that machinery forces. The correct builder has none of it.

That also resolves an inconsistency in the record: the consumer's own captures show ~1029-1030 pairs,
which matches the `a240` tree rather than `417c`'s 919. The consumer's bound table address alternates
between the two across frames, so it walks a good tree on some frames and the broken one on others —
which is why the hang arrives at a particular dispatch rather than the first.

## The failure mode, exactly: one of the two per-parent stores does not execute

Established from both images of 18 clean -> cyclic transitions.

**Ground truth.** Two clean captures give the target unambiguously: parent-multiplicity
`{2: 1030, 1: 2}` — every parent has exactly two children — with 1,032 distinct parent values over
0..2062, 515 of them above 1031, and 1,031 indices never a parent. Leaves and internals are
interleaved, not segregated into low and high halves.

**`0x413dc3400`'s output** measures `{2: ~872, 1: 220..461}`. For 220–461 parents **exactly one of
the two children carries that parent**, and the missing child's slot still holds a *stale* value —
`0x09249249`, `0x12492492` or `0x2db6db6d`, which are the Morton dilation constant times 1, 2 and 5,
or plain zero. Those slots were never written by this dispatch. **So this is a store that does not
execute, not a store that computes a wrong address or a wrong value.**

### The store site

All six stores sit in six separately exec-predicated blocks of identical shape (mnemonics from
prosper's own decoder, `rdna2_to_spirv.cpp` VOP2 0x16 `v_lshrrev_b32`, 0x1B `v_and_b32`,
0x25 `v_add_nc_u32`):

```
pc=0611  v_and_b32          v36, 7, v69      ; tag = child_ref & 7
pc=0612  v_cmp_eq_u32       s8,  2, v36
pc=0614  v_cmp_eq_u32       vcc, 5, v36
pc=0615  s_or_b64           vcc, vcc, s8     ; tag == 2 || tag == 5
pc=0616  s_and_saveexec_b64 vcc, vcc
pc=0617  s_cbranch_execz    -> 622
pc=0618  v_lshrrev_b32      v36, 3, v69      ; index = child_ref >> 3
pc=0619  v_add_nc_u32       v36, -4, v36     ; index -= 4
pc=0620  buffer_store_dword v40, v36, s0, 0  ; idxen=1, offen=0
pc=0622  s_mov_b64          exec, vcc
```

A child is written only when its reference's low three bits are 2 or 5, into slot `(ref >> 3) - 4`.
**Whether tags outside `{2, 5}` are written by a LATER PASS is the open question** — if they are,
a stale slot here is expected and the defect is a missing program rather than this one (#2542).

### A gap in the toucher census that may matter more

The baseline routed run **skips 13 compute programs as unsupported**, and a skipped program never
realizes a resource table — so `PROSPER_COMPUTE_ADDRESS_WATCH` and the tree watch are
**structurally incapable of seeing any of them**. The clean "no unknown writer" result is therefore
a statement about programs that ran, and only those. Baseline skip list:

```
0x2042f49a00 0x205b545c00 0x205b54ee00 0x205b5e8600 0x205b654a00 0x205b657200
0x205b658800 0x205b67ce00 0x413ce6000 0x413cf9200 0x413cf9a00 0x413cf9d00 0x413d14100
```

If one of those is the pass that fills the slots `0x413dc3400` deliberately leaves alone, the cyclic
table is a **missing program**, not a miscompiled one, and the fix is to make it recompile. Per the
charter's rule that an unsupported program is a fatal gap rather than an acceptable skip, these are
the next thing to implement regardless of how this particular question resolves.

## Ruled out / retracted here

- **"Sibling pairs are `(odd, even)`."** Retracted. The clean table has 204 parents whose children
  are `(2j+1, 2j+2)` and **828** whose children are `(2j, 2j+1)` — pairs are adjacent at arbitrary
  parity. The "first broken pair k=205" figure was computed under the fixed-alignment assumption and
  its alignment part goes with it. The adjacency-based unpaired and cycle counts never assumed
  alignment and stand.
- **"Slot 300's bits[2:0] are an anomaly."** Retracted. `post-36-3.bin` has bits[2:0] == 0 in all
  2,063 records while `live-a240.bin` carries a mix of 0 and 2, so that field varies legitimately
  between clean tables.
- **The LDS-undersizing hypothesis.** Dead: the module declares a Workgroup array of exactly 384
  dwords = 1,536 bytes = 3 × 512-byte `COMPUTE_PGM_RSRC2.LDS_SIZE` granules, so the real allocation
  is plumbed through and matched, not defaulted.
- **Synchronization / race hypotheses for this program.** The broken-pair patterns from two
  different frames share a 60-character suffix exactly, and `min` damage index is identical across
  frames: the failure is deterministic given the input.
- **A lane, wave or workgroup boundary effect.** `k mod 2/3/4/8/16` are all flat.
- **`PROSPER_NO_NATIVE_COMPUTE_SUBGROUP=1` as an A/B arm.** Void, not negative: it skips 15 *more*
  programs including `0x413d88400`, so `0x413dc3400` never dispatches and the phase never runs.

## What the structure IS: an LBVH / binary-radix-tree parent table from Morton keys

Identified by Codex from the retained ISA (#2542), and it reframes everything below.

- **2,063 = 2 × 1,032 − 1** — exactly the node count of a full binary tree with 1,032 leaves and
  1,031 internal nodes.
- **Adjacent pairs are SIBLINGS** sharing one parent; bit 30 identifies the child side. The root is
  the single unpaired node. That explains the count, the pairing, the parent chase and the depth
  parity together, where "union-find with path compression" only explained the chase.
- `0x413cee500` contains the standard 3D Morton bit-dilation constants `0x030000ff`, `0x0300f00f`,
  `0x030c30c3` and **`0x09249249`**, then stores `buffer_store_dwordx3` at pc274 — Morton key
  generation.
- `0x413e1c300` loads two three-dword records, runs a 64-lane LDS compare/exchange loop, and stores
  two three-dword records — a Morton sort/merge pass upstream of topology construction.

**CORRECTION — `0x09249249` is NOT an empty-slot sentinel.** This document said it was. It is the
Morton dilation mask, used as a literal by the key generator. In the captured parent view it happens
to behave as an out-of-range terminator, but that may be an intentional empty value, a Morton key
left in repurposed scratch, or another phase's encoding. Call it an **observed OOB/unused value**
until its producing store is identified.

**Naming:** the "malformed head/tail pair" score is measuring a real property, but the neutral name
is **unpaired sibling records**. Keep it as a quantitative correlation and do NOT promote "must be
zero" to a correctness rule: clean samples tolerate up to 13, and the slot-level causal test failed.
A sharper check is whether the active-node count predicts exactly 1,031 sibling pairs.

## Access direction: eight touchers, FIVE may-writers

Codex joined the census's `fetch-pc` values to the retained guest instructions. MUBUF `0x0c..0x0e`
are loads, `0x1c..0x1e` are stores:

| program | watched instructions | direction |
| --- | --- | --- |
| `0x413ce3400` | pc35, 66, 68 … 421, all `buffer_load_dword{,x2}` | **read only** |
| `0x413ce6000` | pc36 `buffer_load_dword` | **read only** |
| `0x413cea300` | terminator-only, `fetch-pc=0xffffffff` | **no data access** |
| `0x413cee500` | pc274 `buffer_store_dwordx3` | **write** |
| `0x413d88400` | 18 watched pcs, all stores | **write** |
| `0x413dc3400` | pc597/608/620/632/644/656 stores | **write** |
| `0x413dc6700` | pc91 load; pc53/65 and 618..677 stores | **read/write** |
| `0x413e1c300` | pc86/95 loads x3; pc166/176 stores x3 | **read/write** |

So the may-write set is **`0x413cee500`, `0x413d88400`, `0x413dc3400`, `0x413dc6700`,
`0x413e1c300`**. The two read-only programs and the terminator leave the writer investigation.

**A statically writable descriptor whose range contains the address is a CANDIDATE writer**, not
proof that a given invocation wrote that 4-byte slot; changed-byte pre/post evidence closes that.

**Address-boundary caveat that must not be lost:** `0x413e1c300`'s observed view is
`base=0x20f8482140 size=33024`, and `base + size == 0x20f848a240` **exactly**. It therefore may write
the first table and proves nothing about the second. The census must be re-run for `0x20f848a240`
before this eight-program set is carried across both ping-pong tables.

## No guest-side escape makes a reachable cycle safe

Checked by Codex against the retained ISA: the pre-loop bound only excludes padded threads (on the
problematic shape `s18 = 2063` and the launch has 2,063 guest threads, so every active root is
represented); the loop has no iteration count and no cycle detector; `v_cmpx_ne_u32` only retires
lanes reaching index zero, so a lane inside a cycle stays active forever; and the `s24` comparison
happens **after** the walk, so it cannot guard entry.

A builder may hold transiently inconsistent parent links while constructing the hierarchy. Hardware
still cannot launch this consumer on reachable cyclic links — they must be repaired before it,
excluded from its active root set, or absent from the bytes it consumes.

## The current account — read this before anything below it

This document is layered: it grew as an investigation log, and several sections below are historical
transcripts kept for the evidence beside them. **Where a lower section disagrees with this one, this
one is current.** Each layer that was superseded now says so where it sits.

As of 2026-08-14, established and each measured rather than inferred:

1. **The missing world is one compute program.** `0x413dc6700` hangs the GPU into a RADV hard
   recovery. That disables live compute for the whole process, so every later indirect draw is
   dropped — and GTA V's world is GPU-driven, so those indirect operations *are* the world. Skipping
   it (`PROSPER_COMPUTE_SKIP_PROGRAM`) gives 0 device losses and the first real scene content.
2. **The hang is a non-terminating loop in that program.** Its 903-dword body contains exactly one
   backward branch (guest pc97 → pc88). The trip-bound witness fires there and reports that no
   invocation ever reached a dispatch ordinal past the loop body; the fence-wait duration
   for that dispatch is ~2,045 ms against sub-millisecond for every other dispatch in the route; the
   device loss follows on the next dispatch.
3. **The loop's data is cyclic at dispatch time.** Pre-dispatch, 806 of 1,782 reads receive a table
   in which 1,805–2,062 of 2,063 roots lead into a cycle. The guest loop cannot terminate on that.
4. **Our lowering of that loop shape is correct.** A hand-built kernel with the same
   `v_cmpx` / `s_cbranch_execz` / back-edge shape runs correctly on real Vulkan
   (`tests/test_cfg_trip_bound.cpp`), so the recompiler is not what fails to exit.

**So the open question is why the table is cyclic** — not whether the loop spins, and not whether we
lower it correctly.

5. **Eight programs TOUCH the table; the writer set is UNKNOWN.** A containment census over a full
   route names `0x413ce3400`, `0x413ce6000`, `0x413cea300`, `0x413cee500`, `0x413d88400`,
   `0x413dc3400`, `0x413dc6700`, `0x413e1c300`. **"Touch" is a resource binding, not an access
   direction** — the census cannot separate a reader from a writer, so it identifies no writer at
   all. An earlier revision of this list said "at least two programs write the table"; that was a
   matcher artifact (base equality, blind to a view whose base differs) and is withdrawn.
6. **One program's write quality tracks the damage — a correlation, not an identification.**
   `0x413dc3400`'s malformed-pair count separates 54 clean reads (0..13) from 126 cyclic ones
   (19..106) with no overlap. Its slot-level causal test **failed**: 14% of cycle nodes sit on
   malformed slots against a 4.4% base rate. It is one candidate among eight.
7. **`0x413dc3400`'s store path is lane-predicated.** All six of its table stores sit inside an
   `s_and_saveexec_b64` / `s_cbranch_execz` region, so *which slots are written* is decided by a
   per-lane mask — and the defect's character is membership, not arithmetic: every record is
   individually well-formed, in the wrong combination.

**The gating question is which of the eight programs WRITE the table.** The census measures a
resource binding, not an access direction, and every downstream narrowing depends on that
distinction — including whether `0x413dc3400` is a writer at all. Only once the writer set is known
does "why do its writes go bad" become answerable, and even then what exists today is a
correlation on an identical module (same SPIR-V hash, launch and 38 buffers either side of the
transition), not established cause.

**Not established, and explicitly tested:** that each malformed pair becomes a 2-cycle. On a same-run
join only 14% of cycle nodes sit on malformed slots against a 4.4% base rate — real enrichment, not a
mechanism. The dispatch-level correlation stands; the slot-level one does not.

**Superseded:** a paragraph here once named `0x413dc6700` as *the* writer and called it a
self-corrupting kernel. Flips also occur in submits where it writes nothing, so that identification
is dead — and the replacement is not "a second writer" either: **eight programs bind ranges covering
this address and the census cannot say which of them write.**

`0x413ce3400` is **back in scope.** It was marked superseded here on the grounds that it wrote only
"related state"; the containment census lists it among the eight. What remains true of it is
narrower: it is never declined on a routed run, so the "producer was refused" hypothesis is dead.

**What is NOT established**, stated precisely because the wording above is easy to over-read:

- **That `0x413dc3400` causes the cycles.** What is measured is a *dispatch-level correlation*: the
  malformed-pair count of its writes separates 54 clean reads (0..13) from 126 cyclic ones (19..106)
  with no overlap. The slot-level test **failed** — only 14% of cycle nodes sit on malformed slots
  against a 4.4% base rate — so a shared upstream cause is not excluded, and no A/B or
  clean-before/bad-after has been run.
- **That `0x413dc3400` writes this table at all.** The census re-run is complete and reports eight
  programs whose resources *contain* the address; a resource binding is not an access direction.
  Nothing here is a direction-qualified observation, so "its writes go bad" is shorthand for a
  correlation between its dispatches and the damage, not an established write.
- **Which program or store introduces a cycle**, and whether the guest algorithm is behaving
  correctly on inputs produced wrongly upstream.

**The next experiment is direction-qualified attribution**: establish, per program, whether it reads
or writes this range. Every further narrowing depends on it.

## RETRACTED 2026-08-14: "the corrupting write is `0x413dc6700`'s own" was one sample

The self-corrupting account was previously inferred from a writeback trace. It is now a direct
before/after on adjacent dispatches of the same program, in the same submit, on current master:

```text
disp 38  read  0x20f848417c  CLEAN   cycles=0  cyclic-roots=0     oob-roots=1385  max-depth=21
disp 38  execute ok, buffers=43, spirv=61143/177420afa4fd9c50
disp 38  writeback binding=4 addr=0x20f848a240 changed=2357
disp 39  read  0x20f848a240  CYCLIC  cycles=6  cyclic-roots=2062  oob-roots=0     max-depth=32
```

**RETRACTION.** The transition above is real and reproducible, and it does NOT establish cause. Two
observations from a wider census of the same instrumentation kill the inference:

- **A complete rewrite left the table CLEAN.** Submit 4312 dispatch 764 wrote 2,061 of 2,063 slots,
  and every read in that submit reports `cycles=0`. If this program's write corrupted the table, the
  most complete write in the route was the best chance to show it.
- **Tables flip to cyclic with NO write from this program.** Submit 4725 reads `0x20f848417c` with
  986 cyclic roots, and `0x413dc6700` writes nothing to it in that submit at all.

So a flip adjacent to its write is not evidence that the write caused it, because flips also happen
without one. What the original evidence established is that the program **writes the table it later
reads** — which was already known — plus one coincidence.

The generalisable error: a transition was observed next to a candidate cause, and adjacency was
treated as causation without checking whether the transition also occurs WITHOUT the cause. The
negative case is the whole test. Reproduce with `PROSPER_COMPUTELOG_CODE=0x413dc6700` plus
`PROSPER_COMPUTE_PARENT_WALK=0x413dc6700:0x5b:3:0x07FFFFFF:64` and read the interleaved
`parent-walk` / `execute` / `writeback` lines.

### EIGHT programs touch the traversal table — the "two writers" census was wrong

The first census matched `resource.gpu_addr == wanted` over top-level resources. Re-run with a
containment matcher (`wanted >= base && wanted - base < size`, over scalars *and* `table_entries`),
the same route reports **eight** programs, 3,931 hits:

```text
0x413ce3400   0x413ce6000   0x413cea300   0x413cee500
0x413d88400   0x413dc3400   0x413dc6700   0x413e1c300
```

`0x413dc6700` holds nine bindings including the loop's read at fetch pc 91; `0x413e1c300` is the most
frequent of all (four bindings, 308 hits each); `0x413cea300` appears with `fetch-pc=0xffffffff` —
that is the terminator-only program, whose whole body is one `s_endpgm`.

**Zero matches came from `table_entries`**, so on this allocation the array form is not exercised and
the entire gain is base-equality → containment: six programs bind a view whose *base differs* while
its range covers the address.

Consequences, and they are large:

- **"The table has two writers" is withdrawn.** It was an artifact of a matcher that could only see
  an exact base.
- **`0x413ce3400` is back in the picture.** This document marked it superseded on the grounds that it
  was "a writer of related state"; it binds a range containing this table.
- **The `0x413dc3400` correlation is unaffected but much less pointed.** Its malformed-pair count
  still separates 54 clean reads from 126 cyclic ones with no overlap — that measurement stands —
  but with seven other programs touching the allocation, "its writes are what go bad" is one
  candidate among several rather than a narrowing to one.
- **"Touches" is not "writes".** The census lists resources, not access direction. Distinguishing
  readers from writers on this allocation is the next thing to measure, and it is now the gating
  question rather than a detail.

### `0x413dc3400`'s write quality separates clean tables from cyclic ones — correlation, not cause

Tracing `0x413dc3400` — one of the eight touchers, and not established as a writer — with
`PROSPER_COMPUTELOG_CODE=0x413dc3400` plus `PROSPER_COMPUTELOG_CHANGED`
(its table is **binding 23** — binding indices are per-program, and reusing the consumer's numbers
here produced a confident wrong answer first), and scoring each write by how many heads it lands
without a matching tail in the next slot:

| reads | n | prior write's malformed pairs |
| --- | --- | --- |
| `cycles=0` | 54 | **0..13** |
| `cycles>0` | 126 | **19..106** |

**No overlap across 180 reads.** The writes are clean (`MISMATCHED=0`) through submit 7480 and then
jump to 77, 66, 67, … 104, 106, and the tables go cyclic exactly when they do.

That makes `0x413dc3400`'s write quality a **quantitative oracle** — not an identification of the
writer, since the census that produced this framing measured bindings rather than access direction
and eight programs touch the allocation. The malformed-pair count: a fix must drive it to zero, and `cyclic-roots` should follow.

**Stated as correlation, because that is what it is.** 180 reads with clean separation and a mechanism
that explains the shape (a head whose tail is absent leaves an orphan tail from an older generation,
and an orphan tail pointing back at its predecessor is exactly the observed 2-cycle) is strong, but no
A/B has yet shown the cycles following the mismatches. The counter-example that would break it is a
write with a high malformed count followed by a clean read; none occurred in 180 reads.

One earlier inference is already dead by this data: **13 malformed pairs at submit 4162 produced no
cycles at all**, so "any malformed pair corrupts the table" is false. There is a threshold between 13
and 19, which is itself a clue — the structure tolerates some inconsistency.

### The shader does not change across the boundary — only its input does

`0x413dc3400` compiles to the **same module** on both sides of the transition:

```text
submit 7480  groups=9x1x1  buffers=38  spirv=57537/8dbb56b7a4feea9c   MISMATCHED=0
submit 7898  groups=9x1x1  buffers=38  spirv=57537/8dbb56b7a4feea9c   MISMATCHED=77
```

Same SPIR-V hash, same launch, same 38 buffers. So the divergence is **data-dependent**, not a
recompilation difference — which rules out cache/key effects and points at how our execution handles
one particular input.

`0x09249249` is the guest's **empty-slot sentinel**, not a prosper fill (it appears nowhere in our
source). Decoded as a record it yields `next = 19,158,153`, far past the 2,063 records, so a walk
reaching it leaves the array and terminates by the RDNA2 out-of-range rule. Many of the slots the
first dirty write touches held it beforehand, i.e. they were previously empty.

**Falsified while forming it:** "clean tables terminate via those OOB sentinels, and cycles appear
once the table fills up." All 204 clean reads in this run have `oob-roots=0`; the cyclic ones range
0..1,690. The relationship runs the other way from the guess, so OOB termination is not what keeps a
clean table acyclic.

### Two patterns in the malformed set that do NOT generalise

Recorded so nobody re-derives them. Both looked convincing on one dispatch:

- **Stride 16.** The first twenty malformed indices at submit 7898 read `884,885 · 900,901 ·
  916,917 · 932,933` — adjacent pairs exactly 16 apart, which would point straight at a 16-lane
  grouping (DPP row, tile). Across the full set the `index mod 16` histogram is spread over ten
  residues, and it is a *different* spread on every dispatch. The regularity was in the first twenty
  entries, not in the population.
- **A distinguishing prior value.** At submit 7898 the malformed heads' previous contents include
  `0x24924924` twenty-six times while well-formed heads never show it — a clean discriminator, and
  `0x24924924` is `0x09249249 << 2`, the same repeating sentinel at another phase. It does not hold:
  submit 8309's malformed priors are `0` and `2`, submit 14727's are `0x3FFFFFFF` and `0`. No value
  is common to the malformed set across dispatches.

**The honest position after that:** the malformed set has no structural signature I have found, so the
correlation with cyclicity is currently the only handle on it, and pattern-hunting on derived metrics
has stopped paying.

### `0x413dc3400`'s table stores are LANE-PREDICATED

`0x413dc3400` disassembles to 882 dwords / 765 instructions, and its six table stores share one
shape — a store to the table and a store to a second buffer at the *same index*, both inside an
EXEC-predicated region:

```text
pc592  s_or_b64            s[106:107], s[16:17], s[14:15]
pc593  s_and_saveexec_b64  (EXEC &= that; old EXEC saved to s[106:107])
pc594  s_cbranch_execz     -> 599            ; skip the block if no lane qualifies
pc595  buffer_store_dword  v42, v58, s[8:11] ; a second buffer
pc597  buffer_store_dword  v48, v58, s[0:3]  ; THE TABLE, same index v58
pc599  s_mov_b64           exec, s[106:107]  ; restore
```

Six such blocks: (595,597), (606,608), (618,620), (630,632), (642,644), (654,656).

Two consequences worth having:

1. **Which slots get written is decided by a per-lane mask**, so a divergence in EXEC handling shows
   up as *missing or extra* stores rather than wrong values — which matches the observed defect
   (records that are individually well-formed, in the wrong combination) far better than an
   arithmetic error would.
2. The program is lowered through `emit_cfg_state_machine` (it is on the `role=terminal` list), where
   EXEC is emulated per invocation and `s_cbranch_execz` needs a cross-lane vote. That is the same
   machinery the consumer's loop exercised — and a hand-built kernel proved it correct for *that*
   shape (`tests/test_cfg_trip_bound.cpp`), which does not extend to `s_and_saveexec_b64` feeding a
   predicated store.

This is a hypothesis, not a result: no measurement yet shows a lane storing when it should not, or
failing to. The next instrument would compare the set of slots written against the set the mask
selects.

### The test was run, and it does NOT support cause

Same-run capture (120 dumped cyclic tables and 98,837 changed-slot records, so the join is valid),
asking whether the 2-cycles sit at slots the malformed writes touched:

```text
write-submit=10308  malformed=91  cycle-nodes=178  on-malformed-slot=26  (14%)
write-submit=10708  malformed=93  cycle-nodes=178  on-malformed-slot=26  (14%)
```

Stable at **14%** across ten tables. With 91 malformed slots in 2,063 the base rate is 4.4%, so this
is roughly 3x enrichment — a real association, and far from the "cycles land on malformed slots"
that would close the chain. **86% of the cycle nodes are somewhere else.**

So the malformed-pair count remains a strong *dispatch-level* correlate (54 clean reads at 0..13
versus 126 cyclic at 19..106, no overlap) while failing as a *slot-level* mechanism. Two readings
survive, and this data does not choose between them:

- the metric is a proxy for some other property of a bad write, and that property produces the
  cycles; or
- malformed pairs and cycles share an upstream cause and neither produces the other.

**What it rules out:** "each malformed pair becomes a 2-cycle." That was the mechanism I expected
and it is wrong.

### Open: why does it start at submit 7898?





`0x413dc3400` writes 2,061 slots with zero malformed pairs for nine consecutive dispatches, then
never again. Whatever changes at that point is the proximate cause, and it is a much smaller question
than the one this investigation started with.

### FALSIFIED: our store INDICES are wrong

Grouped strictly per `(submit, dispatch)` — the ungrouped form of this analysis is meaningless and
produced a confident wrong answer first — every head a dispatch writes has its matching tail at the
very next slot:

```text
submit 4312 dispatch 764: 2061 slots written, 1031 heads, MISMATCHED=0
submit 5555 dispatch  38: 1721 slots written,  864 heads, MISMATCHED=0
```

So the pair-store index arithmetic is correct in our execution. Whatever produces an orphaned tail,
it is not this program emitting a head and a tail at non-adjacent slots.

### The corruption has a SHAPE: overlapping pair writes, not garbage

`PROSPER_COMPUTE_PARENT_WALK_DUMP` captured 138 cyclic tables. Every cycle in every one of them is a
**2-cycle**, and the records involved are structurally valid — correct tag, plausible index:

```text
rec[452] = 0x00000e32   tag=2  bit30=0  next=454     <== cycle
rec[453] = 0x40000e32   tag=2  bit30=1  next=454
rec[454] = 0x40000e22   tag=2  bit30=1  next=452     <== cycle
rec[455] = 0x00000e7a   tag=2  bit30=0  next=463
```

**The table is a sequence of PAIRS**: `rec[k+1] == rec[k] | 0x40000000`, same payload, bit 30 marking
the second element. 920 such pairs across 2,063 records, and the pairing is not parity-locked (523
begin at an even index, 397 at an odd one), so a pair is simply two consecutive slots.

A `bit30=1` record must therefore be immediately preceded by its `bit30=0` twin. In every cycle it is
not: above, slot 453 holds the *tail* of pair (452,453) while slot 454 holds a tail whose head is
gone. **Two writers claimed slot 453** — one writing the tail of (452,453), one the head of
(453,454) — and the survivor's orphaned partner at 454 points back at 452, closing the cycle.

So this is an **overlapping-allocation / lost-update** signature, and the earlier "61 two-cycles" note
was reading it correctly. What changes is that the falsification recorded against it was measured on
`0x413ce3400`'s instruction footprint, which is only one of eight programs binding this allocation.
Re-opened: it falsifies a lost-atomic hypothesis for one program, and no program is established as
the writer.

Two candidates for how two concurrent STORES land on overlapping slots (a different sense of
"two writers" from the program census above — this is about lanes racing within a dispatch),
neither yet tested:

- **The pair store itself.** The program contains 3 `buffer_store_dwordx2` and 5 `buffer_store_dwordx3`
  among its 23 stores, and a `dwordx2` writes exactly two consecutive dwords — a pair. An addressing
  error in the multi-dword store path (element versus byte, or an off-by-one base) shifts a pair by
  one slot and produces precisely this.
- **The slot allocator.** `ds_add_rtn_u32` at guest pc121 is a bump allocator: each thread atomically
  adds its size to an LDS counter and takes the old value as its base. Overlapping bases would do it.
  **Checked and currently NOT suspect:** the emitter lowers it to `OpAtomicIAdd` with
  `Scope_Workgroup` / `MemSem_WGAcqRel`, which is correct for `local=256` (four 64-wide waves), and
  the program has no global-memory atomics at all — 41 MUBUF ops, every one a plain load or store.

### What the same run also settles

- **Most dispatches write nothing.** 52 of 64 writebacks are `changed=0`; the substantial ones are
  `changed=2357`, `2061`, and a few single-digit updates. So the corrupting event is rare and
  identifiable, not a steady drift.
- **Both tables can be cyclic**, contrary to the earlier per-address reading: `0x20f848417c` is
  162 clean / 30 cyclic and `0x20f848a240` is 40 clean / 120 cyclic across 352 resolved reads. The
  asymmetry is real but it is not a property of the address.
- **The resource table varies per dispatch.** The same program address compiles to different modules
  (`spirv=58649/…` several distinct hashes, `61143/177420afa4fd9c50` on the big-write dispatch), and
  `buffers` ranges from 1 to 43. A single dispatch's resource picture is not the program's.

### The open question, now narrow

Why does that write produce cycles? Candidates, none yet tested:

1. the stored VALUES are wrong (our lowering of the store path computes the wrong record);
2. the stored INDICES are wrong (right values, wrong slots);
3. a concurrency effect at `threads=2063 local=256 groups=9` that the guest's algorithm tolerates on
   hardware and our lowering does not preserve.

The oracle is in place either way: `cyclic-roots` on the read immediately after the write is the
number to move, and dispatch 38 of a `0x413dc6700` pair is where to look.

## Where the world went

As of 2026-08-14 the black world is **one compute program**, and that is established by A/B rather
than inferred:

```
PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700
```

| arm | device losses | frame |
| --- | --- | --- |
| default | 1, at ~59 s — reproduced in every run | black + HUD |
| `0x413dc6700` skipped | **0** across a full 300 s route | sun, anamorphic lens flare, radar with street geometry and blips, first-person tutorial text |

`0x413dc6700` hangs the GPU into a RADV **hard recovery** at `queue-submit`. prosper then disables
live compute for the whole process, and every later dispatch and indirect draw in the frame is
refused — so the frame cannot recover even partially.

**Read that skipped-program frame carefully: it is not "the frame minus that dispatch".** A skip goes
through the same decline path as a real refusal, so it clears `producer_epoch_ok` and the next
`ParserStall` latches `indirect_dependencies_ok` for the rest of the submit. The frame is that frame
minus the dispatch *minus every indirect draw and dispatch after the next parser stall*. The zero
device losses and the appearance of real scene content are unaffected by this; what it costs is the
right to attribute the *remaining* blackness in that image to `0x413dc6700` alone.

What the program probably is, from what survives without it: 43 resources,
`threads=2063x1x1 local=256x1x1 groups=9x1x1`, 61,143 SPIR-V dwords, and sky/flare/radar render while
world geometry does not — the shape of a tiled deferred lighting or light-culling pass.

**The frontier is therefore exactly one sentence: `0x413dc6700` must execute without hanging.**

### Why this took so long to see

Until #2538 the live compute backend had **fifteen** `return false` paths in `execute_item()` that were
silent, `trace`-gated, or logged without naming the dispatch. The executor records such a refusal as
`RealizationFailureReason::Unknown`, and **only when a capture trace is active** — so on a default run
a refused dispatch left no record at all. (`ComputeExecutionDeclined`, which an earlier revision of
this file named, is a classification added later by #2536; it did not exist when the frame went
black.) A gameplay submit's 196 realization failures were 59 anonymous declines and 128 cascade
failures, and the cascade hid the cause:

1. a declined dispatch clears `producer_epoch_ok`
2. the next `ParserStall` latches `indirect_dependencies_ok` false for the **rest of the submit**
   (`gpu_executor.cpp:9609`)
3. every later indirect draw and indirect dispatch short-circuits to `IndirectDependencies` without
   being attempted (`:9354`, `:9422`)

GTA V's world is GPU-driven, so the indirect operations *are* the world. One refusal anywhere early in
a submit removes all of it. Several sessions went to the 72 **direct** draws that execute correctly at
3840x2160 with colour writes enabled — they were never the ones failing.

The first domino turned out to be an **empty guest kernel**: `0x413cea300`, whose entire body is one
`s_endpgm`, declared one raw buffer and was refused as `no-bindable-descriptor` 192+ times per run.
Fixed by proving emptiness from the raw stream (`rdna2_program_is_terminator_only`) and reporting such
a dispatch as a successful no-op. A/B: 192 declines to 0.

### The trip bound measures ONE emitter — a null from it is "not measured"

`PROSPER_CFG_TRIP_BOUND` caps the **CFG dispatcher's** back edge. It does not touch either structured
loop emitter, so a program the structurizer accepts is *structurally unmeasurable* by it and reports
nothing whether or not it runs away. `0x413dc6700` happens to be covered — it reaches `role=terminal`
in the structurizer and is lowered by `emit_cfg_state_machine` — which is why its witness fired at all.

This is recorded because the emitter's own comment asserted the opposite for a while: a correction
saying the structured emitters are NOT covered was added directly above a surviving sentence claiming
all three were. Both are now pinned by assertions — a structured loop must be byte-identical when
armed, a dispatcher loop must not be.

### Diagnostics this investigation added, and what each is safe to conclude from

| switch | default | what it does | the trap it avoids |
| --- | --- | --- | --- |
| `PROSPER_CFG_TRIP_BOUND` + `_PROGRAM` + **required** `_PHASE` | off | caps the CFG dispatcher's back edge for one program and phase, and records a device-side hit witness | covers **only** the CFG dispatcher, so a null on a structurizer-accepted program means *not measured*; without `_PHASE` nothing is emitted, because one record cannot describe two phases |
| `PROSPER_COMPUTELOG_RAW` | off | writes a traced program's guest RDNA2 bytes for `tools/shader_inspect` | `PROSPER_SHADER_DUMP_SUCCESS` names files by hash, so recovering one program by address means hash-matching by hand |
| `PROSPER_INDIRECT_APERTURE_RECOVERY` | **off** | rebuilds a base-less queue-2 indirect argument address from the last-seen SetBase aperture | changes execution: the aperture is learned from any SetBase on any queue, and *mapped* is not *this is the argument buffer* |
| `PROSPER_INDIRECTLOG` | off | per-packet base/offset/queue, the three argument dwords, and an end-of-run outcome census | readability was probed and values were not, so a misread surfaced only as a `workgroup-count-limit` decline thousands of operations later |

The aperture recovery is opt-in on purpose and the trade is recorded rather than implied: **off, 50 of
64 indirect compute dispatches skip as "unreadable arguments"; on, 0 do**, and the probe found the raw
low address unmapped and `aperture | low` mapped on 49 of 49. That is real evidence about where those
arguments live and it is still not provenance — one process VA space can hold mapped allocations under
several high-32 prefixes, so accepting an address because 12 bytes are readable admits dispatching
group counts read out of an unrelated live allocation.

## Ruled out

One line per falsified hypothesis, the evidence that killed it, and where. **Read this before forming
a new one** — and note which entries are *solid* versus *void*, because a void result is not a
falsification.

- **The 72 direct draws are failing / are culled / have colour writes masked.** They execute at full
  3840x2160 with `effective=3f` and zero realization failures in the window that presents black. The
  world is drawn by *indirect* operations, which the latch above dropped untried. #2481.
- **Collapsed AABBs culling the world.** The reduction at `0x413ced900` computes a correct bounding
  box, `(-13.71, -23.65, 0)` to `(14.98, 16.95, 4.101)`. #2481.
- **`CB_TARGET_MASK=0` masking colour writes (the #1946 shape).** The main-view pass reports
  `effective=3f`. The pass that showed `target-mask=0 effective=00` was a **512x512 depth-only shadow
  atlas**, where that is correct — selected by a `MIN_DRAWS` filter, not by phase. Instrument trap 159.
- **The sky renders correctly offline and is lost live.** The `--draw-steps` composite that showed a
  blue gradient with a horizon band is prosper's **seed-miss placeholder**: already 100% non-black at
  step 4, before most draws run, and unchanged through step 78. Instrument trap 161.
- **The whole-frame abort on an unsupported storage image.** `render_draw_pass_rgba`
  (`tests/render_runner.h:3309`) does abandon an entire 4K frame on the first invalid storage-image
  contract, which is a real disproportionality — but gating it so only the offending draw is dropped
  left the black window unchanged. Reverted rather than landed. #2481.
- **A dark frame containing a dim scene.** Frames measuring 0.00% non-black at threshold 8/255 while
  12.7-13.5% of pixels are non-zero at luminance 1-2, with a CRC that changes every second, are
  **animated dither** — amplified 64x there is no structure, edge or geometry. Instrument trap 160.
- **The recompiler is the frontier.** `[compute] skip unsupported program` named 15 programs; retried
  offline against their own captured resource tables, **16 of 20 recompile cleanly**. Only 9 of 196
  failures in the classified capsule are `shader-recompile`. Instrument trap 157.
- **Unresolved descriptors on the hanging program.** All **43** of `0x413dc6700`'s resources resolve
  with real addresses and sizes; none is null. (The zero-address resources in an earlier dump belong to
  `0x413dc3400`, the neighbouring dispatch.) #2481.
- **The hang is an out-of-bounds write from a mis-sized buffer bound.** `RADV_DEBUG=hang` produced a
  report whose **`vm_fault.log` is 0 bytes** — no page fault, no VM fault. This was the leading
  suspect given #2529/#2535's history with `scalar_buffer_dword_count`. #2481.
- **The hang is a non-uniform `OpControlBarrier` deadlock.** The CFG dispatcher's continue block runs
  two workgroup-scope barriers per iteration and its LDS reduction covers only `[base..base+63]`,
  which looks exactly like a per-wave exit under a workgroup-wide barrier at `local=256`/`wave=64`.
  It is not: the loop-exit predicate reads **slot 260**, the whole-workgroup liveness result, so every
  invocation iterates together. #2481.
- **The hang is a non-uniform early exit.** The 14,370-line disassembled module contains exactly
  **one** `OpReturn`, at the end. No `OpKill`, none inside any of the three dispatcher loops. #2481.
- **The hang is a cyclic traversal table (183 cyclic starts).** **RETRACTED — see the retraction
  section above.** The 183 cycles are in `0x20f848417c`, the table read by the dispatch that
  *completed*; the hanging dispatch reads `0x20f848a240`, which is acyclic with a longest chain of 11
  steps. The measurement was taken from the wrong dispatch's binding. #2542.
- **A lost atomic corrupts the traversal table.** **VOID against the current writer — the evidence
  below is about `0x413ce3400`, and the table's writer is `0x413dc6700` itself.** The corruption
  signature (61 two-cycles) still stands as an observation, and so does the instrument note at the end
  of this entry, which is why the entry is kept rather than deleted. What does not stand is the
  falsification: showing that `0x413ce3400` performs no atomic cannot rule out a lost-atomic
  write-path defect in a *different* program. To settle it, re-run the same footprint analysis on
  `0x413dc6700`'s own stores. Everything from here to the end of this bullet is that superseded
  argument. The table that hangs `0x413dc6700` is a linked list
  whose corruption is 61 **two-cycles** (`i` and `i+2` pointing at each other), the classic signature
  of a non-atomic concurrent insertion — two threads each linking to the other because both read the
  head before either wrote. Its producer `0x413ce3400` performs **no atomic operation of any kind**,
  and the strong form of that is structural rather than grep-shaped: its entire memory footprint is
  **29 buffer loads and stores** (`buffer_load_dword` x15, `buffer_load_dwordx2` x9,
  `buffer_store_dword` x4, `buffer_store_dwordx3` x1) and **zero `ds_*` instructions**, so there is no
  LDS family in which an atomic could hide. Zero `OpAtomic*` in its 3,881-line recompiled SPIR-V
  agrees. **Positive control, a different program in a different capsule**: `0x413ced900` contains
  `buffer_atomic_fmax` x3, `buffer_atomic_fmin` x3, `ds_max_f32` x3 and `ds_min_f32` x3 — so the
  instrument fires on the buffer family *and* the LDS family, and the zero is a real negative.
  **Note what the control also demonstrates:** a `(buffer|global|flat|ds)_atomic_*` pattern silently
  misses LDS atomics entirely, because RDNA2 spells them `ds_max_f32`, not `ds_atomic_max` — the
  control's own `ds_min/max_f32` would not have been found by it. That gap is why the claim above
  rests on the absence of the whole DS family rather than on an atomic-shaped pattern. #2542.
- **The producer/consumer ordering violation is `WAIT_REG_MEM` being barreled through.** The guest
  issues waits prosper cannot satisfy (`[agc] WaitRegMem … dependency violated … LABEL-UNMAPPED`, 40
  per route), and by default an unsatisfied wait does not pause the queue — which would let a
  consumer run before its producer's results land, exactly the symptom. **Falsified** with the
  opt-in barrier model `PROSPER_WAIT_DEFER=1` (#312): the device is still lost, at the same program
  and the same dispatch index. **Lever verified before reading the result** (instrument trap 164):
  the run logs 40 `pausing queue (deferred effects)` and the baseline logs 0, against 28 and 40
  `dependency violated` respectively, so the model was genuinely active. Note the recorded #312
  verdict against defaulting this ON was measured entirely on *Dragon Quest VII*'s heap corruption
  and says nothing about GTA V — this is an independent falsification, not a re-derivation of it.
  #2542.
- **The hang is an unconditionally infinite loop.** The same program executes successfully at dispatch
  38 and hangs at dispatch 39 of the same submit. Whatever spins is data-dependent. #2481.
- **Our `v_cmpx` / `s_cbranch_execz` lowering cannot exit the guest's pointer-chasing loop.**
  `0x413dc6700`'s whole 903-dword body contains exactly **one** backward branch, at guest pc97 back to
  pc88, and that loop's only exit is `s_cbranch_execz` after a `v_cmpx_ne_u32`. Nothing else in the
  loop writes EXEC, so if `v_cmpx` narrowed VCC instead of EXEC — plausible, because the e32
  encoding's destination field still reads as VCC and `shader_inspect` prints it as `special:106` —
  the loop could never end. **Falsified by a hand-built kernel of the identical shape**
  (`tests/test_cfg_trip_bound.cpp`): built by hand rather than derived from the capture or the
  recompiler, its body decrements the index instead of chasing a buffer, so only the control flow is
  on trial. On real Vulkan, lane *i* walks exactly *i* steps for all 128 lanes — per-lane EXEC
  narrowing and the cross-lane `execz` vote are both correct. #2542.
- **Bounding the CFG dispatcher's trip count stops the hang.** Tried at 4096 and at 2^20; the device
  was lost both times, so a bound at those values does not rescue the frame. This is **not** in
  tension with the hit witness firing at 4,096 further down: the witness says the loop *reaches* the
  cap, and the cap then truncates that dispatch's control flow, which produces wrong results for
  every later consumer rather than a working frame. "The bound does not fix the title" and "the loop
  runs away" are both true. #2481.

### Void, not falsified — do not cite these as settled

- **Portable wave64 emulation cost is the hang's mechanism.** A run with
  `PROSPER_NATIVE_COMPUTE_MULTIWAVE=1` still hung, which was reported as a falsification and **is
  not one**: the emitted module is **byte-identical** with and without the switch
  (`d04fd09b13408f9b4da7287fae34f692` in both arms and in a capsule from hours earlier), so the two
  runs are the same run. `[subgroup] … native=64 … multiwave=1` reports the resolved *config*, not the
  emitted lowering. The hypothesis is neither confirmed nor refuted — though it is now less likely on
  other grounds, since `native_subgroup_size` was apparently already 64, which suggests those barriers
  are the guest's own `s_barrier`s rather than emulation scaffolding. Reopening it needs a lever
  verified by module hash **before** its result is read. #2481.

## CORRECTED AND CONVERGED: `0x413dc6700` dispatch 39 takes ~2 SECONDS, and the loss follows it

Measuring the DURATION of every compute fence wait, and reporting any wait over 100 ms even when it
succeeds, produced exactly one line in a whole route:

```
[compute] SLOW fence wait 2045.2 ms result=0 program=0x413dc6700 submit=8116 dispatch=39 order=14036 groups=9x1x1
[compute] fatal Vulkan device loss stage=queue-submit … program=0x413dc6700 submit=8116 dispatch=40 order=14041
```

**One abnormal wait in the entire run — 2,045 ms against sub-millisecond for everything else — on the
dispatch every other instrument has named, immediately followed by the device loss on the next
dispatch.**

Three independent instruments now converge on `0x413dc6700` dispatch 39: the trip-bound hit witness
(`trips=4096`, with no invocation ever reaching an ordinal past the loop body), the fence-wait
duration, and the loss ordering. The witness's fields are dispatcher
quantities; resolve an ordinal against the `dispatch map:` line the same phase prints.

### This corrects the section that used to be here

That section argued "no compute dispatch ever hangs", from **0 fence-wait timeouts across 705 waits**.
The count was right and the inference was wrong: the timeout is **30 seconds** and the event is **2
seconds**, so a dispatch can be three orders of magnitude slower than every other one and still never
trip it. A zero timeout count measures only "nothing exceeded 30 s" — it says nothing about whether
anything is pathologically slow, which is the actual question.

The alternative that motivated the check — that a context reset SIGNALS pending fences, so a killed
job returns `VK_SUCCESS` and looks instant — remains untested and is no longer needed to explain
anything.

What survives from that section: the loss is still reported at `queue-submit` on the *next* dispatch,
so the loss line alone never named the culprit. Instrument trap 170 stands as written about
attribution; only the "not compute's" conclusion drawn from it was wrong.

## Superseded: the earlier reframing

###

Counted across every routed run in this investigation:

| observation | count |
| --- | --- |
| compute dispatches that entered their fence wait | **705** |
| compute dispatches that completed it | **705** |
| `[compute-decline] reason=queue-wait` (a 30-second fence timeout) | **0** |
| `[compute-decline] reason=queue-submit` (device ALREADY lost) | **28** |

`execute_item` submits and then waits on a fence with a **30-second** timeout, and reports a timeout as
`queue-wait`. That has never once fired. Every compute dispatch this title issues completes.

The 28 losses are all `stage=queue-submit`, which means `vkQueueSubmit` returned
`VK_ERROR_DEVICE_LOST` — the device was **already** dead when compute next submitted. The dispatch
named in that message is the first one *after* the reset, not the one that caused it.

**So the GPU hang is not caused by a compute dispatch.** It is almost certainly in the graphics
submission path, and the compute backend is a victim that discovers the dead device and then disables
itself process-wide — which is what drops every later indirect draw.

### What this overturns

- **"`0x413dc6700` hangs the GPU."** Its dispatcher loop genuinely runs past 4,096 iterations — the hit
  witness recorded that directly, at `trips=4096`, on dispatch 39 — but it still *finishes* inside the
  30-second fence. A long loop, not a hang.
- **"`0x413e14900` is a second hanging program."** It is a 753-dword module with **zero loops** running
  `threads=42x1x1 groups=1x1x1` with `result=ok`. It was never a candidate.
- **The trip-bound A/B matrix.** Each cell was a single run of a failure that varies run to run — the
  same build died at `0x413dc6700` dispatch 39 in one run and dispatch 40 in another, and reached
  `0x413e14900` in a third. Single-run cells cannot support the causal reading I gave them.

### What survives

- The cyclic traversal tables are real and measured at the correct timing. They make that loop very
  long. They do not, on this evidence, hang the GPU.
- The queue-2 indirect-argument recovery is real and deterministic: 50 of 64 skipped dispatches now
  execute.
- The consequence chain from the loss onward — compute disabled process-wide, `producer_epoch_ok`
  cleared, indirect latch, no world — is unchanged and still explains the black frame.

**The next investigation is the graphics submit path**, not the recompiler and not compute resources.

## FIXED: indirect compute dispatches on queue 2 had no argument base — 50 of 64 were skipped

**What the fix reliably changes:**

| | before | after |
| --- | --- | --- |
| `indirect dispatch skipped: unreadable arguments` | **50 of 64** | **0** |

That is deterministic and verifiable in every run: the skips are gone and those dispatches execute.

**What it does NOT reliably change, corrected after a second run.** The first run with the fix got past
`0x413dc6700` and died later at `0x413e14900` dispatch 52, and its frame showed sun, lens flare and
radar. I wrote that up as a before/after improvement. **A second run with the same build died at
`0x413dc6700` dispatch 40 again.** The hang is data-dependent, so a single run either side proves
nothing about it, and the frame content had already appeared in earlier *skip* runs — so neither the
survival nor the frame is attributable to this fix.

The honest statement is: the fix removes a real and deterministic class of dropped work; whether that
changes the hang is unmeasured, and would need repeated runs on both sides.

### The defect

`SET_BASE_INDIRECT_ARGS` sets one `indirect_compute_base`, and that is **per-fold** state. A PS5
process has one GPU virtual address space, but GTA V's async-compute queue carries `DispatchIndirect`
packets whose 32-bit payload is a full address *within the already-selected aperture* and no SetBase
of its own — so its base is zero and its arguments resolve to an unmapped low address.

Probed on **49 of 49** such dispatches before changing anything:

```
readable? low=0  aperture20=1  hi-dword=0
          (low=0xf8480120  ap20=0x20f8480120  hi64=0x21f8480120)
```

The raw low address is unmapped; `aperture | low` is mapped; folding the modifier as an ADDR_HI is
not.

**That is evidence about where the arguments live, and it is not provenance — which is why the
recovery is opt-in.** This paragraph used to end "so a wrong aperture leaves behaviour exactly as it
was", and that claim does not hold: the aperture is learned from *any* SetBase on *any* queue, and one
process VA space can contain mapped allocations under several high-32 prefixes. A wrong aperture that
happens to land on a mapped allocation is accepted, and the dispatch then reads group counts out of
whatever lives there. Gated behind `PROSPER_INDIRECT_APERTURE_RECOVERY`, default off.

### What it does NOT fix

The world is still black, and the traversal tables are still cyclic in a large minority of reads
(944 cyclic vs 1,179 clean, against 806/1,782 before — proportionally better, not resolved). So the
skipped dispatches were **a** cause of lost work but not the whole cause of the cyclic structure. The
next device loss is `0x413e14900`, which a dispatcher bound does not save either.

## Superseded: the root-cause candidate as first written

`SET_BASE_INDIRECT_ARGS` sets one shared `indirect_compute_base` in the command processor. Logging
base, offset and queue separately at the `DispatchIndirect` site:

| queue | dispatches | base | outcome |
| --- | --- | --- | --- |
| **1** | 14 | `0x205b690f80` | resolve and run |
| **2** | **50** | **`0x0`** | `args = raw offset` -> unreadable -> **skipped** |

The queue-2 offsets are `0xf8480120`, `0xf8480160`, `0xf84801a0`, `0xf84801e0`, `0xf8480220` … stepping
by 0x40 — and the guest arena those dispatches belong to is at **`0x20f8480000`** (347,040 bytes; it is
the buffer the `s_endpgm` kernel declares). So the intended address is almost certainly
`0x20f8480120`, and what is missing is a `0x20_00000000` base that queue 2 never receives.

**50 of 64 indirect compute dispatches in one route are therefore skipped entirely**, with only
`[agc] indirect dispatch skipped: unreadable arguments at 0x…` to show for it — a message that prints
the SUM, so it cannot distinguish a bad offset from a base that was never set.

### Why this is very likely THE defect

It closes the chain that every other measurement in this document constrains:

1. `SET_BASE_INDIRECT_ARGS` for compute is seen only on queue 1; queue 2's folds start with base 0
2. queue-2 indirect dispatches resolve to an unmapped address and are skipped
3. the skipped dispatches are the maintenance passes for the traversal structure
4. the parent array degrades — **once, irreversibly** (each buffer transitions clean->cyclic exactly
   once and never recovers, which is what a half-applied union-find update looks like)
5. `0x413dc6700` walks the cyclic chain and never terminates — recorded directly by the trip-bound
   hit witness (its third field is a dispatch ordinal — see the correction below)
6. GPU watchdog -> RADV hard recovery -> live compute disabled process-wide
7. `producer_epoch_ok` cleared -> `indirect_dependencies_ok` latched -> every remaining indirect draw
   dropped untried -> **no world**

**Not yet proven**: that supplying the missing base makes the tables stay acyclic. That is the next
experiment, and the cycle census is already the oracle for it — the number to move is
`cyclic-roots`, per dispatch, at pre-dispatch timing.

## THE TABLE IS CYCLIC AT DISPATCH TIME — measured with the right instrument at the right moment

`PROSPER_COMPUTE_PARENT_WALK` reads the selected resource **immediately before `compute({item})`** —
the timing the capsule's post-submit snapshots never had — and models this exact loop
(`while (index != 0) index = (records[index] >> shift) & mask`), reporting cycles directly.

```
PROSPER_COMPUTE_PARENT_WALK=0x413dc6700:0x5b:3:0x07FFFFFF:64
```

**1,782 resolved reads across one route:**

| reads | cycles | cyclic roots (of 2,063) | oob roots | max depth |
| --- | --- | --- | --- | --- |
| 16 | 0 | 0 | 0 | 12..15 |
| 960 | 0 | 0 | **1,894** | 19 |
| **795** | **15** | **1,805** | 0 | 22 |
| 5 | 17 | **2,062** | 0 | 20 |
| 6 | 1 | 1,039 | 269 | — |

**806 of 1,782 dispatches receive a table in which 1,805-2,062 of 2,063 roots lead into a cycle.** The
guest loop cannot terminate on that data, so the dispatcher spinning past 4,096 iterations is the
*correct* behaviour for what it was handed. Combined with the hit witness, the chain is closed: bad
data in, non-terminating walk, GPU hang.

The 960 reads with `oob-roots=1894` terminate by the RDNA2 rule that an out-of-range `idxen` load
returns zero — those dispatches complete, which is why only some dispatches hang.

**Read the whole distribution, not the first rows.** The first three log lines of this run show
`cycles=0 cyclic-roots=0`, and stopping there gives exactly the opposite conclusion — "the data is
fine, so the emulation is broken". The clean reads are a real minority of the population, and they
come first.

### And it is ONE buffer, not a random failure

Correlating every resolved read by the table address it walked:

| table address | outcome | reads |
| --- | --- | --- |
| `0x20f848417c` | clean | **966** |
| `0x20f848417c` | cyclic | 6 |
| `0x20f848a240` | **cyclic** | **800** |
| `0x20f848a240` | clean | 10 |

`0x20f848a240` is cyclic in **800 of 810** reads (98.8%); `0x20f848417c` is clean in **966 of 972**
(99.4%). The program alternates between two traversal tables, and **one of them is systematically
corrupt while the other is systematically correct**. That is not a race or a timing artifact — it is
a property of one buffer.

The predecessor correlation carries no signal: both outcomes follow the same preceding dispatch
(`previous-code=0x413dc6700`, realized, mostly not executed) in the same proportion.

Note this is the same address whose *post-submit* snapshot I measured as acyclic, and used to retract
the cyclic-table hypothesis. At dispatch time it is cyclic. The original hypothesis was right in
substance; the capsule's snapshot timing is what made it look wrong.

### The writer is the CONSUMER ITSELF — this kernel corrupts its own structure

`0x413dc6700` both walks the table and writes it. Its own write-back lines say so:

```
[compute] execute submit=4310 dispatch=776 code=0x413dc6700 threads=2063x1x1 local=256x1x1 buffers=43
[compute]   writeback binding=4 addr=0x20f848a240 size=8252 changed=2062 hash=b24dd1c1…->514428f8…
```

It reads through one binding (`fetch=0x5b`, the loop load) and writes through another
(`binding 4`/`binding 5`, `fetch=0x35`/`0x41`), and the two tables swap roles between dispatches — a
double-buffered structure. Combined with the loop shape (chase a parent index to a root, then take the
**parity of the depth**, `v_and_b32 v1, 1, v2` compared against `s24`), this is the shape of a
**union-find with path compression**: walk to the root, then write compressed parents back.

That reframes the defect entirely. There is no upstream producer to blame — **the corruption is
produced by the same program that later chokes on it.** Dispatch N writes a malformed parent array;
a later dispatch reads it, finds a cycle, and never terminates.

It is consistent with the address split: `0x20f848a240` is the buffer this kernel *writes*, and it is
the one that is cyclic on later reads (800 of 810); `0x20f848417c` is clean 966 of 972.

**Correction to an attribution I nearly published.** I first read the trace as naming `0x413cf9a00`
the writer — a program that is genuinely skipped (`[compute] skip unsupported program 0x413cf9a00`) and
which sits on the recompiler-reject list, so it made a very attractive culprit. It was a parsing
error: my attribution regex latched onto the nearest `program 0x…` text, which included the skip line
itself. The real anchor is the `[compute] execute … code=…` line, and it says `0x413dc6700`.

So the next question is not "who failed to write this" but **"why does this kernel's own output
contain cycles"** — a store-path or algorithm-emulation question about `0x413dc6700`, not a missing
producer.

### What this settles, and what it reopens

Settled: the defect is **upstream data**, not the dispatcher's lowering of this loop. The recompiler
faithfully executes a walk that genuinely does not terminate.

Reopened: **why is the table cyclic?** The earlier producer investigation was measured on
post-submit snapshots and on the wrong dispatch's buffer, so it has to be redone against
`PROSPER_COMPUTE_PARENT_WALK` timings.

**The sentence that used to close this paragraph — "`0x413ce3400` remains the only writer identified
so far" — contradicted the section it sits in and is withdrawn.** The writeback evidence directly
above names `0x413dc6700` as the writer of the table it later chokes on. Both readings were live in
this document at once, which is worse than either being wrong: they imply different next experiments
(find the producer vs. audit this kernel's own stores), and the reconciliation is at the top of the
file.

## CONFIRMED BY HIT RECORD: phase 0's dispatcher loop runs past 4,096 iterations on the hanging dispatch

`PROSPER_CFG_TRIP_BOUND=N` (diagnostic) forces a dispatcher loop out after N iterations. With
`N=100000` the run gets **past** `0x413dc6700` and dies much later at a **different** program:

> **The switch has since changed and these runs predate it.** It now bounds only the program and
> phase named by `PROSPER_CFG_TRIP_BOUND_PROGRAM` / `PROSPER_CFG_TRIP_BOUND_PHASE`, and **the phase
> selector is required** — armed without one it emits nothing and says so. The table below was taken
> when the bound applied to every dispatcher phase of the selected program, so reproduce it by naming
> a phase explicitly. The maps every phase prints when the bound is armed list what is available.

| run | device lost at |
| --- | --- |
| default | `0x413dc6700` submit 4535 **dispatch 39** order 5642 |
| + empty-kernel fix | `0x413dc6700` submit 5968 **dispatch 39** order 17056 |
| + no buffer cache | `0x413dc6700` submit 5963 **dispatch 39** order 9755 |
| + `PROSPER_WAIT_DEFER=1` | `0x413dc6700` submit 4643 **dispatch 39** order 1206 |
| **+ `PROSPER_CFG_TRIP_BOUND=100000`** | **`0x413e14900`** submit 5954 **dispatch 52** order **24374** |

**Isolated control — one program instrumented, only the constant differs.** `PROSPER_CFG_TRIP_BOUND`
is targetable with `PROSPER_CFG_TRIP_BOUND_PROGRAM=0xADDR`, which leaves every other recompiled module
byte-identical. Both arms below instrument `0x413dc6700` and nothing else — the run log carries a
single arm line naming that one program:

| bound | target | outcome |
| --- | --- | --- |
| **4,096** | only `0x413dc6700` | **gets past it**, dies later at `0x413e14900` |
| **4,000,000,000** | only `0x413dc6700` | **dies at `0x413dc6700`** |

Same instrumentation, same module perturbation, different constant. That removes the two alternatives
an untargeted bound leaves open: it cannot be an *earlier* truncated shader feeding different data
downstream (no other shader is touched), and it cannot be SPIR-V perturbation changing RADV code
generation (both arms carry the identical counter). This is targeted A/B evidence that phase 0's loop is implicated. It is **not** proof that the
cap fired — see the witness section below.

**Which phase spins — bisected.** `PROSPER_CFG_TRIP_BOUND_PHASE=K` bounds only the K-th dispatcher of
the selected program. Three runs, everything else byte-identical:

| phase bounded | guest pc range | outcome |
| --- | --- | --- |
| **0** | **0..116** | **gets past `0x413dc6700`**, dies later at `0x413e14900` |
| 1 | 117..129 | dies at `0x413dc6700` dispatch 39 |
| 2 | 130..902 | dies at `0x413dc6700` dispatch 39 |

**Only bounding phase 0 saves the device** (targeted A/B evidence, not proof of a cap hit). Phase 0 covers guest pc 0..116, and the program's only
loop is at **pc 88..97** — inside it. That is the pointer chase:

```
 88  v_mov_b32_e32     v2, s22
 89  v_cmpx_ne_u32_e32 0, v1        ; EXEC &= (v1 != 0), never restored in the loop
 90  s_cbranch_execz   7            ; exit when EXEC == 0
 91  buffer_load_dword v1, v1, s[0:3], 0 idxen
 95  v_bfe_u32         v1, v1, 3, 27
 97  s_branch          -10
```

So the runaway dispatcher is the one wrapping the EXEC-narrowing walk. The defect is named to a
program, a phase, and a guest **pc** range — pc is a dword offset and RDNA2 instructions are variable
length, so an instruction count cannot be read off it; this line previously called the range
"117-instruction", which it never was.

(The "maximum chain of 11 steps" quoted here came from the capsule-timing measurement that the
retraction above supersedes. The pre-dispatch census is the current figure: 806 of 1,782 reads receive
a table in which 1,805-2,062 of 2,063 roots lead into a cycle.)

**The spinning dispatcher is the PORTABLE one, and that also explains the earlier void result.** The
emitted module for `0x413dc6700` contains **zero `OpGroupNonUniform*` instructions** — so
`b.native_subgroup_size` was 0 and the branch votes take the portable path: publish into
`vote_pending_var` / `vote_value_var` in the switch case, then reduce through LDS scratch behind two
workgroup barriers in the continue block.

That is forced, not incidental. `rdna2_to_spirv.cpp` sets `b.native_subgroup_size = 0` when
`partial_barrier_phases || exact_partial_dispatcher`, and this dispatch is both barrier-phased and
partial (`threads=2063`, `local=256` — the final workgroup carries 15 real threads of 256).

**This retires the `PROSPER_NATIVE_COMPUTE_MULTIWAVE` mystery recorded above as "void, not
falsified".** That switch only moves `config.native_subgroup_size`, which this line then overrides to
0 for exactly this shape — so the module *could not* change, and the byte-identical hashes were the
correct outcome rather than a broken lever. The result stays void as evidence about emulation cost,
but the reason is now known.

It also narrows where the defect can live: the vote machinery under suspicion is the portable
LDS-reduction path for a barrier-phased dispatcher with a partial final workgroup, not the native
subgroup path. Two things checked there and found correct: the vote mailboxes (`vote_pending_var`,
`vote_value_var`, …) are reset at the top of every dispatcher iteration, so a stale vote cannot carry
over; and each lane's LDS contribution is gated on its own `pending` bit.

**The device-side hit witness WORKS, and the cap fires — on the dispatch that hangs.** The shader
writes into the top of the internal GDS buffer when a cap runs out; the host prepares those dwords
before the selected dispatch, reports them after it, and then RESTORES the guest's original values,
so the shared GDS buffer is byte-identical afterwards for whatever dispatch uses it next. It touches
them only when a witness was actually **emitted** for that program — not merely when the selectors
accept it, since a structured loop or a phase ordinal the program lacks satisfies every selector and
emits nothing. **The current record is five dwords — hit flag, phase, highest trip count, and
the lowest and highest dispatcher switch-case ordinal visited — with the last three reduced across
invocations by device-scope atomics.** The per-invocation "last block index" this line used to name
no longer exists: it was one sample, it could not answer whether the dispatcher was cycling, and its
label was published wrongly twice (instrument trap 172).

**Positive control first**: at bound **2** — which a 15-block dispatcher phase cannot satisfy — the
witness produces **1,606 hit records**. It fires.

At bound **4,096**, on the same program and phase:

```
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=9  trips=4096 submit=5547 dispatch=38
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=14 trips=4096 submit=5547 dispatch=39
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=9  trips=4096 submit=5547 dispatch=40
    ^ HISTORICAL TRANSCRIPT. `last-block` is the label this run printed; the field is a dispatcher
      switch-case ordinal and the per-invocation form was removed rather than renamed again. The
      current line reports `trips` and `dispatch-range` only. See the correction directly below.
```

**Correction (2026-08-14): the third field is a dispatcher switch-case ORDINAL — `next-dispatch`.**
It was first printed as `last-block`, then briefly "corrected" to `next-guest-pc`; both were wrong,
in opposite directions, and the second was published on this branch before being caught. Every write
to the emitter's `pc_var` stores `dispatch_for_block[...]`, so despite the variable's name it is
neither a basic-block index nor a program counter. Instrument trap 172.

**What this does and does not settle.** The numbers 9, 14, 9 are ordinals into phase 0's dispatch
table. Whether ordinal 9 covers the guest's loop body at pc88..97 is a question about that table, not
something derivable by hand from the value — and the hand-mapping is what produced both errors. Each
phase now prints its map when a bound arms:

```text
[cfg-trip-bound] program 0x413dc6700 phase 0 dispatch map: 0:pc0..<N 1:pc... ...
```

so the ordinal resolves against emitted evidence rather than by hand.

**Re-measured on a routed run, 2026-08-14, with the ATOMIC record — and the number changed.**

```text
[cfg-trip-bound] program 0x413dc6700 phase 0 dispatch map:
    0:pc0..<41  1:pc41..<50  2:pc50..<55  3:pc55..<61  4:pc61..<67  5:pc67..<73
    6:pc73..<76  7:pc76..<88  8:pc88..<91  9:pc91..<98  10:pc98..<103 ...
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 trips=4096 dispatch-range=0..9   (5 hits, 1 loss)
```

**The earlier `dispatch-range=6..9` was an artifact of last-writer publication and is withdrawn.**
That record was written with plain stores from every invocation reaching the cap, so it reported one
invocation's local extremes; the reduction over all invocations is `0..9`. The claim built on it —
"the state machine visited only ordinals 6..9, so it is cycling in the loop" — does not survive,
because ordinal 0 is the program entry and every invocation passes through it. A ten-ordinal span is
not evidence of a cycle.

**What the corrected record does establish is the CEILING, and it is stronger.** The maximum ordinal
reached, over every invocation and workgroup, is **9** — the loop body at `pc91..98`. Ordinals 10..14
(`pc98..116`, everything after the loop) were **never reached by any lane**. So no invocation ever
left the loop, which is exactly the claim at issue, and it now rests on a true reduction rather than
on whichever invocation happened to write last.

The lesson is worth keeping separate from the conclusion: the conclusion survived, and the derivation
did not. A last-writer record produced a *narrower*, more striking span than the truth, and a narrower
span is the direction that reads as stronger evidence — which is why nobody questioned it.


### Where to look

The dispatcher exits when the whole-workgroup liveness slot (scratch slot `padded_lanes +
wave_count`) reads zero. Non-termination means at least one lane stays *active* forever — its
`active_var` never clears, or its next-`pc` never reaches an exit block. That is a recompiler
correctness question about `emit_cfg_state_machine`, independent of any resource or descriptor.

## RETRACTED: the cyclic-table root cause. The hanging dispatch's table is ACYCLIC.

Earlier revisions of this file, and several comments on #2542 and #2481, stated that `0x413dc6700`
hangs because its traversal table contains 183 cyclic start indices. **That is wrong**, and it was
wrong in the most embarrassing way: I measured the wrong buffer.

`0x413dc6700` runs many times per submit with **different tables**. Same program, same fetch PC,
different resolved base:

| dispatch | binding 8 @ `pc=0x5b` | outcome | cycles | longest chain |
| --- | --- | --- | --- | --- |
| `compute[37]`, source 38, order 16836 | `0x20f848417c` | **completed** | **183** | 58 |
| `failure[1]`, source 39, order 16841 | `0x20f848a240` | **HUNG** | **0** | **11** |

I dumped `--dump-compute-resource 37:8` believing it was the hanging dispatch's table. It is the
*succeeding* one's. The whole cyclic-table chain was built on the buffer belonging to the dispatch
that worked.

Two conclusions follow, and the second is the useful one:

- **The hanging dispatch's table is well-formed.** Zero cycles from all 2063 starts, longest chain
  **11 steps**, 19,038 total steps across every start. That loop cannot hang on this data — it is
  three orders of magnitude short of a watchdog timeout.
- **A table WITH 183 cycles was walked by a dispatch that completed.** So a cycle in this structure
  does not hang the shader either, and the loop model that predicted it must be incomplete — most
  likely the per-dispatch `s18` bound means the cyclic region (indices 412..1238) is not reachable
  from that dispatch's start set.

**So the cause of the hang is unknown again.** What survives is everything about the *consequence*:
`0x413dc6700` hangs deterministically at the same dispatch index, RADV hard-recovers, live compute is
disabled process-wide, the indirect latch drops every remaining draw, and skipping that one program
yields zero device losses and the first real scene content this title has produced. The mechanism
inside the shader is not established, and the "183 must become 0" oracle is void.

The pointer-chase loop is now *less* likely to be the hang: it is one of three dispatcher loops in the
module, and it is the one just shown to be bounded at 11 iterations on the hanging dispatch's own
data. The other two are unexamined.

## Other open defects

- **#2445** — specific lowercase glyphs (`r`, `s`, `m`) dropped from UI text: "Ente ing Sto y Mode".
  Surrounding text is intact, so it is per-glyph, not a font failure.
- **#2429** — the world cannot render on 32-wide devices: the EXEC-population-count fix requires
  `native_subgroup_size == wave_size == 64`.
- **#2428** (Windows) — frame-rate cliffs of ~60x (62 fps to 0.8 fps) within 60 frames at the gameplay
  transition.
- **#2424** (Windows) — `sceKernelBatchMap` `MAP_DIRECT` fails `ENOMEM` on a
  map-small/unmap/map-larger cycle. No Linux equivalent.

## Instruments worth knowing about here

- **`PROSPER_COMPUTE_SKIP_PROGRAM=0xADDR[,...]`** — decline named compute programs. A bisection tool,
  not a workaround. It announces itself at parse time and reports each skip through the decline census
  as `reason=skipped-by-selector`, so a diagnostic run can never be mistaken for a default one later.
  Ordered **after** the trace and SPIR-V dump, so "dump the module, skip the dispatch" works — which
  is how a recompiler change can be checked against a program that hangs the GPU, at zero device cost.
- **`[compute-decline]`** — every refusal in `execute_item` now names its reason, the dispatch, and a
  **running count**. The count is in the line deliberately: rate-limited diagnostics in this codebase
  have twice been read as censuses.
- **`PROSPER_SUBGROUP_LOG`** — the resolved native-subgroup contract. Read it as an *input*: it says
  what the config resolved to, not what the emitter produced.
- **`RADV_DEBUG=hang`** — the vendor tool, no prosper change needed. Writes a report to
  `~/radv_dumps_<pid>_<ts>` with `vm_fault.log`, `trace.log` (the last CP trace point) and
  `pipeline.log` (shader stats for the hung pipeline). `umr` is not installed here, so per-wave PCs
  are unavailable, and `DISASM` needs an additional RADV flag.

## Shared-GPU policy on this title

Every device-loss experiment here costs a hard recovery on a machine other agents are using. Allow at
most **one expected device loss per hypothesis**, stop that process immediately after the loss, and
investigate before the next routed run. `PROSPER_COMPUTE_SKIP_PROGRAM` plus the SPIR-V dump removes
the need for most of them.
