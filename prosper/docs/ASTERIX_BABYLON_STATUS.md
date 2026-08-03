# Asterix & Obelix: Babylon Mission status

Tracking: [#1599](https://github.com/mattias800/prosper/issues/1599)

Title ID: `PPSA30490`

Engine: Unity 6000.1.5f1 / IL2CPP

Current verified rung: **0 — frame submission, but no real visible graphics**

## Current symptom

The title boots and publishes source-distinct renderer frames, but the measured output is still black.
On the pre-fix baseline, the first instruction shown below was the only unique fragment-stage rejection;
every draw using that stage was dropped. Capturing and independently disassembling the complete 37-dword
shader then showed that it is a four-sample resolve: one consecutive-vaddr load followed by three exact
one-extra-dword NSA loads.

```text
[recompile-reject] pc=17 words=f0000130,00000305 fmt=14 op=0x0
                   dst=3 src=5 dmask=0x1 dim=6 len=2
[exec-recompile-reject] ps=0x20117c0900 vs=1821 fs=0 occurrence=8192
```

```text
image_load v3, v[5:7],       s[0:7] dmask:x dim:2D_MSAA  # sample v7 = 2
image_load v2, [v5, v6, v2], s[0:7] dmask:x dim:2D_MSAA  # sample v2 = 1
image_load v0, [v5, v6, v0], s[0:7] dmask:x dim:2D_MSAA  # sample v0 = 0
image_load v1, [v5, v6, v1], s[0:7] dmask:x dim:2D_MSAA  # sample v1 = 3
```

LLVM `llvm-mc --disassemble -triple=amdgcn-amd-amdhsa -mcpu=gfx1030` independently confirms those
exact operands. In the NSA form, the fixed VADDR byte names X and the extra address dword's low two
bytes name Y/sample. Its descriptor resolves exactly as 1920x1080 `R32_FLOAT`, four samples,
`SW_64KB_Z_X` (mode 24), one guest mip.

On current master all four instructions compile and the live renderer consumes their bindings; no
`[recompile-reject]` or `[render-msaa-reject]` occurs. A 30-second run published six source-distinct
sampled frames while all six remained pure black (`crc=064567f8`). This is generic capability progress,
not visible title progress, and the compatibility rung remains 0.

The historical renderer-publication stop at roughly 125-138 seconds was a separate guest allocation
failure and was fixed by #1748. Bounded current runs no longer reproduce that failure; it is not part
of this MSAA work.

## Current whole-frame discriminator

A current post-MSAA whole-frame bundle at present 180 contains one complete submit: 8 draws, 6
dispatches, 14 ordered operations, and no realization failures. The final two composite draws both
sample the same 1920x1080 R11G11B10 surface, whose captured pre-frame temporal seed is all zero. The
first reading of the dependency graph called that surface an external prior-frame leaf because no
ordinary color write targeted it.

That reading was an instrument defect. Four draws in the submit have
`CB_COLOR_CONTROL.MODE=RESOLVE`: they are fixed-function copies from color0 to raw color1, not shader
draws. Color1 is the composite's sampled surface, and its shader write mask is correctly zero because
the pixel shader does not execute for a resolve. The live backend already implements this contract,
but `gpu_replay --graph` modeled each resolve as an ordinary shader draw: it invented a color0 write,
ignored the color1 destination, and included shader resources that never execute. With resolve-aware
graphing, the retained bundle changes from this false result:

```text
dependency-graph operations=14 edges=0 external-leaves=15
external consumers=2 first=8 future-writer=-1 stage=ps binding=34
         addr=0000002011800000 bytes=8847360 dims=1920x1080
```

to the complete six-edge in-submit chain:

```text
edge producer=4 consumer=5  stage=resolve-src addr=0000002033b10000
edge producer=4 consumer=6  stage=resolve-src addr=0000002033b10000
edge producer=6 consumer=8  stage=ps binding=34 addr=0000002011800000
edge producer=4 consumer=9  stage=resolve-src addr=0000002033b10000
edge producer=4 consumer=10 stage=resolve-src addr=0000002033b10000
edge producer=10 consumer=11 stage=ps binding=34 addr=0000002011800000
```

The corrected graph reports six edges and no external image leaf for the final composite source.
The bounded prefix replay ran through operations 4/6/8/10/11. All five outputs were the same
pure-black image (`hash=792bed5a3f02a383`; RGB min/max/mean zero), although they do not all name the
same surface. Inspect and graph output map them respectively to draws 0/2/4/6/7. Operation 4 is the
first graphics operation and directly renders color0 `0x2033b10000` black; its fragment shader's only
captured input is a zero-filled 128-byte constant buffer, and the temporal seeds in this chain are
also zero.

Ordinary prefix output could not directly observe raw color1 `0x2011800000` for fixed resolves 6 and
10 because it selects their color0 source. #1825's exact post-operation selector has now closed that
apparatus gap for both operations. Each readback proved semantic draw, attachment slot, address,
1920x1080 extent, and raw/backend format 122; both destinations were uniformly black with the same
`792bed5a3f02a383` replay hash and the same BMP SHA-256 as operation 4. Neither fixed resolve is a
useful-color-to-black transition, so this capture still does not justify a product-code fix.

A separate 64.7-second semantic timeline recorded 31,982 complete submits before a recoverable
truncated tail. Submit 1 had 8 draws/10 dispatches; every later submit had 8 draws/6 dispatches, and
all target spans remained 1920x1080. This proves the folded workload did not change its coarse
draw/dispatch/target signature during that run. It does not prove the underlying resources or pixels
were constant. The run also established an apparatus boundary: omitting `PROSPER_RENDER` disables
graphics rendering, but supported compute still initializes and uses the Vulkan device.

## Implemented contract under test

- `2D_MSAA IMAGE_LOAD` lowers to `OpImageFetch` over a single-sample Vulkan 2D-array image. The four
  guest sample planes become four host array layers; the explicit guest sample VGPR becomes the layer
  coordinate. The gate accepts only the independently disassembled consecutive-vaddr packet and the
  exact one-extra-dword NSA `[x,y,sample]` form with unused address bytes zero. This preserves the
  instruction's exact texel-fetch semantics without inventing a Vulkan multisample upload path.
- Descriptor, compiled-shader, capture, upload, and persistent-image identities all retain
  `sample_count`. A 4x module cannot alias the deliberately unsupported 2x variant.
- The CPU detiler uses AMD's published 16-pipe GFX10 `SW_64KB_Z_X` 4xaa pattern. Its output is
  plane-major (`sample * width * height + y * width + x`), directly matching the host array layers.
- The live frontend accepts only the observed 4x/mode-24/R32F/one-mip reflected texel-fetch shape.
  It requires the complete padded tiled allocation, owns the resulting pixels, and reuses them inside
  the submit. Cross-span reuse is allowed only when the ordered write journal proves the complete
  padded source span unchanged.
- Unsupported shapes, short authoritative allocations, and ambiguous compression metadata remain
  visible rejections; no fallback black/zero texture is fabricated.

## HTILE, not DCC

The descriptor's compression bit does not make this mode-24 surface color-DCC data. AMD AddrLib's
`HwlComputeDccInfo` restricts color DCC to `R_X`, while `HwlComputeHtileInfo` accepts pipe-aligned
`SW_64KB_Z_X`. For the project's 16-pipe GFX10 configuration, one HTILE meta block is 32 KiB and
covers 1024x512 pixels; 1920x1080 therefore has an exact 196,608-byte metadata plane.

The live path reads base R32F samples only when the **complete** HTILE plane is uniformly one of PAL's
documented initialization values that disables Z compression:

- depth-only: `0xFFFC000F`
- depth+stencil: `0xFFFFF3FF`

The title-live plane is a third, importantly different exact state: all 196,608 bytes are zero. PAL's
`Gfx9Htile::GetClearValue` proves that this is the depth-only fast-clear encoding for +0.0
(`ZMin=ZMax=0`, `ZMask=0`). In that state metadata is authoritative and the base allocation is stale;
the frontend materializes exact +0.0 into all four host layers without reading base memory. This is
not a fabricated black texture: it is the guest's complete uniform fast-clear value. The gate does not
generalize to nonzero fast clears or depth+stencil encodings.

The descriptor does not otherwise prove which plane kind the title created, so metadata must prove one
exact complete state. A one-bit-near value, mixed values, unsupported uniform nonzero data, and a short
read all reject.

Primary sources:

- [AMD AddrLib GFX10 HTILE/DCC sizing](https://github.com/ROCm/ROCR-Runtime/blob/d614ea8bbd73a7832c265725117274d13041ff06/src/image/addrlib/src/gfx10/gfx10addrlib.cpp)
- [AMD PAL `Gfx9Htile::GetInitialValue`](https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9MaskRam.cpp)

Capture v44 retains both the complete tiled base allocation and this exact HTILE span. Because v44
added sample count in an append-only tail, the reader defers only the 2D-MSAA metadata-footprint equality
until that tail is read, then validates it before accepting the capture. Versions 1-43 keep their
historical single-sample default and cannot claim an MSAA HTILE span.

## Verification

Reduced CPU/structural suites pass for:

- independent sample-bit golden offsets plus multi-block round trips at 1/2/4/8/16 bytes per sample;
- truncated tiled allocations leaving output untouched;
- exact T# decode and resource backing size;
- sample-to-layer `OpImageFetch` structure and reflected 2D-array/non-MS/texel-access contract;
- exact title NSA address-byte mapping, plus rejection of unused bytes and longer NSA forms;
- v44 capture round trip and v43 compatibility;
- 1920x1080 HTILE sizing, both exact decompressed constants, and one-bit/nonuniform/short rejection;
- uniform-zero HTILE producing exact +0.0 in every texel of all four layers while a poison/NaN base is
  ignored; one-word-nonuniform, uniform-nonzero, short-footprint, wrong-count/layout/format arms reject;
- offline replay retaining the complete zero-HTILE blob needed to reproduce metadata-owned output;
- compiled-cache separation of supported 4x and rejected 2x shader variants.

The reduced Vulkan semantic test uploads four distinct R32F planes and fetches four distinct colors.
Its short-span and unsupported-count arms reject before Vulkan work. This is not yet title-live visual
verification and does not advance the compatibility rung.

Mutation evidence:

- aliasing all sample offsets to sample zero breaks the named MSAA golden/round-trip tests;
- skipping the full HTILE-plane scan breaks exactly the named nonuniform-metadata rejection;
- disabling uniform-zero recognition breaks exactly the named poison-base depth-zero materialization;
- removing the len-3 gate breaks the exact title NSA layer test while the consecutive packet survives;
- changing only the shader table's sample count produces a cache miss and the expected 2x rejection.

Live verification was repeated six bounded times after the NSA form compiled. Two early diagnostic runs
hit an intermittent compute `VK_ERROR_DEVICE_LOST` at program `0x2011734400`; four subsequent runs did
not, including the 30-second run, and every run remained black. That event is therefore recorded as
intermittent and non-causal for the stable black output, not promoted to the next blocker without a
self-validating reproduction.

During the CPU follow-up after the exact operation-10 readback, two commands were incorrectly treated
as extraction-only: `gpu_replay --dump-realized-shader` and `--dump-resource`. Each wrote its requested
artifact and then fell through to a complete Vulkan replay; both completed with the same black
`792bed5a3f02a383` output and no error, rejection, device loss, or mismatch. They were disclosed as
unintended runs and add no independent title discriminator. In the same observed path,
`--dump-shader` returned before Vulkan initialization. This apparatus finding is recorded as
instrument trap 63 in `GAME_COMPAT_ORCHESTRATION.md`.

Two independent fresh-save v46 runs on `09be7beccc93c6e2d414657c3570f877271fdcdf`
then retained draw 0 PS binding 32's raw descriptor input. Both selected the same natural submit shape,
reported `raw-identity=full-match`, and retained the same complete zero 128-byte span at realization and
post-submit. A follow-up allocation-census arm again reproduced that exact v46 subject and found one unique
containing direct mapping: `[0x2011500000,0x2012500000)`, physical allocation `0x21500000`, selected offset
`0xdd32f0`. Caller attribution from that arm is **VOID**, not negative: 102 allocation records produced only
two `[dmem-caller]` lines because the diagnostic silently suppresses repeated first-two-frame keys and the
ordinary allocation line carries no correlation token. The target allocation therefore cannot safely be
assigned either retained chain. #1599, #1859.

The corrected caller-chain instrument was then rerun on master `1ccb88da` with the same exact v46
submit/draw/stage/binding selector. The run completed 220 presents and captured submit 181 with the same
8 draws, 6 computes, 14 operations, zero realization failures, raw descriptor full match, and uniformly
zero 128-byte binding. This time every one of 102 main-direct-memory allocations carried correlation:
97 were `caller-chain=1`, five were `caller-chain=2`, and none were unknown or overflow. The selected
address was contained by exactly one map, `[0x2011500000,0x2012500000)`, whose unique allocation record
was `len=0x1000000`, physical `0x21500000`, `caller-chain=1`; the selected offset remained `0xdd32f0`.
Exactly one full definition existed for chain 1. Its first guest frame, `eboot+0x1b2454f`, follows a call
whose PLT relocation resolves to `sceKernelAllocateMainDirectMemory`, independently confirming that the
heuristic stack value is a real return site. Capsule SHA-256 was
`cedf5d7bea196efa58fb90a0d69c1b03359084b90e5ae8cc8cda5ff5b57ca4de`. The earlier caller arm remains
VOID as an apparatus result; the corrected arm establishes the dynamic allocation family and target
membership, not uniqueness within that family. #1599, #1859,
#1863.

#1902 adds the next generic diagnostic contract. Four-field `PROSPER_DMEM_WRITE_TRACE` syntax,
`<caller-chain>:<exact-allocation-size>:<offset>:<bytes>`, requires a unique allocation and rejects a
second match as ambiguous. The optional five-field form inserts a one-based, bounded occurrence before
the offset; it reports requested, observed, and selected occurrences and never accepts a prior run's VA.
It waits for the correlated allocation, arms every known writable alias when the selected physical bytes
are mapped, and records initial plus bounded faulting RIP/thread/before/after history.
Any writable alias added after the initial arm invalidates the trace: that alias existed RW before its
mapping notification, so retroactively protecting it could not prove continuous process-wide coverage.
The Linux canary performs the first selected write from a different thread, an unrelated same-page write,
a second selected write after re-arm, and one event beyond the history cap. A re-arm-bypass mutation fails
the named history checks. It also holds the trace mutex across the first post-store trap and places a marker
store as the exact following instruction: bounded acquisition must snapshot/re-arm while still in that first
trap, with the marker unchanged. Reintroducing the old return-to-guest retry makes that named contention
check fail. Lock-acquisition exhaustion terminates with explicit `step-lock-timeout`; it cannot widen the
claimed single-step window. Because one instruction executes while the page is RW, only an event whose
`coverage-valid-before=yes` has complete process-wide pre-fault coverage; after any step, later negative
coverage is explicitly undetermined (or invalid/overflow), per orchestration trap 68.

Independent review then found and deterministically reproduced three generic apparatus gaps before
merge. Releasing an overlapping production `GuestWriteWatch` could make a still-`Armed` trace page RW;
production re-arm could make the trace's pending one-instruction RW window RO. Page protection now uses
the union of both logical owners, with Stepping as an explicit RW override that refuses production
create/re-arm on covered pages. A sibling invalidation during Stepping now retains the exact published
TID until `TRAP_TRACE`, returns `CompleteInvalid` so the caller clears TF, preserves the first invalid
reason, and skips both stale reads and re-arm. Finally, `si_addr` is no longer treated as an operand
extent: a bounded signal-safe decoder proves contiguous spans for common MOV stores (including the live
event-1 qword form). An unrecognized off-range instruction always reports `selected=unknown`; a changed
post-image is retained separately as `changed-during-window=yes`, because a sibling can write while the
page is process-wide RW. Real fault canaries cover
production reset/re-arm coexistence, sibling physical invalidation between SIGSEGV and TRAP, and a qword
beginning four bytes before the selected interval. These harden the reusable instrument only; no new
Asterix live run or writer conclusion was produced, and both historical live verdicts below remain VOID.

A second exact-head review found five more ways the diagnostic could manufacture or lose evidence.
Strict-unique mode could retain a real occurrence-1 event but still publish `writer-observed` after
occurrence 2 made the subject ambiguous; terminal results now require a still-valid selected identity.
Reconfiguration is refused without mutation while a TF step is pending. `CompleteInvalid` now carries
an unavailable post-image explicitly and prints `post=unavailable`, never its value-initialized zeros.
An unknown off-range instruction remains `selected=unknown` even when selected bytes changed during the
process-wide RW window; the delta is retained separately because a sibling can be its writer. Finally,
an instruction arriving with TF already owned by BP/HWBP/STEPWIN invalidates and opens the trace page but
does not claim or clear that trap. Deterministic real-fault canaries cover selected-event-then-ambiguity,
paused-step reconfiguration, unavailable post formatting, a sibling write during an unknown instruction's
window, and a real `int3`-owned step through the protected store. These are apparatus corrections only and
produce no new title-live conclusion.

A third exact-head review closed two remaining capability/addressing holes in the apparatus. The
retained boolean fault handler cannot request TF or route its completion, so it now preserves an
overlapping production watch's Dirty/Resume behavior while invalidating the diagnostic as
`single-step-unavailable`; it never publishes sentinel TID 0 or leaves selected pages stranded RW.
Separately, a decoded store width is no longer added to page-boundary CR2 as if the first inaccessible
byte were the operand start. A real qword beginning at `P-4`, faulting at P, and ending before a
selection at `P+4` now remains selection-uncertain rather than becoming a false writer. Both real-fault
canaries have defect-shaped mutations that fail only their named checks. These remain reusable
instrument corrections; no new Asterix live run or writer conclusion was produced.

The first exact live arm at commit `d181953e` was **VOID, by a useful self-invalidating result**. It
completed 220 presents and captured submit 181 with the same exact positive control: 8 draws, 6 computes,
14 operations, zero failures, and draw 0 PS binding 32 at `0x20122d32f0`, with 128/128 bytes at both
samples, `full-equal`, raw descriptor `full-match`, and all bytes zero. The allocation census was again
102 records (97 chain 1, five chain 2, none unknown/overflow), but **23** records were chain-1 16 MiB
allocations. The trace selected occurrence 1, physical `0x20500000`, mapped at `0x2010500000`, and armed
`0x20112d32f0`. Occurrence 2 was physical `0x21500000`, mapped at `0x2011500000`, and its offset resolves
to the capture-selected `0x20122d32f0`; its arrival correctly invalidated the trace as
`ambiguous-allocation` before any fault. The terminal summary reported two matches and zero events.
Thus caller-chain plus allocation size is not unique, while two source-distinct runs now identify the
resource as occurrence 2 of that run-relative allocation family. Capsule SHA-256 was
`b4adcf7e1c81032246a03408fd2c469cef12bb0c559741e039730a29237d5391`. #1902.

The exact occurrence-2 arm at commit `1da4d471` then validated that new lever but was also **VOID as a
selected-byte writer result**. It completed 220 presents naturally. The selector reported requested
occurrence 2, selected occurrence 2 at physical `0x21500000`, and armed `0x20122d32f0`; the terminal
census observed all 23 family members. The independent v46 positive control named that exact same VA for
draw 0 PS binding 32, again with 128/128 bytes at both samples, `full-equal`, raw descriptor `full-match`,
and all bytes zero. Thus the run-relative allocation identity is accepted. However, the selected bytes
share a page with heavy earlier activity: all 64 retained events were outside the selected range, spanning
39 addresses from `0x20122d30b8` through `0x20122d3220`, below the selected start at `0x20122d32f0`.
Event 1 had complete pre-fault coverage and was a real guest store at `eboot+0x1a5402e` to
`0x20122d30c0`; independent disassembly confirms `mov QWORD PTR [r8+rdx+0x10],r10`. That first unrelated
page write consumed trap 68's only complete interval. Event 65 then caused explicit overflow before any
selected event. The terminal summary was `status=overflow`, with 23 observed allocations, occurrence 2
selected, 65 page faults, zero selected faults, 64 steps/rearms, 65 coverage gaps, and 64/64 events.
Therefore there is **no selected-byte CPU-writer conclusion** from this arm; merely raising the history
cap could at most produce a later writer hint, not restore process-wide first-writer coverage. Capsule
SHA-256 was `e358535fe2b864394a24df0a0d8735fcbfa1b18a79f57b67a13f0745a3b51f6d`. #1902.

## Ruled out

- **Unresolved T#/S# lookup:** the title-live descriptor resolves; the rejection is the sampled MIMG
  dimension gate, not descriptor discovery.
- **Ordinary 2D or 2D-array sampling:** this packet is integer-coordinate `IMAGE_LOAD`, and the third
  coordinate is an explicit sample index. Normalized sampling would change semantics.
- **The first consecutive-vaddr packet is the complete shader:** the raw 37-dword stage contains three
  further NSA loads. LLVM independently confirms their `[x,y,sample]` address bytes; a len-2-only gate
  was a self-invalidating instrument because it rejected at the first len-3 packet.
- **Ordinary single-sample detiling:** it aliases sample zero and under-reads the padded allocation.
  The published 4xaa sample bits independently select byte offsets 0/4/8/12 for one R32 texel.
- **Color DCC code `0xff`:** mode-24 metadata is HTILE. Applying the existing `R_X` DCC footprint or
  decompressed marker here is the wrong instrument.
- **Treating all compression-enabled base bytes as readable:** only a complete uniform PAL
  decompressed HTILE state authorizes base decoding. Uniform zero is separately a metadata-owned
  depth-only +0.0 fast clear and explicitly ignores base. Anything else still needs a real
  compressed-depth path and is deliberately unsupported.
- **A short/unreadable metadata plane behind the former `base-uncompressed=0`:** the corrected
  diagnostic measured exactly 196,608/196,608 bytes, all zero, on all four bindings.
- **The intermittent compute device loss as the stable black-frame explanation:** it appeared in two
  of six bounded runs. Four clean runs, including 30 seconds, remained identically black.
- **A visible-progress claim from reduced tests:** none has been made. A patched live run and human
  screenshot are still required.
- **The historical ~125-second publication stall as the current blocker:** its command-packet/allocation
  failure was fixed by #1748. The current reproduced blocker is the post-MSAA black output above.
- **The final composite source as an unresolved prior-frame producer:** the old graph omitted
  fixed-function `MODE=RESOLVE` edges. Its `edges=0` result falsely listed `0x2011800000` as an
  external leaf consumed by operations 8/11. The corrected six edges are 4→5, 4→6, 6→8, 4→9,
  4→10, and 10→11, closing that source through four color0-to-color1 copies in the same submit. The
  bounded prefix replay subsequently found the first graphics producer already black; it did not
  reveal a useful-color-to-black boundary or justify a renderer change.
- **The op6 fixed-function resolve turns a nonblack op4 producer black:** #1825's exact post-operation
  selector read raw color1 destination `0x2011800000` immediately after op6 as 1920x1080 R11G11B10,
  hash `792bed5a3f02a383`; an independent pixel census found one opaque color with RGB mean/min/max zero.
  That matches the earlier exact op4 color0 cutoff (the same hash and pixel census), so this capture
  contains no nonblack-to-black resolve transition. This is localization evidence only; rung remains 0.
- **The op10 fixed-function resolve is a later loss boundary:** the exact selector proved operation 10,
  semantic draw 6, raw color1 slot 1, address `0x2011800000`, 1920x1080 extent, and format 122/122 before
  readback. It returned the same `792bed5a3f02a383` uniformly-black pixels and byte-identical BMP as
  operation 4 and the op6 destination. Both resolves therefore copy black to black in this capture;
  neither loses useful color. This is localization evidence only; rung remains 0.
- **An omitted immediately preceding submit as operation 4's black temporal input:** a bounded two-frame
  whole-frame capture contains consecutive submits 180 and 181. Resolve-aware scanning identifies submit
  180 operation 4 as the last semantic writer of submit 181 operation 4's same run-local `0x2033b10000`
  target; later `MODE=RESOLVE` operations read that color0 surface and write color1 rather than overwriting
  it. Submit 181's exact temporal seed is uniformly black and byte-identical to submit 180's pre-submit
  seed, so the included previous frame introduced no useful color. #1599.
- **Operation 4 preserves an old black target because it produces no fragments:** a query-capped prefix
  replay retained both primitives after clipping, counted 519,120 fragment invocations, and counted exactly
  2,073,600 passed samples, one for every target pixel. Replacing only draw 0 PS binding 32 with a finite
  color made all 2,073,600 output pixels that color while leaving the same draw counts. The draw actively
  shades and writes the complete target from that constant. #1599.
- **Draw 0's zero PS binding 32 is introduced after the draw, omitted by capture, or written by earlier
  work in the same submit:** capture v45 sampled the exact 128-byte resource both at draw 0's ordered
  realization point and after submit 181. Both reads were complete and all zero, with the same
  `306c537b2ff38983` content hash (`verdict=full-equal`); the ordinary resource blob independently reports
  `declared=128`, `footprint=128`, `captured=128`, and `nz=0`. Operations 0-3 are the only preceding work
  and are compute dispatches. Their reflected writable ranges are disjoint from
  `[0x20122d32f0,0x20122d3370)`; the nearest preceding descriptor is instead a read-only 16-byte constant
  ending 32 bytes before it. The submit contains no DMA operation, and the resolve-aware dependency graph
  independently reports the selected range as `first=4 future-writer=-1`. The zero therefore exists before
  any captured GPU work can produce the composite and remains unchanged for the rest of the submit. The
  run-local address must be re-derived in future runs. #1599, #1850.
- **A wrong descriptor decode/normalization selects a zero buffer instead of the guest-programmed input:**
  the two independent v46 runs each reported `raw-identity=full-match` for draw 0 PS binding 32. Raw dwords
  `122d32f0,00100020,00000008,0004dfac` independently encode the same run-local address, 16-byte stride,
  eight records (128 bytes), format, and component count as the normalized CB. Both temporal samples and
  the ordinary resource read all 128 bytes, found `nz=0`, and agreed on hash `306c537b2ff38983` with
  `verdict=full-equal`; each submit had 8 draws, 6 computes, 14 operations, and no realization failures.
  Descriptor translation preserves what the guest programmed. The missing content is upstream: either
  the guest CPU intentionally leaves this allocation zero or an expected CPU producer does not populate
  it. The address remains run-local even though deterministic fresh boots reused it. #1599, #1853, #1857.
- **A renderer-disabled title boot as a GPU-free experiment:** live compute remains active without
  `PROSPER_RENDER` and initializes Vulkan when the title dispatches supported compute.

## Next discriminators

1. Treat occurrence 2 as accepted allocation identity: its dynamically selected VA exactly matched the
   independent v46 resource address while the full family census remained 23. Do not return to a fixed VA or
   the ambiguous four-field selector.
2. The first same-page writer is now identified, but it is outside the selected bytes and consumes complete
   coverage. Build the next generic instrument only if it can distinguish selected-range writes without
   silently losing process-wide coverage; a larger history alone can provide a later writer hint, not the
   first selected writer. Keep total faults/history bounded and every coverage gap explicit.
3. If a later diagnostic observes a selected writer, confirm its module+offset in independent disassembly
   before following that producer; do not promote it to process-wide first writer unless the new mechanism
   independently closes trap 68's RW interval.
4. Treat compute program `0x2011734400` as a candidate only if a discriminator independently proves the
   dispatch ran and reproduces its device loss; two failures in six runs are not a stable cause.
5. If output becomes non-black, compare source-distinct/pixel-distinct frames and post a screenshot.
   Until then, do not update `COMPATIBILITY.md` or claim rung progress.
6. Once output is useful, route beyond 125 seconds as a regression check for #1748; do not reopen the
   old allocator diagnosis without a newly reproduced failure.
