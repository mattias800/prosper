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

On this branch all four instructions compile and the live renderer consumes their bindings; no
`[recompile-reject]` or `[render-msaa-reject]` occurs. A 30-second run published six source-distinct
sampled frames while all six remained pure black (`crc=064567f8`). This is generic capability progress,
not visible title progress, and the compatibility rung remains 0.

The historical renderer-publication stop at roughly 125-138 seconds was a separate guest allocation
failure and was fixed by #1748. Bounded current runs no longer reproduce that failure; it is not part
of this MSAA work.

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
  failure was fixed by #1748. The current reproduced blocker is black output from the MSAA shader gap.

## Next discriminators

1. Capture one current black live frame into a replayable bundle and use draw-step/resource inspection
   to identify which final composite draw writes black. The MSAA stage now executes, so another rejected
   or semantically wrong producer/composite is downstream.
2. Treat compute program `0x2011734400` as a candidate only if a discriminator independently proves the
   dispatch ran and reproduces its device loss; two failures in six runs are not a stable cause.
3. If output becomes non-black, compare source-distinct/pixel-distinct frames and post a screenshot.
   Until then, do not update `COMPATIBILITY.md` or claim rung progress.
4. Once output is useful, route beyond 125 seconds as a regression check for #1748; do not reopen the
   old allocator diagnosis without a newly reproduced failure.
