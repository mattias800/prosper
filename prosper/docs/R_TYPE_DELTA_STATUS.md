# R-Type Delta: HD Boosted status

`PPSA26414` is currently blocked first by the title's logged-in-user startup race (#1746). A local
diagnostic that caps the title's initial sleep reaches its frame loop, where the movie conversion
draw was then dropped at fragment-shader `IMAGE_SAMPLE` pc 52 (#1753). This note records the static
causal chain behind that downstream failure. The startup diagnostic is not a product fix.

## Movie resource failure

The rejected fragment shader is `shader/movie_yuv_p.ags`. Its AGC metadata declares two direct
textures at user SGPRs 0 and 8, two direct samplers at SGPRs 16 and 20, and one constant buffer at
SGPR 24. The shader samples texture slot 1 / sampler slot 1 first at pc 52 (`T# s8`, `S# s20`), then
texture slot 0 / sampler slot 0 at pc 57. The pc-52 instruction is a supported 2D `IMAGE_SAMPLE`;
the failure is that texture slot 1 was never constructed.

Guest disassembly at `eboot+0x19d10` explains why:

1. The title reads `AvPlayerFrameInfoEx::video.height` at frame offset `0x1c` and `pitch` at `0x3c`.
2. It constructs the luma R8 T# using `pitch` as the descriptor width and `height` as its height.
3. It asks the guest AGC library for that descriptor's allocation size and requires it to equal
   `pitch * height`.
4. Prosper previously published a tight 1920-byte pitch. The AGC sampled-linear footprint is
   256-byte aligned, so the descriptor occupies `2048 * 1080 = 2,211,840` bytes while the title
   expects `1920 * 1080 = 2,073,600`. The title prints `unexpected y_texture size` and returns
   before constructing the chroma T# in texture slot 1.
5. The one-time initialization flag has already been cleared. Later frames bind the valid luma T#
   and the never-constructed chroma T#, so pc 52 cannot resolve SGPR 8.

The PS5 `AvPlayerVideoEx` ABI keeps physical and visible extents separate: crop offsets are four
32-bit pixel counts at offsets 20, 24, 28 and 32, followed by the physical pitch at offset 36.
Prosper now stages 1920 visible bytes into a 2048-byte physical row and publishes
`crop_right_offset=128`. R-Type's own movie code subtracts left/right crop from pitch when it builds
the visible quad, so the padding is not displayed.

This is also the cross-title contract needed by GRIS. Its PS5 video wrapper computes visible width
as `pitch - crop_left_offset - crop_right_offset`. The earlier experiment in #1393 that padded the
pitch but left all crop fields zero correctly removed the stride corruption, but exposed the
128-pixel tail as a right-edge strip. Padding and crop are one contract; testing only the first half
made the correct physical layout look invalid.

## CPU guard and mutation evidence

`test_avplayer` supplies a 1920x1080 NV12 frame with 2048-byte decoder strides and checks all of the
following without launching Vulkan:

- the callback-owned frame ring has 2048-byte luma and chroma rows;
- only 1920 visible bytes per row are copied and padding remains cleared;
- stream and frame metadata publish pitch 2048, crop right 128 and zero for the other crop offsets;
- luma and chroma guest addresses preserve exact 2048-byte layout provenance;
- the R-Type descriptor footprint equals `pitch * height`;
- the GRIS visible-width expression evaluates back to 1920.

Two mutations prove the checks exercise both failure mechanisms. Restoring `pitch=width` makes the
named R-Type footprint check fail. Keeping pitch 2048 but forcing crop right to zero leaves the
R-Type footprint check green and makes the named GRIS right-edge-strip check fail.

## Live validation

Candidate `4114391c` was run for 60 seconds through the bounded #1746 startup diagnostic. The three
byte patches applied at the expected addresses, AvPlayer opened `CRG.mp4`, allocated two 3,317,760
byte guest texture buffers, returned those buffers alternately, and advanced timestamps. All #1753
failure signals were absent: no `unexpected y_texture size`, no pc-52 `mimg-unresolved` from SGPR 8,
no pc-52 `recompile-reject` for `f080030a`, and no execution reject for fragment program
`0x2011c0d700`. No other shader/resource rejection appeared. This proves the upstream descriptor
construction and shader-resolution defect is fixed.

It does **not** establish a visual milestone. All 49 retained 1920x1080 frames were inspected. Early
and late samples were solid clears; the only non-flat interval contained 2--75 colours per frame and
showed broad diagonal purple/grey gradients rather than recognizable movie imagery. There is no
screenshot because incorrect diagnostic output is not progression evidence. The remaining movie
sampling/content frontier is tracked separately in #1807.

The same candidate completed the GRIS cross-title route with 45 source-distinct and 42 pixel-distinct
frames, maximum pixel staleness 1.0 seconds, and status `ok`. All 45 frames were inspected. The
Nomada/Devolver opening logos and title screen have no 128-pixel right strip, horizontal stride
corruption, or U/V colour cast. This verifies that the padded-pitch/crop contract does not regress
the known consumer that exposed the earlier half-fix.

## Movie sampler-coordinate recovery

An F9 capture of present 1740 retained the next submit as a replayable two-draw frame. The first
draw uses fragment program `0x2011c0d700` and samples the staged movie planes at the exact AvPlayer
addresses: an R8 2048x1080 luma plane and an RG8 1024x540 chroma plane. Both resources are complete
and non-flat. Converting those captured bytes as padded NV12 on the CPU produces a sharp,
full-colour movie frame, while replaying the same submit on unmodified `dad8f518` produces the
purple diagonal gradient (`c87261d44cd4426b`). That separates decoder/staging correctness from the
downstream sampling failure without another title boot.

Both direct S# descriptors set `FORCE_UNNORMALIZED`, and the vertex shader supplies pixel/texel
coordinates rather than normalized UVs. The descriptor decoder already preserved that bit, but no
renderer consumer applied it. Treating coordinates such as 1920 as ordinary normalized values made
the sampler wrap or clamp almost the whole quad onto a few edge texels; the triangle-strip primitive
boundary made the resulting alias appear as a hard diagonal.

The generic lowering keeps the ordinary Vulkan sampler, preserving the guest's wrap modes and LOD
state, and scales only spatial sample coordinates and explicit gradients by the corresponding
resource extent in the shader recompiler. Array layers, DREF, LOD/bias, packed texel offsets and
integer image loads keep their established meanings. Because the reciprocal extents are embedded in
SPIR-V, the compiled-shader cache key includes the unnormalized contract and its dimensions.

Recompiling every retained raw shader on `4dfcaa94` changes the exact replay hash to
`51d9511c303d9182` and recovers recognizable, detailed movie content. Disabling only the coordinate
lowering returns byte-for-byte to `c87261d44cd4426b`; the focused live-renderer mutation likewise
keeps its named pixel check executing and makes only that check fail. This is real visual progress,
but it is not yet visually correct: the replay has a strong green/purple chroma cast and scales the
movie into top/bottom black bars. It also does not bypass the default-launch startup race (#1746),
so the compatibility rung is unchanged.

<p align="center">
  <img src="../../assets/screenshots/rtype-delta-movie-coordinate-progress.png" alt="R-Type Delta movie replay after texel-coordinate recovery; detailed content is visible but chroma and vertical scaling remain incorrect"><br>
  <sub>Retained-frame replay after the coordinate fix. Detail is recovered; the green/purple cast and top/bottom bars remain known defects.</sub>
</p>

## Ruled out

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| pc-52 `IMAGE_SAMPLE` is an unsupported shader opcode | **Falsified.** The opcode/dimension/dmask are already implemented. The exact shader metadata resolves pc 52 to direct texture slot 1 at SGPR 8 and sampler slot 1 at SGPR 20; the missing texture is upstream. | #1753, this doc |
| SGPR 8 is a bindless or dynamically indexed resource that needs a new resolver | **Falsified.** `movie_yuv_p.ags` declares direct texture offsets 0 and 8 and direct sampler offsets 16 and 20. The title explicitly binds texture slots 0 and 1. | this doc |
| The chroma bind packet is lost by Prosper | **Falsified.** The title returns before constructing the chroma descriptor, then later binds that uninitialized slot itself. There is no valid descriptor for a packet decoder to recover. | guest `eboot+0x19d10`, this doc |
| AvPlayer output must remain tight because padded output caused GRIS's right-edge strip | **Falsified.** GRIS subtracts the ABI crop offsets from pitch. The prior padded experiment left crop right zero; the missing crop, not the physical padding, exposed the strip. | #1393, GRIS `eboot+0xf6d5ab`, this doc |
| Reporting pitch 2048 while retaining tight rows is sufficient | **Falsified by construction.** The title advances plane addresses by the published physical extent. The CPU fixture verifies row-by-row storage and both plane bases, not metadata alone. | `test_avplayer` |
| The pc-52 resource rejection was the only movie-visual blocker | **Falsified live.** The size mismatch and all shader/resource rejects are gone, but retained frames contain only flat clears and low-colour gradients. The downstream frontier is #1807. | bounded `4114391c` run, #1807 |
| The decoded/staged movie planes contain the low-colour gradient | **Falsified.** The exact captured R8/RG8 bytes produce a sharp full-colour frame through an independent CPU NV12 conversion; only Prosper's replay of the same bytes collapses to the gradient. | present 1740 capture, #1807 |
| Vulkan's native unnormalized-coordinate sampler can apply `FORCE_UNNORMALIZED` directly | **Falsified against the live S#.** R-Type combines the bit with wrap addressing (`CLAMP_X/Y=0`) and `maxLod=15.9961`; Vulkan requires clamp addressing and zero LOD for an unnormalized sampler. The native candidate correctly became fail-visible but could not represent the guest contract. | retained submit 1741, #1807 |
| The diagonal proves the fullscreen strip's topology or interpolation is wrong | **Falsified by the coordinate-only A/B.** Recompiling the same raw VS/FS and changing only texel-to-normalized lowering recovers detailed content and removes the diagonal; disabling only that lowering reproduces the exact old hash. | hashes `51d9511c303d9182` / `c87261d44cd4426b`, #1807 |

## Next frontier

The coordinate failure in #1807 is isolated and mutation-proven. The next movie-rendering frontier
is the green/purple chroma cast: compare the raw shader's sampled component/swizzle flow and YUV
conversion constants with the captured RG plane before changing upload order. The vertical bars are
separate: compare the movie VS constant buffer and generated strip positions with the full-height CPU
frame. Default launch still stops earlier on #1746; neither replay improvement changes that product
route or the compatibility rung.
