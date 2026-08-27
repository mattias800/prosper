# Tomb Raider I-III Remastered (`PPSA16901`) — status

Tracker: [#2990](https://github.com/mattias800/prosper/issues/2990).
Route files and the menu map: `prosper/scripts/tomb-raider-PPSA16901/AGENTS.md`.

**Rung 3** — Croft Manor runs with real GPU draws and the scene renders with correct geometry.
Every surface is still untextured ([#2998](https://github.com/mattias800/prosper/issues/2998)),
which is what keeps this below a rung-4 claim.

Prior to 2026-08-26 the title had no record anywhere in this repository — no `COMPATIBILITY.md`
row, no tracker, no mention in any source file or script. It reached rung 2 with **no prosper code
change at all**; rung 3 needed exactly one fix, below.

## The dump

6.1 GB. A launcher `eboot.bin` plus three self-contained games under `1/`, `2/` and `3/`, each with
its own `tombN.prx`, and `pros.sdk.Prospero-9.000.prx` alongside. **SDK 9**, so this title is on the
pre-13 side of the post-submit completion-visibility contract (#2219) — worth remembering if a
submit-race signature ever shows up here, though nothing so far points that way.

## What renders

| State | Result |
| --- | --- |
| EULA gate (40 pages) | renders and scrolls; body text garbled (#2999) |
| Publisher/developer logos | correct — the Saber Interactive logo is pixel-clean |
| Tomb Raider I title screen | logo, Lara portrait, animated ring menu and labels all correct |
| Tomb Raider II title screen | logo, Lara model, villain silhouette and background all correct |
| Tomb Raider III title screen | art correct; ring area is a solid violet block |
| Croft Manor (`Lara's Home`) | correct geometry, correct character models, **untextured** (#2998) |

## The one fix that produced rung 3: an unannounced 32-bit index buffer with large indices

The world used to render as long stretched triangles radiating from a point, while character meshes
in the same frame were shaped correctly. That was **#304's defect, in a form its detector could not
see**.

This title announces an index size **exactly never**: `index_type=0` on all **508,688** indexed
draws of a boot to Croft Manor. Its world index buffers are nevertheless 32-bit, because the level
is drawn from one shared **775,111-vertex** pool — which 16-bit indices cannot address. Read as
16-bit, each 32-bit index becomes two, and every triangle collapses to a degenerate `(N, K, N)`
sliver.

`index_buffer_is_unannounced_32bit` already recovers exactly this case for DOLL, but its fingerprint
requires every high half to be **zero**, i.e. every index below 65536. Here the indices sit in a
64 KiB window *above* zero, so the repeated high half is a small **non-zero** constant and the
fingerprint cannot match. `index_buffer_is_unannounced_32bit_high` handles that form.

**The byte pattern alone is not enough, and that is the part a review caught.** With no
`DrawIndexOffset` the caller passes the same pointer twice, so the two readings are the same bytes —
and a genuine 16-bit buffer with a period-2 pattern (a fan or cone encoded as a triangle strip
`[rim, apex, rim, apex, …]`, or a line list radiating from one hub) is then *byte-identical* to a
clustered 32-bit list. Constructed by hand and confirmed against the shipped header: a 64-spoke line
list to hub vertex 7, and a triangle-strip cone with apex 12, both satisfied every byte-pattern
clause of the first version. No further test on those two pointers can separate them.

So the deciding evidence comes from **outside** the buffer: an index must address a vertex that
exists. The detector takes the bound vertex buffer's **unclamped** record count and requires every
32-bit index to fall below it. The cases separate at once — the cone's 32-bit reading demands 786,640
vertices from a mesh holding tens, while Tomb Raider's demands 775,111 from a pool holding exactly
775,111. Note *unclamped*: the executor's own `vb_entries` is capped at 65,536, and that cap would
reject precisely the case the detector exists for. A caller with no bound passes 0 and the detector
declines, because a discriminator that cannot see is not a licence to guess.

Two further details cost time and are worth keeping:

- **The constant lands on either PARITY.** Which one depends on the alignment of the 16-bit address
  against the 32-bit grid: a `DrawIndexOffset` scaled by 2 instead of 4 lands 2 mod 4 as often as
  not. On the sampled frame the **even** parity carried it for 55,677 draws against 21,871 for the
  odd one, so a first version that checked only odd words fixed a visible minority of the scene and
  left the world shattered. Check both.
- **The 16-bit and 32-bit addresses are different memory**, not two views of the same bytes, for any
  `DrawIndexOffset` — `index_base + offset*2` against `index_base + offset*4`. The regression test
  therefore feeds the detector two independent arrays, as the executor does.
- **The 64-entry sample cap is a PARTIAL mitigation, not a total one.** It exists so a run straddling
  a 64 KiB boundary (which carries two high halves) is not rejected on the parity clause — but the
  parity loop reads the first 64 *words*, i.e. dwords 0..31, while the range loop reads 64 *dwords*.
  Those are different extents, and only crossings at dword 32 or later are rescued: swept, crossings
  at dword 5, 22 and 31 are still rejected. Both halves are pinned by tests, because stated as a
  total mitigation it reads as a guarantee the code does not provide.

## Open defects

1. **Every world and character surface is untextured** and the scene is over-bright
   ([#2998](https://github.com/mattias800/prosper/issues/2998)). The atlas is bound, nothing is
   rejected, and the menus are fully textured.
   **The "84 MB non-zero" this line used to carry is withdrawn (2026-08-27).** A full-byte census
   of the guest allocation reads `filled=19/344` 256 KiB buckets -- about **4.75 MB of 86 MiB** --
   with `unreadable=0`, which the old figure cannot be reconciled with. It arrived in `227f5fbe`
   (#3006) without a recorded method, and **what it actually measured is not established** -- so it
   is withdrawn rather than reinterpreted. Do not repair it by guessing a plausible source (the
   decoded host image is the obvious guess, and guessing is how the head-sample figure survived);
   re-derive it or leave it out. See `## Ruled out`.
2. **Text is intermittently garbled** ([#2999](https://github.com/mattias800/prosper/issues/2999)) —
   the EULA body and the title-screen game selector draw the wrong glyphs while the same font
   renders headings, numerals and every ring label correctly.
3. **Solid untextured quads on the menus** — small pale quads on the Tomb Raider I and II rings and
   a large violet block over the whole Tomb Raider III ring.
4. **No savedata is written, and the ring menu has no passport item.** No `savedata0` directory
   appears across a full boot, and the five-item ring carries no New Game or Load Game. Whether
   those are one defect or two is unestablished.

## Guest 2D_ARRAY textures (#325)

This title textures its whole world from one 512x512x**256** BC7 array, so until #325 the world was
untextured. What makes an array work is that **three** places agree on the same question, and the
predicate is written once — `guest_texture_is_uploaded_array()` in `gpu/texture/bc_decode.hpp`:

| side | follows |
| --- | --- |
| uploader | decodes `depth` slices, publishes the count through `sample_count` |
| view | `VK_IMAGE_VIEW_TYPE_2D_ARRAY` iff `sample_count > 1` |
| recompiler | `OpTypeImage` `Arrayed` at **every** declaration site for such a resource; non-array instructions get layer 0 |

**Arrayed-ness is a property of the RESOURCE, not of the instruction.** The uploader picks the view
type from the guest T# and cannot see which opcode will sample it, so a declaration keyed on the
instruction produces a descriptor mismatch for any DIM=1 sample of an array texture.

The predicate is **block-compressed only**, and that is a real limit rather than caution: the
per-slice decoder carries tiled and linear BC, plain byte-per-texel surfaces, fp16 and unorm16 — but
not fp32 or the 4-byte narrow formats the single-surface decoder converts, so a Float32 array decodes
to **black** (measured). Widening it means teaching the slice loop those two cases first.

Measured layout, for anyone extending this: `layer_stride = 352256` for the 512x512 BC7 atlas —
262144 for mip 0 plus a 6-level chain — with `layer_mip_offset = 90112` selecting the level.
`PROSPER_SLICESTRIDE=1` prints it, once per texture address.

## Ruled out

- **RESOLVED 2026-08-27 — the interior wrong-texture defect (#2998) was the decode cache's
  validation span.** `persistent_base_source_size` returned ONE surface for a layered array, so the
  cache proved reuse against **262144 of 90177536 bytes (0.29%)** of the 256-layer world atlas.
  The title fills that atlas progressively: prosper decoded it once at roughly 5% populated, the
  guest reached 93% by gameplay (measured from a `.prgbundle`: `nz=84070204` of `footprint=90177536`),
  and the cache served the early decode for the rest of the run — so interiors sampled layers still
  holding the previous occupants of that memory. Fixed by `layered_array_source_size`, mirroring
  `layered_cube_source_size`, whose comment already stated the invariant: *"cache reuse can never be
  proved against less memory than the layer-aware decoder reads."* Control: toggling the fix with one
  `false &&` on the same route flips `source_size` 90087424 ↔ 262144 and the frame between the
  correct scene (245141 colours) and passports on the walls (124916), mean difference 51.48/255.
  **The hypotheses this retires as symptoms rather than causes:** layer indexing, `base_array`,
  per-slice stride, and "the atlas is unpopulated" — the guest memory was right all along; prosper
  was reading a stale copy of it. (#2998, #2990)

- **"The world is textured" (2026-08-27 blog entry / #3050 progression evidence)** — RETRACTED the
  same day. The capture was the game blitting its own pre-rendered loading picture,
  `2/PIX/HD/MANSION.DDS`: pixel-compared against the published screenshot it scores **mean abs diff
  0.02/255, 100.00% of pixels within 8/255**, while the next closest asset scores 52.77. Blitting a
  full-screen 2D image requires no world rendering, so the frame never evidenced what it was
  captioned as. Recorded as **instrument trap 230**. Before publishing an unusually good frame from
  this title, decode every same-size `PIX/**/*.DDS` in the dump and diff against it — the check is
  seconds and the real match is unambiguous. (#2990, #2998)

- **The atlas is only ever ~4.75 MiB populated of its 86 MiB span, for the whole run** — falsified
  2026-08-27, but **read the scope before using this**. With the write trace armed at bucket 200
  (`offset=0x83cc000`, target `0x204c3cc000`) that bucket takes **32 events, `selected=yes
  changed-during-window=yes`**, against a positive control at bucket 0 that also fired 32. So the
  allocation is filled progressively and `filled=19/344` is one instant, not a steady state.
  **What this does NOT establish:** bucket 200 is byte 52,428,800, i.e. slice **148.8** at
  `layer_stride=352256` — it sits *below* the interior's slices 186-248, which are buckets
  **249.9-333.2**. This probe never touched the region the interior samples, and the separate
  finding that slice 248 reads zero throughout still stands (re-confirmed 2026-08-27). An earlier
  version of this row said "do not restart 'the guest never fills the upper atlas'" without that
  qualification, which overstated a probe taken below the range in question. (#2998)
- **A stale persistent decode cache holds an early, near-empty snapshot** — falsified 2026-08-27.
  `PROSPER_NO_TEXTURE_DECODE_CACHE=1` over `scripts/tomb-raider-PPSA16901/reach-gameplay.pad`
  changes nothing: the atlas still decodes once. Caveat on the instrument: `[occupancy]` is gated
  on `is_array && depth > 64`, so "one print" is **not** "one decode" — do not quote it as one.
  (#2998)
- **`base_array` is dropped between the T# and the uploader** — falsified 2026-08-27. It is
  applied in `agc_shader_layout.cpp` `image_base_level_view()`:
  `view.base += (uint64_t)d.base_array * stride;` with `view.layer_stride = stride` set alongside,
  on the `thin_2d_layered` `depth > 1` path. `live_renderer.cpp` never names `base_array` because
  the shift is already folded into `gpu_addr`, which is correct, not a gap. (#2998)
- **The guest CPU writer of the atlas cannot be observed** — falsified 2026-08-27, and the cause
  was an allocator, not the watchpoint. `PROSPER_DMEM_WRITE_TRACE` could not target this title
  because only `sceKernelAllocateMainDirectMemory` published a caller chain, while the title
  allocates through `sceKernelAllocateDirectMemory`, which minted none — so no selector could name
  a byte it owns. With both entry points sharing one attribution helper the writer is named
  immediately: `eboot+0xebb82..0xebbfa`, an unrolled AVX scatter storing 8x16 B per iteration to
  offsets computed in `%ymm10`.
  **Reproduction needs #3054**, which made `sceKernelAllocateDirectMemory` publish a caller chain.
  Before it, this title produced no `[dmem-caller]` records at all and the trace refused to arm for
  any configuration — which is the very failure this row retires, so on an older tree do not read a
  failed arm as contradicting it. (#2998, #3054)

- **Nothing maps into the atlas per-slice, and no host/kernel write streams into it** — both with
  controls, both new instruments. `PROSPER_MAPWATCH` sees one 1 GiB `map_dmem` covering the atlas,
  established once, never per-slice (control: watching all of dmem shows two maps).
  `PROSPER_HOSTWRITEWATCH` sees **zero** writes into the atlas while its control — the same watch
  over all of dmem — records **107**. So the ~18 slices that do arrive are written by guest CPU
  stores, which no instrument in the tree could observe **when this was written**. Superseded
  2026-08-27: they are observable now — see the writer row in `## Ruled out` (`eboot+0xebb82..`),
  which also records why the trace could not previously be aimed at this title.
  That is what makes **#3054** the blocking tool: `PROSPER_HWWATCH` is register-relative and cannot
  arm on a fixed guest address, so the one mechanism that would catch a plain guest store is
  unreachable through its current interface.

- **At the moment prosper decodes it, the atlas allocation holds about 4.75 MB of content and
  nothing above it, measured WITHOUT any stride assumption.** (Scope corrected 2026-08-27: this is
  one instant, not the whole run. Bucket 200 is written later — see `## Ruled out`. The interior's
  own slices 186-248, buckets 249.9-333.2, are a separate question and remain unwritten.) `PROSPER_OCCUPANCY=1` walks the guest allocation in 256 KiB buckets and asks
  only "is there content here", so it cannot be fooled by a wrong per-slice layout — which every
  earlier measurement could, since they all addressed slices through `layer_stride`:

  ```
  [occupancy] 0x20491cc000 span=86 MiB bucket=256KiB FULL-BUCKET scan filled=19/344 scanned
              (of 344 total) last_filled_bucket=18 unreadable=0
  ###################...........(325 more)
  ```

  Content sits in buckets 0-18 and stops. At `layer_stride=352256` over `depth=256`, the interior's
  slices 186-248 begin **65.5 MB** into the allocation and end at 87.4 MB. So the data those draws
  need is not in this allocation *at the moment of decode*, at any layout, and the earlier per-slice
  findings were not artefacts of the stride the probes assumed.

  **Two corrections to what this box used to say.** The reading was `filled=18/344` with bucket 0
  shown EMPTY, and the range was given as "48-87 MB". Both were wrong. `[occupancy]` sampled only
  the first 512 B of each 256 KiB bucket (0.195%) and reported that head sample as a census, so a
  bucket whose content began past the window read as empty -- which is exactly what happened to
  bucket 0, and `[slicemap]` contradicted it at the time. It now scans every byte of each bucket and
  distinguishes unreadable (`?`) from empty (`.`). The lower bound of the slice range was simple
  arithmetic error. Neither changes the conclusion, and note the scope this box does NOT establish:
  it is one observation at decode time, not a statement about the whole run -- see the entry below.

- **The allocation is NOT static: it is filled progressively, and a region ABOVE the decode-time
  content is written later in the run.** With the write trace armed at bucket 200
  (`PROSPER_DMEM_WRITE_TRACE=2:0x40000000:0x83cc000:128`, target `0x204c3cc000`) that bucket takes
  **32 events, `selected=yes changed-during-window=yes`**, against a positive control at bucket 0
  that also fires 32. So the occupancy box above measures the state at the one moment prosper
  decodes, not the whole run.
  **Read the scope.** Bucket 200 is byte 52,428,800 = slice **148.8** at `layer_stride=352256`,
  which sits *below* the interior's slices 186-248 (buckets **249.9-333.2**). This probe says
  nothing about the interior's own slices, and the finding that slice 248 reads zero throughout
  stands. An earlier version of this bullet concluded "so 'the guest never fills the upper atlas'
  is false" without that qualification; that conclusion is retracted -- see `## Ruled out`.
  The writer is `eboot+0xebb82..0xebbfa`, an unrolled AVX scatter storing 8x16 B per iteration to
  offsets computed in `%ymm10`.
- **The title reads its textures; they simply do not arrive here.** `PROSPER_READBYTES=1` counts
  bytes delivered per path with no stack walk: **`.DDS open-events=299 paths-read=297`** over a
  route, alongside `.TRM 98/52` and smaller sets. (Those two fields carry DIFFERENT units -- open
  *events* against distinct *paths* read -- and were previously printed as `opened=`/`read=`, which
  invited reading them as a matched pair.) So asset I/O works and 297 texture files are read, while this atlas
  receives about eighteen slices' worth. Whatever moves DDS content into this allocation is doing so
  for a small fraction of what the title loads.

- **Everything on prosper's side of the interior defect is measured and correct; the slices the
  interior samples are genuinely unpopulated in guest memory.** (Scoped 2026-08-27 — see
  `## Ruled out`.) Twelve candidates eliminated, each with its control or
  its stated scope: layer index (traced twice to a constant 248), flat interpolation (#3051, the
  title never sets FLAT_SHADE), mixed-slice triangles (constant per draw), stale decode cache (guest
  memory at the sampled slice probed ~138,000 times, never becomes non-zero), per-slice stride and mip offset (both
  layouts tested), `depth` mis-decode (raw T# confirms 256 / base_array 0), truncated reads
  (`short_reads=0`), partial residency (one `rw-s` mapping covers the whole span), base-address
  mis-decode (dword0 `<< 8` matches), compute STORE (control fired, no compute binds it), recorded
  PM4 write paths (all four recorders armed, zero hits), and an unimplemented
  `sceAgcDriverInitResourceRegistration` (its family is profiler labelling, not memory). The
  remaining work is a forward investigation from the title's asset/streaming logic, not from the
  frame. Full evidence on #2998.

- **The interior's wrong textures are not prosper mis-reading the atlas. The slices the interior
  samples are genuinely empty in GUEST memory for the whole run** — narrowed 2026-08-27 from "the
  atlas is almost empty for the whole run", which is false: the allocation IS filled progressively
  below slice ~149. The claim that survives is about the interior's OWN slice range. Measured, each point with the control that
  makes it mean something:
  - `PROSPER_SLICEMAP=1`: the 512x512x256 world atlas decodes with **slices 0-13 populated and
    14-255 empty** — contiguous, from the full 256-character map. **`short_reads=0`**, so every
    slice was read completely; "empty" here is not "unreadable". (The count is a lower bound: the
    scan samples one byte in 61.)
  - `PROSPER_SLICEWATCH=1` samples GUEST memory at slice 248 on the reuse path, which runs about
    138,000 times per route: it reads **zero on the first reference and never becomes non-zero**.
    So the guest never fills **slice 248**, at any point — which retires the decode-cache
    hypothesis properly. Note the scope: this is one slice inside the interior's range, and is NOT
    evidence about the allocation as a whole (`## Ruled out` records a write at bucket 200).
    An earlier attempt to retire it was **void**: the `PROSPER_NO_TEXTURE_DECODE_CACHE` run never
    left the title screen (its colour count sits at the title's ~210k for all 300 s), so its 233
    re-decodes all happened before the level streamed anything.
  - `PROSPER_COMPUTE_BINDS` over the atlas plus three controls: the one line printed is a **control**
    — the compute program binds a different 4 MiB buffer and executes — so no compute STORE produces
    this atlas.
  - The world shader samples slice **248**, traced twice: `vertexIndex*24 + 3` from binding 7, a byte
    constant across all 4,206 vertices, through `OpBitFieldUExtract(dword, 24, 8)` to the fragment
    varying.
  So the guest samples a slice that nothing ever wrote — that slice, not the whole allocation —
  and prosper reads that correctly. **The
  missing piece is an atlas upload path prosper does not perform**, not the array plumbing, which is
  now measured end to end. Compute is excluded; DMA is untested (a `PROSPER_DMA_WATCH_DST` run was
  silent but had no control, so it is not evidence).

- **`r.depth = 256` is NOT a mis-decode.** The raw descriptor (`PROSPER_TDUMP=512x512`) reads
  `t = 20491cc0 cb600000 007fc07f d0550fac 000000ff 00700050 00000000 00000000` ->
  `512x512 fmt=182 type=13 tile=5 mips=0..5 depth=256 base_array=0`. The layer count, the base slice
  and the mip range are exactly what the decode reports, so the layer mapping is faithful and the
  question is not "how many layers".
- **Nor is it the slice LAYOUT.** With the array treated as fully tight -- stride = one selected-mip
  surface AND no per-slice mip offset, the combination an earlier test got wrong by changing only the
  stride -- the populated count moves 14 -> 19 of 256 and the interior is still wrong. Every layout
  tried leaves the same invariant **at decode time**: only about 5 MB of the atlas's 67 MB is
  non-zero, whichever way the slices are addressed. (67 MB here is the mip-0-only footprint,
  256 x 262,144; the 86 MiB used elsewhere in this doc is the full `layer_stride=352256` span. Both
  are correct for what they measure -- name which one you mean.) That figure is a decode-time
  observation: the allocation IS filled progressively above it, though not, as far as anything has
  measured, in the interior's own slice range.
  So the remaining question is not how prosper reads the atlas but **why the slices the interior
  samples were never written**: the guest declares 256 slices at `0x20491cc000` and samples slice
  248, while the memory there is zero. That points at a content path prosper does not observe (a DMA or compute upload
  into the atlas) rather than at the array plumbing, which is now measured end to end.

- **Missing SPIR-V `Flat` on the varying that carries the array slice is NOT why interiors sample
  the wrong layers.** The gap is real — `is_flat_shaded()` is decoded, used to build the guest's PS
  input control word, and never reaches the SPIR-V (#3051) — but this title never asks for it. A
  live `PROSPER_INTERPLOG=1` route (224,363 `[interp]` lines) shows Location 1, the slot carrying the
  slice, with control words `0x1` / `0x0` / `0x4` and **no control word anywhere in the run setting
  the `0x400` FLAT_SHADE bit**. #3051 stays open as a latent defect with no observed instance.
- **This question cannot be answered offline.** `gpu_replay` on a `.prgcap` emits **zero**
  `[interp]` lines: a capture replays pre-decoded draws, so recompile-time diagnostics never fire
  (instrument trap 229). The falsification above had to be measured on a live route.

> The `PROSPER_*` probes cited below are **not on master**. They live on the unmerged WIP branch
> `wip/issue-325-texture-arrays` (`e1b0fbb2` and later), which exists so these measurements stay
> reproducible. Check that branch out before trying to re-run one.

- **"Memory pressure is why 256 decoded layers fail" — false.** A `PROSPER_ARRAY_MAX_LAYERS` bisect
  over an early array upload rendered at 1 layer and failed identically at **4, 16 and 64**. Four
  layers is trivially small, so the failure was structural, not a working-set problem. The actual
  cause was `backend_texture_plane_span_valid`, which admitted `sample_count > 1` only for the
  four-plane R32_SFLOAT guest-MSAA shape and rejected everything else (#3043).
- **"The world samples its array with a non-array instruction, so it always reads slice 0" — false.**
  A full census over a gameplay route — the whole population, not a sample — found **every** MIMG
  touching an array texture uses `DIM=5`: op `0x2f` on bindings 39-46 at depth 1 (64 events), and
  op `0x20` on bindings 34/36/39/47 (17 events, of which 14 are at depth 256; binding 34's four span
  depths 1/29/32/256). The census is not circular: it selects on the resource's `img_dim` and
  reports the *instruction's* `mimg_dim`, which are independent fields. There is no DIM≠5 case to
  explain the flat world (#2998).
- **The decoded slices are not duplicates of slice 0.** Per-layer checksums (`PROSPER_ARRAYTEX`) give
  13 distinct hashes for one array binding and 8 for another, so the decoder does not replicate the
  base slice (#2998).
- **Forcing the array layer to a constant changes nothing for the plain-SAMPLE path.** A probe
  substituting a constant layer *and printing when it does* fired on all four bindings reached by op
  `0x20` (34/36/39/47), and the gameplay frame was unchanged — identical average- and difference-hash, luma
  differing in one byte of 288. **Scope, because the first version of this line overstated it:** the
  probe sits in the implicit-LOD array sampler only, so it covers the `0x20` events and
  *structurally cannot* cover the 64 `SAMPLE_C_LZ` (`0x2f`) events, which take a different lowering.
  Within that scope the layer coordinate was not the variable; the resource was uploaded with one
  layer (#2998).
- **Binding numbers are not texture identities.** A binding is a per-shader descriptor slot, and the
  decoded-texture cache keys by guest address — so "binding 36 never reaches the array decode gate"
  does not mean its texture is never decoded there. It is decoded once under whatever slot referenced
  it first, then served from cache under every later slot. Chasing a binding number instead of an
  address cost a full measurement cycle (#2998).

- **The byte pattern of a misread 32-bit index buffer is NOT sufficient to identify one.** A genuine
  16-bit buffer with a period-2 pattern — a fan or cone as a triangle strip `[rim, apex, rim, apex, …]`,
  or a line list radiating from one hub — is **byte-identical** to a clustered 32-bit list whenever the
  guest supplies no `DrawIndexOffset`, because the caller then passes the same pointer twice. Two such
  buffers were constructed by hand and both satisfied every byte-pattern clause of the first version of
  `index_buffer_is_unannounced_32bit_high`. **No further test over those two pointers can separate
  them** — do not try to tighten the pattern. The deciding evidence has to come from outside the
  buffer, and today that is the bound vertex buffer's record count. Residual, measured: a bound of
  65,602 records still admits the case; #3009 is the real fix. Found by independent review of PR #3006.

- **The renderer is not rejecting anything, and never was.** A full boot-to-gameplay run produces
  **zero** `[recompile-reject]` lines and **zero** `[compute] skip` lines. Neither the shattered
  geometry (now fixed) nor the missing textures (still open) is an unsupported-op gap — prosper
  executes the title's draws and gets a wrong answer, rather than declining them. (run08, `606fd6ae`.)
- **The vertex data and its descriptors were never wrong.** The world's positions decode cleanly as
  `Sint16 x 4` at stride 20 — quantized on the 1024-unit grid the original games used, spanning a
  sane room-sized box — and the four attribute V#s tile the 20-byte record exactly. The shattering
  was entirely downstream, in the index buffer. Do not re-open the vertex-format path on the strength
  of the geometry looking torn.
- **#305's user-data window mismatch is not involved.** That path skips draws fail-visibly; here
  every draw executes.
- **The EULA hold is not a hang, and not a renderer stall.** A no-input launch sits on one distinct
  frame for its whole run, which reads exactly like a stalled title; it is the game waiting for input
  on page 1 of 40, and only reaching page 40 clears it. Cross, Circle and Options are all inert
  before then.
- **Acceptance is not persisted**, so route desync is not a savedata-staleness effect. No
  `savedata0` directory is created at all and the EULA re-shows identically on every cold start.

## Instrument note, and a trap worth avoiding

**A `.prgbundle` replays PRE-DECODED draws.** An instrument in the index or vertex decode path does
not fire under `gpu_replay --bundle` — measured here: the `[idxtype]` diagnostic printed 0 lines
offline and 508,688 live. The offline bundle is the right tool for "which draw wrote this pixel" and
the wrong tool for anything upstream of the decoded draw, and the two are not distinguishable from
the replay output. `PROSPER_INDEXTYPE_LOG=1` (bounded to 64 lines) prints what the guest announced
beside both readings of its own bytes, and must be run **live**.

## Not yet investigated

- Whether the missing New Game item is caused by the absent savedata, or is independent.
- Why `reach-title-screen.pad` arrives on Tomb Raider I while `reach-gameplay.pad` arrives on
  Tomb Raider II. Each is internally consistent across runs (3 of 3 for the gameplay route), so both
  are reproducible, but the reason they differ is unexplained and no mechanism should be assumed.
- Tomb Raider III beyond its title screen.
