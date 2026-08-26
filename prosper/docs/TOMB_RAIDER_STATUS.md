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
   ([#2998](https://github.com/mattias800/prosper/issues/2998)). The atlas is bound and holds real
   data (84 MB non-zero), nothing is rejected, and the menus are fully textured.
2. **Text is intermittently garbled** ([#2999](https://github.com/mattias800/prosper/issues/2999)) —
   the EULA body and the title-screen game selector draw the wrong glyphs while the same font
   renders headings, numerals and every ring label correctly.
3. **Solid untextured quads on the menus** — small pale quads on the Tomb Raider I and II rings and
   a large violet block over the whole Tomb Raider III ring.
4. **No savedata is written, and the ring menu has no passport item.** No `savedata0` directory
   appears across a full boot, and the five-item ring carries no New Game or Load Game. Whether
   those are one defect or two is unestablished.

## Ruled out

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
  op `0x20` on bindings 34/36/47 (20 events, of which 12 are at depth 256; binding 34's four span
  depths 1/29/32/256). The census is not circular: it selects on the resource's `img_dim` and
  reports the *instruction's* `mimg_dim`, which are independent fields. There is no DIM≠5 case to
  explain the flat world (#2998).
- **The decoded slices are not duplicates of slice 0.** Per-layer checksums (`PROSPER_ARRAYTEX`) give
  13 distinct hashes for one array binding and 8 for another, so the decoder does not replicate the
  base slice (#2998).
- **Forcing the array layer to a constant changes nothing for the plain-SAMPLE path.** A probe
  substituting a constant layer *and printing when it does* fired on the four bindings reached by op
  `0x20`, and the gameplay frame was unchanged — identical average- and difference-hash, luma
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
