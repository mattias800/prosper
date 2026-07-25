# Blue Prince (PPSA25009, Unity/IL2CPP) — status & investigation map

Last revised: 2026-07-25 (the depth/stencil-clear + arena-recovery session). Prior history: #1216
(closed — boot/blank-frame era), #1264 (Day One hold investigation), #1287 (visuals umbrella, open).

## Ladder position

**Rung 3 — gameplay with real GPU draws, sustained.** The scripted fresh-save route plays the
intro cinematic, loads Day One, walks the manor approach, opens the front door, and explores
Mount Holly's entrance hall/vestibule with the footstep HUD advancing. Composition is not yet
hardware-faithful (rung 4 blocked on the #1287 families below).

Reaching the frame loop uses the standard gated switches:

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_IME_AUTOKEY=1 PROSPER_RENDER=1 \
  ./build-linux/screenshot <root>/PPSA25009-app0 --seconds 3 --count 650 --timeout 2000 --no-manifest
```

`PROSPER_IME_AUTOKEY=1` answers the save-name IME; the route needs no pad script. Timing:
intro cinematic ≈ shots 20–70, Day One yard ≈ 90–115, front door/vestibule ≈ 115+ (plain run);
capture-armed runs pace 2.4–4× slower — budget `--count 730` to pass renderer submit 3300.
The `blue-prince-title` snapshot route guards the title screen.

## Landed fixes this arc (evidence on the linked issues/PRs)

- **#1335/#1344** — the Day One frozen/blown frame: ONE stale arena slot folded
  `(PA_SC_SCREEN_SCISSOR_BR, 0)`; degenerate-SCREEN-pair recovery in the scissor combine.
- **#1349/#1350** — `PA_SU_POLY_OFFSET` depth bias decoded + applied (capture v29 retains it;
  `gpu_replay --inspect-only` prints per-draw `bias=`); #1351/#1357 hardening.
- **#1352/#1354** — the per-light shadow-plane destroyer: light-loop rects with
  `DEPTH_CLEAR_ENABLE` + `DB_DEPTH_CONTROL==0` were treated as clears and clobbered the just-
  rendered shadow maps via sampled-bridge recency; clears now act only through the enabled write
  path. Restored real per-texel shadow structure (marble floor, sconce pools, stained-glass
  door detail). Stencil twin: #1355/#1359; initial-value contract: #1361/#1362.
- **#1353/#1363** — shadow casters decoded `(Z_READ=P, Z_WRITE=0)` from another stale arena slot;
  lone-zero DB base halves now recover from their partner (GFX10 equal-bases contract), unifying
  the persistent-DS identities to `(P,P)`.
- **#1364 (open)** — the family-level arena problem behind all of the above: Gen5 indirect
  register arenas interleave non-register records inside the counted window; full anatomy and
  the `[dbbase-clobber]` capture tool are on the issue.

## Open defect families (#1287)

1. **Concentric ring/band moiré** radiating from lights (ceiling medallion, light pools).
   PROVEN NOT: shadow-plane content (#1352 A/B), caster depth bias (forced-bias A/B via the
   patch in `~/bp-1264/force-bias-diagnostic.patch`), texture mips/LOD, sampler filters (ramp +
   cookie decode LINEAR and honored). Draw-step bisection: each additive light draw paints its
   own ring family; the G-buffer beneath is clean. Lives in the light-accumulation/falloff math —
   same family as Dead Cells #781 (which rejected depth/history/sampler/interp). FS-tap analysis
   is hampered in the hall scene by additive blending into a saturated post-grade frame; next
   productive step is a joint #781 dissection on the isolated Dead Cells pass, or a Blue Prince
   capture before the grade flip.
2. **Overbright/white blowout** of lit surfaces (facade patches, wallpaper striping).
3. **Full-scene magenta grade flip** at ~frame 2973 — discrete, possibly game-intent
   (evening→night grading?); needs a post-intro PS5 oracle from the user to judge.
4. **Floating black rectangles** at wall-lamp positions; **RGB-noise ball** center-hall.
   Untriaged. #1334 (MODE=3 resolve freshness, parked branch) may relate to stuck present
   regions.

## Capsule & artifact inventory (`~/bp-1264/`, all run-local addresses)

- `hall.prgcap` — entrance hall, submit 3302, pre-v29 (no bias state), exact live parity
  `41cc4a6524492a6a` at capture time; post-#1352 replay `a56192c29c21c375`. THE workhorse for
  offline light-pass iteration (its light draws carry full scissors).
- `hall3.prgcap` — v29 sibling; **bad lighting oracle**: 1299/1533 draws genuinely
  empty-scissored that frame (per-pass vport closes), so its lighting rides RTT seeds.
  **Always check a capture with `gpu_replay --inspect-only | grep scissor` before trusting it
  for lighting work.**
- `dayone_healthy.prgcap`, `f9grabs/dayone_v24.prgcap`, `bp2.prgtl` — Day One era artifacts
  (#1264/#1335 evidence chain).
- `clobberdump.log` — six full arena arrays around DB-base clobbers (#1364 raw material).
- Route scripts: `caphall*.sh` (capture-armed), `hall1.sh`/`hallfix.sh` (plain 120-shot route).

## Diagnostics cheat-sheet (all env-gated, off by default)

- `PROSPER_DBBASETRACE=1` — logs cx writes to the DB Z/STENCIL base registers with fold path;
  dumps the full array on a within-array clobber (`[dbbase-clobber]`, #1364).
- `PROSPER_SCISSORLOG=1` — empty-combine dumps; `[scissor] degenerate SCREEN pair recovered` and
  `[render_state] lone-zero DB … recovered` fire on the two landed arena recoveries.
- `PROSPER_DSLOG=1` / `PROSPER_DSBRIDGE_LOG=1` — persistent-DS call keys and sampled-bridge
  HIT/miss (how #1352 was found).
- `PROSPER_FS_TAP=DRAW:PC` on a v19+ capsule — visualize a light-FS intermediate offline.
- The #1352-era scissor three-stage traces are parked on branch
  `fix/issue-1335-regindirect-tags`.

## Do-not-redo list

Exonerated with evidence (do not re-chase without new contradictory evidence): present staleness,
capture transposition, texture/descriptor decode for the Day One hold (#1287/#1334/#1335 record);
mips/CBR/checkerboard, shadow acne, caster bias, and sampler filtering for the ring family
(#1287 comment trail, 2026-07-25).
