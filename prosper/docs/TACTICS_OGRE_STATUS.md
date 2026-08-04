# Tactics Ogre: Reborn status (`PPSA03839`)

## Current state

Tactics Ogre: Reborn is at compatibility rung 3. A fresh-save route reaches the native 1920x1080
title menu, completes character setup, advances through the Chapter 1 movie at natural EOF, and
reaches an interactive move command in the first tutorial battle. The route and current screenshots
are indexed by game tracker [#1892](https://github.com/mattias800/prosper/issues/1892).

The startup HEVC movie now presents recognizable, changing decoded imagery through the normal Linux
Vulkan renderer. A direct 16-second `screenshot`-frontend run retained the Square Enix, Algebra
Factory, and Basiscape logos, then showed the movie's `SQUARE ENIX PRESENTS` card, cloth-covered stone,
moving pale ribbon/smoke, and lit figure. It produced 16 source-distinct samples and 14 pixel-distinct
samples, with at most two seconds of pixel staleness. The representative checked-in movie frame is
`assets/screenshots/tactics-ogre-hevc-movie.png`. This resolves the flat gray/green movie defect
tracked in [#1903](https://github.com/mattias800/prosper/issues/1903).

Gameplay is not visually complete. Several character sprites remain flat silhouettes, some unit
cards have solid-colour backings, and the lower-left HUD contains an opaque black block and unrelated
red diamond. Those defects remain tracked in [#1913](https://github.com/mattias800/prosper/issues/1913).

## HEVC presentation contract

The title's AGC DMA builder uses two distinct source forms. `sourceKind=0` selects the immediate or
offset in argument 1; `sourceKind=2` selects the 64-bit address in stack argument 8. The movie upload
uses the latter for each decoded NV12 row. Keeping this distinction in the emitted packet preserves
malformed address forms for fail-visible executor rejection rather than silently turning them into
immediate fills.

The destination NV12 allocations are committed direct memory mapped CPU-read-only. A GPU device write
is legal there even though a CPU store through the guest VA is not. The command processor therefore
accepts such a destination only when the authoritative kernel mapping table proves that the complete
range belongs to one committed direct mapping. The kernel helper writes through the shared physical
backing while holding the mapping topology stable, retains the guest VA's CPU protection, invalidates
physical aliases after releasing the topology lock, and rejects private, untracked, malformed, or
cross-boundary destinations.

The presented planes use 1920 visible bytes inside a 2048-byte physical row. Luma is R8 and chroma is
interleaved RG8 with the title's `(X,X,X,Y)` descriptor swizzle. The live renderer preserves the exact
physical pitch provenance and strips row padding during upload; capture replay recognizes the same
narrow contract from matching 2:1 plane extents, pitch, swizzle, and adjacent or 64-KiB-aligned plane
placement.

## Reproduction

Use an FFmpeg installation with an HEVC decoder and the normal full-cadence renderer:

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_RENDER=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA03839-app0 \
  --seconds 1 --count 16 --timeout 25 --out ~/tactics-hevc
```

The movie begins after the static startup logos. Do not use `PROSPER_RENDER_SCALE`,
`PROSPER_RENDER_EVERY`, or a forced guest-state diagnostic for acceptance evidence.

## Verification

- `avplayer_hle`, `agc_dcb_pm4`, and `eop_write` pass in the Vulkan-enabled Linux build.
- The DMA tests exercise address and immediate forms into two CPU-read-only aliases of one direct
  allocation, require exact bytes through both aliases, and require the VAs to remain read-only.
- A destructive `destination = source + 4` copy through two different VAs of the same physical
  direct-memory range requires the prior memmove result. The backing helper resolves this hidden
  physical overlap, stages bounded chunks, and copies in physical address order on POSIX and Windows.
- Private anonymous read-only memory and a direct range crossing its mapping boundary both reject
  without changing bytes or publishing a renderer write notification.
- A defect-shaped mutation that removes direct-backing eligibility makes exactly the three protected
  direct-memory copy, fill, and physical-alias-overlap checks fail while the private and
  cross-boundary rejection checks remain green.
- The valid live run logged successful protected-backing DMA copy ordinals from `#1` through at least
  `#2048`, with no invalid DMA row, and the direct frames changed continuously through the movie.

## Ruled out

- **The HEVC decoder is absent.** A codec-enabled FFmpeg build selects libavcodec after the expected
  VAAPI initialization failure, and focused decoding produces distinct NV12 frames with monotonic
  timestamps. Decoder packaging remains separately tracked by #1784.
- **The title movie is a flat or malformed source asset.** Offline FFmpeg decoding of
  `Data/movie/title_en-us.pmf` produces ordinary changing 1920x1080 HEVC imagery.
- **AvPlayer publishes a stalled or corrupted host frame.** In a bounded live discriminator, every
  sampled decoded input hash matched the corresponding staged guest-frame hash, and PTS advanced.
- **The renderer alone created the old dark diagonal field.** Raw Y/UV bytes extracted from the final
  captured resource already contained that field; the compositor faithfully exposed stale destination
  data. Physical pitch still matters for correct rows, but it was not the reason those bytes stayed
  stale.
- **The movie upload uses argument 1 as its address source.** The live call has `sourceKind=2`, a zero
  argument 1, and the changing decoded row address in stack argument 8. Encoding argument 1 turned the
  upload into immediate-zero fills.
- **A whole retained-DMA submit must reject whenever any later packet reads guest memory.** Tactics
  Ogre's later waits and indirect-register arrays are physically disjoint from the 1,620 movie-row
  destinations. Authoritative VA-to-physical topology proves the safe case; same-physical aliases and
  unknown topology remain fail-closed.
- **CPU read-only protection forbids the device write.** The destination is readable but not CPU
  writable, while its complete range is a committed shared direct-memory mapping. Writing the physical
  backing preserves CPU protection and restores the exact decoded movie.
