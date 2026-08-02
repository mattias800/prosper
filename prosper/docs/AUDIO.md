# Audio (`libSceAudioOut`)

prosper implements the PS5 `sceAudioOut*` API as a **headless, backend-agnostic core** with
**pluggable frontends**. The HLE never depends on any audio library — it decodes the guest's calls
into a port lifecycle plus interleaved PCM "grains" and forwards them to an installed `AudioSink`.
This keeps `prosper_core` dependency-free and unit-testable, and lets any frontend (SDL3, a file
recorder, a network sink, …) be swapped in at runtime.

## Layers

| Layer | File | In `prosper_core`? |
|---|---|---|
| Backend interface | `src/hle/audio.hpp` (`AudioSink`, `AudioPortInfo`, format decode) | yes (header) |
| HLE + default sink | `src/hle/hle_audio.cpp` | yes |
| SDL3 frontend (optional) | `frontends/audio_sdl3/` | **no** (separate target) |

### HLE coverage (`hle_audio.cpp`)
`sceAudioOutInit`, `sceAudioOutOpen`, `sceAudioOutOutput`, `sceAudioOutOutputs`,
`sceAudioOutSetVolume`, `sceAudioOutClose`, `sceAudioOutGetPortState`. A 16-port table tracks each
open port's `{freq, channels, format, grain}`. `sceAudioOutOpen`'s `param` is decoded into channels
+ sample format (`SceAudioOutParamFormat` 0–7: S16/F32 × mono/stereo/8ch). PCM pointers are read
directly (guest memory is 1:1-mapped), so grains are forwarded to the sink with **zero copies** in
the HLE.

### Default backend — silent, real-time paced
With no frontend installed, `prosper_core` uses a built-in `RealtimeSilentSink`. On real hardware,
`sceAudioOutOutput` **blocks until the audio ring has room**, which is what paces a game's audio
thread. The default sink reproduces that pacing by sleeping each grain's wall-clock duration
(`grain / freq` seconds), so the guest advances at the correct speed with **no device attached** —
correct timing, no sound. This is the right default for headless bring-up.

### Installing a frontend
```cpp
#include "hle/audio.hpp"
prosper::audio_set_sink(&my_sink);   // AudioSink*; pass nullptr to restore the silent default
```

## SDL3 frontend (optional)

Builds a real audio backend that plays the guest's output on the host's default device.

```sh
cmake -S prosper -B build -DPROSPER_AUDIO_SDL3=ON   # uses system SDL3 if present, else fetches release-3.2.30
```
- Uses a system SDL3 (`find_package(SDL3)`) if available, otherwise fetches and builds an
  **audio-only** SDL3 (video/render/joystick/etc. disabled — no X11/Wayland needed).
- One `SDL_AudioStream` per PS5 port; `output()` pushes PCM and blocks while the device queue is full
  (matching hardware pacing). Per-channel volumes are mapped to the stream gain.
- When enabled, `boot_trace` calls `install_sdl3_audio_sink()` at startup automatically.

## Diagnosing a silent title — `PROSPER_AUDIO_FLOW=1`

"No audio" has three mutually exclusive causes that are **indistinguishable from outside** and need
completely different fixes:

1. the guest never submits PCM — the defect is upstream (voice/mixer setup, or an HLE return that
   makes the title's audio engine give up);
2. the guest submits PCM but it is all zero — the defect is in the title's own mixing/decode;
3. the guest submits real PCM and prosper loses it — the defect is ours (routing, sink, volume).

`PROSPER_AUDIO_FLOW=1` reports **unconditionally**, once per second per AudioOut2 context, from both
`ContextAdvance` and `ContextPush`, so each branch has its own signature:

```text
[audio-flow] ctx0 created: grain=256 queue_depth=2 (host sink port 17)
[audio-flow] port1 created: type=0x0 data_format=0xc00 flags=0x0 (MAIN -> mixed to host)
[audio-flow] ctx0 advance=188 push=188 (with-pcm=0 silent-paced=188 not-ready=0) sink=never-opened \
             port=17 grains=0 bytes=0 bed-peak=0.00000 bed-rms=0.00000 bed-nonzero=0/96256 \
             bed-nan=0 | BED LIFE: nonzero=0/3944448 peak=0.00000 rms=0.00000 nan=0 \
             | lifetime: ctx=2 ports=2 pcm-published=15039
[audio-flow]   port1 type=0x0 fmt=0xc00 reads=188 mixed=0 frames=48128 bytes=2310144 \
               peak=0.15349 rms=0.01289 nonzero=86528/577536 nan=0 no-pcm=0 skip-fmt=188 \
               skip-not-main=0 short=0 \
               | LIFE: nonzero=1782048/18456576 peak=0.38976 rms=0.01033 nan=0
```

That is a real *Dragon Quest VII* capture from **before #1700**, and the port line alone diagnoses the
title: the guest is handing us 2,310,144 bytes per second of genuine audio (`LIFE:` run peak
**0.38976**, 1.78 M non-zero samples, `nan=0` ruling out a NaN artifact), and `mixed=0` with
`skip-fmt=188` says prosper discards every grain of it because the port declares 12 channels. Its
context correspondingly shows `sink=never-opened` and `BED LIFE: nonzero=0/3944448` — nothing from it
ever reaches the device. It is kept here because it is the clearest example of the signature; on
current master the same route reports `mixed=188 skip-fmt=0 sink=open`.

Note how little the interval columns would tell you on their own: this line's `peak=0.15349` happens
to be a quiet moment, and other intervals in the same run read `nonzero=0/577536` outright. The
verdict is in `LIFE:`.

| Observation | Meaning |
|---|---|
| no `[audio-flow]` lines at all | the guest never created an AudioOut2 context |
| `ctx…created` but no interval lines | a context exists, but the pump loop never runs |
| `advance=N push=0` | the pump runs but never submits — **case 1** |
| `push=N`, **`BED LIFE: nonzero=0/M`** | real submissions carrying exact silence — **case 2** |
| `push=N`, **`BED LIFE: nonzero>0`** | signal reaches the host sink — **case 3**, look at the sink/volume |
| any port with `LIFE: nonzero>0` but `mixed=0` | **prosper is discarding real audio** — read `skip-fmt` / `skip-not-main` to see why |

The last row is the one that matters most and is invisible in the bed alone: a port can carry the
title's entire soundtrack and never reach the mix. That is what *Dragon Quest VII* does (#1692).

Read the counters, not just presence:

- **Reached-ness** (`lifetime: ctx= ports= pcm-published=`) makes a null reading a *finding*: zero
  submissions means something different when the guest never built an audio graph
  (`ports=0`), built one but never handed over a buffer (`pcm-published=0`), or wired everything up
  and still never pushed.
- **`nonzero=N/M` is the measure that decides silent vs quiet.** Peak is raised by a single stray
  sample and RMS needs a threshold that will call some genuinely-mixed quiet passage silent; only an
  exact `nonzero=0/M` proves the guest submitted a cleared buffer. The rule lives in
  `AudioSignalStats` (`src/hle/audio.hpp`) and is unit-tested against both alternatives.
- **Judge a port by its `LIFE:` totals, never by one interval.** Per-interval counters reset every
  second and real playback has gaps, so any single line can read `nonzero=0/M` on a port carrying
  strong signal over the run. The never-reset totals are what make one line sufficient — reading
  only the last interval inverted this investigation's first conclusion (instrument-trap 39).
- **`BED LIFE:` and a port's `LIFE:` do not share a denominator.** They start at different moments —
  the context's lifetime begins at ContextCreate, each port's at PortCreate — so the two adjacent
  `nonzero=N/M` values on consecutive lines are *not* directly comparable, and in the example above
  they cover roughly 41 and 32 seconds respectively. Compare each against itself over time, or
  against its own `samples`; never read one as a fraction of the other.
- **`nan=N` names the one remaining input class.** NaN is neither silence nor signal: the sink
  clamps it to zero, so a NaN-producing guest is audibly silent while its buffers are "full". The
  probe counts NaN separately on both port and bed lines and then treats it as the silence it
  becomes, so `nonzero>0 nan>0` is a decode bug, not a working mix.
- **Per-port attribution** locates loss the mixed bed hides: `skip-not-main` (a port with signal
  discarded for not being MAIN), `skip-fmt` (unsupported layout), `no-pcm` (no buffer published),
  `short` (a partial/faulting guest read).
- **`skip-fmt` now means "prosper will not MIX this port"**, and since #1700 that is a much smaller
  set: every channel count from 1 to 16 is folded, so a `skip-fmt` port is one whose `data_format`
  low bits are neither `0` (f32) nor `1` (s16), or whose declared width exceeds 16. A width-rejected
  port is **not** clamped and mixed — reading a 20-channel grain at a 16-channel stride would walk
  the guest's buffer and mix garbage — but it **is** still read and measured under this probe,
  because a rejected port is exactly where lost audio hides and that is how #1700 was found. Its
  declared count gives a well-defined stride, so the measurement is sound even where the placement
  is not. Only an unknown sample type is genuinely unsizable; that arm records the skip alone and
  its signal columns stay zero. Either way the reject is no longer silent: the mix loop prints
  `[audio2] portN DISCARDED: …` unconditionally, rate-limited per port. Implement the format; do not
  leave it skipped.
- Treat a **rejected** port's measurement as slightly lower confidence than a mixed port's: the read
  length comes from a layout prosper does not support, so an over-read past the guest's buffer would
  be attributed to that port. It is fault-safe and cannot invent silence, but it could in principle
  inflate `nonzero` — confirm a surprising result against the producing code before acting on it.
- **`peak=inf` / `rms=inf` is a correct reading, not a probe bug.** Infinities are deliberately not
  filtered the way NaN is: the output path clamps `+Inf` to `+1.0`, so an Inf-producing guest is
  genuinely, loudly audible and belongs in the signal statistics. Because `LIFE:` never resets, one
  Inf pins the run peak and rms for the rest of the session — read the per-interval columns to see
  where it came from, and treat it as a guest-side mixing defect worth its own investigation.

## Multichannel MAIN beds and the stereo fold — `PROSPER_AUDIO_LAYOUT=1`

prosper's host sink is stereo, so a MAIN port wider than two channels has to be folded down. The
fold is a pure function, `audio_stereo_downmix(channels, out, capacity)` in `src/hle/audio.hpp`,
returning one `{left, right}` gain per **source** channel plus a count of channels it could **not**
place. The mix loop does nothing but apply it, so the whole correctness surface of multichannel
output is unit-tested in `test_audio` without a device or a guest.

| channels | placement |
|---|---|
| 1 | mono to both sides |
| 2 | identity |
| 3 | FL FR FC (centre -3 dB to both) |
| 4 | quad: FL FR SL SR |
| 5 | FL FR FC SL SR |
| 6 | 5.1: FL FR FC LFE(-6 dB, both) SL SR |
| 7 | 6.1: adds a rear centre, -3 dB halved across both sides |
| 8 | 7.1: adds a second surround pair |
| 9..16 | first eight as 7.1; the remaining height tier is **unplaced** and reported |

A fold is only correct if you know each source channel's **side**. `data_format` carries a channel
count and nothing else, so the order is not derivable from it — and a wrong order sounds plausible,
which is worse than silence because it reads as working. `PROSPER_AUDIO_LAYOUT=1` measures the
guest's own PCM instead of assuming an enumeration, reporting every five seconds per port:

```text
[audio-layout] port1 ctx0 type=0x0 fmt=0xc00 channels=12 grain=256 frames=6856704 \
               stride=n/a (single grain buffer 0x301e393a88, 26784 publications)
[audio-layout]   ch0  rms=0.028960 peak=0.389781 nonzero= 89.8% hf=0.1210 corr= +1.00 +0.46 …
```

- **`rms` / `peak` / `nonzero%`** — which channels carry the bed *at all*. A height channel a title
  never uses is then measurably absent rather than assumed quiet. Read `nonzero%` alongside `rms`:
  the two are not the same finding, and a formatted `rms=0.000000` covers both an exactly-zero
  channel and a ~1e-9 residue.
- **`hf` = RMS(x[n]−x[n−1]) / RMS(x)** — spectral tilt. An LFE feed is low-passed near 120 Hz, so at
  48 kHz it reads ~0.016 while full-band content reads ~0.3..1.5. This identifies LFE from content
  alone, at any index.
- **`corr`** — the full normalized cross-correlation matrix. A front pair is strongly but not
  perfectly correlated, a duplicated channel reads +1.00, an independent channel reads ~0. This is
  what sorts the channels into a left GROUP and a right GROUP, which is most of what a stereo fold
  needs. It cannot say WHICH group is the left one — correlation is symmetric under a swap — so that
  last step rests on the index-0 = FrontLeft convention noted below.
- **`stride`** — the smallest distance between two distinct grain pointers the guest publishes. A
  double-buffering title makes this one grain's byte size, an **independent** measure of the channel
  count rather than a restatement of the `data_format` decode. A single-buffer title reports
  `stride=n/a` with the pointer and publication count, which is a finding, not a missing value.

`PROSPER_AUDIO_LAYOUT_DUMP=PATH` additionally appends each measured port's raw interleaved grain to
`PATH.portN.<channels>ch.f32` as float32 (both sample types converted), so the measurement can be re-derived,
analysed differently, or rendered to something audible offline.

**The channel count is confirmed from the guest's own code, independently of prosper's decode.**
`data_format` bits 8..15 being a channel count is not just prosper's reading of it: Dragon Quest
VII's own format selector (eboot `0x605b02a` in a flattened image, adjacent to the CRIWARE wrapper
that owns the `E2020070207:sceAudioOut2PortCreate() is failed` string) is a four-armed switch that
writes `data_format` and a channel count into parallel fields — `0x100`/1, `0x200`/2, `0x800`/8,
and `0xc00`/**12**. That is the cross-check instrument-trap 43 asks for: a measurement whose value
does not come from the decode being tested. Note what it does NOT establish — the *order* of those
12 channels appears nowhere in it.

**What this established for *Dragon Quest VII Reimagined* (#1700).** Over 19,962,368 frames of its
12-channel MAIN port: all content is in ch0/ch1 (rms 3.1e-2 / 4.3e-2, peak 0.407, correlation
+0.46 — a real decorrelated stereo pair); ch2 and ch4..ch7 hold a ~1e-9 residue; **ch3 and
ch8..ch11 are exactly zero**. The title writes a stereo mix into a 12-channel container, never
sends LFE, and never writes the height tier. Within the residue, ch4/ch6 correlate +0.96 and
ch5/ch7 +0.95 while ch4/ch5 is −0.04, so the surround tier pairs even=left / odd=right, and ch2
correlates near-equally with both groups — corroborating the 6..8 mapping above from live content
(CONFIDENCE: MED; the level is far below anything audible). **Channels 8..15 have no measured side
anywhere yet**, so they are deliberately left unplaced and logged rather than folded on a guess;
`sceAudioOut2PortCreate`'s `portParam` carries no layout field (`{u16 type, u16 pad, u32
dataFormat, u32 sampleRate, u32 flags, u64 userHandle}` — live-captured), and this title never
calls `sceAudioOut2GetSpeakerInfo`, so nothing in the guest's own setup names the order.

**The wide-bed fold has NOT yet been shown to generalise past Dragon Quest VII — one first-party
cross-check came back negative.** *Astro Bot* (`PPSA21564`) was the obvious candidate: a Sony
first-party title, and audible, so if it depended on #1700 that would be the first evidence the fix
matters beyond the title it was found on. `PROSPER_AUDIO_FLOW=1` over a routed boot says it does
not. Astro creates one context with 23 ports, exactly one of them MAIN, and that port is
`data_format=0x800` — **8** channels, not 12. The pre-#1700 gate rejected `channels > 8`, so an
8-channel bed was always accepted and **Astro Bot was already audible before `75e4417f`**. Its
lifetime counters confirm nothing is being lost on the way: `skip-fmt=0 skip-not-main=0 no-pcm=0
short=0`, port1 lifetime peak 0.27977 / rms 0.02749 with 63.6 M of 76.1 M samples non-zero, and the
mixed bed reaching the host sink at peak 0.30449 / rms 0.0545. The 22 non-MAIN ports are `0x100`
(1 ch) apart from one `0x200` and one `0x800`, and are correctly not mixed. So the 12-channel
container remains a Dragon Quest VII observation; do not assume the next silent title has one, and
read its `data_format` before designing for it.

**A sustained left/right level imbalance in the output is CONTENT, not the fold — measured, so it
does not have to be re-investigated.** Dragon Quest VII's mixed bed runs about 38 % louder on the
right across a whole capture, which is large enough for a music bed to look like an unequal gain or
a mis-sided channel, and small enough to sound like ordinary mixing — exactly the class of defect a
listening test cannot catch. One run capturing both the raw port
(`PROSPER_AUDIO_LAYOUT_DUMP`) and the host sink (`PROSPER_AUDIO_DUMP`) settles it, because the two
imbalances can be compared directly:

| | ratio | |
|---|---|---|
| source `rms(ch1)/rms(ch0)` | **1.3842** | 2.5999e-2 -> 3.5989e-2 |
| output `rms(R)/rms(L)` | **1.3842** | 2.6002e-2 -> 3.5991e-2 |
| ratio of ratios | **1.00000** | the fold adds no imbalance of its own |

Stronger than the ratio, and the check to prefer: applying the fold table above to the raw capture
**offline** reproduces prosper's actual host output to within **2.7e-14** peak absolute difference
over 5,240,320 frames, with zero frames differing by more than 1e-6. The mix loop is doing exactly
the documented matrix on the guest's own samples, so any asymmetry in the result came in with the
content. Note what this does *not* show: it proves the fold is internally faithful, **not** that
prosper's left is the game's intended left — that is the orientation question below. It does make
that question sharper, because a bed this asymmetric would make a swapped mapping measurable against
a hardware reference rather than a matter of taste.

**What no amount of this evidence can settle, and what would.** The probe **groups** channels by
side; it does not **orient** the groups, and those are different claims. Correlation is symmetric,
so it proves ch0/ch1 are a genuine front pair — and that {ch4,ch6} and {ch5,ch7} are the two
surround groups — but it cannot distinguish any of those from a left/right **swap**.

> **`CONFIDENCE: MED` for the left/right orientation of every pair in the fold table above, and its
> basis is convention, not evidence.** Index 0 = FrontLeft is universal across published
> multichannel bed layouts, and prosper's own v1 `sceAudioOut` path assumes the same. **Two**
> converging
> conventions earn MED rather than LOW — but a convention is not a measurement, and this one has
> never been tested against a PS5 title. Note precisely where the gap is: `test_audio` DOES pin
> index -> side for widths 6, 7 and 8, so a build that swapped prosper's own fold fails in CI.
> What is unconfirmed is whether that mapping matches hardware.
>
> **A third convention was checked and rejected, and the rejection is the useful part.** CRI
> Atom's `EAtomSpeakerID` names are present in this eboot and order FrontLeft first, which is
> a different kind of claim from "the wider world does this": if the title mixed its bed
> through Atom, the bed's order would be Atom's order, and that is a property of the code
> path rather than an inherited assumption. It does not hold. `tools/re/xref.py` reports
> **zero code references** to that string and one data-pointer relocation into a UE
> reflection table — the names are Blueprint metadata that survived into the image, and
> nothing calls them. A symbol in a binary is not a code path, and three conventions
> agreeing is not corroboration when they may share one ancestor. A mono or stereo bed has no orientation to get wrong, so
> 1..2 channels remain `CONFIDENCE: HIGH`. **A listening test does
not settle a fold-down either** — and for this title it cannot settle anything about the layout,
because ten of its twelve channels are empty, so *every* mapping that routes ch0 and ch1 to the two
sides produces a host bed differing only by the ~1e-9 residue on ch2 and ch4..ch7 — roughly 150 dB
below the content. Deliberately not called "bit-identical": ch3 and ch8..ch11 are exactly zero and
the others are a residue, and this page's whole point is that those are different findings, so the
strongest claim on it must not be the one place they get re-merged. A human confirming "it sounds
right" therefore establishes
that real audio reaches the sink at sane levels through the guest's own path (bring-up rung 4), and
nothing about the channel order. Broadly stereo music folds down plausibly under a swap, which is
exactly why it cannot arbitrate one. **Do not cite a listening confirmation as evidence for the
mapping.** The discriminating evidence is content with **distinct per-channel placement** — a
hard-panned effect whose true side is known from the game, or centre-channel dialogue — measured
with this probe; it settles orientation in one run. That is
the experiment #1720 asks for, and it is why a plausible-sounding wrong order is more dangerous than
a reported gap.

Related, narrower probes: `PROSPER_AUDIOLOG=1` (legacy `sceAudioOut` path: per-port calls, bytes,
peak and RMS), `PROSPER_AUDIO_DUMP=PATH` (raw `PATH.portN.raw` PCM for offline inspection),
`PROSPER_AUDIO2LOG=1` (full AudioOut2 call trace with hexdumps), `PROSPER_AUDIO2_PROBE=1` (per-port
PCM sampling, signal-bearing ports only) and `PROSPER_AUDIO2_CONTROL_PROBE=1` (control surface).
Note that the first two fall silent in cases 1 **and** 2, which is why `PROSPER_AUDIO_FLOW` exists.

## Tests (`ctest`)
- **`audio_hle`** (`tests/test_audio.cpp`) — always built. Drives the real HLE entrypoints through the
  dispatch table with a capturing sink: format decode (all 8 formats + unknown), port open/close,
  byte-exact PCM forwarding, `Output`/`Outputs`, volume, port state, exhaustion, and error paths.
- **`audio_sdl3`** (`frontends/audio_sdl3/test_audio_sdl3.cpp`) — built with `-DPROSPER_AUDIO_SDL3=ON`.
  End-to-end through the real SDL3 backend under the SDL **dummy** audio driver, so it runs on
  headless CI with no sound hardware.

`test_audio` also covers `AudioSignalStats`, the silent-vs-quiet decision rule behind
`PROSPER_AUDIO_FLOW`, including the quiet-but-real case that peak and RMS alone cannot separate
from silence.
