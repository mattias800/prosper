# `libSceVideodec2` — the access-unit video decoder

`sceVideodec2` is a **raw elementary-stream decoder**, and that is the whole reason it needs its own
document alongside `sceAvPlayer`. AvPlayer is handed a container and owns `open -> demux -> decode`;
Videodec2's caller (CRI's `criMvPly`, and the CRI Mana backends built on it) demuxes **itself** and
submits one compressed access unit at a time, expecting at most one picture back per unit.

Both now run on the same host decoder. `prosper::video::VideoBackend` carries two entry points onto
it (`src/hle/video_backend.hpp`):

| shape | entry points | used by |
| --- | --- | --- |
| stream | `open` / `open_memory` / `info` / `next_video` / `seek` / `close` | `sceAvPlayer` |
| access unit | `open_decoder` / `decode_au` / `reset_decoder` / `close_decoder` | `sceVideodec2` |

The Linux implementation is `frontends/video_vaapi/vaapi_backend.cpp`. `decode_au` is
`avcodec_send_packet` / `avcodec_receive_frame`, which is natively access-unit shaped — so this is a
second door onto libavcodec, not a second decoder. VA-API is requested first (its H.264 surfaces are
already NV12, so that path does no colour conversion at all); a host without it falls back to
software decode plus a per-frame YUV420P→NV12 chroma interleave.

## What a title actually submits

Captured live with `PROSPER_VDEC2_CONTRACT=1` — *Tales of Graces f Remastered* (`PPSA19991`), the
first title in this repository's history to reach the decoder lifecycle (#1658, #1687):

```
[vdec-contract] create codec=1 profile=100 max=1920x1088 dpb=-1 input_depth=4
                | mem cpu=65536 gpu=65536 shared=65536 max_frame=3133440 align=256
[vdec-contract] decode#0 au_bytes=3984 frame_buf=3133440 head=00 00 00 01 09 10 00 00 00 01 67 64 00 28
[vdec-contract] decode#1 au_bytes=565  frame_buf=3133440 head=00 00 00 01 09 30 00 00 01 06 01 02 00 80
```

Read off those bytes rather than assumed:

- **Annex-B H.264.** `00 00 00 01` is a 4-byte start code; `09` is an access-unit delimiter, so every
  access unit is self-delimiting. The first one continues into `67 64 00 28` — an SPS with
  `profile_idc = 100` (High) and `level_idc = 40`, agreeing with the `profile=100` the guest asked
  for through `VdecConfig`. Two independent signals for the same conclusion.
- **1088 is 1080 rounded up to an H.264 macroblock row**, which is the coded height, not a typo and
  not a safety margin.
- **The output layout is YUV 4:2:0.** `max_frame_size / (max_width * max_height)` = 3,133,440 /
  2,088,960 = **1.5000 bytes per pixel, exactly**. YUV422 would be 2.0, RGB24 3.0, RGBA8 4.0. This is
  a derivation from bytes the guest wrote, not a plausible inference.
- **Access units are per-frame NALs**, 565–3,984 bytes at the start of the movie and tens of
  kilobytes once it is running.

## Verifying a decoded picture — do not trust your eyes

A picture that "looks like a movie" is not evidence the decode is correct. A wrong chroma plane, a
swapped U/V, a stale reference frame and a dropped access unit all produce something still perfectly
filmic. This project has a recorded trap for exactly this failure (a plausible palette convinced the
eye; a gradient beat real content on a metric).

**H.264 reconstruction is normatively exact**, so the check that cannot be faked is byte equality
against an independent decoder. `PROSPER_VDEC2_DUMP_DIR` exists to make that a two-command job:

```bash
mkdir -p ~/vdec-check
PROSPER_VDEC2_DUMP_DIR=~/vdec-check PROSPER_VDEC2_DUMP_FRAMES=16 \
    PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
    ./build-linux/screenshot <DUMP_ROOT>/<TITLE_ID>-app0 --seconds 15 --count 5

cd ~/vdec-check
ffmpeg -i au.bin -f rawvideo -pix_fmt nv12 ref.nv12   # an independent decode of the GUEST's own bits
cmp ref.nv12 pic.nv12                                  # exit 0 == byte-identical
```

`au.bin` is every access unit the guest submitted, concatenated — for an Annex-B stream that
concatenation is itself a playable elementary stream. `pic.nv12` is the exact bytes prosper wrote
into the guest's frame buffer, including the copy. Both are capped at
`PROSPER_VDEC2_DUMP_FRAMES` (default 16) because a 1920×1088 NV12 picture is 3.1 MB and these titles
feed thousands of them.

Results on `PPSA19991`, 2026-08-17 — three runs, `cmp` exit 0 every time: **16 frames /
50,135,040 bytes**, **146 frames / 457,482,240 bytes**, and **64 frames / 200,540,160 bytes** after
the copy path was rewritten to write straight into the guest's buffer. The regression test
`test_video_vaapi` runs the same comparison offline against a committed 128×96 Annex-B asset whose
per-frame hashes came from a host `ffmpeg` decode, over **both** the hardware and software paths.

## Three lifecycle verbs, three contracts

`libSceVideodec2` exports eighteen functions (PS5 3.20 firmware symbol database), among them
`CreateDecoder`, `Decode`, `Flush`, `Reset` and `DeleteDecoder` as five separate entry points with no
re-initialise call between them. Three of them dispose of decoder state and
they are **not** interchangeable — implementing one as another is #2585, and it is the kind of defect
that shows up as a hang rather than as an error:

| call | what it disposes of | carries a `VdecFrame`/`VdecOutput` |
| --- | --- | --- |
| `sceVideodec2Flush` | **drains** — hands the buffered pictures back | yes |
| `sceVideodec2Reset` | **discards** — throws the buffered state away | no |
| `sceVideodec2DeleteDecoder` | **destroys** — the decoder and the handle are gone | no |

**The warrant for flushing is the dominance argument below — read that first.** Whether the Sony
contract keeps or drops the parsed sequence headers is *not* established (`CONFIDENCE: MED`), and it
does not need to be, because `avcodec_flush_buffers` dominates:

| if the true contract is... | close and reopen (what prosper did) | flush in place (what it does) |
| --- | --- | --- |
| drop the DPB, **keep** the parameter sets | **broken** — the decoder cannot decode until the next in-band SPS | correct |
| drop the DPB **and** the parameter sets | works only if the stream repeats its parameter sets | **no worse either way, and strictly better under row 1**: a stream that repeats them replaces the retained ones in-band by id, and a stream that does not repeat them is one where retaining is the only thing that decodes at all. Under this row retention can make prosper decode where a faithful implementation would not — the decision is unaffected, but the claim is "no worse", not "correct" |

There is no reading of the contract under which closing wins, so the fix does not depend on resolving
the open half.

A weaker supporting claim, deliberately **not** the reason: **Reset must leave the decoder usable**,
derivable from the export list rather than assumed — the library has no re-initialise entry point
between `CreateDecoder` and `DeleteDecoder`, so a Reset that destroyed the decoder would leave the
guest no way to rebuild it except Delete + Create with the whole `VdecConfig`, which is what Delete
is already for. `CONFIDENCE: HIGH` — **but it is not load-bearing**, because the pre-fix code already
satisfied it in the guest-visible sense: the handle stayed valid and the next `Decode` reopened. Cite
the dominance argument as the warrant, not this.

Both halves of what `avcodec_flush_buffers` actually does were **measured against this
build's libavcodec**, not read off its documentation, using the committed asset and a hand-built
positive instance of each case (`test_video_vaapi`, and `VaapiBackend::reset_decoder`'s comment):

- **parameter sets are kept** — a flushed context fed access units 6–11 with every SPS and PPS NAL
  stripped out decodes **6** pictures; a **fresh** context fed the same bytes decodes **0**.
- **the DPB is dropped** — a flushed context fed access units 7–11 (non-IDR slices, no IDR among
  them) decodes **0** pictures; an **unflushed** one decodes against the stale pre-reset references.
  **That control's margin is one picture, not five** (the decoder errors out on the rest once the
  references diverge), independently reproduced against libavcodec 8.1.2 with a separate access-unit
  splitter. The test therefore asserts `> 0` rather than a fixed count — **do not "tighten" it to 5**;
  a fixed count is brittle across libavcodec versions and would fail for a reason unrelated to Reset.

A backend with no in-place reset falls back to close-and-reopen and **says so once**. That forgets too
much rather than too little, which is the right direction to fail: leaving the DPB live would decode
the guest's next access units against references it just asked us to forget, and that is a corrupt
picture rather than an error.

**The decoder writes into the CALLER's buffer, not into its own.** `decode_au` takes a destination
and copies under its own lock, and `AuPicture` carries no plane pointers at all. That is deliberate:
the earlier shape handed back pointers into the decoder's staging buffer that stayed valid only
"until the next `decode_au` on this id", so a concurrent `Reset`/`DeleteDecoder` could free the
source of an in-flight copy. Handing the destination down removes the lifetime question instead of
documenting it — and drops a 3.1 MB per-frame copy on the way.

## Environment variables

| variable | effect |
| --- | --- |
| `PROSPER_VDEC2_CONTRACT` | print what the guest configures and submits (capped at 12 decodes) |
| `PROSPER_VDEC2_DUMP_DIR` | write `au.bin` + `pic.nv12` for the reference comparison above |
| `PROSPER_VDEC2_DUMP_FRAMES` | cap for both dump files (default 16) |
| `PROSPER_VDEC2_NO_DECODE` | restore the pre-#2270 no-picture behaviour, for the A/B |
| `PROSPER_VDEC2_FORMAT` | sweep candidate values for the unestablished `VdecOutput::format` |

## Which titles reach this path

Every executable image in the local corpus scanned for `sceVideodec2Decode`'s NID `852F5+q6+iM`
(44 dumps, 589 `eboot.bin`/`.prx`/`.sprx` files; independently reproduced in #2571's review):

| title | reaches `Decode` | codec | measured under default-on decoding |
| --- | --- | --- | --- |
| *Tales of Graces f Remastered* `PPSA19991` | yes | 1 (AVC High, 1920x1088) | plays its opening movie; 15 distinct samples against 12 identical before |
| *Sonic Origins* `PPSA05325` | yes | 1 (AVC High, 3840x2160) | the pure-white hold is **passed through, not eliminated** — still one flat white frame at 80 s, then 14,871 / 8,978 / 13,351 / 9,294 / 9,372 colours to the end of a 180 s window. #2267 |
| *Sonic Racing: CrossWorlds* `PPSA08804` | yes | 2382845 (VP9) | carried **through** its post-logo wall — another lane's measurement, `SONIC_CROSSWORLDS_STATUS.md` |
| *Dragon Quest VII Reimagined* `PPSA17942` | no — `QueryComputeMemoryInfo` only | — | unaffected |
| *Balan Wonderworld* `PPSA02058` | no record | — | unaffected |
| *Crisis Core –FFVII– Reunion* `PPSA07809` | no record | — | unaffected |

A title that never reaches `Decode` cannot be affected: the decode path is entered only from inside
`sceVideodec2Decode` on a live decoder handle.

**All six also import `sceVideodec2Reset` (`wJXikG6QFN8`), and `sceVideodec2Flush` (`l1hXwscLuCY`).**
Re-measured over the same 44 dumps / 589 executable images, with `sceVideodec2Decode`'s own NID as
the positive control — it returns exactly the six titles above, so the method reproduces a known
answer before being trusted on a new one. Note what this does and does not say: **an import is a
linked symbol, not a call.** CRI's player links the surface it may use. But it does replace #2585's
original framing — "no title is recorded calling `sceVideodec2Reset` at all", which was a fact about
a handler that had no logging rather than about any title — with a measured one: every affected title
links it, and three of the six are recorded reaching `Decode` live.

Seven of `libSceVideodec2`'s eighteen exports are still unregistered: `CreateDecoderBid`,
`CreateHevcDecoder`, `QueryHevcDecoderMemoryInfo`, `GetHevcPictureInfo`, `GetVp9PictureInfo`,
`MapMemory`, `MapDirectMemory`. **Zero titles in the local corpus import any of them** (same scan,
same control), which is why this is recorded rather than urgent — but an unregistered NID answers
`SCE_OK` without writing its out-parameter, so an HEVC title would receive a successful-looking
decoder handle that was never written. It is loud (`prosper_on_unimpl` prints once per NID), so the
gap is recorded rather than silent; it is the return *value* that is wrong. #2630.

**Decoding a movie does not make prosper render it.** On `PPSA05325` every presented sample is
`source=guest_scanout` and **68 of 68** `[rtt] GUEST SCANOUT` lines still report *"no present source
and no renderer target"* — the same figure #2267 measured before any of this. The renderer authors
nothing on that title either way; what changed is that the buffer the guest flips now contains
decoded pictures instead of a cleared page. Do not read a title advancing past a movie as evidence
that its rendering works.

## Open

- **`VdecOutput::format`** is the one field still unestablished, and prosper writes 0. It is a Sony
  enum whose values are not derivable from anything local, and a wrong constant is a
  correctly-decoded picture the guest reads wrongly — silent.

  **The strongest evidence on it is not in this document's own lane**: #2267 records a four-value
  sweep (0/1/2/4) on *Sonic Origins*' intro measuring mean channel values over non-background pixels,
  and all four arms agree to one decimal — so the field does not drive that title's colour path.
  That comment is also worth reading for how it nearly went wrong: the first comparison was frame
  *hashes*, which differ across arms **by construction** when the subject is a playing video, and a
  guaranteed difference was about to be reported as signal.

  Two more titles now composite a byte-verified picture with `format = 0`. Taken together that is
  evidence these guests tolerate 0 and **not** evidence that no guest reads the field — one title's
  null does not generalise, and a value that happens to mean something specific could still change
  behaviour. `PROSPER_VDEC2_FORMAT` sweeps candidates without a rebuild, and a ctest arm pins that
  the value it names reaches the guest's struct, so a sweep's null is about the guest rather than
  about the instrument.
- **`sceVideodec2Flush` does not drain the decoder** — #2562. Note it is also still **unlogged**, so
  "no title calls it" remains a statement about the instrument rather than about any title.
- **`sceVideodec2GetAvcPictureInfo`'s structure** is unestablished — #1658.
- **Which sequence-header disposition Sony specifies for `sceVideodec2Reset`** — `CONFIDENCE: MED`,
  and the fix does not depend on it (see the dominance table above). Resolving it wants a title that
  actually resets; `sceVideodec2Reset` now `svc_log`s, so the next boot that makes one will say so.
- **Windows.** `frontends/video_mf` implements the stream shape only, so Videodec2 titles get the
  honest no-decoder announcement there rather than a picture. #2563.

## Ruled out

- **"Closing the backend decoder is an acceptable way to express `sceVideodec2Reset`, because
  forgetting more than a reset needs to is a safe direction to err."** Falsified by measurement
  against this build's libavcodec: a **fresh** `AVCodecContext` — exactly what close-and-reopen
  produces — decodes **0** pictures from access units whose SPS/PPS have been stripped, while a
  **flushed** one decodes all 6. Over-forgetting is not safe here, because the parsed sequence
  headers are the one piece of state a mid-stream reset may be unable to re-supply: Videodec2's
  caller demuxes itself, so whether parameter sets are repeated in-band is title-dependent. #2585,
  and both arms plus their controls are in `test_video_vaapi`.
- **"No title in this repository's history calls `sceVideodec2Reset`, so the mechanism is bounded."**
  **Void, not false** — and *unfalsifiable in principle*, not merely unobserved. `wJXikG6QFN8` has
  been a **registered** NID since #1368, and `prosper_on_unimpl` fires only for imports with **no**
  registered handler, so a registered handler with no `svc_log` leaves no trace in *any* instrument on
  *any* boot. The claim was about the instrument, not about any title, and it was used as a reason the
  defect was not urgent. `sceVideodec2Reset` now `svc_log`s. #2585.
  **It is not true that the other entry points already did.** Of the eleven registered Videodec2
  handlers, **six log and five do not** — `Flush`, `DeleteDecoder`, `ReleaseComputeQueue`, and both
  `GetPictureInfo` forms are still silent. Said explicitly because an earlier draft of this row
  claimed Reset "now logs like every other Videodec2 entry point", which contradicted this document's
  own open-issues entry for `Flush` (#2562) — and a `## Ruled out` row asserting Flush already logs is
  exactly the sentence that stops the next reader checking.
- **"The two-pass access-unit decode test covers the hardware and the software path."** True as a
  *request* and unverified as an *outcome*: both passes are software on a host with no usable VA-API
  device, which is precisely the headless CI where the claim was being relied on. `AuPicture` now
  carries `hardware`, set from the pixel format a frame actually came back in, and the run states its
  own coverage. #2586.
  **The guarantee is deliberately one-sided, and this was measured, not reasoned about.** On a host
  with VA-API (12 of 12 pictures in the hardware format), forcing `hardware` **true** fails two
  assertions, and forcing it **false** *passes* while printing "this host negotiated NO VA-API
  hardware decode". So the test can prove a pass ran in software and that two passes differed; it
  **cannot** prove a host's hardware path went unexercised — "the flag is stuck false" and "this host
  has no VA-API" are the same observation, and the second is a supported state. A created device does
  not settle it either: libavcodec may negotiate a software format behind an attached VA-API device,
  which is why `open_decoder` logs *"requested"*.
- **"The output `format` enum must be established before decoding can be enabled."** This kept the
  decode path opt-in behind `PROSPER_VDEC2_DECODE` (#2281). **Falsified by measurement, and the
  measurement is what carries it**: on `PPSA19991` decode-off freezes the composite at one CRC for
  the rest of a 150 s run while decode-on plays the opening movie across 15 distinct samples — and
  the picture it composites is byte-identical to an independent decode **while `format` is 0**, which
  is direct evidence that this guest tolerates 0. On `PPSA05325` the same flip moves the title off
  the flat white it had held indefinitely, to 8 distinct 4K samples.
  A supporting symmetry argument exists — with the path off the guest receives `format = 0` *and*
  `pictures = 0`, so the field is no less unestablished either way — but **it is weaker than it
  looks and must not be quoted as the reason**: with `pictures = 0` the field is inert, because no
  guest reads a picture it was never given, and with `pictures = 1` it becomes live. The trade is
  "an inert unknown" against "a live unknown plus a verified image", not "the same wrongness plus an
  image". It resolves the same way; it does so on the evidence. (#2270, corrected by #2571 review N2
  — a true statement reached by a route that does not establish it is this project's recurring
  failure mode, and a `## Ruled out` row is exactly where nobody re-derives one.)
- **"The stream-shaped `VideoBackend` cannot serve Videodec2, so this needs a new decoder or an
  access-unit accumulator."** The framing in #1688 and in #2270's original text. `send_packet` /
  `receive_frame` is *already* an access-unit API and FFmpeg was already linked, so it needed a
  second entry point onto the existing decoder and no new dependency. #2281.
- **"An honest error return is the cheap interim fix."** Measured with a gated experiment arm on
  *Sonic Origins* (`PPSA05325`): returning `VDEC_ERR_PIPE` moves the title from holding white to
  holding black. It changes which stuck state a title sits in; it does not advance one. The
  fail-visible **log** is what was worth landing, not the error. #2270.
