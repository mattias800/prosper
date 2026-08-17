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
| access unit | `open_decoder` / `decode_au` / `close_decoder` | `sceVideodec2` |

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
| *Sonic Origins* `PPSA05325` | yes | 1 (AVC High, 3840x2160) | moves off the flat white it held indefinitely; 8 distinct 4K samples |
| *Sonic Racing: CrossWorlds* `PPSA08804` | yes | 2382845 (VP9) | carried **through** its post-logo wall — another lane's measurement, `SONIC_CROSSWORLDS_STATUS.md` |
| *Dragon Quest VII Reimagined* `PPSA17942` | no — `QueryComputeMemoryInfo` only | — | unaffected |
| *Balan Wonderworld* `PPSA02058` | no record | — | unaffected |
| *Crisis Core –FFVII– Reunion* `PPSA07809` | no record | — | unaffected |

A title that never reaches `Decode` cannot be affected: the decode path is entered only from inside
`sceVideodec2Decode` on a live decoder handle.

## Open

- **`VdecOutput::format`** is the one field still unestablished, and prosper writes 0. It is a Sony
  enum whose values are not derivable from anything local, and a wrong constant is a
  correctly-decoded picture the guest reads wrongly — silent. Two titles now composite a
  byte-verified picture with `format = 0`, which is evidence they tolerate it and **not** evidence
  that no guest reads the field. `PROSPER_VDEC2_FORMAT` sweeps candidates without a rebuild, and a
  ctest arm pins that the value it names reaches the guest's struct — so a sweep's null is about the
  guest rather than about the instrument.
- **`sceVideodec2Flush` does not drain the decoder** — #2562.
- **`sceVideodec2Reset` closes and reopens instead of `avcodec_flush_buffers`**, which discards the
  parsed SPS/PPS as well as the DPB — #2585. `CONFIDENCE: MED` on the current mechanism.
- **`sceVideodec2GetAvcPictureInfo`'s structure** is unestablished — #1658.
- **The two-pass decode test cannot prove it ran two different paths** — #2586.
- **Windows.** `frontends/video_mf` implements the stream shape only, so Videodec2 titles get the
  honest no-decoder announcement there rather than a picture. #2563.

## Ruled out

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
