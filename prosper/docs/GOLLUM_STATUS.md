# The Lord of the Rings: Gollum (`PPSA06367`) — status and evidence

Unreal Engine 4. **Rung 0** as of 2026-08-22. The title boots, links every module, reaches the live
renderer and issues real GPU work, but **no frame with content has ever been observed**: it dies in
its startup movie roughly five seconds in, before anything but a flat clear reaches the screen.

First brought up 2026-08-22 from `b5aeefbf`. No prior work, no tracker, no `COMPATIBILITY.md` row
existed before that date.

## Where it stands

A default launch gets a long way in a short time and then stops hard:

- Every module links, including the 32 Wwise plug-in `.prx`es under `prx/`.
- The AGC path is live: `[agc] register defaults requested for SDK version 10`, DCC fast-clears at
  **2560x1440**, real draws, `seed-skip-verify` full coverage on a long list of shader codes.
- The live renderer composites and presents. At t=5.0 s the one sample a run has ever captured is
  `2560x1440`, `source=composited`, `distinct_rgb_colors=1`, every pixel non-black — a **flat white
  frame**. That is a clear, not content, so it does **not** count as rung 1.
- Then `ElectraPlayer::` — Unreal's media player, decoding `tom/content/movies/nacon.mp4` — takes a
  worker-thread fault and `_exit(90)` ends the process.

The run bound is not the limit; the crash is. Every arm so far has ended the same way at the same
point.

## The blocker: `sceVideodec2GetPictureInfo` — FIXED as far as the false-success defect; one ownership question remains

**2026-08-23: the structure is now filled from the title's own bitstream.** `sceVideodec2GetPictureInfo`
(`NtXRa3dRzU0`) writes the SPS/VUI-derived field set (crop flag + quad, aspect-ratio idc +
SAR, timing pair) parsed from the access unit the guest itself submitted at Decode time,
plus a per-picture record pointer at +0x20. The original SIGSEGV at `eboot+0xec6394`
(deref of the never-written +0x20) is gone and Electra proceeds through its whole
picture-info consumer. What remains is below.

`sceVideodec2GetPictureInfo` (`NtXRa3dRzU0`) is registered, and its handler
(`src/hle/service/hle_service.cpp`, `vdec_picture_info`) validates `a0->size` and returns `0`
**without writing the caller's picture-info structure**. The guest then reads a pointer out of the
block it zeroed before the call, and dereferences it:

```text
eboot+0xec4b07   lea  r12,[rsp+0x110]            ; &pictureInfo
eboot+0xec4b36   mov  QWORD PTR [rsp+0x110],0x78 ; pictureInfo.size = 0x78 (120 bytes)
eboot+0xec4b42   call sceVideodec2GetPictureInfo(&outputInfo, &pictureInfo, 0, 0)
eboot+0xec4b47   mov  r13,QWORD PTR [rsp+0x130]  ; pictureInfo + 0x20  -- a POINTER
eboot+0xec4b5b   call 0xec5960                   ; (..., &pictureInfo, pictureInfo[+0x20])
  eboot+0xec6394   mov  rsi,QWORD PTR [r12]      ; r12 == 0  ->  SIGSEGV at (nil)
```

`[prosper] WORKER-THREAD FAULT: sig=11 addr=(nil) rip=0x410ec6394 … thread='ElectraPlayer::'`.

This is the *same defect class* as the audio one below — a query answered `SCE_OK` with its
out-parameters untouched — but the fix is not available: the picture-info layout is not derivable
from this title's bytes beyond the fields it happens to read.

### What this title establishes about the layout (the evidence #1658 asked for)

[#1658](https://github.com/mattias800/prosper/issues/1658) closed asking for exactly this: "a title
that calls it, traced with the struct it passes and the `size` field it sets." `PPSA06367` is that
title. Measured on `b5aeefbf`:

- The structure is **0x78 (120) bytes**, and its **first field is a `u64` size** set to `0x78`
  (`mov QWORD PTR [rsp+0x110],0x78`). The guest zeroes `+0x08 .. +0x77` before the call, so every
  field is the library's to write.
- The live call is `a0=<stack> a1=<stack> a2=0 a3=0`, i.e. `(outputInfo, pictureInfo, 0, 0)`.
- Fields the guest reads afterwards, with `r13 = &pictureInfo` in the consumer at `0xec5960`:

  | offset | width | read at | reads as |
  | --- | --- | --- | --- |
  | `+0x20` | qword | `eboot+0xec4b47` | **a pointer**, passed on and dereferenced — the fatal one |
  | `+0x35` | byte | `eboot+0xec5a22` | a flag guarding `+0x38..+0x44` |
  | `+0x38` `+0x3c` `+0x40` `+0x44` | dword ×4 | `0xec5a80` … `0xec5c34` | summed pairwise (`[+0x3c]+[+0x38]`, `[+0x44]+[+0x40]`) — a crop rectangle |
  | `+0x48` | byte | `eboot+0xec5f34` | a flag guarding `+0x49..+0x4c` |
  | `+0x49` | byte | `eboot+0xec5f42` | |
  | `+0x4a` `+0x4c` | word ×2 | `0xec5f7f` `0xec5f84` | |
  | `+0x55` | byte | `eboot+0xec6155` | a flag guarding `+0x58..+0x5c` |
  | `+0x58` `+0x5c` | dword ×2 | `0xec6226` `0xec61ac` | |

  The shape is **H.264 SPS/VUI**: a cropping flag with four crop values, an
  aspect-ratio-present flag with an idc byte and two 16-bit sample-aspect values, and a
  timing-present flag with two 32-bit values. `CONFIDENCE: HIGH` that these offsets are read;
  `CONFIDENCE: MED` on the SPS/VUI naming, which is an inference from the access pattern, not from
  a symbol.

**Do not fill this structure from the table above.** Two thirds of 120 bytes are unobserved, and
`+0x20` is a pointer to something of unknown shape and unknown ownership — inventing it is the
failure mode this project has a rule against. What would settle it is a second caller (any other
Electra title) to cross-check the offsets, or `sceVideodec2GetAvcPictureInfo`'s (`kjrLbcyhEiw`)
distinct behaviour, which no title here has been observed to call at all.

## Fixed on the way here: `libSceAudiodec` was entirely unimplemented

**This was the first wall and it is gone.** Before 2026-08-22 the whole of `libSceAudiodec` fell to
the dispatcher's default `return 0`. `0` is `SCE_OK` for a status-returning contract, so Electra
believed it had created an AAC decoder, ran a decode that wrote nothing, and divided by the channel
count it had never been given:

```text
eboot+0xec15e1   mov  esi, DWORD PTR [r13+0x240]   ; bsiInfo.uiChannels == 0
eboot+0xec15f6   add  rsi,rsi                      ; 2 bytes/sample * 0 channels
eboot+0xec15f9   div  rsi                          ; #DE -> SIGFPE, process dies at ~5 s
```

The faulting function names itself: it logs `"AudioAACConvertOutput"` (`eboot+0x7e81031`).

`src/hle/audio/hle_audiodec.cpp` now implements the library against an ABI recovered from this
title's own call sites, decoding through the FFmpeg backend in `frontends/audio_ffmpeg`. A live boot
reports `48000Hz 2ch` and 4096 bytes of interleaved S16 per access unit — exactly one AAC-LC frame —
against `ffprobe`'s independent reading of the same file (`aac`, `profile=LC`, `48000`, `2 ch`).

The parameter block turned out to be **an unpacked MPEG-4 AudioSpecificConfig**, which is what makes
a raw (MP4 `esds`) AAC stream decodable at all: `{objectType=2, samplingFrequencyIndex=3,
channelConfiguration=2}`. See `hle_audiodec.cpp`'s header comment for the full derivation.

## Reproduction

```bash
export PROSPER_GAME_ROOT=<DUMP_ROOT>
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  ./prosper/build-linux/screenshot $PROSPER_GAME_ROOT/PPSA06367-app0 \
      --seconds 5 --count 36 --out <OUT_DIR> --timeout 220
```

Expect `shot_rc=90` and a `WORKER-THREAD FAULT … thread='ElectraPlayer::'` within ~10 s. No pad
route exists yet, and none would help: the title dies before it polls a controller.

`PROSPER_AUDIODEC_LOG=1` prints the AAC parameter block and every decode. It exists because a single
log line is what corrected this lane's first (wrong) reading of the parameter layout — see the
`## Ruled out` row.

## Ruled out

One line per hypothesis this lane killed, with the evidence and the change. Read these before
forming a new one.

| Hypothesis | Verdict and evidence |
| --- | --- |
| "The startup crash is a renderer or shader defect." | **False.** The fault is a guest `div` by zero on a worker thread named `ElectraPlayer::`, in a function that logs `AudioAACConvertOutput`. The renderer had already composited and presented; the GPU path is not implicated at any point in this crash. `eboot+0xec15f9`, run of 2026-08-22. |
| "`libSceAudiodec` is unimplemented but harmless — the movie just plays silently." | **False, and it is fatal.** The dispatcher's `return 0` reads as `SCE_OK`, so the guest proceeds into arithmetic on out-parameters that were never written. Implementing the library moves the boot past it entirely. |
| "`SceAudiodecParamAac + 0x0c` is a sample rate in Hz." | **False.** It is the MPEG-4 **samplingFrequencyIndex**. A live boot prints `+0x0c=3` for a stream `ffprobe` reports as 48000 Hz, and index 3 *is* 48000 in ISO/IEC 14496-3 Table 1.18. Configuring a decoder with "3 Hz" fails at the first access unit — which is exactly what the first implementation did. |
| "Returning an error from `sceVideodec2GetPictureInfo` would let the title continue without video." | **False, and it makes nothing better.** `nid_gate_scan` classifies both call sites (`0xec4b42`, `0xec8239`) as `ignored`: the return value is never tested. The guest reads the unwritten structure regardless, so the crash is identical. An error return is still the more honest answer, but it is not a route past this. |
| "The guest's own decode-error path can be used as a fallback." | **False.** Returning an error from `sceAudiodecDecode` (an earlier arm, when the AAC config was still wrong) sent the guest down a path that null-dereferenced at `eboot+0xec6394` — the same address the video blocker reaches. This title's media error paths are not survivable; the calls have to actually succeed. |
| "VA-API decodes this title's H.264." | **False, and it is not the blocker.** `No support for codec h264 profile 77` — the stream is Main profile and the host VA-API driver declines it, so prosper falls back to software decode (`#2270`). The fallback *works*: `au#1 bytes=508 send=0 recv=0 \| pictures=1`. The crash is downstream of a successful decode. |

## The remaining frontier: who owns the +0x20 record?

With the structure filled, Electra now walks the record at +0x20 (`P = *(void**)record`,
C-string at `P+0x08`, ownership flag at `P+0x30`) — and then **frees the record through
its own allocator**: prosper's calloc'd block came back as
`FMallocBinned3 Attempt to free an unrecognized block <the same address>`, fatal. So the
real decoder hands back a block the GUEST ALLOCATOR owns (or a value whose free path is
guarded). Two readings were measured and both falsified:

- A zeroed prosper-owned block with an empty string: guest walks it fine but the post-consumer
  free fatals (above).
- Echoing `out->frame` (the decoded NV12 buffer): the guest derefs `*record` as a pointer to
  the payload, reads the frame data as a pointer, and faults at `image+0x105ed54`
  (`PROSPER_VDEC2_PICINFO_ECHO_FRAME=1` boot). The field really is a pointer TO a payload,
  not itself payload.

The decisive question: where does Electra allocate (or register) that block so the real
decoder can hand it back owned? Candidates: an echo of something passed in via
`VdecInput`/`CreateDecoder` (Beast's identity arrays suggest a registered token), or an
allocation from a shared pool MB3 recognizes. Beast (`PPSA29343`) live-traced at its three
call sites would settle what the arrays at `this+0x150/0x170/0x178` hold.

## What would move this to rung 1

1. **Settle the +0x20 ownership contract** (above) so Electra's post-picture free succeeds.
2. Then whatever the movie pipeline wants next — every crash since the fix has moved
   further into Electra's own logic, which is exactly the shape of progress.
3. Failing that, establish whether Unreal's Electra has a configuration under which `PPSA06367`
   skips its startup movies through an ordinary guest path rather than a forced one. Nothing here
   is currently known about that, and a forced skip would not be acceptance evidence anyway.
