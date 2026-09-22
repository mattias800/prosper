# `video_mf` — Windows video decode through Media Foundation

The Windows implementation of `hle/video/video_backend.hpp`. The HLE owns every guest-facing
contract; this folder only turns compressed bytes into NV12 on the host. Linux's counterpart is
`frontends/video_vaapi` (FFmpeg + VA-API), and the two must answer the backend contract the same
way — the HLE cannot tell them apart and must not need to.

Two pipelines live here, because the guest has two different shapes of request:

- `media_foundation_backend` — **sceAvPlayer**: the guest names a whole media file and prosper does
  demux, decode and pacing. Media Foundation's Source Reader does all three, on a worker thread per
  stream. Hardware decode is required unless `PROSPER_AVP_ALLOW_SOFTWARE=1`.
- `mf_access_unit_decoder` — **sceVideodec2** (#2983): the guest demuxes itself and submits one
  access unit at a time. There is no file for a Source Reader, so this drives a decoder MFT directly.
  DXVA is requested and software is the fallback, as on Linux; `AuPicture::hardware` reports which
  one each picture actually came from.

Both run every Media Foundation and D3D11 call on a host thread of their own, never on the guest
thread that made the HLE call. Keep it that way: the system decoders are opaque code, and a guest
thread's stack belongs to the title.

Codec availability is the host's, not prosper's. H.264 is built into Windows; VP9 comes from
Microsoft's *VP9 Video Extensions* package (present on most Windows 10/11 installs). Nothing here is
shipped with the release archive, so when a decoder is missing the right answer is a refusal that
names the package — never a substitute picture.

`vp9_uncompressed_header.hpp` exists because an MFT is configured with a frame size before it sees a
byte, and a bare VP9 elementary stream states its size only in its key-frame header.
