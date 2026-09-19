# `video` — the guest's video decode and playback libraries

The Sony libraries a title calls to get a movie on screen, and the host backend they run on.

- `video_backend` — the host decode backend contract. `avplayer.cpp` and the decoder libraries talk
  to this, never to FFmpeg or VA-API directly; the concrete backends live in `frontends/video_*`.
- `h264_sps` — SPS/VUI extraction. Its consumer is `GetPictureInfo`, which has to answer questions
  about a stream (dimensions, aspect, frame rate) that only the bitstream itself carries.
- `avplayer` — libSceAvPlayer: the playback lifecycle, the media clock, and the guest
  file-replacement reader. Split out of `hle/service/hle_service.cpp` in #3735.
- `videodec2` — libSceVideodec2, its software-decode sibling libSceVdecsw, and libScePsml. Same
  split. Videodec2 and Vdecsw share this file because Vdecsw's decoder config *is* Videodec2's plus
  eight bytes and the two are read by the same code; libScePsml is here because it is a decode
  library, not because it shares anything — it is four unimplemented NIDs whose names were never
  recovered.

## What belongs here

A Sony library whose subject is video. That is now the whole decode-and-play surface, and
`hle/service/` keeps only the things that are not about a medium.

Audio is `hle/audio/`, not here, even where a title drives both through one AvPlayer handle — the
split follows the Sony library, not the guest's call graph.

## Registration

Each library file owns a `register_<lib>_hle()` that `register_builtin_hle()` calls, declared in
`hle/dispatch/dispatch.hpp`. That is what lets every handler stay `static` to the file implementing
it, which is in turn what lets these be separate translation units at all. A new library here adds
its own entry point rather than extending somebody else's.
