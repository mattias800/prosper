#pragma once

// Install FFmpeg's software audio codecs as the host backend for compressed AJM streams. The
// backend currently exposes MP3 (AJM codec 0); unsupported codecs truthfully remain unavailable.
namespace prosper::ajm {

bool install_ffmpeg_decoder_backend();
void uninstall_ffmpeg_decoder_backend();

} // namespace prosper::ajm
