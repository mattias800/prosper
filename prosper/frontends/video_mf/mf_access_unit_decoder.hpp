#pragma once

#include "hle/video/video_backend.hpp"

#include <cstddef>
#include <cstdint>

// sceVideodec2's access-unit decode on Windows (#2983), through a Media Foundation decoder MFT.
//
// The guest demuxes itself and submits one compressed access unit at a time, expecting at most one
// picture back (see VideoBackend::decode_au). A Media Foundation decoder transform is exactly that
// contract -- ProcessInput one sample, ProcessOutput what it produced -- so this drives the MFT
// directly instead of going through the Source Reader, which only knows how to read whole files and
// is what MediaFoundationBackend's AvPlayer path uses.
//
// The MediaFoundationBackend methods of the same names forward here. It is a separate translation
// unit because it is a separate pipeline: no demux, no clock, no audio, no queue feeding a consumer.
namespace prosper::video::mf_au {

int  open_decoder(uint32_t codec);
VideoBackend::AuResult decode_au(int id, const uint8_t* au, size_t bytes, uint8_t* dst,
                                 uint64_t dst_bytes, VideoBackend::AuPicture& out);
bool reset_decoder(int id);
void close_decoder(int id);

} // namespace prosper::video::mf_au
