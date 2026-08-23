#pragma once
// H.264 SPS/VUI extraction for sceVideodec2GetPictureInfo (#2898).
//
// The Gollum/Beast Electra callers read a specific field set out of the callee-filled
// picture-info structure (eboot evidence in #2898): a frame-cropping flag with four
// offsets, an aspect-ratio flag with idc + two SAR values, and a timing-info flag with
// two 32-bit values. That is exactly the SPS VUI grammar, and every one of those values
// is present in the access-unit bytes the guest already handed us at Decode time -- so
// the fill derives from the title's own bitstream rather than inventing an ABI.
//
// Everything here is pure and host-side: no guest memory, no globals, fully unit-tested.

#include <cstddef>
#include <cstdint>

namespace prosper::h264 {

// The picture-info fields the guest reads, exactly as parsed from the stream's first
// SPS. Field names follow the H.264 spec grammar; the mapping to guest offsets lives in
// fill_picture_info() where the offset evidence is cited.
struct SpsPictureMeta {
    bool crop_flag = false;
    uint32_t crop[4] = {0, 0, 0, 0};   // left, right, top, bottom (frame_crop_*_offset)
    bool ar_flag = false;
    uint8_t ar_idc = 0;
    uint16_t sar_w = 0, sar_h = 0;     // RESOLVED values (idc table applied), not raw
    bool timing_flag = false;
    uint32_t num_units_in_tick = 0, time_scale = 0;
};

// Scan one Annex-B access unit for its first SPS NAL (type 7) and parse the subset of
// syntax through VUI timing info. Returns false when no parseable SPS is present (the
// caller then reports no metadata rather than zeros-that-look-parsed).
//
// Handles emulation-prevention byte removal and the High-profile scaling-list skip,
// which real movie streams carry routinely.
bool parse_first_sps(const uint8_t* au, size_t n, SpsPictureMeta* out);

// aspect_ratio_idc -> SAR per the H.264 Table E-1. Extended SAR is idc 255 (caller
// supplies w/h); unknown idcs resolve to 1:1 rather than guessing.
void resolve_sar(uint8_t idc, uint32_t ext_w, uint32_t ext_h, uint32_t* w, uint32_t* h);

// Fill the caller's picture-info buffer from parsed metadata.
//
// Offset evidence (#2898, PPSA06367 eboot; identical reads at both call sites):
//   +0x20 qword  pointer to a decoder-side record (Beast compares it for identity;
//                Gollum reads a C-string at +0x08 of it and a flag byte at +0x30).
//   +0x35 byte  crop flag guarding +0x38..+0x44
//   +0x38..+0x44 dword x4  crop offsets, summed pairwise by the guest
//   +0x48 byte  aspect-ratio flag; +0x49 idc byte; +0x4a/+0x4c u16 SAR pair
//   +0x55 byte  timing flag; +0x58/+0x5c u32 num_units_in_tick / time_scale
//
// `record` is written verbatim into +0x20; it is owned by the caller (hle_service keeps
// it alive with the decoder). Offsets above +0x5f are unobserved and left untouched --
// this function never writes a byte the guest has not been seen reading.
//
// CONFIDENCE: HIGH that these offsets are read at these instructions; MED on the
// left/right/top/bottom ORDER within the crop quad (the guest only sums the pairs) and
// on the SPS/VUI naming itself, which is an inference from the access pattern.
// Returns false when `size` cannot hold the observed fields (< 0x60), in which case the
// caller must not write anything.
bool fill_picture_info(const SpsPictureMeta& meta, void* record, void* pic_info,
                       size_t size);

}  // namespace prosper::h264
