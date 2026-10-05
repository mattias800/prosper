// libSceNgs2 waveform surface: sceNgs2ParseWaveformData, sceNgs2CalcWaveformBlock and
// sceNgs2SystemResetOption, re-derived from libSceNgs2.native.sprx (the module PS5 titles bind;
// see ngs2_waveform.cpp for the per-rule evidence). The thin HLE entry points live with the rest of
// NGS2 in hle_audio.cpp; this file is the contract, kept apart so it is testable without the
// dispatcher and so hle_audio.cpp does not grow past its size cap.
#pragma once

#include <cstdint>

namespace prosper::ngs2_waveform {

// The native module's own error family for these three exports (it is NOT the 0x804a04xx family
// the non-native libSceNgs2.sprx uses). Returned as the 32-bit value the module puts in %eax.
constexpr uint32_t kNgs2NativeErrNullOut = 0x804a8010;            // NULL info/block/option pointer
constexpr uint32_t kNgs2NativeErrInvalidData = 0x804a8430;        // bad source, magic or chunk walk
constexpr uint32_t kNgs2NativeErrNullFormat = 0x804a8431;         // CalcWaveformBlock: NULL format
constexpr uint32_t kNgs2NativeErrUnsupportedFormat = 0x804a8432;  // codec/bit depth not handled

constexpr uint32_t kNgs2NativeSystemOptionBytes = 0x90;
constexpr uint32_t kNgs2NativeWaveformBlockBytes = 0x28;
constexpr uint32_t kNgs2NativeWaveformInfoHeaderBytes = 0x48;
constexpr uint32_t kNgs2NativeWaveformMaxBlocks = 4;   // the info struct's capacity (0xe8 bytes)

// Waveform types the module reports in SceNgs2WaveformFormat::type.
constexpr uint32_t kNgs2NativeTypePcmFirst = 0x10;   // 0x10..0x1b: linear PCM / float variants
constexpr uint32_t kNgs2NativeTypePcmLast = 0x1b;
constexpr uint32_t kNgs2NativeTypeVag = 0x1c;
constexpr uint32_t kNgs2NativeTypeAtrac9 = 0x40;
constexpr uint32_t kNgs2NativeTypeExt43 = 0x43;      // third extensible SubFormat GUID (see .cpp)

// Guest-facing operations. Every pointer is a guest address; reads and writes are fault-safe.
uint32_t parse_waveform_data(uint64_t data, uint64_t size, uint64_t info);
uint32_t calc_waveform_block(uint64_t format, uint32_t sample_pos, uint32_t num_samples,
                             uint64_t block);
uint32_t system_reset_option(uint64_t option);

// Pure block mapping (the body of CalcWaveformBlock after its pointer checks). Always zeroes
// `block` first, as the module does, then fills it on success.
uint32_t calc_block(uint32_t type, uint32_t channels, uint32_t config, uint32_t sample_pos,
                    uint32_t num_samples, uint8_t block[kNgs2NativeWaveformBlockBytes]);

}   // namespace prosper::ngs2_waveform
