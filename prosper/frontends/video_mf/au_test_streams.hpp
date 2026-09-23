#pragma once

// Shared by the two Media Foundation access-unit tests (test_video_mf, test_videodec2_mf): reading
// the committed elementary streams, splitting them the way a guest demuxer would, hashing pictures,
// the independent reference values for the VP9 asset, and asking Media Foundation -- not the code
// under test -- whether this host has a decoder at all.

#include <windows.h>
#include <combaseapi.h>
#include <mfapi.h>
#include <mftransform.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace prosper::video::au_test {

using Unit = std::pair<size_t, size_t>;   // offset, length

inline std::vector<uint8_t> read_file(const char* path) {
    std::vector<uint8_t> bytes;
    if (std::FILE* f = std::fopen(path, "rb")) {
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (size > 0) {
            bytes.resize(static_cast<size_t>(size));
            if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
        }
        std::fclose(f);
    }
    return bytes;
}

inline uint64_t fnv1a(const uint8_t* data, size_t bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < bytes; ++i) { h ^= data[i]; h *= 0x100000001b3ull; }
    return h;
}

// IVF: a 32-byte file header, then per frame a 12-byte header (u32 size, u64 pts) and the frame.
// Each frame is one VP9 access unit exactly as a demuxer hands it on -- a superframe (hidden alt-ref
// plus the frame it serves) stays ONE unit, which is how a guest demuxer would submit it.
inline std::vector<Unit> split_ivf(const std::vector<uint8_t>& b) {
    std::vector<Unit> units;
    if (b.size() < 32 || std::memcmp(b.data(), "DKIF", 4) != 0) return units;
    size_t off = 32;
    while (off + 12 <= b.size()) {
        const uint32_t size = b[off] | (b[off + 1] << 8) | (b[off + 2] << 16) |
                              (static_cast<uint32_t>(b[off + 3]) << 24);
        if (off + 12 + size > b.size()) break;
        units.emplace_back(off + 12, size);
        off += 12 + size;
    }
    return units;
}

// Whether THIS HOST has a synchronous Media Foundation decoder for the format, asked of Media
// Foundation directly rather than of the code under test. Codecs are the host's: H.264 ships with
// Windows, VP9 comes from the separately installed VP9 Video Extensions, and a Windows Server CI image
// may have neither. A host without one is a supported configuration, so the decode assertions are
// conditional on this -- and the refusal is asserted instead, so a skip is never silent and a
// backend that fabricated a decoder where none exists would fail.
inline bool host_has_decoder(const GUID& subtype) {
    const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    MFT_REGISTER_TYPE_INFO input{MFMediaType_Video, subtype};
    IMFActivate** activates = nullptr;
    UINT32 count = 0;
    const HRESULT hr = MFTEnumEx(MFT_CATEGORY_VIDEO_DECODER,
                                 MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT |
                                     MFT_ENUM_FLAG_SORTANDFILTER,
                                 &input, nullptr, &activates, &count);
    for (UINT32 i = 0; SUCCEEDED(hr) && i < count; ++i) activates[i]->Release();
    CoTaskMemFree(activates);
    if (SUCCEEDED(com)) CoUninitialize();
    return SUCCEEDED(hr) && count > 0;
}

// tests/assets/vp9_testpattern.ivf, 160x90, 24 shown frames. Produced by: ffmpeg -i <asset> -f
// rawvideo -pix_fmt nv12 ref.nv12 (libavcodec's own VP9 decoder, not Media Foundation), then
// FNV-1a-64 over each 160*90 + 2*80*45 = 21600-byte frame. All 24 differ.
constexpr uint32_t kVp9Width = 160, kVp9Height = 90;
inline constexpr uint64_t kVp9Expected[24] = {
    0xc5d881f5ad18d6e3ull, 0x0a104ad23033be37ull, 0x0c671d7cbaffd853ull,
    0xb050c7d781be8167ull, 0x30114c4d0db7d39dull, 0x64bde2949d311befull,
    0x437df2ad43354a19ull, 0x75ba29d1fff0cd40ull, 0x19f7fb26cfc95d38ull,
    0x3a15afbba23fda2eull, 0xbd6bfff70be35829ull, 0x762c1ff5bc183ce6ull,
    0x3218d18e7619cc16ull, 0x5069861f5e6eb052ull, 0xbcb8cbd90ac03d32ull,
    0xfcaab884215e640eull, 0xa8609984e0cd3d83ull, 0x895e7e704f12516aull,
    0xb33fcf120f8f743full, 0x9b769661ad98990bull, 0x0dc8693197c79bcbull,
    0x2ef06dd3c0b3d647ull, 0x4fee04f0e7dbf647ull, 0x82a50de1bd934ad4ull,
};

} // namespace prosper::video::au_test
