#include "h264_sps.hpp"

#include <vector>
#include <cstdio>
#include <cstdlib>

namespace prosper::h264 {
namespace {

// ---- Annex-B NAL scanning -------------------------------------------------

// Index of the next start-code suffix (the 01 of 00 00 01 / 00 00 00 01) at or after
// `from`, or `n` when absent.
size_t find_start_code(const uint8_t* p, size_t n, size_t from) {
    for (size_t i = from; i + 2 < n; ++i)
        if (p[i] == 0 && p[i + 1] == 0 && p[i + 2] == 1) return i;
    return n;
}

// Strip emulation-prevention bytes (00 00 03 -> 00 00) into `dst`; dst must hold n
// bytes (output is never longer than input).
size_t unescape(const uint8_t* src, size_t n, uint8_t* dst) {
    size_t o = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i + 2 < n && src[i] == 0 && src[i + 1] == 0 && src[i + 2] == 3) {
            dst[o++] = 0;
            dst[o++] = 0;
            i += 2;
        } else {
            dst[o++] = src[i];
        }
    }
    return o;
}

// ---- Bit reader with ue/se ------------------------------------------------

struct BitReader {
    const uint8_t* p;
    size_t n;
    size_t bit = 0;

    bool u(uint32_t count, uint32_t* out) {
        uint32_t v = 0;
        for (uint32_t i = 0; i < count; ++i) {
            size_t byte = bit >> 3;
            if (byte >= n) return false;
            v = (v << 1) | ((p[byte] >> (7 - (bit & 7))) & 1);
            ++bit;
        }
        *out = v;
        return true;
    }

    bool u8(uint8_t* out) {
        uint32_t v;
        if (!u(8, &v)) return false;
        *out = (uint8_t)v;
        return true;
    }

    // Exp-Golomb unsigned: z leading zero bits, a terminating 1, then z suffix bits.
    bool ue(uint32_t* out) {
        uint32_t zeros = 0, b = 0;
        for (;;) {
            if (!u(1, &b)) return false;
            if (b) break;
            if (++zeros > 31) return false;
        }
        uint32_t suffix = 0;
        if (!u(zeros, &suffix)) return false;
        *out = (zeros ? (1u << zeros) - 1u : 0u) + suffix;
        return true;
    }

    bool se(uint32_t* out) {
        uint32_t k;
        if (!ue(&k)) return false;
        *out = (k & 1) ? (k + 1) / 2 : 0u - (k / 2);
        return true;
    }
};

}  // namespace

void resolve_sar(uint8_t idc, uint32_t ext_w, uint32_t ext_h, uint32_t* w, uint32_t* h) {
    // H.264 Table E-1: the aspect_ratio_idc values a movie stream can carry.
    static const struct { uint8_t idc; uint16_t w, h; } kTable[] = {
        {1, 1, 1},    {2, 12, 11},   {3, 10, 11},  {4, 16, 11},  {5, 40, 33},
        {6, 24, 11},  {7, 20, 11},   {8, 32, 11},  {9, 80, 33},  {10, 18, 11},
        {11, 15, 11}, {12, 64, 33},  {13, 160, 99}, {14, 4, 3},  {15, 3, 2},
        {16, 2, 1},
    };
    if (idc == 255) { *w = ext_w; *h = ext_h; return; }
    for (const auto& e : kTable)
        if (e.idc == idc) { *w = e.w; *h = e.h; return; }
    *w = 1; *h = 1;  // reserved/unspecified idc: square pixels rather than a guess
}

bool parse_first_sps(const uint8_t* au, size_t n, SpsPictureMeta* out) {
    if (!au || !out || n < 4) return false;

    // Find the first SPS NAL, skipping SVC/MVC prefix NALs whose payload re-announces
    // the following NAL's type.
    size_t sc = 0, nal_start = 0, nal_end = 0;
    bool found = false;
    while (!found) {
        sc = find_start_code(au, n, sc);
        if (sc >= n) return false;
        size_t hdr = sc + 3;
        if (hdr >= n) return false;
        uint8_t first_byte = au[hdr];
        size_t next = find_start_code(au, n, hdr);
        // Trim the trailing zero bytes that belong to the NEXT start code's prefix --
        // but only when there IS one: a well-formed RBSP carries a stop bit, yet this
        // tool must not depend on that, and the final NAL's trailing bytes are data.
        size_t end = next;
        if (next < n)
            while (end > hdr && au[end - 1] == 0) --end;
        sc = next;
        if (first_byte == 0x0e || first_byte == 0x14) continue;  // prefix NAL
        if ((first_byte & 0x1f) != 7) continue;                  // not an SPS
        nal_start = hdr + 1;
        nal_end = end;
        found = true;
    }

    std::vector<uint8_t> rbsp(nal_end - nal_start);
    size_t rlen = unescape(au + nal_start, nal_end - nal_start, rbsp.data());
    rbsp.resize(rlen);

    BitReader br{rbsp.data(), rbsp.size()};
    uint32_t profile_idc, constraint_flags, level_idc, sps_id;
    if (!br.u(8, &profile_idc) || !br.u(8, &constraint_flags) || !br.u(8, &level_idc))
        return false;
    if (!br.ue(&sps_id)) return false;

    static const uint32_t kHighProfiles[] = {100, 110, 122, 244, 44, 83, 86, 118,
                                             128, 138, 139, 134, 135};
    bool high = false;
    for (uint32_t hp : kHighProfiles) high = high || profile_idc == hp;

    if (high) {
        uint32_t chroma_format_idc;
        if (!br.ue(&chroma_format_idc)) return false;
        if (chroma_format_idc == 3) {
            uint32_t separate;
            if (!br.u(1, &separate)) return false;
        }
        uint32_t bd_luma, bd_chroma, qpprime, scaling;
        if (!br.ue(&bd_luma) || !br.ue(&bd_chroma) || !br.u(1, &qpprime) ||
            !br.u(1, &scaling))
            return false;
        if (scaling) {
            // seq_scaling_matrix(): 8 lists, 12 when chroma_format_idc == 3.
            uint32_t lists = chroma_format_idc != 3 ? 8u : 12u;
            for (uint32_t i = 0; i < lists; ++i) {
                uint32_t present;
                if (!br.u(1, &present)) return false;
                if (!present) continue;
                // scaling_list(): delta_scale per coefficient unless a coefficient
                // keeps next_scale == 0, which ends the list early. Getting this wrong
                // desyncs every field after it, so follow the grammar exactly.
                uint32_t size_of_scaling_list = i < 6 ? 16u : 64u;
                int32_t last_scale = 8, next_scale = 8;
                for (uint32_t j = 0; j < size_of_scaling_list; ++j) {
                    if (next_scale != 0) {
                        uint32_t delta_signed;
                        if (!br.se(&delta_signed)) return false;
                        next_scale = (last_scale + (int32_t)delta_signed + 256) % 256;
                    }
                    last_scale = (next_scale == 0) ? last_scale : next_scale;
                }
            }
        }
    }

    uint32_t log2_mfn, poc_type;
    if (!br.ue(&log2_mfn) || !br.ue(&poc_type)) return false;
    if (poc_type == 0) {
        uint32_t lsb;
        if (!br.ue(&lsb)) return false;
    } else if (poc_type == 1) {
        uint32_t flag, v, cycles;
        if (!br.u(1, &flag) || !br.se(&v) || !br.se(&v) || !br.ue(&cycles)) return false;
        for (uint32_t i = 0; i < cycles; ++i)
            if (!br.se(&v)) return false;
    }

    uint32_t refs, gaps, w_mbs, h_map, mbs_only, d8x8;
    if (!br.ue(&refs) || !br.u(1, &gaps) || !br.ue(&w_mbs) || !br.ue(&h_map) ||
        !br.u(1, &mbs_only) || !br.u(1, &d8x8))
        return false;
    if (!mbs_only) {
        uint32_t mb_adaptive;
        if (!br.u(1, &mb_adaptive)) return false;
    }

    SpsPictureMeta meta;
    uint32_t crop_flag;
    if (!br.u(1, &crop_flag)) return false;
    meta.crop_flag = crop_flag != 0;
    if (meta.crop_flag)
        for (int i = 0; i < 4; ++i) {
            if (!br.ue(&meta.crop[i])) return false;
        }

    uint32_t vui_present;
    if (!br.u(1, &vui_present)) return false;
    if (vui_present) {
        uint32_t ar_present;
        if (!br.u(1, &ar_present)) return false;
        meta.ar_flag = ar_present != 0;
        if (meta.ar_flag) {
            uint8_t idc;
            if (!br.u8(&idc)) return false;
            meta.ar_idc = idc;
            uint32_t w = 0, h = 0;
            if (idc == 255 && (!br.u(16, &w) || !br.u(16, &h))) return false;
            resolve_sar(idc, w, h, &w, &h);
            meta.sar_w = (uint16_t)w;
            meta.sar_h = (uint16_t)h;
        }
        uint32_t f;
        if (!br.u(1, &f)) return false;              // overscan_info_present_flag
        if (f && !br.u(1, &f)) return false;         // overscan_appropriate_flag
        if (!br.u(1, &f)) return false;              // video_signal_type_present_flag
        if (f) {
            // video_format u(3), full_range u(1), colour_description_present u(1).
            uint32_t colour_desc;
            if (!br.u(3, &f) || !br.u(1, &f) || !br.u(1, &colour_desc)) return false;
            if (colour_desc &&
                (!br.u(8, &f) || !br.u(8, &f) || !br.u(8, &f)))
                return false;
        }
        // chroma_loc_info_present_flag is a TOP-LEVEL VUI field -- it does not sit
        // inside video_signal_type. Nesting it there desynced every stream that omits
        // video signal info and carried timing instead.
        {
            uint32_t chroma_loc;
            if (!br.u(1, &chroma_loc)) return false;
            if (chroma_loc && !br.ue(&f)) return false;
        }
        uint32_t timing;
        if (!br.u(1, &timing)) return false;
        meta.timing_flag = timing != 0;
        if (meta.timing_flag) {
            uint32_t units, scale, fixed;
            if (!br.u(32, &units) || !br.u(32, &scale) || !br.u(1, &fixed)) return false;
            meta.num_units_in_tick = units;
            meta.time_scale = scale;
        }
        // VUI syntax after timing_info (hrd, bitstream restriction, ...) is not read by
        // any observed caller; parsing further would only add ways to fail on real
        // streams. Stop here by design.
    }

    *out = meta;
    return true;
}

bool fill_picture_info(const SpsPictureMeta& meta, void* record, void* pic_info,
                       size_t size) {
    // 0x5c..0x5f is the highest observed read (time_scale), so anything smaller cannot
    // hold the fields the guest will fetch.
    if (size < 0x60) return false;
    auto* p = (uint8_t*)pic_info;
    auto w32 = [&](size_t off, uint32_t v) {
        p[off] = (uint8_t)v;
        p[off + 1] = (uint8_t)(v >> 8);
        p[off + 2] = (uint8_t)(v >> 16);
        p[off + 3] = (uint8_t)(v >> 24);
    };
    auto w16 = [&](size_t off, uint16_t v) {
        p[off] = (uint8_t)v;
        p[off + 1] = (uint8_t)(v >> 8);
    };

    uint64_t rec = (uint64_t)(uintptr_t)record;
    for (int i = 0; i < 8; ++i) p[0x20 + i] = (uint8_t)(rec >> (8 * i));

    p[0x35] = meta.crop_flag ? 1 : 0;
    if (meta.crop_flag) {
        w32(0x38, meta.crop[0]);
        w32(0x3c, meta.crop[1]);
        w32(0x40, meta.crop[2]);
        w32(0x44, meta.crop[3]);
    }
    p[0x48] = meta.ar_flag ? 1 : 0;
    if (meta.ar_flag) {
        p[0x49] = meta.ar_idc;
        w16(0x4a, meta.sar_w);
        w16(0x4c, meta.sar_h);
    }
    p[0x55] = meta.timing_flag ? 1 : 0;
    if (meta.timing_flag) {
        w32(0x58, meta.num_units_in_tick);
        w32(0x5c, meta.time_scale);
    }
    return true;
}

}  // namespace prosper::h264
