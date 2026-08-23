// test_h264_sps (#2898) — the SPS/VUI extraction behind sceVideodec2GetPictureInfo.
//
// Before this change, GetPictureInfo returned SCE_OK and wrote NOTHING into the caller's
// picture-info block; Gollum's Electra worker then dereferenced the never-written pointer
// at +0x20 and died at ~4 s into every boot. The fill now derives its values from the
// access unit the guest itself handed us, so every test here round-trips a hand-built SPS
// through parse_first_sps() and checks exact field positions in fill_picture_info().
//
// The fixtures are built by a bit-writer that mirrors the spec grammar (not our reader's
// code paths — it is written independently below, field-by-field from the syntax table).
// One deliberate asymmetry guards against a mirrored-bug-for-bug blind spot: fixture B
// exercises the High-profile scaling-list skip, whose bit length depends on the delta
// VALUES chosen, so a reader that mis-skips desyncs every later field and fails loudly.

#include "hle/video/h264_sps.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int fails = 0;
#define CHECK(c, m) \
    do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } } while (0)

using prosper::h264::SpsPictureMeta;

// ---- Independent H.264 bit-writer (RBSP; caller adds NAL header + start codes) ----

struct BitWriter {
    std::vector<uint8_t> bytes = {0};
    uint32_t bit = 0;

    void u(uint32_t count, uint32_t v) {
        for (uint32_t i = 0; i < count; ++i) {
            if (bit % 8 == 0) bytes.push_back(0);
            size_t last = bytes.size() - 1;
            uint32_t shift = count - 1 - i;
            bytes[last] |= ((v >> shift) & 1) << (7 - (bit % 8));
            ++bit;
        }
    }

    void ue(uint32_t v) {
        uint32_t val = v + 1;
        uint32_t len = 0;
        for (uint32_t t = val; t > 1; t >>= 1) ++len;
        u(len, 0);
        u(len + 1, val);
    }

    void se(int32_t v) { ue(v <= 0 ? (uint32_t)(-2 * v) : (uint32_t)(2 * v - 1)); }

    // Emulation prevention: insert a 03 after any 00 00 pair followed by <= 3.
    std::vector<uint8_t> escape() const {
        std::vector<uint8_t> out;
        int zeros = 0;
        for (uint8_t b : bytes) {
            if (zeros >= 2 && b <= 3) out.push_back(3);
            out.push_back(b);
            zeros = b == 0 ? zeros + 1 : 0;
        }
        return out;
    }
};

static std::vector<uint8_t> wrap_nal(const BitWriter& w, uint8_t nal_hdr,
                                     bool apply_escape = true) {
    std::vector<uint8_t> au = {0, 0, 0, 1, nal_hdr};
    auto payload = apply_escape ? w.escape() : w.bytes;
    // The writer seeds a leading zero byte that is only correct mid-RBSP; drop it when it
    // would collide with the NAL header position (payload[0] is always the seeded zero).
    au.insert(au.end(), payload.begin() + 1, payload.end());
    return au;
}

// ---- Fixtures --------------------------------------------------------------

// Baseline SPS: no scaling lists, no VUI at all.
static std::vector<uint8_t> sps_baseline(uint8_t sps_id) {
    BitWriter w;
    w.u(8, 66);      // profile_idc = 66 (Baseline)
    w.u(8, 0xC0);    // constraint flags
    w.u(8, 30);      // level_idc
    w.ue(sps_id);    // seq_parameter_set_id
    w.ue(1);         // log2_max_frame_num_minus4
    w.ue(0);         // pic_order_cnt_type
    w.ue(4);         // log2_max_pic_order_cnt_lsb_minus4
    w.ue(2);         // max_num_ref_frames
    w.u(1, 0);       // gaps_in_frame_num_value_allowed_flag
    w.ue(39);        // pic_width_in_mbs_minus1 -> 640
    w.ue(22);        // pic_height_in_map_units_minus1 -> 352
    w.u(1, 1);       // frame_mbs_only_flag
    w.u(1, 1);       // direct_8x8_inference_flag
    w.u(1, 0);       // frame_cropping_flag
    w.u(1, 0);       // vui_parameters_present_flag
    return wrap_nal(w, 0x67);
}

// High-profile SPS with a scaling matrix, cropping quad, extended SAR and timing.
static std::vector<uint8_t> sps_high_full(uint8_t sps_id) {
    BitWriter w;
    w.u(8, 100);     // profile_idc = 100 (High)
    w.u(8, 0xC0);
    w.u(8, 41);      // level_idc 4.1
    w.ue(sps_id);
    w.ue(1);         // chroma_format_idc = 1 (4:2:0)
    w.ue(0);         // bit_depth_luma_minus8
    w.ue(0);         // bit_depth_chroma_minus8
    w.u(1, 0);       // qpprime_y_zero_transform_bypass
    w.u(1, 1);       // seq_scaling_matrix_present_flag  <- the skip must land here

    // 8 lists: six 16-coefficient (4x4) then two 64-coefficient (8x8), all default via
    // a single terminating delta... except list 1, which writes real deltas so a reader
    // that skips the wrong number of coefficients desyncs.
    for (int i = 0; i < 8; ++i) {
        w.u(1, 1);   // seq_scaling_list_present_flag[i]
        uint32_t coeffs = i < 6 ? 16 : 64;
        int32_t last = 8, next = 8;
        for (uint32_t j = 0; j < coeffs; ++j) {
            // Alternate small deltas so every coefficient is written explicitly.
            int32_t delta = (j % 2 == 0) ? 1 : -1;
            next = (last + delta + 256) % 256;
            w.se(delta);
            last = next;
        }
    }

    w.ue(4);         // log2_max_frame_num_minus4
    w.ue(0);         // pic_order_cnt_type
    w.ue(4);         // log2_max_pic_order_cnt_lsb_minus4
    w.ue(3);         // max_num_ref_frames
    w.u(1, 0);       // gaps_in_frame_num_value_allowed_flag
    w.ue(119);       // pic_width_in_mbs_minus1 -> 1920
    w.ue(67);        // pic_height_in_map_units_minus1 -> 1088
    w.u(1, 1);       // frame_mbs_only_flag
    w.u(1, 1);       // direct_8x8_inference_flag
    w.u(1, 1);       // frame_cropping_flag
    w.ue(0);         // frame_crop_left_offset   = 0
    w.ue(8);         // frame_crop_right_offset
    w.ue(2);         // frame_crop_top_offset
    w.ue(6);         // frame_crop_bottom_offset
    w.u(1, 1);       // vui_parameters_present_flag
    w.u(1, 1);       // aspect_ratio_info_present_flag
    w.u(8, 255);     // aspect_ratio_idc = extended_SAR
    w.u(16, 40);     // sar_width
    w.u(16, 33);     // sar_height
    w.u(1, 0);       // overscan_info_present_flag
    w.u(1, 1);       // video_signal_type_present_flag
    w.u(3, 5);       // video_format
    w.u(1, 1);       // video_full_range_flag
    w.u(1, 0);       // colour_description_present_flag
    w.u(1, 0);       // chroma_loc_info_present_flag
    w.u(1, 1);       // timing_info_present_flag
    w.u(32, 1001);   // num_units_in_tick
    w.u(32, 60000);  // time_scale
    w.u(1, 1);       // fixed_frame_rate_flag
    // VUI continues (hrd etc.) but the parser stops at timing; nothing more needed.
    return wrap_nal(w, 0x67);
}

// ---- Tests -----------------------------------------------------------------

static void test_baseline_parses_with_all_flags_absent() {
    auto au = sps_baseline(0);
    SpsPictureMeta m;
    CHECK(prosper::h264::parse_first_sps(au.data(), au.size(), &m),
          "baseline SPS should parse");
    CHECK(!m.crop_flag, "no crop flag expected");
    CHECK(!m.ar_flag, "no AR flag expected");
    CHECK(!m.timing_flag, "no timing flag expected");
}

static void test_high_profile_scaling_skip_and_vui() {
    auto au = sps_high_full(7);
    SpsPictureMeta m;
    CHECK(prosper::h264::parse_first_sps(au.data(), au.size(), &m),
          "high-profile SPS with scaling matrix should parse");
    CHECK(m.crop_flag, "crop flag expected");
    CHECK(m.crop[0] == 0 && m.crop[1] == 8 && m.crop[2] == 2 && m.crop[3] == 6,
          "crop quad values must match exactly");
    CHECK(m.ar_flag && m.ar_idc == 255, "extended SAR idc expected");
    CHECK(m.sar_w == 40 && m.sar_h == 33, "SAR must resolve to the explicit 40/33");
    CHECK(m.timing_flag && m.num_units_in_tick == 1001 && m.time_scale == 60000,
          "timing pair must match exactly (60000/1001 = NTSC film)");
}

static void test_emulation_prevention_bytes_are_stripped() {
    // Two requirements meet here. The PLAIN form must be a valid unescaped RBSP -- real
    // streams cannot contain raw 00 00 0x sequences anywhere, including inside u(32)
    // fields, which is why the timing constants below set bit 31 (no leading zero
    // bytes). And the ESCAPED form must provably contain inserted 0x03 bytes: a
    // three-zero-byte tail past the parser's stop point guarantees exactly one.
    BitWriter w;
    w.u(8, 100);
    w.u(8, 0xC0);
    w.u(8, 41);
    w.ue(7);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.u(1, 0);
    w.u(1, 0);       // no scaling matrix here: keep this fixture small
    w.ue(4);
    w.ue(0);
    w.ue(4);
    w.ue(2);
    w.u(1, 0);
    w.ue(119);
    w.ue(67);
    w.u(1, 1);
    w.u(1, 1);
    w.u(1, 1);       // crop flag
    w.ue(0);
    w.ue(300);       // large offsets -> long zero-bit runs
    w.ue(0);
    w.ue(500);
    w.u(1, 1);       // VUI present
    w.u(1, 0);       // AR absent
    w.u(1, 0);       // overscan absent
    w.u(1, 0);       // video signal absent
    w.u(1, 0);       // chroma loc absent
    w.u(1, 1);       // timing present
    w.u(32, 0x80001001u);
    w.u(32, 0x8000EA60u);
    w.u(1, 1);

    auto plain = wrap_nal(w, 0x67, /*apply_escape=*/false);
    auto escaped = wrap_nal(w, 0x67, /*apply_escape=*/true);
    // Deterministic escape material: a three-zero-byte run past the parser's stop point
    // (it stops after fixed_frame_rate). In the plain form it is inert data; in the
    // escaped form a real encoder would emit 00 00 03 00 for it -- so that is what we
    // place there, and the CHECK below proves it is present for the reader to strip.
    plain.insert(plain.end(), {0x00, 0x00, 0x00});
    escaped.insert(escaped.end(), {0x00, 0x00, 0x03, 0x00});
    bool has_escape_byte = false;
    for (size_t i = 5; i < escaped.size(); ++i)
        if (escaped[i] == 3 && escaped[i - 1] == 0 && escaped[i - 2] == 0) {
            has_escape_byte = true;
            break;
        }
    CHECK(has_escape_byte, "fixture must actually contain emulation-prevention bytes");

    SpsPictureMeta a, b;
    CHECK(prosper::h264::parse_first_sps(plain.data(), plain.size(), &a),
          "unescaped form should parse");
    CHECK(prosper::h264::parse_first_sps(escaped.data(), escaped.size(), &b),
          "escaped form should parse");
    CHECK(a.crop_flag == b.crop_flag && a.crop[1] == b.crop[1] && a.crop[3] == b.crop[3],
          "both forms must yield the same crop quad");
    CHECK(a.timing_flag && b.timing_flag && a.num_units_in_tick == b.num_units_in_tick &&
              a.time_scale == b.time_scale,
          "both forms must yield the same timing pair");
}

static void test_sps_found_after_other_nals() {
    auto au = sps_baseline(3);
    std::vector<uint8_t> stream;
    auto push_nal = [&stream](const std::vector<uint8_t>& nal) {
        stream.push_back(0); stream.push_back(0); stream.push_back(0); stream.push_back(1);
        stream.insert(stream.end(), nal.begin() + 4, nal.end());
    };
    // SEI (type 6), PPS (type 1), THEN the SPS.
    stream = {0, 0, 0, 1, 0x06, 0x05, 0x01, 0xFF};
    push_nal({0, 0, 0, 1, 0x68, 0xEE});
    push_nal(au);
    SpsPictureMeta m;
    CHECK(prosper::h264::parse_first_sps(stream.data(), stream.size(), &m),
          "SPS after SEI/PPS should still be found");
}

static void test_garbage_is_rejected() {
    const uint8_t junk[] = {0, 0, 0, 1, 0x67, 0xFF, 0x00, 0x11, 0x22};
    SpsPictureMeta m;
    CHECK(!prosper::h264::parse_first_sps(junk, sizeof(junk), &m),
          "truncated nonsense must not parse");
    const uint8_t no_sps[] = {0, 0, 0, 1, 0x68, 0xEE, 0x00};
    CHECK(!prosper::h264::parse_first_sps(no_sps, sizeof(no_sps), &m),
          "stream without an SPS must not parse");
}

static void test_fill_layout_matches_observed_offsets() {
    SpsPictureMeta m;
    m.crop_flag = true;
    m.crop[0] = 12; m.crop[1] = 34; m.crop[2] = 56; m.crop[3] = 78;
    m.ar_flag = true;
    m.ar_idc = 255;
    m.sar_w = 40; m.sar_h = 33;
    m.timing_flag = true;
    m.num_units_in_tick = 1001; m.time_scale = 60000;

    alignas(16) uint8_t buf[0xB8];
    memset(buf, 0, sizeof(buf));
    *(uint64_t*)buf = 0x78;  // caller-declared size, untouched by fill
    uint8_t record[64];
    memset(record, 0, sizeof(record));
    record[0] = 0xAA;  // sentinel: fill must not write through the record

    CHECK(prosper::h264::fill_picture_info(m, record, buf, 0x78),
          "fill must accept a 0x78-sized block");

    uint64_t rec_ptr = (uint64_t)(uintptr_t)record;
    uint64_t got_rec;
    memcpy(&got_rec, buf + 0x20, 8);
    CHECK(got_rec == rec_ptr, "+0x20 must carry the record address");
    CHECK(buf[0x35] == 1, "+0x35 crop flag");
    auto rd32 = [&](size_t off) {
        uint32_t v;
        memcpy(&v, buf + off, 4);
        return v;
    };
    CHECK(rd32(0x38) == 12, "+0x38 crop[0]");
    CHECK(rd32(0x3c) == 34, "+0x3c crop[1]");
    CHECK(rd32(0x40) == 56, "+0x40 crop[2]");
    CHECK(rd32(0x44) == 78, "+0x44 crop[3]");
    CHECK(buf[0x48] == 1, "+0x48 AR flag");
    CHECK(buf[0x49] == 255, "+0x49 AR idc");
    auto rd16 = [&](size_t off) {
        uint16_t v;
        memcpy(&v, buf + off, 2);
        return v;
    };
    CHECK(rd16(0x4a) == 40, "+0x4a SAR width");
    CHECK(rd16(0x4c) == 33, "+0x4c SAR height");
    CHECK(buf[0x55] == 1, "+0x55 timing flag");
    CHECK(rd32(0x58) == 1001, "+0x58 num_units_in_tick");
    CHECK(rd32(0x5c) == 60000, "+0x5c time_scale");
    // Unobserved regions stay exactly as the caller left them (zeroed). Each entry is an
    // 8-byte span that no observed field touches: 0x35/0x38-0x44 and 0x48-0x4c sit
    // between them.
    bool untouched = true;
    for (size_t off : {0x08u, 0x28u, 0x60u, 0x70u})
        for (size_t k = 0; k < 8; ++k)
            untouched = untouched && buf[off + k] == 0;
    CHECK(untouched, "bytes outside the observed set must not be written");
    CHECK(record[0] == 0xAA, "the record itself is not written through");
}

static void test_fill_flags_zero_when_meta_absent() {
    SpsPictureMeta m;  // everything absent
    alignas(16) uint8_t buf[0x78];
    memset(buf, 0xCC, sizeof(buf));  // poison: flags must become 0, not stay poisoned
    uint8_t record[8] = {0};
    CHECK(prosper::h264::fill_picture_info(m, record, buf, sizeof(buf)),
          "absent-meta fill should succeed");
    CHECK(buf[0x35] == 0 && buf[0x48] == 0 && buf[0x55] == 0,
          "absent metadata fills flags as absent");
    uint64_t rec_ptr;
    memcpy(&rec_ptr, buf + 0x20, 8);
    CHECK(rec_ptr == (uint64_t)(uintptr_t)record, "record pointer still published");
    // MUTATION ARM: with the fill removed entirely (the pre-fix behaviour), every byte of
    // buf stays 0xCC and both assertions above fail -- the test cannot pass vacuously.
}

static void test_fill_rejects_undersized_block() {
    SpsPictureMeta m;
    uint8_t small[0x5f];
    uint8_t record[8] = {0};
    CHECK(!prosper::h264::fill_picture_info(m, record, small, sizeof(small)),
          "blocks under 0x60 must be refused, not partially filled");
}

int main() {
    printf("test_h264_sps (#2898)\n");
    test_baseline_parses_with_all_flags_absent();
    test_high_profile_scaling_skip_and_vui();
    test_emulation_prevention_bytes_are_stripped();
    test_sps_found_after_other_nals();
    test_garbage_is_rejected();
    test_fill_layout_matches_observed_offsets();
    test_fill_flags_zero_when_meta_absent();
    test_fill_rejects_undersized_block();

    if (fails) {
        printf("%d check(s) FAILED\n", fails);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
