// #5225: five existing nonfused F32 encodings share finite normal inputs, independent expected
// rounding, and real guest buffer loads/stores. No half or mixed form is approximated here.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>
namespace prosper::test::mad_rounding {
using namespace prosper::test::ir;
using prosper::gpu::Rdna2Format;
enum class Addend { Source2, Destination, Literal };
struct Encoding {
    const char* name;
    Rdna2Format format;
    uint32_t opcode;
    Addend addend;
    bool literal_multiplier;
};
inline constexpr std::array<Encoding, 5> kEncodings = {{
    {"mac_e32", Rdna2Format::VOP2, 0x1f, Addend::Destination, false},
    {"madmk", Rdna2Format::VOP2, 0x20, Addend::Source2, true},
    {"madak", Rdna2Format::VOP2, 0x21, Addend::Literal, false},
    {"mac_e64", Rdna2Format::VOP3, 0x11f, Addend::Destination, false},
    {"mad_e64", Rdna2Format::VOP3, 0x141, Addend::Source2, false},
}};
inline constexpr uint32_t kLiteralMultiplier = 0x3f7ffffe, kLiteralAddend = 0xbf800000;
struct Shape {
    // 0 registers; 1 inline SRC0; 2 SGPR SRC0; 3 SRC0 aliases VDST; 4 literal SRC0;
    // 5 literal SRC2 (e64 MAD only).
    uint32_t form = 0, abs = 0, neg = 0, omod = 0;
    bool clamp = false;
};
inline bool supports(const Encoding& e, Shape s) {
    if (e.format == Rdna2Format::VOP2) {
        if (s.abs || s.neg || s.omod || s.clamp || s.form == 5) return false;
        return !((e.literal_multiplier || e.addend == Addend::Literal) && s.form == 4);
    }
    return s.form != 5 || e.addend == Addend::Source2;
}
inline std::vector<uint32_t> words(const Encoding& e, Shape s = {}) {
    const uint32_t src0 = s.form == 1   ? 242u
                          : s.form == 2 ? 0u
                          : s.form == 3 ? 256u + kVdst
                          : s.form == 4 ? 255u
                                        : 256u + kVsrc0;
    if (e.format == Rdna2Format::VOP2) {
        std::vector<uint32_t> out = {vop2(e.opcode, kVdst, src0)};
        if (e.literal_multiplier)
            out.push_back(kLiteralMultiplier);
        else if (e.addend == Addend::Literal)
            out.push_back(kLiteralAddend);
        else if (s.form == 4)
            out.push_back(0x3f800001);
        return out;
    }
    const uint32_t src2 = e.addend == Addend::Destination ? 0u : s.form == 5 ? 255u : 256u + kVsrc2;
    std::vector<uint32_t> out = {
        0xd0000000u | (e.opcode << 16) | kVdst | (s.abs << 8) | (s.clamp ? 1u << 15 : 0u),
        src0 | ((256u + kVsrc1) << 9) | (src2 << 18) | (s.omod << 27) | (s.neg << 29)};
    if (s.form == 4 || s.form == 5) out.push_back(s.form == 4 ? 0x3f800001 : kLiteralAddend);
    return out;
}
inline LaneInputs inputs(const Encoding* encoding = nullptr) {
    auto in = lane_inputs(5225);
    const uint32_t aa[] = {0x3f800001, 0x3fc00000, 0xbf800000, 0x40000000};
    const uint32_t bb[] = {0x3f7ffffe, 0x40000000, 0x40000000, 0x3f000000};
    const uint32_t cc[] = {0xbf800000, 0x3f000000, 0x40400000, 0xbf800000};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = aa[lane % 4];
        in.s1[lane] = encoding && encoding->literal_multiplier ? cc[lane % 4] : bb[lane % 4];
        in.old_d[lane] = cc[lane % 4];
        in.s2.push_back(cc[lane % 4]);
    }
    // Lane 4 is an active cancellation witness; lane 0 is its inactive counterpart.
    in.active[4] = true;
    return in;
}
// Keep the canonical cancellation dataset unchanged. Clamped controls avoid the oracle's
// independently tested zero/zero NMin/NMax tie, whose signed result is outside its domain.
inline LaneInputs inputs(const Encoding& e, Shape s) {
    auto in = inputs(&e);
    if (s.clamp)
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            if (lane % 4 == 0) in.s0[lane] = 0x3f800002;
            if (lane % 4 == 3) in.s0[lane] = 0x40400000;
        }
    return in;
}
inline LaneProgram program(const Encoding& e, Shape s, uint32_t wave, uint32_t rsrc1) {
    auto out = lane_program(words(e, s), 64, wave, true);
    out.config.compute_pgm_rsrc1 = rsrc1;
    if (s.form == 2) out.code.insert(out.code.begin(), {0xbe8003ffu, 0x3f800001u});
    return out;
}
inline std::array<uint32_t, 3> sources(const Encoding& e, Shape s, const LaneInputs& in,
                                       size_t lane) {
    uint32_t a = s.form == 1                  ? 0x3f800000u
                 : s.form == 2 || s.form == 4 ? 0x3f800001u
                 : s.form == 3                ? in.old_d[lane]
                                              : in.s0[lane];
    uint32_t b = e.literal_multiplier ? kLiteralMultiplier : in.s1[lane];
    uint32_t c = e.addend == Addend::Destination              ? in.old_d[lane]
                 : e.addend == Addend::Literal || s.form == 5 ? kLiteralAddend
                 : e.literal_multiplier                       ? in.s1[lane]
                                                              : in.s2[lane];
    std::array<uint32_t, 3> out = {a, b, c};
    for (uint32_t slot = 0; slot < 3; ++slot) {
        if (s.abs & (1u << slot)) out[slot] &= 0x7fffffffu;
        if (s.neg & (1u << slot)) out[slot] ^= 0x80000000u;
    }
    return out;
}
inline uint32_t expected(const Encoding& e, Shape s, const LaneInputs& in, size_t lane,
                         uint32_t rsrc1) {
    if (!in.active[lane]) return in.old_d[lane];
    const auto bits = sources(e, s, in, lane);
    const float a = std::bit_cast<float>(bits[0]), b = std::bit_cast<float>(bits[1]),
                c = std::bit_cast<float>(bits[2]);
    // Materialize binary32 before addition. This oracle must never contract the non-fused arm.
    volatile float product = a * b;
    float result = product + c;
    if (s.omod && !(rsrc1 & ((0x20u << 12) | (1u << 23))))
        result *= s.omod == 1 ? 2.0f : s.omod == 2 ? 4.0f : 0.5f;
    if (s.clamp) result = std::fmin(std::fmax(result, 0.0f), 1.0f);
    return std::bit_cast<uint32_t>(result);
}
inline std::vector<uint32_t> buffer(const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (const auto* a : {&in.s0, &in.s1, &in.old_d}) out.insert(out.end(), a->begin(), a->end());
    for (bool active : in.active) out.push_back(active ? 1u : 0u);
    out.insert(out.end(), in.s2.begin(), in.s2.end());
    return out;
}
inline std::vector<uint32_t> execute(const std::vector<uint32_t>& module, const LaneInputs& in,
                                     std::string& error) {
    bpermute_oracle::Interpreter vm(module);
    auto out = vm.run_buffers(64, {{3, buffer(in)}}, 3);
    error = vm.error;
    out.resize(error.empty() ? 64 : 0);
    return out;
}
} // namespace prosper::test::mad_rounding
