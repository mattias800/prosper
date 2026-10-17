// #5225: raw half sources, real RDNA2 buffer programs and an independent exact RTE oracle.
#pragma once
#include <cstddef>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "fixtures/half_fma_reference.hpp"
#include "fixtures/half_fma_vectors.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace prosper::test::half_fma {
using namespace prosper::test::ir;
enum class Kind { Native, Packed, PackedMac };
inline constexpr std::array<Kind, 3> kKinds = {Kind::Native, Kind::Packed, Kind::PackedMac};
inline const char* name(Kind kind) {
    return kind == Kind::Native ? "native" : kind == Kind::Packed ? "packed" : "packed_mac";
}
struct Shape {
    uint32_t low = 0, high = 7, absolute = 0, neg_low = 0, neg_high = 0, form = 0, omod = 0;
    bool clamp = false;
};
// forms: VGPR, inline integer 1, inline float 1.0, genuine SGPR, SRC0 aliases VDST,
// literal SRC0, SRC2 aliases VDST. No encoding admission is changed by these tests.
inline std::vector<uint32_t> words(Kind kind, Shape s = {}) {
    const uint32_t src0 = s.form == 1   ? 129
                          : s.form == 2 ? 242
                          : s.form == 3 ? 0
                          : s.form == 4 ? 256 + kVdst
                          : s.form == 5 ? 255
                                        : 256 + kVsrc0;
    if (kind == Kind::PackedMac) {
        std::vector<uint32_t> out = {vop2(0x3c, kVdst, src0)};
        if (s.form == 5) out.push_back(0x3c01b838);
        return out;
    }
    const auto src2 = 256 + (s.form == 6 ? kVdst : kVsrc2);
    std::vector<uint32_t> out;
    if (kind == Kind::Native)
        out = {0xd4000000u | (0x34bu << 16) | kVdst | (s.absolute << 8) | (s.low << 11) |
                   (s.clamp ? 1u << 15 : 0),
               src0 | ((256 + kVsrc1) << 9) | (src2 << 18) | (s.omod << 27) | (s.neg_low << 29)};
    else
        out = {0xcc0e0000u | kVdst | (s.neg_high << 8) | (s.low << 11) | ((s.high >> 2) << 14) |
                   (s.clamp ? 1u << 15 : 0),
               src0 | ((256 + kVsrc1) << 9) | (src2 << 18) | ((s.high & 3) << 27) |
                   (s.neg_low << 29)};
    if (s.form == 5) out.push_back(0x3c01b838);
    return out;
}
inline LaneProgram program(Kind kind, Shape shape, uint32_t wave, uint32_t rsrc1 = 0xc0000) {
    auto out = lane_program(words(kind, shape), 64, wave, kind != Kind::PackedMac);
    out.config.compute_pgm_rsrc1 = rsrc1;
    if (shape.form == 3) out.code.insert(out.code.begin(), {0xbe8003ffu, 0x3c01b838u});
    return out;
}
inline LaneInputs inputs(Kind kind, uint32_t offset = 0) {
    auto in = lane_inputs(5225);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const size_t index = lane == 4 ? 0 : (lane + offset) % std::size(kHalfFmaVectors);
        const auto& low = kHalfFmaVectors[index];
        const auto& high = kHalfFmaVectors[(index + 1) % std::size(kHalfFmaVectors)];
        in.s0[lane] = low.a | (uint32_t(high.a) << 16);
        in.s1[lane] = low.b | (uint32_t(high.b) << 16);
        const auto addend = low.c | (uint32_t(high.c) << 16);
        in.s2.push_back(addend);
        in.old_d[lane] = kind == Kind::PackedMac ? addend : 0x35553a11u + uint32_t(lane);
    }
    in.active[4] = true;
    return in;
}
// Fresh clamp controls use strictly positive inputs; the legacy oracle's signed-zero
// NMin/NMax tie domain is independent of this static policy/byte-identity control.
inline LaneInputs inputs(Kind kind, Shape shape, uint32_t offset = 0) {
    auto in = inputs(kind, offset);
    if (shape.clamp)
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            in.s0[lane] = in.s1[lane] = 0x3c003c00;
            in.s2[lane] = 0x38003800;
            if (kind == Kind::PackedMac) in.old_d[lane] = 0x38003800;
        }
    return in;
}
inline uint32_t legacy_mode(uint32_t rsrc1) {
    return rsrc1 != UINT32_MAX && ((rsrc1 >> 12) & 0xcc) == 0xc0 ? rsrc1 | 0x4000 : rsrc1;
}
inline bool profile(uint32_t rsrc1, Shape shape) {
    return rsrc1 != UINT32_MAX && ((rsrc1 >> 12) & 0xcc) == 0xc0 && !shape.omod && !shape.clamp;
}
inline std::vector<uint32_t> buffer(const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (const auto* a : {&in.s0, &in.s1, &in.old_d}) out.insert(out.end(), a->begin(), a->end());
    for (bool active : in.active) out.push_back(active ? 1 : 0);
    if (!in.s2.empty()) out.insert(out.end(), in.s2.begin(), in.s2.end());
    return out;
}
inline std::vector<uint32_t> compile(Kind kind, Shape shape, uint32_t wave,
                                     uint32_t rsrc1 = 0xc0000) {
    const auto p = program(kind, shape, wave, rsrc1);
    return prosper::gpu::recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
}
inline std::vector<uint32_t> execute(const std::vector<uint32_t>& module, const LaneInputs& in,
                                     std::string& error) {
    bpermute_oracle::Interpreter vm(module);
    // This models the old Pack/Unpack reference only. Corrected raw integer results
    // bypass both operations; host Vulkan input/output denorm support is not assumed.
    vm.half_rte_preserve_denorm = true;
    auto out = vm.run_buffers(64, {{3, buffer(in)}}, 3);
    error = vm.error;
    out.resize(error.empty() ? 64 : 0);
    return out;
}
inline std::array<uint16_t, 3> sources(Kind kind, Shape s, const LaneInputs& in, size_t lane,
                                       uint32_t high) {
    uint32_t raw0 = s.form == 1                  ? 1u
                    : s.form == 2                ? (kind == Kind::PackedMac ? 0x3f800000u : 0x3c00u)
                    : s.form == 3 || s.form == 5 ? 0x3c01b838u
                    : s.form == 4                ? in.old_d[lane]
                                                 : in.s0[lane];
    const std::array<uint32_t, 3> raw = {
        raw0, in.s1[lane], kind == Kind::PackedMac || s.form == 6 ? in.old_d[lane] : in.s2[lane]};
    const uint32_t selectors = kind == Kind::PackedMac ? high * 7 : high ? s.high : s.low;
    const uint32_t neg = high ? s.neg_high : s.neg_low;
    std::array<uint16_t, 3> out;
    for (uint32_t k = 0; k < 3; ++k) {
        auto bits = uint16_t(raw[k] >> (((selectors >> k) & 1) * 16));
        if (s.absolute & (1u << k)) bits &= 0x7fff;
        if (neg & (1u << k)) bits ^= 0x8000;
        out[k] = bits;
    }
    return out;
}
inline uint32_t expected(Kind kind, Shape s, const LaneInputs& in, size_t lane, uint32_t fallback) {
    if (!in.active[lane]) return in.old_d[lane];
    auto out = kind == Kind::Native ? in.old_d[lane] : fallback;
    for (uint32_t high = 0; high < (kind == Kind::Native ? 1u : 2u); ++high) {
        const auto raw = sources(kind, s, in, lane, high);
        const auto want = reference(raw[0], raw[1], raw[2]);
        const uint32_t destination = kind == Kind::Native ? ((s.low >> 3) & 1) : high;
        const auto shift = destination * 16;
        const auto bits = want.eligible ? want.bits : uint16_t(fallback >> shift);
        out = (out & ~(0xffffu << shift)) | (uint32_t(bits) << shift);
    }
    return out;
}
}   // namespace prosper::test::half_fma
