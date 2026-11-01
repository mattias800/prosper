#pragma once
// LLVM gfx1030 fixture: an entry automatic parameter word serves native interpolation;
// the same SGPR then has a real numerical lifetime. A fixed READLANE relay forces the
// guarded Arbitrary retry, so an absent real input cannot hide behind legacy strict zeros.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace prosper::test::native_pixel_parameter {
using Words = std::vector<uint32_t>;
inline uint32_t rsrc2(uint32_t count) {
    return ((count & 31u) << 1) | ((count >> 5) << 27);
}
inline gpu::PixelSystemInputMapping mapping(uint32_t count = 30, bool smooth = false) {
    // P1/P2 consume the real perspective-center I/J pair in v0/v1. Flat P0 needs only POS_X.
    const uint32_t fields = (1u << 8) | (smooth ? 1u << 1 : 0u);
    gpu::PixelSystemInputMapping result{fields, fields};
    result.parameter_launch = {true, rsrc2(count)};
    return result;
}
inline Words program(uint32_t word = 30, bool smooth = false) {
    Words code{0xbefc031eu, 0xbe9e03f0u, 0xc81a0002u, 0x7e0e0314u, 0xd7600028u,
               0x00010107u, 0x7d840e28u, 0xd5010008u, 0x01aa0c80u, 0x7e12021eu,
               0x7e1402f2u, 0xf800080fu, 0x0a080908u, 0xbf810000u};
    code[0] = (code[0] & ~0xffu) | word;              // M0 = the entry word
    code[1] = (code[1] & ~(127u << 16)) | (word << 16); // overwrite that SGPR with 0.5
    code[9] = (code[9] & ~511u) | word;               // observe its new numerical lifetime
    if (smooth) {
        code[2] = 0xc8180000u;
        code.insert(code.begin() + 3, 0xc8190001u);
    }
    return code;
}
struct Compilation {
    Words module;
    std::string reason;
};
inline Compilation compile(const Words& code, const gpu::PixelSystemInputMapping& inputs,
                           bool wave32 = false) {
    static std::atomic<uint64_t> next{0x5543000000ull};
    const auto address = next.fetch_add(0x100);
    auto module =
        gpu::recompile_fragment(code.data(), code.size(), nullptr, &inputs, UINT32_MAX, nullptr,
                                wave32, {gpu::RecompileDiagnosticStage::Fragment, address}, {},
                                nullptr, {gpu::FloatTransportProfile::ExplicitNonFinite32});
    return {std::move(module), gpu::last_terminal_reject_reason(address)};
}
} // namespace prosper::test::native_pixel_parameter
