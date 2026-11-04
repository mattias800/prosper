#pragma once
// LLVM gfx1030 Wave64 encodings; the producing profile explicitly selects guest width.
// A fresh unloaded marker selects the guarded retry. v0/v1 are unsupplied launch I/J;
// each writer replaces v0 only inside current EXEC, then copies it to an initialized sink.
#include "fixtures/native_pixel_parameter.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>

namespace prosper::test::inactive_old_word {
using Words = std::vector<uint32_t>;
enum class Writer { Interpolate, Multiply, Accumulate, Half, Missing };
inline constexpr std::array kAccumulate{0x56000404u};
inline constexpr std::array kCapture{0x7e0c0300u};
inline constexpr std::array kExport{0xf800180fu, 0x08060606u, 0xbf810000u};
inline constexpr std::array kHalf{0x7e0002f9u, 0x00061402u};
inline constexpr std::array kInterpolate{0xc8000000u, 0xc8010001u};
inline constexpr std::array kMissing{0x10000204u};
inline constexpr std::array kMultiply{0x10000404u};
inline constexpr std::array kPrefix{0xbefc0380u, 0xbe8403ffu, 0x3e000000u, 0x7e0c02ffu, 0x3e800000u,
                                    0x7e1002f2u, 0x7e0e037fu, 0xd7600028u, 0x00010107u, 0x7d840e28u,
                                    0xbe94047eu, 0x7d8604ffu, 0x40200000u, 0xbefe046au};
inline constexpr std::array kProvided{0x7e0002ffu, 0x3f600000u};
inline constexpr std::array kRestore{0xbefe0414u};

inline Words program(Writer writer, bool provided = false, bool expanded = false,
                     float threshold = 2.5f) {
    Words code;
    const auto append = [&](std::span<const uint32_t> words) {
        code.insert(code.end(), words.begin(), words.end());
    };
    if (provided) append(kProvided);
    append(kPrefix);
    code.at(code.size() - 2) = std::bit_cast<uint32_t>(threshold);
    switch (writer) {
        case Writer::Interpolate: append(kInterpolate); break;
        case Writer::Multiply: append(kMultiply); break;
        case Writer::Accumulate: append(kAccumulate); break;
        case Writer::Half: append(kHalf); break;
        case Writer::Missing: append(kMissing); break;
    }
    if (expanded) append(kRestore);
    append(kCapture);
    if (!expanded) append(kRestore);
    append(kExport);
    return code;
}
inline uint32_t writer_pc(Writer writer, bool provided = false) {
    return kPrefix.size() + (provided ? kProvided.size() : 0) +
           (writer == Writer::Interpolate ? 1u : 0u);
}
inline auto mapping() {
    return native_pixel_parameter::mapping(0, true);
}
} // namespace prosper::test::inactive_old_word
