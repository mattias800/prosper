#pragma once
// LLVM gfx1030: a genuine per-lane POS_X input must become one current-EXEC scalar value.
// The independent unloaded READLANE/compare requests the guarded retry before READFIRST.
#include "fixtures/native_pixel_parameter.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace prosper::test::native_fragment_readfirst {
using Words = std::vector<uint32_t>;
inline Words program(bool first = true, bool high = false, float threshold = 2.5f,
                     bool marker = true) {
    const Words first_ordinary{
        0x7e0e037fu, 0xd7600028u, 0x00010107u, 0x7d840e28u, 0xbe94047eu, 0x7d8600ffu,
        0x40200000u, 0xbefe046au, 0x7e100500u, 0xbefe0414u, 0x7e040208u, 0x100404ffu,
        0x3e000000u, 0x7e0602f2u, 0xf800080fu, 0x03020202u, 0xbf810000u,
    };
    const Words first_high{
        0x7e0e037fu, 0xd7600028u, 0x00010107u, 0x7d840e28u, 0xbe94047eu, 0x7d8600ffu,
        0x40200000u, 0xbefe046au, 0x7ed60500u, 0xbefe0414u, 0x7e04026bu, 0x100404ffu,
        0x3e000000u, 0x7e0602f2u, 0xf800080fu, 0x03020202u, 0xbf810000u,
    };
    const Words fixed_ordinary{
        0x7e0e037fu, 0xd7600028u, 0x00010107u, 0x7d840e28u, 0xbe94047eu, 0x7d8600ffu,
        0x40200000u, 0xbefe046au, 0xd7600008u, 0x00010100u, 0xbefe0414u, 0x7e040208u,
        0x100404ffu, 0x3e000000u, 0x7e0602f2u, 0xf800080fu, 0x03020202u, 0xbf810000u,
    };
    const Words fixed_high{
        0x7e0e037fu, 0xd7600028u, 0x00010107u, 0x7d840e28u, 0xbe94047eu, 0x7d8600ffu,
        0x40200000u, 0xbefe046au, 0xd760006bu, 0x00010100u, 0xbefe0414u, 0x7e04026bu,
        0x100404ffu, 0x3e000000u, 0x7e0602f2u, 0xf800080fu, 0x03020202u, 0xbf810000u,
    };

    auto code = first ? (high ? first_high : first_ordinary) : (high ? fixed_high : fixed_ordinary);
    const auto at = std::find(code.begin(), code.end(), 0x40200000u);
    if (at == code.end()) throw std::runtime_error("missing fixture threshold");
    *at = std::bit_cast<uint32_t>(threshold);
    if (!marker) code.erase(code.begin(), code.begin() + 4);
    return code;
}
inline gpu::PixelSystemInputMapping mapping(bool missing = false) {
    const auto fields = (1u << 8) | (missing ? 1u << 1 : 0u);
    gpu::PixelSystemInputMapping result{fields, fields};
    result.parameter_launch = {true, 0};
    return result;
}
} // namespace prosper::test::native_fragment_readfirst
