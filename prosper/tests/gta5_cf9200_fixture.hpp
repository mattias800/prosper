#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace prosper::test {

// Independent copy of the complete consumed production program. The recognizer cannot provide its
// expected bytes to the test: otherwise an accidental edit to both identity and behavior would make
// the same-site mutation arms circular.
inline constexpr char kGta5Cf9200ProgramHex[] =
"0300a0bf8902007e000108f4c00000fa7fc08cbf000030e00011018001ff09880000e000000388be81038abeff038bbe04620100703f8cbf8a22022c1105187e82222428f9228a7d80828606c0020228c124204abf020436c002064a8004847dff060436c0ffffff02030202010d047e0105207e0c2070e0000101800257027e8010ea81ff020210feff7f4f010f027e0105d67e6a6b6a936a6bea9a6b6a6a816a0c919a11106a936a22024e1002044ef902867d108c86061004867d020001d5010532006a0cea87f904867d108e86061004064ef902030280068686020001d502073a00016a28d51102320080048a7d800202500204eabe002070e000010180740074e000100280330087bfc102007ec102027ec102047ec102067eff02087e0000807fff020a7e0000807fff020c7e0000807fff020e7e000080ffff02107e000080ffff02127e000080ffff02147e0000807fff02167e0000807fff02187e0000807fff021a7e000080ffff021c7e000080ffff021e7e000080ff000204f4500000fa81038abeff038bbe046201107fc08cbf961d89be000078e000000280700078e000000280100078e000040280200078e000080280300078e0000c0280400078e000040280500078e000080280600078e0000c02809f22084a8402007e010046d7120309038302047e8102067e8508082c82020a7e002070e002010180002070e003100180002070e000040180002070e005120180000081bf";

consteval uint8_t gta5_cf9200_nibble(char value) {
    return value >= '0' && value <= '9' ? static_cast<uint8_t>(value - '0')
                                        : static_cast<uint8_t>(value - 'a' + 10);
}

template <size_t Characters>
consteval auto gta5_cf9200_decode_hex(const char (&hex)[Characters]) {
    static_assert((Characters - 1u) % 8u == 0u);
    std::array<uint32_t, (Characters - 1u) / 8u> words{};
    for (size_t byte = 0; byte < (Characters - 1u) / 2u; ++byte) {
        const uint8_t value = static_cast<uint8_t>(
            gta5_cf9200_nibble(hex[byte * 2u]) << 4u |
            gta5_cf9200_nibble(hex[byte * 2u + 1u]));
        words[byte / 4u] |= static_cast<uint32_t>(value) << ((byte % 4u) * 8u);
    }
    return words;
}

inline constexpr auto kGta5Cf9200Program =
    gta5_cf9200_decode_hex(kGta5Cf9200ProgramHex);
static_assert(kGta5Cf9200Program.size() == 135u);

inline constexpr std::array<uint32_t, 56> gta5_cf9200_source_and_output_null_root() {
    std::array<uint32_t, 56> root{};
    root[0] = 0x55du;
    root[1] = 0x565u;
    root[2] = root[3] = UINT32_MAX;
    root[16] = 0x5ddu;
    root[17] = 0x5e5u;
    root[18] = root[19] = UINT32_MAX;
    root[48] = 0xau;
    root[49] = 0x5204u;
    root[50] = root[51] = UINT32_MAX;
    return root;
}

} // namespace prosper::test
