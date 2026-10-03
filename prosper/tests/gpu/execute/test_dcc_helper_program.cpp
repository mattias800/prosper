// AGC's colour-block utility pixel programs (dcc_helper_program.hpp). A DCC_DECOMPRESS draw ignores
// the pixel export, so a recognised helper must not run as colour; an ordinary shader under a stale
// DCC_DECOMPRESS mode must NOT be recognised (it is still drawn).
#include "gpu/execute/dcc_helper_program.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using prosper::gpu::is_agc_dcc_helper_program;

namespace {
bool helper(const std::vector<uint32_t>& code) {
    return is_agc_dcc_helper_program(code.data(), code.size());
}
} // namespace

TEST(DccHelperProgram, RecognisesBothObservedAgcHelpers) {
    // v_mov_b32 v0, 0 ; exp mrt0 v0, v0 (R,G) done vm ; s_endpgm -- the clear-RG helper.
    EXPECT_TRUE(helper({0x7e000280u, 0xf8001803u, 0x00000000u, 0xbf810000u}));
    // v_mov_b32 v0, 0x3c003c00 (fp16 1.0,1.0) ; exp mrt0 v0, v0 compr (RGBA) done vm ; s_endpgm --
    // Hollow Knight: Silksong's, which painted its scanout white when it ran as an ordinary draw.
    EXPECT_TRUE(helper({0x7e0002ffu, 0x3c003c00u, 0xf8001c0fu, 0x00000000u, 0xbf810000u}));
    // Words after s_endpgm (the next program, padding) do not matter.
    EXPECT_TRUE(helper({0x7e000280u, 0xf8001803u, 0x00000000u, 0xbf810000u, 0xdeadbeefu}));
}

TEST(DccHelperProgram, AnOrdinaryConstantColourShaderIsNotAHelper) {
    // test_gpu_execute's green fill: four constant moves and one RGBA MRT0 export. Structurally the
    // same family as the helpers -- which is exactly why recognition is by exact program: under a
    // stale DCC_DECOMPRESS mode this shader must still draw.
    EXPECT_FALSE(helper({0x7e000280u, 0x7e0202f2u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu,
                         0x03020100u, 0xbf810000u}));
    // The Silksong helper with a different literal (exports something other than 1.0) is not it.
    EXPECT_FALSE(helper({0x7e0002ffu, 0x3f800000u, 0xf8001c0fu, 0x00000000u, 0xbf810000u}));
    // The clear-RG helper exporting to MRT1 instead of MRT0 is not it.
    EXPECT_FALSE(helper({0x7e000280u, 0xf8001813u, 0x00000000u, 0xbf810000u}));
}

TEST(DccHelperProgram, NeverReadsPastTheBoundOrANullProgram) {
    const std::vector<uint32_t> full = {0x7e0002ffu, 0x3c003c00u, 0xf8001c0fu, 0x00000000u, 0xbf810000u};
    for (size_t n = 0; n < full.size(); ++n)
        EXPECT_FALSE(is_agc_dcc_helper_program(full.data(), n)) << "truncated to " << n << " dwords";
    EXPECT_TRUE(is_agc_dcc_helper_program(full.data(), full.size()));
    EXPECT_FALSE(is_agc_dcc_helper_program(nullptr, 16));
}
