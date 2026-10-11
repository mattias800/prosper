// #4899: a fresh SAVEEXEC mask must retire an earlier opaque M0 save before loop PHIs.
// Unrelated and still-live opaque words remain forbidden as numeric input.
#include "fixtures/saveexec_m0.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace fixture = prosper::test::saveexec_m0;
using namespace prosper::gpu;

TEST(SaveexecM0, LowWordOverwriteStartsFreshLoopMask) {
    const auto code = fixture::loop(14);
    for (uint32_t native : {0u, 64u})
        EXPECT_FALSE(fixture::compile(code, fixture::address(), native).empty());
}

TEST(SaveexecM0, HighWordOverwriteStartsFreshLoopMask) {
    const auto code = fixture::loop(15);
    for (uint32_t native : {0u, 64u})
        EXPECT_FALSE(fixture::compile(code, fixture::address(), native).empty());
}

TEST(SaveexecM0, PlainDataControlCompilesThroughSameLoop) {
    auto code = fixture::loop();
    code[0] = 0xbe8e0387u; // ordinary s_mov_b32 s14,7; tracked M0 round trip
    EXPECT_FALSE(fixture::compile(code, fixture::address()).empty());
}

TEST(SaveexecM0, UnrelatedOpaqueWordCannotBecomeNumericData) {
    const std::vector<uint32_t> code{
        0xbe82037cu, // opaque s2=M0
        0xbe8e247eu, // overwrite only s[14:15], preserves s2's token
        0xbefe040eu, // restore EXEC
        0x4a020002u, // v_add_nc_u32 v1,s2,v0 must refuse
        0xbf810000u,
    };
    auto control = code;
    control[0] = 0xbe820387u;
    ASSERT_FALSE(fixture::compile(control, fixture::address()).empty());
    const uint64_t address = fixture::address();
    EXPECT_TRUE(fixture::compile(code, address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("pc=3"), std::string::npos);
}

TEST(SaveexecM0, UnreplacedOpaqueWordStillRefusesAtLoopHeader) {
    auto code = fixture::loop(14);
    code[3] = 0xbe90247eu; // save EXEC into s[16:17], leaves s14 opaque
    code[4] = 0xbefe0410u;
    const uint64_t address = fixture::address();
    EXPECT_TRUE(fixture::compile(code, address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("entry-M0 token is loop-carried (s14)"),
              std::string::npos);
}
