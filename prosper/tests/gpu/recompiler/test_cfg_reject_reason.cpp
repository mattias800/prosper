// A Wave64 CFG terminal record must retain the actual placeholder cause and roots.
// Losing that cause sends a refusal census to missing lowerings instead of missing launch inputs.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
struct Result {
    std::vector<uint32_t> module;
    std::string reason;
};
Result compile(uint32_t width, bool supplied, uint64_t address) {
    const std::vector<uint32_t> words{
        supplied ? 0x7e020204u : 0x7e020211u, // v1 = supplied s4 or unknown s17
        0x7d840301u, // VCC = (v1 == v1)
        0xbf810000u,
    };
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = width;
    c.user_sgprs.resize(16);
    c.force_exchange_dispatcher = width == 64; // the exact exchange retry supports Wave64 only
    prosper::test::complete_compute_launch(c);
    c.dispatch_launch.known = 0; // the missing scratch-offset input must remain Hard
    auto module = recompile_compute(words.data(), words.size(), nullptr, c,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
void expect_named_refusal(uint32_t width, uint64_t address) {
    const auto r = compile(width, false, address);
    EXPECT_TRUE(r.module.empty());
    EXPECT_NE(r.reason.find(width == 64 ? "cfg-recompile-reject" : "recompile-reject"),
              std::string::npos)
        << r.reason;
    EXPECT_NE(r.reason.find("reason=fabricated-fresh-mask-source"), std::string::npos) << r.reason;
    EXPECT_NE(r.reason.find("origin=unknown-launch word=s17 at=entry"), std::string::npos)
        << r.reason;
    EXPECT_NE(r.reason.find("pc=1 "), std::string::npos) << r.reason;
    EXPECT_NE(r.reason.find("words=7d840301"), std::string::npos) << r.reason;
}
} // namespace

TEST(CfgRejectReason, Wave64RetainsTheActualFreshMaskCauseAndRoot) {
    expect_named_refusal(64, 0x4808200100ull);
}
TEST(CfgRejectReason, Wave32KeepsTheExistingStraightLineCauseAndRoot) {
    expect_named_refusal(32, 0x4808200200ull);
}
TEST(CfgRejectReason, SuppliedSourceIsAcceptedAtBothWidths) {
    for (uint32_t width : {32u, 64u}) {
        const auto r = compile(width, true, 0x4808200300ull + width);
        EXPECT_FALSE(r.module.empty()) << r.reason;
    }
}
