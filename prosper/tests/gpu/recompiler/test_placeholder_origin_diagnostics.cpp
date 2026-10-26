#include "gpu/recompiler/rdna2_diagnostic_identity.hpp"
#include <string_view>
// Origin details survive retry transactions without truncation or stale-attempt contamination.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_diagnostic_attempt.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace {
using namespace prosper::gpu;
RecompileDiagnosticContext context(uint64_t address, const std::string& roots) {
    return {RecompileDiagnosticStage::Compute, address, std::make_shared<const std::string>(roots)};
}
} // namespace

TEST(PlaceholderOriginDiagnostics, EveryRootSurvivesDeferredTerminalFormatting) {
    std::string roots;
    for (int reg = 0; reg < 200; ++reg) {
        if (reg) roots += ';';
        roots += "origin=unloaded word=v" + std::to_string(reg) + " at=entry";
    }
    constexpr uint64_t address = 0x5497a00400ull;
    TerminalRejectCapture capture;
    RecompileDiagnosticAttempt attempt;
    log_recompile_diagnostic(context(address, roots), "recompile-reject", "terminal",
                             "reason=fabricated-fresh-mask-source pc=3");
    attempt.commit();
    const auto records = capture.take();
    ASSERT_EQ(records.size(), 1u);
    const auto expected = "placeholder-roots=[" + roots + "]";
    EXPECT_NE(records.front().second.find(expected), std::string::npos);
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find(expected), std::string::npos);
    EXPECT_EQ(reason.find("placeholder-roots=", reason.find("placeholder-roots=") + 1),
              std::string::npos);
}

TEST(PlaceholderOriginDiagnostics, UnrelatedRefusalsDoNotInheritAnOrigin) {
    constexpr uint64_t address = 0x5497a00500ull;
    log_recompile_diagnostic(context(address, "origin=spill word=spill-v20:0 at=entry"),
                             "recompile-reject", "terminal", "reason=unresolved-operand pc=3");
    EXPECT_EQ(last_terminal_reject_reason(address).find("origin="), std::string::npos);
}

TEST(PlaceholderOriginDiagnostics, DiscardedAttemptsPublishNoRoots) {
    constexpr uint64_t address = 0x5497a00600ull;
    TerminalRejectCapture capture;
    {
        RecompileDiagnosticAttempt discarded;
        log_recompile_diagnostic(context(address, "origin=spill word=spill-v20:0 at=entry"),
                                 "recompile-reject", "terminal",
                                 "reason=fabricated-fresh-mask-source pc=3");
    }
    RecompileDiagnosticAttempt selected;
    log_recompile_diagnostic(context(address, "origin=unloaded word=v6 at=entry"),
                             "recompile-reject", "terminal",
                             "reason=fabricated-fresh-mask-source pc=3");
    selected.commit();
    ASSERT_EQ(capture.take().size(), 1u);
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("origin=unloaded"), std::string::npos);
    EXPECT_EQ(reason.find("origin=spill"), std::string::npos);
}

TEST(PlaceholderOriginDiagnostics, LoopRefusalsCarryTheirRootOrigins) {
    constexpr uint64_t address = 0x5497a00800ull;
    log_recompile_diagnostic(context(address, "origin=spill word=spill-v20:3 at=entry"),
                             "recompile-reject", "terminal",
                             "loop-carried s10 is fabricated on the back edge but was read as "
                             "lane bits in the loop (header pc=3)");
    EXPECT_NE(last_terminal_reject_reason(address).find("origin=spill word=spill-v20:3 at=entry"),
              std::string::npos);
}

TEST(PlaceholderOriginDiagnostics, ReplayIdentityExcludesOnlyTheDiagnosticEnvelope) {
    const std::string reason = "fabricated-fresh-mask-source pc=3 placeholder-roots=["
                               "origin=unloaded word=v8 at=entry]";
    const std::string_view identity = recompile_refusal_identity(reason);
    EXPECT_EQ(identity, "fabricated-fresh-mask-source pc=3");
    EXPECT_EQ(identity.data(), reason.data());
    EXPECT_EQ(
        recompile_refusal_identity("loop-carried s10 is fabricated on the back edge "
                                   "placeholder-roots=[origin=spill word=spill-v20:3 at=entry]"),
        "loop-carried s10 is fabricated on the back edge");
}

TEST(PlaceholderOriginDiagnostics, ReplayIdentityPreservesTheReasonSiteAndMalformedSuffix) {
    EXPECT_NE(
        recompile_refusal_identity("fabricated-fresh-mask-source pc=3 placeholder-roots=[x]"),
        recompile_refusal_identity("fabricated-fresh-mask-source pc=4 placeholder-roots=[x]"));
    for (const std::string_view reason :
         {"unresolved-operand pc=3 placeholder-roots=[x]",
          "fabricated-fresh-mask-source pc=3 placeholder-roots=[unterminated"})
        EXPECT_EQ(recompile_refusal_identity(reason), reason);
}
