// Dependency re-emission must preserve successful routes and cannot grant Arbitrary admission.
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;
PlaceholderRoots roots(PlaceholderOrigin origin) {
    return {{origin, {PlaceholderFile::Scalar, 60, -1}, UINT32_MAX}};
}
} // namespace

TEST(ScalarDependencyRetry, OriginalSuccessKeepsItsBytesAndDoesNotRetry) {
    unsigned attempts = 0;
    const Words original{1, 2, 3};
    const auto result = recompile_with_arbitrary_placeholders({}, [&] {
        ++attempts;
        EXPECT_FALSE(allow_packed_scalar_merge());
        return Words(original);
    });
    EXPECT_EQ(result, original);
    EXPECT_EQ(attempts, 1u);
}

TEST(ScalarDependencyRetry, RepairDoesNotGrantArbitraryAdmission) {
    unsigned attempts = 0;
    const Words repaired{4, 5, 6};
    const auto result = recompile_with_arbitrary_placeholders({}, [&] {
        if (++attempts == 1) {
            EXPECT_FALSE(allow_packed_scalar_merge());
            return Words{};
        }
        EXPECT_TRUE(scalar_dependency_repair_active());
        EXPECT_TRUE(allow_packed_scalar_merge());
        EXPECT_FALSE(arbitrary_placeholder_retry_active());
        SpirvCompute b;
        EXPECT_FALSE(admit_arbitrary_placeholder_read(b, 7, ArbitraryPlaceholderRead::FreshMask,
                                                      roots(PlaceholderOrigin::Unloaded)));
        EXPECT_TRUE(b.debug.empty());
        return Words(repaired);
    });
    EXPECT_EQ(result, repaired);
    EXPECT_EQ(attempts, 2u);
}

TEST(ScalarDependencyRetry, AnUnloadedReadStillNeedsTheSeparateAdmissionAttempt) {
    unsigned attempts = 0;
    const auto result = recompile_with_arbitrary_placeholders({}, [&] {
        ++attempts;
        const bool packed = allow_packed_scalar_merge();
        SpirvCompute b;
        const bool admitted = admit_arbitrary_placeholder_read(
            b, 7, ArbitraryPlaceholderRead::FreshMask, roots(PlaceholderOrigin::Unloaded));
        if (!packed || !admitted) return Words{};
        Words module{0x07230203u, 0x00010300u, 0, 1, 0};
        module.insert(module.end(), b.debug.begin(), b.debug.end());
        return module;
    });
    EXPECT_EQ(attempts, 3u);
    EXPECT_EQ(arbitrary_placeholder_reads(result).fresh_mask, 1u);
}

TEST(ScalarDependencyRetry, AHardRootKeepsOnlyTheFinalRefusal) {
    constexpr uint64_t address = 0x5521cc9000ull;
    const RecompileDiagnosticContext context{RecompileDiagnosticStage::Compute, address};
    for (const auto kind : {PlaceholderOrigin::LaunchInput, PlaceholderOrigin::Spill,
                            PlaceholderOrigin::UnknownLaunch, PlaceholderOrigin::LostDefinition}) {
        unsigned attempts = 0;
        TerminalRejectCapture capture;
        const auto result = recompile_with_arbitrary_placeholders(context, [&] {
            const bool first = ++attempts == 1;
            const bool packed = allow_packed_scalar_merge();
            EXPECT_EQ(packed, !first);
            SpirvCompute b;
            EXPECT_FALSE(admit_arbitrary_placeholder_read(b, 7, ArbitraryPlaceholderRead::FreshMask,
                                                          roots(kind)));
            log_recompile_diagnostic(context, "recompile-reject", "terminal",
                                     first ? "original route" : "actual Hard root");
            return Words{};
        });
        EXPECT_TRUE(result.empty());
        EXPECT_EQ(attempts, 2u);
        const auto records = capture.take();
        ASSERT_EQ(records.size(), 1u);
        EXPECT_NE(records.front().second.find("actual Hard root"), std::string::npos);
        EXPECT_EQ(records.front().second.find("original route"), std::string::npos);
    }
}

TEST(ScalarDependencyRetry, ANamelessFailedRepairKeepsTheOriginalCause) {
    constexpr uint64_t address = 0x5521cc9100ull;
    const RecompileDiagnosticContext context{RecompileDiagnosticStage::Compute, address};
    TerminalRejectCapture capture;
    unsigned attempts = 0;
    const auto result = recompile_with_arbitrary_placeholders(context, [&] {
        const bool first = ++attempts == 1;
        EXPECT_EQ(allow_packed_scalar_merge(), !first);
        log_recompile_diagnostic(context, "recompile-reject",
                                 first ? "route-decline" : "consequent",
                                 first ? "original missing input" : "body failed");
        return Words{};
    });
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(attempts, 2u);
    const auto records = capture.take();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_NE(records.front().second.find("original missing input"), std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(address).find("original missing input"),
              std::string::npos);
}

TEST(ScalarDependencyRetry, ANamelessThirdAttemptKeepsTheLastMeaningfulCause) {
    constexpr uint64_t address = 0x5521cc9200ull;
    const RecompileDiagnosticContext context{RecompileDiagnosticStage::Compute, address};
    for (bool meaningful_repair : {false, true}) {
        TerminalRejectCapture capture;
        unsigned attempts = 0;
        const auto result = recompile_with_arbitrary_placeholders(context, [&] {
            ++attempts;
            const bool packed = allow_packed_scalar_merge();
            EXPECT_EQ(packed, attempts != 1);
            SpirvCompute b;
            if (attempts == 1)
                EXPECT_FALSE(admit_arbitrary_placeholder_read(
                    b, 7, ArbitraryPlaceholderRead::FreshMask, roots(PlaceholderOrigin::Unloaded)));
            const bool cause = attempts == 1 || (attempts == 2 && meaningful_repair);
            log_recompile_diagnostic(context, "recompile-reject",
                                     cause ? "route-decline" : "consequent",
                                     attempts == 1 ? "original cause" : "repair cause");
            return Words{};
        });
        EXPECT_TRUE(result.empty());
        EXPECT_EQ(attempts, 3u);
        const auto records = capture.take();
        ASSERT_EQ(records.size(), 1u);
        EXPECT_NE(
            records.front().second.find(meaningful_repair ? "repair cause" : "original cause"),
            std::string::npos);
    }
}
