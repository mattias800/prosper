// One Hard root vetoes admission; empty/lost/unknown slices must never become unloaded.
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_diagnostic_attempt.hpp"
#include "diagnostics/perf/perf_ledger.hpp"
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <string>
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "bpermute_spirv_oracle.hpp"
#include "diagnostics/perf/arbitrary_placeholder.hpp"
#include "diagnostics/perf/perf_alarms.hpp"
#include "diagnostics/exit_reports.hpp"
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
PlaceholderRoots roots(PlaceholderOrigin kind) {
    return {{PlaceholderOrigin::Unloaded, {PlaceholderFile::Vector, 8}},
            {kind, {PlaceholderFile::Scalar, 60}}};
}
} // namespace

TEST(ArbitraryPlaceholderPolicy, EveryOtherOriginIsAHardVeto) {
    EXPECT_TRUE(placeholder_roots_are_unloaded(roots(PlaceholderOrigin::Unloaded)));
    EXPECT_FALSE(placeholder_roots_are_unloaded({}));
    for (auto kind :
         {PlaceholderOrigin::LaunchInput, PlaceholderOrigin::Spill,
          PlaceholderOrigin::UnknownLaunch, PlaceholderOrigin::LostDefinition,
          PlaceholderOrigin::UnmodelledTransfer, PlaceholderOrigin::TraceBudget,
          PlaceholderOrigin::Unreachable, PlaceholderOrigin::UnprovedMemoryResult,
          PlaceholderOrigin::UnprovedStrongRead, PlaceholderOrigin::UnmodelledScalarPublication})
        EXPECT_FALSE(placeholder_roots_are_unloaded(roots(kind))) << static_cast<int>(kind);
}

TEST(ArbitraryPlaceholderPolicy, ASecondOperandCannotHideALostDefinition) {
    const std::vector<uint32_t> code{0x7e0c0281u, 0x7d841106u, 0xbf810000u};
    auto program = std::make_shared<PlaceholderOriginProgram>();
    ASSERT_EQ(rdna2_walk(code.data(), code.size(), program->instructions), code.size());
    program->entry_full_exec = program->vector_profile_known = true;
    RegState rs;
    rs.placeholder_origin_program = program;
    rs.vreg[6] = rs.vreg[8] = 1;
    rs.vreg_source_marks[6].placeholder = rs.vreg_source_marks[8].placeholder = true;
    const auto found = fresh_placeholder_roots(rs, program->instructions[1]);
    const auto text = format_placeholder_origins(found);
    EXPECT_NE(text.find("origin=lost-definition word=v6"), std::string::npos) << text;
    EXPECT_NE(text.find("origin=unloaded word=v8"), std::string::npos) << text;
    EXPECT_FALSE(placeholder_roots_are_unloaded(found));
}

TEST(ArbitraryPlaceholderPolicy, OnlyTheSelectedRetryPublishesReadMetadata) {
    unsigned attempts = 0;
    const auto result = recompile_with_arbitrary_placeholders({}, [&] {
        ++attempts;
        SpirvCompute b;
        if (!admit_arbitrary_placeholder_read(b, 7, ArbitraryPlaceholderRead::FreshMask,
                                              roots(PlaceholderOrigin::Unloaded)))
            return std::vector<uint32_t>{};
        std::vector<uint32_t> module{0x07230203u, 0x00010300u, 0, 1, 0};
        module.insert(module.end(), b.debug.begin(), b.debug.end());
        return module;
    });
    EXPECT_EQ(attempts, 2u);
    EXPECT_EQ(arbitrary_placeholder_reads(result).fresh_mask, 1u);
}

TEST(ArbitraryPlaceholderPolicy, OneChangedLaunchOrSpillRootKeepsTheReadRefused) {
    for (auto kind : {PlaceholderOrigin::LaunchInput, PlaceholderOrigin::Spill}) {
        unsigned attempts = 0;
        const auto result = recompile_with_arbitrary_placeholders({}, [&] {
            ++attempts;
            SpirvCompute b;
            if (attempts == 1) {
                EXPECT_FALSE(admit_arbitrary_placeholder_read(
                    b, 7, ArbitraryPlaceholderRead::FreshMask, roots(PlaceholderOrigin::Unloaded)));
                return std::vector<uint32_t>{};
            }
            EXPECT_FALSE(admit_arbitrary_placeholder_read(b, 7, ArbitraryPlaceholderRead::FreshMask,
                                                          roots(kind)));
            EXPECT_TRUE(b.debug.empty());
            return std::vector<uint32_t>{};
        });
        EXPECT_TRUE(result.empty());
        EXPECT_EQ(attempts, 2u);
    }
}

TEST(ArbitraryPlaceholderPolicy, ASuccessfulOriginalAttemptIsNeverRetried) {
    unsigned attempts = 0;
    const std::vector<uint32_t> original{1, 2, 3};
    const auto result = recompile_with_arbitrary_placeholders({}, [&] {
        ++attempts;
        SpirvCompute b;
        EXPECT_FALSE(admit_arbitrary_placeholder_read(b, 7, ArbitraryPlaceholderRead::FreshMask,
                                                      roots(PlaceholderOrigin::Unloaded)));
        return std::vector<uint32_t>(original);
    });
    EXPECT_EQ(result, original);
    EXPECT_EQ(attempts, 1u);
}

TEST(ArbitraryPlaceholderPolicy, SelectingANestedAttemptKeepsOnlyItsDiagnostics) {
    constexpr uint64_t address = 0x5497b00300ull;
    RecompileDiagnosticContext context{RecompileDiagnosticStage::Compute, address};
    TerminalRejectCapture capture;
    RecompileDiagnosticAttempt original;
    log_recompile_diagnostic(context, "recompile-reject", "terminal", "original record");
    RecompileDiagnosticAttempt selected;
    log_recompile_diagnostic(context, "recompile-reject", "terminal", "selected record");
    original.replace_with(selected);
    const auto records = capture.take();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_NE(records.front().second.find("selected record"), std::string::npos);
    EXPECT_EQ(records.front().second.find("original record"), std::string::npos);
}

TEST(ArbitraryPlaceholderPolicy, ARejectedRetryReportsItsActualHardCause) {
    constexpr uint64_t address = 0x5497b00500ull;
    const RecompileDiagnosticContext context{RecompileDiagnosticStage::Compute, address};
    TerminalRejectCapture capture;
    unsigned attempts = 0;
    const auto result = recompile_with_arbitrary_placeholders(context, [&] {
        SpirvCompute b;
        const bool first = ++attempts == 1;
        const auto input = roots(first ? PlaceholderOrigin::Unloaded : PlaceholderOrigin::Spill);
        EXPECT_FALSE(
            admit_arbitrary_placeholder_read(b, 7, ArbitraryPlaceholderRead::FreshMask, input));
        log_recompile_diagnostic(context, "recompile-reject", "terminal",
                                 first ? "original unloaded refusal" : "actual missing spill");
        return std::vector<uint32_t>{};
    });
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(attempts, 2u);
    const auto records = capture.take();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_NE(records.front().second.find("actual missing spill"), std::string::npos);
    EXPECT_EQ(records.front().second.find("original unloaded refusal"), std::string::npos);
}

TEST(ArbitraryPlaceholderPolicy, ActualCompilerAdmissionReachesTheWindowAndExitReport) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    namespace perf = prosper::diagnostics::perf;
    auto& ledger = perf::ledger();
    perf::AlarmEngine engine({.window_ns = 1'000'000'000, .log = nullptr});
    engine.on_flip(1'000'000'000, ledger, 60);
    const auto words = fixture::scalar_relay();
    const auto resources = fixture::table();
    constexpr uint64_t address = 0x5497b10400ull;
    testing::internal::CaptureStderr();
    const auto module = recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                                          {RecompileDiagnosticStage::Compute, address});
    prosper::diagnostics::flush_exit_reports();
    const auto output = testing::internal::GetCapturedStderr();
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    const auto reads = arbitrary_placeholder_reads(module);
    EXPECT_GT(reads.total(), 0u);
    const auto alarms = engine.on_flip(2'000'000'000, ledger, 60);
    const auto alarm = std::find_if(alarms.begin(), alarms.end(), [](const auto& row) {
        return std::string_view(row.rule) == "arbitrary-placeholder-admitted";
    });
    ASSERT_NE(alarm, alarms.end());
    EXPECT_EQ(alarm->value, 1);
    EXPECT_NE(alarm->detail.find("reads=" + std::to_string(reads.total())), std::string::npos);
    EXPECT_NE(output.find("rule=arbitrary-placeholder-admitted program=0x5497b10400"),
              std::string::npos);
    EXPECT_NE(output.find("summary observer=arbitrary-placeholder-admitted program=0x5497b10400"),
              std::string::npos);
}

TEST(ArbitraryPlaceholderPolicy, AnAtomicAtAnUnloadedAddressKeepsItsResultHard) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    auto words = fixture::Words{0xd8800000u, 0x06000007u, 0x7d840d06u};
    fixture::append_output(words);
    const auto resources = fixture::table();
    constexpr uint64_t address = 0x5497b00600ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("origin=unproved-memory-result word=v6 at=0"), std::string::npos)
        << reason;
}

TEST(ArbitraryPlaceholderPolicy, AnUnboundedLoopReadRetainsItsStrongObligation) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    auto words = fixture::loop_relay();
    words.insert(words.begin(), 0x7e0c0283u); // genuine preheader v6=3
    words.insert(words.begin() + 2, {0xd8d80000u, 0x08000006u}); // unbounded LDS address v6
    words[8] = 0xbf85fff9u; // back edge to the LDS read at header2
    const auto resources = fixture::table();
    constexpr uint64_t address = 0x5497b00700ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("origin=unproved-strong-read word=v6 at=2"), std::string::npos) << reason;
}

TEST(ArbitraryPlaceholderPolicy, AReadfirstLoopCannotInventTheMissingScalarPublication) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    const auto words = fixture::readfirst_loop_relay();
    const auto resources = fixture::table();
    constexpr uint64_t address = 0x5497b00900ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("origin=unloaded word=s60"), std::string::npos) << reason;
    EXPECT_NE(reason.find("origin=unmodelled-scalar-publication word=s20 at=2"), std::string::npos)
        << reason;
}

TEST(ArbitraryPlaceholderPolicy, AStraightReadfirstCannotInventTheMissingScalarPublication) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    const auto words = fixture::readfirst_straight_relay();
    const auto resources = fixture::table();
    constexpr uint64_t address = 0x5497b00a00ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    if (!module.empty()) {
        // Mutation witness: a missing safeguard produces all11, while a real broadcast
        // makes only lane0 equal to its original vector value.
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.supplied_push_constants = fixture::config().user_sgprs;
        const auto actual = vm.run(64, fixture::Words(64, 0xdeadbeefu));
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(actual, fixture::cmpx_expected());
    }
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("origin=unloaded word=s60"), std::string::npos) << reason;
    EXPECT_NE(reason.find("origin=unmodelled-scalar-publication word=s20 at=3"), std::string::npos)
        << reason;
}

TEST(ArbitraryPlaceholderPolicy, AnIndependentPublicationIsHardBeforeAndAfterAdmission) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    for (bool after : {false, true}) {
        const auto words = fixture::readfirst_independent_relay(after);
        const auto resources = fixture::table();
        const uint64_t address = 0x5497b00b00ull + after;
        const auto module =
            recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                              {RecompileDiagnosticStage::Compute, address});
        EXPECT_TRUE(module.empty());
        const auto reason = last_terminal_reject_reason(address);
        const auto expected =
            "origin=unmodelled-scalar-publication word=s20 at=" + std::to_string(after ? 3 : 1);
        EXPECT_NE(reason.find(expected), std::string::npos) << reason;
    }
}

TEST(ArbitraryPlaceholderPolicy, AnUnrelatedUnknownLaunchReadStaysHardBeforeAndAfterThePredicate) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    const auto resources = fixture::table();
    for (bool before : {false, true}) {
        const auto words = fixture::data_after_predicate(true, before);
        const uint64_t address = 0x5497b00e00ull + before;
        const auto module =
            recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                              {RecompileDiagnosticStage::Compute, address});
        EXPECT_TRUE(module.empty());
        EXPECT_NE(last_terminal_reject_reason(address).find("origin=unknown-launch word=s12"),
                  std::string::npos)
            << last_terminal_reject_reason(address);
    }
}

TEST(ArbitraryPlaceholderPolicy, ADirectExportCannotPublishAnUnsuppliedLaunchWordAfterAdmission) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 1; // raw perspective v0/v1 are real, unsupplied launch inputs
    for (bool unknown : {false, true}) {
        const auto words = fixture::pixel_export_after_predicate(unknown);
        const uint64_t address = 0x5497b00f00ull + unknown;
        const auto module =
            recompile_fragment(words.data(), words.size(), nullptr, &inputs, UINT32_MAX, nullptr,
                               false, {RecompileDiagnosticStage::Fragment, address});
        EXPECT_EQ(module.empty(), unknown) << last_terminal_reject_reason(address);
        if (!unknown) EXPECT_GT(arbitrary_placeholder_reads(module).total(), 0u);
        if (unknown)
            EXPECT_NE(last_terminal_reject_reason(address).find("origin=launch-input word=v0"),
                      std::string::npos)
                << last_terminal_reject_reason(address);
    }
}

TEST(ArbitraryPlaceholderPolicy, ADroppedDefinitionDoesNotBecomeUnloadedAtTheJoin) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    const auto words = fixture::dropped_definition_after_join();
    const auto resources = fixture::table();
    for (uint32_t path : {0u, 1u}) {
        auto launch = fixture::config();
        launch.user_sgprs[4] = path;
        launch.force_exchange_dispatcher = true;
        TerminalRejectCapture capture;
        const auto module = recompile_compute(words.data(), words.size(), &resources, launch);
        EXPECT_TRUE(module.empty());
        if (!module.empty()) {
            bpermute_oracle::Interpreter vm(module);
            vm.native_subgroup_width = 64;
            vm.supplied_push_constants = launch.user_sgprs;
            const auto actual = vm.run(64, fixture::Words(64, 0xdeadbeefu));
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            EXPECT_EQ(actual, fixture::Words(64, path ? 11 : 22));
        } else {
            std::string reason;
            for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
            EXPECT_NE(reason.find("origin=lost-definition word=s40 at=2"), std::string::npos)
                << reason;
        }
    }
}

TEST(ArbitraryPlaceholderPolicy, ImplicitScalarOperandsRemainHardOnTheRetry) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    const auto words = fixture::implicit_compare_after_predicate(true);
    const auto resources = fixture::table();
    constexpr uint64_t address = 0x5497b01200ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("origin=unknown-launch word=s12"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(ArbitraryPlaceholderPolicy, ABufferStoreCannotPublishAnUnsuppliedLaunchWordAfterAdmission) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 1;
    const auto resources = fixture::table();
    for (bool unknown : {false, true}) {
        const auto words = fixture::pixel_store_after_predicate(unknown);
        const uint64_t address = 0x5497b01400ull + unknown;
        const auto module =
            recompile_fragment(words.data(), words.size(), &resources, &inputs, UINT32_MAX, nullptr,
                               false, {RecompileDiagnosticStage::Fragment, address});
        EXPECT_EQ(module.empty(), unknown) << last_terminal_reject_reason(address);
        if (!unknown) EXPECT_GT(arbitrary_placeholder_reads(module).total(), 0u);
        if (unknown)
            EXPECT_NE(last_terminal_reject_reason(address).find("origin=launch-input word=v0"),
                      std::string::npos)
                << last_terminal_reject_reason(address);
    }
}

TEST(ArbitraryPlaceholderPolicy, ACarriedOrdinaryValueCannotHideAnUntrackedSpillRead) {
    namespace fixture = prosper::test::arbitrary_placeholder;
    const auto words = fixture::untracked_spill_in_loop();
    const auto resources = fixture::table();
    for (uint32_t path : {0u, 1u}) {
        auto launch = fixture::config();
        launch.user_sgprs[0] = 7;
        launch.user_sgprs[4] = path;
        TerminalRejectCapture capture;
        const auto module = recompile_compute(words.data(), words.size(), &resources, launch);
        EXPECT_TRUE(module.empty());
        std::string reason;
        for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
        EXPECT_NE(reason.find("origin=spill word=spill-v255:0"), std::string::npos) << reason;
    }
}
