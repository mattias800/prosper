// #4907: a proven phase split must not be preempted by ordinary portable READLANE's barrier
// refusal. Execute actual production modules, so common-barrier divergence and wrong peers fail.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/readlane_barrier.hpp"
#include "gpu/recompiler/rdna2_readlane_route.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace fixture = prosper::test::readlane_barrier;
using namespace prosper::gpu;
namespace {
uint32_t count(const std::vector<uint32_t>& module, uint32_t opcode) {
    uint32_t found = 0;
    for (size_t at = 5; at < module.size();) {
        const uint32_t size = module[at] >> 16u;
        if (!size || size > module.size() - at) return 0;
        found += (module[at] & 0xffffu) == opcode;
        at += size;
    }
    return found;
}

void execute(fixture::Options options = {}) {
    const auto p = fixture::program(options);
    const auto address = fixture::address();
    const auto module = fixture::compile(p, fixture::table(), address);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    EXPECT_GE(count(module, Op_Switch), options.barrier ? 2u : 1u);
    EXPECT_GT(count(module, Op_ControlBarrier), 2u);
    EXPECT_EQ(count(module, Op_GroupNonUniformShuffle), 0u);
    bpermute_oracle::Interpreter vm(module);
    const auto actual = vm.run_buffers(fixture::lanes, {{3u, fixture::input()}}, 3u);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, fixture::expected(options))
        << "every wave's peers, retained scalar words, LDS data and untouched input planes";
}

void refuse(fixture::Unsafe unsafe,
            std::string_view expected_reason = "portable-readlane-dispatcher-unsafe") {
    const auto p = fixture::program({.unsafe = unsafe});
    std::vector<Rdna2Inst> instructions;
    ASSERT_EQ(rdna2_walk(p.words.data(), p.words.size(), instructions), p.words.size());
    EXPECT_FALSE(can_retry_readlane_barrier_phases(instructions, {}));
    const auto address = fixture::address();
    EXPECT_TRUE(fixture::compile(p, fixture::table(), address).empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find(expected_reason), std::string::npos) << reason;
}
} // namespace

TEST(ReadlaneBarrier, BalancedPeerValuesCrossGuestLdsAndRegisterBoundary) {
    execute({.loop = false, .differing_peers = false});
}
TEST(ReadlaneBarrier, DifferentGuestWavesExecuteDifferentReadlaneSites) {
    execute({.loop = false});
}
TEST(ReadlaneBarrier, DifferentTripCountsRetainCompletedWavesAtCommonBarriers) {
    execute();
}
TEST(ReadlaneBarrier, InactiveSelectedPeersStillSupplyActualValues) {
    execute({.narrowed_readlane = true});
}
TEST(ReadlaneBarrier, ExistingBarrierFreeReadlaneRouteIsUnchanged) {
    execute({.barrier = false});
}
TEST(ReadlaneBarrier, CrossingEdgeRemainsRefused) {
    // The terminal edge is rejected by the counted-loop prelude before the READLANE route.
    // Pin that actual production refusal in addition to the failed phase-proof assertion.
    refuse(fixture::Unsafe::CrossingEdge, "counted-loop prelude cfg");
}
TEST(ReadlaneBarrier, NonuniformTerminalGuardRemainsRefused) {
    refuse(fixture::Unsafe::NonuniformGuard);
}
TEST(ReadlaneBarrier, PartialExtentDoesNotAuthorizeNonuniformOuterGuard) {
    const auto p = fixture::program({.unsafe = fixture::Unsafe::NonuniformGuard});
    auto config = fixture::config();
    config.exact_thread_extent = true;
    config.threads_x = fixture::lanes - 1u;
    const auto resources = fixture::table();
    const auto address = fixture::address();
    EXPECT_TRUE(recompile_compute(p.words.data(), p.words.size(), &resources, config,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_FALSE(last_terminal_reject_reason(address).empty());
}
TEST(ReadlaneBarrier, EarlyTrapRemainsRefused) {
    refuse(fixture::Unsafe::EarlyTrap);
}
TEST(ReadlaneBarrier, IndirectPcRemainsRefused) {
    refuse(fixture::Unsafe::IndirectPc);
}
TEST(ReadlaneBarrier, OpaqueM0CannotBecomeAValidZeroAtNextPhase) {
    refuse(fixture::Unsafe::M0);
}
TEST(ReadlaneBarrier, MixedWritelaneServiceIsOutsideTheNewRetry) {
    refuse(fixture::Unsafe::Writelane);
}
TEST(ReadlaneBarrier, IncomingOpaqueTokenIsNotPhaseAuthority) {
    const auto p = fixture::program();
    std::vector<Rdna2Inst> instructions;
    ASSERT_EQ(rdna2_walk(p.words.data(), p.words.size(), instructions), p.words.size());
    RegState state;
    EXPECT_TRUE(can_retry_readlane_barrier_phases(instructions, state));
    state.sreg_entry_m0.insert(30);
    EXPECT_FALSE(can_retry_readlane_barrier_phases(instructions, state));
}
TEST(ReadlaneBarrier, RelativeVectorSourceDoesNotGainM0Authority) {
    const auto p = fixture::program();
    std::vector<Rdna2Inst> instructions;
    ASSERT_EQ(rdna2_walk(p.words.data(), p.words.size(), instructions), p.words.size());
    auto& writer = instructions.front();
    const auto original = writer;
    writer.fmt = Rdna2Format::VOP1;
    writer.opcode = 0x43;
    EXPECT_FALSE(can_retry_readlane_barrier_phases(instructions, {}));
    writer = original;
    EXPECT_TRUE(can_retry_readlane_barrier_phases(instructions, {}));
}
TEST(ReadlaneBarrier, IndependentOracleDetectsWrongWaveAndWrongPeer) {
    const auto wanted = fixture::expected();
    const auto same_peer = fixture::expected({.differing_peers = false});
    EXPECT_NE(wanted[fixture::lanes * 3u + 64u], same_peer[fixture::lanes * 3u + 64u]);
    // A missing wave base would consume wave0's selected value in every other wave.
    const uint32_t wrong = fixture::data(64u) + fixture::trips[1] + fixture::data(63u) +
                           fixture::trips[0] + fixture::data(95u);
    EXPECT_NE(wanted[fixture::lanes * 3u + 64u], wrong);
}
