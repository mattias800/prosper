// #4840: ordinary LOAD_* range packets and SET_* shadow writes obey CONTEXT_CONTROL's
// independent update bits. Dropping the control or reading packed pairs restores unrelated state.
// Synthetic packets pin the published AMD ABI; no platform initialization image is assumed.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/command_processor.hpp"

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace prosper::gpu {
namespace {
using Words = std::vector<uint32_t>;

void packet(Words& buffer, uint32_t opcode, std::initializer_list<uint32_t> payload,
            bool compute = false) {
    buffer.push_back(0xc0000000u | (uint32_t(payload.size() - 1) << 16) | (opcode << 8) |
                     (compute ? 2u : 0u));
    buffer.insert(buffer.end(), payload);
}

void control(Words& buffer, uint32_t load, uint32_t shadow) {
    packet(buffer, 0x28, {load, shadow});
}

void load(Words& buffer, uint32_t opcode, const void* base, uint32_t offset, uint32_t count,
          bool compute = false) {
    const uint64_t address = reinterpret_cast<uint64_t>(base);
    packet(buffer, opcode, {uint32_t(address), uint32_t(address >> 32), offset, count}, compute);
}

class Pm4Context : public ::testing::Test {
protected:
    void SetUp() override { prosper_gpu_drain_completion_writes(); }
    void TearDown() override { prosper_gpu_drain_completion_writes(); }
};

// Execution-time register effects need the controls at their packet, not the final submit's
// enables or bases. In particular, a count-zero LOAD changes a shadow base without loading values.
TEST_F(Pm4Context, SnapshotsRetainControlUpdatesAtEachDraw) {
    Words buffer;
    control(buffer, 0x80010000, 0x80010000);
    packet(buffer, 0x2d, {3, 2});
    control(buffer, 0, 0x80000000);
    packet(buffer, 0x2d, {3, 2});
    control(buffer, 0x80000000, 0);
    packet(buffer, 0x2d, {3, 2});
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.draws.size(), 3u);
    const auto& first = state.state_at_draw(0).context_control;
    const auto& second = state.state_at_draw(1).context_control;
    const auto& third = state.state_at_draw(2).context_control;
    EXPECT_TRUE(first.load_known);
    EXPECT_EQ(first.load, 0x10000u);
    EXPECT_EQ(first.shadow, 0x10000u);
    EXPECT_TRUE(second.load_known);
    EXPECT_EQ(second.load, 0x10000u);
    EXPECT_EQ(second.shadow, 0u);
    EXPECT_TRUE(third.load_known);
    EXPECT_EQ(third.load, 0u);
    EXPECT_EQ(third.shadow, 0u);
    EXPECT_NE(state.draws[0].state, state.draws[1].state);
    EXPECT_NE(state.draws[1].state, state.draws[2].state);
    EXPECT_TRUE(state.sh.empty());
}

TEST_F(Pm4Context, CountZeroRangeBaseChangesReachSnapshots) {
    std::array<uint32_t, 4> first_backing{}, second_backing{};
    Words buffer;
    control(buffer, 0x80010000, 0x80010000);
    load(buffer, 0x5f, first_backing.data(), 0, 0);
    packet(buffer, 0x2d, {3, 2});
    load(buffer, 0x5f, second_backing.data(), 0, 0);
    packet(buffer, 0x2d, {3, 2});
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.draws.size(), 2u);
    EXPECT_EQ(state.state_at_draw(0).context_control.bases[2],
              reinterpret_cast<uint64_t>(first_backing.data()));
    EXPECT_EQ(state.state_at_draw(1).context_control.bases[2],
              reinterpret_cast<uint64_t>(second_backing.data()));
    EXPECT_NE(state.draws[0].state, state.draws[1].state);
    EXPECT_TRUE(state.sh.empty());
    EXPECT_FALSE(state.dma_execution_rejected);
}

TEST_F(Pm4Context, UnselectedAndUnchangedControlsShareSnapshot) {
    Words buffer;
    control(buffer, 0x80010000, 0x80000000);
    packet(buffer, 0x2d, {3, 2});
    control(buffer, 0x10000, 0x10000); // Neither word selects an update.
    packet(buffer, 0x2d, {3, 2});
    control(buffer, 0x80010000, 0x80000000);   // Selected, but identical to the existing controls.
    packet(buffer, 0x2d, {3, 2});
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.draws.size(), 3u);
    EXPECT_EQ(state.draws[0].state, state.draws[1].state);
    EXPECT_EQ(state.draws[1].state, state.draws[2].state);
}

TEST_F(Pm4Context, OrdinaryRangesUseOffsetsInMemoryAndRegisterSpace) {
    std::array<uint32_t, 32> backing{};
    backing[3] = 0x1234;
    backing[4] = 0x5678;
    backing[9] = 0xabcd;
    Words buffer;
    control(buffer, 0x80000002, 0x80000000);
    const uint64_t address = reinterpret_cast<uint64_t>(backing.data());
    packet(buffer, 0x61, {uint32_t(address), uint32_t(address >> 32), 3, 2, 9, 1});
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.cx.count(3), 1u);
    EXPECT_EQ(state.cx.at(3), 0x1234u);
    EXPECT_EQ(state.cx.at(4), 0x5678u);
    EXPECT_EQ(state.cx.at(9), 0xabcdu);
    EXPECT_FALSE(state.dma_execution_rejected);
}

TEST_F(Pm4Context, UpdateBitsPreserveTheOtherControlWord) {
    std::array<uint32_t, 16> backing{};
    backing[2] = 10;
    Words buffer;
    control(buffer, 0x80000002, 0x80000000);
    load(buffer, 0x61, backing.data(), 0, 0);
    control(buffer, 0, 0x80000002);  // Enable shadowing without changing the load enables.
    packet(buffer, 0x69, {2, 20});
    load(buffer, 0x61, backing.data(), 2, 1);
    GpuState state;
    prosper_gpu_submit_scope_begin();
    run_command_buffer(buffer.data(), buffer.size(), state);
    const uint32_t visible = backing[2];
    const bool executed = execute_nonrender_submit_work(state, 1);
    prosper_gpu_submit_scope_end();
    EXPECT_TRUE(executed);
    EXPECT_EQ(visible, 10u) << "a parser read must not publish the private shadow write";
    ASSERT_EQ(state.cx.count(2), 1u);
    EXPECT_EQ(state.cx.at(2), 20u);
    EXPECT_FALSE(state.dma_execution_rejected);
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(backing[2], 20u);
}

TEST_F(Pm4Context, DisabledLoadsStillAllowDirectRegisterWrites) {
    std::array<uint32_t, 16> backing{};
    backing[2] = 99;
    Words buffer;
    control(buffer, 0x80000002, 0x80000000);
    load(buffer, 0x61, backing.data(), 2, 1);
    control(buffer, 0x80000000, 0);
    packet(buffer, 0x69, {2, 77});
    load(buffer, 0x61, backing.data(), 2, 1);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.cx.count(2), 1u);
    EXPECT_EQ(state.cx.at(2), 77u);
    EXPECT_FALSE(state.dma_execution_rejected);
}

TEST_F(Pm4Context, ContextUserConfigAndShaderLoadSelectorsAreIndependent) {
    std::array<uint32_t, 16> cx{}, uc{}, gfx{}, cs{};
    cx[2] = 11;
    uc[2] = 22;
    gfx[2] = 33;
    cs[3] = 44;
    Words buffer;
    control(buffer, 0x80018002, 0x80000000);  // Cx, Uc, Gfx SH; CS SH stays disabled.
    load(buffer, 0x61, cx.data(), 2, 1);
    load(buffer, 0x5e, uc.data(), 2, 1);
    load(buffer, 0x5f, gfx.data(), 2, 1);
    load(buffer, 0x5f, cs.data(), 3, 1, true);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.cx.count(2), 1u);
    EXPECT_EQ(state.cx.at(2), 11u);
    EXPECT_EQ(state.uc.at(2), 22u);
    EXPECT_EQ(state.sh.at(2), 33u);
    EXPECT_EQ(state.sh.count(3), 0u);
    buffer.clear();
    control(buffer, 0x81000000, 0);
    load(buffer, 0x5f, cs.data(), 3, 1, true);
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.sh.count(3), 1u);
    EXPECT_EQ(state.sh.at(3), 44u);
}

TEST_F(Pm4Context, ZeroCountInstallsShadowBaseAndConsecutiveSetShadowsAllWords) {
    std::array<uint32_t, 16> backing{};
    Words buffer;
    control(buffer, 0x80000002, 0x80000002);
    load(buffer, 0x61, backing.data(), 0, 0);
    packet(buffer, 0x69, {3, 0x1111, 0x2222, 0x3333});
    load(buffer, 0x61, backing.data(), 3, 3);
    GpuState state;
    prosper_gpu_submit_scope_begin();
    run_command_buffer(buffer.data(), buffer.size(), state);
    const auto visible = backing;
    const bool executed = execute_nonrender_submit_work(state, 2);
    prosper_gpu_submit_scope_end();
    EXPECT_TRUE(executed);
    EXPECT_EQ(visible[3], 0u);
    EXPECT_FALSE(state.dma_execution_rejected);
    ASSERT_EQ(state.cx.count(3), 1u);
    EXPECT_EQ(state.cx.at(3), 0x1111u);
    EXPECT_EQ(state.cx.at(4), 0x2222u);
    EXPECT_EQ(state.cx.at(5), 0x3333u);
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(backing[3], 0x1111u);
    EXPECT_EQ(backing[4], 0x2222u);
    EXPECT_EQ(backing[5], 0x3333u);
}

TEST_F(Pm4Context, DisabledShadowingDoesNotWriteTheEstablishedBase) {
    std::array<uint32_t, 16> backing{};
    backing[2] = 99;
    Words buffer;
    control(buffer, 0x80000002, 0x80000002);
    load(buffer, 0x61, backing.data(), 0, 0);
    control(buffer, 0, 0x80000000);
    packet(buffer, 0x69, {2, 77});
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(backing[2], 99u);
    EXPECT_EQ(state.cx.at(2), 77u);
}

TEST_F(Pm4Context, IndexedPairLoadIsIndependentOfOrdinaryLoadEnable) {
    const ShaderReg pair{3, 0x1234};
    const uint64_t address = reinterpret_cast<uint64_t>(&pair);
    Words buffer;
    control(buffer, 0x80000000, 0x80000000);
    packet(buffer, 0x9f, {uint32_t(address), uint32_t(address >> 32), 0x80000000, 1});
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.cx.count(3), 1u);
    EXPECT_EQ(state.cx.at(3), 0x1234u);
}

TEST_F(Pm4Context, UnretiredComputeProducerRefusesBeforeReadingRange) {
    std::array<uint32_t, 16> backing{};
    backing[2] = 99;
    Words buffer;
    control(buffer, 0x80000002, 0x80000000);
    packet(buffer, 0x15, {1, 1, 1, 1});
    load(buffer, 0x61, backing.data(), 2, 1);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_TRUE(state.dma_execution_rejected);
    EXPECT_EQ(state.cx.count(2), 0u);
}

TEST_F(Pm4Context, UnmappedEnabledRangeRefusesBeforeReadingMemory) {
    Words buffer;
    control(buffer, 0x80000002, 0x80000000);
    load(buffer, 0x61, nullptr, 2, 1);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_TRUE(state.dma_execution_rejected);
    EXPECT_EQ(state.cx.count(2), 0u);
}

TEST_F(Pm4Context, ReservedRangeFieldsAndWrongControlLengthStayUnknown) {
    for (auto payload : {Words{0, 0, 0x10000, 1}, Words{0, 0, 0, 0x4000}, Words{1, 0, 0, 1}}) {
        Words buffer{0xc0036100};
        buffer.insert(buffer.end(), payload.begin(), payload.end());
        std::vector<Pm4Command> decoded;
        EXPECT_EQ(decode_pm4(buffer.data(), buffer.size(), decoded), buffer.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded.front().kind, Pm4Command::Kind::Unknown);
    }
    Words buffer;
    packet(buffer, 0x28, {0x80000000});
    std::vector<Pm4Command> decoded;
    decode_pm4(buffer.data(), buffer.size(), decoded);
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded.front().kind, Pm4Command::Kind::Unknown);
}

TEST_F(Pm4Context, LoadAndShadowWriteUseExactlyTheMappedDword) {
#ifdef _WIN32
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const size_t page = info.dwPageSize;
    auto* memory = static_cast<uint8_t*>(
        VirtualAlloc(nullptr, page * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    ASSERT_NE(memory, nullptr);
    DWORD old_protection = 0;
    ASSERT_TRUE(VirtualProtect(memory + page, page, PAGE_NOACCESS, &old_protection));
#else
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto* memory = static_cast<uint8_t*>(
        mmap(nullptr, page * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(memory, MAP_FAILED);
    ASSERT_EQ(mprotect(memory + page, page, PROT_NONE), 0);
#endif
    auto* word = reinterpret_cast<uint32_t*>(memory + page - 4);
    *word = 10;
    Words buffer;
    control(buffer, 0x80000002, 0x80000002);
    load(buffer, 0x61, word, 0, 1);
    packet(buffer, 0x69, {0, 20});
    load(buffer, 0x61, word, 0, 1);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_FALSE(state.dma_execution_rejected);
    ASSERT_EQ(state.cx.count(0), 1u);
    EXPECT_EQ(state.cx.at(0), 20u);
    EXPECT_TRUE(execute_nonrender_submit_work(state, 3));
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(*word, 20u);
#ifdef _WIN32
    EXPECT_TRUE(VirtualFree(memory, 0, MEM_RELEASE));
#else
    EXPECT_EQ(munmap(memory, page * 2), 0);
#endif
}

TEST_F(Pm4Context, FailedComputeProducerDoesNotPublishFollowingShadowWrite) {
    std::array<uint32_t, 16> backing{};
    Words buffer;
    control(buffer, 0x80000002, 0x80000002);
    load(buffer, 0x61, backing.data(), 0, 0);
    packet(buffer, 0x15, {1, 1, 1, 1});
    packet(buffer, 0x69, {2, 99});
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.dispatches.size(), 1u);
    execute_nonrender_submit_work(state, 4);
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(backing[2], 0u);
}

TEST_F(Pm4Context, PendingCompletionAliasStaysPrivateWhileRangeUsesItsLatestShadowValue) {
    std::array<uint32_t, 16> backing{};
    backing[2] = 10;
    const uint64_t address = reinterpret_cast<uint64_t>(&backing[2]);
    Words buffer;
    control(buffer, 0x80000002, 0x80000002);
    load(buffer, 0x61, backing.data(), 0, 0);
    packet(buffer, 0x49, {0, 1u << 29, uint32_t(address), uint32_t(address >> 32), 20, 0, 0});
    packet(buffer, 0x69, {2, 30});
    load(buffer, 0x61, backing.data(), 2, 1);
    GpuState state;
    prosper_gpu_submit_scope_begin();
    run_command_buffer(buffer.data(), buffer.size(), state);
    const bool executed = execute_nonrender_submit_work(state, 5);
    const uint32_t visible = backing[2];
    prosper_gpu_submit_scope_end();
    EXPECT_TRUE(executed);
    EXPECT_EQ(visible, 10u);
    EXPECT_FALSE(state.dma_execution_rejected);
    EXPECT_EQ(state.cx.at(2), 30u);
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(backing[2], 30u);
}

TEST_F(Pm4Context, EnabledLoadWithoutEstablishedEnablesRefusesInsteadOfGuessing) {
    const uint32_t word = 99;
    Words buffer;
    load(buffer, 0x61, &word, 0, 1);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_TRUE(state.dma_execution_rejected);
    EXPECT_EQ(state.cx.count(0), 0u);
}
}  // namespace
}  // namespace prosper::gpu
