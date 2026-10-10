// #4840: COND_EXEC skips raw dwords before decoding, and ATOMIC_MEM performs an ordered native
// integer RMW. Ignoring either changes which commands run or publishes stale memory to consumers.
// Hand-built packets pin the published GFX10 ABI independently of Prosper's packet builders.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_memory.hpp"

#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <thread>
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
using Kind = Pm4Command::Kind;

uint32_t header(uint32_t op, size_t dwords) {
    return 0xc0000000u | (static_cast<uint32_t>(dwords - 2) << 16) | (op << 8);
}

void append(Words& buffer, uint32_t op, std::initializer_list<uint32_t> payload) {
    buffer.push_back(header(op, payload.size() + 1));
    buffer.insert(buffer.end(), payload);
}

void conditional(Words& buffer, uint64_t address, uint32_t dwords) {
    append(buffer, IT_COND_EXEC,
           {static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32), 0, dwords});
}

void atomic(Words& buffer, uint32_t op, uint64_t address, uint64_t source, uint64_t compare = 0) {
    append(buffer, IT_ATOMIC_MEM,
           {op, static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32),
            static_cast<uint32_t>(source), static_cast<uint32_t>(source >> 32),
            static_cast<uint32_t>(compare), static_cast<uint32_t>(compare >> 32), 0});
}

void write(Words& buffer, uint64_t address, uint32_t value) {
    append(buffer, IT_WRITE_DATA,
           {5u << 8, static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32), value});
}

void release(Words& buffer, uint64_t address, uint32_t value) {
    append(buffer, IT_RELEASE_MEM,
           {0, 1u << 29, static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32),
            value, 0, 0});
}

void reg(Words& buffer, uint32_t value) {
    append(buffer, IT_SET_CONTEXT_REG, {0x40, value});
}

class Pm4Memory : public ::testing::Test {
protected:
    void SetUp() override { prosper_gpu_drain_completion_writes(); }
    void TearDown() override { prosper_gpu_drain_completion_writes(); }
};

TEST_F(Pm4Memory, FalseConditionalSkipsUnparseableRawSpan) {
    uint32_t predicate = 0;
    Words buffer;
    conditional(buffer, reinterpret_cast<uint64_t>(&predicate), 4);
    buffer.insert(buffer.end(), {0, 0xcfffffff, 0xdeadbeef, 0x80000000});
    reg(buffer, 0x1234);
    GpuState state;
    size_t consumed = 0;
    EXPECT_EQ(run_command_buffer(buffer.data(), buffer.size(), state, &consumed), 2u);
    EXPECT_EQ(consumed, buffer.size());
    EXPECT_FALSE(state.dma_execution_rejected);
    EXPECT_EQ(state.cx[0x40], 0x1234u);
}

TEST_F(Pm4Memory, NonzeroConditionalExecutesAndZeroSkips) {
    for (uint32_t predicate : {0u, 1u, 0x80000000u}) {
        Words buffer;
        conditional(buffer, reinterpret_cast<uint64_t>(&predicate), 3);
        reg(buffer, 0xabc);
        GpuState state;
        run_command_buffer(buffer.data(), buffer.size(), state);
        EXPECT_EQ(state.cx.count(0x40), predicate ? 1u : 0u);
    }
}

TEST_F(Pm4Memory, ConditionalReadsOnlyLowDword) {
    uint64_t predicate = 0xffffffff00000000ull;
    Words buffer;
    conditional(buffer, reinterpret_cast<uint64_t>(&predicate), 3);
    reg(buffer, 1);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_EQ(state.cx.count(0x40), 0u);
}

TEST_F(Pm4Memory, ConditionalDoesNotReadPastMappingEdge) {
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
    auto* predicate = reinterpret_cast<uint32_t*>(memory + page - 4);
    *predicate = 1;
    Words buffer;
    conditional(buffer, reinterpret_cast<uint64_t>(predicate), 3);
    reg(buffer, 0x5678);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_EQ(state.cx[0x40], 0x5678u);
#ifdef _WIN32
    EXPECT_TRUE(VirtualFree(memory, 0, MEM_RELEASE));
#else
    EXPECT_EQ(munmap(memory, page * 2), 0);
#endif
}

TEST_F(Pm4Memory, NestedConditionAndTypeTwoFillerPreserveSpan) {
    uint32_t zero = 0, one = 1;
    Words buffer{0x80000000};
    conditional(buffer, reinterpret_cast<uint64_t>(&one), 8);
    conditional(buffer, reinterpret_cast<uint64_t>(&zero), 3);
    reg(buffer, 2);
    reg(buffer, 3);
    GpuState state;
    EXPECT_EQ(run_command_buffer(buffer.data(), buffer.size(), state), 3u);
    EXPECT_EQ(state.cx[0x40], 3u);
}

TEST_F(Pm4Memory, ConditionalRejectsUnmappedPredicateAndOversizedSpan) {
    uint32_t zero = 0;
    for (uint64_t address : {uint64_t{0x1000}, reinterpret_cast<uint64_t>(&zero)}) {
        Words buffer;
        conditional(buffer, address, 4);
        reg(buffer, 1);
        GpuState state;
        run_command_buffer(buffer.data(), buffer.size(), state);
        EXPECT_TRUE(state.dma_execution_rejected);
        EXPECT_EQ(state.cx.count(0x40), 0u);
    }
}

TEST_F(Pm4Memory, ConditionalUsesPendingWriteWithoutPublishingIt) {
    uint32_t predicate = 0;
    Words buffer;
    write(buffer, reinterpret_cast<uint64_t>(&predicate), 1);
    conditional(buffer, reinterpret_cast<uint64_t>(&predicate), 3);
    reg(buffer, 0x55);
    GpuState state;
    prosper_gpu_submit_scope_begin();
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_EQ(predicate, 0u);
    EXPECT_FALSE(state.dma_execution_rejected);
    EXPECT_EQ(state.cx[0x40], 0x55u);
    prosper_gpu_submit_scope_end();
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(predicate, 1u);
}

TEST_F(Pm4Memory, ConditionalRefusesUnretiredComputeProducer) {
    uint32_t predicate = 1;
    Words buffer;
    append(buffer, IT_DISPATCH_DIRECT, {1, 1, 1, 1});
    conditional(buffer, reinterpret_cast<uint64_t>(&predicate), 3);
    reg(buffer, 0x55);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_TRUE(state.dma_execution_rejected);
    EXPECT_EQ(state.cx.count(0x40), 0u);
}

TEST_F(Pm4Memory, InvalidPacketFieldsAndTruncationRemainUnsupported) {
    uint32_t predicate = 1;
    Words valid;
    conditional(valid, reinterpret_cast<uint64_t>(&predicate), 0);
    for (size_t index : {size_t{1}, size_t{3}, size_t{4}}) {
        Words bad = valid;
        bad[index] |= index == 4 ? 1u << 14 : 1;
        std::vector<Pm4Command> commands;
        EXPECT_EQ(decode_pm4(bad.data(), bad.size(), commands), bad.size());
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, Kind::Unknown);
    }
    std::vector<Pm4Command> commands;
    EXPECT_EQ(decode_pm4(valid.data(), valid.size() - 1, commands), 0u);
    EXPECT_TRUE(commands.empty());
    for (uint32_t op : {0u, 1u, 0x100u | 15u, 0x80000000u | 15u}) {
        Words bad;
        atomic(bad, op, reinterpret_cast<uint64_t>(&predicate), 1);
        commands.clear();
        decode_pm4(bad.data(), bad.size(), commands);
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, Kind::Unknown);
    }
}

TEST_F(Pm4Memory, AtomicAndFollowingWritesExecuteInStreamOrderWithoutRenderer) {
    alignas(8) uint32_t word = 3;
    Words buffer;
    atomic(buffer, 15, reinterpret_cast<uint64_t>(&word), 5);
    write(buffer, reinterpret_cast<uint64_t>(&word), 21);
    atomic(buffer, 0x4f, reinterpret_cast<uint64_t>(&word), 2);
    GpuState state;
    EXPECT_EQ(run_command_buffer(buffer.data(), buffer.size(), state), 3u);
    EXPECT_EQ(word, 3u);
    ASSERT_EQ(state.ordered_memory_effects.size(), 3u);
    EXPECT_TRUE(execute_nonrender_submit_work(state, 1));
    EXPECT_EQ(word, 23u);
}

TEST_F(Pm4Memory, AtomicRefusesFailedProducerEpoch) {
    uint32_t word = 0;
    Words buffer;
    // No code/resource binds: the dispatch cannot produce its promised write. A later RMW must
    // not replace that failed GPU input with whatever happened to be in the mapped CPU bytes.
    append(buffer, IT_DISPATCH_DIRECT, {1, 1, 1, 1});
    atomic(buffer, 15, reinterpret_cast<uint64_t>(&word), 1);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    ASSERT_EQ(state.dispatches.size(), 1u);
    execute_nonrender_submit_work(state, 9);
    EXPECT_EQ(word, 0u);
}

TEST_F(Pm4Memory, PresentEntryExecutesAtomicOnlySubmit) {
    uint32_t word = 1;
    Words buffer;
    atomic(buffer, 15, reinterpret_cast<uint64_t>(&word), 7);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    execute_ordered_and_present(state, 0, 0, 2, false);
    EXPECT_EQ(word, 8u);
}

TEST_F(Pm4Memory, AtomicAndCompletionAliasKeepPrivateFifoOrder) {
    uint32_t resource = 3, label = 0;
    Words buffer;
    atomic(buffer, 15, reinterpret_cast<uint64_t>(&resource), 5);
    release(buffer, reinterpret_cast<uint64_t>(&label), 1);
    write(buffer, reinterpret_cast<uint64_t>(&label), 2);
    GpuState state;
    prosper_gpu_submit_scope_begin();
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_TRUE(execute_nonrender_submit_work(state, 7));
    EXPECT_EQ(resource, 8u);
    EXPECT_EQ(label, 0u);
    prosper_gpu_submit_scope_end();
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(label, 2u);
}

TEST_F(Pm4Memory, AtomicAliasingEarlierQueuedCompletionWaitsForItsValue) {
    uint32_t label = 0;
    Words buffer;
    release(buffer, reinterpret_cast<uint64_t>(&label), 7);
    atomic(buffer, 15, reinterpret_cast<uint64_t>(&label), 2);
    GpuState state;
    prosper_gpu_submit_scope_begin();
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_TRUE(execute_nonrender_submit_work(state, 8));
    EXPECT_EQ(label, 0u);
    prosper_gpu_submit_scope_end();
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(label, 9u);
}

TEST_F(Pm4Memory, AtomicWrapAndCompareExchangeUseRequestedWidth) {
    alignas(8) uint64_t word = 0xaabbccddffffffffull;
    Words buffer;
    atomic(buffer, 15, reinterpret_cast<uint64_t>(&word), 1);
    atomic(buffer, 8, reinterpret_cast<uint64_t>(&word), 7, 0);
    atomic(buffer, 8, reinterpret_cast<uint64_t>(&word), 9, 0);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_TRUE(execute_nonrender_submit_work(state, 3));
    EXPECT_EQ(word, 0xaabbccdd00000007ull);
    buffer.clear();
    atomic(buffer, 0x2f, reinterpret_cast<uint64_t>(&word), 0x55443322fffffff9ull);
    state = GpuState{};
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_TRUE(execute_nonrender_submit_work(state, 4));
    EXPECT_EQ(word, 0u);
}

TEST_F(Pm4Memory, AtomicIntegerOperationsMatchArchitecturalResults) {
    struct Case {
        uint32_t op;
        uint64_t old_value, source, compare, result;
    };
    const Case cases[] = {
        {7, 3, 9, 0, 9},
        {16, 0, 1, 0, UINT32_MAX},
        {17, 0xfffffffe, 1, 0, 0xfffffffe},
        {18, 0xfffffffe, 1, 0, 1},
        {19, 0xfffffffe, 1, 0, 1},
        {20, 0xfffffffe, 1, 0, 0xfffffffe},
        {21, 0xaa, 0x0f, 0, 0x0a},
        {22, 0xa0, 0x0f, 0, 0xaf},
        {23, 0xaf, 0x0f, 0, 0xa0},
        {24, 4, 5, 0, 5},
        {24, 5, 5, 0, 0},
        {25, 0, 5, 0, 5},
        {25, 6, 5, 0, 5},
        {25, 4, 5, 0, 3},
        {0x31, UINT64_MAX - 1, 1, 0, UINT64_MAX - 1},
        {0x38, UINT64_MAX, UINT64_MAX, 0, 0},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.op);
        alignas(8) uint64_t word = c.old_value;
        Words buffer;
        atomic(buffer, c.op, reinterpret_cast<uint64_t>(&word), c.source, c.compare);
        GpuState state;
        run_command_buffer(buffer.data(), buffer.size(), state);
        EXPECT_TRUE(execute_nonrender_submit_work(state, 5));
        EXPECT_EQ(word, c.result);
    }
}

TEST_F(Pm4Memory, ConditionalOverlaysEarlierIntegerAtomicWithoutEarlyRmw) {
    uint32_t word = 0;
    Words buffer;
    atomic(buffer, 15, reinterpret_cast<uint64_t>(&word), 1);
    conditional(buffer, reinterpret_cast<uint64_t>(&word), 3);
    reg(buffer, 42);
    GpuState state;
    run_command_buffer(buffer.data(), buffer.size(), state);
    EXPECT_EQ(word, 0u);
    EXPECT_EQ(state.cx[0x40], 42u);
    EXPECT_TRUE(execute_nonrender_submit_work(state, 6));
    EXPECT_EQ(word, 1u);
}

TEST_F(Pm4Memory, AtomicRejectsMisalignedAndUnmappedOperands) {
    alignas(8) uint64_t word = 9;
    Pm4Command command;
    command.kind = Kind::AtomicMem;
    command.atomic_op = 15;
    command.atomic_source = 1;
    for (uint64_t address :
         {uint64_t{0}, uint64_t{0x1000}, reinterpret_cast<uint64_t>(&word) + 1}) {
        command.atomic_addr = address;
        EXPECT_FALSE(execute_atomic_mem(command));
        EXPECT_EQ(word, 9u);
    }
}

TEST_F(Pm4Memory, AtomicRmwDoesNotLoseConcurrentCpuUpdates) {
    alignas(8) uint64_t word = 0;
    Pm4Command command;
    command.kind = Kind::AtomicMem;
    command.atomic_op = 0x2f;
    command.atomic_addr = reinterpret_cast<uint64_t>(&word);
    command.atomic_source = 1;
    std::thread cpu([&] {
        for (unsigned i = 0; i < 1000; ++i)
            std::atomic_ref<uint64_t>(word).fetch_add(1, std::memory_order_seq_cst);
    });
    for (unsigned i = 0; i < 1000; ++i) EXPECT_TRUE(execute_atomic_mem(command));
    cpu.join();
    EXPECT_EQ(word, 2000u);
}

TEST_F(Pm4Memory, AtomicRejectsReadOnlyMemory) {
#ifdef _WIN32
    auto* word = static_cast<uint32_t*>(
        VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    ASSERT_NE(word, nullptr);
    *word = 7;
    DWORD old_protection = 0;
    ASSERT_TRUE(VirtualProtect(word, 4096, PAGE_READONLY, &old_protection));
#else
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto* word = static_cast<uint32_t*>(
        mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(word, MAP_FAILED);
    *word = 7;
    ASSERT_EQ(mprotect(word, page, PROT_READ), 0);
#endif
    Pm4Command command;
    command.kind = Kind::AtomicMem;
    command.atomic_op = 15;
    command.atomic_addr = reinterpret_cast<uint64_t>(word);
    command.atomic_source = 1;
    EXPECT_FALSE(execute_atomic_mem(command));
    EXPECT_EQ(*word, 7u);
#ifdef _WIN32
    EXPECT_TRUE(VirtualFree(word, 0, MEM_RELEASE));
#else
    EXPECT_EQ(munmap(word, page), 0);
#endif
}

} // namespace
} // namespace prosper::gpu
