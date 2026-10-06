// extract_render_state reuses its answer while the register files have not changed, and it knows
// that from their content ids rather than from their entries. So the thing to pin is the id: every
// way a register file's entries can change must change it, and nothing that leaves two files with
// different entries may leave them with the same one. Each arm below reads a field through the
// reusing extraction BEFORE the change, so the answer it must not see again is in the memo.
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/state/render_state.hpp"
#include <atomic>
#include <cstdlib>
#include <gtest/gtest.h>
#include <set>
#include <thread>
#include <utility>
#include <vector>

using namespace prosper::gpu;
namespace {
namespace P = prosper::agc::Pm4;

// CB_COLOR0_BASE holds address bits [39:8], so a written word W reads back as W << 8.
uint64_t color0(const GpuState& state) {
    return extract_render_state(state).color0_base;
}
uint64_t color0_computed(const GpuState& state) {
    return extract_render_state_uncached(state).color0_base;
}
GpuState with_color0(uint32_t word) {
    GpuState state;
    state.cx[P::CB_COLOR0_BASE] = word;
    return state;
}

// The two calls below compute `expected` times, or the process exits 1. Run in a child that has
// extracted nothing yet, so a switch armed here is read after it is set.
[[noreturn]] void exit_zero_if_two_calls_compute(const char* armed_switch, uint64_t expected) {
    if (armed_switch) {
#ifdef _WIN32
        _putenv_s(armed_switch, "1");
#else
        setenv(armed_switch, "1", 1);   // NOLINT(concurrency-mt-unsafe): a single-threaded child
#endif
    }
    const GpuState state = with_color0(0x77u);
    const uint64_t before = render_state_computations();
    (void)extract_render_state(state);
    (void)extract_render_state(state);
    std::_Exit(render_state_computations() - before == expected ? 0 : 1);
}
}   // namespace

// The switches that turn reuse off are read once, at the first extraction, so this process cannot
// try them: each runs in a re-executed child. A name misspelt in render_state.cpp would otherwise
// turn nothing red, and would make the comparison arm of a measurement the same as the other one.
TEST(RenderStateMemoDeathTest, EachSwitchThatNeedsEveryCallMakesEveryCallCompute) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    for (const char* name : {"PROSPER_NO_RENDER_STATE_MEMO", "PROSPER_MSAA_LOG",
                             "PROSPER_STENCILLOG", "PROSPER_SCISSORLOG"})
        EXPECT_EXIT(exit_zero_if_two_calls_compute(name, 2), testing::ExitedWithCode(0), "")
            << name;
    // The control for the arms above: with nothing armed the same child computes once.
    EXPECT_EXIT(exit_zero_if_two_calls_compute(nullptr, 1), testing::ExitedWithCode(0), "");
}

TEST(RenderStateMemo, AnUnchangedStateAnswersTheSameAndMatchesTheComputation) {
    const GpuState state = with_color0(0x1234u);
    EXPECT_EQ(color0(state), 0x123400u);
    EXPECT_EQ(color0(state), 0x123400u);
    EXPECT_EQ(color0(state), color0_computed(state));
}

TEST(RenderStateMemo, AWriteThroughTheIndexOperatorIsSeen) {
    GpuState state = with_color0(0x1234u);
    ASSERT_EQ(color0(state), 0x123400u);
    state.cx[P::CB_COLOR0_BASE] = 0x5678u;
    EXPECT_EQ(color0(state), 0x567800u);
}

TEST(RenderStateMemo, AWriteThroughMutableAtIsSeen) {
    GpuState state = with_color0(0x1234u);
    ASSERT_EQ(color0(state), 0x123400u);
    state.cx.at(P::CB_COLOR0_BASE) = 0x5678u;
    EXPECT_EQ(color0(state), 0x567800u);
}

TEST(RenderStateMemo, AnErasedRegisterIsSeen) {
    GpuState state = with_color0(0x1234u);
    ASSERT_EQ(color0(state), 0x123400u);
    ASSERT_EQ(state.cx.erase(P::CB_COLOR0_BASE), 1u);
    EXPECT_EQ(color0(state), 0u);
}

TEST(RenderStateMemo, AClearedFileIsSeen) {
    GpuState state = with_color0(0x1234u);
    ASSERT_EQ(color0(state), 0x123400u);
    state.cx.clear();
    EXPECT_EQ(color0(state), 0u);
}

TEST(RenderStateMemo, AWriteToEachOfTheThreeFilesIsSeen) {
    GpuState state = with_color0(0x1234u);
    ASSERT_EQ(extract_render_state(state).ps_addr, 0u);
    state.sh[P::SPI_SHADER_PGM_LO_PS] = 0x42u;
    EXPECT_EQ(extract_render_state(state).ps_addr, 0x4200u);
    // The user-config file is the third key. Which field it feeds is not this test's business:
    // whatever the computation says after the write is what the reusing call must say too.
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4u;
    const RenderState reused = extract_render_state(state);
    const RenderState computed = extract_render_state_uncached(state);
    EXPECT_EQ(reused.prim_type, computed.prim_type);
    EXPECT_EQ(reused.prim_type, 4u);
}

// A snapshot is a copy, and the folding state goes on changing after it is taken. Both must keep
// their own answer, in either order of asking.
TEST(RenderStateMemo, ACopyAndItsSourceDivergeIndependently) {
    GpuState folding = with_color0(0x1234u);
    const GpuState snapshot = folding;
    ASSERT_EQ(color0(snapshot), 0x123400u);
    folding.cx[P::CB_COLOR0_BASE] = 0x5678u;
    EXPECT_EQ(color0(folding), 0x567800u);
    EXPECT_EQ(color0(snapshot), 0x123400u);
    EXPECT_EQ(color0(folding), 0x567800u);
}

// Copy assignment replaces a file's entries wholesale; the id has to come with them.
TEST(RenderStateMemo, AssigningOverAStateReplacesItsAnswer) {
    GpuState state = with_color0(0x1234u);
    ASSERT_EQ(color0(state), 0x123400u);
    const GpuState other = with_color0(0x5678u);
    state.cx = other.cx;
    EXPECT_EQ(color0(state), 0x567800u);
    state = with_color0(0x9abcu);
    EXPECT_EQ(color0(state), 0x9abc00u);
}

// A moved-from file is empty. If it kept its id it would go on answering as what it used to hold.
TEST(RenderStateMemo, AMovedFromStateAnswersAsEmpty) {
    GpuState source = with_color0(0x1234u);
    ASSERT_EQ(color0(source), 0x123400u);
    GpuState taken = std::move(source);
    EXPECT_EQ(color0(taken), 0x123400u);
    // The moved-from states ARE the subject of the three reads below.
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(source.cx.empty());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(color0(source), 0u);
    GpuState assigned;
    assigned = std::move(taken);
    EXPECT_EQ(color0(assigned), 0x123400u);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(color0(taken), 0u);
}

// More distinct states than the memo holds, asked for round-robin: an entry that was displaced
// and one that never was must both come back right. (A plain check that displacement does not
// confuse entries; no mutation was found that only this arm catches.)
TEST(RenderStateMemo, MoreStatesThanTheMemoHoldsStayDistinct) {
    std::vector<GpuState> states;
    states.reserve(9);
    for (uint32_t i = 0; i < 9; ++i) states.push_back(with_color0(0x100u + i));
    for (int round = 0; round < 3; ++round)
        for (uint32_t i = 0; i < states.size(); ++i)
            EXPECT_EQ(color0(states[i]), uint64_t{0x100u + i} << 8) << "round " << round;
}

TEST(RegisterFileContentId, NamesContentsNotObjects) {
    RegisterFile empty, also_empty;
    EXPECT_EQ(empty.content_id(), 0u);
    EXPECT_EQ(also_empty.content_id(), 0u);

    RegisterFile file;
    file[0x10] = 1u;
    const uint64_t first = file.content_id();
    EXPECT_NE(first, 0u);
    const RegisterFile copy = file;
    EXPECT_EQ(copy.content_id(), first) << "a copy holds the same entries";

    // Reading leaves it alone: the snapshot of a draw is read through const access only.
    (void)copy.find(0x10);
    (void)copy.count(0x10);
    (void)copy.at(0x10);
    EXPECT_EQ(copy.content_id(), first);

    file[0x10] = 2u;
    EXPECT_NE(file.content_id(), first);
    EXPECT_EQ(copy.content_id(), first);

    // The same value written twice into two files makes equal contents with different ids. That
    // is allowed (it costs a recomputation); the reverse never is.
    RegisterFile twin;
    twin[0x10] = 2u;
    EXPECT_NE(twin.content_id(), file.content_id());

    file.clear();
    EXPECT_EQ(file.content_id(), 0u);
    EXPECT_EQ(file.erase(0x99), 0u);
    EXPECT_EQ(file.content_id(), 0u) << "erasing an absent register changed nothing";
}

// `set` is what the command fold writes with. A copy taken before it must keep the old value
// under the old id, and the file must not carry that id once the value is different.
TEST(RegisterFileContentId, SetWritesTheValueAndRenamesTheFile) {
    // Read through a const reference: the mutable `at()` hands out a writable reference and so
    // renames the file, which is the very thing being observed here.
    RegisterFile file;
    const RegisterFile& read = file;
    file.set(0x10, 1u);
    const uint64_t first = file.content_id();
    EXPECT_NE(first, 0u);
    EXPECT_EQ(read.at(0x10), 1u);
    const RegisterFile before = file;
    file.set(0x10, 2u);
    file.set(0x08, 3u);   // inserted in front of an existing entry
    EXPECT_EQ(read.at(0x10), 2u);
    EXPECT_EQ(read.at(0x08), 3u);
    EXPECT_EQ(file.size(), 2u);
    EXPECT_NE(file.content_id(), first);
    EXPECT_EQ(before.content_id(), first);
    EXPECT_EQ(before.at(0x10), 1u);

    GpuState state;
    state.cx.set(P::CB_COLOR0_BASE, 0x1234u);
    ASSERT_EQ(color0(state), 0x123400u);
    state.cx.set(P::CB_COLOR0_BASE, 0x5678u);
    EXPECT_EQ(color0(state), 0x567800u);
}

// Move assignment empties its source. Onto itself, that must not empty the file.
TEST(RegisterFileContentId, MovingAFileOntoItselfKeepsItWhole) {
    RegisterFile file;
    file.set(0x10, 7u);
    const uint64_t id = file.content_id();
    RegisterFile& same = file;
    file = std::move(same);
    EXPECT_EQ(file.count(0x10), 1u);
    EXPECT_EQ(file.content_id(), id);
}

// Ids are drawn by every thread that folds a command stream, and a snapshot made on one thread is
// read on another. Two files must never be handed the same one.
TEST(RegisterFileContentId, IsUniqueAcrossThreads) {
    constexpr int kThreads = 4, kWrites = 2000;
    std::vector<std::vector<uint64_t>> seen(kThreads);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    // All four start drawing ids together; threads that ran one after another would not show a
    // counter that is shared but not atomic.
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back([&seen, &ready, &go, t] {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            RegisterFile file;
            for (int i = 0; i < kWrites; ++i) {
                file[0x10] = static_cast<uint32_t>(i);
                seen[t].push_back(file.content_id());
            }
        });
    while (ready.load() != kThreads) std::this_thread::yield();
    go.store(true);
    for (auto& thread : threads) thread.join();
    std::set<uint64_t> all;
    for (const auto& ids : seen) all.insert(ids.begin(), ids.end());
    EXPECT_EQ(all.size(), size_t{kThreads} * kWrites);
    EXPECT_FALSE(all.contains(0u));
}

// The other half: reuse has to HAPPEN, or every arm above passes on an extraction that computes
// each time and the cost this exists to remove is back with nothing red. A draw holds a snapshot,
// which is a copy of the folding state, so the copy must share its source's answer.
TEST(RenderStateMemo, AnUnchangedStateIsComputedOnce) {
    const GpuState state = with_color0(0x4321u);
    const uint64_t before = render_state_computations();
    (void)extract_render_state(state);
    (void)extract_render_state(state);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): the copy is the subject
    const GpuState snapshot = state;
    (void)extract_render_state(snapshot);
    EXPECT_EQ(render_state_computations() - before, 1u);
    (void)extract_render_state_uncached(state);
    EXPECT_EQ(render_state_computations() - before, 2u) << "the computing call always computes";
}

// A draw's questions alternate with questions about the draw whose output it reads, so one
// remembered answer would be displaced on every call.
TEST(RenderStateMemo, TwoAlternatingStatesAreEachComputedOnce) {
    const GpuState reader = with_color0(0x2001u), producer = with_color0(0x2002u);
    const uint64_t before = render_state_computations();
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(color0(reader), 0x200100u);
        EXPECT_EQ(color0(producer), 0x200200u);
    }
    EXPECT_EQ(render_state_computations() - before, 2u);
}
