// The ordering contract for pipelined GPU work (ADR 0009, PERF-P8), asserted without Vulkan.
//
// Two layers: hand-written cases for each rule in retirement_queue.hpp, and a model check that runs
// random operation streams both sequentially (every writeback applied immediately -- today's
// behaviour) and pipelined through the queue, and requires every input an operation reads and every
// CPU observation to see byte-identical guest memory. That second layer is the guarantee: it does
// not assert that the queue "looks right", it asserts that the pipelined run is indistinguishable
// from the sequential one for the guest, whatever the stream.
#include "shared/compute/retirement_queue.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <map>
#include <vector>

namespace {

// Deterministic test-data generator (splitmix64). Not for any security purpose.
struct SplitMix {
    uint64_t state;
    explicit SplitMix(uint64_t seed) : state(seed) {}
    uint32_t operator()() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return static_cast<uint32_t>((z ^ (z >> 31)) >> 16);
    }
};

using prosper::frontend::GuestRange;
using prosper::frontend::PendingOperation;
using prosper::frontend::RetirementQueue;

PendingOperation op(uint64_t id, std::vector<GuestRange> reads, std::vector<GuestRange> writes) {
    return {id, std::move(reads), std::move(writes)};
}

TEST(RetirementQueue, IndependentOperationsStayPending) {
    RetirementQueue q(8);
    q.admit(op(1, {{0, 16}}, {{100, 16}}));
    EXPECT_TRUE(q.required_before_admit(op(2, {{200, 16}}, {{300, 16}})).empty());
}

TEST(RetirementQueue, ReadAfterWriteRetiresTheWriter) {
    RetirementQueue q(8);
    q.admit(op(1, {}, {{100, 16}}));
    EXPECT_EQ(q.required_before_admit(op(2, {{108, 4}}, {})), (std::vector<uint64_t>{1}));
}

TEST(RetirementQueue, WriteAfterWriteAndWriteAfterReadConflict) {
    RetirementQueue q(8);
    q.admit(op(1, {}, {{100, 16}}));
    EXPECT_EQ(q.required_before_admit(op(2, {}, {{110, 16}})), (std::vector<uint64_t>{1}));
    RetirementQueue r(8);
    r.admit(op(1, {{100, 16}}, {}));
    EXPECT_EQ(r.required_before_admit(op(2, {}, {{100, 4}})), (std::vector<uint64_t>{1}));
}

TEST(RetirementQueue, ReadAfterReadDoesNotConflict) {
    RetirementQueue q(8);
    q.admit(op(1, {{100, 16}}, {}));
    EXPECT_TRUE(q.required_before_admit(op(2, {{100, 16}}, {})).empty());
}

TEST(RetirementQueue, ConflictRetiresEveryEarlierOperationNotJustTheConflictingOne) {
    RetirementQueue q(8);
    q.admit(op(1, {}, {{0, 8}}));
    q.admit(op(2, {}, {{100, 8}}));
    q.admit(op(3, {}, {{200, 8}}));
    // Conflicts only with 2, but 1 precedes it and must retire first; 3 may stay pending.
    EXPECT_EQ(q.required_before_admit(op(4, {{100, 8}}, {})), (std::vector<uint64_t>({1, 2})));
}

TEST(RetirementQueue, GuestVisibleEffectRetiresEverything) {
    RetirementQueue q(8);
    q.admit(op(1, {}, {{0, 8}}));
    q.admit(op(2, {}, {{100, 8}}));
    EXPECT_EQ(q.required_for_effect(), (std::vector<uint64_t>({1, 2})));
}

TEST(RetirementQueue, ObservationRetiresUpToTheLastWriterOfTheRange) {
    RetirementQueue q(8);
    q.admit(op(1, {}, {{0, 8}}));
    q.admit(op(2, {}, {{100, 8}}));
    q.admit(op(3, {}, {{200, 8}}));
    EXPECT_EQ(q.required_for_observation({100, 4}), (std::vector<uint64_t>({1, 2})));
    EXPECT_TRUE(q.required_for_observation({500, 4}).empty());
    // A CPU store also has to wait for pending READERS of what it overwrites.
    RetirementQueue r(8);
    r.admit(op(1, {{100, 8}}, {}));
    EXPECT_TRUE(r.required_for_observation({100, 4}, false).empty());
    EXPECT_EQ(r.required_for_observation({100, 4}, true), (std::vector<uint64_t>{1}));
}

TEST(RetirementQueue, DepthIsBounded) {
    RetirementQueue q(2);
    q.admit(op(1, {}, {{0, 8}}));
    q.admit(op(2, {}, {{100, 8}}));
    EXPECT_EQ(q.required_before_admit(op(3, {}, {{200, 8}})), (std::vector<uint64_t>{1}));
}

TEST(RetirementQueue, RetirementIsStrictlyInOrder) {
    RetirementQueue q(8);
    q.admit(op(1, {}, {{0, 8}}));
    q.admit(op(2, {}, {{100, 8}}));
    EXPECT_FALSE(q.retire({2}));          // the newest cannot leave first
    EXPECT_TRUE(q.retire({1}));
    EXPECT_FALSE(q.retire({1}));          // already gone
    EXPECT_TRUE(q.retire({2}));
    EXPECT_TRUE(q.empty());
}

// ---- model check: pipelined == sequential, for every stream ---------------------------------

constexpr size_t kSlots = 6;   // each slot is one 16-byte guest range
GuestRange slot_range(size_t i) { return {i * 16u, 16u}; }

struct Step {
    enum Kind { Dispatch, CpuRead, CpuWrite, Effect } kind = Dispatch;
    std::vector<size_t> reads, writes;   // slots
    size_t slot = 0;                     // CpuRead / CpuWrite
};

std::vector<Step> random_stream(SplitMix& rng, size_t length) {
    std::vector<Step> steps;
    for (size_t i = 0; i < length; ++i) {
        Step s;
        const unsigned pick = rng() % 10;
        if (pick < 6) {
            s.kind = Step::Dispatch;
            for (size_t k = rng() % 3; k > 0; --k) s.reads.push_back(rng() % kSlots);
            for (size_t k = 1 + rng() % 2; k > 0; --k) s.writes.push_back(rng() % kSlots);
        } else if (pick < 8) {
            s.kind = Step::CpuRead; s.slot = rng() % kSlots;
        } else if (pick < 9) {
            s.kind = Step::CpuWrite; s.slot = rng() % kSlots;
        } else {
            s.kind = Step::Effect;
        }
        steps.push_back(s);
    }
    return steps;
}

// What a dispatch computes: a value derived from every slot it read, so a stale input shows up.
uint64_t compute(const std::array<uint64_t, kSlots>& memory, const Step& s, uint64_t id) {
    uint64_t v = id * 1000003u;
    for (size_t r : s.reads) v = v * 31u + memory[r];
    return v;
}

struct Trace { std::vector<uint64_t> seen; std::array<uint64_t, kSlots> final_memory{}; };

Trace run_sequential(const std::vector<Step>& steps) {
    Trace t; std::array<uint64_t, kSlots> mem{};
    uint64_t id = 0, store = 7;
    for (const Step& s : steps) {
        ++id;
        switch (s.kind) {
            case Step::Dispatch: {
                const uint64_t v = compute(mem, s, id);
                t.seen.push_back(v);
                for (size_t w : s.writes) mem[w] = v;
                break;
            }
            case Step::CpuRead: t.seen.push_back(mem[s.slot]); break;
            case Step::CpuWrite: mem[s.slot] = ++store; break;
            case Step::Effect: t.seen.push_back(0xE); break;
        }
    }
    t.final_memory = mem;
    return t;
}

Trace run_pipelined(const std::vector<Step>& steps, size_t depth) {
    Trace t; std::array<uint64_t, kSlots> mem{};
    struct Pending { uint64_t id; std::vector<size_t> writes; uint64_t value; };
    std::map<uint64_t, Pending> in_flight;
    RetirementQueue q(depth);
    const auto retire = [&](const std::vector<uint64_t>& ids) {
        ASSERT_TRUE(q.retire(ids)) << "retirement out of order";
        for (uint64_t id : ids) {
            for (size_t w : in_flight[id].writes) mem[w] = in_flight[id].value;
            in_flight.erase(id);
        }
    };
    uint64_t id = 0, store = 7;
    for (const Step& s : steps) {
        ++id;
        switch (s.kind) {
            case Step::Dispatch: {
                PendingOperation p; p.id = id;
                for (size_t r : s.reads) p.reads.push_back(slot_range(r));
                for (size_t w : s.writes) p.writes.push_back(slot_range(w));
                retire(q.required_before_admit(p));
                // The GPU reads guest memory as the host sees it NOW (the upload), so every
                // pending writer that matters has retired by this point.
                const uint64_t v = compute(mem, s, id);
                t.seen.push_back(v);
                in_flight[id] = {id, s.writes, v};
                q.admit(p);
                break;
            }
            case Step::CpuRead:
                retire(q.required_for_observation(slot_range(s.slot)));
                t.seen.push_back(mem[s.slot]);
                break;
            case Step::CpuWrite:
                retire(q.required_for_observation(slot_range(s.slot), true));
                mem[s.slot] = ++store;
                break;
            case Step::Effect:
                retire(q.required_for_effect());
                t.seen.push_back(0xE);
                break;
        }
    }
    retire(q.required_for_effect());   // submit end
    t.final_memory = mem;
    return t;
}

TEST(RetirementQueueModel, PipelinedRunIsIndistinguishableFromSequential) {
    for (uint32_t seed = 1; seed <= 400; ++seed) {
        SplitMix rng(seed);
        const std::vector<Step> steps = random_stream(rng, 60);
        const Trace expected = run_sequential(steps);
        for (const size_t depth : {size_t{1}, size_t{2}, size_t{4}, size_t{16}}) {
            const Trace actual = run_pipelined(steps, depth);
            ASSERT_EQ(actual.seen, expected.seen) << "seed " << seed << " depth " << depth;
            ASSERT_EQ(actual.final_memory, expected.final_memory)
                << "seed " << seed << " depth " << depth;
        }
    }
}

// The control: a deliberately broken policy (no retirement before admit, as Stage 2 behaved for a
// label read) must be CAUGHT by the same model, otherwise the check above proves nothing.
TEST(RetirementQueueModel, ARunThatSkipsConflictRetirementIsDetected) {
    bool diverged = false;
    for (uint32_t seed = 1; seed <= 50 && !diverged; ++seed) {
        SplitMix rng(seed);
        const std::vector<Step> steps = random_stream(rng, 60);
        const Trace expected = run_sequential(steps);
        // Pipelined without retiring on conflict: writes only land at effects/submit end.
        Trace t; std::array<uint64_t, kSlots> mem{};
        std::vector<std::pair<std::vector<size_t>, uint64_t>> pending;
        uint64_t id = 0, store = 7;
        const auto flush = [&] { for (auto& [w, v] : pending) for (size_t s : w) mem[s] = v; pending.clear(); };
        for (const Step& s : steps) {
            ++id;
            if (s.kind == Step::Dispatch) { const uint64_t v = compute(mem, s, id); t.seen.push_back(v); pending.push_back({s.writes, v}); }
            else if (s.kind == Step::CpuRead) t.seen.push_back(mem[s.slot]);
            else if (s.kind == Step::CpuWrite) mem[s.slot] = ++store;
            else { flush(); t.seen.push_back(0xE); }
        }
        flush();
        diverged = t.seen != expected.seen;
    }
    EXPECT_TRUE(diverged) << "the model must detect a policy that skips conflict retirement";
}

}  // namespace
