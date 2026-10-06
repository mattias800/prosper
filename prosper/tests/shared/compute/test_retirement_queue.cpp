// The ordering rules of RetirementQueue (ADR 0009, PERF-P8), asserted without Vulkan.
//
// What this proves, and what it does not. The tests exercise the QUEUE, a model of the contract;
// no production code uses it yet, so nothing here shows the backend obeys the rules. Two layers:
//   1. a hand-written case per rule;
//   2. a model check that runs random streams sequentially (every writeback applied immediately,
//      today's behaviour) and pipelined through the queue, where a dispatch READS guest memory when
//      the "GPU" executes it (at a random point between admission and retirement) and its writes
//      are applied at retirement, and requires every dispatch result, every CPU read and every
//      guest-visible effect to be identical. That model can fail on: a missed read-after-write at
//      admission, a CPU read of a pending writer, a CPU store over a pending reader, an effect that
//      does not retire the queue, and an unbounded queue. Each of those has its own negative test
//      that switches the rule off IN THE REAL QUEUE (RetirementPolicy) and requires the model check
//      to diverge, so the check cannot pass vacuously.
// It cannot see write-after-write between dispatches: in-order retirement already applies writes in
// order, so that clause is redundant in this model and rests on its hand-written case.
#include "shared/compute/retirement_queue.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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
using prosper::frontend::RetirementPolicy;
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

using Memory = std::array<uint64_t, kSlots>;

// What a dispatch computes: a value derived from every slot it read, so a stale input shows up.
uint64_t compute(const Memory& memory, const Step& s, uint64_t id) {
    uint64_t v = id * 1000003u;
    for (size_t r : s.reads) v = v * 31u + memory[r];
    return v;
}

// A guest-visible effect observes ALL of memory (a label, a flip, an EOP event).
uint64_t observe_all(const Memory& memory) {
    uint64_t v = 0xE;
    for (uint64_t word : memory) v = v * 1099511628211ull + word;
    return v;
}

struct Trace {
    std::vector<uint64_t> seen;   // indexed by step: a dispatch's value, a read, an effect's view
    Memory final_memory{};
    bool depth_exceeded = false;
};

Trace run_sequential(const std::vector<Step>& steps) {
    Trace t; t.seen.assign(steps.size(), 0);
    Memory mem{};
    uint64_t store = 7;
    for (size_t i = 0; i < steps.size(); ++i) {
        const Step& s = steps[i];
        switch (s.kind) {
            case Step::Dispatch: {
                const uint64_t v = compute(mem, s, i + 1);
                t.seen[i] = v;
                for (size_t w : s.writes) mem[w] = v;
                break;
            }
            case Step::CpuRead: t.seen[i] = mem[s.slot]; break;
            case Step::CpuWrite: mem[s.slot] = ++store; break;
            case Step::Effect: t.seen[i] = observe_all(mem); break;
        }
    }
    t.final_memory = mem;
    return t;
}

// The pipelined run: a dispatch executes ("the GPU reads guest memory") at a random point after
// admission, in submission order, and applies its writes at retirement.
Trace run_pipelined(const std::vector<Step>& steps, size_t depth, RetirementPolicy policy,
                    uint32_t schedule_seed) {
    Trace t; t.seen.assign(steps.size(), 0);
    Memory mem{};
    struct Pending { size_t step = 0; uint64_t value = 0; bool executed = false; };
    std::map<uint64_t, Pending> in_flight;
    std::vector<uint64_t> unexecuted;      // ids in submission order
    RetirementQueue q(depth, policy);
    SplitMix schedule(schedule_seed);
    const auto execute = [&](uint64_t id) {
        Pending& p = in_flight[id];
        if (p.executed) return;
        p.value = compute(mem, steps[p.step], id);
        p.executed = true;
        t.seen[p.step] = p.value;
        unexecuted.erase(std::find(unexecuted.begin(), unexecuted.end(), id));
    };
    const auto retire = [&](const std::vector<uint64_t>& ids) {
        // Execution is in order, so executing the requested prefix executes everything older too.
        for (uint64_t id : ids) execute(id);
        EXPECT_TRUE(q.retire(ids)) << "retirement out of order";
        for (uint64_t id : ids) {
            for (size_t w : steps[in_flight[id].step].writes) mem[w] = in_flight[id].value;
            in_flight.erase(id);
        }
    };
    uint64_t store = 7;
    for (size_t i = 0; i < steps.size(); ++i) {
        const Step& s = steps[i];
        const uint64_t id = i + 1;
        switch (s.kind) {
            case Step::Dispatch: {
                PendingOperation p; p.id = id;
                for (size_t r : s.reads) p.reads.push_back(slot_range(r));
                for (size_t w : s.writes) p.writes.push_back(slot_range(w));
                retire(q.required_before_admit(p));
                in_flight[id] = Pending{i, 0, false};
                unexecuted.push_back(id);
                q.admit(p);
                if (q.size() > depth) t.depth_exceeded = true;
                break;
            }
            case Step::CpuRead:
                retire(q.required_for_observation(slot_range(s.slot)));
                t.seen[i] = mem[s.slot];
                break;
            case Step::CpuWrite:
                retire(q.required_for_observation(slot_range(s.slot), true));
                mem[s.slot] = ++store;
                break;
            case Step::Effect:
                retire(q.required_for_effect());
                t.seen[i] = observe_all(mem);
                break;
        }
        // The GPU makes progress at its own pace between steps.
        for (unsigned k = schedule() % 3; k > 0 && !unexecuted.empty(); --k)
            execute(unexecuted.front());
    }
    // Submit end. Under a policy that never retires at an effect this flush still applies everything.
    for (uint64_t id : std::vector<uint64_t>(unexecuted)) execute(id);
    for (auto& [id, pending] : in_flight)
        for (size_t w : steps[pending.step].writes) mem[w] = pending.value;
    t.final_memory = mem;
    return t;
}

constexpr size_t kDepths[] = {1, 2, 4, 16};

// True when some stream, depth or schedule makes the pipelined run differ from the sequential one.
bool model_diverges(RetirementPolicy policy) {
    for (uint32_t seed = 1; seed <= 400; ++seed) {
        SplitMix rng(seed);
        const std::vector<Step> steps = random_stream(rng, 60);
        const Trace expected = run_sequential(steps);
        for (const size_t depth : kDepths) {
            const Trace actual = run_pipelined(steps, depth, policy, seed * 7919u + depth);
            if (actual.seen != expected.seen || actual.final_memory != expected.final_memory ||
                actual.depth_exceeded)
                return true;
        }
    }
    return false;
}

TEST(RetirementQueueModel, PipelinedRunIsIndistinguishableFromSequential) {
    EXPECT_FALSE(model_diverges(RetirementPolicy{}))
        << "the full contract must be indistinguishable from sequential execution";
}

// Each rule is necessary: turning it off in the REAL queue makes the same check fail.
TEST(RetirementQueueModel, DroppingConflictRetirementIsDetected) {
    RetirementPolicy p; p.retire_on_conflict = false;
    EXPECT_TRUE(model_diverges(p));
}
TEST(RetirementQueueModel, DroppingObservationRetirementIsDetected) {
    RetirementPolicy p; p.retire_on_observation = false;
    EXPECT_TRUE(model_diverges(p));
}
TEST(RetirementQueueModel, DroppingReaderWaitOnStoreIsDetected) {
    RetirementPolicy p; p.retire_readers_on_store = false;
    EXPECT_TRUE(model_diverges(p));
}
TEST(RetirementQueueModel, DroppingEffectRetirementIsDetected) {
    RetirementPolicy p; p.retire_on_effect = false;
    EXPECT_TRUE(model_diverges(p));
}
TEST(RetirementQueueModel, DroppingTheDepthBoundIsDetected) {
    RetirementPolicy p; p.bound_depth = false;
    EXPECT_TRUE(model_diverges(p));
}

}  // namespace
