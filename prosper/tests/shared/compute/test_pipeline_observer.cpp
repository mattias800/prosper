// The pipelining observer (ADR 0009 Stage 1, step 2) counts what the RetirementQueue contract WOULD
// have allowed; it changes nothing in the backend. These tests pin the counting: what is "free",
// what is attributed to a conflict versus the depth bound, what a graphics span or submit boundary
// does in each model, and that the wait-weighted fraction uses the dispatch's own measured wait.
#include "shared/compute/pipeline_observer.hpp"

#include <gtest/gtest.h>

namespace {

using prosper::frontend::BindingRange;
using prosper::frontend::DispatchTimes;
using prosper::frontend::ObservedOperation;
using prosper::frontend::PipelineModel;
using prosper::frontend::PipelineObserver;

// Only the submit-to-fence time matters for the overlap counts; the timeline tests set all three.
DispatchTimes W(double gpu_ms) { return {0.0, gpu_ms, 0.0}; }

BindingRange rd(uint64_t addr, uint64_t bytes = 16) { return {addr, bytes, true, false}; }
BindingRange wr(uint64_t addr, uint64_t bytes = 16) { return {addr, bytes, false, true}; }

TEST(PipelineObserver, IndependentDispatchesAreFreeAndDepthEventuallyForcesRetirement) {
    PipelineModel m("t", 2, false);
    m.dispatch(1, {rd(0), wr(100)}, W(1.0));
    m.dispatch(1, {rd(0), wr(200)}, W(1.0));
    EXPECT_EQ(m.stats().free_admits, 2u);
    m.dispatch(1, {rd(0), wr(300)}, W(1.0));   // queue holds 2: the oldest must retire for the bound
    EXPECT_EQ(m.stats().free_admits, 2u);
    EXPECT_EQ(m.stats().forced_admits, 1u);
    EXPECT_EQ(m.stats().depth_retires, 1u);
    EXPECT_EQ(m.stats().conflict_retires, 0u);
    EXPECT_EQ(m.stats().max_pending, 2u);
}

TEST(PipelineObserver, ReadAfterWriteIsAttributedToTheConflictNotTheDepth) {
    PipelineModel m("t", 8, false);
    m.dispatch(1, {wr(100)}, W(2.0));
    m.dispatch(1, {rd(100), wr(200)}, W(3.0));   // reads what the first wrote
    EXPECT_EQ(m.stats().free_admits, 1u);
    EXPECT_EQ(m.stats().forced_admits, 1u);
    EXPECT_EQ(m.stats().conflict_retires, 1u);
    EXPECT_EQ(m.stats().depth_retires, 0u);
}

TEST(PipelineObserver, ConflictRetiresEveryEarlierDispatchToo) {
    PipelineModel m("t", 8, false);
    m.dispatch(1, {wr(100)}, W(1.0));
    m.dispatch(1, {wr(200)}, W(1.0));
    m.dispatch(1, {wr(300)}, W(1.0));
    m.dispatch(1, {rd(200)}, W(1.0));   // conflicts with the second; the first precedes it
    EXPECT_EQ(m.stats().conflict_retires, 2u);
}

TEST(PipelineObserver, OnlyTheBarrierModelRetiresAtOtherOperations) {
    PipelineModel optimistic("o", 8, false), pessimistic("p", 8, true);
    for (PipelineModel* m : {&optimistic, &pessimistic}) {
        m->dispatch(1, {wr(100)}, W(1.0));
        m->other_operation();                // a graphics span ran between the dispatches
        m->dispatch(1, {rd(500), wr(600)}, W(1.0));
    }
    EXPECT_EQ(optimistic.stats().barrier_retires, 0u);
    EXPECT_EQ(pessimistic.stats().barrier_retires, 1u);
    EXPECT_EQ(pessimistic.stats().max_pending, 1u);   // the barrier emptied the queue
    EXPECT_EQ(optimistic.stats().max_pending, 2u);
}

TEST(PipelineObserver, ASubmitBoundaryRetiresEverythingInEveryModel) {
    PipelineModel m("t", 8, false);
    m.dispatch(1, {wr(100)}, W(1.0));
    m.dispatch(1, {wr(200)}, W(1.0));
    m.dispatch(2, {wr(300)}, W(1.0));   // next submit: the guest observes completion at submit end
    EXPECT_EQ(m.stats().barrier_retires, 2u);
    EXPECT_EQ(m.stats().max_pending, 2u);
}

TEST(PipelineObserver, WaitWeightedFractionUsesTheMeasuredWait) {
    PipelineModel m("t", 8, false);
    m.dispatch(1, {wr(100)}, W(9.0));            // free, long wait
    m.dispatch(1, {rd(100), wr(200)}, W(1.0));   // forced, short wait
    EXPECT_DOUBLE_EQ(m.stats().free_fraction(), 0.5);
    EXPECT_DOUBLE_EQ(m.stats().wait_free_fraction(), 0.9);
    // The second one waited for the first (a conflict), so the queue was empty when it was admitted.
    EXPECT_DOUBLE_EQ(m.stats().wait_overlapped_fraction(), 0.0);
    PipelineModel independent("i", 8, false);
    independent.dispatch(1, {wr(100)}, W(9.0));
    independent.dispatch(1, {wr(200)}, W(1.0));   // joins the pending first one
    EXPECT_DOUBLE_EQ(independent.stats().wait_overlapped_fraction(), 0.1);
}

TEST(PipelineObserver, DispatchWithNoRangesIsAlwaysFreeAndConflictsWithNothing) {
    PipelineModel m("t", 8, false);
    m.dispatch(1, {}, W(1.0));
    m.dispatch(1, {{0, 0, true, true}}, W(1.0));   // a zero-byte binding carries no range
    EXPECT_EQ(m.stats().free_admits, 2u);
}

TEST(PipelineObserver, TheFourModelsDifferWhereTheirAssumptionsDiffer) {
    PipelineObserver observer;
    // Ten independent dispatches with a graphics span after each one.
    for (uint64_t i = 0; i < 10; ++i) {
        observer.dispatch(1, {rd(0), wr(1000 + i * 100)}, W(1.0));
        observer.other_operation(ObservedOperation::GraphicsSpan);
    }
    const auto ranges_deep = observer.stats(1);
    const auto barriers_deep = observer.stats(3);
    EXPECT_EQ(ranges_deep.free_admits, 10u);       // nothing in the ranges forces a wait
    EXPECT_EQ(ranges_deep.max_pending, 10u);
    EXPECT_EQ(barriers_deep.free_admits, 10u);     // each dispatch is admitted before its barrier...
    EXPECT_EQ(barriers_deep.max_pending, 1u);      // ...but nothing ever stays pending across one
    EXPECT_EQ(barriers_deep.barrier_retires, 10u);
    EXPECT_EQ(barriers_deep.overlapped_admits, 0u);   // so no two dispatches were ever in flight together
    EXPECT_EQ(ranges_deep.overlapped_admits, 9u);     // all but the first joined a pending one
    EXPECT_EQ(observer.dispatches(), 10u);
    EXPECT_EQ(observer.report().size(), 4u);
    EXPECT_NE(observer.report()[0].find("ranges-only/depth4"), std::string::npos);
}

// Timeline: the CPU sets up and submits, the GPU runs after what it is already running, and a
// writeback is paid when the dispatch retires. Hand-computed.
TEST(PipelineObserver, IndependentDispatchesOverlapSetupWithTheGpuAndSaveTime) {
    PipelineModel m("t", 8, false);
    m.dispatch(1, {wr(100)}, {10.0, 10.0, 5.0});
    m.dispatch(1, {wr(200)}, {10.0, 10.0, 5.0});
    // cpu: setup1 -> 10, gpu1 [10,20]; setup2 -> 20, gpu2 [20,30]. Drain: max(20,20)+5=25, max(25,30)+5=35.
    EXPECT_DOUBLE_EQ(m.stats().sequential_ms, 50.0);
    EXPECT_DOUBLE_EQ(m.stats().pipelined_ms, 35.0);
    EXPECT_DOUBLE_EQ(m.stats().predicted_ratio(), 0.7);
}

TEST(PipelineObserver, ADependentDispatchSavesNothing) {
    PipelineModel m("t", 8, false);
    m.dispatch(1, {wr(100)}, {10.0, 10.0, 5.0});
    m.dispatch(1, {rd(100), wr(200)}, {10.0, 10.0, 5.0});   // reads what the first wrote
    // The first must retire before the second is set up: cpu 10 -> max(10,20)+5=25, setup2 -> 35,
    // gpu2 [35,45], drain 45+5=50: exactly the sequential time.
    EXPECT_DOUBLE_EQ(m.stats().pipelined_ms, 50.0);
    EXPECT_DOUBLE_EQ(m.stats().predicted_ratio(), 1.0);
}

TEST(PipelineObserver, ABarrierBetweenDispatchesRemovesTheOverlap) {
    PipelineModel optimistic("o", 8, false), pessimistic("p", 8, true);
    for (PipelineModel* m : {&optimistic, &pessimistic}) {
        m->dispatch(1, {wr(100)}, {10.0, 10.0, 5.0});
        m->other_operation();
        m->dispatch(1, {wr(200)}, {10.0, 10.0, 5.0});
    }
    EXPECT_DOUBLE_EQ(optimistic.stats().pipelined_ms, 35.0);
    EXPECT_DOUBLE_EQ(pessimistic.stats().pipelined_ms, 50.0);
}

TEST(PipelineObserver, TheDepthBoundCapsHowMuchAFastCpuCanRunAhead) {
    // GPU-bound work: the CPU could race ahead, but depth 1 forces one retirement per dispatch.
    PipelineModel shallow("s", 1, false), deep("d", 8, false);
    for (PipelineModel* m : {&shallow, &deep})
        for (int i = 0; i < 4; ++i) m->dispatch(1, {wr(100 + 100 * i)}, {1.0, 10.0, 1.0});
    EXPECT_GT(shallow.stats().pipelined_ms, deep.stats().pipelined_ms);
}

}  // namespace
