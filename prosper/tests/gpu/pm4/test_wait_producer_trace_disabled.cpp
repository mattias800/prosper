// An unarmed tracer must not create tags, counters or records. Actual queued writes and predicates
// still use the existing execution path; this is not a helper-only disabled stub test.
#include "wait_producer_fixture.hpp"

namespace {
using namespace wait_producer_test;
class WaitProducerTraceDisabled : public Fixture {};
TEST_F(WaitProducerTraceDisabled, RealQueueWorksWithoutTraceStateOrOutput) {
    ASSERT_FALSE(wait_producer_trace_enabled());
    acb(0x20, write(address(&label), 4));
    acb(0x38, wait(address(&label), 4));
    EXPECT_EQ(label, 0u);
    const auto output = finish();
    EXPECT_EQ(label, 4u);
    EXPECT_EQ(output.find("[wait-producer]"), std::string::npos);
    const auto counters = wait_producer_trace_counters();
    EXPECT_EQ(counters.producers, 0u);
    EXPECT_EQ(counters.decisions, 0u);
    EXPECT_EQ(counters.stores, 0u);
    EXPECT_EQ(counters.records, 0u);
}
} // namespace
