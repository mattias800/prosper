// #5070: a missing newest tag must erase an older known source, while the existing overlay still
// determines the predicate. This isolated process exhausts the documented diagnostic bounds.
#include "wait_producer_fixture.hpp"

namespace {
using namespace wait_producer_test;
class WaitProducerTraceBudget : public Fixture {};
TEST_F(WaitProducerTraceBudget, MissingLatestTagAndBoundedRecordsStayExplicit) {
    ASSERT_EQ(wait_producer_trace_counters().producers, 0u);
    const uint64_t target = address(&label);
    fold(wait(target, 0));
    std::array<uint64_t, 64> additional_labels{};
    for (size_t i = 0; i < 63; ++i) fold(wait(address(&additional_labels[i]), 0));
    fold(wait(address(&additional_labels.back()), 0)); // Cohort 65 is explicitly omitted.
    auto packet = write(target, 1);
    for (size_t i = 0; i < 4096; ++i) fold(packet);
    packet[5] = 2;
    fold(packet); // The cap prevents this newest producer from receiving an identity.
    fold(wait(target, 2));
    EXPECT_EQ(label, 0u);
    const auto output = finish();
    const auto decisions = records(output, "wait");
    ASSERT_EQ(decisions.size(), 65u);
    EXPECT_EQ(decisions.back().at("producer-present"), "0");
    EXPECT_EQ(decisions.back().at("producer-queue-known"), "0");
    EXPECT_EQ(decisions.back().at("overlay"), "1");
    EXPECT_EQ(decisions.back().at("effective"), "0x2");
    EXPECT_EQ(decisions.back().at("satisfied"), "1");
    EXPECT_TRUE(at_address(decisions, address(&additional_labels.back())).empty());
    EXPECT_EQ(label, 2u);
    const auto counters = wait_producer_trace_counters();
    EXPECT_EQ(counters.producers, 4096u);
    EXPECT_EQ(counters.missing_producers, 1u);
    EXPECT_EQ(counters.stores, 4097u);
    EXPECT_EQ(counters.labels, 64u);
    EXPECT_EQ(counters.label_overflow, 1u);
    EXPECT_EQ(counters.records, 512u);
    EXPECT_GT(counters.record_overflow, 0u);
    EXPECT_NE(output.find("limit=producers omitted-ordinal=1 data=TRUNCATED"), std::string::npos);
    EXPECT_NE(output.find("limit=records omitted-ordinal=1 data=TRUNCATED"), std::string::npos);
    EXPECT_NE(output.find("limit=labels omitted-ordinal=1 data=TRUNCATED"), std::string::npos);
}
} // namespace
