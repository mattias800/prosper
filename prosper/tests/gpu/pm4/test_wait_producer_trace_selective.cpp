// The legacy selective renderer drain moves a PendWrite differently from the batch and FIFO
// routes. A producer tag must travel with the write rather than be looked up by current context.
#include "wait_producer_fixture.hpp"

namespace {
using namespace wait_producer_test;
class WaitProducerTraceSelective : public Fixture {};
TEST_F(WaitProducerTraceSelective, SelectiveRendererDrainKeepsOriginalIdentity) {
    const uint64_t target = address(&label);
    acb(0x20, write(target, 7));
    acb(0x38, release(address(&other), 8));
    acb(0x38, wait(target, 7));
    prosper_gpu_drain_renderer_writes();
    EXPECT_EQ(label, 7u);
    EXPECT_EQ(other, 0u);
    const auto output = finish();
    const auto decision = at_address(records(output, "wait"), target);
    const auto store = at_address(records(output, "store-notified"), target);
    ASSERT_FALSE(decision.empty());
    ASSERT_FALSE(store.empty());
    EXPECT_EQ(store.at("producer-occurrence"), decision.at("producer-occurrence"));
    EXPECT_EQ(store.at("producer-queue"), "0x20");
    EXPECT_NE(store.at("producer-submit"), decision.at("consumer-submit"));
    EXPECT_EQ(other, 8u);
}
} // namespace
