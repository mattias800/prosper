// #5070: log adjacency cannot identify a pending producer. Exercise actual HLE enqueue, modeled
// overlay and all notified writes without a device. None of these tests claims hardware ordering.
#include "wait_producer_fixture.hpp"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
using namespace wait_producer_test;
class WaitProducerTrace : public Fixture {};

TEST_F(WaitProducerTrace, RealQueueAndRecycledPacketStayWithTheirOwnWriter) {
    const uint64_t target = address(&label);
    auto producer = write(target, 0x63);
    const uint64_t packet = address(producer.data());
    acb(0xdead000000000020ull, producer);
    producer[5] = 0xbad;  // Recycle command storage after enqueue, before overlay/application.
    const auto other_producer = write(address(&other), 0x38);
    acb(0x38, other_producer);
    const auto consumer = wait(target, 0x63);
    acb(0x38, consumer);
    EXPECT_EQ(label, 0u) << "instrumentation must not publish a private pending write";
    const auto output = finish();
    const auto decision = at_address(records(output, "wait"), target);
    const auto store = at_address(records(output, "store-notified"), target);
    ASSERT_FALSE(decision.empty());
    ASSERT_FALSE(store.empty());
    EXPECT_EQ(decision.at("consumer-queue"), "0x38");
    EXPECT_EQ(decision.at("producer-queue-known"), "1");
    EXPECT_EQ(decision.at("producer-queue"), "0x20");
    EXPECT_EQ(decision.at("producer-submit-known"), "1");
    EXPECT_NE(decision.at("producer-submit"), decision.at("consumer-submit"));
    EXPECT_EQ(decision.at("producer-packet"), hex(packet));
    EXPECT_EQ(decision.at("producer-order-known"), "1");
    EXPECT_EQ(decision.at("producer-payload"), "0x63");
    EXPECT_EQ(decision.at("raw"), "0x0");
    EXPECT_EQ(decision.at("effective"), "0x63");
    EXPECT_EQ(decision.at("satisfied"), "1");
    EXPECT_EQ(store.at("producer-occurrence"), decision.at("producer-occurrence"));
    EXPECT_EQ(store.at("producer-submit"), decision.at("producer-submit"));
    EXPECT_EQ(store.at("producer-payload"), "0x63");
    EXPECT_EQ(label, 0x63u);
    EXPECT_EQ(other, 0x38u);
}

TEST_F(WaitProducerTrace, SamePacketReusedByAnotherQueueGetsANewOccurrence) {
    const uint64_t target = address(&label);
    auto packet = write(target, 1);
    acb(0x20, packet);
    acb(0x20, wait(target, 1));
    packet[5] = 2;
    acb(0x38, packet);
    acb(0x38, wait(target, 2));
    const auto output = finish();
    const auto decisions = records(output, "wait");
    ASSERT_EQ(decisions.size(), 2u);
    EXPECT_EQ(decisions[0].at("producer-packet"), decisions[1].at("producer-packet"));
    EXPECT_NE(decisions[0].at("producer-occurrence"), decisions[1].at("producer-occurrence"));
    EXPECT_EQ(decisions[0].at("producer-queue"), "0x20");
    EXPECT_EQ(decisions[1].at("producer-queue"), "0x38");
    EXPECT_EQ(label, 2u);
}

TEST_F(WaitProducerTrace, DirectFoldDoesNotBorrowPreviousSubmitIdentity) {
    acb(0x20, write(address(&other), 7));
    fold(write(address(&label), 9));
    fold(wait(address(&label), 9));
    const auto output = finish();
    const auto decision = at_address(records(output, "wait"), address(&label));
    ASSERT_FALSE(decision.empty());
    EXPECT_EQ(decision.at("producer-present"), "1");
    EXPECT_EQ(decision.at("producer-queue-known"), "0");
    EXPECT_EQ(decision.at("producer-submit-known"), "0");
    EXPECT_EQ(decision.at("consumer-queue-known"), "0");
    EXPECT_EQ(decision.at("consumer-submit-known"), "0");
    EXPECT_EQ(decision.at("producer-payload"), "0x9");
    EXPECT_EQ(label, 9u);
}

TEST_F(WaitProducerTrace, RawMatchAndOverlayMismatchRemainDifferent) {
    label = 0xf111ed;
    dcb(write(address(&label), 0x63));
    acb(0x20, wait(address(&label), 0xf111ed));
    const auto output = finish();
    const auto decision = at_address(records(output, "wait"), address(&label));
    ASSERT_FALSE(decision.empty());
    EXPECT_EQ(decision.at("producer-origin"), "1");
    EXPECT_EQ(decision.at("producer-submit-known"), "1");
    EXPECT_EQ(decision.at("producer-queue-known"), "0") << "DCB supplies no queue handle";
    EXPECT_EQ(decision.at("raw"), "0xf111ed");
    EXPECT_EQ(decision.at("effective"), "0x63");
    EXPECT_EQ(decision.at("satisfied"), "0");
    EXPECT_EQ(label, 0x63u);
}

TEST_F(WaitProducerTrace, NoMatchingPendingWriteDoesNotInventAProducer) {
    acb(0x20, write(address(&other), 3));
    label = 4;
    acb(0x38, wait(address(&label), 4));
    const auto output = finish();
    const auto decision = at_address(records(output, "wait"), address(&label));
    ASSERT_FALSE(decision.empty());
    EXPECT_EQ(decision.at("producer-present"), "0");
    EXPECT_EQ(decision.at("producer-queue-known"), "0");
    EXPECT_EQ(decision.at("overlay"), "0");
    EXPECT_EQ(decision.at("satisfied"), "1");
}

TEST_F(WaitProducerTrace, RendererBatchCarriesIdentityAcrossQueueMoves) {
    const uint64_t target = address(&label);
    acb(0x20, write(target, 5));
    acb(0x38, release(address(&other), 8));
    acb(0x38, wait(target, 5));
    prosper_gpu_drain_renderer_writes();
    EXPECT_EQ(label, 5u);
    EXPECT_EQ(other, 0u) << "observing a renderer write must not release its completion neighbor";
    const auto output = finish();
    const auto decision = at_address(records(output, "wait"), target);
    const auto store = at_address(records(output, "store-notified"), target);
    ASSERT_FALSE(decision.empty());
    ASSERT_FALSE(store.empty());
    EXPECT_EQ(store.at("producer-occurrence"), decision.at("producer-occurrence"));
    EXPECT_EQ(store.at("producer-queue"), "0x20");
    EXPECT_EQ(other, 8u);
}

TEST_F(WaitProducerTrace, UnmappedApplicationDoesNotReportAnActualStore) {
#ifdef _WIN32
    void* page = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    ASSERT_NE(page, nullptr);
#else
    const auto size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    void* page = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(page, MAP_FAILED);
#endif
    const uint64_t target = address(page);
    acb(0x20, write(target, 6));
    acb(0x38, wait(target, 6));
#ifdef _WIN32
    ASSERT_TRUE(VirtualFree(page, 0, MEM_RELEASE));
#else
    ASSERT_EQ(munmap(page, size), 0);
#endif
    const auto output = finish();
    EXPECT_EQ(records(output, "wait").size(), 1u);
    EXPECT_TRUE(records(output, "store-notified").empty())
        << "an application attempt skipped by the mapped-target guard is not a store";
    EXPECT_NE(output.find("write SKIPPED"), std::string::npos);
}

TEST_F(WaitProducerTrace, NestedDirectStoreCannotBorrowPendingApplicationIdentity) {
    const uint64_t target = address(&label);
    acb(0x20, wait(target, 0)); // Enroll the observed label through the real fold.
    const auto packet = write(target, 11);
    std::vector<Pm4Command> decoded;
    ASSERT_EQ(decode_pm4(packet.data(), packet.size(), decoded), packet.size());
    ASSERT_EQ(decoded.size(), 1u);
    decoded.front().stream_order = 1;
    const auto tag = capture_pending_writer_identity(decoded.front());
    ASSERT_NE(tag, nullptr);
    // A distinct command with the same trace metadata, including packet and stream order.
    GpuState::MemoryEffect unrelated(decoded.front(), 1);
    ASSERT_EQ(unrelated.cmd.stream_order, decoded.front().stream_order);
    ASSERT_EQ(unrelated.cmd.payload, decoded.front().payload);
    {
        PendingWriterTraceApplyScope pending(tag.get(), decoded.front());
        execute_ordered_memory_effect(unrelated);
    }
    const auto output = finish();
    EXPECT_EQ(label, 11u) << "the independent ordered memory effect really executed";
    EXPECT_TRUE(records(output, "store-notified").empty())
        << "current application context must not assign its identity to a distinct command";
}
} // namespace
