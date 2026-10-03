// The guest expects a real result plus a separate submit handle: a successful file read
// cannot become status=token, and a failed/pending read cannot be cleared by submit.
#include "hle/dispatch/dispatch.hpp"
#include "hle/kernel/apr_submission.hpp"
#include "host/image/exec_image.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace prosper {
uint32_t prosper_apr_register(const std::string&, uint64_t);
}
using namespace prosper;
// The stub is a GUEST entry point, so on Windows the host must call it with the System V ABI.
using GuestReadFile = uint64_t(PROSPER_GUEST_ABI *)(uint64_t,uint64_t,uint64_t,uint64_t,
                                                  uint64_t,uint64_t,uint64_t,uint64_t,uint64_t);
namespace {
uint64_t ptr(const void* p) { return reinterpret_cast<uint64_t>(p); }
class AprSubmission : public testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        submit = Hle::lookup("ASoW5WE-UPo");
        submit_id = Hle::lookup("qvMUCyyaCSI");
        wait = Hle::lookup("rqwFKI4PAiM");
        reset = Hle::lookup("baQO9ez2gL4");
        ASSERT_TRUE(submit && submit_id && wait && reset);
        path = prosper_test::test_scratch_file("apr-submit-result.bin");
        FILE* f = std::fopen(path.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        ASSERT_EQ(std::fwrite(expected.data(), 1, expected.size(), f), expected.size());
        ASSERT_EQ(std::fclose(f), 0);
        file_id = prosper_apr_register(path, expected.size());
        // install_stubs claims a fixed aperture, and every platform refuses a second claim of the
        // same base, so a direct all-cases run failed from its second case. Every case needs the same
        // one slot: install it once. The slot vector is static because dispatch keeps a pointer to it.
        static const std::vector<ImportSlot> slots{{"libSceAmpr","mQ16-QdKv7k"}};
        static const std::string install_error = [] {
            std::string error;
            if (install_stubs(slots, 0x720000000ull, 96, &error))
                return std::string();
            return error.empty() ? std::string("install_stubs failed") : error;
        }();
        ASSERT_TRUE(install_error.empty()) << install_error;
        read = reinterpret_cast<GuestReadFile>(stub_addr(0));
        reset(ptr(cb.data()), 0, 0, 0, 0, 0);
        output.fill(0xa5);
    }
    void TearDown() override { reset(ptr(cb.data()), 0, 0, 0, 0, 0); std::remove(path.c_str()); }
    uint64_t read_bytes(uint32_t id) {
        return read(ptr(cb.data()), ptr(cb.data())+0x18, ptr(cb.data())+0x20,
                    id, ptr(dst.data()), expected.size(), 0, 0, 0);
    }
    uint32_t word(size_t offset) { uint32_t value; std::memcpy(&value, output.data()+offset, 4); return value; }
    uint64_t publish() { return submit(ptr(cb.data()), 6, ptr(output.data()+4), ptr(output.data()+12), 0, 0); }
    HleFn submit{}, submit_id{}, wait{}, reset{};
    GuestReadFile read{};
    std::array<uint8_t,0x48> cb{};
    const std::array<uint8_t,8> expected{3,17,42,99,8,6,2,1};
    std::array<uint8_t,8> dst{};
    std::array<uint8_t,24> output{};
    std::string path;
    uint32_t file_id{};
};
}
TEST_F(AprSubmission, SuccessfulReadPublishesBothResultWordsAndWaitableId) {
    ASSERT_EQ(read_bytes(file_id), 0u);
    ASSERT_EQ(dst, expected) << "real file bytes must reach the guest before a success result";
    ASSERT_EQ(publish(), 0u);
    EXPECT_EQ(word(4), 0u);
    EXPECT_EQ(word(8), 0u);
    EXPECT_NE(word(12), 0xa5a5a5a5u);
    EXPECT_EQ(wait(word(12),0,0,0,0,0), 0u);
    EXPECT_EQ(wait(word(12),0,0,0,0,0), 0u);
    for (size_t i=16;i<output.size();++i) EXPECT_EQ(output[i],0xa5) << "ID must not overwrite its neighbor";
}
TEST_F(AprSubmission, FailedReadIsRetainedUntilCommandBufferReset) {
    ASSERT_EQ(std::remove(path.c_str()), 0);
    ASSERT_NE(read_bytes(file_id), 0u);
    EXPECT_NE(publish(), 0u);
    for (uint8_t b:output) EXPECT_EQ(b,0xa5) << "failed execution must not publish a completed result";
    reset(ptr(cb.data()),0,0,0,0,0);
    EXPECT_EQ(publish(),0u) << "a real rewind discards the failed batch";
}
TEST_F(AprSubmission, PendingExecutionDoesNotPublishOrCompleteAnotherBuffer) {
    {
        AprReadExecution pending(ptr(cb.data()));
        EXPECT_NE(publish(),0u);
        for (uint8_t b:output) EXPECT_EQ(b,0xa5);
        std::array<uint8_t,0x48> other{};
        EXPECT_EQ(submit(ptr(other.data()),1,ptr(output.data()+4),ptr(output.data()+12),0,0),0u);
        pending.finish(0);
    }
    EXPECT_EQ(publish(),0u);
}
TEST_F(AprSubmission, GetIdWritesExactlyFourBytesAndReturnsStatus) {
    ASSERT_EQ(read_bytes(file_id),0u);
    ASSERT_EQ(submit_id(ptr(cb.data()),6,ptr(output.data()+4),0,0,0),0u);
    EXPECT_EQ(wait(word(4),0,0,0,0,0),0u);
    for (size_t i=8;i<output.size();++i) EXPECT_EQ(output[i],0xa5);
}
TEST_F(AprSubmission, UnwritableOutputReturnsErrorWithoutHostFault) {
    ASSERT_EQ(read_bytes(file_id),0u);
    EXPECT_NE(submit(ptr(cb.data()),6,1,ptr(output.data()+12),0,0),0u);
    EXPECT_EQ(word(12),0xa5a5a5a5u);
    EXPECT_EQ(publish(),0u) << "failed output publication does not consume the batch";
}

TEST_F(AprSubmission, UndeliveredGatherSegmentCannotBecomeSubmitSuccess) {
    ASSERT_EQ(read_bytes(file_id),0u);
    HleFn gather = Hle::lookup("BVmR1H8l+XI");
    ASSERT_NE(gather,nullptr);
    ASSERT_NE(gather(ptr(cb.data()),ptr(cb.data())+0x18,ptr(cb.data())+0x20,
                     1,expected.size(),0),0u) << "actual host read cannot reach destination1";
    EXPECT_NE(publish(),0u);
    for (uint8_t b:output) EXPECT_EQ(b,0xa5);
}
TEST_F(AprSubmission, LaterSuccessfulReadDoesNotEraseEarlierFailure) {
    {
        AprReadExecution failure(ptr(cb.data()));
    }
    ASSERT_EQ(read_bytes(file_id),0u);
    EXPECT_NE(publish(),0u);
    for (uint8_t b:output) EXPECT_EQ(b,0xa5);
}
TEST_F(AprSubmission, BoundBufferStillWritesResultAndSeparateId) {
    HleFn bind = Hle::lookup("H896Pt-yB4I");
    ASSERT_NE(bind,nullptr);
    // A recorded binding with no equeue still used to suppress both outputs. A nonzero tag
    // must not stand in for a kernel submit ID regardless of event delivery.
    ASSERT_EQ(bind(ptr(cb.data()),0,0,0x14000000000003e8ull,0,0),0u);
    ASSERT_EQ(read_bytes(file_id),0u);
    ASSERT_EQ(publish(),0u);
    EXPECT_EQ(word(4),0u);
    EXPECT_EQ(word(8),0u);
    EXPECT_NE(word(12),0xa5a5a5a5u);
    EXPECT_EQ(wait(word(12),0,0,0,0,0),0u);
    HleFn destroy = Hle::lookup("Qs1xtplKo0U");
    ASSERT_NE(destroy,nullptr);
    destroy(ptr(cb.data()),0,0,0,0,0);
}
