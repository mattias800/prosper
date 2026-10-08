// test_agc_multi_acb -- sceAgcDriverSubmitMultiAcbs (NID HF3YllT3mXU) folds a batch of async-compute
// streams into its queue's own register context, in order, in firmware-sized chunks.
//
// THE DEFECT. The call was unregistered: the dispatcher answered 0 ("unimplemented, returning 0"),
// so a title's async-compute work was never folded at all. Assassin's Creed Black Flag Resynced calls
// it during boot. The contract is read from the project's testdata/sprx/libSceAgcDriver.sprx (sha256
// 7399b4eb...92ba9b; export 0x4830, worker 0x4570):
//   (uint32 queue, const uint64_t* streams, const uint32_t* dwords, uint32 count);
//   count == 0 -> 0x8a6d0000; at most 0x80 descriptors per queue submit, repeated until the count is
//   consumed; the first non-zero chunk result is returned. No queue-id range check in that build.
// The empty-batch code is BUILD-DEPENDENT (CONFIDENCE: MED): another build of the module (worker
// 0x4af0, #4741) returns 0x8a6d0109. This test pins the project copy's constant.
// The 0x8a6c000a refusals are prosper's own fail-visible policy; the firmware worker validates nothing.
//
// WHAT EACH ASSERTION KILLS:
//   RegisteredAndFoldsIntoItsQueueOnly   unregistered again (stub returning 0) -> no submission and
//                                        no register; folded into the GRAPHICS context instead of the
//                                        queue's -> the queue-0 probe sees the write, or the queue's
//                                        probe does not
//   EmptyBatchReturnsTheDriverCode       count == 0 returns 0, another constant, or submits anyway
//   ChunksAtOneHundredTwentyEight        a batch above 128 descriptors is truncated at 128 (the tail
//                                        writes are missing) or is folded as one submission (the
//                                        submit counter moves by 1, not 2)
//   RefusalInSecondChunkKeepsTheFirst    a refusal in chunk 2 that also unwinds chunk 1 (its writes
//                                        missing, submits +0), or that is swallowed (0 returned), or
//                                        that commits chunk 2's valid prefix (descriptor 128 present)
//   ZeroLengthSegmentIsSkipped           a zero-length descriptor aborts or drops its neighbour
//   UnreadableArraysAreRejected          unreadable descriptor arrays are dereferenced (crash) or
//                                        answered with success
//   UnreadableDwordsArrayAloneIsRejected only the streams array is checked (dwords read -> crash)
//   UnalignedStreamIsRejected            the alignment refusal is removed (the stream is folded)
//   OversizedStreamIsRejected            the stream-extent refusal is removed (the byte extent wraps)
//   OversizedCountIsRejected             the count-extent refusal is removed (the array extent wraps)
//   DescriptorOrderIsPreserved           descriptors are folded out of order: the last write to the
//                                        same register must win
//   SubmitAcbIgnoresTheQueuesUpperHalf   sceAgcDriverSubmitAcb keys its compute context on all 64 bits
//                                        of the first argument (#4743), so a guest that leaves rdi's
//                                        upper half dirty lands in a different context from the one
//                                        sceAgcDriverSubmitMultiAcbs names with the same low 32 bits
#include "hle/dispatch/dispatch.hpp"
#include "gpu/pm4/command_processor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper;
extern "C" void prosper_agc_submit_stats(uint64_t* submits, uint64_t* draws);
extern "C" bool prosper_agc_submit_sh_reg(uint64_t queue, uint32_t offset, uint32_t* value);

namespace {
constexpr const char* kNid = "HF3YllT3mXU";   // sceAgcDriverSubmitMultiAcbs
// Written as literals so a mutated constant in the handler cannot make the arms agree with itself.
constexpr uint64_t kEmptyBatch = 0x8a6d0000ull;   // project copy; see header
constexpr uint64_t kInvalidArg = 0x8a6c000aull;

uint64_t U(const void* p) {
    return reinterpret_cast<uintptr_t>(p);
}
uint32_t header(uint32_t words, uint32_t op, uint32_t sub = 0) {
    return 0xc0000000u | ((words - 2) << 16) | (op << 8) | (sub << 2);
}

// One SET_SH_REG stream writing `value` to SH register `offset`.
std::vector<uint32_t> sh_write(uint32_t offset, uint32_t value) {
    return {header(3, gpu::IT_SET_SH_REG), offset, value};
}

// Calls the handler and its paired return hook, as the generated import trampoline does.
uint64_t call(uint64_t queue, const uint64_t* streams, const uint32_t* words, uint32_t count) {
    HleFn fn = Hle::lookup(kNid);
    auto hook = Hle::return_hook_of(kNid);
    if (!fn || !hook) {
        // An unregistered NID is a failure to report, not a null call to make.
        ADD_FAILURE() << "sceAgcDriverSubmitMultiAcbs (" << kNid << ") is not registered";
        return ~0ull;
    }
    const uint64_t rc = fn(queue, U(streams), U(words), count, 0, 0);
    hook();
    return rc;
}

uint64_t submits() {
    uint64_t s = 0, d = 0;
    prosper_agc_submit_stats(&s, &d);
    return s;
}
bool sh(uint64_t queue, uint32_t offset, uint32_t* value) {
    return prosper_agc_submit_sh_reg(queue, offset, value);
}

class MultiAcb : public ::testing::Test {
protected:
    void SetUp() override { register_builtin_hle(); }
};
}   // namespace

TEST_F(MultiAcb, RegisteredAndFoldsIntoItsQueueOnly) {
    ASSERT_NE(Hle::lookup(kNid), nullptr) << "sceAgcDriverSubmitMultiAcbs must be registered";
    const auto a = sh_write(0x7a1, 0x11111111u);
    const auto b = sh_write(0x7a2, 0x22222222u);
    const uint64_t streams[2] = {U(a.data()), U(b.data())};
    const uint32_t words[2] = {uint32_t(a.size()), uint32_t(b.size())};
    const uint64_t before = submits();
    EXPECT_EQ(call(/* queue */ 5, streams, words, 2), 0u);
    EXPECT_EQ(submits(), before + 1) << "a batch inside one chunk is one submission";
    uint32_t v = 0;
    ASSERT_TRUE(sh(5, 0x7a1, &v));
    EXPECT_EQ(v, 0x11111111u);
    ASSERT_TRUE(sh(5, 0x7a2, &v));
    EXPECT_EQ(v, 0x22222222u);
    // The async-compute context is the queue's own: the same numeric offsets in the graphics context
    // (queue 0) are untouched.
    EXPECT_FALSE(sh(0, 0x7a1, &v)) << "an Acb batch must not write the graphics SH context";
    EXPECT_FALSE(sh(0, 0x7a2, &v));
    // And a different compute queue keeps its own context too.
    EXPECT_FALSE(sh(6, 0x7a1, &v)) << "queue 6 never received this batch";
}

TEST_F(MultiAcb, EmptyBatchReturnsTheDriverCode) {
    const uint64_t before = submits();
    EXPECT_EQ(call(5, nullptr, nullptr, 0), kEmptyBatch);
    EXPECT_EQ(submits(), before) << "an empty batch folds nothing";
}

TEST_F(MultiAcb, ChunksAtOneHundredTwentyEight) {
    // 130 descriptors: the firmware submits 128, then the remaining 2.
    constexpr uint32_t kCount = 130;
    std::vector<std::vector<uint32_t>> bodies;
    std::vector<uint64_t> streams;
    std::vector<uint32_t> words;
    for (uint32_t i = 0; i < kCount; ++i) {
        bodies.push_back(sh_write(0x800 + i, 0x1000 + i));
        streams.push_back(U(bodies.back().data()));
        words.push_back(uint32_t(bodies.back().size()));
    }
    const uint64_t before = submits();
    EXPECT_EQ(call(7, streams.data(), words.data(), kCount), 0u);
    EXPECT_EQ(submits(), before + 2) << "130 descriptors are two queue submits (128 + 2)";
    for (uint32_t i = 0; i < kCount; ++i) {
        uint32_t v = 0;
        ASSERT_TRUE(sh(7, 0x800 + i, &v)) << "descriptor " << i << " was dropped";
        EXPECT_EQ(v, 0x1000 + i) << "descriptor " << i;
    }
}

TEST_F(MultiAcb, RefusalInSecondChunkKeepsTheFirst) {
    // 130 descriptors; descriptor 129 (in the second chunk) names an unreadable stream. Chunk 1 has
    // already been folded when chunk 2 is refused, and chunk 2 is refused whole (prosper policy).
    constexpr uint32_t kCount = 130;
    constexpr uint32_t kBad = 129;
    std::vector<std::vector<uint32_t>> bodies;
    std::vector<uint64_t> streams;
    std::vector<uint32_t> words;
    for (uint32_t i = 0; i < kCount; ++i) {
        bodies.push_back(sh_write(0x900 + i, 0x2000 + i));
        streams.push_back(U(bodies.back().data()));
        words.push_back(uint32_t(bodies.back().size()));
    }
    streams[kBad] = 0x10;   // page 0 is never mapped
    const uint64_t before = submits();
    EXPECT_EQ(call(10, streams.data(), words.data(), kCount), kInvalidArg)
        << "the refusal in chunk 2 is the call's result";
    EXPECT_EQ(submits(), before + 1) << "chunk 1 was folded, chunk 2 was not";
    for (uint32_t i = 0; i < 128; ++i) {
        uint32_t v = 0;
        ASSERT_TRUE(sh(10, 0x900 + i, &v)) << "chunk 1 descriptor " << i << " was unwound";
        EXPECT_EQ(v, 0x2000 + i) << "descriptor " << i;
    }
    uint32_t v = 0;
    EXPECT_FALSE(sh(10, 0x900 + 128, &v)) << "chunk 2's valid prefix must not be folded";
    EXPECT_FALSE(sh(10, 0x900 + kBad, &v));
}

TEST_F(MultiAcb, ZeroLengthSegmentIsSkipped) {
    alignas(16) static uint32_t empty[4] = {};
    const auto real = sh_write(0x7b1, 0xdeadbeefu);
    const uint64_t streams[2] = {U(empty), U(real.data())};
    const uint32_t words[2] = {0, uint32_t(real.size())};
    const uint64_t before = submits();
    EXPECT_EQ(call(8, streams, words, 2), 0u);
    EXPECT_EQ(submits(), before + 1);
    uint32_t v = 0;
    ASSERT_TRUE(sh(8, 0x7b1, &v));
    EXPECT_EQ(v, 0xdeadbeefu) << "the real segment still lands beside the empty one";
}

TEST_F(MultiAcb, UnreadableArraysAreRejected) {
    const uint64_t before = submits();
    // Page 0 is never mapped: neither descriptor array is readable.
    EXPECT_EQ(call(5, reinterpret_cast<const uint64_t*>(0x10),
                   reinterpret_cast<const uint32_t*>(0x10), 4),
              kInvalidArg);
    EXPECT_EQ(submits(), before) << "a rejected batch folds nothing";
}

TEST_F(MultiAcb, UnreadableDwordsArrayAloneIsRejected) {
    const auto a = sh_write(0x7d1, 0x33333333u);
    const uint64_t streams[1] = {U(a.data())};
    const uint64_t before = submits();
    EXPECT_EQ(call(11, streams, reinterpret_cast<const uint32_t*>(0x10), 1), kInvalidArg);
    EXPECT_EQ(submits(), before) << "a rejected batch folds nothing";
    uint32_t v = 0;
    EXPECT_FALSE(sh(11, 0x7d1, &v));
}

TEST_F(MultiAcb, UnalignedStreamIsRejected) {
    // A valid stream copied one byte past a 4-byte boundary: readable, but not dword-aligned.
    const auto a = sh_write(0x7d2, 0x44444444u);
    alignas(16) static uint8_t raw[16 + 1] = {};
    memcpy(raw + 1, a.data(), a.size() * sizeof(uint32_t));
    const uint64_t streams[1] = {U(raw + 1)};
    const uint32_t words[1] = {uint32_t(a.size())};
    const uint64_t before = submits();
    EXPECT_EQ(call(11, streams, words, 1), kInvalidArg);
    EXPECT_EQ(submits(), before);
    uint32_t v = 0;
    EXPECT_FALSE(sh(11, 0x7d2, &v));
}

TEST_F(MultiAcb, OversizedStreamIsRejected) {
    // 0x40000003 dwords is just over 4 GiB. Its byte extent wraps a uint32 to exactly 12 bytes, which
    // this 3-dword stream really has, so only the extent refusal (not the readability check) can
    // reject it; 0x40000000 would wrap to 0 and be refused as unreadable even without the guard.
    const auto a = sh_write(0x7d3, 0x55555555u);
    const uint64_t streams[1] = {U(a.data())};
    const uint32_t words[1] = {0x40000003u};
    const uint64_t before = submits();
    EXPECT_EQ(call(11, streams, words, 1), kInvalidArg);
    EXPECT_EQ(submits(), before);
}

TEST_F(MultiAcb, OversizedCountIsRejected) {
    // 0x40000001 descriptors: the streams extent (x8) and dwords extent (x4) wrap a uint32 to 8 and 4
    // bytes, which these real 128-entry arrays cover, so only the count-extent refusal rejects it.
    // Null arrays would be refused as unreadable even without that guard.
    const auto a = sh_write(0x7d4, 0x66666666u);
    std::vector<uint64_t> streams(128, U(a.data()));
    std::vector<uint32_t> words(128, uint32_t(a.size()));
    const uint64_t before = submits();
    EXPECT_EQ(call(11, streams.data(), words.data(), 0x40000001u), kInvalidArg);
    EXPECT_EQ(submits(), before);
}

TEST_F(MultiAcb, SubmitAcbIgnoresTheQueuesUpperHalf) {
    // Both libSceAgcDriver entry points compare `edi`: the queue is a 32-bit argument, so the upper
    // half of rdi (legal garbage under the SysV ABI) is not part of its identity.
    constexpr const char* kAcbNid = "gSRnr79F8tQ";   // sceAgcDriverSubmitAcb
    HleFn acb = Hle::lookup(kAcbNid);
    auto acb_hook = Hle::return_hook_of(kAcbNid);
    ASSERT_NE(acb, nullptr);
    ASSERT_NE(acb_hook, nullptr);

    const auto one = sh_write(0x7e1, 0x0a0a0a0au);
    // SubmitAcb reads a 16-byte record {stream address, uint32 dwords, uint32 flags}.
    struct Record {
        uint64_t stream;
        uint32_t dwords;
        uint32_t flags;
    } record{U(one.data()), uint32_t(one.size()), 0};
    constexpr uint64_t kQueue = 0x12;
    constexpr uint64_t kDirty = 0xdead0000'00000000ull | kQueue;
    EXPECT_EQ(acb(kDirty, U(&record), record.dwords, 0, 0, 0), 0u);
    acb_hook();

    // The same hardware queue through the other entry point, with a clean register.
    const auto two = sh_write(0x7e2, 0x0b0b0b0bu);
    const uint64_t streams[1] = {U(two.data())};
    const uint32_t words[1] = {uint32_t(two.size())};
    EXPECT_EQ(call(kQueue, streams, words, 1), 0u);

    // One queue, one context: both writes are visible under the 32-bit queue id.
    uint32_t v = 0;
    ASSERT_TRUE(sh(kQueue, 0x7e1, &v)) << "SubmitAcb with a dirty rdi upper half missed queue 0x12";
    EXPECT_EQ(v, 0x0a0a0a0au);
    ASSERT_TRUE(sh(kQueue, 0x7e2, &v));
    EXPECT_EQ(v, 0x0b0b0b0bu);
}

TEST_F(MultiAcb, DescriptorOrderIsPreserved) {
    // Two writes to the SAME register: the later descriptor must win.
    const auto first = sh_write(0x7c1, 0xaaaaaaaau);
    const auto second = sh_write(0x7c1, 0xbbbbbbbbu);
    const uint64_t streams[2] = {U(first.data()), U(second.data())};
    const uint32_t words[2] = {uint32_t(first.size()), uint32_t(second.size())};
    EXPECT_EQ(call(9, streams, words, 2), 0u);
    uint32_t v = 0;
    ASSERT_TRUE(sh(9, 0x7c1, &v));
    EXPECT_EQ(v, 0xbbbbbbbbu);
}
