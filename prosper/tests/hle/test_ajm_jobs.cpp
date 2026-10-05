// test_ajm_jobs — the direct AJM job shapes queue onto the batch for BatchStart to execute.
//
// DecodeSingle/Run/RunSplit were unregistered, so the dispatcher answered `0` for jobs nobody
// queued. Every arm drives the real NIDs: validation arms refuse bad inputs, the execution arms
// prove a queued job reaches the decode pipeline (its sideband carries the decode error for
// undecodable input), and a Run given a short sideband gets a truncated write, never an overflow.
// Exports with outputs prosper does not compute stay unregistered, and that is pinned too.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

using Hle8Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t);
using Hle10Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                             uint64_t, uint64_t, uint64_t);
static uint64_t addr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}

static constexpr uint64_t kInvalidParameter = 0xffffffff80930005ull;
static constexpr uint64_t kDecodeError = 0xffffffff80930005ull;  // sideband iResult value

static Hle10Fn job_fn(const char* name) { return (Hle10Fn)Hle::lookup(nid_hash(name)); }

// Run every job queued under `batch` so no test leaves jobs in the process-global map.
static void drain(uint64_t batch) {
    Hle10Fn start2 = job_fn("sceAjmBatchStart");
    ASSERT_NE(start2, nullptr);
    uint32_t batch_id = 0;
    start2(1, batch, 0, 0, addr(&batch_id), 0, 0, 0, 0, 0);
}

TEST(AjmJobs, BoundAndDeliberatelyUnbound) {
    register_builtin_hle();
    for (const char* name : {"sceAjmBatchJobDecodeSingle", "sceAjmBatchJobRun",
                             "sceAjmBatchJobRunSplit", "sceAjmMemoryRegister",
                             "sceAjmMemoryUnregister", "sceAjmDecAt9ParseConfigData",
                             "sceAjmStrError"})
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    // Registering these would only hide them from the unimplemented-call alarm: their outputs
    // (encoded data, codec/gapless/stream info, MP3 frame headers) are not computed, and the
    // control/resample payloads would be dropped.
    for (const char* name : {"sceAjmBatchJobEncode", "sceAjmBatchJobControl",
                             "sceAjmBatchJobGetCodecInfo", "sceAjmBatchJobGetGaplessDecode",
                             "sceAjmBatchJobGetInfo", "sceAjmBatchJobSetResampleParameters",
                             "sceAjmDecMp3ParseFrame"})
        EXPECT_EQ(Hle::lookup(nid_hash(name)), nullptr) << name << " must stay unregistered";
}

TEST(AjmJobs, DirectShapesValidate) {
    register_builtin_hle();
    Hle10Fn decode_single = job_fn("sceAjmBatchJobDecodeSingle");
    Hle10Fn run = job_fn("sceAjmBatchJobRun");
    Hle10Fn run_split = job_fn("sceAjmBatchJobRunSplit");
    for (void* f : {(void*)decode_single, (void*)run, (void*)run_split}) ASSERT_NE(f, nullptr);
    uint8_t io[64]{};
    uint8_t result[32]{};
    const uint64_t batch = 0xBEEF0001ull;  // batch-info key; jobs queue under it
    EXPECT_EQ(decode_single(0, 1, addr(io), 64, addr(io), 64, addr(result), 0, 0, 0),
              kInvalidParameter)
        << "null batch";
    EXPECT_EQ(decode_single(batch, 1, 0, 64, addr(io), 64, addr(result), 0, 0, 0),
              kInvalidParameter)
        << "null input";
    EXPECT_EQ(decode_single(batch, 1, addr(io), 64, 0, 64, addr(result), 0, 0, 0),
              kInvalidParameter)
        << "null output";
    EXPECT_EQ(decode_single(batch, 1, addr(io), 0, addr(io), 64, addr(result), 0, 0, 0), 0u)
        << "a zero input size is accepted, as by sceAjmBatchJobDecode (end of stream)";
    EXPECT_EQ(run(batch, 1, 0, 0, 64, addr(io), 64, 0, 0, 0), kInvalidParameter) << "null input";
    EXPECT_EQ(run(0, 1, 0, addr(io), 64, addr(io), 64, 0, 0, 0), kInvalidParameter) << "null batch";
    EXPECT_EQ(run(batch, 1, 0, addr(io), 64, addr(io), 64, 0, 0, 0), 0u) << "valid Run queues";

    uint64_t in_desc[2 * 5];
    uint64_t out_desc[2 * 3];
    for (int i = 0; i < 5; ++i) { in_desc[2 * i] = addr(io); in_desc[2 * i + 1] = 16; }
    for (int i = 0; i < 3; ++i) { out_desc[2 * i] = addr(io); out_desc[2 * i + 1] = 16; }
    EXPECT_EQ(run_split(batch, 1, 0, 0, 1, addr(out_desc), 1, 0, 0, 0), kInvalidParameter)
        << "null descriptor arrays";
    EXPECT_EQ(run_split(batch, 1, 0, addr(in_desc), 5, addr(out_desc), 1, 0, 0, 0),
              kInvalidParameter)
        << "five input fragments overflow the four-entry job";
    EXPECT_EQ(run_split(batch, 1, 0, addr(in_desc), 1, addr(out_desc), 3, 0, 0, 0),
              kInvalidParameter)
        << "three outputs overflow the two-entry job";
    EXPECT_EQ(run_split(batch, 1, 0, addr(in_desc), 4, addr(out_desc), 2, 0, 0, 0), 0u)
        << "the four-in/two-out maximum queues";
    drain(batch);
}

TEST(AjmJobs, RunNeverWritesPastTheGuestSideband) {
    register_builtin_hle();
    Hle10Fn run = job_fn("sceAjmBatchJobRun");
    ASSERT_NE(run, nullptr);
    uint8_t garbage[64];
    std::memset(garbage, 0xAB, sizeof garbage);
    uint8_t pcm[512];
    uint8_t sideband[32];
    std::memset(sideband, 0xDD, sizeof sideband);
    const uint64_t batch = 0xBEEF0003ull;
    // An undecodable job on an unknown instance: execution writes the error sideband, but the
    // guest gave only 8 bytes for it.
    ASSERT_EQ(run(batch, 0x7FFFFFFFu, 0, addr(garbage), sizeof garbage, addr(pcm), sizeof pcm,
                  addr(sideband), 8, 0),
              0u);
    drain(batch);
    int32_t i_result = 0;
    std::memcpy(&i_result, sideband, sizeof i_result);
    EXPECT_EQ((uint64_t)(uint32_t)i_result, (uint64_t)(uint32_t)kDecodeError)
        << "the result header is written";
    for (size_t i = 8; i < sizeof sideband; ++i)
        EXPECT_EQ(sideband[i], 0xDDu) << "byte " << i << " lies past the guest's 8-byte sideband";
}

TEST(AjmJobs, QueuedJobExecutesWithErrorSideband) {
    register_builtin_hle();
    Hle10Fn decode_single = (Hle10Fn)Hle::lookup(nid_hash("sceAjmBatchJobDecodeSingle"));
    Hle10Fn start2 = (Hle10Fn)Hle::lookup(nid_hash("sceAjmBatchStart"));
    ASSERT_NE(decode_single, nullptr);
    ASSERT_NE(start2, nullptr);
    // Garbage bytes are not valid compressed audio: execution must report the decode error in
    // the sideband rather than writing silence. The instance id is deliberately unknown, which
    // exercises the missing-instance path too.
    uint8_t garbage[64];
    std::memset(garbage, 0xAB, sizeof garbage);
    uint8_t pcm[512];
    std::memset(pcm, 0xCC, sizeof pcm);
    uint8_t result[32];
    std::memset(result, 0xDD, sizeof result);
    const uint64_t batch = 0xBEEF0002ull;
    ASSERT_EQ(decode_single(batch, 0x7FFFFFFFu, addr(garbage), sizeof garbage, addr(pcm),
                            sizeof pcm, addr(result), 0, 0, 0),
              0u)
        << "queueing succeeds; execution reports";
    uint32_t batch_id = 0;
    ASSERT_EQ(start2(1, batch, 0, 0, addr(&batch_id), 0, 0, 0, 0, 0), 0u);
    EXPECT_NE(batch_id, 0u) << "BatchStart published a batch id: it ran";
    int32_t i_result = 0;
    std::memcpy(&i_result, result, sizeof i_result);
    EXPECT_EQ((uint64_t)(uint32_t)i_result, (uint64_t)(uint32_t)kDecodeError)
        << "undecodable input reports the decode error in the sideband";
    EXPECT_EQ(pcm[0], 0xCC) << "failed decode writes no PCM";
}

TEST(AjmJobs, ConfigParseAndMisc) {
    register_builtin_hle();
    HleFn parse = Hle::lookup(nid_hash("sceAjmDecAt9ParseConfigData"));
    HleFn mem_reg = Hle::lookup(nid_hash("sceAjmMemoryRegister"));
    HleFn mem_unreg = Hle::lookup(nid_hash("sceAjmMemoryUnregister"));
    HleFn strerror = Hle::lookup(nid_hash("sceAjmStrError"));
    for (HleFn f : {parse, mem_reg, mem_unreg, strerror})
        ASSERT_NE(f, nullptr);
    // FE 72 09 F0: the ATRAC9 config from the B2 comment = 48 kHz stereo.
    const uint8_t config[4] = {0xFE, 0x72, 0x09, 0xF0};
    uint32_t info[5]{};
    ASSERT_EQ(parse(addr(config), addr(info), 0, 0, 0, 0), 0u);
    EXPECT_EQ(info[0], 2u) << "channels";
    EXPECT_EQ(info[1], 48000u) << "sample rate";
    EXPECT_GT(info[2], 0u) << "frame samples";
    EXPECT_GT(info[4], 0u) << "superframe bytes";
    const uint8_t bad_config[4] = {0x00, 0x00, 0x00, 0x00};
    EXPECT_EQ(parse(addr(bad_config), addr(info), 0, 0, 0, 0), kInvalidParameter)
        << "bad config fails";
    EXPECT_EQ(parse(0, addr(info), 0, 0, 0, 0), kInvalidParameter) << "null config";
    EXPECT_EQ(mem_reg(1, addr(info), 8, 0, 0, 0), 0u);
    EXPECT_EQ(mem_unreg(1, addr(info), 0, 0, 0, 0), 0u);
    const uint64_t msg = strerror(5, 0, 0, 0, 0, 0);
    EXPECT_NE(msg, 0u) << "StrError never returns null (a %s formatter may not survive it)";
}
