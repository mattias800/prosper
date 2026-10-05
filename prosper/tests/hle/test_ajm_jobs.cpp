// test_ajm_jobs — the direct AJM job shapes queue onto the batch for BatchStart to execute.
//
// These 14 exports were unregistered, so the dispatcher answered `0`. The builders
// (Decode/Run/RunSplit/Encode) would then report queued jobs nobody queued, and the getters
// success over unwritten result sidebands. Every arm drives the real NIDs: validation arms
// refuse bad inputs, and the execution arm proves a queued job actually reaches the decode
// pipeline (its sideband carries the decode error for undecodable input) instead of
// evaporating.
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

TEST(AjmJobs, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceAjmBatchJobDecodeSingle",
        "sceAjmBatchJobRun",
        "sceAjmBatchJobRunSplit",
        "sceAjmBatchJobEncode",
        "sceAjmBatchJobControl",
        "sceAjmBatchJobGetCodecInfo",
        "sceAjmBatchJobGetGaplessDecode",
        "sceAjmBatchJobGetInfo",
        "sceAjmBatchJobSetResampleParameters",
        "sceAjmMemoryRegister",
        "sceAjmMemoryUnregister",
        "sceAjmDecAt9ParseConfigData",
        "sceAjmDecMp3ParseFrame",
        "sceAjmStrError",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 14, "the 14 direct-job exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(AjmJobs, DirectShapesValidate) {
    register_builtin_hle();
    Hle10Fn decode_single = (Hle10Fn)Hle::lookup(nid_hash("sceAjmBatchJobDecodeSingle"));
    Hle10Fn run = (Hle10Fn)Hle::lookup(nid_hash("sceAjmBatchJobRun"));
    Hle10Fn run_split = (Hle10Fn)Hle::lookup(nid_hash("sceAjmBatchJobRunSplit"));
    Hle10Fn encode = (Hle10Fn)Hle::lookup(nid_hash("sceAjmBatchJobEncode"));
    Hle10Fn control = (Hle10Fn)Hle::lookup(nid_hash("sceAjmBatchJobControl"));
    Hle10Fn get_info = (Hle10Fn)Hle::lookup(nid_hash("sceAjmBatchJobGetInfo"));
    for (void* f : {(void*)decode_single, (void*)run, (void*)run_split, (void*)encode,
                    (void*)control, (void*)get_info})
        ASSERT_NE(f, nullptr);
    uint8_t io[64]{};
    uint8_t result[32]{};
    const uint64_t batch = 0xBEEF0001ull;  // batch-info key; jobs queue under it
    // Nulls and over-long sizes are refused before anything queues.
    EXPECT_EQ(decode_single(batch, 1, 0, 64, addr(io), 64, addr(result), 0, 0, 0),
              kInvalidParameter)
        << "null input";
    EXPECT_EQ(decode_single(batch, 1, addr(io), 64, 0, 64, addr(result), 0, 0, 0),
              kInvalidParameter)
        << "null output";
    EXPECT_EQ(run(batch, 1, 0, 0, 64, addr(io), 64, 0, 0, 0), kInvalidParameter)
        << "Run validates like DecodeSingle";
    EXPECT_EQ(run(batch, 1, 0, addr(io), 64, addr(io), 64, 0, 0, 0), 0u) << "valid Run queues";
    EXPECT_EQ(run_split(batch, 1, 0, 0, 1, addr(io), 1, 0, 0, 0), kInvalidParameter)
        << "null descriptor arrays";
    EXPECT_EQ(encode(batch, 1, addr(io), 64, addr(io), 64, addr(result), 0, 0, 0), 0u)
        << "valid encode queues (execution reports the error, not the call)";
    EXPECT_EQ(control(batch, 1, 0, 0, 64, addr(io), 64, 0, 0, 0), 0u);
    EXPECT_EQ(control(batch, 1, 0, 0, 0, 0, 0, 0, 0, 0), 0u) << "sidebands optional";
    EXPECT_EQ(get_info(batch, 1, addr(result), 64, 0, 0, 0, 0, 0, 0), 0u);
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
    HleFn mp3parse = Hle::lookup(nid_hash("sceAjmDecMp3ParseFrame"));
    HleFn set_resample = Hle::lookup(nid_hash("sceAjmBatchJobSetResampleParameters"));
    for (HleFn f : {parse, mem_reg, mem_unreg, strerror, mp3parse, set_resample})
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
    EXPECT_EQ(set_resample(1, 1, 0, 0, 0, 0), 0u);
    EXPECT_EQ(mp3parse(0, 0, 0, 0, 0, 0), 0u);
    const uint64_t msg = strerror(5, 0, 0, 0, 0, 0);
    EXPECT_NE(msg, 0u) << "StrError never returns null (a %s formatter may not survive it)";
}
