// test_audio_spatial — libSceAudio3d / libSceAudioPropagation / libSceAcm contracts.
//
// The defect this pins is the audio folder's recurring killer (audio/AGENTS.md): all three
// libraries were unregistered, so the dispatcher answered `0` — SCE_OK — while writing nothing.
// Audio out-parameters are overwhelmingly divisors (queue depths, ray counts, batch ids), so a
// title proceeds on garbage and dies in its own arithmetic far from the cause. Every TEST below
// therefore asserts what was WRITTEN on success, and that refused calls write NOTHING while
// answering a negative error the guest sign-checks.
//
// NID provenance: the 7 Audio3d NIDs are the PS5 3.20 firmware set; `ZIXln2K3XMk` is the orphan
// libSceAcm NID from a live SYBERIA boot, resolved here because nid_hash("sceAcmContextCreate")
// reproduces it exactly. The remaining 10 NIDs are nid_hash over the firmware stub export names
// (the hash itself is pinned 7/7 against firmware values, so the computed ones are trustworthy).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

using HleFn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }
static uint64_t sx(uint32_t v) { return (uint64_t)(int64_t)(int32_t)v; }
static bool is_error(uint64_t r) { return (int64_t)r < 0; }

static constexpr uint64_t kInvalidPort = (uint64_t)(int64_t)(int32_t)0x80EA0002u;
static constexpr uint64_t kInvalidParam = (uint64_t)(int64_t)(int32_t)0x80EA0004u;
static constexpr uint64_t kNoResources = (uint64_t)(int64_t)(int32_t)0x80EA0006u;
static constexpr uint64_t kNotReady = (uint64_t)(int64_t)(int32_t)0x80EA0007u;
static constexpr uint64_t kNoBackend = (uint64_t)(int64_t)(int32_t)0x80410002u;
static constexpr uint32_t kPortInvalid = 0xFFFFFFFFu;

TEST(AudioSpatial, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        // libSceAudio3d
        "sceAudio3dInitialize", "sceAudio3dGetDefaultOpenParameters", "sceAudio3dPortOpen",
        "sceAudio3dPortSetAttribute", "sceAudio3dPortGetQueueLevel", "sceAudio3dPortAdvance",
        "sceAudio3dPortPush",
        // libSceAcm
        "sceAcmContextCreate", "sceAcmContextDestroy", "sceAcmBatchStartBuffer",
        "sceAcmBatchStartBuffers", "sceAcmBatchWait",
        // libSceAudioPropagation
        "sceAudioPropagationSystemCreate", "sceAudioPropagationSystemQueryMemory",
        "sceAudioPropagationRoomCreate", "sceAudioPropagationSystemRegisterMaterial",
        "sceAudioPropagationSystemSetAttributes", "sceAudioPropagationSystemGetRays",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 18,
                  "the table must cover all 18 exports of the three libraries");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(AudioSpatial, NidsResolveToFirmwareValues) {
    // The 8 independently known NIDs: 7 from the PS5 3.20 set, plus the SYBERIA orphan that
    // nid_hash("sceAcmContextCreate") reproduces. A wrong literal must NOT match — that is the
    // positive control proving this test discriminates rather than echoing the hash.
    EXPECT_EQ(nid_hash("sceAudio3dInitialize"), "UmCvjSmuZIw");
    EXPECT_EQ(nid_hash("sceAudio3dGetDefaultOpenParameters"), "Im+jOoa5WAI");
    EXPECT_EQ(nid_hash("sceAudio3dPortOpen"), "XeDDK0xJWQA");
    EXPECT_EQ(nid_hash("sceAudio3dPortSetAttribute"), "Yq9bfUQ0uJg");
    EXPECT_EQ(nid_hash("sceAudio3dPortGetQueueLevel"), "YaaDbDwKpFM");
    EXPECT_EQ(nid_hash("sceAudio3dPortAdvance"), "lw0qrdSjZt8");
    EXPECT_EQ(nid_hash("sceAudio3dPortPush"), "VEVhZ9qd4ZY");
    EXPECT_EQ(nid_hash("sceAcmContextCreate"), "ZIXln2K3XMk");
    EXPECT_NE(nid_hash("sceAudio3dInitialize"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

// Must run before any Initialize: the NOT_READY contract is only observable pre-init.
TEST(Audio3d, RequiresInit) {
    register_builtin_hle();
    HleFn open = Hle::lookup(nid_hash("sceAudio3dPortOpen"));
    ASSERT_NE(open, nullptr);
    uint8_t params[0x20]{};
    uint32_t id = 0x12345678u;
    EXPECT_EQ(open(0, addr(params), addr(&id), 0, 0, 0), kNotReady);
    EXPECT_EQ(id, kPortInvalid) << "a refused open must still publish PORT_INVALID";
}

TEST(Audio3d, InitializeContract) {
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("sceAudio3dInitialize"));
    ASSERT_NE(init, nullptr);
    EXPECT_EQ(init(1, 0, 0, 0, 0, 0), kInvalidParam) << "nonzero reserved is rejected";
    EXPECT_EQ(init(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(init(0, 0, 0, 0, 0, 0), 0u) << "initialization is idempotent";
}

TEST(Audio3d, DefaultParameters) {
    register_builtin_hle();
    HleFn defaults = Hle::lookup(nid_hash("sceAudio3dGetDefaultOpenParameters"));
    ASSERT_NE(defaults, nullptr);
    EXPECT_EQ(defaults(0, 0, 0, 0, 0, 0), kInvalidParam) << "null out-pointer is rejected";
    uint8_t params[0x20 + 8];
    std::memset(params, 0xAA, sizeof params);
    EXPECT_EQ(defaults(addr(params), 0, 0, 0, 0, 0), 0u);
    uint64_t size_this = 0;
    uint32_t w[6]{};
    std::memcpy(&size_this, params + 0x00, 8);
    std::memcpy(w, params + 0x08, sizeof w);
    EXPECT_EQ(size_this, 0x20u) << "size_this versions the 0x20 prefix";
    EXPECT_EQ(w[0], 0x100u) << "granularity is a divisor the guest mixes against";
    EXPECT_EQ(w[1], 0u) << "rate 0 = 48 kHz";
    EXPECT_EQ(w[2], 512u) << "max_objects is nonzero";
    EXPECT_EQ(w[3], 2u) << "queue_depth is the divisor GetQueueLevel paces against";
    EXPECT_EQ(w[4], 2u) << "buffer_mode ADVANCE_AND_PUSH";
    EXPECT_EQ(w[5], 0u) << "pad is zero";
    EXPECT_EQ(params[0x20], 0xAA) << "nothing is written past the versioned prefix";
}

TEST(Audio3d, PortOpenValidation) {
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("sceAudio3dInitialize"));
    HleFn open = Hle::lookup(nid_hash("sceAudio3dPortOpen"));
    HleFn defaults = Hle::lookup(nid_hash("sceAudio3dGetDefaultOpenParameters"));
    ASSERT_NE(init, nullptr);
    ASSERT_NE(open, nullptr);
    ASSERT_NE(defaults, nullptr);
    ASSERT_EQ(init(0, 0, 0, 0, 0, 0), 0u);

    uint8_t params[0x20]{};
    ASSERT_EQ(defaults(addr(params), 0, 0, 0, 0, 0), 0u);
    uint32_t id = 0x12345678u;
    EXPECT_EQ(open(0, 0, addr(&id), 0, 0, 0), kInvalidParam);
    EXPECT_EQ(id, kPortInvalid);
    EXPECT_EQ(open(0, addr(params), 0, 0, 0, 0), kInvalidParam) << "null id pointer is rejected";

    auto open_with = [&](uint64_t size_this, uint32_t rate, uint32_t granularity,
                         uint32_t max_objects, uint32_t queue_depth,
                         uint32_t buffer_mode) -> uint64_t {
        uint8_t p[0x20]{};
        std::memcpy(p + 0x00, &size_this, 8);
        std::memcpy(p + 0x08, &granularity, 4);
        std::memcpy(p + 0x0c, &rate, 4);
        std::memcpy(p + 0x10, &max_objects, 4);
        std::memcpy(p + 0x14, &queue_depth, 4);
        std::memcpy(p + 0x18, &buffer_mode, 4);
        id = 0x12345678u;
        uint64_t r = open(0, addr(p), addr(&id), 0, 0, 0);
        EXPECT_EQ(id, kPortInvalid) << "a refused open publishes PORT_INVALID";
        return r;
    };
    EXPECT_EQ(open_with(0x30, 0, 0x100, 512, 2, 2), kInvalidParam) << "unknown struct version";
    EXPECT_EQ(open_with(0x20, 1, 0x100, 512, 2, 2), kInvalidParam) << "non-48kHz rate";
    EXPECT_EQ(open_with(0x20, 0, 0x80, 512, 2, 2), kInvalidParam) << "granularity under 0x100";
    EXPECT_EQ(open_with(0x20, 0, 0x180, 512, 2, 2), kInvalidParam) << "unaligned granularity";
    EXPECT_EQ(open_with(0x20, 0, 0x100, 0, 2, 2), kInvalidParam) << "zero objects";
    EXPECT_EQ(open_with(0x20, 0, 0x100, 512, 0, 2), kInvalidParam) << "zero queue depth";
    EXPECT_EQ(open_with(0x20, 0, 0x100, 512, 2, 3), kInvalidParam) << "unknown buffer mode";

    id = 0x12345678u;
    ASSERT_EQ(open(0, addr(params), addr(&id), 0, 0, 0), 0u);
    EXPECT_LE(id, 3u) << "port ids come from the 0..3 table";
}

TEST(Audio3d, QueueLevelPushAdvance) {
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("sceAudio3dInitialize"));
    HleFn open = Hle::lookup(nid_hash("sceAudio3dPortOpen"));
    HleFn level = Hle::lookup(nid_hash("sceAudio3dPortGetQueueLevel"));
    HleFn push = Hle::lookup(nid_hash("sceAudio3dPortPush"));
    HleFn advance = Hle::lookup(nid_hash("sceAudio3dPortAdvance"));
    HleFn setattr = Hle::lookup(nid_hash("sceAudio3dPortSetAttribute"));
    for (HleFn f : {init, open, level, push, advance, setattr}) ASSERT_NE(f, nullptr);
    ASSERT_EQ(init(0, 0, 0, 0, 0, 0), 0u);

    uint8_t params[0x20]{};
    uint32_t id = kPortInvalid;
    // A non-default depth proves GetQueueLevel reports the port's own depth, not a constant.
    uint32_t depth = 7;
    std::memcpy(params + 0x00, "\x20\x00\x00\x00\x00\x00\x00\x00", 8);  // size_this = 0x20
    std::memcpy(params + 0x08, "\x00\x01\x00\x00", 4);                 // granularity = 0x100
    std::memcpy(params + 0x10, "\x00\x02\x00\x00", 4);                 // max_objects = 512
    std::memcpy(params + 0x14, &depth, 4);                            // queue_depth = 7
    std::memcpy(params + 0x18, "\x02\x00\x00\x00", 4);                 // buffer_mode = 2
    ASSERT_EQ(open(0, addr(params), addr(&id), 0, 0, 0), 0u);
    ASSERT_NE(id, kPortInvalid);

    uint32_t qlevel = 0xDEADu, qavail = 0xDEADu;
    ASSERT_EQ(level(id, addr(&qlevel), addr(&qavail), 0, 0, 0), 0u);
    EXPECT_EQ(qlevel, 0u) << "the headless queue is always empty";
    EXPECT_EQ(qavail, 7u) << "available reports the port's own depth: the push-pacing divisor";
    EXPECT_EQ(level(id, 0, 0, 0, 0, 0), kInvalidParam) << "both null out-pointers rejected";
    EXPECT_EQ(push(id, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(advance(id, 0, 0, 0, 0, 0), 0u);
    uint8_t attr[16]{};
    EXPECT_EQ(setattr(id, 3, addr(attr), sizeof attr, 0, 0), 0u);
    EXPECT_EQ(setattr(id, 3, 0, sizeof attr, 0, 0), kInvalidParam);

    const uint32_t bogus = kPortInvalid;
    EXPECT_EQ(level(bogus, addr(&qlevel), addr(&qavail), 0, 0, 0), kInvalidPort);
    EXPECT_EQ(push(bogus, 0, 0, 0, 0, 0), kInvalidPort);
    EXPECT_EQ(advance(bogus, 0, 0, 0, 0, 0), kInvalidPort);
    EXPECT_EQ(setattr(bogus, 3, addr(attr), sizeof attr, 0, 0), kInvalidPort);
}

TEST(Audio3d, PortsExhaust) {
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("sceAudio3dInitialize"));
    HleFn open = Hle::lookup(nid_hash("sceAudio3dPortOpen"));
    HleFn defaults = Hle::lookup(nid_hash("sceAudio3dGetDefaultOpenParameters"));
    ASSERT_NE(init, nullptr);
    ASSERT_NE(open, nullptr);
    ASSERT_NE(defaults, nullptr);
    ASSERT_EQ(init(0, 0, 0, 0, 0, 0), 0u);
    uint8_t params[0x20]{};
    ASSERT_EQ(defaults(addr(params), 0, 0, 0, 0, 0), 0u);
    // Earlier TESTs may already hold ports; opening until refusal must end in OUT_OF_RESOURCES
    // with every granted id inside 0..3 — never a silent success past the table.
    bool saw_exhaustion = false;
    for (int i = 0; i < 6; i++) {
        uint32_t id = kPortInvalid;
        const uint64_t r = open(0, addr(params), addr(&id), 0, 0, 0);
        if (r == kNoResources) {
            saw_exhaustion = true;
            EXPECT_EQ(id, kPortInvalid);
            break;
        }
        ASSERT_EQ(r, 0u);
        EXPECT_LE(id, 3u);
    }
    EXPECT_TRUE(saw_exhaustion) << "the 4-port table must eventually refuse";
}

TEST(Acm, ContextLifecycle) {
    register_builtin_hle();
    HleFn create = Hle::lookup(nid_hash("sceAcmContextCreate"));
    HleFn destroy = Hle::lookup(nid_hash("sceAcmContextDestroy"));
    ASSERT_NE(create, nullptr);
    ASSERT_NE(destroy, nullptr);
    EXPECT_TRUE(is_error(create(0, 0, 0, 0, 0, 0))) << "null context pointer fails, writes nothing";
    uint32_t ctx = 0;
    ASSERT_EQ(create(addr(&ctx), 0, 0, 0, 0, 0), 0u);
    EXPECT_NE(ctx, 0u) << "the guest gets a valid local context id";
    EXPECT_TRUE(is_error(destroy(99, 0, 0, 0, 0, 0))) << "foreign context id fails";
    ASSERT_EQ(destroy(ctx, 0, 0, 0, 0, 0), 0u);
    EXPECT_TRUE(is_error(destroy(ctx, 0, 0, 0, 0, 0))) << "destroy frees exactly once";
}

TEST(Acm, BatchUnavailable) {
    register_builtin_hle();
    HleFn create = Hle::lookup(nid_hash("sceAcmContextCreate"));
    HleFn start1 = Hle::lookup(nid_hash("sceAcmBatchStartBuffer"));
    HleFn startn = Hle::lookup(nid_hash("sceAcmBatchStartBuffers"));
    HleFn wait = Hle::lookup(nid_hash("sceAcmBatchWait"));
    for (HleFn f : {create, start1, startn, wait}) ASSERT_NE(f, nullptr);
    uint32_t ctx = 0;
    ASSERT_EQ(create(addr(&ctx), 0, 0, 0, 0, 0), 0u);

    // prosper has no batch engine: every batch call fails WITHOUT touching the sideband
    // out-parameters, which the guest would otherwise read as a completed batch.
    uint8_t sideband[64];
    std::memset(sideband, 0xAA, sizeof sideband);
    uint8_t batch[8];
    std::memset(batch, 0xBB, sizeof batch);
    EXPECT_TRUE(is_error(start1(ctx, 0, 0, addr(sideband), addr(batch), 0)));
    EXPECT_EQ(sideband[0], 0xAA) << "batch_error untouched";
    EXPECT_EQ(batch[0], 0xBB) << "batch id untouched";
    std::memset(sideband, 0xAA, sizeof sideband);
    EXPECT_TRUE(is_error(startn(ctx, 0, 0, addr(sideband), addr(batch), 0)));
    EXPECT_EQ(sideband[0], 0xAA);
    EXPECT_EQ(batch[0], 0xBB);
    EXPECT_TRUE(is_error(wait(ctx, 1, 0, 0, 0, 0)));
}

TEST(Propagation, SubsystemUnavailable) {
    register_builtin_hle();
    HleFn create = Hle::lookup(nid_hash("sceAudioPropagationSystemCreate"));
    HleFn query = Hle::lookup(nid_hash("sceAudioPropagationSystemQueryMemory"));
    HleFn room = Hle::lookup(nid_hash("sceAudioPropagationRoomCreate"));
    HleFn material = Hle::lookup(nid_hash("sceAudioPropagationSystemRegisterMaterial"));
    HleFn attrs = Hle::lookup(nid_hash("sceAudioPropagationSystemSetAttributes"));
    HleFn rays = Hle::lookup(nid_hash("sceAudioPropagationSystemGetRays"));
    for (HleFn f : {create, query, room, material, attrs, rays}) ASSERT_NE(f, nullptr);

    // No ray tracer in prosper: everything fails, every out-parameter keeps its sentinel.
    // In particular QueryMemory must NOT answer a zero size — the guest would allocate a
    // zero-byte system block and overrun it (the AudioOut2 speaker-array lesson).
    uint8_t out[64];
    std::memset(out, 0xAA, sizeof out);
    EXPECT_TRUE(is_error(query(0, addr(out), 0, 0, 0, 0)));
    EXPECT_EQ(out[0], 0xAA);
    std::memset(out, 0xAA, sizeof out);
    EXPECT_TRUE(is_error(create(0, addr(out), addr(out + 32), 0, 0, 0)));
    EXPECT_EQ(out[0], 0xAA);
    EXPECT_EQ(out[32], 0xAA);
    std::memset(out, 0xAA, sizeof out);
    EXPECT_TRUE(is_error(room(1, addr(out), 0, 0, 0, 0)));
    EXPECT_EQ(out[0], 0xAA);
    std::memset(out, 0xAA, sizeof out);
    EXPECT_TRUE(is_error(material(1, 0, addr(out), 0, 0, 0)));
    EXPECT_EQ(out[0], 0xAA);
    EXPECT_TRUE(is_error(attrs(1, 0, 0, 0, 0, 0)));
    std::memset(out, 0xAA, sizeof out);
    uint32_t n = 0xDEADu;
    EXPECT_TRUE(is_error(rays(1, addr(out), addr(&n), 0, 0, 0)));
    EXPECT_EQ(out[0], 0xAA);
    EXPECT_EQ(n, 0xDEADu) << "ray count untouched: the guest must not divide by it";
}
