// test_audio_spatial — libSceAudio3d / libSceAcm contracts, read from the PS5 system modules.
//
// Both libraries were unregistered, so the dispatcher answered `0` — SCE_OK — while writing
// nothing. Every TEST below asserts what was WRITTEN on success, that refused calls write NOTHING,
// and the exact error each refusal returns.
//
// Audio3d has a single global port, so each TEST starts from `reset_audio3d()` (close the port if
// open, then terminate) and leaves the port closed; the TESTs are independent of run order.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

using HleFn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }

static constexpr uint64_t sx(uint32_t v) { return (uint64_t)(int64_t)(int32_t)v; }
static constexpr uint64_t kInvalidPort = sx(0x80EA0002u);
static constexpr uint64_t kInvalidParam = sx(0x80EA0004u);
static constexpr uint64_t kNoResources = sx(0x80EA0006u);
static constexpr uint64_t kNotReady = sx(0x80EA0007u);
static constexpr uint64_t kUnsupportedMode = sx(0x80EA0008u);
static constexpr uint64_t kAcmGeneric = sx(0x81940001u);
static constexpr uint64_t kAcmBadContext = sx(0x81940002u);
static constexpr uint64_t kAcmNullPointer = sx(0x81940006u);
static constexpr uint32_t kUser = 0xFFu;
static constexpr uint32_t kSentinel = 0x12345678u;

namespace {

struct Audio3d {
    HleFn init, terminate, defaults, open, close, setattr, level, advance, push;
    Audio3d() {
        register_builtin_hle();
        init = Hle::lookup(nid_hash("sceAudio3dInitialize"));
        terminate = Hle::lookup(nid_hash("sceAudio3dTerminate"));
        defaults = Hle::lookup(nid_hash("sceAudio3dGetDefaultOpenParameters"));
        open = Hle::lookup(nid_hash("sceAudio3dPortOpen"));
        close = Hle::lookup(nid_hash("sceAudio3dPortClose"));
        setattr = Hle::lookup(nid_hash("sceAudio3dPortSetAttribute"));
        level = Hle::lookup(nid_hash("sceAudio3dPortGetQueueLevel"));
        advance = Hle::lookup(nid_hash("sceAudio3dPortAdvance"));
        push = Hle::lookup(nid_hash("sceAudio3dPortPush"));
    }
    bool bound() const {
        for (HleFn f : {init, terminate, defaults, open, close, setattr, level, advance, push})
            if (!f) return false;
        return true;
    }
    // Back to "not initialised, port closed" whatever an earlier TEST left behind.
    void reset() const {
        close(0, 0, 0, 0, 0, 0);
        terminate(0, 0, 0, 0, 0, 0);
    }
};

// An open-parameter block in the module's normalised 0x28 layout.
struct Params {
    uint8_t raw[0x28]{};
    Params(uint64_t size_this, uint32_t buffer_mode = 2, uint32_t queue_depth = 2,
           uint32_t granularity = 0x100, uint32_t rate = 0, uint32_t max_objects = 0x200,
           uint32_t beds = 2) {
        std::memcpy(raw + 0x00, &size_this, 8);
        std::memcpy(raw + 0x08, &granularity, 4);
        std::memcpy(raw + 0x0c, &rate, 4);
        std::memcpy(raw + 0x10, &max_objects, 4);
        std::memcpy(raw + 0x14, &queue_depth, 4);
        std::memcpy(raw + 0x18, &buffer_mode, 4);
        std::memcpy(raw + 0x20, &beds, 4);
    }
    uint64_t ptr() const { return addr(raw); }
};

}  // namespace

TEST(AudioSpatial, RegisteredNidsBound) {
    register_builtin_hle();
    // The 14 functions this file registers (the three libraries export 31/10/28 in 3.20; the rest
    // stay unregistered). libSceAudioPropagation is deliberately absent.
    static const char* registered[] = {
        "sceAudio3dInitialize", "sceAudio3dTerminate", "sceAudio3dGetDefaultOpenParameters",
        "sceAudio3dPortOpen", "sceAudio3dPortClose", "sceAudio3dPortSetAttribute",
        "sceAudio3dPortGetQueueLevel", "sceAudio3dPortAdvance", "sceAudio3dPortPush",
        "sceAcmContextCreate", "sceAcmContextDestroy", "sceAcmBatchStartBuffer",
        "sceAcmBatchStartBuffers", "sceAcmBatchWait",
    };
    for (const char* name : registered)
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    EXPECT_EQ(Hle::lookup(nid_hash("sceAudioPropagationSystemQueryMemory")), nullptr)
        << "AudioPropagation must stay unregistered: its one importer asserts on a failure";
}

TEST(AudioSpatial, NidsMatchFirmwareValues) {
    // Known-answer NIDs from the PS5 3.20 export list, plus the libSceAcm orphan seen in a live
    // boot that nid_hash("sceAcmContextCreate") reproduces.
    EXPECT_EQ(nid_hash("sceAudio3dInitialize"), "UmCvjSmuZIw");
    EXPECT_EQ(nid_hash("sceAudio3dGetDefaultOpenParameters"), "Im+jOoa5WAI");
    EXPECT_EQ(nid_hash("sceAudio3dPortOpen"), "XeDDK0xJWQA");
    EXPECT_EQ(nid_hash("sceAudio3dPortSetAttribute"), "Yq9bfUQ0uJg");
    EXPECT_EQ(nid_hash("sceAudio3dPortGetQueueLevel"), "YaaDbDwKpFM");
    EXPECT_EQ(nid_hash("sceAudio3dPortAdvance"), "lw0qrdSjZt8");
    EXPECT_EQ(nid_hash("sceAudio3dPortPush"), "VEVhZ9qd4ZY");
    EXPECT_EQ(nid_hash("sceAcmContextCreate"), "ZIXln2K3XMk");
}

TEST(Audio3d, PortOpenRequiresInitAndWritesNothing) {
    Audio3d a;
    ASSERT_TRUE(a.bound());
    a.reset();
    Params p(0x20);
    uint32_t id = kSentinel;
    EXPECT_EQ(a.open(kUser, p.ptr(), addr(&id), 0, 0, 0), kNotReady);
    EXPECT_EQ(id, kSentinel) << "the module leaves *port untouched on NOT_READY";
}

TEST(Audio3d, InitializeAndTerminate) {
    Audio3d a;
    ASSERT_TRUE(a.bound());
    a.reset();
    EXPECT_EQ(a.terminate(0, 0, 0, 0, 0, 0), kNotReady) << "terminate before init";
    EXPECT_EQ(a.init(1, 0, 0, 0, 0, 0), kInvalidParam) << "nonzero reserved is rejected";
    EXPECT_EQ(a.init(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(a.init(0, 0, 0, 0, 0, 0), 0u) << "a repeat Initialize is answered 0 (see source)";

    Params p(0x20);
    uint32_t id = kSentinel;
    ASSERT_EQ(a.open(kUser, p.ptr(), addr(&id), 0, 0, 0), 0u);
    EXPECT_EQ(a.terminate(0, 0, 0, 0, 0, 0), kNotReady) << "terminate refuses while the port is open";
    ASSERT_EQ(a.close(id, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(a.terminate(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(a.open(kUser, p.ptr(), addr(&id), 0, 0, 0), kNotReady) << "terminated again";
}

TEST(Audio3d, DefaultParameters) {
    Audio3d a;
    ASSERT_TRUE(a.bound());
    EXPECT_EQ(a.defaults(0, 0, 0, 0, 0, 0), 0u) << "void in the module: a null pointer is a no-op";
    uint8_t params[0x28];
    std::memset(params, 0xAA, sizeof params);
    a.defaults(addr(params), 0, 0, 0, 0, 0);
    uint64_t size_this = 0;
    uint32_t w[5]{};
    std::memcpy(&size_this, params + 0x00, 8);
    std::memcpy(w, params + 0x08, sizeof w);
    EXPECT_EQ(size_this, 0x20u);
    EXPECT_EQ(w[0], 0x100u) << "granularity";
    EXPECT_EQ(w[1], 0u) << "rate 0 = 48 kHz";
    EXPECT_EQ(w[2], 0x200u) << "max_objects";
    EXPECT_EQ(w[3], 2u) << "queue_depth";
    EXPECT_EQ(w[4], 2u) << "buffer_mode advance-and-push";
    EXPECT_EQ(params[0x1c], 0xAA) << "the module never writes the pad at 0x1c";
    EXPECT_EQ(params[0x20], 0xAA) << "nothing past the 0x20 struct";
}

TEST(Audio3d, PortOpenValidationWritesNothing) {
    Audio3d a;
    ASSERT_TRUE(a.bound());
    a.reset();
    ASSERT_EQ(a.init(0, 0, 0, 0, 0, 0), 0u);
    uint32_t id = kSentinel;
    auto refused = [&](const Params& p, uint32_t user, const char* why) {
        id = kSentinel;
        EXPECT_EQ(a.open(user, p.ptr(), addr(&id), 0, 0, 0), kInvalidParam) << why;
        EXPECT_EQ(id, kSentinel) << why << ": *port must stay untouched";
    };
    refused(Params(0x20), 0, "user id must be 0xFF");
    refused(Params(0x21), kUser, "size 0x21 is not one of the four exact sizes");
    refused(Params(0x30), kUser, "unknown struct version");
    refused(Params(0x20, 2, 2, 0x100, 1), kUser, "non-48kHz rate");
    refused(Params(0x20, 2, 2, 0x80), kUser, "granularity under 0x100");
    refused(Params(0x20, 2, 2, 0x180), kUser, "unaligned granularity");
    refused(Params(0x20, 2, 2, 0x100, 0, 0), kUser, "zero objects");
    refused(Params(0x20, 2, 0), kUser, "zero queue depth");
    refused(Params(0x20, 3), kUser, "unknown buffer mode");
    refused(Params(0x28, 2, 2, 0x100, 0, 0x200, 4), kUser, "beds must be 2 or 3");
    EXPECT_EQ(a.open(kUser, 0, addr(&id), 0, 0, 0), kInvalidParam) << "null params";
    EXPECT_EQ(a.open(kUser, Params(0x20).ptr(), 0, 0, 0, 0), kInvalidParam) << "null port pointer";

    // Every exact size the module accepts opens port 0; a deep queue is not capped.
    for (uint64_t size : {0x10u, 0x18u, 0x20u, 0x28u}) {
        id = kSentinel;
        ASSERT_EQ(a.open(kUser, Params(size, 2, 0x100).ptr(), addr(&id), 0, 0, 0), 0u) << size;
        EXPECT_EQ(id, 0u) << "the only port id is 0";
        ASSERT_EQ(a.close(id, 0, 0, 0, 0, 0), 0u);
    }
    a.reset();
}

TEST(Audio3d, SinglePortCloseAndReopen) {
    Audio3d a;
    ASSERT_TRUE(a.bound());
    a.reset();
    ASSERT_EQ(a.init(0, 0, 0, 0, 0, 0), 0u);
    Params p(0x28);
    uint32_t id = kSentinel;
    ASSERT_EQ(a.open(kUser, p.ptr(), addr(&id), 0, 0, 0), 0u);
    EXPECT_EQ(id, 0u);

    uint32_t second = kSentinel;
    EXPECT_EQ(a.open(kUser, p.ptr(), addr(&second), 0, 0, 0), kNoResources)
        << "one port: a second open while it is held is OUT_OF_RESOURCES";
    EXPECT_EQ(second, kSentinel);
    EXPECT_EQ(a.close(1, 0, 0, 0, 0, 0), kInvalidPort) << "only id 0 maps to the port";

    // BALAN's teardown: close, then reopen through its refcount — repeatedly. Every reopen must
    // get port 0 back, never run out.
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(a.close(id, 0, 0, 0, 0, 0), 0u) << "cycle " << i;
        EXPECT_EQ(a.close(id, 0, 0, 0, 0, 0), kInvalidPort) << "closing twice";
        id = kSentinel;
        ASSERT_EQ(a.open(kUser, p.ptr(), addr(&id), 0, 0, 0), 0u) << "reopen " << i;
        EXPECT_EQ(id, 0u);
    }
    ASSERT_EQ(a.close(id, 0, 0, 0, 0, 0), 0u);
    a.reset();
}

TEST(Audio3d, QueueLevelPushAdvance) {
    Audio3d a;
    ASSERT_TRUE(a.bound());
    a.reset();
    ASSERT_EQ(a.init(0, 0, 0, 0, 0, 0), 0u);
    uint32_t id = kSentinel;
    // A non-default depth proves GetQueueLevel reports the port's own depth, not a constant.
    ASSERT_EQ(a.open(kUser, Params(0x20, 2, 7).ptr(), addr(&id), 0, 0, 0), 0u);

    uint32_t qlevel = 0xDEADu, qavail = 0xDEADu;
    ASSERT_EQ(a.level(id, addr(&qlevel), addr(&qavail), 0, 0, 0), 0u);
    EXPECT_EQ(qlevel, 0u) << "the headless queue is always empty";
    EXPECT_EQ(qavail, 7u) << "available reports the port's own depth";
    qavail = 0xDEADu;
    EXPECT_EQ(a.level(id, 0, addr(&qavail), 0, 0, 0), 0u) << "a null level pointer is valid";
    EXPECT_EQ(qavail, 7u);
    EXPECT_EQ(a.level(id, 0, 0, 0, 0, 0), kInvalidParam);
    EXPECT_EQ(a.push(id, 1, 0, 0, 0, 0), 0u);
    EXPECT_EQ(a.advance(id, 0, 0, 0, 0, 0), 0u);
    uint8_t attr[16]{};
    EXPECT_EQ(a.setattr(id, 3, addr(attr), sizeof attr, 0, 0), 0u);
    EXPECT_EQ(a.setattr(id, 3, 0, sizeof attr, 0, 0), kInvalidParam);

    EXPECT_EQ(a.level(1, addr(&qlevel), addr(&qavail), 0, 0, 0), kInvalidPort);
    EXPECT_EQ(a.push(1, 0, 0, 0, 0, 0), kInvalidPort);
    EXPECT_EQ(a.advance(1, 0, 0, 0, 0, 0), kInvalidPort);
    EXPECT_EQ(a.setattr(1, 3, addr(attr), sizeof attr, 0, 0), kInvalidPort);
    ASSERT_EQ(a.close(id, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(a.push(id, 0, 0, 0, 0, 0), kInvalidPort) << "a closed port no longer exists";
    a.reset();
}

TEST(Audio3d, UnsupportedBufferModes) {
    Audio3d a;
    ASSERT_TRUE(a.bound());
    a.reset();
    ASSERT_EQ(a.init(0, 0, 0, 0, 0, 0), 0u);
    uint32_t id = kSentinel, avail = 0;
    // Mode 1: Push and GetQueueLevel refuse, Advance accepts.
    ASSERT_EQ(a.open(kUser, Params(0x20, 1).ptr(), addr(&id), 0, 0, 0), 0u);
    EXPECT_EQ(a.push(id, 0, 0, 0, 0, 0), kUnsupportedMode);
    EXPECT_EQ(a.level(id, 0, addr(&avail), 0, 0, 0), kUnsupportedMode);
    EXPECT_EQ(a.advance(id, 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(a.close(id, 0, 0, 0, 0, 0), 0u);
    // Mode 0 (also the 0x10 default): all three refuse.
    ASSERT_EQ(a.open(kUser, Params(0x10).ptr(), addr(&id), 0, 0, 0), 0u);
    EXPECT_EQ(a.push(id, 0, 0, 0, 0, 0), kUnsupportedMode);
    EXPECT_EQ(a.level(id, 0, addr(&avail), 0, 0, 0), kUnsupportedMode);
    EXPECT_EQ(a.advance(id, 0, 0, 0, 0, 0), kUnsupportedMode);
    ASSERT_EQ(a.close(id, 0, 0, 0, 0, 0), 0u);
    a.reset();
}

TEST(Acm, ContextLifecycle) {
    register_builtin_hle();
    HleFn create = Hle::lookup(nid_hash("sceAcmContextCreate"));
    HleFn destroy = Hle::lookup(nid_hash("sceAcmContextDestroy"));
    ASSERT_NE(create, nullptr);
    ASSERT_NE(destroy, nullptr);
    EXPECT_EQ(create(0, 0, 0, 0, 0, 0), kAcmNullPointer);
    uint32_t ctx = 0;
    ASSERT_EQ(create(addr(&ctx), 0, 0, 0, 0, 0), 0u);
    EXPECT_NE(ctx, 0u);
    EXPECT_NE(ctx, 0xFFFFFFFFu) << "a live context is not the -1 the module presets";
    EXPECT_EQ(destroy(99, 0, 0, 0, 0, 0), kAcmBadContext) << "foreign context id";
    ASSERT_EQ(destroy(ctx, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(destroy(ctx, 0, 0, 0, 0, 0), kAcmBadContext) << "destroy frees exactly once";
}

TEST(Acm, BatchUnavailableWritesNothing) {
    register_builtin_hle();
    HleFn create = Hle::lookup(nid_hash("sceAcmContextCreate"));
    HleFn destroy = Hle::lookup(nid_hash("sceAcmContextDestroy"));
    HleFn start1 = Hle::lookup(nid_hash("sceAcmBatchStartBuffer"));
    HleFn startn = Hle::lookup(nid_hash("sceAcmBatchStartBuffers"));
    HleFn wait = Hle::lookup(nid_hash("sceAcmBatchWait"));
    for (HleFn f : {create, destroy, start1, startn, wait}) ASSERT_NE(f, nullptr);
    uint32_t ctx = 0;
    ASSERT_EQ(create(addr(&ctx), 0, 0, 0, 0, 0), 0u);

    // No batch engine: the catch-all failure, and the batch id (written by the module only on
    // success) keeps its preset.
    uint32_t batch = kSentinel;
    uint8_t list[16]{};
    EXPECT_EQ(startn(ctx, 1, addr(list), 0, addr(&batch), 0), kAcmGeneric);
    EXPECT_EQ(batch, kSentinel);
    EXPECT_EQ(start1(ctx, addr(list), 0, 0, addr(&batch), 0), kAcmGeneric);
    EXPECT_EQ(batch, kSentinel);
    EXPECT_EQ(wait(ctx, 1, 0, 0, 0, 0), kAcmGeneric);
    // An unknown context is BAD_CONTEXT on every batch call.
    EXPECT_EQ(startn(99, 1, addr(list), 0, addr(&batch), 0), kAcmBadContext);
    EXPECT_EQ(start1(99, addr(list), 0, 0, addr(&batch), 0), kAcmBadContext);
    EXPECT_EQ(wait(99, 1, 0, 0, 0, 0), kAcmBadContext);
    ASSERT_EQ(destroy(ctx, 0, 0, 0, 0, 0), 0u);
}
