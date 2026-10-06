// test_ngs2_lifecycle — NGS2 lifecycle contracts as libSceNgs2.native.sprx answers them.
//
// PS5 titles link the .native module (every NGS2 importer in the local corpus imports NIDs only it
// exports); the plain libSceNgs2.sprx is the PS4-compatibility build and answers a different
// 0x804a0xxx error family. Every expected code below is the native module's, and each arm names the
// native check it pins. Handles validate, double-free fails, and destroying a system reclaims its
// racks.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper;

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }
static constexpr uint64_t sx(uint32_t v) { return (uint64_t)(int64_t)(int32_t)v; }

static constexpr uint64_t kInvalidOut = sx(0x804a8010u);
static constexpr uint64_t kOutTooSmall = sx(0x804a8011u);
static constexpr uint64_t kOptionSize = sx(0x804a8013u);
static constexpr uint64_t kInvalidBufferInfo = sx(0x804a8020u);
static constexpr uint64_t kInvalidAllocator = sx(0x804a8023u);
static constexpr uint64_t kWaveformType = sx(0x804a8050u);
static constexpr uint64_t kSampleRate = sx(0x804a8051u);
static constexpr uint64_t kNumChannels = sx(0x804a8052u);
static constexpr uint64_t kInvalidSystem = sx(0x804a8201u);
static constexpr uint64_t kInvalidRack = sx(0x804a8202u);
static constexpr uint64_t kInvalidVoice = sx(0x804a8203u);
static constexpr uint64_t kRenderBuffer = sx(0x804a8251u);
static constexpr uint64_t kInvalidRackId = sx(0x804a8300u);
static constexpr uint64_t kVoiceIndex = sx(0x804a8305u);
static constexpr uint64_t kRenderCount = sx(0x804a830bu);
static constexpr uint64_t kInvalidGrain = sx(0x804a8310u);
static constexpr uint64_t kInvalidMaxGrain = sx(0x804a8311u);
static constexpr uint64_t kNullParamList = sx(0x804a8360u);

// SceNgs2SystemOption as the native core reads it (0x1e7e0): size 0x90, maxGrainSamples @0x6c,
// numGrainSamples @0x70, sampleRate @0x74, numChannels @0x78.
struct SystemOption {
    uint64_t size = 0x90;
    uint8_t name_and_schedulers[0x60] = {};
    uint32_t flags = 0, max_grain = 512, num_grain = 256, sample_rate = 48000, channels = 8;
    uint8_t rest[0x14] = {};
};
static_assert(sizeof(SystemOption) == 0x90, "SceNgs2SystemOption is 0x90 bytes");
static_assert(offsetof(SystemOption, max_grain) == 0x6c, "maxGrainSamples");
static constexpr uint32_t kSampler = 0x1000;  // a rack id the native table knows

TEST(Ngs2Lifecycle, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceNgs2SystemCreateWithAllocator", "sceNgs2SystemDestroy",
        "sceNgs2SystemSetGrainSamples",     "sceNgs2RackCreateWithAllocator",
        "sceNgs2RackLock",                  "sceNgs2RackUnlock",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 6, "the 6 lifecycle exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(Ngs2Lifecycle, NidsResolveToFirmwareValues) {
    // All six NIDs reproduce the PS5 3.20 firmware export set verbatim.
    EXPECT_EQ(nid_hash("sceNgs2SystemCreateWithAllocator"), "mPYgU4oYpuY");
    EXPECT_EQ(nid_hash("sceNgs2RackCreateWithAllocator"), "U546k6orxQo");
    EXPECT_EQ(nid_hash("sceNgs2SystemDestroy"), "u-WrYDaJA3k");
    EXPECT_EQ(nid_hash("sceNgs2SystemSetGrainSamples"), "l4Q2dWEH6UM");
    EXPECT_EQ(nid_hash("sceNgs2RackLock"), "MzTa7VLjogY");
    EXPECT_EQ(nid_hash("sceNgs2RackUnlock"), "++YZ7P9e87U");
}

TEST(Ngs2Lifecycle, SystemCreateDestroyReclaimsRacks) {
    register_builtin_hle();
    HleFn create = Hle::lookup(nid_hash("sceNgs2SystemCreateWithAllocator"));
    HleFn destroy = Hle::lookup(nid_hash("sceNgs2SystemDestroy"));
    HleFn grain = Hle::lookup(nid_hash("sceNgs2SystemSetGrainSamples"));
    HleFn rack_create = Hle::lookup(nid_hash("sceNgs2RackCreateWithAllocator"));
    HleFn lock = Hle::lookup(nid_hash("sceNgs2RackLock"));
    HleFn unlock = Hle::lookup(nid_hash("sceNgs2RackUnlock"));
    for (HleFn f : {create, destroy, grain, rack_create, lock, unlock}) ASSERT_NE(f, nullptr);

    uint64_t sys = 0;
    // An allocator whose allocHandler (+0) is set; the callbacks are never invoked.
    uint64_t alloc[3] = {0x1000, 0x2000, 0};
    uint64_t no_handler[3] = {0, 0x2000, 0};
    EXPECT_EQ(create(0, 0, 0, 0, 0, 0), kInvalidAllocator)
        << "the allocator is checked before the handle out";
    EXPECT_EQ(create(0, 0, addr(&sys), 0, 0, 0), kInvalidAllocator) << "null allocator";
    EXPECT_EQ(create(0, addr(no_handler), addr(&sys), 0, 0, 0), kInvalidAllocator)
        << "an allocator without an allocHandler";
    EXPECT_EQ(create(0, addr(alloc), 0, 0, 0, 0), kInvalidOut) << "null handle out";
    EXPECT_EQ(sys, 0u) << "a refused create writes no handle";
    ASSERT_EQ(create(0, addr(alloc), addr(&sys), 0, 0, 0), 0u);
    EXPECT_NE(sys, 0u) << "CreateWithAllocator writes a tagged handle";
    // A NULL option means maxGrainSamples 512 (the native default option at 0x54800).
    EXPECT_EQ(grain(sys, 512, 0, 0, 0, 0), 0u);
    EXPECT_EQ(grain(sys, 64, 0, 0, 0, 0), 0u);
    EXPECT_EQ(grain(sys, 100, 0, 0, 0, 0), kInvalidGrain) << "not a multiple of 64";
    EXPECT_EQ(grain(sys, 0, 0, 0, 0, 0), kInvalidGrain) << "below the minimum";
    EXPECT_EQ(grain(sys, 576, 0, 0, 0, 0), kInvalidGrain) << "above this system's maxGrainSamples";
    EXPECT_EQ(grain(0xDEADu, 512, 0, 0, 0, 0), kInvalidSystem) << "foreign system refused";

    uint64_t rack = 0;
    EXPECT_EQ(rack_create(sys, kSampler, 0, 0, addr(&rack), 0), kInvalidAllocator) << "null allocator";
    EXPECT_EQ(rack_create(sys, kSampler, 0, addr(no_handler), addr(&rack), 0), kInvalidAllocator);
    EXPECT_EQ(rack_create(sys, kSampler, 0, addr(alloc), 0, 0), kInvalidOut) << "null handle out";
    EXPECT_EQ(rack_create(sys, 1, 0, addr(alloc), addr(&rack), 0), kInvalidRackId)
        << "an id outside the native rack table";
    EXPECT_EQ(rack_create(0xDEADu, 1, 0, addr(alloc), addr(&rack), 0), kInvalidRackId)
        << "the rack id is checked before the system";
    EXPECT_EQ(rack_create(0xDEADu, kSampler, 0, addr(alloc), addr(&rack), 0), kInvalidSystem);
    EXPECT_EQ(rack, 0u) << "a refused create writes no handle";
    ASSERT_EQ(rack_create(sys, kSampler, 0, addr(alloc), addr(&rack), 0), 0u);
    // A second system with its own rack: destroying the first must leave it alone.
    uint64_t sys2 = 0, rack2 = 0;
    ASSERT_EQ(create(0, addr(alloc), addr(&sys2), 0, 0, 0), 0u);
    ASSERT_EQ(rack_create(sys2, 0x4003, 0, addr(alloc), addr(&rack2), 0), 0u);
    EXPECT_NE(rack, 0u);
    EXPECT_EQ(lock(rack, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(unlock(rack, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(lock(0xDEADu, 0, 0, 0, 0, 0), kInvalidRack) << "foreign rack refused";
    EXPECT_EQ(destroy(0xDEADu, 0, 0, 0, 0, 0), kInvalidSystem);
    uint8_t released[0x40];
    memset(released, 0xAB, sizeof(released));  // nonzero sentinel proves the zero-write
    ASSERT_EQ(destroy(sys, addr(released), 0, 0, 0, 0), 0u);
    for (size_t i = 0; i < sizeof(released); ++i) {
        EXPECT_EQ(released[i], 0u) << "released-context OUT block is zeroed at byte " << i;
    }
    EXPECT_EQ(destroy(sys, 0, 0, 0, 0, 0), kInvalidSystem) << "destroy frees exactly once";
    EXPECT_EQ(lock(rack, 0, 0, 0, 0, 0), kInvalidRack)
        << "destroying the system reclaims its racks";
    EXPECT_EQ(lock(rack2, 0, 0, 0, 0, 0), 0u) << "another system's rack survives the destroy";
    EXPECT_EQ(destroy(sys2, 0, 0, 0, 0, 0), 0u);
}

// The SceNgs2SystemOption checks the native core applies to SystemQueryBufferSize, SystemCreate and
// SystemCreateWithAllocator alike, in the order the core runs them.
TEST(Ngs2Lifecycle, SystemOptionIsValidatedLikeTheNativeCore) {
    register_builtin_hle();
    HleFn query = Hle::lookup(nid_hash("sceNgs2SystemQueryBufferSize"));
    HleFn create = Hle::lookup(nid_hash("sceNgs2SystemCreate"));
    HleFn create_alloc = Hle::lookup(nid_hash("sceNgs2SystemCreateWithAllocator"));
    HleFn destroy = Hle::lookup(nid_hash("sceNgs2SystemDestroy"));
    HleFn grain = Hle::lookup(nid_hash("sceNgs2SystemSetGrainSamples"));
    for (HleFn f : {query, create, create_alloc, destroy, grain}) ASSERT_NE(f, nullptr);

    struct BufferInfo { uint64_t host_buffer, host_buffer_size, reserved[5], user_data; } info{};
    memset(&info, 0xEE, sizeof info);
    EXPECT_EQ(query(0, 0, 0, 0, 0, 0), kInvalidOut);
    ASSERT_EQ(query(0, addr(&info), 0, 0, 0, 0), 0u);
    EXPECT_EQ(info.host_buffer, 0u);
    EXPECT_NE(info.host_buffer_size, 0u);
    EXPECT_EQ(info.reserved[4], 0u);
    EXPECT_EQ(info.user_data, 0xEEEEEEEEEEEEEEEEull) << "userData (+0x38) is the caller's: not cleared";

    auto rejects = [&](const SystemOption& opt, uint64_t want, const char* what) {
        BufferInfo q{};
        EXPECT_EQ(query(addr(&opt), addr(&q), 0, 0, 0, 0), want) << "query: " << what;
        uint64_t alloc[3] = {0x1000, 0x2000, 0};
        uint64_t sys = 0;
        EXPECT_EQ(create_alloc(addr(&opt), addr(alloc), addr(&sys), 0, 0, 0), want) << what;
        EXPECT_EQ(sys, 0u) << what << ": no handle on refusal";
    };
    SystemOption opt;
    opt.size = 0x40;
    rejects(opt, kOptionSize, "size != 0x90");
    opt = {}; opt.max_grain = 4096;
    rejects(opt, kInvalidMaxGrain, "maxGrainSamples above 2048");
    opt = {}; opt.max_grain = 100;
    rejects(opt, kInvalidMaxGrain, "maxGrainSamples not a multiple of 64");
    opt = {}; opt.num_grain = 1024;
    rejects(opt, kInvalidGrain, "numGrainSamples above maxGrainSamples");
    opt = {}; opt.max_grain = 4096; opt.num_grain = 1024;
    rejects(opt, kInvalidMaxGrain, "the max is checked before the grain");
    opt = {}; opt.sample_rate = 44000;
    rejects(opt, kSampleRate, "not one of the ten rates");
    for (uint32_t rate : {11025u, 12000u, 22050u, 24000u, 44100u, 48000u, 88200u, 96000u, 176400u,
                          192000u}) {
        opt = {}; opt.sample_rate = rate;
        BufferInfo q{};
        EXPECT_EQ(query(addr(&opt), addr(&q), 0, 0, 0, 0), 0u) << "rate " << rate << " is accepted";
    }
    opt = {}; opt.channels = 0;
    rejects(opt, kNumChannels, "no channels");
    opt = {}; opt.channels = 38;
    rejects(opt, kNumChannels, "38 channels");

    // A valid option is accepted and its maxGrainSamples bounds later SetGrainSamples calls.
    opt = {}; opt.max_grain = 2048; opt.num_grain = 1024; opt.sample_rate = 44100;
    std::vector<uint8_t> work(0x1000);
    BufferInfo buf{};
    buf.host_buffer = addr(work.data());
    buf.host_buffer_size = work.size();
    uint64_t sys = 0;
    ASSERT_EQ(create(addr(&opt), addr(&buf), addr(&sys), 0, 0, 0), 0u);
    EXPECT_EQ(grain(sys, 2048, 0, 0, 0, 0), 0u) << "within this system's maxGrainSamples";
    EXPECT_EQ(grain(sys, 2112, 0, 0, 0, 0), kInvalidGrain);
    EXPECT_EQ(destroy(sys, 0, 0, 0, 0, 0), 0u);

    // SystemCreate refuses a missing buffer or hostBuffer before the handle out.
    BufferInfo no_host{};
    EXPECT_EQ(create(0, 0, 0, 0, 0, 0), kInvalidBufferInfo);
    EXPECT_EQ(create(0, addr(&no_host), addr(&sys), 0, 0, 0), kInvalidBufferInfo);
    EXPECT_EQ(create(0, addr(&buf), 0, 0, 0, 0), kInvalidOut);
}

// The rack, voice and render handlers: native handle codes, check order, and the refusals that
// replaced success or a plain-module code.
TEST(Ngs2Lifecycle, RackVoiceAndRenderAnswerNativeCodes) {
    register_builtin_hle();
    auto fn = [](const char* name) { return Hle::lookup(nid_hash(name)); };
    HleFn create = fn("sceNgs2SystemCreateWithAllocator"), destroy = fn("sceNgs2SystemDestroy");
    HleFn rack_query = fn("sceNgs2RackQueryBufferSize"), rack_create = fn("sceNgs2RackCreate");
    HleFn rack_destroy = fn("sceNgs2RackDestroy"), get_voice = fn("sceNgs2RackGetVoiceHandle");
    HleFn control = fn("sceNgs2VoiceControl"), run = fn("sceNgs2VoiceRunCommands");
    HleFn state = fn("sceNgs2VoiceGetState"), flags = fn("sceNgs2VoiceGetStateFlags");
    HleFn render = fn("sceNgs2SystemRender");
    for (HleFn f : {create, destroy, rack_query, rack_create, rack_destroy, get_voice, control, run,
                    state, flags, render})
        ASSERT_NE(f, nullptr);

    uint64_t alloc[3] = {0x1000, 0x2000, 0};
    uint64_t sys = 0;
    ASSERT_EQ(create(0, addr(alloc), addr(&sys), 0, 0, 0), 0u);

    uint8_t qinfo[0x40];
    EXPECT_EQ(rack_query(kSampler, 0, 0, 0, 0, 0), kInvalidOut);
    EXPECT_EQ(rack_query(0x1234, 0, addr(qinfo), 0, 0, 0), kInvalidRackId);
    EXPECT_EQ(rack_query(0x4004, 0, addr(qinfo), 0, 0, 0), 0u);

    std::vector<uint8_t> work(0x4000);
    uint64_t buf[8] = {addr(work.data()), work.size()};
    uint64_t no_host[8] = {};
    uint64_t rack = 0;
    EXPECT_EQ(rack_create(sys, kSampler, 0, 0, addr(&rack), 0), kInvalidBufferInfo);
    EXPECT_EQ(rack_create(sys, kSampler, 0, addr(no_host), addr(&rack), 0), kInvalidBufferInfo);
    EXPECT_EQ(rack_create(sys, kSampler, 0, addr(buf), 0, 0), kInvalidOut);
    EXPECT_EQ(rack_create(sys, 0x2002, 0, addr(buf), addr(&rack), 0), kInvalidRackId);
    ASSERT_EQ(rack_create(sys, kSampler, 0, addr(buf), addr(&rack), 0), 0u);

    uint64_t voice = 0;
    EXPECT_EQ(get_voice(0xDEADu, 0, 0, 0, 0, 0), kInvalidRack) << "the rack is resolved before the out";
    EXPECT_EQ(get_voice(rack, 0, 0, 0, 0, 0), kInvalidOut);
    EXPECT_EQ(get_voice(rack, 64, addr(&voice), 0, 0, 0), kVoiceIndex) << "64 default voices";
    ASSERT_EQ(get_voice(rack, 3, addr(&voice), 0, 0, 0), 0u);

    uint8_t st[0x30];
    EXPECT_EQ(state(0xDEADu, addr(st), sizeof st, 0, 0, 0), kInvalidVoice);
    EXPECT_EQ(state(voice, 0, 4, 0, 0, 0), kOutTooSmall) << "the size is checked before the pointer";
    EXPECT_EQ(state(voice, addr(st), 7, 0, 0, 0), kOutTooSmall);
    EXPECT_EQ(state(voice, 0, sizeof st, 0, 0, 0), kInvalidOut);
    EXPECT_EQ(state(voice, addr(st), 8, 0, 0, 0), 0u);
    std::vector<uint8_t> big(0x2000, 0xAB);
    EXPECT_EQ(state(voice, addr(big.data()), big.size(), 0, 0, 0), 0u) << "no upper size bound";
    uint32_t f = 7;
    EXPECT_EQ(flags(0xDEADu, addr(&f), 0, 0, 0, 0), kInvalidVoice);
    EXPECT_EQ(flags(voice, 0, 0, 0, 0, 0), kInvalidOut);
    EXPECT_EQ(control(voice, 0, 0, 0, 0, 0), kNullParamList);
    EXPECT_EQ(control(0xDEADu, 0, 0, 0, 0, 0), kNullParamList) << "the list is checked before the voice";
    uint8_t list[16] = {};
    EXPECT_EQ(control(0xDEADu, addr(list), 0, 0, 0, 0), kInvalidVoice);
    EXPECT_EQ(run(0xDEADu, addr(list), 0, 0, 0, 0), kInvalidVoice);

    struct RenderInfo { uint64_t buffer, size; uint32_t waveform_type, channels; };
    uint8_t pcm[256];
    RenderInfo out{addr(pcm), sizeof pcm, 0x18, 2};
    EXPECT_EQ(render(0xDEADu, addr(&out), 1, 0, 0, 0), kInvalidSystem);
    EXPECT_EQ(render(sys, 0, 0, 0, 0, 0), 0u) << "zero buffers is accepted";
    RenderInfo many[32] = {};
    EXPECT_EQ(render(sys, addr(many), 32, 0, 0, 0), kRenderCount) << "at most 31 buffers";
    RenderInfo bad_type{addr(pcm), sizeof pcm, 0, 2};
    EXPECT_EQ(render(sys, addr(&bad_type), 1, 0, 0, 0), kWaveformType);
    RenderInfo no_buffer{0, sizeof pcm, 0x12, 2};
    EXPECT_EQ(render(sys, addr(&no_buffer), 1, 0, 0, 0), kRenderBuffer);
    for (uint32_t t : {0x12u, 0x13u, 0x18u, 0x19u}) {
        out.waveform_type = t;
        EXPECT_EQ(render(sys, addr(&out), 1, 0, 0, 0), 0u) << "waveform type 0x" << std::hex << t;
    }

    EXPECT_EQ(rack_destroy(rack, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(rack_destroy(rack, 0, 0, 0, 0, 0), kInvalidRack) << "a destroyed rack is not live";
    EXPECT_EQ(destroy(sys, 0, 0, 0, 0, 0), 0u);
}
