// test_voice — libSceVoice reports its subsystem unavailable at every entry point.
//
// prosper has no voice-capture backend (no microphone / network voice path), so sceVoiceInit
// reports voice unavailable and GTA V skips its voice-chat setup instead of #DE'ing on
// never-written voice-parameter divisors (#1158). The 14 port entry points answer what the shipped
// libSceVoice.sprx answers after a failed Init: SCE_VOICE_ERROR_LIBVOICE_NOT_INIT (0x804E0801),
// checked before any argument is read. Every arm drives the real NIDs, passes a sentinel buffer in
// EVERY argument slot (CreatePort's out-parameter is a0, GetPortAttr's a2, ...), and asserts the
// whole buffer is untouched.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

static constexpr uint64_t kVoiceInitUnavailable = (uint64_t)(int64_t)(int32_t)0x80410002u;
static constexpr uint64_t kVoiceNotInit = (uint64_t)(int64_t)(int32_t)0x804E0801u;

namespace {
void expect_refuses_untouched(const char* name, uint64_t expected) {
    HleFn fn = Hle::lookup(nid_hash(name));
    ASSERT_NE(fn, nullptr) << name << " is not registered";
    uint8_t out[64];
    std::memset(out, 0xAA, sizeof out);
    const uint64_t p = (uint64_t)(uintptr_t)out;
    EXPECT_EQ(fn(p, p, p, p, p, p), expected) << name;
    for (uint8_t b : out) ASSERT_EQ(b, 0xAA) << name << " writes nothing on refusal";
}
}  // namespace

TEST(Voice, InitReportsVoiceUnavailable) {
    register_builtin_hle();
    for (const char* name : {"sceVoiceInit", "sceVoiceInitHQ"})
        expect_refuses_untouched(name, kVoiceInitUnavailable);
}

TEST(Voice, PortSurfaceAnswersNotInitialized) {
    register_builtin_hle();
    static const char* table[] = {
        "sceVoiceCreatePort",
        "sceVoiceDeletePort",
        "sceVoiceStart",
        "sceVoiceStop",
        "sceVoiceConnectIPortToOPort",
        "sceVoiceDisconnectIPortFromOPort",
        "sceVoiceGetBitRate",
        "sceVoiceGetPortAttr",
        "sceVoiceGetPortInfo",
        "sceVoiceGetVolume",
        "sceVoiceSetVolume",
        "sceVoiceSetThreadsParams",
        "sceVoiceReadFromOPort",
        "sceVoiceWriteToIPort",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 14, "14 port entry points");
    for (const char* name : table) expect_refuses_untouched(name, kVoiceNotInit);
}

// libSceVoiceQoS sits on top of a voice session, and there is no voice backend: sceVoiceQoSInit must
// refuse exactly as sceVoiceInit does and leave the guest's memory block alone (the real guest passes
// a freshly allocated 0x40000-byte block and tests the result with `test eax,eax`).
TEST(Voice, QoSInitReportsVoiceUnavailableAndTouchesNothing) {
    register_builtin_hle();
    HleFn fn = Hle::lookup(nid_hash("sceVoiceQoSInit"));
    ASSERT_NE(fn, nullptr) << "sceVoiceQoSInit is not registered";
    uint8_t block[64];
    std::memset(block, 0xAA, sizeof block);
    const uint64_t r = fn((uint64_t)(uintptr_t)block, sizeof block, 0x20000000u, 0, 0, 0);
    EXPECT_EQ(r, kVoiceInitUnavailable);
    EXPECT_LT((int64_t)r, 0) << "negative as int64 too, not a zero-extended error";
    for (uint8_t b : block) ASSERT_EQ(b, 0xAA);
    EXPECT_EQ(fn(0, 0, 0, 0, 0, 0), kVoiceInitUnavailable)
        << "a null block is refused the same way";
}
