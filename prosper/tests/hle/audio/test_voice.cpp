// test_voice — libSceVoice reports its subsystem unavailable at every entry point.
//
// prosper has no voice-capture backend (no microphone / network voice path), so sceVoiceInit
// reports voice unavailable and GTA V skips its voice-chat setup instead of #DE'ing on
// never-written voice-parameter divisors (#1158). The 14 port entry points below must agree
// with that answer: handing out working port handles after Init reported voice off would
// contradict it, and any title that proceeds past a failed Init anyway must land on error
// branches rather than dividing by unwritten parameters. Every arm drives the real NIDs and
// asserts the negative answer with caller buffers left untouched.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

static constexpr uint64_t kVoiceUnavailable = (uint64_t)(int64_t)(int32_t)0x80410002u;

TEST(Voice, UnavailableEverywhere) {
    register_builtin_hle();
    static const char* table[] = {
        "sceVoiceInit",
        "sceVoiceInitHQ",
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
    static_assert(sizeof(table) / sizeof(table[0]) == 16, "init pair + 14 port entry points");
    for (const char* name : table) {
        HleFn fn = Hle::lookup(nid_hash(name));
        ASSERT_NE(fn, nullptr) << name << " is not registered";
        uint8_t out[64];
        std::memset(out, 0xAA, sizeof out);
        EXPECT_EQ(fn(1, (uint64_t)(uintptr_t)out, 0, 0, 0, 0), kVoiceUnavailable) << name;
        EXPECT_EQ(out[0], 0xAA) << name << " writes nothing on refusal";
    }
    EXPECT_EQ(Hle::lookup("AAAAAAAAAAA"), nullptr)
        << "positive control: the lookup rejects an unknown NID";
}
