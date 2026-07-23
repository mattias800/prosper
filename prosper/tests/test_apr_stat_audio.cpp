// test_apr_stat_audio — regression guards for two of the three net-new GTA V (PPSA04263, RAGE) HLE
// behaviors added by #1139 (issue #1142 follow-up). The third, the AGC GetSize cluster, is already
// covered by test_agc_getsize (#1208), so this file covers the remaining two:
//
//   1. sceKernelAprGetFileStat (hle_file.cpp) — #1133. Stubbing it to success WITHOUT filling the
//      output made RAGE convert an uninitialized SceKernelStat as a FILETIME and abort streaming init
//      with a fatal (int 0x41). The handler must host-stat the id's file and write the real fields.
//      Guarded: valid id -> 0 with size@0x48 / mtime@0x28 populated from the host stat; unknown id ->
//      ENOENT with the caller's buffer left UNTOUCHED; null out-ptr -> EINVAL.
//   2. sceAudioOut2GetSpeakerArrayMemorySize (hle_audio.cpp) — #1134. It RETURNS the allocation size
//      in rax; the guest allocates exactly that many bytes. Stubbed to 0 the guest made a zero-byte
//      speaker buffer and overran it (the deterministic pre-render crash). Guarded: returns the fixed
//      non-zero, 0x10-aligned size so the allocation is large enough.
//
// Both handlers are `static` (file-local), so they're reached the way the loader reaches them: by NID
// through the dispatch registry. P() is identity in the HLE layer, so a host pointer passed as the
// guest stat buffer resolves to itself.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/stat.h>

#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"

namespace prosper {
    uint32_t prosper_apr_register(const std::string& path, uint64_t size);
    void     prosper_apr_reset_for_test();
}
using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else          { std::printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    std::printf("== test_apr_stat_audio (#1142: AprGetFileStat + AudioOut2 speaker-array) ==\n");
    register_file_hle();
    register_audio_hle();

    // ---- 1. sceKernelAprGetFileStat ------------------------------------------------------------
    HleFn stat_fn = Hle::lookup(nid_hash("sceKernelAprGetFileStat"));
    CHECK(stat_fn != nullptr, "sceKernelAprGetFileStat registered");

    // A fixture file with a known, non-trivial size that the handler will host-stat.
    const char* fixture = "test_apr_stat_fixture.bin";
    {
        FILE* f = std::fopen(fixture, "wb");
        CHECK(f != nullptr, "create APR stat fixture file");
        if (f) { const char payload[1234] = {0}; std::fwrite(payload, 1, sizeof payload, f); std::fclose(f); }
    }
    struct stat host_st{};
    const bool have_host = (::stat(fixture, &host_st) == 0);
    CHECK(have_host, "host-stat the fixture");

    prosper_apr_reset_for_test();
    uint32_t id = prosper_apr_register(fixture, have_host ? (uint64_t)host_st.st_size : 0);
    CHECK(id == 1, "fixture registered with 1-based id");

    if (stat_fn && have_host) {
        // Valid id: success, and the output is the REAL stat (a non-filling stub would leave the
        // pre-poisoned buffer, failing the size check).
        uint8_t buf[0x78];
        std::memset(buf, 0xAB, sizeof buf);
        uint64_t r = stat_fn(id, (uint64_t)(uintptr_t)buf, 0, 0, 0, 0);
        CHECK(r == 0, "valid id -> 0 (success)");
        CHECK(*(int64_t*)(buf + 0x48) == (int64_t)host_st.st_size,
              "size @0x48 == host file size (buffer filled, not stubbed)");
        CHECK(*(int64_t*)(buf + 0x28) != 0, "mtime @0x28 populated (non-zero)");
#ifndef _WIN32
        CHECK(*(int64_t*)(buf + 0x28) == (int64_t)host_st.st_mtime, "mtime @0x28 == host mtime");
#endif

        // Unknown id: ENOENT, and the caller's buffer must be left UNTOUCHED.
        uint8_t buf2[0x78];
        std::memset(buf2, 0xCD, sizeof buf2);
        uint64_t r2 = stat_fn(0x7fffffffu, (uint64_t)(uintptr_t)buf2, 0, 0, 0, 0);
        CHECK(r2 == 0x80020002ull, "unknown id -> ENOENT (0x80020002)");
        bool untouched = true;
        for (unsigned i = 0; i < sizeof buf2; i++) if (buf2[i] != 0xCD) untouched = false;
        CHECK(untouched, "unknown id leaves the out buffer untouched");

        // Null out pointer: EINVAL.
        uint64_t r3 = stat_fn(id, 0, 0, 0, 0, 0);
        CHECK(r3 == 0x80020016ull, "null out ptr -> EINVAL (0x80020016)");
    }
    std::remove(fixture);

    // ---- 2. sceAudioOut2GetSpeakerArrayMemorySize ----------------------------------------------
    HleFn spk = Hle::lookup(nid_hash("sceAudioOut2GetSpeakerArrayMemorySize"));
    CHECK(spk != nullptr, "sceAudioOut2GetSpeakerArrayMemorySize registered");
    if (spk) {
        uint64_t sz = spk(8, 0, 0, 0, 0, 0);   // rdi=8 (speaker config) per the live RAGE disassembly
        CHECK(sz != 0, "speaker-array size is non-zero (guards the zero-alloc overrun crash)");
        CHECK((sz & 0xfull) == 0, "speaker-array size is 0x10-aligned (usable as an allocation size)");
        CHECK(sz == 0x100000ull, "speaker-array size == 1 MiB (0x100000), the null-backend contract");
    }

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
