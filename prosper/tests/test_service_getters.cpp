// test_service_getters — guards the UserService/libkernel startup stubs reduced in the HLE audit.
// The UserService getters are (userId, out*) queries that must write a deterministic default to the
// output pointer (accessibility off = 0, age level = adult) rather than leave it uninitialized; the
// libkernel registration/hook calls must resolve to a benign OK-returning no-op. Registered by raw
// NID, so this looks them up by NID and exercises the output contract.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/callback_fs.hpp"
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include "test_scratch.h"

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Call an (userId, int* out) getter and return the value it wrote (sentinel-initialized to catch a
// no-write).
static int32_t call_int_getter(HleFn fn, int32_t sentinel) {
    int32_t out = sentinel;
    fn(1 /*userId*/, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0);
    return out;
}

static void set_game_intent_activity(const char* value) {
#ifdef _WIN32
    _putenv_s("PROSPER_GAME_INTENT_ACTIVITY_ID", value ? value : "");
#else
    if (value) setenv("PROSPER_GAME_INTENT_ACTIVITY_ID", value, 1);
    else unsetenv("PROSPER_GAME_INTENT_ACTIVITY_ID");
#endif
}

int main() {
    printf("== test_service_getters ==\n");
    register_builtin_hle();

    // PlayGo must describe the chunks actually present in an Unreal IoStore dump. In particular,
    // invalid ids must not be reported LOCAL_FAST: DOLL probes sequential ids until BAD_CHUNK_ID,
    // and the old all-local answer ran it through a 1000-id safety cap while pakchunk1 content was
    // waiting to become available (#1373).
    {
        namespace fs = std::filesystem;
        const fs::path app0 = prosper_test::test_scratch_dir() /
            ("prosper-playgo-" + std::to_string((uintptr_t)&fails));
        std::error_code ec;
        fs::remove_all(app0, ec);
        const fs::path paks = app0 / "DOLL" / "Content" / "Paks";
        fs::create_directories(paks, ec);
        std::ofstream(paks / "pakchunk0-ps5.utoc", std::ios::binary).put('\1');
        std::ofstream(paks / "pakchunk0-ps5.ucas", std::ios::binary).put('\1');
        std::ofstream(paks / "pakchunk1-ps5.utoc", std::ios::binary).put('\1');
        std::ofstream(paks / "pakchunk1-ps5.ucas", std::ios::binary).put('\1');
        std::ofstream(paks / "pakchunk2-ps5.utoc", std::ios::binary); // empty placeholder
        std::ofstream(paks / "pakchunk3-ps5.utoc", std::ios::binary).put('\1'); // missing data
        set_app0_root(app0.string());

        HleFn init = Hle::lookup(nid_hash("scePlayGoInitialize"));
        HleFn get_ids = Hle::lookup(nid_hash("scePlayGoGetChunkId"));
        HleFn get_locus = Hle::lookup(nid_hash("scePlayGoGetLocus"));
        CHECK(init && get_ids && get_locus, "PlayGo chunk-query functions registered");
        if (init && get_ids && get_locus) {
            init(0, 0, 0, 0, 0, 0);
            uint32_t entries = 0;
            CHECK(get_ids(1, 0, 0, (uint64_t)(uintptr_t)&entries, 0, 0) == 0 && entries == 2,
                  "GetChunkId reports both non-empty IoStore chunks");
            uint16_t ids[4] = {0xffff, 0xffff, 0xffff, 0xffff};
            entries = 0;
            CHECK(get_ids(1, (uint64_t)(uintptr_t)ids, 4,
                          (uint64_t)(uintptr_t)&entries, 0, 0) == 0 &&
                      entries == 2 && ids[0] == 0 && ids[1] == 1,
                  "GetChunkId requires non-empty IoStore index and data files");
            int8_t loci[2] = {-1, -1};
            CHECK(get_locus(1, (uint64_t)(uintptr_t)ids, 2,
                            (uint64_t)(uintptr_t)loci, 0, 0) == 0 &&
                      loci[0] == 3 && loci[1] == 3,
                  "GetLocus reports discovered chunks LOCAL_FAST");
            const uint16_t missing = 2;
            int8_t missing_locus = -1;
            CHECK(get_locus(1, (uint64_t)(uintptr_t)&missing, 1,
                            (uint64_t)(uintptr_t)&missing_locus, 0, 0) == 0x80B2000Cull &&
                      missing_locus == 0,
                  "GetLocus marks an absent chunk not-downloaded and returns BAD_CHUNK_ID");
            fs::remove(paks / "pakchunk0-ps5.ucas", ec);
            fs::remove(paks / "pakchunk1-ps5.ucas", ec);
            init(0, 0, 0, 0, 0, 0);
            entries = 99;
            CHECK(get_ids(1, 0, 0, (uint64_t)(uintptr_t)&entries, 0, 0) == 0 && entries == 0,
                  "GetChunkId does not apply the non-IoStore fallback to a partial IoStore dump");
        }
        fs::remove_all(app0, ec);
    }

#ifndef _WIN32
    // Guest callback shims must recover saved r11 from the import stub's real +0x28 slot. +0x18 is
    // forwarded arg9 and can legitimately contain an eboot pointer; treating it as %fs crashes the
    // next host-libc call on executable memory.
    {
        struct alignas(16) FakeTcb { uint8_t bytes[0x110]; } tcb{};
        uint64_t tp = (uint64_t)(uintptr_t)&tcb;
        *(uint64_t*)(tcb.bytes + 0x00) = tp;
        *(uint32_t*)(tcb.bytes + 0x108) = 0x50524F53u;
        uint64_t frame[7] = {};
        frame[0] = 0x600001000ull;    // return into an import swap stub
        frame[3] = 0x400000000ull;    // arg9 decoy: the old +0x18 bug selected this as %fs
        frame[5] = tp;                // +0x28: saved r11 / real guest %fs
        CHECK(callback_guest_fs_from_entry_stack((uint64_t)(uintptr_t)frame) == tp,
              "callback shim recovers guest fs from saved-r11 +0x28, not arg9 +0x18");
        *(uint32_t*)(tcb.bytes + 0x108) = 0;
        CHECK(callback_guest_fs_from_entry_stack((uint64_t)(uintptr_t)frame) == 0,
              "callback shim rejects a value without the guest-TCB magic");
        frame[0] = 0x400001000ull;
        CHECK(callback_guest_fs_from_entry_stack((uint64_t)(uintptr_t)frame) == 0,
              "host-context callback entry does not invent a guest fs frame");
    }
#endif

    // UserService getters: age level -> adult (18), accessibility -> off (0). Must WRITE the output.
    struct { const char* nid; int32_t want; const char* what; } getters[] = {
        {"woNpu+45RLk", 18, "GetAgeLevel -> adult (18)"},
        {"rnEhHqG-4xo",  0, "GetAccessibilityChatTranscription -> 0"},
        {"O6IW1-Dwm-w",  0, "GetAccessibilityZoomFollowFocus -> 0"},
        {"-3Y5GO+-i78",  0, "GetAccessibilityTriggerEffect -> 0"},
    };
    for (auto& g : getters) {
        HleFn fn = Hle::lookup(g.nid);
        CHECK(fn != nullptr, g.what);
        if (fn) CHECK(call_int_getter(fn, (int32_t)0xDEAD) == g.want, g.what);
    }

    {
        HleFn fn = Hle::lookup("qbwy0Ub8b3M");
        CHECK(fn != nullptr, "sceUserServiceGetUserNumber registered");
        if (fn) CHECK(call_int_getter(fn, (int32_t)0xDEAD) == 1,
                      "GetUserNumber writes default local-user number 1");
    }

    {
        HleFn fn = Hle::lookup("qP-EvQRl2Hc");
        CHECK(fn != nullptr, "sceLoginDialogInitialize registered");
        if (fn) CHECK(fn(0, 0, 0, 0, 0, 0) == 0,
                      "LoginDialog initialization succeeds without opening UI");
    }

    {
        HleFn fn = Hle::lookup("RnDibcGCPKw");
        CHECK(fn != nullptr, "sceVideodec2QueryComputeMemoryInfo registered");
        if (fn) {
            uint64_t info[3] = {24, 0xDEAD, 0xDEAD};
            CHECK(fn((uint64_t)(uintptr_t)info, 0, 0, 0, 0, 0) == 0,
                  "Videodec2 compute-memory query succeeds for the 24-byte ABI");
            CHECK(info[1] == (64ull << 10) && info[2] == 0,
                  "Videodec2 query writes its one-page no-picture requirement and null allocation");

            auto alloc = Hle::lookup("eD+X2SmxUt4");
            auto query_decoder = Hle::lookup("qqMCwlULR+E");
            auto create = Hle::lookup("CNNRoRYd8XI");
            auto decode = Hle::lookup("852F5+q6+iM");
            auto flush = Hle::lookup("l1hXwscLuCY");
            auto destroy = Hle::lookup("jwImxXRGSKA");
            CHECK(alloc && query_decoder && create && decode && flush && destroy,
                  "Videodec2 compute/decoder lifecycle registered");
            if (alloc && query_decoder && create && decode && flush && destroy) {
                alignas(256) static uint8_t memory[3][64u << 10];
                struct { uint64_t size; uint16_t pipe, queue; uint8_t check, r0; uint16_t r1; }
                    cq{16, 0, 0, 0, 0, 0};
                info[2] = (uint64_t)(uintptr_t)memory[0];
                uint64_t compute_queue = 0;
                CHECK(alloc((uint64_t)(uintptr_t)&cq, (uint64_t)(uintptr_t)info,
                            (uint64_t)(uintptr_t)&compute_queue, 0, 0, 0) == 0 &&
                          compute_queue == info[2],
                      "Videodec2 allocate publishes the caller-owned compute queue");

                struct Config {
                    uint64_t size; uint32_t resource, codec, profile, level;
                    int32_t width, height, dpb; uint32_t depth; uint64_t queue, affinity;
                    int32_t priority; uint8_t optimize, check, r0, r1; uint64_t extra;
                } config{72, 1, 7, 0, 0, 1920, 1080, 4, 2, compute_queue, 0, 0, 0, 0, 0, 0, 0};
                uint64_t decoder_memory[9] = {72};
                CHECK(query_decoder((uint64_t)(uintptr_t)&config,
                                    (uint64_t)(uintptr_t)decoder_memory, 0, 0, 0, 0) == 0 &&
                          decoder_memory[1] == (64ull << 10) && decoder_memory[7] == (64ull << 10),
                      "Videodec2 decoder-memory query initializes all size requirements");
                decoder_memory[2] = (uint64_t)(uintptr_t)memory[0];
                decoder_memory[4] = (uint64_t)(uintptr_t)memory[1];
                decoder_memory[6] = (uint64_t)(uintptr_t)memory[2];
                uint64_t decoder = 0;
                CHECK(create((uint64_t)(uintptr_t)&config, (uint64_t)(uintptr_t)decoder_memory,
                             (uint64_t)(uintptr_t)&decoder, 0, 0, 0) == 0 && decoder != 0,
                      "Videodec2 decoder create publishes an opaque handle");
                uint64_t input[6] = {48, (uint64_t)(uintptr_t)memory[0], 16, 0, 0, 0};
                struct { uint64_t size, data, bytes; uint8_t accepted, pad[7]; }
                    frame{32, (uint64_t)(uintptr_t)memory[1], sizeof(memory[1]), 1, {}};
                uint8_t output[56]{}; *(uint64_t*)output = 56;
                CHECK(decode(decoder, (uint64_t)(uintptr_t)input, (uint64_t)(uintptr_t)&frame,
                             (uint64_t)(uintptr_t)output, 0, 0) == 0 && !frame.accepted,
                      "Videodec2 no-picture decode consumes an access unit deterministically");
                alignas(8) uint8_t short_output[64];
                std::memset(short_output, 0xA5, sizeof short_output);
                *(uint64_t*)short_output = 48;
                uint8_t short_before[sizeof short_output];
                std::memcpy(short_before, short_output, sizeof short_output);
                frame.accepted = 1;
                CHECK(decode(decoder, (uint64_t)(uintptr_t)input, (uint64_t)(uintptr_t)&frame,
                             (uint64_t)(uintptr_t)short_output, 0, 0) == 0x811d0101ull &&
                          frame.accepted == 1 &&
                          std::memcmp(short_output, short_before, sizeof short_output) == 0,
                      "Videodec2 decode rejects the 48-byte output ABI without writing past it");
                CHECK(flush(decoder, (uint64_t)(uintptr_t)&frame,
                            (uint64_t)(uintptr_t)short_output, 0, 0, 0) == 0x811d0101ull &&
                          frame.accepted == 1 &&
                          std::memcmp(short_output, short_before, sizeof short_output) == 0,
                      "Videodec2 flush rejects the 48-byte output ABI without writing past it");
                CHECK(destroy(decoder, 0, 0, 0, 0, 0) == 0,
                      "Videodec2 decoder lifecycle tears down cleanly");
            }
        }
    }

    // libkernel registration/no-op calls: resolve + return OK (0), no crash.
    const char* noops[] = {"rNhWz+lvOMU", "pB-yGZ2nQ9o", "WhCc1w3EhSI",
                           "p5EcQeEeJAE", "bnZxYgAFeA0", "DGMG3JshrZU"};
    for (const char* nid : noops) {
        HleFn fn = Hle::lookup(nid);
        CHECK(fn != nullptr, nid);
        if (fn) CHECK(fn(0, 0, 0, 0, 0, 0) == 0, nid);
    }

    // Native SDK modules ask libkernel to turn an address into the stable linked-module handle.
    // The result lives after 32 opaque qwords at ModuleInfo+0x108, not in the leading size field.
    {
        UnwindModuleDesc modules[2]{};
        modules[0].lo = 0x410000000ull; modules[0].hi = 0x420000000ull;
        modules[1].lo = 0x5c0000000ull; modules[1].hi = 0x5d0000000ull;
        set_unwind_modules(modules, 2);
        HleFn fn = Hle::lookup("f7KBOafysXo");
        CHECK(fn != nullptr, "sceKernelGetModuleInfoFromAddr registered");
        if (fn) {
            alignas(8) uint8_t info[0x1a8];
            memset(info, 0xA5, sizeof info);
            *(uint64_t*)info = sizeof info;
            CHECK(fn(0x5c0001234ull, 2, (uint64_t)(uintptr_t)info, 0, 0, 0) == 0,
                  "GetModuleInfoFromAddr finds an address in a linked module");
            CHECK(*(int32_t*)(info + 0x108) == 0x10001,
                  "GetModuleInfoFromAddr writes the load-order module handle at +0x108");
            *(int32_t*)(info + 0x108) = (int32_t)0xDEADBEEF;
            CHECK(fn(0x700000000ull, 2, (uint64_t)(uintptr_t)info, 0, 0, 0) == (uint64_t)-1 &&
                  *(int32_t*)(info + 0x108) == 0,
                  "GetModuleInfoFromAddr reports an unknown address and clears its handle");
            CHECK(fn(0x410000000ull, 1, (uint64_t)(uintptr_t)info, 0, 0, 0) == 0x80020016ull,
                  "GetModuleInfoFromAddr rejects an unsupported query selector");
        }
    }

    const char* startup_zero[] = {"g0VTBxfJyu0", "Tz4RNUCBbGI", "jh+8XiK4LeE"};
    for (const char* nid : startup_zero) {
        HleFn fn = Hle::lookup(nid);
        CHECK(fn != nullptr, nid);
        if (fn) CHECK(fn(0, 0, 0, 0, 0, 0) == 0, nid);
    }

    {
        auto init = Hle::lookup(nid_hash("scePthreadAttrInit"));
        HleFn guard = Hle::lookup("El+cQ20DynU");
        auto destroy = Hle::lookup(nid_hash("scePthreadAttrDestroy"));
        CHECK(init && guard && destroy, "scePthreadAttrSetguardsize lifecycle registered");
        if (init && guard && destroy) {
            void* attr = nullptr;
            CHECK(init((uint64_t)(uintptr_t)&attr, 0, 0, 0, 0, 0) == 0 && attr,
                  "pthread attributes initialize before setting guard size");
            CHECK(guard((uint64_t)(uintptr_t)&attr, 0x4000, 0, 0, 0, 0) == 0,
                  "scePthreadAttrSetguardsize accepts a valid guard size");
            destroy((uint64_t)(uintptr_t)&attr, 0, 0, 0, 0, 0);
        }
    }

    // Sanitizer malloc replacement query returns a real empty table, not nullptr. Runtimes inspect
    // callback slots even when no sanitizer is active (Dead Cells reads slot +0x18 during startup).
    {
        HleFn fn = Hle::lookup("py6L8jiVAN8");
        CHECK(fn != nullptr, "sceKernelGetSanitizerMallocReplaceExternal registered");
        if (fn) {
            uint64_t addr = fn(0, 0, 0, 0, 0, 0);
            CHECK(addr != 0, "sanitizer malloc replacement table is non-null");
            if (addr) {
                const uint64_t* table = (const uint64_t*)(uintptr_t)addr;
                CHECK(table[0] == 0x70, "sanitizer malloc replacement table reports size 0x70");
                bool empty = true;
                for (int i = 1; i < 14; ++i) empty &= table[i] == 0;
                CHECK(empty, "sanitizer malloc replacement callbacks default to null");
            }
        }
    }

    // Sonic's disconnected network stack still needs valid local context IDs. NP signup is a
    // byte-sized bool output (the live title passes an odd address), and must not clobber neighbors.
    {
        auto pool = Hle::lookup("dgJBaeJnGpo");
        auto ssl = Hle::lookup("hdpVEUDFW3s");
        auto http2 = Hle::lookup("3JCe3lCbQ8A");
        auto web = Hle::lookup("+o9816YQhqQ");
        auto user = Hle::lookup("sk54bi6FtYM");
        auto result = Hle::lookup("0cBgduPRR+M");
        auto signed_up = Hle::lookup("Oad3rvY-NJQ");
        CHECK(pool && ssl && http2 && web && user && result && signed_up,
              "offline network context and NP signup functions registered");
        if (pool && ssl && http2 && web && user && result && signed_up) {
            const char name[] = "SonicWebApi";
            uint64_t pool_id = pool((uint64_t)(uintptr_t)name, 0x8000, 0, 0, 0, 0);
            uint64_t ssl_id = ssl(0x64000, 0, 0, 0, 0, 0);
            uint64_t http_id = http2(pool_id, ssl_id, 0x58000, 3, 0, 0);
            uint64_t web_id = web(http_id, 0x10000, 0, 0, 0, 0);
            uint64_t user_id = user(web_id, 1, 0, 0, 0, 0);
            CHECK(pool_id > 0 && ssl_id > 0 && http_id > 0 && web_id > 0 && user_id > 0,
                  "disconnected console initialization returns valid local context IDs");
            int32_t error = (int32_t)0xDEADBEEF;
            CHECK(result(1, (uint64_t)(uintptr_t)&error, 0, 0, 0, 0) == 0 && error == 0,
                  "NetCtlGetResult writes a successful DISCONNECTED callback result");
            uint8_t bytes[3] = {0xA5, 0xFF, 0x5A};
            CHECK(signed_up(1, (uint64_t)(uintptr_t)&bytes[1], 0, 0, 0, 0) == 0 &&
                      bytes[1] == 0,
                  "sceNpHasSignedUp writes false for the offline user");
            CHECK(bytes[0] == 0xA5 && bytes[2] == 0x5A,
                  "sceNpHasSignedUp writes exactly one byte, preserving odd-address neighbors");
        }
    }

    // Sonic's last two startup imports have no output parameters. Keep the required input and
    // return contract explicit so they cannot regress to the generic success stub unnoticed.
    {
        auto share_param = Hle::lookup("7QZtURYnXG4");
        auto live_init = Hle::lookup("kvYEw2lBndk");
        CHECK(share_param && live_init,
              "Share content-param and GameLiveStreaming initialization registered");
        if (share_param && live_init) {
            const char content[] = "Sonic Origins";
            CHECK(share_param((uint64_t)(uintptr_t)content, 0, 0, 0, 0, 0) == 0,
                  "sceShareSetContentParam accepts a valid content string");
            CHECK(share_param(0, 0, 0, 0, 0, 0) == 0x81960002ull,
                  "sceShareSetContentParam rejects a null string with INVALID_PARAM");
            CHECK(live_init(0x4000, 0, 0, 0, 0, 0) == 0,
                  "sceGameLiveStreamingInitialize accepts its heap-size argument");
        }
    }

    // Message-dialog lifecycle (#144): GetStatus must report NONE before an Open (the old handler
    // returned FINISHED unconditionally, so a guest guarding on GetStatus saw "done" at the wrong
    // stage). Transitions: Initialize -> INITIALIZED(1); Open -> FINISHED(3, auto-dismiss); Close ->
    // NONE(0). Registered by name; look up the NIDs.
    {
        auto init   = Hle::lookup(nid_hash("sceMsgDialogInitialize"));
        auto open   = Hle::lookup(nid_hash("sceMsgDialogOpen"));
        auto close  = Hle::lookup(nid_hash("sceMsgDialogClose"));
        auto status = Hle::lookup(nid_hash("sceMsgDialogGetStatus"));
        auto upd    = Hle::lookup(nid_hash("sceMsgDialogUpdateStatus"));
        CHECK(init && open && close && status && upd, "MsgDialog lifecycle functions registered");
        if (init && open && close && status && upd) {
            close(0,0,0,0,0,0);   // reset to a known idle state
            CHECK(status(0,0,0,0,0,0) == 0, "GetStatus before Initialize -> NONE(0) (was FINISHED)");
            init(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 1, "after Initialize -> INITIALIZED(1)");
            CHECK(upd(0,0,0,0,0,0) == 1, "UpdateStatus before Open also reports INITIALIZED, not FINISHED");
            open(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 3, "after Open -> FINISHED(3) (auto-dismiss)");
            close(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 0, "after Close -> back to NONE(0)");
        }
    }

    // SaveDataDialog lifecycle (#768): returning generic success (0 == NONE) from UpdateStatus makes
    // a title poll forever. Headless Open auto-dismisses to FINISHED and preserves the proven result
    // prefix without touching the reserved tail.
    {
        auto init   = Hle::lookup("s9e3+YpRnzw");
        auto open   = Hle::lookup("4tPhsP6FpDI");
        auto close  = Hle::lookup("fH46Lag88XY");
        auto status = Hle::lookup("ERKzksauAJA");
        auto update = Hle::lookup("KK3Bdg1RWK0");
        auto result = Hle::lookup("yEiJ-qqr6Cg");
        auto term   = Hle::lookup("YuH2FA7azqQ");
        CHECK(init && open && close && status && update && result && term,
              "SaveDataDialog lifecycle functions registered");
        if (init && open && close && status && update && result && term) {
            term(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 0, "SaveDataDialog before Initialize -> NONE(0)");
            init(0,0,0,0,0,0);
            CHECK(update(0,0,0,0,0,0) == 1, "SaveDataDialog Initialize -> INITIALIZED(1)");
            uint8_t param[0x98]{};
            *(uint32_t*)(param + 0x34) = 3;
            *(uint64_t*)(param + 0x70) = 0x123456789abcdef0ull;
            open((uint64_t)(uintptr_t)param,0,0,0,0,0);
            CHECK(update(0,0,0,0,0,0) == 3, "SaveDataDialog Open auto-dismisses -> FINISHED(3)");
            uint8_t out[0x50]; memset(out, 0xAB, sizeof out);
            result((uint64_t)(uintptr_t)out,0,0,0,0,0);
            CHECK(*(uint32_t*)(out + 0x00) == 3 && *(uint32_t*)(out + 0x04) == 0 &&
                  *(uint32_t*)(out + 0x08) == 0,
                  "SaveDataDialog result reports mode and neutral OK/INVALID outcome");
            CHECK(*(uint64_t*)(out + 0x20) == 0x123456789abcdef0ull,
                  "SaveDataDialog result preserves caller userData");
            CHECK(out[0x0c] == 0xAB && *(uint64_t*)(out + 0x10) == 0xABABABABABABABABull &&
                  *(uint64_t*)(out + 0x18) == 0xABABABABABABABABull,
                  "SaveDataDialog result preserves ABI padding and caller-owned output pointers");
            CHECK(out[0x28] == 0xAB && out[0x4f] == 0xAB,
                  "SaveDataDialog result does not overwrite its reserved tail");
            close(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 0, "SaveDataDialog Close -> NONE(0)");
        }
    }

    // SystemService / AppContent / NP getters: each must WRITE its out-param with the CORRECT value —
    // returning success with an unfilled (or wrong-valued) out is the harmful-stub class whose downstream
    // effects are hard to trace (a 0.0 safe-area ratio collapses the viewport; lang 0 localizes to
    // Japanese; NP state SIGNED_IN would send an offline title into network flows). These handlers carry
    // careful fixes but had no test locking them; this does.
    {
        // sceSystemServiceParamGetInt(paramId=LANG(1), int* out) -> en-US(1), not Japanese(0).
        if (HleFn f = Hle::lookup(nid_hash("sceSystemServiceParamGetInt"))) {
            int32_t out = (int32_t)0xDEAD; f(1 /*LANG*/, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0);
            CHECK(out == 1, "ParamGetInt(LANG) -> en-US(1), not the uninitialized/Japanese default");

            out = (int32_t)0xDEAD;
            f(2 /*DATE_FORMAT*/, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0);
            CHECK(out == 2,
                  "ParamGetInt(DATE_FORMAT) -> MM/DD/YYYY(2), matching the en-US default");
        } else CHECK(false, "sceSystemServiceParamGetInt registered");

        // sceSystemServiceGetStatus(status*) -> fills 12 bytes, byte[6] (isCpuMode7CpuNormal)=1, and
        // MUST NOT write past 12 (the oversized-write class — cf. the pad-overflow crash).
        if (HleFn f = Hle::lookup(nid_hash("sceSystemServiceGetStatus"))) {
            uint8_t buf[16]; memset(buf, 0xAB, sizeof buf);
            f((uint64_t)(uintptr_t)buf, 0, 0, 0, 0, 0);
            bool body_ok = true; for (int i = 0; i < 12; i++) if (buf[i] != (i == 6 ? 1 : 0)) body_ok = false;
            CHECK(body_ok, "GetStatus fills 12 bytes (byte6=isCpuMode7CpuNormal=1, rest 0)");
            CHECK(buf[12] == 0xAB && buf[15] == 0xAB, "GetStatus does NOT write past its 12-byte struct");
        } else CHECK(false, "sceSystemServiceGetStatus registered");

        // sceSystemServiceGetDisplaySafeAreaInfo(info*) -> ratio=1.0 (a 0.0 ratio collapses the viewport).
        if (HleFn f = Hle::lookup("1n37q1Bvc5Y")) {
            uint8_t buf[132]; memset(buf, 0xAB, sizeof buf);
            f((uint64_t)(uintptr_t)buf, 0, 0, 0, 0, 0);
            float ratio; memcpy(&ratio, buf, 4);
            CHECK(ratio == 1.0f, "GetDisplaySafeAreaInfo -> ratio 1.0 (full display, no viewport collapse)");
        } else CHECK(false, "sceSystemServiceGetDisplaySafeAreaInfo registered");

        // sceSystemServiceGetHdrToneMapLuminance(out*) -> Kyty-derived three-float layout. A
        // success-only stub left all three display-setup inputs poisoned, while an oversized fill
        // would corrupt adjacent guest stack storage.
        if (HleFn f = Hle::lookup("mPpPxv5CZt4")) {
            struct GuardedLuminance { float value[3]; uint8_t canary[4]; } out;
            memset(&out, 0xAB, sizeof out);
            CHECK(f((uint64_t)(uintptr_t)&out.value, 0, 0, 0, 0, 0) == 0,
                  "GetHdrToneMapLuminance -> OK");
            CHECK(out.value[0] == 80.0f && out.value[1] == 1000.0f && out.value[2] == 0.0f,
                  "GetHdrToneMapLuminance writes the deterministic Kyty-derived fallback values");
            bool canary_ok = true;
            for (uint8_t byte : out.canary) if (byte != 0xAB) canary_ok = false;
            CHECK(canary_ok, "GetHdrToneMapLuminance writes only its modeled 12-byte output");
            CHECK(f(0, 0, 0, 0, 0, 0) == 0x80A10003ull,
                  "GetHdrToneMapLuminance rejects a null output pointer");
        } else CHECK(false, "sceSystemServiceGetHdrToneMapLuminance registered");

        // sceAppContentTemporaryDataMount2(opt, char mp[16]) -> exactly "/temp0\0" (7 bytes), never 16.
        if (HleFn f = Hle::lookup("buYbeLOGWmA")) {
            char mp[16]; memset(mp, 0xAB, sizeof mp);
            f(0, (uint64_t)(uintptr_t)mp, 0, 0, 0, 0);
            CHECK(memcmp(mp, "/temp0\0", 7) == 0, "TemporaryDataMount2 writes \"/temp0\\0\" (game builds paths from it)");
            CHECK((uint8_t)mp[7] == 0xAB, "TemporaryDataMount2 writes exactly 7 bytes, not the full 16");
        } else CHECK(false, "sceAppContentTemporaryDataMount2 registered");

        // sceAppContentTemporaryDataGetAvailableSpaceKb(mp, uint64_t* kb) -> nonzero free (1 GiB).
        if (HleFn f = Hle::lookup("SaKib2Ug0yI")) {
            uint64_t kb = 0xDEAD; f(0, (uint64_t)(uintptr_t)&kb, 0, 0, 0, 0);
            CHECK(kb == 1048576ull, "TemporaryDataGetAvailableSpaceKb -> 1 GiB free (not uninitialized)");
        } else CHECK(false, "sceAppContentTemporaryDataGetAvailableSpaceKb registered");

        // sceNpGetState(userId, SceNpState* out) -> SIGNED_OUT(1). SIGNED_IN would push an offline title
        // into network flows that never complete.
        if (HleFn f = Hle::lookup(nid_hash("sceNpGetState"))) {
            int32_t st = (int32_t)0xDEAD; f(1, (uint64_t)(uintptr_t)&st, 0, 0, 0, 0);
            CHECK(st == 1, "sceNpGetState -> SIGNED_OUT(1), written (offline)");
        } else CHECK(false, "sceNpGetState registered");

        // sceNpGetAccountIdA(userId, uint64_t* out) -> signed-out error AND a written 0 (no stale id).
        if (HleFn f = Hle::lookup(nid_hash("sceNpGetAccountIdA"))) {
            uint64_t id = 0xDEAD; uint64_t r = f(1, (uint64_t)(uintptr_t)&id, 0, 0, 0, 0);
            CHECK(r != 0, "sceNpGetAccountIdA -> signed-out error (not spurious success)");
            CHECK(id == 0, "sceNpGetAccountIdA writes 0 to the account-id out (no uninitialized id)");
        } else CHECK(false, "sceNpGetAccountIdA registered");
    }

    // Dead Cells late-startup services (#552): opaque UDS objects must be non-null, and the
    // synchronous SaveData backend must expose one completion event per unmount instead of
    // reporting success with an untouched type-0 event.
    {
        auto create_event = Hle::lookup("p+GcLqwpL9M");
        auto set_string   = Hle::lookup("MfDb+4Nln64");
        auto post_event   = Hle::lookup("CzkKf7ahIyU");
        auto destroy_event= Hle::lookup("wG+84pnNIuo");
        CHECK(create_event && set_string && post_event && destroy_event,
              "NpUniversalDataSystem event lifecycle registered");
        if (create_event && set_string && post_event && destroy_event) {
            uint64_t event = 0, properties = 0;
            uint64_t r = create_event((uint64_t)(uintptr_t)"ActivityStart", 0,
                                      (uint64_t)(uintptr_t)&event,
                                      (uint64_t)(uintptr_t)&properties, 0, 0);
            CHECK(r == 0, "CreateEvent succeeds for valid output pointers");
            CHECK(event != 0 && properties != 0 && event != properties,
                  "CreateEvent writes distinct non-null opaque objects");
            CHECK(set_string(properties, (uint64_t)(uintptr_t)"activityId",
                             (uint64_t)(uintptr_t)"MainQuest", 0, 0, 0) == 0,
                  "EventPropertyObjectSetString accepts the returned object");
            CHECK(post_event(1, 1, event, 1, 0, 0) == 0,
                  "PostEvent accepts the returned event");
            CHECK(destroy_event(event, 0, 0, 0, 0, 0) == 0,
                  "DestroyEvent accepts the returned event");
        }

        auto umount = Hle::lookup("uW4vfTwMQVo");
        auto get_event = Hle::lookup("j8xKtiFj0SY");
        CHECK(umount && get_event, "SaveData unmount event lifecycle registered");
        if (umount && get_event) {
            uint8_t event[112]; memset(event, 0xAB, sizeof event);
            CHECK(get_event(0, (uint64_t)(uintptr_t)event, 0, 0, 0, 0) == 0x809F0018ull,
                  "GetEventResult with no completion -> NO_EVENT");
            umount(0, 0, 0, 0, 0, 0);
            CHECK(get_event(0, (uint64_t)(uintptr_t)event, 0, 0, 0, 0) == 0,
                  "GetEventResult consumes the queued unmount completion");
            CHECK(*(uint32_t*)(event + 0) == 1, "SaveData completion type -> UMOUNT_BACKUP(1)");
            CHECK(*(int32_t*)(event + 4) == 0 && *(int32_t*)(event + 8) == 1,
                  "SaveData completion reports success for initial user");
            bool empty_tail = true; for (size_t i = 12; i < 104; ++i) empty_tail &= event[i] == 0;
            CHECK(empty_tail, "GetEventResult zeroes the empty title, dir name, and reserved tail");
            CHECK(event[104] == 0xAB && event[111] == 0xAB,
                  "GetEventResult writes exactly the 104-byte event struct");
            CHECK(get_event(0, (uint64_t)(uintptr_t)event, 0, 0, 0, 0) == 0x809F0018ull,
                  "GetEventResult completion is one-shot");
        }

        if (HleFn entitlement = Hle::lookup("xddD23+8TfQ")) {
            uint8_t info[32]; memset(info, 0xAB, sizeof info);
            CHECK(entitlement(0, (uint64_t)(uintptr_t)"DEADCELLSBASESEED",
                              (uint64_t)(uintptr_t)info, 0, 0, 0) != 0,
                  "unknown addcont entitlement fails cleanly while signed out");
            bool untouched = true; for (uint8_t b : info) untouched &= b == 0xAB;
            CHECK(untouched, "failed entitlement query leaves output untouched");
        } else CHECK(false, "sceNpEntitlementAccessGetAddcontEntitlementInfo registered");

        HleFn game_intent_init = Hle::lookup("m87BHxt-H60");
        HleFn game_intent_term = Hle::lookup("0HBYxYAjmf0");
        HleFn game_intent_receive = Hle::lookup("jEIXUAr9XE8");
        HleFn game_intent_get_string = Hle::lookup("rPl0INNc-M8");
        HleFn system_receive = Hle::lookup("656LMQSrg6U");
        HleFn system_status = Hle::lookup("rPo6tV8D9bM");
        CHECK(game_intent_init && game_intent_term && game_intent_receive &&
              game_intent_get_string && system_receive && system_status,
              "Game Intent lifecycle and system-event delivery registered");
        if (game_intent_init && game_intent_term && game_intent_receive &&
            game_intent_get_string && system_receive && system_status) {
            uint64_t init[5] = {0x28, 0, 0, 0, 0};
            alignas(8) uint8_t info[16712];
            set_game_intent_activity(nullptr);
            CHECK(game_intent_init((uint64_t)(uintptr_t)init, 0, 0, 0, 0, 0) == 0,
                  "sceNpGameIntentInitialize accepts the 0x28-byte inert init struct");
            memset(info, 0xAB, sizeof(info));
            *(uint64_t*)info = 16704;
            CHECK(game_intent_receive((uint64_t)(uintptr_t)info, 0, 0, 0, 0, 0) ==
                      0x80553806ull,
                  "Game Intent without a host activity reports INTENT_NOT_FOUND");
            CHECK(*(uint64_t*)info == 16704 && *(int32_t*)(info + 8) == -1,
                  "no-intent receive preserves size and writes invalid user id");
            bool type_empty = true;
            for (size_t i = 12; i < 45; ++i) type_empty &= info[i] == 0;
            bool reserved_untouched = true;
            for (size_t i = 45; i < 308; ++i) reserved_untouched &= info[i] == 0xAB;
            bool data_empty = true;
            for (size_t i = 308; i < 16700; ++i) data_empty &= info[i] == 0;
            bool tail_untouched = true;
            for (size_t i = 16700; i < sizeof(info); ++i) tail_untouched &= info[i] == 0xAB;
            CHECK(type_empty && reserved_untouched && data_empty && tail_untouched,
                  "no-intent receive writes only the documented user/type/data fields");
            char value[40]; memset(value, 0xAB, sizeof(value));
            CHECK(game_intent_get_string((uint64_t)(uintptr_t)(info + 308),
                                         (uint64_t)(uintptr_t)"activityId",
                                         (uint64_t)(uintptr_t)value, sizeof(value), 0, 0) ==
                      0x80553807ull && value[0] == 0,
                  "no-intent property lookup reports VALUE_NOT_FOUND with an empty output");
            CHECK(system_receive((uint64_t)(uintptr_t)info, 0, 0, 0, 0, 0) == 0x80A10004ull,
                  "no host activity leaves the system event stream idle");
            CHECK(game_intent_term(0, 0, 0, 0, 0, 0) == 0,
                  "sceNpGameIntentTerminate succeeds");

            set_game_intent_activity("TITLE_SONIC_1_CLASSIC");
            CHECK(game_intent_init((uint64_t)(uintptr_t)init, 0, 0, 0, 0, 0) == 0,
                  "Game Intent reinitializes for a host-selected activity");
            uint8_t system_status_bytes[12]; memset(system_status_bytes, 0xAB, sizeof(system_status_bytes));
            CHECK(system_status((uint64_t)(uintptr_t)system_status_bytes, 0, 0, 0, 0, 0) == 0 &&
                  *(int32_t*)system_status_bytes == 1,
                  "System Service status advertises one pending GAME_INTENT event");
            uint8_t system_event[8196]; memset(system_event, 0xAB, sizeof(system_event));
            CHECK(system_receive((uint64_t)(uintptr_t)system_event, 0, 0, 0, 0, 0) == 0 &&
                  *(uint32_t*)system_event == 0x10000017u,
                  "host activity produces one GAME_INTENT system event");
            CHECK(system_event[4] == 0xAB && system_event[sizeof(system_event) - 1] == 0xAB,
                  "GAME_INTENT delivery writes only the event type");
            CHECK(system_receive((uint64_t)(uintptr_t)system_event, 0, 0, 0, 0, 0) == 0x80A10004ull,
                  "GAME_INTENT system event is one-shot");
            CHECK(system_status((uint64_t)(uintptr_t)system_status_bytes, 0, 0, 0, 0, 0) == 0 &&
                  *(int32_t*)system_status_bytes == 0,
                  "System Service status clears event_num after GAME_INTENT delivery");

            memset(info, 0xAB, sizeof(info));
            *(uint64_t*)info = 16704;
            CHECK(game_intent_receive((uint64_t)(uintptr_t)info, 0, 0, 0, 0, 0) == 0 &&
                  *(int32_t*)(info + 8) == 1 &&
                  strcmp((char*)info + 12, "launchActivity") == 0,
                  "selected activity receives as a launchActivity for initial user 1");
            memset(value, 0xAB, sizeof(value));
            CHECK(game_intent_get_string((uint64_t)(uintptr_t)(info + 308),
                                         (uint64_t)(uintptr_t)"activityId",
                                         (uint64_t)(uintptr_t)value, sizeof(value), 0, 0) == 0 &&
                  strcmp(value, "TITLE_SONIC_1_CLASSIC") == 0,
                  "activityId property returns the exact host-selected activity");
            char unknown_key[sizeof("activityId")] = "otherKey";
            memset(value, 0xAB, sizeof(value));
            CHECK(game_intent_get_string((uint64_t)(uintptr_t)(info + 308),
                                         (uint64_t)(uintptr_t)unknown_key,
                                         (uint64_t)(uintptr_t)value, sizeof(value), 0, 0) ==
                      0x80553807ull && value[0] == 0,
                  "unknown Game Intent property reports VALUE_NOT_FOUND");
            CHECK(game_intent_term(0, 0, 0, 0, 0, 0) == 0,
                  "Game Intent terminates after activity delivery");
            set_game_intent_activity(nullptr);
        }
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
