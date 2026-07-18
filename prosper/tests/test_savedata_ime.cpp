// test_savedata_ime (#191) — libSceImeDialog lifecycle + the libSceSaveData "save-data memory" API.
// The ImeDialog auto-completes (no keyboard UI) so a poll loop can't hang; the SaveData memory API is
// a real per-(user,slot) block: Setup allocates, Set writes guest->block, Get reads block->guest,
// Sync commits. Struct layouts mirror shadPS4 save_data/savedata.cpp (offsets asserted below).
#include "../src/hle/dispatch.hpp"
#include <cstdio>
#include <cstdlib>   // setenv (test-private save dir)
#include <cstdint>
#include <cstddef>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Guest struct mirrors (natural x86-64 alignment == the guest's LP64 layout).
struct MemData    { void* buf; uint64_t bufSize; int64_t offset; uint8_t rsv[40]; };
struct Setup2     { uint32_t option; int32_t userId; uint64_t memSize; uint64_t iconSize;
                    const void* initParam; const void* initIcon; uint32_t slotId; uint8_t rsv[20]; };
struct SetupResult{ uint64_t existed; uint8_t rsv[16]; };
struct Set2       { int32_t userId; uint8_t pad[4]; const MemData* data; const void* param;
                    const void* icon; uint32_t dataNum; uint32_t slotId; uint8_t rsv[32]; };
struct Get2       { int32_t userId; uint8_t pad[4]; MemData* data; const void* param;
                    const void* icon; uint32_t slotId; uint8_t rsv[28]; };
static_assert(offsetof(MemData, buf) == 0 && offsetof(MemData, bufSize) == 8 && offsetof(MemData, offset) == 16, "MemData");
static_assert(offsetof(Setup2, userId) == 4 && offsetof(Setup2, memSize) == 8 && offsetof(Setup2, slotId) == 40, "Setup2");
static_assert(offsetof(Set2, data) == 8 && offsetof(Set2, dataNum) == 32 && offsetof(Set2, slotId) == 36, "Set2");
static_assert(offsetof(Get2, data) == 8 && offsetof(Get2, slotId) == 32, "Get2");

int main() {
    printf("== test_savedata_ime ==\n");
    // Hermeticity: SaveDataMemory now persists a synced slot to PROSPER_SAVEDATA_DIR (#432), which
    // survives across ctest runs -- so the "fresh slot" check below would see a PRIOR run's file. Point
    // it at a test-private dir and clear that run's slot file so fresh/resume are deterministic.
#ifndef _WIN32
    setenv("PROSPER_SAVEDATA_DIR", "/tmp/prosper-savemem-selftest", 1);
    remove("/tmp/prosper-savemem-selftest/savemem_1_0.bin");
#endif
    register_builtin_hle();

    // ---- libSceImeDialog: auto-completing lifecycle ----
    HleFn ime_init = Hle::lookup("NUeBrN7hzf0"), ime_status = Hle::lookup("IADmD4tScBY"),
          ime_result = Hle::lookup("x01jxu+vxlc"), ime_term = Hle::lookup("gyTyVn+bXMw"),
          ime_abort = Hle::lookup("oBmw4xrmfKs");
    CHECK(ime_init && ime_status && ime_result && ime_term && ime_abort, "ImeDialog functions registered");
    if (ime_init) {
        ime_term(0,0,0,0,0,0);
        CHECK(ime_status(0,0,0,0,0,0) == 0, "ImeDialog status is NONE(0) before Init");
        ime_init(0,0,0,0,0,0);
        CHECK(ime_status(0,0,0,0,0,0) == 2, "after Init -> OrbisImeDialogStatus::Finished(2) so a poll loop exits");
        int32_t endStatus = (int32_t)0xDEAD;
        ime_result((uint64_t)(uintptr_t)&endStatus, 0, 0, 0, 0, 0);
        CHECK(endStatus == 0, "GetResult writes endStatus = OK(0)");
        ime_term(0,0,0,0,0,0);
        CHECK(ime_status(0,0,0,0,0,0) == 0, "after Term -> NONE(0)");
        ime_init(0,0,0,0,0,0); ime_abort(0,0,0,0,0,0);
        CHECK(ime_status(0,0,0,0,0,0) == 0, "Abort -> NONE(0)");
    }

    // ---- libSceSaveData memory API: Setup -> Set -> Get round-trip ----
    HleFn setup = Hle::lookup("oQySEUfgXRA"), set = Hle::lookup("cduy9v4YmT4"),
          get = Hle::lookup("QwOO7vegnV8"), sync = Hle::lookup("wiT9jeC7xPw");
    CHECK(setup && set && get && sync, "SaveData memory functions registered");
    if (setup && set && get && sync) {
        // Setup a 256-byte block for (user 1, slot 0): a fresh slot reports existed == 0.
        Setup2 su{}; su.userId = 1; su.memSize = 256; su.slotId = 0;
        SetupResult res{}; res.existed = (uint64_t)-1;
        CHECK(setup((uint64_t)(uintptr_t)&su, (uint64_t)(uintptr_t)&res, 0,0,0,0) == 0, "SetupSaveDataMemory2 -> OK");
        CHECK(res.existed == 0, "fresh slot reports existedMemorySize 0 (first run)");

        // Set: write 16 bytes at offset 32.
        uint8_t src[16]; for (int i = 0; i < 16; i++) src[i] = (uint8_t)(0xA0 + i);
        MemData sd{}; sd.buf = src; sd.bufSize = 16; sd.offset = 32;
        Set2 st{}; st.userId = 1; st.data = &sd; st.dataNum = 1; st.slotId = 0;
        CHECK(set((uint64_t)(uintptr_t)&st, 0,0,0,0,0) == 0, "SetSaveDataMemory2 -> OK");

        // Get: read 16 bytes from offset 32 -> must equal what Set wrote.
        uint8_t dst[16]; memset(dst, 0, sizeof dst);
        MemData gd{}; gd.buf = dst; gd.bufSize = 16; gd.offset = 32;
        Get2 gt{}; gt.userId = 1; gt.data = &gd; gt.slotId = 0;
        CHECK(get((uint64_t)(uintptr_t)&gt, 0,0,0,0,0) == 0, "GetSaveDataMemory2 -> OK");
        CHECK(memcmp(src, dst, 16) == 0, "Set -> Get round-trips the exact bytes at the offset");

        // Sync commits; Setup again on the same slot now reports the existing size.
        CHECK(sync((uint64_t)(uintptr_t)&st, 0,0,0,0,0) == 0, "SyncSaveDataMemory -> OK");
        SetupResult res2{}; res2.existed = (uint64_t)-1;
        setup((uint64_t)(uintptr_t)&su, (uint64_t)(uintptr_t)&res2, 0,0,0,0);
        CHECK(res2.existed == 256, "re-Setup reports existedMemorySize 256 (resume, not first run)");

        // Get on an un-Setup slot is MEMORY_NOT_READY (not a spurious success on uninitialized data).
        Get2 g5{}; g5.userId = 1; g5.data = &gd; g5.slotId = 5;
        CHECK(get((uint64_t)(uintptr_t)&g5, 0,0,0,0,0) == 0x809F0012ull, "Get before Setup -> MEMORY_NOT_READY");

        // A null param is rejected, never a wild write.
        CHECK(set(0,0,0,0,0,0) == 0x809F0000ull, "Set(NULL) -> PARAMETER error (no wild deref)");
    }

    // The original scalar-argument API uses slot 0 and must share the same backing behavior as *2.
    HleFn setup_v1 = Hle::lookup("v7AAAMo0Lz4"), set_v1 = Hle::lookup("h3YURzXGSVQ"),
          get_v1 = Hle::lookup("7Bt5pBC-Aco");
    CHECK(setup_v1 && set_v1 && get_v1, "SaveData memory v1 functions registered");
    if (setup_v1 && set_v1 && get_v1) {
        constexpr uint64_t user = 0x392;
        CHECK(setup_v1(user, 64, 0, 0, 0, 0) == 0, "SetupSaveDataMemory(v1) -> OK");
        uint8_t src[8] = {0x39, 0x20, 0x17, 0x07, 0x18, 0x01, 0x08, 0x00};
        CHECK(set_v1(user, (uint64_t)(uintptr_t)src, sizeof src, 11, 0, 0) == 0,
              "SetSaveDataMemory(v1) -> OK");
        uint8_t dst[8]{};
        CHECK(get_v1(user, (uint64_t)(uintptr_t)dst, sizeof dst, 11, 0, 0) == 0,
              "GetSaveDataMemory(v1) -> OK");
        CHECK(memcmp(src, dst, sizeof src) == 0, "v1 Set -> Get round-trips bytes through slot 0");
        CHECK(get_v1(user + 1, (uint64_t)(uintptr_t)dst, sizeof dst, 0, 0, 0) == 0x809F0012ull,
              "v1 Get before Setup -> MEMORY_NOT_READY");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
