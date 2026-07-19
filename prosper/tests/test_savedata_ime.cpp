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
#include <filesystem>
#include <string>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

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
struct SaveDirName { char data[32]; };
struct LegacyMount {
    int32_t userId; uint32_t pad; const void* titleId; const SaveDirName* dirName;
    const void* fingerprint; uint64_t blocks; uint32_t mode; uint8_t reserved[32];
};
struct LegacyMount2 {
    int32_t userId; uint32_t pad; const SaveDirName* dirName; uint64_t blocks;
    uint32_t mode; uint8_t reserved[32]; uint32_t tailPad;
};
struct MountResult {
    char mountPoint[16]; uint64_t requiredBlocks; uint32_t unused; uint32_t status;
    uint8_t reserved[28]; uint32_t tailPad;
};
static_assert(offsetof(MemData, buf) == 0 && offsetof(MemData, bufSize) == 8 && offsetof(MemData, offset) == 16, "MemData");
static_assert(offsetof(Setup2, userId) == 4 && offsetof(Setup2, memSize) == 8 && offsetof(Setup2, slotId) == 40, "Setup2");
static_assert(offsetof(Set2, data) == 8 && offsetof(Set2, dataNum) == 32 && offsetof(Set2, slotId) == 36, "Set2");
static_assert(offsetof(Get2, data) == 8 && offsetof(Get2, slotId) == 32, "Get2");
static_assert(sizeof(LegacyMount) == 0x50 && offsetof(LegacyMount, dirName) == 0x10 &&
              offsetof(LegacyMount, mode) == 0x28, "legacy Mount ABI");
static_assert(sizeof(LegacyMount2) == 0x40 && offsetof(LegacyMount2, dirName) == 0x08 &&
              offsetof(LegacyMount2, mode) == 0x18, "legacy Mount2 ABI");
static_assert(sizeof(MountResult) == 0x40 && offsetof(MountResult, status) == 0x1c,
              "SaveData MountResult ABI");

static bool all_bytes_are(const void* data, size_t size, uint8_t value) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) if (bytes[i] != value) return false;
    return true;
}

int main() {
    printf("== test_savedata_ime ==\n");
    const int process_id =
#ifdef _WIN32
        _getpid();
#else
        (int)getpid();
#endif
    const std::filesystem::path test_root = std::filesystem::temp_directory_path() /
        ("prosper-savedata-selftest-" + std::to_string(process_id));
    const std::filesystem::path mount_root = test_root / "mount";
    const std::filesystem::path memory_root = test_root / "memory";
    std::error_code mount_ec;
    std::filesystem::remove_all(test_root, mount_ec);
    std::filesystem::create_directories(mount_root, mount_ec);
    std::filesystem::create_directories(memory_root, mount_ec);
    const std::string mount_root_string = mount_root.string();
    const std::string memory_root_string = memory_root.string();
#ifdef _WIN32
    _putenv_s("PROSPER_SAVE0", mount_root_string.c_str());
    _putenv_s("PROSPER_SAVEDATA_DIR", memory_root_string.c_str());
#else
    setenv("PROSPER_SAVE0", mount_root_string.c_str(), 1);
    setenv("PROSPER_SAVEDATA_DIR", memory_root_string.c_str(), 1);
#endif
    // Hermeticity: SaveDataMemory now persists a synced slot to PROSPER_SAVEDATA_DIR (#432), which
    // survives across ctest runs. Both save backends use this process-private root, so concurrent
    // agents and repeated runs cannot turn the fresh-save assertions into order-dependent tests.
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

    // ---- PS4-inherited SaveData mount ABIs: exact layouts share the /savedata0 backend ----
    {
        HleFn mount = Hle::lookup("32HQAQdwM2o"), mount2 = Hle::lookup("0z45PIH+SNI"),
              mount3 = Hle::lookup("ZP4e7rlzOUk"), umount = Hle::lookup("BMR4F-Uek3E");
        CHECK(mount && mount2 && mount3 && umount,
              "SaveData Mount/Mount2/Mount3/Umount functions registered");
        if (mount && mount2 && mount3 && umount) {
            SaveDirName dir2{}; memcpy(dir2.data, "LegacyMount2", 13);
            LegacyMount2 input2{}; input2.userId = 1; input2.dirName = &dir2;
            input2.blocks = 96; input2.mode = 1; // RDONLY: missing save must not be invented
            struct GuardedResult { MountResult result; uint8_t canary[8]; } guarded;
            memset(&guarded, 0xAB, sizeof guarded);
            CHECK(mount2((uint64_t)(uintptr_t)&input2, (uint64_t)(uintptr_t)&guarded.result,
                         0,0,0,0) == 0x809F0008ull,
                  "Mount2 RDONLY of a missing save -> NOT_FOUND");
            CHECK(all_bytes_are(&guarded, sizeof guarded, 0xAB),
                  "failed Mount2 leaves its result and post-object canary untouched");

            input2.mode = 4; // CREATE
            CHECK(mount2((uint64_t)(uintptr_t)&input2, (uint64_t)(uintptr_t)&guarded.result,
                         0,0,0,0) == 0,
                  "Mount2 CREATE makes and mounts the named save");
            CHECK(strcmp(guarded.result.mountPoint, "/savedata0") == 0 &&
                  guarded.result.requiredBlocks == 0 && guarded.result.status == 1,
                  "Mount2 writes the shared mount point and CREATED status");
            CHECK(guarded.result.unused == 0 && guarded.result.tailPad == 0 &&
                  all_bytes_are(guarded.result.reserved, sizeof guarded.result.reserved, 0),
                  "Mount2 zero-initializes the complete result body");
            CHECK(all_bytes_are(guarded.canary, sizeof guarded.canary, 0xAB),
                  "Mount2 writes exactly the 0x40-byte result");
            const auto expected2 = (mount_root / "LegacyMount2" / "probe.bin").lexically_normal();
            CHECK(std::filesystem::path(resolve_guest_path("/savedata0/probe.bin")).lexically_normal() == expected2,
                  "Mount2 activates /savedata0 path translation");

            CHECK(umount((uint64_t)(uintptr_t)guarded.result.mountPoint, 0,0,0,0,0) == 0,
                  "legacy Umount releases an active mount synchronously");
            CHECK(resolve_guest_path("/savedata0/probe.bin") == "/savedata0/probe.bin",
                  "legacy Umount removes /savedata0 path translation");
            CHECK(umount((uint64_t)(uintptr_t)guarded.result.mountPoint, 0,0,0,0,0) == 0x809F0008ull,
                  "legacy Umount of an inactive mount -> NOT_FOUND");

            memset(&guarded, 0xAB, sizeof guarded);
            input2.mode = 4; // CREATE is exclusive
            CHECK(mount2((uint64_t)(uintptr_t)&input2, (uint64_t)(uintptr_t)&guarded.result,
                         0,0,0,0) == 0x809F0007ull,
                  "Mount2 CREATE of an existing save -> EXISTS");
            CHECK(all_bytes_are(&guarded, sizeof guarded, 0xAB) &&
                  resolve_guest_path("/savedata0/probe.bin") == "/savedata0/probe.bin",
                  "failed exclusive CREATE leaves its result untouched and does not mount");

            memset(&guarded, 0xAB, sizeof guarded);
            input2.mode = 0x20; // CREATE2 opens or creates
            CHECK(mount2((uint64_t)(uintptr_t)&input2, (uint64_t)(uintptr_t)&guarded.result,
                         0,0,0,0) == 0 && guarded.result.status == 0,
                  "Mount2 CREATE2 opens an existing save with OPENED status");
            CHECK(umount((uint64_t)(uintptr_t)guarded.result.mountPoint, 0,0,0,0,0) == 0,
                  "legacy Umount releases the CREATE2-opened save");

            SaveDirName create2_dir{}; memcpy(create2_dir.data, "LegacyCreate2", 14);
            input2.dirName = &create2_dir;
            memset(&guarded, 0xAB, sizeof guarded);
            CHECK(mount2((uint64_t)(uintptr_t)&input2, (uint64_t)(uintptr_t)&guarded.result,
                         0,0,0,0) == 0 && guarded.result.status == 1,
                  "Mount2 CREATE2 creates a missing save with CREATED status");
            CHECK(umount((uint64_t)(uintptr_t)guarded.result.mountPoint, 0,0,0,0,0) == 0,
                  "legacy Umount releases the CREATE2-created save");

            memset(&guarded, 0xAB, sizeof guarded);
            input2.dirName = &dir2;
            input2.mode = 1;
            CHECK(mount2((uint64_t)(uintptr_t)&input2, (uint64_t)(uintptr_t)&guarded.result,
                         0,0,0,0) == 0 && guarded.result.status == 0,
                  "Mount2 RDONLY opens an existing save with OPENED status");
            CHECK(umount((uint64_t)(uintptr_t)guarded.result.mountPoint, 0,0,0,0,0) == 0,
                  "legacy Umount releases the reopened save");

            SaveDirName dir1{}; memcpy(dir1.data, "LegacyMount1", 13);
            LegacyMount input1{}; input1.userId = 1;
            input1.titleId = reinterpret_cast<const void*>(uintptr_t{1}); // decoy: dirName is @0x10
            input1.dirName = &dir1; input1.blocks = 96; input1.mode = 4;
            memset(&guarded, 0xAB, sizeof guarded);
            CHECK(mount((uint64_t)(uintptr_t)&input1, (uint64_t)(uintptr_t)&guarded.result,
                        0,0,0,0) == 0 && guarded.result.status == 1,
                  "legacy Mount decodes its distinct dirName/mode offsets and creates the save");
            const auto expected1 = (mount_root / "LegacyMount1" / "probe.bin").lexically_normal();
            CHECK(std::filesystem::path(resolve_guest_path("/savedata0/probe.bin")).lexically_normal() == expected1,
                  "legacy Mount activates the same /savedata0 backend");
            CHECK(umount((uint64_t)(uintptr_t)guarded.result.mountPoint, 0,0,0,0,0) == 0,
                  "legacy Umount releases the Mount-created save");

            // The shared result writer must preserve the already-live PS5-native Mount3 ABI too.
            SaveDirName dir3{}; memcpy(dir3.data, "NativeMount3", 13);
            alignas(8) uint8_t input3[0x30]{};
            *(const SaveDirName**)(input3 + 0x08) = &dir3;
            *(uint64_t*)(input3 + 0x10) = 96;
            *(uint32_t*)(input3 + 0x20) = 4;
            memset(&guarded, 0xAB, sizeof guarded);
            CHECK(mount3((uint64_t)(uintptr_t)input3, (uint64_t)(uintptr_t)&guarded.result,
                         0,0,0,0) == 0 && guarded.result.status == 1,
                  "shared mount helper preserves Mount3 create/result behavior");
            CHECK(umount((uint64_t)(uintptr_t)guarded.result.mountPoint, 0,0,0,0,0) == 0,
                  "legacy Umount releases a Mount3-created save");

            CHECK(mount2(0, (uint64_t)(uintptr_t)&guarded.result, 0,0,0,0) == 0x809F0000ull &&
                  mount2((uint64_t)(uintptr_t)&input2, 0, 0,0,0,0) == 0x809F0000ull &&
                  umount(0,0,0,0,0,0) == 0x809F0000ull,
                  "legacy mount lifecycle rejects null ABI pointers");
        }
    }

    std::filesystem::remove_all(test_root, mount_ec);

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
