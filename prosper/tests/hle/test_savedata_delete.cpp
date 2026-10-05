// test_savedata_delete (#2081) — sceSaveDataDelete must delete, and refuse what it cannot.
//
// `sceSaveDataDelete` (NID `S1GkePI17zQ`, libSceSaveData_native) was unregistered, so the
// dispatcher answered SCE_OK while deleting nothing: a title told its save was gone would go on
// to recreate or ignore a slot that is still on disk — "reports a deletion that did not happen",
// with a 37-title reach in the #2081 census.
//
// Every arm goes through the real NIDs: Mount3 creates the save, Umount2 releases it, Delete
// removes it. A test calling an internal helper would pass on a build where the NID is still
// absent and the guest still gets the stub.
//
// WHAT EACH ARM KILLS:
//   RemovesAnExistingSave  success without removal (the old stub); the dir must be gone from DISK
//   AbsentSaveIsSuccess    a NOT_FOUND answer for an idempotent delete (shadPS4-compatible)
//   Refusals               null/empty/traversal names and the live mount reading as success
//   NidResolves            the NID staying unregistered; a fabricated NID must NOT resolve
//                          (positive control: the lookup discriminates rather than echoing)
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

using namespace prosper;
namespace fs = std::filesystem;

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

constexpr uint64_t kParameter = 0x809F0000ull;

struct MountPoint {
    char data[16]{};
};
MountPoint mount_point(const char* path) {
    MountPoint mp{};
    snprintf(mp.data, sizeof mp.data, "%s", path);
    return mp;
}

// OrbisSaveDataDelete { u32 userId @0; u32 pad; const TitleId* @8; const DirName* @16;
// u32 unused; u8 reserved[32]; s32 pad } — 64 bytes. titleId stays null on purpose: the
// implementation must never dereference it, and this arm would fault if it did.
struct DeleteDesc {
    uint32_t user_id = 1;
    uint32_t pad0 = 0;
    uint64_t title_id = 0;
    uint64_t dir_name = 0;
    uint32_t unused = 0;
    uint8_t reserved[32]{};
    int32_t pad1 = 0;
};

// Mount3 desc { u32 userId @0; const char* dirName @8; u64 blocks @0x10; u64 reserved @0x18;
// u32 mode @0x20; u32 txres @0x28 }. The reserved word matters: without it mode lands at
// +0x18, reads back 0 (OPEN), and the CREATE mount fails NOT_FOUND.
struct Mount3Desc {
    uint32_t user_id = 1;
    uint32_t pad0 = 0;
    uint64_t dir_name = 0;
    uint64_t blocks = 0x60;
    uint64_t reserved = 0;
    uint32_t mode = 5;  // CREATE|RDONLY
    uint32_t txres = 0;
    uint8_t rest[16]{};
};

uint64_t ptr(const void* p) { return (uint64_t)(uintptr_t)p; }

struct Api {
    HleFn mount3 = nullptr, umount2 = nullptr, del = nullptr;
};

Api registered() {
    register_builtin_hle();
    Api api;
    api.mount3 = Hle::lookup("ZP4e7rlzOUk");
    api.umount2 = Hle::lookup("uW4vfTwMQVo");
    api.del = Hle::lookup("S1GkePI17zQ");
    return api;
}

bool api_ok(const Api& api) { return api.mount3 && api.umount2 && api.del; }

std::string use_scratch(const char* tag) {
    const fs::path scratch = prosper_test::test_scratch_dir() / tag;
    std::error_code ec;
    fs::remove_all(scratch, ec);
    const fs::path save_root = scratch / "save0";
    fs::create_directories(save_root, ec);
    set_env("PROSPER_SAVE0", save_root.string().c_str());
    return save_root.string();
}

uint64_t mount_save(const Api& api, const char* name) {
    Mount3Desc desc{};
    desc.dir_name = ptr(name);
    uint8_t result[0x40]{};
    return api.mount3(ptr(&desc), ptr(result), 0, 0, 0, 0);
}

uint64_t umount_save(const Api& api) {
    const MountPoint mp = mount_point("/savedata0");
    return api.umount2(0, ptr(&mp), 0, 0, 0, 0);
}

uint64_t delete_save(const Api& api, const char* name) {
    DeleteDesc del{};
    del.dir_name = ptr(name);
    return api.del(ptr(&del), 0, 0, 0, 0, 0);
}

}  // namespace

TEST(SaveDataDelete, NidResolves) {
    register_builtin_hle();
    EXPECT_NE(Hle::lookup("S1GkePI17zQ"), nullptr) << "sceSaveDataDelete must be registered";
    EXPECT_EQ(nid_hash("sceSaveDataDelete"), "S1GkePI17zQ")
        << "the registered NID is the firmware export's hash";
    EXPECT_EQ(Hle::lookup("AAAAAAAAAAA"), nullptr)
        << "positive control: the lookup rejects an unknown NID";
}

TEST(SaveDataDelete, RemovesAnExistingSave) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    use_scratch("savedata-delete");
    ASSERT_EQ(mount_save(api, "SLOT0"), 0u);
    ASSERT_EQ(umount_save(api), 0u);
    ASSERT_EQ(delete_save(api, "SLOT0"), 0u);
    // Gone from DISK, not just from a process-global map: re-listing the save root must not
    // find it, which is what a save browser reads after a restart.
    std::error_code ec;
    bool any_slot0 = false;
    const fs::path root = fs::path(getenv("PROSPER_SAVE0"));
    for (const auto& title : fs::directory_iterator(root, ec)) {
        if (!title.is_directory()) continue;
        for (const auto& slot : fs::directory_iterator(title.path(), ec)) {
            if (slot.path().filename() == "SLOT0") any_slot0 = true;
        }
    }
    EXPECT_FALSE(any_slot0) << "SLOT0 must be gone from the save root";
}

TEST(SaveDataDelete, AbsentSaveIsSuccess) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    use_scratch("savedata-delete-absent");
    // Idempotent: deleting what is not there reports success, matching the reference behavior.
    // A NOT_FOUND here would send titles that delete-before-create into an error path.
    EXPECT_EQ(delete_save(api, "NOPE"), 0u);
}

TEST(SaveDataDelete, Refusals) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    use_scratch("savedata-delete-refuse");
    DeleteDesc del{};
    EXPECT_EQ(api.del(0, 0, 0, 0, 0, 0), kParameter) << "null descriptor is refused";
    del.dir_name = 0;
    EXPECT_EQ(api.del(ptr(&del), 0, 0, 0, 0, 0), kParameter) << "null dirName is refused";
    EXPECT_EQ(delete_save(api, ""), kParameter) << "empty dirName is refused";
    EXPECT_EQ(delete_save(api, "../escape"), kParameter) << "traversal is refused";
    EXPECT_EQ(delete_save(api, ".dotdir"), 0u) << "dotted names are ordinary saves";

    // The live mount's backing dir cannot be removed from under it.
    ASSERT_EQ(mount_save(api, "LIVE"), 0u);
    EXPECT_EQ(delete_save(api, "LIVE"), kParameter) << "deleting the live mount is refused";
    ASSERT_EQ(umount_save(api), 0u);
    EXPECT_EQ(delete_save(api, "LIVE"), 0u) << "after unmount the same delete succeeds";
}
