// test_savedata_icon (#2081) — sceSaveDataLoadIcon must read back what SaveIcon wrote.
//
// `sceSaveDataLoadIcon` (NID `cGjO3wM3V28`) was unregistered, so the dispatcher answered SCE_OK
// over an unwritten icon buffer: a title showing the save's picture displayed whatever its own
// heap already held. This completes the round trip the merged SaveIcon handler started — save a
// PNG through the real SaveIcon NID, read it back through the real LoadIcon NID, compare bytes.
//
// WHAT EACH ARM KILLS:
//   NidResolves        the NID staying unregistered (plus an unknown-NID positive control)
//   RoundTrip          success without bytes (the old stub); dataSize must report the FULL file
//                      even when the caller buffer only fits a prefix
//   Refusals           null/wrong-mount/unmounted/missing-file reading as success; the missing
//                      file must leave the caller buffer untouched
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

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
constexpr uint64_t kNotMounted = 0x809F0004ull;
constexpr uint64_t kNotFound = 0x809F0008ull;

struct MountPoint {
    char data[16]{};
};
MountPoint mount_point(const char* path) {
    MountPoint mp{};
    snprintf(mp.data, sizeof mp.data, "%s", path);
    return mp;
}

// Mount3 desc { u32 userId @0; const char* dirName @8; u64 blocks @0x10; u64 reserved @0x18;
// u32 mode @0x20 }. The reserved word matters: without it mode lands at +0x18, reads back 0
// (OPEN), and the CREATE mount fails NOT_FOUND.
struct Mount3Desc {
    uint32_t user_id = 1;
    uint32_t pad0 = 0;
    uint64_t dir_name = 0;
    uint64_t blocks = 96;
    uint64_t reserved = 0;
    uint32_t mode = 5;  // CREATE|RDONLY
    uint32_t txres = 0;
    uint8_t rest[16]{};
};

// SceSaveDataIcon { void* buf @0; size_t bufSize @8; size_t dataSize @0x10; u8 reserved[32] }.
struct Icon {
    uint64_t buf = 0;
    uint64_t buf_size = 0;
    uint64_t data_size = 0;
    uint8_t reserved[32]{};
};

uint64_t ptr(const void* p) { return (uint64_t)(uintptr_t)p; }

struct Api {
    HleFn mount3 = nullptr, umount2 = nullptr, save_icon = nullptr, load_icon = nullptr;
};

Api registered() {
    register_builtin_hle();
    Api api;
    api.mount3 = Hle::lookup("ZP4e7rlzOUk");
    api.umount2 = Hle::lookup("uW4vfTwMQVo");
    api.save_icon = Hle::lookup("c88Yy54Mx0w");
    api.load_icon = Hle::lookup("cGjO3wM3V28");
    return api;
}

bool api_ok(const Api& api) { return api.mount3 && api.umount2 && api.save_icon && api.load_icon; }

void use_scratch(const char* tag) {
    const fs::path scratch = prosper_test::test_scratch_dir() / tag;
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch / "save0", ec);
    set_env("PROSPER_SAVE0", (scratch / "save0").string().c_str());
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

// 256 bytes with no zero run long enough to hide a short read: every position is checked.
std::vector<uint8_t> png_bytes() {
    std::vector<uint8_t> v(256);
    for (size_t i = 0; i < v.size(); i++) v[i] = (uint8_t)(0x89 + i * 37);
    return v;
}

}  // namespace

TEST(SaveDataIcon, NidResolves) {
    register_builtin_hle();
    EXPECT_NE(Hle::lookup("cGjO3wM3V28"), nullptr) << "sceSaveDataLoadIcon must be registered";
    EXPECT_EQ(nid_hash("sceSaveDataLoadIcon"), "cGjO3wM3V28")
        << "the registered NID is the firmware export's hash";
    EXPECT_EQ(Hle::lookup("AAAAAAAAAAA"), nullptr)
        << "positive control: the lookup rejects an unknown NID";
}

TEST(SaveDataIcon, RoundTrip) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    use_scratch("savedata-icon-roundtrip");
    ASSERT_EQ(mount_save(api, "SLOT0"), 0u);
    const MountPoint good = mount_point("/savedata0");

    const std::vector<uint8_t> png = png_bytes();
    Icon save{};
    save.buf = ptr(png.data());
    save.buf_size = png.size();
    save.data_size = png.size();
    ASSERT_EQ(api.save_icon(ptr(&good), ptr(&save), 0, 0, 0, 0), 0u);

    // Full-size read: bytes equal, dataSize reports the file.
    std::vector<uint8_t> out(png.size(), 0xAA);
    Icon load{};
    load.buf = ptr(out.data());
    load.buf_size = out.size();
    ASSERT_EQ(api.load_icon(ptr(&good), ptr(&load), 0, 0, 0, 0), 0u);
    EXPECT_EQ(load.data_size, png.size());
    EXPECT_EQ(out, png) << "the icon reads back byte-identical";

    // Prefix read: min(bufSize, file) lands, dataSize still reports the whole file so the
    // caller learns its buffer was short instead of receiving a silent truncation.
    std::vector<uint8_t> small(64, 0xBB);
    Icon load_small{};
    load_small.buf = ptr(small.data());
    load_small.buf_size = small.size();
    ASSERT_EQ(api.load_icon(ptr(&good), ptr(&load_small), 0, 0, 0, 0), 0u);
    EXPECT_EQ(load_small.data_size, png.size());
    EXPECT_TRUE(std::equal(small.begin(), small.end(), png.begin())) << "prefix matches";
    ASSERT_EQ(umount_save(api), 0u);
}

TEST(SaveDataIcon, Refusals) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    use_scratch("savedata-icon-refuse");
    const MountPoint good = mount_point("/savedata0");
    const MountPoint bad = mount_point("/savedata1");
    Icon icon{};
    uint8_t buf[64];
    std::memset(buf, 0xAA, sizeof buf);
    icon.buf = ptr(buf);
    icon.buf_size = sizeof buf;

    EXPECT_EQ(api.load_icon(0, ptr(&icon), 0, 0, 0, 0), kParameter) << "null mount point";
    EXPECT_EQ(api.load_icon(ptr(&good), 0, 0, 0, 0, 0), kParameter) << "null icon";
    EXPECT_EQ(api.load_icon(ptr(&bad), ptr(&icon), 0, 0, 0, 0), kParameter) << "wrong mount";
    Icon null_buf = icon;
    null_buf.buf = 0;
    EXPECT_EQ(api.load_icon(ptr(&good), ptr(&null_buf), 0, 0, 0, 0), kParameter)
        << "null buffer";
    EXPECT_EQ(api.load_icon(ptr(&good), ptr(&icon), 0, 0, 0, 0), kNotMounted)
        << "nothing mounted";

    // Mounted save, no icon saved: NOT_FOUND with the caller buffer untouched.
    ASSERT_EQ(mount_save(api, "EMPTY"), 0u);
    EXPECT_EQ(api.load_icon(ptr(&good), ptr(&icon), 0, 0, 0, 0), kNotFound);
    EXPECT_EQ(buf[0], 0xAA) << "missing icon leaves the buffer untouched";
    ASSERT_EQ(umount_save(api), 0u);
}
