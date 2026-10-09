// test_guest_namespace — a guest path that no served mount owns never reaches the host
// filesystem, and is answered from the guest namespace model instead (#4782).
//
// Before #4782 such a path was handed to the host verbatim: a guest rmdir("/oracle_nonexistent_dir")
// ran a real rmdir on the host's root, a relative guest path landed in the host's working directory,
// and /app0 or /savedata0 with nothing behind them composed paths on the host's root. The console
// answers these from read-only system space (console-oracle measurements, #4780):
//   sceKernelRmdir("/oracle_nonexistent_dir")    = 0x8002001e (EROFS)
//   sceKernelUnlink("/oracle_nonexistent/file")  = 0x80020002 (ENOENT)
//   sceKernelOpen / sceKernelStat of that file   = 0x80020002 (ENOENT); open("") likewise.
//
// Two kinds of evidence, because a code alone cannot tell "answered from the model" from "the host
// happened to say the same thing" — on an image-based read-only host root the old passthrough
// returned EROFS for the root-level arms too:
//   - SENTINELS: a host path IS a guest path outside every mount, so the old code acted on it. Each
//     sentinel arm points the guest at a real host file or directory this test owns and checks it is
//     untouched. These run on every host.
//   - A HOST-CALL RECORDER (Linux): rmdir/mkdir/unlink/rename are interposed for this binary
//     (prosper_core is a static library, so its calls bind here) and must never see a guest path
//     the namespace does not serve. This is what makes the root-level console-code arms fail on a
//     read-only host root, where the sentinel trick cannot reach.
//
// Mutations each arm was written against (all redden at least one arm here):
//   M1  resolve_guest() returns the guest spelling as a host path for an unmapped path (the
//       pre-#4782 passthrough)          -> every sentinel and recorder arm
//   M2  unmapped_errno() answers ENOENT for every operation   -> the EROFS arms
//   M3  f_open treats O_CREAT as a lookup                     -> the open(O_CREAT) EROFS arms
//   M4  a relative path is not rooted at the guest root        -> RelativePaths...
//   M5  a mount with an empty host root is served ("/app0" before set_app0_root composes
//       "<empty>/sub", a path on the host's root)              -> UnconfiguredApp0...
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#if defined(__linux__)
#include <dlfcn.h>
#include <mutex>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace prosper;

namespace {

constexpr uint64_t kSceEnoent = 0x80020002ull;
constexpr uint64_t kSceErofs = 0x8002001eull;
constexpr uint64_t kGuestOWriteOnly = 0x0001;
constexpr uint64_t kGuestOCreate = 0x0200;
constexpr int64_t kPosixFailure = -1;

// The root-level spelling every EROFS arm uses. The marker is what the recorder looks for.
constexpr const char* kMarker = "prosper_4782";
constexpr const char* kRootName = "/prosper_4782_absent_entry";

#if defined(__linux__)
std::mutex g_record_mx;
std::vector<std::string> g_recorded;

void record_host_call(const char* op, const char* path) {
    if (!path) return;
    std::lock_guard<std::mutex> lock(g_record_mx);
    g_recorded.push_back(std::string(op) + " " + path);
}

template <typename Fn>
Fn real_symbol(const char* name) {
    return reinterpret_cast<Fn>(dlsym(RTLD_NEXT, name));
}
#endif

uint64_t ptr(const void* p) { return (uint64_t)(uintptr_t)p; }

uint64_t call(const char* name, uint64_t a = 0, uint64_t b = 0, uint64_t c = 0) {
    HleFn fn = Hle::lookup(nid_hash(name));
    EXPECT_NE(fn, nullptr) << name << " is registered";
    return fn ? fn(a, b, c, 0, 0, 0) : ~uint64_t{0};
}

struct PosixResult {
    int64_t result;
    int error;
};

PosixResult posix(const char* name, uint64_t a = 0, uint64_t b = 0, uint64_t c = 0) {
    errno = 0;
    const int64_t result = (int64_t)call(name, a, b, c);
    return {result, errno};
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Every case owns a fresh scratch tree: `app0/` is the /app0 mount, and everything else under the
// root is a HOST path that no mount serves.
class GuestNamespace : public ::testing::Test {
protected:
    void SetUp() override {
        root_ = prosper_test::test_scratch_path("guest-namespace");
        std::error_code ec;
        fs::remove_all(root_, ec);
        fs::create_directories(root_ / "app0");
        { std::ofstream(root_ / "app0" / "fixture.bin", std::ios::binary) << "app0"; }
        register_file_hle();
        set_app0_root((root_ / "app0").string());
#if defined(__linux__)
        std::lock_guard<std::mutex> lock(g_record_mx);
        g_recorded.clear();
#endif
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    // The recorded host calls whose path names something this test only ever spelled as a guest
    // path outside the mounts.
    static std::vector<std::string> leaked_host_calls() {
        std::vector<std::string> leaked;
#if defined(__linux__)
        std::lock_guard<std::mutex> lock(g_record_mx);
        for (const auto& entry : g_recorded)
            if (entry.find(kMarker) != std::string::npos) leaked.push_back(entry);
#endif
        return leaked;
    }

    fs::path root_;
};

} // namespace

#if defined(__linux__)
// The interposed namespace-changing calls. Each records, then forwards to the C library.
extern "C" int rmdir(const char* path) noexcept {
    record_host_call("rmdir", path);
    static auto real = real_symbol<int (*)(const char*)>("rmdir");
    return real ? real(path) : (errno = ENOSYS, -1);
}
extern "C" int unlink(const char* path) noexcept {
    record_host_call("unlink", path);
    static auto real = real_symbol<int (*)(const char*)>("unlink");
    return real ? real(path) : (errno = ENOSYS, -1);
}
extern "C" int mkdir(const char* path, mode_t mode) noexcept {
    record_host_call("mkdir", path);
    static auto real = real_symbol<int (*)(const char*, mode_t)>("mkdir");
    return real ? real(path, mode) : (errno = ENOSYS, -1);
}
extern "C" int rename(const char* from, const char* to) noexcept {
    record_host_call("rename-from", from);
    record_host_call("rename-to", to);
    static auto real = real_symbol<int (*)(const char*, const char*)>("rename");
    return real ? real(from, to) : (errno = ENOSYS, -1);
}

// Positive control: the recorder sees prosper_core's calls. Without it, every "no host call" arm
// below would pass on a recorder that was never linked in.
TEST_F(GuestNamespace, RecorderSeesTheMountedPathsHostCalls) {
    const std::string guest = "/app0/prosper_4782_control_dir";
    EXPECT_EQ(call("mkdir", ptr(guest.c_str()), 0777), 0u);
    EXPECT_EQ(call("rmdir", ptr(guest.c_str())), 0u);
    const auto seen = leaked_host_calls();
    ASSERT_EQ(seen.size(), 2u) << "the mapped mkdir and rmdir both reached the recorder";
    EXPECT_EQ(seen[0].rfind("mkdir ", 0), 0u);
    EXPECT_EQ(seen[1].rfind("rmdir ", 0), 0u);
    EXPECT_NE(seen[0].find((root_ / "app0").string()), std::string::npos)
        << "the recorded path is the mount's host path: " << seen[0];
}
#endif

// rmdir is the measured case; mkdir and unlink share its namei shape (a CREATE/DELETE of a name
// directly under a read-only root). CONFIDENCE: HIGH for rmdir (console), MED for the siblings.
TEST_F(GuestNamespace, RootLevelNamespaceChangesAnswerErofs) {
    struct Case {
        const char* posix_name;
        const char* sce_name;
        uint64_t mode;
    };
    for (const Case& c : {Case{"rmdir", "sceKernelRmdir", 0}, Case{"mkdir", "sceKernelMkdir", 0777},
                          Case{"unlink", "sceKernelUnlink", 0}}) {
        const PosixResult p = posix(c.posix_name, ptr(kRootName), c.mode);
        EXPECT_EQ(p.result, kPosixFailure) << c.posix_name;
        EXPECT_EQ(p.error, EROFS) << c.posix_name << " publishes the bare errno";
        EXPECT_EQ(call(c.sce_name, ptr(kRootName), c.mode), kSceErofs)
            << c.sce_name << " returns SCE_KERNEL_ERROR_EROFS, as the console does for rmdir";
    }
    EXPECT_TRUE(leaked_host_calls().empty())
        << "a root-level guest name reached a host call: " << leaked_host_calls().front();
}

TEST_F(GuestNamespace, RootLevelCreateOpenAnswersErofs) {
    const PosixResult p = posix("open", ptr(kRootName), kGuestOWriteOnly | kGuestOCreate, 0600);
    EXPECT_EQ(p.result, kPosixFailure);
    EXPECT_EQ(p.error, EROFS);
    EXPECT_EQ(call("sceKernelOpen", ptr(kRootName), kGuestOWriteOnly | kGuestOCreate, 0600),
              kSceErofs);
    // fopen's 'w' creates, so it is the same namei CREATE.
    errno = 0;
    EXPECT_EQ(call("fopen", ptr(kRootName), ptr("w")), 0u);
    EXPECT_EQ(errno, EROFS);
    // Without O_CREAT the open is a lookup, and nothing is there.
    EXPECT_EQ(call("sceKernelOpen", ptr(kRootName), kGuestOWriteOnly, 0), kSceEnoent);
}

TEST_F(GuestNamespace, RenameAcrossTheMountBoundaryAnswersErofs) {
    // Out of a mount: the source exists, so the target's RENAME lookup answers.
    const PosixResult out = posix("rename", ptr("/app0/fixture.bin"), ptr(kRootName));
    EXPECT_EQ(out.result, kPosixFailure);
    EXPECT_EQ(out.error, EROFS);
    EXPECT_EQ(call("sceKernelRename", ptr("/app0/fixture.bin"), ptr(kRootName)), kSceErofs);
    EXPECT_EQ(read_file(root_ / "app0" / "fixture.bin"), "app0") << "the source did not move";
    // Into a mount: the source's DELETE lookup answers first.
    EXPECT_EQ(call("sceKernelRename", ptr(kRootName), ptr("/app0/moved.bin")), kSceErofs);
    EXPECT_FALSE(fs::exists(root_ / "app0" / "moved.bin"));
    // A source missing inside the mount is ENOENT before the unmapped target is examined.
    EXPECT_EQ(call("sceKernelRename", ptr("/app0/missing.bin"), ptr(kRootName)), kSceEnoent);
    EXPECT_TRUE(leaked_host_calls().empty())
        << "a root-level guest name reached a host call: " << leaked_host_calls().front();
}

// The console-oracle shapes: a deeper path's parent does not exist, so every operation is ENOENT.
TEST_F(GuestNamespace, LookupsAndDeepPathsAnswerEnoent) {
    const char* deep = "/oracle_nonexistent/file";
    EXPECT_EQ(call("sceKernelUnlink", ptr(deep)), kSceEnoent) << "console: kx_file_unlink_missing";
    EXPECT_EQ(call("sceKernelOpen", ptr(deep), 0, 0), kSceEnoent) << "console: kx_file_open_missing";
    EXPECT_EQ(call("sceKernelOpen", ptr(""), 0, 0), kSceEnoent) << "console: kx_file_open_empty_path";
    uint8_t stat_buffer[0x78] = {};
    EXPECT_EQ(call("sceKernelStat", ptr(deep), ptr(stat_buffer)), kSceEnoent)
        << "console: kx_file_stat_missing";
    struct Lookup {
        const char* name;
        uint64_t second;   // open flags, a stat buffer, or an access mode
    };
    for (const Lookup& l : {Lookup{"open", 0}, Lookup{"stat", ptr(stat_buffer)},
                            Lookup{"lstat", ptr(stat_buffer)}, Lookup{"access", 0}}) {
        const PosixResult p = posix(l.name, ptr(deep), l.second);
        EXPECT_EQ(p.result, kPosixFailure) << l.name;
        EXPECT_EQ(p.error, ENOENT) << l.name;
    }
    // A root-level name is looked up, not changed, so it is absent rather than read-only.
    EXPECT_EQ(call("sceKernelStat", ptr(kRootName), ptr(stat_buffer)), kSceEnoent);
    EXPECT_EQ(call("sceKernelCheckReachability", ptr(kRootName)), kSceEnoent);
    EXPECT_EQ(call("sceKernelChmod", ptr(kRootName), 0644), kSceEnoent);
    EXPECT_EQ(call("sceKernelTruncate", ptr(kRootName), 0), kSceEnoent);
    // A deeper namespace change fails on its missing parent, as the console's unlink does.
    EXPECT_EQ(call("sceKernelRmdir", ptr("/prosper_4782_absent/dir")), kSceEnoent);
    EXPECT_EQ(call("sceKernelMkdir", ptr("/prosper_4782_absent/dir"), 0777), kSceEnoent);
    // /savedata0 with no save mounted is not served (it used to reach the host's "/savedata0/...").
    EXPECT_EQ(call("sceKernelUnlink", ptr("/savedata0/prosper_4782_save.bin")), kSceEnoent);
    EXPECT_TRUE(leaked_host_calls().empty())
        << "an unserved guest path reached a host call: " << leaked_host_calls().front();
}

// A host path is a guest path outside every mount. Under the old passthrough each of these acted on
// the real host entry this test created; now none of them may.
TEST_F(GuestNamespace, HostPathsAreNotGuestPaths) {
    const fs::path dir = root_ / "sentinel_dir";
    const fs::path file = root_ / "sentinel_file";
    fs::create_directories(dir);
    { std::ofstream(file, std::ios::binary) << "keep"; }
    const std::string dir_s = dir.string(), file_s = file.string();
    const std::string new_dir = (root_ / "created_dir").string();
    const std::string new_file = (root_ / "created_file").string();
    const std::string renamed = (root_ / "renamed_file").string();

    EXPECT_EQ(call("sceKernelRmdir", ptr(dir_s.c_str())), kSceEnoent);
    const PosixResult rm = posix("rmdir", ptr(dir_s.c_str()));
    EXPECT_EQ(rm.result, kPosixFailure);
    EXPECT_EQ(rm.error, ENOENT);
    EXPECT_TRUE(fs::is_directory(dir)) << "guest rmdir removed a host directory";

    EXPECT_EQ(call("sceKernelUnlink", ptr(file_s.c_str())), kSceEnoent);
    EXPECT_EQ(call("unlink", ptr(file_s.c_str())), (uint64_t)kPosixFailure);
    EXPECT_EQ(call("sceKernelTruncate", ptr(file_s.c_str()), 0), kSceEnoent);
    EXPECT_EQ(call("sceKernelRename", ptr(file_s.c_str()), ptr(renamed.c_str())), kSceEnoent);
    EXPECT_EQ(read_file(file), "keep") << "guest unlink/truncate/rename changed a host file";
    EXPECT_FALSE(fs::exists(renamed));

    EXPECT_EQ(call("sceKernelMkdir", ptr(new_dir.c_str()), 0777), kSceEnoent);
    EXPECT_FALSE(fs::exists(new_dir)) << "guest mkdir created a host directory";
    EXPECT_EQ(call("sceKernelOpen", ptr(new_file.c_str()), kGuestOWriteOnly | kGuestOCreate, 0600),
              kSceEnoent);
    EXPECT_EQ(call("fopen", ptr(new_file.c_str()), ptr("w")), 0u);
    EXPECT_FALSE(fs::exists(new_file)) << "guest open(O_CREAT)/fopen created a host file";

    // A lookup of a host file that really exists still finds nothing in the guest namespace.
    uint8_t stat_buffer[0x78] = {};
    EXPECT_EQ(call("sceKernelStat", ptr(file_s.c_str()), ptr(stat_buffer)), kSceEnoent);
    EXPECT_EQ(call("sceKernelOpen", ptr(file_s.c_str()), 0, 0), kSceEnoent);
    EXPECT_EQ(call("sceKernelCheckReachability", ptr(dir_s.c_str())), kSceEnoent);
}

// A relative guest path resolves against the guest's working directory, which prosper models as the
// guest root (CONFIDENCE: LOW on the PS5's initial cwd). It must never be the HOST's working
// directory, which is where the old passthrough sent it.
TEST_F(GuestNamespace, RelativePathsDoNotReachTheHostWorkingDirectory) {
    const fs::path cwd = root_ / "cwd";
    fs::create_directories(cwd);
    { std::ofstream(cwd / "prosper_4782_cwd_file", std::ios::binary) << "keep"; }
    std::error_code ec;
    const fs::path previous = fs::current_path(ec);
    ASSERT_FALSE(ec);
    fs::current_path(cwd, ec);
    ASSERT_FALSE(ec);

    const PosixResult unlinked = posix("unlink", ptr("prosper_4782_cwd_file"));
    const uint64_t opened = call("sceKernelOpen", ptr("prosper_4782_cwd_file"), 0, 0);
    const uint64_t made = call("sceKernelMkdir", ptr("prosper_4782_cwd_dir"), 0777);
    const uint64_t created =
        call("sceKernelOpen", ptr("prosper_4782_cwd_new"), kGuestOWriteOnly | kGuestOCreate, 0600);

    fs::current_path(previous, ec);
    EXPECT_EQ(unlinked.result, kPosixFailure);
    EXPECT_EQ(unlinked.error, EROFS) << "a relative name is a root-level name of the guest";
    EXPECT_EQ(opened, kSceEnoent) << "the host file in the host cwd is not visible to the guest";
    EXPECT_EQ(made, kSceErofs);
    EXPECT_EQ(created, kSceErofs);
    EXPECT_EQ(read_file(cwd / "prosper_4782_cwd_file"), "keep");
    EXPECT_FALSE(fs::exists(cwd / "prosper_4782_cwd_dir"));
    EXPECT_FALSE(fs::exists(cwd / "prosper_4782_cwd_new"));
}

#if !defined(_WIN32)
// /app0 with no root configured is not served. The old composition was "<empty root>" + sub, so a
// guest "/app0/<host path>" named that host path directly. (POSIX only: a Windows host path does
// not compose into a valid spelling this way.)
TEST_F(GuestNamespace, UnconfiguredApp0IsNotTheHostRoot) {
    const fs::path file = root_ / "sentinel_file";
    { std::ofstream(file, std::ios::binary) << "keep"; }
    set_app0_root("");
    const std::string guest = "/app0" + file.string();
    const uint64_t unlinked = call("sceKernelUnlink", ptr(guest.c_str()));
    const uint64_t opened = call("sceKernelOpen", ptr(guest.c_str()), 0, 0);
    set_app0_root((root_ / "app0").string());
    EXPECT_EQ(unlinked, kSceEnoent);
    EXPECT_EQ(opened, kSceEnoent);
    EXPECT_EQ(read_file(file), "keep") << "an unconfigured /app0 reached the host's root";
    EXPECT_EQ(resolve_guest_path("/app0/fixture.bin"), (root_ / "app0").string() + "/fixture.bin")
        << "the configured mount is unaffected";
}
#endif
