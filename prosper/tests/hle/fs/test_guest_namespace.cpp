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
// Each arm fails without the fix: against the pre-#4782 hle_file.cpp all seven non-control cases
// fail. Mutations of the fix, and the cases each one reddens (all measured):
//   M1  resolve_guest() hands an unmapped path to the host as if mapped (the old passthrough)
//       -> RootLevelNamespaceChanges, RootLevelCreateOpen, RenameAcross, LookupsAndDeepPaths,
//          HostPathsAreNotGuestPaths, RelativePaths
//   M2  unmapped_errno() answers ENOENT for every operation
//       -> RootLevelNamespaceChanges, RootLevelCreateOpen, RenameAcross, RelativePaths
//   M3  f_open treats O_CREAT as a lookup       -> RootLevelCreateOpen, RelativePaths
//   M4  the line rooting a relative path at the guest root is deleted, so a relative spelling of
//       a mount ("app0/fixture.bin") falls outside every mount   -> RelativeMountSpelling
//   M6  "/dev/urandom" is not a served device (guest_device_named answers nothing)
//       -> RandomDevices*
//   M7  a device descriptor's read is not intercepted (it reads the null backing object)
//       -> RandomDevicesReadEntropy
//   M8  closing a device descriptor leaves its mark, so the closed number still reads entropy
//       -> RandomDeviceMarkDoesNotOutliveItsDescriptor
//   M9  only ".." (not ".") triggers normalization, so "/." is absent  -> DotSpellings...
// (The PlayGo half of the review is guarded in test_service_getters: dropping the empty-/app0 guard
// in discover_playgo_chunks() reddens ServiceGetters.Contract.)
//   M5  a mount with an empty host root is served, composing "<empty>/sub" on the host's root
//       -> LookupsAndDeepPaths (/savedata0), UnconfiguredApp0
// The recorder below defines open/openat. A fortified <fcntl.h> declares them as inline wrappers,
// which would collide with those definitions, so this file is built unfortified.
#undef _FORTIFY_SOURCE
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <cstdarg>
#include <dlfcn.h>
#include <fcntl.h>
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

uint64_t ptr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}

uint64_t call(const char* name, uint64_t a = 0, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0) {
    HleFn fn = Hle::lookup(nid_hash(name));
    EXPECT_NE(fn, nullptr) << name << " is registered";
    return fn ? fn(a, b, c, d, 0, 0) : ~uint64_t{0};
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
        // resolve_guest() re-roots an empty /app0 from PROSPER_APP0, so a developer's exported
        // variable must not decide what these cases see.
#ifdef _WIN32
        _putenv_s("PROSPER_APP0", "");
#else
        unsetenv("PROSPER_APP0");   // NOLINT(concurrency-mt-unsafe): before any thread starts
#endif
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
    static std::vector<std::string> leaked_host_calls(const char* marker = kMarker) {
        std::vector<std::string> leaked;
#if defined(__linux__)
        std::lock_guard<std::mutex> lock(g_record_mx);
        for (const auto& entry : g_recorded)
            if (entry.find(marker) != std::string::npos) leaked.push_back(entry);
#else
        (void)marker;
#endif
        return leaked;
    }

    fs::path root_;
};

}   // namespace

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
// open/openat, in both the plain and the LFS spelling, so a create or a lookup that reached the
// host is seen too. The mode is read only when the flags say one was passed.
namespace {
mode_t open_mode(int flags, va_list args) {
    return (flags & O_CREAT) || (flags & __O_TMPFILE) == __O_TMPFILE ? (mode_t)va_arg(args, int)
                                                                     : 0;
}
}   // namespace
extern "C" int open(const char* path, int flags, ...) {
    va_list args;
    va_start(args, flags);
    const mode_t mode = open_mode(flags, args);
    va_end(args);
    record_host_call("open", path);
    static auto real = real_symbol<int (*)(const char*, int, ...)>("open");
    return real ? real(path, flags, mode) : (errno = ENOSYS, -1);
}
extern "C" int open64(const char* path, int flags, ...) {
    va_list args;
    va_start(args, flags);
    const mode_t mode = open_mode(flags, args);
    va_end(args);
    record_host_call("open", path);
    static auto real = real_symbol<int (*)(const char*, int, ...)>("open64");
    return real ? real(path, flags, mode) : (errno = ENOSYS, -1);
}
extern "C" int openat(int dir, const char* path, int flags, ...) {
    va_list args;
    va_start(args, flags);
    const mode_t mode = open_mode(flags, args);
    va_end(args);
    record_host_call("openat", path);
    static auto real = real_symbol<int (*)(int, const char*, int, ...)>("openat");
    return real ? real(dir, path, flags, mode) : (errno = ENOSYS, -1);
}
extern "C" int openat64(int dir, const char* path, int flags, ...) {
    va_list args;
    va_start(args, flags);
    const mode_t mode = open_mode(flags, args);
    va_end(args);
    record_host_call("openat", path);
    static auto real = real_symbol<int (*)(int, const char*, int, ...)>("openat64");
    return real ? real(dir, path, flags, mode) : (errno = ENOSYS, -1);
}

// Positive control: the recorder sees prosper_core's calls. Without it, every "no host call" arm
// below would pass on a recorder that was never linked in.
TEST_F(GuestNamespace, RecorderSeesTheMountedPathsHostCalls) {
    const std::string guest = "/app0/prosper_4782_control_dir";
    EXPECT_EQ(call("mkdir", ptr(guest.c_str()), 0777), 0u);
    EXPECT_EQ(call("rmdir", ptr(guest.c_str())), 0u);
    const auto seen = leaked_host_calls();
    auto saw = [&](const char* op) {
        for (const auto& entry : seen)
            if (entry.rfind(op, 0) == 0 &&
                entry.find((root_ / "app0").string()) != std::string::npos)
                return true;
        return false;
    };
    EXPECT_TRUE(saw("mkdir ")) << "the mapped mkdir reached the recorder with its host path";
    EXPECT_TRUE(saw("rmdir ")) << "the mapped rmdir reached the recorder with its host path";
    // The open interposer is armed too.
    { std::ofstream(root_ / "app0" / "prosper_4782_control_file", std::ios::binary) << "x"; }
    const uint64_t fd = call("sceKernelOpen", ptr("/app0/prosper_4782_control_file"), 0, 0);
    EXPECT_LT(fd, 0x80000000u);
    if (fd < 0x80000000u) call("sceKernelClose", fd);
    bool opened = false;
    for (const auto& entry : leaked_host_calls())
        if (entry.rfind("open", 0) == 0 &&
            entry.find("prosper_4782_control_file") != std::string::npos)
            opened = true;
    EXPECT_TRUE(opened) << "the mapped open reached the open recorder";
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
    EXPECT_TRUE(leaked_host_calls().empty())
        << "a root-level guest name reached a host call: " << leaked_host_calls().front();
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
    EXPECT_EQ(call("sceKernelOpen", ptr(deep), 0, 0), kSceEnoent)
        << "console: kx_file_open_missing";
    EXPECT_EQ(call("sceKernelOpen", ptr(""), 0, 0), kSceEnoent)
        << "console: kx_file_open_empty_path";
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

// M4: the rooting line matters only for a relative spelling of a mount. Rooted, "app0/fixture.bin"
// is "/app0/fixture.bin"; unrooted, it would fall outside every mount.
TEST_F(GuestNamespace, RelativeMountSpellingResolvesThroughTheMount) {
    const uint64_t fd = call("sceKernelOpen", ptr("app0/fixture.bin"), 0, 0);
    ASSERT_LT(fd, 0x80000000u) << "the relative spelling of /app0 opens the mounted file";
    char bytes[8] = {};
    EXPECT_EQ(call("sceKernelRead", fd, ptr(bytes), sizeof bytes), 4u);
    EXPECT_EQ(std::string(bytes, 4), "app0");
    call("sceKernelClose", fd);
}

// "/." and a relative "." are the root, as "/" and "/app0/.." are (#1234/#1323).
TEST_F(GuestNamespace, DotSpellingsAreTheVirtualRoot) {
    const std::string root = resolve_guest_path("/");
    ASSERT_FALSE(root.empty());
    EXPECT_EQ(resolve_guest_path("/."), root);
    EXPECT_EQ(resolve_guest_path("/./"), root);
    for (const char* spelling : {"/.", "."}) {
        const uint64_t fd = call("sceKernelOpen", ptr(spelling), 0, 0);
        EXPECT_LT(fd, 0x80000000u) << spelling << " opens as the virtual root";
        if (fd < 0x80000000u) call("sceKernelClose", fd);
    }
    // A "." component inside a mount is the same file.
    EXPECT_EQ(resolve_guest_path("/app0/./fixture.bin"), resolve_guest_path("/app0/fixture.bin"));
}

namespace {
constexpr uint64_t kSceEbadf = 0x80020009ull;
constexpr uint64_t kSceEacces = 0x8002000dull;
constexpr uint64_t kSceEexist = 0x80020011ull;
constexpr uint64_t kGuestONonblock = 0x0004;
constexpr uint64_t kGuestOExclusive = 0x0800;

bool is_error(uint64_t sce_result) {
    return sce_result >= 0x80000000u;
}
}   // namespace

// Every IL2CPP title's crypto provider opens /dev/urandom O_RDONLY, falls back to /dev/random with
// O_NONBLOCK, and reads 16 bytes (The Messenger's eboot +0xab8b50). Both must be served, from the
// host CSPRNG, without the guest's spelling reaching the host.
TEST_F(GuestNamespace, RandomDevicesReadEntropy) {
    struct Name {
        const char* path;
        uint64_t flags;
    };
    for (const Name& n : {Name{"/dev/urandom", 0}, Name{"/dev/random", kGuestONonblock}}) {
        const uint64_t fd = call("sceKernelOpen", ptr(n.path), n.flags, 0);
        ASSERT_FALSE(is_error(fd)) << n.path << " opens";
        uint8_t first[16], second[16];
        std::memset(first, 0, sizeof first);
        std::memset(second, 0, sizeof second);
        EXPECT_EQ(call("sceKernelRead", fd, ptr(first), sizeof first), 16u) << n.path;
        EXPECT_EQ(call("read", fd, ptr(second), sizeof second), 16u) << n.path;
        // Two 128-bit draws from a CSPRNG collide with probability 2^-128; zeros mean "not filled".
        EXPECT_NE(std::memcmp(first, second, sizeof first), 0) << n.path << " returns fresh bytes";
        static const uint8_t zeros[16] = {};
        EXPECT_NE(std::memcmp(first, zeros, sizeof zeros), 0) << n.path << " filled the buffer";
        uint8_t positioned[8] = {};
        EXPECT_EQ(call("sceKernelPread", fd, ptr(positioned), sizeof positioned, 1 << 20), 8u);
        // Written through: the descriptor is read-only, as a write to an O_RDONLY descriptor is.
        EXPECT_EQ(call("sceKernelWrite", fd, ptr(first), sizeof first),
                  0xffffffff00000000ull | kSceEbadf);
        call("sceKernelClose", fd);
    }
    EXPECT_TRUE(leaked_host_calls("random").empty())
        << "the guest's device spelling reached a host call: "
        << leaked_host_calls("random").front();
}

TEST_F(GuestNamespace, RandomDevicesAreServedReadOnly) {
    const char* dev = "/dev/urandom";
    EXPECT_EQ(call("sceKernelOpen", ptr(dev), kGuestOWriteOnly, 0), kSceEacces);
    EXPECT_EQ(call("sceKernelOpen", ptr(dev), kGuestOCreate | kGuestOExclusive, 0600), kSceEexist);
    uint8_t stat_buffer[0x78] = {};
    EXPECT_EQ(call("sceKernelStat", ptr(dev), ptr(stat_buffer)), 0u);
    uint16_t mode = 0;
    std::memcpy(&mode, stat_buffer + 0x08, sizeof mode);
    EXPECT_EQ(mode & 0xf000u, 0x2000u) << "a character device";
    EXPECT_EQ(call("sceKernelCheckReachability", ptr(dev)), 0u);
    EXPECT_EQ(call("access", ptr(dev), 4 /*R_OK*/), 0u);
    const PosixResult writable = posix("access", ptr(dev), 2 /*W_OK*/);
    EXPECT_EQ(writable.result, kPosixFailure);
    EXPECT_EQ(writable.error, EACCES);
    EXPECT_EQ(call("sceKernelUnlink", ptr(dev)), kSceEacces);
    EXPECT_EQ(call("sceKernelMkdir", ptr(dev), 0777), kSceEexist);
    EXPECT_EQ(call("sceKernelRename", ptr("/app0/fixture.bin"), ptr(dev)), kSceEacces);
    EXPECT_EQ(read_file(root_ / "app0" / "fixture.bin"), "app0");
    // The HLE stdio path cannot serve entropy, so it refuses visibly (titles with libc.prx never
    // reach it).
    errno = 0;
    EXPECT_EQ(call("fopen", ptr(dev), ptr("r")), 0u);
    EXPECT_EQ(errno, ENODEV);
    // No other /dev name is served: its parent is not listed, so it is absent.
    EXPECT_EQ(call("sceKernelOpen", ptr("/dev/prosper_4782_null"), 0, 0), kSceEnoent);
    EXPECT_EQ(call("sceKernelMkdir", ptr("/dev/prosper_4782_dir"), 0777), kSceEnoent);
    EXPECT_TRUE(leaked_host_calls().empty())
        << "an unserved /dev name reached a host call: " << leaked_host_calls().front();
}

// The device mark follows the descriptor: a dup is a device too, and a closed number that is
// reused for a file reads the file.
TEST_F(GuestNamespace, RandomDeviceMarkDoesNotOutliveItsDescriptor) {
    const uint64_t dev = call("sceKernelOpen", ptr("/dev/urandom"), 0, 0);
    ASSERT_FALSE(is_error(dev));
    const uint64_t copy = call("sceKernelDup", dev);
    ASSERT_FALSE(is_error(copy));
    EXPECT_EQ(call("sceKernelClose", dev), 0u);
    uint8_t bytes[16] = {};
    EXPECT_EQ(call("sceKernelRead", dev, ptr(bytes), sizeof bytes),
              0xffffffff00000000ull | kSceEbadf)
        << "a closed device descriptor is closed, not a source of entropy";
    EXPECT_EQ(call("sceKernelRead", copy, ptr(bytes), sizeof bytes), 16u) << "the dup is a device";
    EXPECT_EQ(call("sceKernelClose", copy), 0u);
    const uint64_t file = call("sceKernelOpen", ptr("/app0/fixture.bin"), 0, 0);
    ASSERT_FALSE(is_error(file));
    ASSERT_TRUE(file == dev || file == copy)
        << "precondition: the file reuses a closed device descriptor number (got " << file << ")";
    char text[8] = {};
    EXPECT_EQ(call("sceKernelRead", file, ptr(text), sizeof text), 4u);
    EXPECT_EQ(std::string(text, 4), "app0");
    call("sceKernelClose", file);
}
