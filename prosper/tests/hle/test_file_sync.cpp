// #2651: observe the real HLE's barrier selection without issuing ANY host writeback barrier.
// Symbol interception applies to the old ::sync as well as the new per-filesystem primitive.
#include "hle/dispatch/dispatch.hpp"
#include "hle/fs/save_paths.hpp"
#include "fixtures/test_scratch.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <set>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using namespace prosper;
namespace {
int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("[FAIL] %s\n", m); ++fails; } \
                        else std::printf("[ok] %s\n", m); } while (0)
std::atomic<int> host_sync_calls{0};
std::mutex observer_mutex;
std::condition_variable observer_cv;
std::set<dev_t> observed_devices;
bool block_barrier = false, entered_barrier = false, release_barrier = false;
bool invalid_anchor = false;
int barrier_calls = 0;
int injected_error = 0, interrupted_calls = 0;

int observe_barrier(int fd) {
    std::unique_lock lock(observer_mutex);
    ++barrier_calls;
    struct stat st{};
    if (::fstat(fd, &st) != 0) invalid_anchor = true;
    else observed_devices.insert(st.st_dev);
    if (block_barrier) {
        entered_barrier = true;
        observer_cv.notify_all();
        // A failed implementation must fail a bounded test, never strand the laptop.
        if (!observer_cv.wait_for(lock, std::chrono::seconds(5), [] { return release_barrier; }))
            invalid_anchor = true;
        if (::fstat(fd, &st) != 0) invalid_anchor = true;
    }
    if (interrupted_calls > 0) { --interrupted_calls; errno = EINTR; return -1; }
    if (injected_error) { errno = injected_error; return -1; }
    return 0;
}
uint64_t ptr(const void* p) { return (uint64_t)(uintptr_t)p; }
HleFn fn(const char* name) { return Hle::lookup(nid_hash(name)); }
uint64_t call(const char* name, uint64_t a = 0, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0) {
    auto f = fn(name);
    if (!f) { std::printf("[FAIL] unregistered %s\n", name); ++fails; return (uint64_t)-1; }
    return f(a, b, c, d, 0, 0);
}
dev_t device(const std::string& path) {
    struct stat st{};
    CHECK(::stat(path.c_str(), &st) == 0, "fixture filesystem is identifiable");
    return st.st_dev;
}
off_t size(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 ? st.st_size : -1;
}
} // namespace

#ifdef __APPLE__
extern "C" void sync() { ++host_sync_calls; }
#else
extern "C" void sync() noexcept { ++host_sync_calls; }
#endif
#ifdef __APPLE__
extern "C" int fsync_volume_np(int fd, int flags) {
    if (flags != (SYNC_VOLUME_WAIT | SYNC_VOLUME_FULLSYNC)) invalid_anchor = true;
    const int saved_errno = errno;
    const int result = observe_barrier(fd);
    const int error = result == 0 ? 0 : errno;
    errno = saved_errno;
    return error;
}
#else
extern "C" int syncfs(int fd) noexcept { return observe_barrier(fd); }
#endif

int main() {
    register_file_hle();
    ::sync(); // Positive control: even the old call is intercepted, and does no real writeback.
    CHECK(host_sync_calls.exchange(0) == 1, "whole-host interceptor is active");
    CHECK(call("sceKernelSync") == 0 && observed_devices.empty(),
          "no guest storage means no filesystem barrier, with the existing success result");

    const auto root = prosper_test::test_scratch_dir();
    const std::string app = (root / "app").string();
    std::filesystem::create_directory(app);
    const std::string saves = (root / "saves").string();
    setenv("PROSPER_SAVE0", saves.c_str(), 1);
    setenv("PROSPER_SAVEDATA_DIR", (root / "memory").c_str(), 1);
    setenv("PROSPER_TEMP0", (root / "temp").c_str(), 1);
    setenv("PROSPER_DOWNLOAD0", (root / "download").c_str(), 1);
    set_app0_root(app);

    const char data[] = "guest save bytes";
    const int fd = (int)call("sceKernelOpen", ptr("/app0/closed"), 0x601, 0600);
    CHECK(fd >= 3, "open a writable guest descriptor");
    if (fd < 3) return 1;
    CHECK(call("sceKernelWrite", fd, ptr(data), sizeof(data)) == sizeof(data), "write through guest fd");
    CHECK(call("sceKernelClose", fd) == 0, "close before the barrier");
    CHECK(savedata0_mount("slot", SaveDataMountPolicy::OpenOrCreate) == SaveDataMountOutcome::Created,
          "create a guest save mount");
    const std::string save_file = savedata0_mounted_dir() + "/buffered";
    FILE* stream = (FILE*)(uintptr_t)call("fopen", ptr("/savedata0/buffered"), ptr("wb"));
    CHECK(stream != nullptr, "open a writable guest stdio stream");
    if (!stream) return 1;
    char buffer[4096];
    CHECK(::setvbuf(stream, buffer, _IOFBF, sizeof(buffer)) == 0, "force a bounded stdio buffer");
    CHECK(call("fwrite", ptr(data), 1, sizeof(data), ptr(stream)) == sizeof(data), "buffer a guest write");
    CHECK(size(save_file) == 0, "control: the guest payload has not reached the OS yet");
    CHECK(savedata0_umount(), "unmount before the barrier, with a live stream");

    FILE* unrelated = ::fopen((root / "host-buffered").c_str(), "wb");
    CHECK(unrelated != nullptr, "open an unrelated host stdio control");
    char host_buffer[4096];
    if (unrelated) {
        ::setvbuf(unrelated, host_buffer, _IOFBF, sizeof(host_buffer));
        ::fwrite(data, 1, sizeof(data), unrelated);
    }
    barrier_calls = 0;
    errno = ERANGE;
    CHECK(call("sceKernelSync") == 0 && errno == ERANGE, "barrier preserves guest result and caller errno");
    CHECK(size(save_file) == sizeof(data), "barrier flushes guest buffering after save unmount");
    CHECK(size((root / "host-buffered").string()) == 0, "barrier does not flush unrelated host stdio");
    CHECK(barrier_calls == 1 && observed_devices == std::set<dev_t>{device(app)},
          "roots on one filesystem share one barrier");

#ifdef __linux__
    // A separate tmpfs is a small, disposable second filesystem. No loop mount or dirty workload.
    char other_template[] = "/dev/shm/prosper-file-sync-XXXXXX";
    char* other = ::mkdtemp(other_template);
    CHECK(other != nullptr, "create the second-filesystem fixture");
    if (!other) return 1;
    const std::string other_root = other;
    const dev_t other_device = device(other_root);
    CHECK(other_device != device(app), "positive control: roots occupy distinct filesystems");
    setenv("PROSPER_SAVE0", other, 1);
    CHECK(savedata0_mount("second", SaveDataMountPolicy::OpenOrCreate) == SaveDataMountOutcome::Created,
          "mount save storage on a second filesystem");
    const int saved_fd = (int)call("sceKernelOpen", ptr("/savedata0/closed"), 0x601, 0600);
    CHECK(saved_fd >= 3 && call("sceKernelWrite", saved_fd, ptr(data), sizeof(data)) == sizeof(data),
          "write a file on the second filesystem");
    if (saved_fd >= 3) call("sceKernelClose", saved_fd);
    CHECK(savedata0_umount(), "retire the second save mount before sync");
    set_app0_root(other_root); // Switch app filesystems as well as retiring the save mount.
    observed_devices.clear();
    barrier_calls = 0;
    CHECK(call("sceKernelSync") == 0 && barrier_calls == 2 && observed_devices == std::set<dev_t>({device(app), other_device}),
          "closed files and retired mounts on both filesystems remain covered");
#endif

    // Hold the intercepted barrier after its anchor snapshot, and close/reuse guest resources.
    {
        std::lock_guard lock(observer_mutex);
        block_barrier = true;
    }
    std::thread syncing([] { call("sceKernelSync"); });
    bool entered;
    {
        std::unique_lock lock(observer_mutex);
        entered = observer_cv.wait_for(lock, std::chrono::seconds(2), [] { return entered_barrier; });
    }
    CHECK(entered, "barrier reached the controlled syscall interception point");
    CHECK(call("fclose", ptr(stream)) == 0, "concurrent fclose completes while filesystem sync is held");
    const int reused = ::open((root / "reused").c_str(), O_CREAT | O_RDWR, 0600);
    CHECK(reused >= 0 && ::dup2(reused, fd) == fd, "reuse the old guest descriptor number");
    if (reused >= 0 && reused != fd) ::close(reused);
    if (reused >= 0) ::close(fd);
    {
        std::lock_guard lock(observer_mutex);
        release_barrier = true;
        observer_cv.notify_all();
    }
    syncing.join();
    block_barrier = false;
    CHECK(!invalid_anchor, "retained barrier descriptors survive concurrent close and descriptor reuse");
    interrupted_calls = 1;
    const int before_interrupt = barrier_calls;
    CHECK(call("sceKernelSync") == 0 && interrupted_calls == 0 &&
          barrier_calls == before_interrupt + (int)observed_devices.size() + 1,
          "retry a controlled EINTR rather than dropping the interrupted filesystem");
    injected_error = EIO;
    CHECK(call("sceKernelSync") == 0, "host writeback errors retain the descriptor-free success ABI");
    injected_error = 0;
    CHECK(host_sync_calls == 0, "the guest never calls whole-host sync");
    if (unrelated) ::fclose(unrelated);
#ifdef __linux__
    std::filesystem::remove_all(other_root);
#endif
    std::printf("failures: %d\n", fails);
    return fails ? 1 : 0;
}
