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
#include <map>
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
std::map<dev_t, int> observed_fds;
bool block_barrier = false, entered_barrier = false, release_barrier = false;
int held_fd = -1;
bool invalid_anchor = false;
int barrier_calls = 0;
int injected_error = 0, interrupted_calls = 0;

int observe_barrier(int fd) {
    std::unique_lock lock(observer_mutex);
    ++barrier_calls;
    struct stat st{};
    if (::fstat(fd, &st) != 0) invalid_anchor = true;
    else { observed_devices.insert(st.st_dev); observed_fds[st.st_dev] = fd; }
    const auto original = st;
    if (block_barrier) {
        held_fd = fd;
        entered_barrier = true;
        observer_cv.notify_all();
        // A failed implementation must fail a bounded test, never strand the laptop.
        if (!observer_cv.wait_for(lock, std::chrono::seconds(5), [] { return release_barrier; }))
            invalid_anchor = true;
        if (::fstat(fd, &st) != 0 || st.st_dev != original.st_dev || st.st_ino != original.st_ino)
            invalid_anchor = true;
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

    const dev_t app_device = device(app);
    const int replacement = (int)call("sceKernelOpen", ptr("/app0/replacement"), 0x601, 0600);
    CHECK(replacement >= 3, "open a real guest dup2 source on the second filesystem");
    if (replacement < 3) return 1;
    if (observed_fds.count(app_device)) {
        const int anchor_fd = observed_fds.at(app_device);
        CHECK(call("sceKernelClose", anchor_fd) == 0x80020009ull,
              "a private anchor slot is not a closeable guest descriptor");
        CHECK(call("sceKernelDup2", replacement, anchor_fd) == (uint64_t)anchor_fd,
              "guest dup2 may select an actual anchor slot as its destination");
        struct stat replaced{};
        CHECK(::fstat(anchor_fd, &replaced) == 0 && replaced.st_dev == other_device,
              "dup2 returns the requested guest descriptor on the source filesystem");
        observed_devices.clear(); observed_fds.clear(); barrier_calls = 0;
        call("sceKernelSync");
        CHECK(barrier_calls == 2 && observed_devices == std::set<dev_t>({app_device, other_device}) &&
              observed_fds.count(app_device) && observed_fds.at(app_device) != anchor_fd,
              "relocating the real anchor retains both filesystem durability obligations");
        CHECK(call("sceKernelClose", anchor_fd) == 0, "close the guest replacement in the former anchor slot");
        const int reused_anchor = ::open((root / "anchor-reused").c_str(), O_CREAT | O_RDWR, 0600);
        CHECK(reused_anchor >= 0 && ::dup2(reused_anchor, anchor_fd) == anchor_fd,
              "reuse the closed former anchor slot with a host file");
        if (reused_anchor >= 0 && reused_anchor != anchor_fd) ::close(reused_anchor);
        if (reused_anchor >= 0) ::close(anchor_fd);
        observed_devices.clear();
        call("sceKernelSync");
        CHECK(observed_devices == std::set<dev_t>({app_device, other_device}),
              "closing and reusing the former anchor slot does not lose either filesystem");
    } else CHECK(false, "the syscall observer captured the first filesystem's actual anchor");
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
#ifdef __linux__
    std::atomic<bool> replacing_started{false}, replacing_finished{false};
    uint64_t replacing_result = (uint64_t)-1;
    const int collision_fd = held_fd;
    std::thread replacing([&] {
        replacing_started = true;
        observer_cv.notify_all();
        replacing_result = call("sceKernelDup2", replacement, collision_fd);
        replacing_finished = true;
        observer_cv.notify_all();
    });
    {
        std::unique_lock lock(observer_mutex);
        CHECK(observer_cv.wait_for(lock, std::chrono::seconds(2), [&] { return replacing_started.load(); }),
              "start guest dup2 against the anchor held inside the intercepted syscall");
        CHECK(!observer_cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return replacing_finished.load(); }),
              "dup2 waits for the active barrier before replacing its descriptor identity");
    }
#endif
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
#ifdef __linux__
    replacing.join();
    CHECK(replacing_result == (uint64_t)collision_fd, "in-flight anchor relocation preserves the guest dup2 result");
    if ((int64_t)replacing_result >= 0) call("sceKernelClose", collision_fd);
    call("sceKernelClose", replacement);
#endif
    block_barrier = false;
    CHECK(!invalid_anchor, "retained barrier descriptors survive concurrent close and descriptor reuse");
#ifdef __linux__
    observed_devices.clear();
    call("sceKernelSync");
    CHECK(observed_devices == std::set<dev_t>({device(app), other_device}),
          "both filesystem obligations survive replacement during an in-flight barrier");
#endif
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
