// guest_devices.cpp — see guest_devices.hpp.
#include "hle/fs/guest_devices.hpp"

#include "host/platform/host_entropy.hpp"
#include "host/platform/null_device.hpp"

#include <array>
#include <atomic>
#include <cerrno>

namespace prosper {
namespace {

// FreeBSD open(2) flags (the guest's), not the host's.
constexpr uint64_t kGuestAccessMode = 0x0003;
constexpr uint64_t kGuestCreate = 0x0200;
constexpr uint64_t kGuestExclusive = 0x0800;
constexpr uint64_t kGuestDirectory = 0x00020000;

// One mark per descriptor number. Lock-free so the read path pays one relaxed load, and only when
// some device descriptor is open at all.
constexpr int kMaxTrackedFd = 1 << 16;
std::array<std::atomic<uint8_t>, kMaxTrackedFd> g_device_fd{};
std::atomic<int> g_open_device_fds{0};

void mark(int fd, bool device) {
    if (fd < 0 || fd >= kMaxTrackedFd) return;
    const uint8_t was = g_device_fd[(size_t)fd].exchange(device ? 1 : 0);
    if (was && !device) g_open_device_fds.fetch_sub(1);
    if (!was && device) g_open_device_fds.fetch_add(1);
}

}   // namespace

std::optional<GuestDevice> guest_device_named(std::string_view rooted_path) {
    if (rooted_path == "/dev/urandom" || rooted_path == "/dev/random") return GuestDevice::Random;
    return std::nullopt;
}

int guest_device_open_errno(GuestDevice, uint64_t guest_flags) {
    if ((guest_flags & kGuestCreate) && (guest_flags & kGuestExclusive)) return EEXIST;
    if (guest_flags & kGuestDirectory) return ENOTDIR;
    if ((guest_flags & kGuestAccessMode) != 0) return EACCES;   // served read-only
    return 0;
}

int guest_device_open_fd(GuestDevice) {
    // A real descriptor, so close/dup/fstat/lseek behave natively; its reads are answered from the
    // host CSPRNG by the read paths in hle_file.cpp, and the null device itself is never read.
    const int fd = host::open_null_device_readonly();
    if (fd < 0) return -1;
    if (fd >= kMaxTrackedFd) {
        host::close_null_device(fd);
        errno = EMFILE;
        return -1;
    }
    mark(fd, true);
    return fd;
}

bool guest_device_is_fd(int fd) {
    if (g_open_device_fds.load(std::memory_order_relaxed) == 0) return false;
    return fd >= 0 && fd < kMaxTrackedFd && g_device_fd[(size_t)fd].load(std::memory_order_relaxed);
}

void guest_device_forget_fd(int fd) {
    mark(fd, false);
}

void guest_device_copy_fd(int from, int to) {
    if (to < 0 || to == from) return;
    mark(to, guest_device_is_fd(from));
}

bool guest_device_fill(void* buffer, std::size_t count) {
    return count == 0 || host::host_entropy_fill(buffer, count);
}

}   // namespace prosper
