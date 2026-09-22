// guest_memory_copy.cpp — see guest_memory_copy.hpp for the policy (#3734).
#include "host/memory/guest_memory_copy.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include "host/platform/posix_shim.hpp"   // Darwin process_vm_readv/writev via mach
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace prosper::host {
namespace {

// Readability changes only at page boundaries. 4 KiB is the smallest page on every host prosper
// runs on, so walking at this granularity is exact everywhere (merely finer than needed on a 16 KiB
// host).
constexpr uint64_t kPage = 0x1000;

struct ErrnoGuard {
    int saved = errno;
    ~ErrnoGuard() { errno = saved; }
};

// One kernel copy of [addr, addr+n). Returns the bytes it moved (0 on failure). Linux may return a
// prefix; Darwin and Windows report all or nothing in practice.
size_t raw_read(uint64_t addr, void* dst, size_t n) {
#ifdef _WIN32
    SIZE_T got = 0;
    ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(static_cast<uintptr_t>(addr)),
                      dst, n, &got);
    return static_cast<size_t>(got);
#else
    for (;;) {
        iovec local{dst, n};
        iovec remote{reinterpret_cast<void*>(static_cast<uintptr_t>(addr)), n};
        const ssize_t got = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
        if (got < 0 && errno == EINTR) continue;
        return got > 0 ? static_cast<size_t>(got) : 0;
    }
#endif
}

size_t raw_write(uint64_t addr, const void* src, size_t n) {
#ifdef _WIN32
    SIZE_T put = 0;
    WriteProcessMemory(GetCurrentProcess(), reinterpret_cast<void*>(static_cast<uintptr_t>(addr)),
                       src, n, &put);
    return static_cast<size_t>(put);
#else
    for (;;) {
        iovec local{const_cast<void*>(src), n};
        iovec remote{reinterpret_cast<void*>(static_cast<uintptr_t>(addr)), n};
        const ssize_t put = process_vm_writev(getpid(), &local, 1, &remote, 1, 0);
        if (put < 0 && errno == EINTR) continue;
        return put > 0 ? static_cast<size_t>(put) : 0;
    }
#endif
}

// Repeats the kernel copy while it makes progress. Returns the total moved.
template <class Copy>
size_t copy_while_progress(uint64_t addr, size_t bytes, Copy&& copy) {
    size_t done = 0;
    while (done < bytes) {
        const size_t got = copy(addr + done, done, bytes - done);
        if (got == 0) break;
        done += got;
    }
    return done;
}

bool wraps(uint64_t addr, size_t bytes) {
    return bytes != 0 && addr > UINT64_MAX - (static_cast<uint64_t>(bytes) - 1);
}

}  // namespace

bool guest_read_exact(uint64_t src, void* dst, size_t bytes) {
    if (bytes == 0) return true;
    if (src == 0 || dst == nullptr || wraps(src, bytes)) return false;
    ErrnoGuard keep;
    auto* out = static_cast<uint8_t*>(dst);
    return copy_while_progress(src, bytes, [&](uint64_t a, size_t off, size_t n) {
               return raw_read(a, out + off, n);
           }) == bytes;
}

size_t guest_read_prefix(uint64_t src, void* dst, size_t bytes) {
    if (bytes == 0 || src == 0 || dst == nullptr) return 0;
    // Clamp to the top of the address space rather than wrapping (src != 0, so this cannot overflow).
    const uint64_t room = UINT64_MAX - src + 1;
    if (static_cast<uint64_t>(bytes) > room) bytes = static_cast<size_t>(room);
    ErrnoGuard keep;
    auto* out = static_cast<uint8_t*>(dst);
    // Fast path: the whole range at once. Where the kernel returns a prefix (Linux) this already is
    // the answer; everywhere else it is the common all-readable case.
    size_t done = copy_while_progress(src, bytes, [&](uint64_t a, size_t off, size_t n) {
        return raw_read(a, out + off, n);
    });
    // Slow path, only after a short copy: finish page by page, so a host that fails a range
    // wholesale still yields the exact readable prefix. On Linux this is one failing probe.
    while (done < bytes) {
        const uint64_t a = src + done;
        const size_t chunk = static_cast<size_t>(
            std::min<uint64_t>(bytes - done, kPage - (a & (kPage - 1))));
        const size_t got = raw_read(a, out + done, chunk);
        done += got;
        if (got != chunk) break;
    }
    return done;
}

bool guest_write_exact(uint64_t dst, const void* src, size_t bytes) {
    if (bytes == 0) return true;
    if (dst == 0 || src == nullptr || wraps(dst, bytes)) return false;
    ErrnoGuard keep;
    const auto* in = static_cast<const uint8_t*>(src);
    return copy_while_progress(dst, bytes, [&](uint64_t a, size_t off, size_t n) {
               return raw_write(a, in + off, n);
           }) == bytes;
}

}  // namespace prosper::host
