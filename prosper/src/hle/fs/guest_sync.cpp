#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "hle/fs/guest_sync.hpp"
#include <cerrno>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace prosper {
namespace {
struct Anchor {
    int fd;
    explicit Anchor(int value) : fd(value) {}
    ~Anchor() { ::close(fd); }
};
struct SyncState {
    std::mutex mutex;
    std::map<dev_t, std::shared_ptr<Anchor>> filesystems;
    std::set<FILE*> streams;
};
SyncState& state() { static SyncState s; return s; }
void report(const char* operation, int error) {
    // The guest's descriptor-free sync has no error return. Keep failures visible on the host
    // without inventing an SCE error or escalating to whole-host sync.
    std::fprintf(stderr, "[file-sync] %s failed (host error %d)\n", operation, error);
}
} // namespace

void guest_sync_note_fd(int fd) {
    if (fd < 0) return;
    const int saved_errno = errno;
    struct stat st{};
    if (::fstat(fd, &st) != 0) report("filesystem identification", errno);
    else if (S_ISREG(st.st_mode) || S_ISDIR(st.st_mode)) {
        auto& s = state();
        std::lock_guard lock(s.mutex);
        if (!s.filesystems.count(st.st_dev)) {
            const int retained = ::fcntl(fd, F_DUPFD_CLOEXEC, 3);
            if (retained < 0) report("filesystem retention", errno);
            else s.filesystems.emplace(st.st_dev, std::make_shared<Anchor>(retained));
        }
    }
    errno = saved_errno;
}

void guest_sync_note_root(const std::string& root) {
    if (root.empty()) return;
    const int saved_errno = errno;
    const int fd = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NONBLOCK);
    if (fd >= 0) { guest_sync_note_fd(fd); ::close(fd); }
    else report("root retention", errno);
    errno = saved_errno;
}

void guest_sync_note_path(const std::string& path) {
    // Path mutations can reach a nested mount or a symlink target on another filesystem.
    // Never open a device/FIFO merely to register it. The parent covers creation/removal and
    // rename metadata; an existing regular file/directory also covers its target filesystem.
    if (path.empty()) return;
    const int saved_errno = errno;
    auto parent = std::filesystem::path(path).parent_path();
    guest_sync_note_root(parent.empty() ? "." : parent.string());
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0 && (S_ISREG(st.st_mode) || S_ISDIR(st.st_mode))) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
        if (fd >= 0) { guest_sync_note_fd(fd); ::close(fd); }
        else report("path retention", errno);
    }
    errno = saved_errno;
}

void guest_sync_note_stream(FILE* stream, bool writable) {
    if (!stream) return;
    guest_sync_note_fd(::fileno(stream));
    if (writable) {
        auto& s = state();
        std::lock_guard lock(s.mutex);
        s.streams.insert(stream);
    }
}

int guest_sync_close_stream(FILE* stream) {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    s.streams.erase(stream);
    // Serialize fclose with the barrier's fflush so no FILE* can be freed/reused mid-flush.
    return ::fclose(stream);
}

void guest_sync() {
    const int saved_errno = errno;
    std::vector<std::shared_ptr<Anchor>> anchors;
    {
        auto& s = state();
        std::lock_guard lock(s.mutex);
        for (FILE* stream : s.streams)
            if (::fflush(stream) != 0) report("stdio flush", errno);
        for (const auto& [device, anchor] : s.filesystems) anchors.push_back(anchor);
    }
    // Private anchors stay live across concurrent guest close/dup2, mount changes and other
    // barriers. Do not hold the stream registry lock during filesystem writeback.
    for (const auto& anchor : anchors) {
        int error;
        do {
#ifdef __APPLE__
            // Apple Libc gen/sync_volume_np.3: data + metadata, wait for hardware completion.
            error = ::fsync_volume_np(anchor->fd, SYNC_VOLUME_WAIT | SYNC_VOLUME_FULLSYNC);
#else
            error = ::syncfs(anchor->fd) == 0 ? 0 : errno;
#endif
        } while (error == EINTR);
        if (error) report("filesystem sync", error);
    }
    errno = saved_errno;
}
} // namespace prosper
#else
namespace prosper {
void guest_sync_note_fd(int) {}
void guest_sync_note_root(const std::string&) {}
void guest_sync_note_path(const std::string&) {}
void guest_sync_note_stream(FILE*, bool) {}
int guest_sync_close_stream(FILE* stream) { return ::fclose(stream); }
void guest_sync() { (void)::_flushall(); }
} // namespace prosper
#endif
