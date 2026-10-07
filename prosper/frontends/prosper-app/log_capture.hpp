#pragma once
// log_capture.hpp — re-points file descriptors (stdout/stderr in the app) at pipes whose reader
// threads forward every byte to the original destination AND keep a line copy in a LogRing. It
// feeds the library's Game Log panel. Header-only and free of SDL/ImGui so the teardown contract
// is unit-tested (test_log_capture.cpp).
//
// The teardown contract, and why it is shaped this way: a reader sees EOF only when EVERY write
// end of its pipe is closed, and a child process spawned while the capture is installed (SDL's
// xdg-open for "Show in Explorer", say) inherits fds 1/2, which ARE write ends. So uninstall()
// never waits for EOF. It restores the original fds first (which protects every byte written
// afterwards), asks the readers to stop, and waits a BOUNDED time for each to report its pipe
// empty. A reader that drained but has no EOF yet is detached and keeps forwarding whatever the
// child still writes until the child closes its copy; it owns its read end and closes it itself,
// so nothing is closed under it.
//
// Install is all-or-nothing: any failure unwinds every dup2 and descriptor it made, so a target fd
// is never left pointing at a pipe nobody reads (which would block the writer after ~64 KiB).

#include "log_ring.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>   // _O_BINARY, _O_NOINHERIT
#include <io.h>      // _pipe, _dup, _dup2, _read, _write, _close, _get_osfhandle
#else
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

namespace prosper::frontend {

class LogCapture {
public:
    struct Options {
        // TEST SEAM: readers do not read until stop is requested, so a test can put bytes in the
        // pipe that only the bounded drain can deliver. Never set outside tests.
        bool hold_readers_until_stop = false;
    };

    LogCapture() = default;
    // A process that exits while still installed must not std::terminate on a joinable reader
    // in a static destructor; the readers die with the process.
    ~LogCapture() {
        for (Stream& s : streams_)
            if (s.reader.joinable()) s.reader.detach();
    }
    LogCapture(const LogCapture&) = delete;
    LogCapture& operator=(const LogCapture&) = delete;

    // Capture every fd in `targets`. `ring` must outlive every reader (the app never frees it).
    // Returns false, with every target exactly as it was, when any step fails.
    bool install(LogRing* ring, const std::vector<int>& targets) {
        return install(ring, targets, Options());
    }
    bool install(LogRing* ring, const std::vector<int>& targets, Options opts) {
        if (installed_ || !ring || targets.empty()) return false;
        auto shared = std::make_shared<Shared>();
        shared->hold = opts.hold_readers_until_stop;
        std::vector<Stream> streams;
        auto unwind = [&streams] {
            for (Stream& s : streams) {
                if (s.redirected) sys_dup2(s.saved, s.target);
                for (int fd : {s.pipe_rd, s.pipe_wr, s.saved})
                    if (fd >= 0) sys_close(fd);
            }
            return false;
        };
        fflush(stdout);
        fflush(stderr);
        for (int target : targets) {
            streams.push_back(Stream{});
            Stream& s = streams.back();
            s.target = target;
            int fds[2] = {-1, -1};
            if (!make_pipe(fds)) return unwind();
            s.pipe_rd = fds[0];
            s.pipe_wr = fds[1];
            s.saved = dup_noinherit(target);
            if (s.saved < 0) return unwind();
        }
        for (Stream& s : streams) {
            if (sys_dup2(s.pipe_wr, s.target) < 0) return unwind();
            s.redirected = true;
        }
        // Nothing below can fail, so the readers start only once the capture is complete.
        for (Stream& s : streams) {
            s.reader = std::thread(reader_main, shared, ring, s.pipe_rd, s.saved);
            s.pipe_rd = -1;   // the reader owns and closes it
        }
        streams_ = std::move(streams);
        shared_ = std::move(shared);
        installed_ = true;
        return true;
    }

    // Restore every target, drain what is already in the pipes, and return within roughly `bound`
    // even when a child still holds a write end. Returns true when every reader reached EOF and
    // was joined; false when at least one had to be left running (detached) because something
    // outside this object still holds its pipe open. Idempotent; a no-op when not installed.
    bool uninstall(std::chrono::milliseconds bound) {
        if (!installed_) return true;
        installed_ = false;
        fflush(stdout);
        fflush(stderr);
        for (Stream& s : streams_) {
            sys_dup2(s.saved, s.target);
            sys_close(s.pipe_wr);   // ours; a child's inherited copy may remain
            s.pipe_wr = -1;
        }
        const size_t n = streams_.size();
        {
            std::unique_lock<std::mutex> lock(shared_->mu);
            shared_->stop = true;
            shared_->cv.notify_all();
            shared_->cv.wait_for(lock, bound, [&] { return shared_->drained >= n; });
        }
        bool all_joined = true;
        {
            std::lock_guard<std::mutex> lock(shared_->mu);
            all_joined = shared_->exited >= n;
        }
        for (Stream& s : streams_) {
            if (!s.reader.joinable()) continue;
            if (all_joined)
                s.reader.join();
            else
                s.reader.detach();   // safe: it holds its own shared state and read end
        }
        // The saved descriptors stay open for process life: a detached reader may still forward
        // to them, and closing a console fd a thread writes to would be worse than leaking it.
        streams_.clear();
        shared_.reset();
        return all_joined;
    }

    // Crash path: point `target` back at its original destination now. No lock, no allocation,
    // no wait; bytes still in the pipe may be lost, which is better than a hung crash report.
    void restore_now(int target) const {
        for (const Stream& s : streams_)
            if (s.target == target && s.saved >= 0) sys_dup2(s.saved, s.target);
    }

    bool installed() const { return installed_; }

private:
    struct Shared {
        std::mutex mu;
        std::condition_variable cv;
        bool stop = false;
        bool hold = false;
        size_t drained = 0;   // readers that saw their pipe empty (or EOF) after stop
        size_t exited = 0;   // readers that reached EOF or an error and returned
    };
    struct Stream {
        int target = -1;
        int pipe_rd = -1;
        int pipe_wr = -1;
        int saved = -1;
        bool redirected = false;
        std::thread reader;
    };

    // Waits up to `ms` for `fd` to be readable. 1 = readable (data or EOF), 0 = nothing yet,
    // -1 = the pipe is gone.
    static int wait_readable(int fd, int ms) {
#ifdef _WIN32
        const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
        DWORD avail = 0;
        if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr))
            return GetLastError() == ERROR_BROKEN_PIPE ? 1 : -1;   // broken = EOF: _read reports 0
        if (avail > 0) return 1;
        if (ms > 0) Sleep(static_cast<DWORD>(ms < 10 ? ms : 10));   // anonymous pipes cannot wait
        return 0;
#else
        pollfd p{fd, POLLIN, 0};
        for (;;) {
            const int r = ::poll(&p, 1, ms);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) return -1;
            return r > 0 ? 1 : 0;
        }
#endif
    }

    static void reader_main(std::shared_ptr<Shared> sh, LogRing* ring, int src, int forward) {
        std::string carry;
        char buf[4096];
        bool reported_drained = false;
        auto report = [&](bool exiting) {
            std::lock_guard<std::mutex> lock(sh->mu);
            if (!reported_drained) {
                reported_drained = true;
                sh->drained++;
            }
            if (exiting) sh->exited++;
            sh->cv.notify_all();
        };
        if (sh->hold) {
            std::unique_lock<std::mutex> lock(sh->mu);
            sh->cv.wait(lock, [&] { return sh->stop; });
        }
        for (;;) {
            bool stopping;
            {
                std::lock_guard<std::mutex> lock(sh->mu);
                stopping = sh->stop;
            }
            // Before stop: wake periodically to notice it. After the drain is reported: only a
            // child's late output or its EOF is left, so block in longer steps.
            const int ms = !stopping ? 50 : (reported_drained ? 200 : 0);
            const int ready = wait_readable(src, ms);
            if (ready < 0) break;
            if (ready == 0) {
                // Stop was requested BEFORE this empty check, and the targets were restored
                // before stop, so every byte written through them is already forwarded.
                if (stopping && !reported_drained) report(false);
                continue;
            }
            const long n = sys_read(src, buf, sizeof buf);
            if (n <= 0) break;   // EOF: every write end, ours and any child's, is closed
            long done = 0;
            while (done < n) {   // the original destination stays the primary record
                const long w = sys_write(forward, buf + done, n - done);
                if (w <= 0) break;
                done += w;
            }
            carry.append(buf, static_cast<size_t>(n));
            size_t pos;
            while ((pos = carry.find('\n')) != std::string::npos) {
                std::string line = carry.substr(0, pos);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                ring->push(line);
                carry.erase(0, pos + 1);
            }
            if (carry.size() > 65536) {   // output without newlines must not grow without bound
                ring->push(carry);
                carry.clear();
            }
        }
        if (!carry.empty()) ring->push(carry);
        sys_close(src);
        report(true);
    }

    static bool make_pipe(int fds[2]) {
#ifdef _WIN32
        // _O_NOINHERIT: a child must not inherit the extra pipe descriptors (fds 1/2 themselves
        // are inherited regardless, which is what the bounded teardown is for).
        return _pipe(fds, 4096, _O_BINARY | _O_NOINHERIT) == 0;
#else
        if (::pipe(fds) != 0) return false;
        ::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
        ::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
        return true;
#endif
    }
    static int dup_noinherit(int fd) {
#ifdef _WIN32
        const int d = _dup(fd);
        if (d >= 0)
            SetHandleInformation(reinterpret_cast<HANDLE>(_get_osfhandle(d)), HANDLE_FLAG_INHERIT,
                                 0);
        return d;
#else
        return ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
#endif
    }
#ifdef _WIN32
    static int sys_dup2(int a, int b) { return _dup2(a, b); }
    static int sys_close(int fd) { return _close(fd); }
    static long sys_read(int fd, char* buf, size_t n) {
        return _read(fd, buf, static_cast<unsigned>(n));
    }
    static long sys_write(int fd, const char* buf, long n) {
        return _write(fd, buf, static_cast<unsigned>(n));
    }
#else
    static int sys_dup2(int a, int b) {
        int r;
        do r = ::dup2(a, b);
        while (r < 0 && errno == EINTR);
        return r;
    }
    static int sys_close(int fd) { return ::close(fd); }
    static long sys_read(int fd, char* buf, size_t n) {
        long r;
        do r = static_cast<long>(::read(fd, buf, n));
        while (r < 0 && errno == EINTR);
        return r;
    }
    static long sys_write(int fd, const char* buf, long n) {
        long r;
        do r = static_cast<long>(::write(fd, buf, static_cast<size_t>(n)));
        while (r < 0 && errno == EINTR);
        return r;
    }
#endif

    bool installed_ = false;
    std::vector<Stream> streams_;
    std::shared_ptr<Shared> shared_;
};

}   // namespace prosper::frontend
