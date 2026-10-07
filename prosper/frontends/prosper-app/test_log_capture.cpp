// test_log_capture — the Game Log capture's teardown contract (log_capture.hpp). No SDL, no ImGui:
// each case captures a temp file's descriptor instead of stdout/stderr, so the bytes that reach
// the "console" can be read back exactly.
//
// The cases pin the three ways this capture has failed in review: bytes left in the pipe at
// teardown (lost output), a teardown that waits for an EOF a child process prevents (a hung game
// boot), and a half-failed install that leaves a descriptor pointing at a pipe nobody reads.
#include "log_capture.hpp"
#include "log_ring.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#ifdef _WIN32
#include <io.h>
#else
#include <csignal>
#include <ctime>
#include <sys/wait.h>
#include <unistd.h>
#endif

using prosper::frontend::LogCapture;
using prosper::frontend::LogRing;
using Clock = std::chrono::steady_clock;

namespace {

#ifdef _WIN32
int fd_of(FILE* f) {
    return _fileno(f);
}
int dup_fd(int fd) {
    return _dup(fd);
}
int close_fd(int fd) {
    return _close(fd);
}
long long seek_fd(int fd, long long off, int whence) {
    return _lseeki64(fd, off, whence);
}
long read_fd(int fd, char* buf, size_t n) {
    return _read(fd, buf, static_cast<unsigned>(n));
}
void write_all(int fd, const std::string& s) {
    size_t done = 0;
    while (done < s.size()) {
        const int w = _write(fd, s.data() + done, static_cast<unsigned>(s.size() - done));
        ASSERT_GT(w, 0);
        done += static_cast<size_t>(w);
    }
}
#else
int fd_of(FILE* f) {
    return fileno(f);
}
int dup_fd(int fd) {
    return ::dup(fd);
}
int close_fd(int fd) {
    return ::close(fd);
}
long long seek_fd(int fd, long long off, int whence) {
    return ::lseek(fd, off, whence);
}
long read_fd(int fd, char* buf, size_t n) {
    return static_cast<long>(::read(fd, buf, n));
}
void write_all(int fd, const std::string& s) {
    size_t done = 0;
    while (done < s.size()) {
        const ssize_t w = ::write(fd, s.data() + done, s.size() - done);
        ASSERT_GT(w, 0);
        done += static_cast<size_t>(w);
    }
}
#endif

// Reads the whole file through `probe`, a dup taken BEFORE install: it shares the file's open
// description but is not a captured descriptor, so it still refers to the file while the captured
// one points at a pipe. The shared offset is put back so later writes still append.
std::string read_back(int probe) {
    const long long at = seek_fd(probe, 0, SEEK_CUR);
    EXPECT_GE(at, 0);
    EXPECT_EQ(seek_fd(probe, 0, SEEK_SET), 0);
    std::string out;
    char buf[4096];
    long n;
    while ((n = read_fd(probe, buf, sizeof buf)) > 0) out.append(buf, static_cast<size_t>(n));
    seek_fd(probe, at, SEEK_SET);
    return out;
}

// Small enough to sit in a pipe whole (Windows' is 4 KiB), with lines the ring can count.
std::string payload() {
    std::string s;
    for (int i = 0; i < 40; i++) s += "game log line " + std::to_string(i) + "\n";
    return s;
}

// A ring a detached reader may still use after the test body ends: never freed.
LogRing* leaked_ring() {
    return new LogRing();
}

} // namespace

// (a) Bytes still in the pipe when teardown starts are forwarded before uninstall() returns. The
// readers are held until stop, so ONLY the bounded drain can deliver them.
TEST(LogCapture, TeardownForwardsEveryByteStillInThePipe) {
    FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    const int target = fd_of(f);
    const int probe = dup_fd(target);
    ASSERT_GE(probe, 0);
    LogRing* ring = leaked_ring();
    LogCapture cap;
    LogCapture::Options opts;
    opts.hold_readers_until_stop = true;
    ASSERT_TRUE(cap.install(ring, {target}, opts));
    const std::string bytes = payload();
    write_all(target, bytes);
    EXPECT_EQ(read_back(probe), "") << "the readers were held; nothing may have been forwarded yet";

    EXPECT_TRUE(cap.uninstall(std::chrono::milliseconds(2000)))
        << "with no outside holder every reader reaches EOF and is joined";
    EXPECT_FALSE(cap.installed());
    EXPECT_EQ(read_back(probe), bytes);
    const auto lines = ring->snapshot();
    ASSERT_EQ(lines.size(), 40u);
    EXPECT_EQ(lines.front(), "game log line 0");
    EXPECT_EQ(lines.back(), "game log line 39");
    // The target is the file again: a write now lands directly, with no reader involved.
    write_all(target, "after\n");
    EXPECT_EQ(read_back(probe), bytes + "after\n");
    close_fd(probe);
    fclose(f);
}

// (b) A second reference to the pipe's write end (what a child inherits as fd 1/2) prevents EOF.
// Teardown must still return within its bound, having forwarded what was already written. The
// watchdog releases the holder after 3 s so a regressed, unbounded teardown fails on the elapsed
// time instead of hanging the suite.
TEST(LogCapture, TeardownIsBoundedWhileAnotherDescriptorHoldsTheWriteEnd) {
    FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    const int target = fd_of(f);
    const int probe = dup_fd(target);
    ASSERT_GE(probe, 0);
    LogCapture cap;
    ASSERT_TRUE(cap.install(leaked_ring(), {target}));
    const int held = dup_fd(target);   // a write end of the capture pipe, outside LogCapture
    ASSERT_GE(held, 0);
    const std::string bytes = payload();
    write_all(target, bytes);

    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    std::thread watchdog([&] {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait_for(lock, std::chrono::seconds(3), [&] { return done; });
        close_fd(held);
    });
    const auto t0 = Clock::now();
    const bool joined = cap.uninstall(std::chrono::milliseconds(200));
    const auto elapsed = Clock::now() - t0;
    {
        std::lock_guard<std::mutex> lock(mu);
        done = true;
    }
    cv.notify_all();
    watchdog.join();

    EXPECT_LT(elapsed, std::chrono::milliseconds(1500))
        << "teardown waited for an EOF that only the outside holder could deliver";
    EXPECT_FALSE(joined) << "the reader could not reach EOF, so it must have been left running";
    EXPECT_EQ(read_back(probe), bytes) << "the bounded drain must still forward what was written";
    close_fd(probe);
    fclose(f);
}

#ifndef _WIN32
// (b), as the app meets it: a child process (xdg-open, a file manager) inherited the captured fd.
TEST(LogCapture, TeardownIsBoundedWhileAChildProcessHoldsTheWriteEnd) {
    FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    const int target = fd_of(f);
    const int probe = dup_fd(target);
    ASSERT_GE(probe, 0);
    LogCapture cap;
    ASSERT_TRUE(cap.install(leaked_ring(), {target}));
    const std::string bytes = payload();
    write_all(target, bytes);

    const pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        // Holds every inherited descriptor, including `target` (the pipe's write end). Only
        // async-signal-safe calls after fork in a threaded parent.
        timespec ts{5, 0};
        while (nanosleep(&ts, &ts) != 0) {}
        _exit(0);
    }
    const auto t0 = Clock::now();
    const bool joined = cap.uninstall(std::chrono::milliseconds(200));
    const auto elapsed = Clock::now() - t0;
    kill(child, SIGKILL);
    int status = 0;
    waitpid(child, &status, 0);

    EXPECT_LT(elapsed, std::chrono::milliseconds(1500))
        << "teardown waited for the child to close its inherited write end";
    EXPECT_FALSE(joined);
    EXPECT_EQ(read_back(probe), bytes);
    close_fd(probe);
    fclose(f);
}
#endif

// A failed install leaves every target exactly as it was: the good target must not be left
// pointing at a pipe with no reader (the writer would block once the pipe fills).
TEST(LogCapture, FailedInstallLeavesEveryTargetUntouched) {
    FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    const int good = fd_of(f);
    const int probe = dup_fd(good);
    ASSERT_GE(probe, 0);
    // A descriptor number that can never be open. (Not "open one and close it": install's own
    // pipe() reuses the lowest free numbers, so a just-closed number becomes valid again.)
#ifdef _WIN32
    // The CRT reports a bad descriptor through its invalid-parameter handler, which terminates
    // by default; a no-op handler makes _dup return -1/EBADF as POSIX dup does.
    const _invalid_parameter_handler previous = _set_invalid_parameter_handler(
        [](const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {});
    const int bad = 9999;
#else
    const int bad = INT_MAX;
#endif

    LogCapture cap;
    EXPECT_FALSE(cap.install(leaked_ring(), {good, bad}));
#ifdef _WIN32
    _set_invalid_parameter_handler(previous);
#endif
    EXPECT_FALSE(cap.installed());
    write_all(good, "direct\n");
    EXPECT_EQ(read_back(probe), "direct\n") << "the good target was left redirected into a pipe";
    EXPECT_TRUE(cap.uninstall(std::chrono::milliseconds(100)))
        << "uninstall without install is a no-op";
    close_fd(probe);
    fclose(f);
}
