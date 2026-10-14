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
#include <cerrno>
#include <ctime>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
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
// Redirect only this test's standard streams to synthetic files. A stdin gate holds the shell
// child until capture teardown has finished; no timing guess or application launch is involved.
class ChildOutputCapture : public testing::Test {
protected:
    void SetUp() override {
        for (int fd = 0; fd < 3; ++fd) {
            saved_[fd] = ::fcntl(fd, F_DUPFD_CLOEXEC, 3);
            ASSERT_GE(saved_[fd], 0);
        }
        out_ = std::tmpfile();
        err_ = std::tmpfile();
        ASSERT_NE(out_, nullptr);
        ASSERT_NE(err_, nullptr);
        ASSERT_EQ(::pipe(gate_), 0);
        for (int fd : {gate_[0], gate_[1], fd_of(out_), fd_of(err_)})
            ASSERT_EQ(::fcntl(fd, F_SETFD, FD_CLOEXEC), 0);
        ASSERT_EQ(::dup2(gate_[0], 0), 0);
        ASSERT_EQ(::dup2(fd_of(out_), 1), 1);
        ASSERT_EQ(::dup2(fd_of(err_), 2), 2);
        LogCapture::Options opts;
        opts.hold_readers_until_stop = true;
        ASSERT_TRUE(capture_.install(ring_, {1, 2}, opts));
    }

    void TearDown() override {
        if (child_ > 0) {
            ::kill(child_, SIGKILL);   // only the synthetic child created by this test
            while (::waitpid(child_, nullptr, 0) < 0 && errno == EINTR) {}
        }
        capture_.uninstall(std::chrono::milliseconds(2000));
        for (int fd = 0; fd < 3; ++fd) {
            if (saved_[fd] < 0) continue;
            ::dup2(saved_[fd], fd);
            ::close(saved_[fd]);
        }
        for (int fd : gate_)
            if (fd >= 0) ::close(fd);
        if (HasFailure()) {
            if (out_) std::fprintf(stderr, "captured stdout:\n%s", read_back(fd_of(out_)).c_str());
            if (err_) std::fprintf(stderr, "captured stderr:\n%s", read_back(fd_of(err_)).c_str());
        }
        if (out_) std::fclose(out_);
        if (err_) std::fclose(err_);
    }

    LogCapture capture_;
    LogRing* ring_ = leaked_ring();
    FILE* out_ = nullptr;
    FILE* err_ = nullptr;
    int saved_[3] = {-1, -1, -1};
    int gate_[2] = {-1, -1};
    pid_t child_ = -1;
};

TEST_F(ChildOutputCapture, ChildWritesAfterLibraryCaptureTeardown) {
    std::string exe = "/bin/sh";
    std::string flag = "-c";
    std::string script = "printf 'child ready\\n'; read token || exit 9; printf 'child stdout\\n'; "
                         "printf 'child stderr\\n' >&2";
    char* args[] = {exe.data(), flag.data(), script.data(), nullptr};
    ASSERT_EQ(capture_.spawn_child(child_, args, environ), 0);
    ASSERT_GT(child_, 0);
    EXPECT_TRUE(capture_.installed()) << "spawning must retain the library's capture";
    // Spawn completion does not synchronize the test with the shell's startup. Wait for its
    // actual script before requiring EOF: transient inherited write ends may still be closing.
    // pread observes readiness without moving the file offset shared with the child's stdout.
    const auto child_ready = [&] {
        constexpr char marker[] = "child ready\n";
        char seen[sizeof(marker) - 1]{};
        return ::pread(fd_of(out_), seen, sizeof(seen), 0) == static_cast<ssize_t>(sizeof(seen)) &&
               std::string(seen, sizeof(seen)) == marker;
    };
    const auto ready_deadline = Clock::now() + std::chrono::seconds(3);
    while (!child_ready() && Clock::now() < ready_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_TRUE(child_ready()) << "the child must reach its stdin gate before capture teardown";
    write_all(1, "library stdout\n");
    write_all(2, "library stderr\n");
    EXPECT_EQ(read_back(fd_of(out_)), "child ready\n")
        << "the library still writes through held readers";
    EXPECT_EQ(read_back(fd_of(err_)), "");

    EXPECT_TRUE(capture_.uninstall(std::chrono::milliseconds(2000)))
        << "the waiting game must hold no write end of the library's output pipes";
    EXPECT_EQ(read_back(fd_of(out_)), "child ready\nlibrary stdout\n");
    EXPECT_EQ(read_back(fd_of(err_)), "library stderr\n");
    write_all(gate_[1], "go\n");
    int status = 0;
    pid_t reaped = 0;
    const auto deadline = Clock::now() + std::chrono::seconds(3);
    do {
        reaped = ::waitpid(child_, &status, WNOHANG);
        if (reaped != 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (Clock::now() < deadline);
    ASSERT_EQ(reaped, child_) << "the released synthetic child must finish";
    child_ = -1;
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    EXPECT_EQ(read_back(fd_of(out_)), "child ready\nlibrary stdout\nchild stdout\n");
    EXPECT_EQ(read_back(fd_of(err_)), "library stderr\nchild stderr\n");
    EXPECT_EQ(ring_->snapshot().size(), 2u) << "child output must bypass the library's log ring";
}

TEST(LogCapture, SpawnFailureDoesNotStartAChildOrUninstallCapture) {
    FILE* f = std::tmpfile();
    ASSERT_NE(f, nullptr);
    LogCapture capture;
    ASSERT_TRUE(capture.install(leaked_ring(), {fd_of(f)}));
    std::string exe = "/prosper-test-nonexistent/executable";
    char* args[] = {exe.data(), nullptr};
    pid_t child = -1;
    EXPECT_EQ(capture.spawn_child(child, args, environ), ENOENT);
    EXPECT_EQ(child, -1);
    EXPECT_TRUE(capture.installed());
    EXPECT_EQ(capture.spawn_child(child, nullptr, environ), EINVAL);
    EXPECT_EQ(capture.spawn_console_actions(nullptr), EINVAL);
    capture.uninstall(std::chrono::milliseconds(2000));
    fclose(f);
}

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

#ifndef _WIN32
namespace {

// Spawn `/bin/sh -c <script>`; a null `actions` inherits every descriptor as-is. The child
// gets an empty environment: echo needs none, and this avoids any platform's `environ` plumbing.
pid_t spawn_shell(const char* script, posix_spawn_file_actions_t* actions) {
    char* const argv[] = {const_cast<char*>("sh"), const_cast<char*>("-c"),
                          const_cast<char*>(script), nullptr};
    char* const envp[] = {nullptr};
    pid_t child = -1;
    const int rc = ::posix_spawn(&child, "/bin/sh", actions, nullptr, argv, envp);
    EXPECT_EQ(rc, 0) << "posix_spawn of /bin/sh failed, cannot exercise the child";
    return child;
}

bool wait_exited_zero(pid_t child) {
    int status = 0;
    while (::waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}   // namespace

// Positive control for the test below: a child whose stdout is a pipe with no readers dies of
// SIGPIPE on its first write. If this ever stops reporting the death, the regression test's
// green no longer means anything — it would pass with a detector that cannot see the failure.
TEST(LogCapture, ChildOnAReaderlessPipeDiesOfSigpipe) {
    int doomed[2] = {-1, -1};
    ASSERT_EQ(::pipe(doomed), 0);
    ASSERT_EQ(::close(doomed[0]), 0);   // no readers from here on
    posix_spawn_file_actions_t actions;
    ASSERT_EQ(::posix_spawn_file_actions_init(&actions), 0);
    ASSERT_EQ(::posix_spawn_file_actions_adddup2(&actions, doomed[1], 1), 0);
    const pid_t child = spawn_shell("echo MARK_DOOMED", &actions);
    ::posix_spawn_file_actions_destroy(&actions);
    ::close(doomed[1]);
    ASSERT_GE(child, 0);
    int status = 0;
    while (::waitpid(child, &status, 0) < 0) ASSERT_EQ(errno, EINTR);
    // A shell whose builtin echo dies of SIGPIPE either dies by the signal itself or exits
    // 128+SIGPIPE, depending on whether echo ran in-process or forked.
    const bool signaled = WIFSIGNALED(status) && WTERMSIG(status) == SIGPIPE;
    const bool reported = WIFEXITED(status) && WEXITSTATUS(status) == 128 + SIGPIPE;
    EXPECT_TRUE(signaled || reported) << "wait status " << status << " shows neither death";
}

// A game the library launches may outlive the library, so spawn_console_actions must route its
// streams to the pre-capture destination rather than the capture pipes. Routing is observable
// without killing any reader: pipe-routed bytes pass through the ring, console-routed bytes
// bypass it. Without the actions both markers would reach the ring and this would be red.
//
// Deliberately no ASSERT past the fd 1 redirect below: an early return would strand fd 1 on the
// console file and swallow every later failure's text, so everything after it is EXPECT, and fd 1
// is handed back before any assertion runs.
TEST(LogCapture, SpawnConsoleActionsBypassTheCapturePipes) {
    FILE* console = std::tmpfile();   // "the console": what fd 1 meant before capture
    ASSERT_NE(console, nullptr);
    const int probe = dup_fd(fd_of(console));
    ASSERT_GE(probe, 0);
    const int real_stdout = dup_fd(1);
    ASSERT_GE(real_stdout, 0);
    ASSERT_EQ(::dup2(fd_of(console), 1), 1);

    LogRing* ring = leaked_ring();
    LogCapture cap;
    const bool installed = cap.install(ring, {1});
    EXPECT_TRUE(installed);
    posix_spawn_file_actions_t actions;
    bool have_actions = false;
    if (installed) {
        have_actions = cap.spawn_console_actions(&actions) == 0;
        EXPECT_TRUE(have_actions);
    }
    pid_t fixed = -1, plain = -1;
    if (have_actions) fixed = spawn_shell("echo MARK_FIX", &actions);
    plain = spawn_shell("echo MARK_CTRL", nullptr);
    if (have_actions) ::posix_spawn_file_actions_destroy(&actions);
    if (fixed >= 0) EXPECT_TRUE(wait_exited_zero(fixed));
    if (plain >= 0) EXPECT_TRUE(wait_exited_zero(plain));
    // Joined (nothing holds a write end): every byte is delivered before this returns.
    if (installed) EXPECT_TRUE(cap.uninstall(std::chrono::milliseconds(2000)));
    // fd 1 still points at the console tmpfile (uninstall restores the saved destination, which
    // IS that file here): hand it back first, so failure text below reaches the real stdout.
    EXPECT_EQ(::dup2(real_stdout, 1), 1);
    EXPECT_EQ(close_fd(real_stdout), 0);
    const std::string text = read_back(probe);
    EXPECT_NE(text.find("MARK_FIX"), std::string::npos) << "the fixed child never reached fd 1";
    EXPECT_NE(text.find("MARK_CTRL"), std::string::npos) << "the plain child never reached fd 1";
    std::string logged;
    for (const auto& line : ring->snapshot()) {
        logged += line;
        logged += '\n';
    }
    EXPECT_EQ(logged.find("MARK_FIX"), std::string::npos)
        << "the fixed child's bytes passed through the pipe instead of the console";
    EXPECT_NE(logged.find("MARK_CTRL"), std::string::npos)
        << "the plain child's bytes never reached the capture";
    EXPECT_EQ(close_fd(probe), 0);
    EXPECT_EQ(std::fclose(console), 0);
}
#endif
