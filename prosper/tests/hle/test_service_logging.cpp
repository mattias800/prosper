// test_service_logging — PROSPER_SVCLOG must not fault on, or lie about, an argument it cannot read.
//
// The service logger prints its arguments through P(), which is the identity map, so a caller that
// hands it a pointer-shaped value it cannot dereference used to fault inside the logger. The three
// arms below are the shapes that make that reachable: a reserved-but-uncommitted page, a committed
// PROT_NOACCESS page, and a guard page. The fourth arm is the review finding -- the mapping can
// disappear between the readability probe and the read -- so the snapshot is driven against a page
// being torn down and remapped under it.
#include "fixtures/test_scratch.h"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#ifdef _WIN32
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
constexpr uint64_t kLoggedWord = 0x1122334455667788ull;

// The three deliberately supplied argument-2 values are pointer-shaped but cannot safely be
// dereferenced for every requested word.
bool call_open(prosper::HleFn open, void* pointer) {
    // sceImeKeyboardOpen now consumes its second argument as a real keyboard parameter. Keep that
    // ABI input valid while argument 2 remains the deliberately unreadable value exercised by the
    // service logger below.
    uint64_t keyboard_param[4]{};
    return open(1, reinterpret_cast<uintptr_t>(keyboard_param),
                reinterpret_cast<uintptr_t>(pointer), 1, 0x83, 0) == 0;
}

// The pages the arms need. Two are unreadable by construction, one carries a value that sits in the
// last 8 bytes before an unreadable page (so a 16-byte read walks off the end), and one is a guard
// page (so a read faults on the access itself).
struct ServiceLogPages {
#ifdef _WIN32
    static constexpr size_t kPageSize = 0x1000;
    void* reserved = VirtualAlloc(nullptr, kPageSize, MEM_RESERVE, PAGE_NOACCESS);
    void* protected_page =
        VirtualAlloc(nullptr, kPageSize, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    void* boundary = VirtualAlloc(nullptr, kPageSize * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    void* guard_page = VirtualAlloc(nullptr, kPageSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    ServiceLogPages() {
        if (!reserved || !protected_page || !boundary || !guard_page) return;
        *reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(boundary) + kPageSize - 8) = kLoggedWord;
        DWORD old_protect = 0;
        if (!VirtualProtect(static_cast<uint8_t*>(boundary) + kPageSize, kPageSize, PAGE_NOACCESS,
                            &old_protect))
            return;
        if (!VirtualProtect(guard_page, kPageSize, PAGE_READWRITE | PAGE_GUARD, &old_protect))
            return;
        ready = true;
    }
    ~ServiceLogPages() {
        VirtualFree(reserved, 0, MEM_RELEASE);
        VirtualFree(protected_page, 0, MEM_RELEASE);
        VirtualFree(guard_page, 0, MEM_RELEASE);
        VirtualFree(boundary, 0, MEM_RELEASE);
    }
    void* boundary_value() const { return static_cast<uint8_t*>(boundary) + kPageSize - 8; }
#else
    size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    void* reserved = MAP_FAILED;
    void* protected_page = MAP_FAILED;
    void* boundary = MAP_FAILED;
    void* guard_page = MAP_FAILED;
    ServiceLogPages() {
        reserved = mmap(nullptr, page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        protected_page = mmap(nullptr, page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        boundary = mmap(nullptr, page_size * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                        -1, 0);
        if (reserved == MAP_FAILED || protected_page == MAP_FAILED || boundary == MAP_FAILED)
            return;
        *reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(boundary) + page_size - 8) = kLoggedWord;
        if (mprotect(static_cast<uint8_t*>(boundary) + page_size, page_size, PROT_NONE) != 0)
            return;
        guard_page = protected_page;   // PROT_NONE already faults the access, as a guard page would
        ready = true;
    }
    ~ServiceLogPages() {
        if (reserved != MAP_FAILED) munmap(reserved, page_size);
        if (protected_page != MAP_FAILED) munmap(protected_page, page_size);
        if (boundary != MAP_FAILED) munmap(boundary, page_size * 2);
    }
    void* boundary_value() const { return static_cast<uint8_t*>(boundary) + page_size - 8; }
#endif
    // Set only as the constructor's last statement: any early return (a failed allocation OR a
    // failed protect) leaves the fixture not ready, so a case cannot pass without its guard pages.
    bool ready = false;
    bool complete() const { return ready; }
};

prosper::HleFn ime_keyboard_open() {
#ifdef _WIN32
    _putenv_s("PROSPER_SVCLOG", "1");
#else
    setenv("PROSPER_SVCLOG", "1", 1);
#endif
    prosper::register_builtin_hle();
    return prosper::Hle::lookup("eaFXjfJv3xs");   // sceImeKeyboardOpen
}

// Call open with a pointer that reads correctly, capturing stderr, and require the logger to have
// printed BOTH the raw argument list and the full 64-bit word -- a truncated read is the defect
// this exists for, and "logged something" would not see it.
bool logs_full_values(prosper::HleFn open, void* pointer) {
    const std::string capture_path = prosper_test::test_scratch_file("service_log_capture.txt");
    FILE* capture = std::fopen(capture_path.c_str(), "w+b");
    if (!capture) return false;
    if (std::fflush(stderr) != 0) {
        std::fclose(capture);
        return false;
    }

#ifdef _WIN32
    const int stderr_fd = _fileno(stderr);
    const int saved_stderr = _dup(stderr_fd);
    const bool redirected = saved_stderr >= 0 && _dup2(_fileno(capture), stderr_fd) == 0;
#else
    const int stderr_fd = fileno(stderr);
    const int saved_stderr = dup(stderr_fd);
    const bool redirected = saved_stderr >= 0 && dup2(fileno(capture), stderr_fd) >= 0;
#endif
    if (!redirected) {
        if (saved_stderr >= 0) {
#ifdef _WIN32
            _close(saved_stderr);
#else
            close(saved_stderr);
#endif
        }
        std::fclose(capture);
        return false;
    }

    uint64_t keyboard_param[4]{};
    const bool called = open(1, reinterpret_cast<uintptr_t>(keyboard_param),
                             reinterpret_cast<uintptr_t>(pointer), 1, 0x83, kLoggedWord) == 0;
    const bool flushed = std::fflush(stderr) == 0;
#ifdef _WIN32
    const bool restored = _dup2(saved_stderr, stderr_fd) == 0;
    _close(saved_stderr);
#else
    const bool restored = dup2(saved_stderr, stderr_fd) >= 0;
    close(saved_stderr);
#endif

    char output[512]{};
    std::rewind(capture);
    const size_t bytes = std::fread(output, 1, sizeof(output) - 1, capture);
    std::fclose(capture);
    std::remove(capture_path.c_str());
    output[bytes] = '\0';
    const bool full_argument = std::strstr(output, ", 0x1122334455667788)") != nullptr;
    const bool full_word = std::strstr(output, "[svc]   a2 -> 1122334455667788") != nullptr;
    if (!full_argument || !full_word) std::fprintf(stderr, "captured service log:\n%s", output);
    return called && flushed && restored && full_argument && full_word;
}
}   // namespace

TEST(ServiceLogging, ImeKeyboardOpenIsRegistered) {
    EXPECT_NE(ime_keyboard_open(), nullptr) << "sceImeKeyboardOpen was not registered";
}

TEST(ServiceLogging, UnreadableArgumentValuesAreAcceptedRatherThanRead) {
    const prosper::HleFn open = ime_keyboard_open();
    ASSERT_NE(open, nullptr);
    ServiceLogPages pages;
    ASSERT_TRUE(pages.complete()) << "could not reserve the unreadable test pages";
    EXPECT_TRUE(call_open(open, pages.reserved)) << "a reserved-only page is accepted";
    EXPECT_TRUE(call_open(open, pages.protected_page))
        << "a committed PROT_NOACCESS page is accepted";
    EXPECT_TRUE(call_open(open, pages.guard_page)) << "a guard page is accepted";
}

TEST(ServiceLogging, AReadableValueIsLoggedInFull) {
    const prosper::HleFn open = ime_keyboard_open();
    ASSERT_NE(open, nullptr);
    ServiceLogPages pages;
    ASSERT_TRUE(pages.complete()) << "could not reserve the unreadable test pages";
    EXPECT_TRUE(logs_full_values(open, pages.boundary_value()))
        << "the logger printed neither the full argument list nor the full 64-bit word";
}

TEST(ServiceLogging, AMappingMayDisappearWhileTheLoggerSnapshotsIt) {
    // The review finding: the mapping can disappear concurrently with the snapshot, so the logger
    // must tolerate it for a whole run of calls against a page that is repeatedly torn down.
    const prosper::HleFn open = ime_keyboard_open();
    ASSERT_NE(open, nullptr);
    ServiceLogPages pages;
    ASSERT_TRUE(pages.complete()) << "could not reserve the unreadable test pages";

    std::atomic<bool> run{true};
    std::atomic<bool> race_failed{false};
    std::atomic<unsigned> changes{0};
    std::thread toggler([&] {
        while (run.load(std::memory_order_relaxed)) {
#ifdef _WIN32
            const bool unmapped =
                VirtualFree(pages.boundary, pages.kPageSize, MEM_DECOMMIT) != FALSE;
            const bool remapped = VirtualAlloc(pages.boundary, pages.kPageSize, MEM_COMMIT,
                                               PAGE_READWRITE) == pages.boundary;
#else
            const bool unmapped =
                mmap(pages.boundary, pages.page_size, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == pages.boundary;
            const bool remapped =
                mmap(pages.boundary, pages.page_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == pages.boundary;
#endif
            if (!unmapped || !remapped) {
                race_failed.store(true, std::memory_order_release);
                break;
            }
            changes.fetch_add(1, std::memory_order_release);
        }
    });
    while (changes.load(std::memory_order_acquire) == 0 &&
           !race_failed.load(std::memory_order_acquire))
        std::this_thread::yield();

    bool ok = true;
    for (int i = 0; i < 1000 && ok; ++i) ok = call_open(open, pages.boundary);
    run.store(false, std::memory_order_relaxed);
    toggler.join();
    EXPECT_TRUE(ok) << "the logger faulted or refused while the mapping was being replaced";
    EXPECT_FALSE(race_failed.load(std::memory_order_relaxed))
        << "the page could not be torn down and remapped, so this arm proved nothing";
    EXPECT_GT(changes.load(std::memory_order_relaxed), 0u)
        << "the mapping never changed, so this arm proved nothing";
}