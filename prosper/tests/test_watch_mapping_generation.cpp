// test_watch_mapping_generation — a software watchpoint must publish its protection changes (#2393).
//
// prosper's Linux fault handler hosts several in-process debug instruments (the same class of thing
// gdb or Wine's relay debugging provides, pointed at this emulator's own process). Two of them --
// PROSPER_WATCH_LABEL/_ABS/_HOT and PROSPER_WATCH_COMPANION -- work by mprotect-ing a live guest
// page down to PROT_READ and single-stepping over each write.
//
// gpu_executor.cpp's guest_readable/guest_writable caches (#2387/#2389) memoize POSITIVE permission
// answers and drop them only when host::guest_mapping_generation() moves. So a watchpoint that
// flips a page to read-only without advancing that counter leaves guest_writable answering "yes,
// writable" for a page that is currently read-only, from cache, until something unrelated moves the
// generation. That is a stale positive on a WRITE permission check, which is the silent kind: the
// caller writes where it should have refused.
//
// STRUCTURE. The two arms below differ in ONE variable -- whether the protection change advanced
// the generation -- and are otherwise the same page, the same cache and the same query:
//
//   Arm A (control, and the pre-fix behavior reproduced in-process): revoke write permission with a
//          bare mprotect, deliberately not notifying. guest_writable must still answer TRUE. This is
//          the defect, and it doubles as the test's own warmth precondition: an OS probe of an
//          `r--p` page cannot return true, so a true here PROVES the answer came from the memo. If
//          the cache were disabled or never populated this arm fails, and the test cannot reach the
//          measurement below in a state where it would pass vacuously.
//
//   Arm B (the fix): revoke write permission through the PRODUCTION arm path,
//          prosper_arm_label_watch_force(), which routes its mprotect through watch_mprotect() and
//          therefore advances the generation. guest_writable must now answer FALSE.
//
// Revert the notify in watch_mprotect() and arm B produces arm A's `true` and fails. The mutation
// arm is thus structural rather than something a reader has to take on trust.
//
// Arm C is a wiring check over the production source, in the shape #1932/#2576 established: the
// live arms can only exercise the call sites they happen to reach, and the handler has thirteen.
// It asserts that EVERY protection change in exec_image_linux.cpp routes through the notifying
// wrapper, and that the notifier itself is still lock-free -- a "helpful" change that made it also
// maintain the mapping registry would take a mutex and reintroduce the signal-handler deadlock that
// is the reason this entry point exists at all.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(__linux__)

#include <sys/mman.h>
#include <unistd.h>

#include "gpu/gpu_execute.hpp"
#include "host/guest_memory_map.hpp"

// Armed by the AGC fence builder in production; called directly here because the arm's mprotect is
// one of the thirteen protection changes the invariant covers, and the only one reachable without
// installing the fault handler and taking a real SIGSEGV.
extern "C" void prosper_arm_label_watch_force(uint64_t addr);

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::fprintf(stderr, "%s %s\n", ok ? "[ ok ]" : "[FAIL]", what);
    if (!ok) ++failures;
}

void test_live_arms() {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    if (page != 0x1000) {
        // The watch arms a hard-coded 0x1000 span. Rather than silently testing something else,
        // fail loudly: a skip here would be indistinguishable from a pass.
        std::fprintf(stderr, "[FAIL] host page size is %zu, not 0x1000; the watch's fixed 0x1000 "
                             "arm span does not match this platform\n", page);
        ++failures;
        return;
    }

    void* p = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(p != MAP_FAILED, "mapped a read-write page to watch");
    if (p == MAP_FAILED) return;
    const uint64_t addr = (uint64_t)(uintptr_t)p;

    check(prosper::gpu::guest_writable(addr, 8), "the page is writable (and now cached)");

    // ---- Arm A: protection revoked WITHOUT advancing the generation --------------------------
    // Bare mprotect on purpose. This is exactly what the handler did before this fix.
    check(mprotect(p, page, PROT_READ) == 0, "revoked write permission with a bare mprotect");
    check(prosper::gpu::guest_writable(addr, 8),
          "PRECONDITION ESTABLISHED, NOT ASSUMED: an unpublished revocation still answers "
          "writable, which is only possible from a warm cache (an r--p probe cannot)");

    // Restore, and clear the artificial state so arm B starts from a genuinely warm cache over a
    // genuinely writable page rather than inheriting arm A's staleness.
    check(mprotect(p, page, PROT_READ | PROT_WRITE) == 0, "restored write permission");
    prosper::host::detail::advance_guest_mapping_generation();
    check(prosper::gpu::guest_writable(addr, 8), "writable again, re-probed and re-cached");

    // ---- Arm B: protection revoked THROUGH the production arm path ---------------------------
    const uint64_t before = prosper::host::guest_mapping_generation();
    prosper_arm_label_watch_force(addr);
    check(prosper::host::guest_mapping_generation() != before,
          "arming the watch advanced the guest mapping generation");
    check(!prosper::gpu::guest_writable(addr, 8),
          "THE FIX: a page the watch has made read-only is NOT reported writable");

    // The invalidation must not overshoot into a wrong negative: the page is still readable, and a
    // watchpoint that made guest_readable answer false would break the guarded derefs that probe
    // exists for.
    check(prosper::gpu::guest_readable(addr, 8), "the watched page is still readable");

    mprotect(p, page, PROT_READ | PROT_WRITE);
    prosper::host::notify_guest_page_protection_changed();
    munmap(p, page);
}

bool read_file(const char* path, std::string& out) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char buffer[8192];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof buffer, f)) > 0) out.append(buffer, got);
    std::fclose(f);
    return true;
}

// Count occurrences of `needle` that are NOT already prefixed by `prefix`.
size_t count_unwrapped(const std::string& text, const char* needle, const char* prefix) {
    const size_t plen = std::strlen(prefix);
    size_t found = 0;
    for (size_t at = text.find(needle); at != std::string::npos;
         at = text.find(needle, at + 1)) {
        if (at >= plen && text.compare(at - plen, plen, prefix) == 0) continue;
        ++found;
    }
    return found;
}

// One bare call may remain per wrapper: the one inside the wrapper's own body.
void check_wrapped(const std::string& text, const char* needle, const char* prefix,
                   const char* what) {
    const size_t bare = count_unwrapped(text, needle, prefix);
    if (bare != 1)
        std::fprintf(stderr, "  (found %zu unwrapped `%s` call(s); expected exactly 1 -- the one "
                             "inside the wrapper's own body)\n", bare, needle);
    check(bare == 1, what);
}

void test_production_wiring() {
    // Both paths are supplied by the build (CMAKE_CURRENT_SOURCE_DIR) and deliberately never
    // printed: they are absolute paths on whoever's machine ran the build.
#if defined(PROSPER_EXEC_IMAGE_LINUX_SOURCE) && defined(PROSPER_GUEST_MEMORY_MAP_SOURCE) && \
    defined(PROSPER_GUEST_WRITE_WATCH_SOURCE)
    std::string handler;
    if (!read_file(PROSPER_EXEC_IMAGE_LINUX_SOURCE, handler)) {
        std::fprintf(stderr, "[FAIL] cannot read the Linux fault-handler source configured at "
                             "build time, so the #2393 call sites could not be confirmed\n");
        ++failures;
        return;
    }
    check_wrapped(handler, "mprotect(", "watch_",
                  "wiring: every protection change in the Linux fault handler routes through "
                  "watch_mprotect()");
    check(handler.find("const int rc = mprotect(addr, len, prot);\n"
                       "        prosper::host::notify_guest_page_protection_changed();")
              != std::string::npos,
          "wiring: the fault handler's watch_mprotect() advances the generation after the mprotect");

    // The same invariant, the same remedy, in the file that arms the production write-watch. Both
    // of its arms are checked from here even though only the POSIX one compiles on Linux -- a
    // source scan is the only check that can see the Windows arm from this build at all, and an
    // unpublished VirtualProtect there is the identical defect.
    std::string watch;
    if (!read_file(PROSPER_GUEST_WRITE_WATCH_SOURCE, watch)) {
        std::fprintf(stderr, "[FAIL] cannot read guest_write_watch.cpp configured at build time\n");
        ++failures;
        return;
    }
    check_wrapped(watch, "mprotect(", "watch_",
                  "wiring: every protection change in guest_write_watch.cpp's POSIX arm routes "
                  "through watch_mprotect()");
    check_wrapped(watch, "VirtualProtect(", "watch_virtual_",
                  "wiring: every protection change in guest_write_watch.cpp's Windows arm routes "
                  "through watch_virtual_protect()");
    check(watch.find("notify_guest_page_protection_changed();") != std::string::npos,
          "wiring: guest_write_watch.cpp advances the guest mapping generation");

    std::string header;
    if (!read_file(PROSPER_GUEST_MEMORY_MAP_SOURCE, header)) {
        std::fprintf(stderr, "[FAIL] cannot read guest_memory_map.hpp configured at build time\n");
        ++failures;
        return;
    }
    // The notifier's body must stay lock-free. It is called from a SIGSEGV/SIGTRAP handler, where
    // taking guest_mapping_mutex can deadlock against a fault that arrives on a thread already
    // holding it -- which is precisely why this entry point exists instead of reusing
    // notify_guest_mapping_added/removed.
    const size_t body = header.find("inline void notify_guest_page_protection_changed() {");
    check(body != std::string::npos, "guest_memory_map.hpp still defines the signal-safe notifier");
    if (body != std::string::npos) {
        const size_t end = header.find("\n}", body);
        const std::string fn = header.substr(body, end == std::string::npos ? end : end - body);
        check(fn.find("lock_guard") == std::string::npos &&
                  fn.find("guest_mapping_mutex") == std::string::npos,
              "the signal-safe notifier takes no lock (a mutex here deadlocks the fault handler)");
    }
#else
    std::fprintf(stderr, "[FAIL] the production source paths were not defined by the build\n");
    ++failures;
#endif
}

}  // namespace

int main() {
    std::fprintf(stderr, "== test_watch_mapping_generation ==\n");
    test_live_arms();
    test_production_wiring();
    std::fprintf(stderr, failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}

#else   // !__linux__

int main() {
    // The invariant is platform-independent and the notifier is compiled everywhere, but the
    // software watchpoints that motivated it are in the Linux fault handler.
    std::printf("test_watch_mapping_generation: skipped (Linux-only handler)\n");
    return 0;
}

#endif
