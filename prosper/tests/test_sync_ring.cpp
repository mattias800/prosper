// PROSPER_SYNC_RING: the retained tail of synchronisation history, used to diagnose a total guest
// deadlock (#2139) where a thread snapshot shows where every thread stopped but nothing shows how
// they got there.
//
// The ring is sized from the environment at static-initialisation time, so this test is registered
// with ENVIRONMENT "PROSPER_SYNC_RING=64" -- a deliberately tiny ring, because the property that
// matters most is what the dump says when it has DROPPED events. A diagnostic that silently
// truncates is worse than no diagnostic: it reads as "this is everything that happened".
#include "../src/hle/sync_futex.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <pthread.h>
#endif

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else std::printf("  [ok]   %s\n", msg); } while (0)

#ifdef _WIN32
static std::string read_all(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return {};
    std::string out;
    char buffer[4096];
    size_t got;
    while ((got = std::fread(buffer, 1, sizeof buffer, f)) > 0) out.append(buffer, got);
    std::fclose(f);
    return out;
}

static size_t count_occurrences(const std::string& haystack, const char* needle) {
    size_t count = 0, at = 0;
    const size_t len = std::strlen(needle);
    while ((at = haystack.find(needle, at)) != std::string::npos) { ++count; at += len; }
    return count;
}
#endif

int main() {
#ifndef _WIN32
    std::printf("  [skip] PROSPER_SYNC_RING is a Windows-only diagnostic\n");
    std::printf("== PASS ==\n");
    return 0;
#else
    const char* path = "test_sync_ring.out";
    std::remove(path);

    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

    // 100 events into a 64-event ring: enough to wrap it comfortably.
    constexpr int kEmitted = 100;
    for (int i = 0; i < kEmitted; ++i) prosper::interruptible_cond_signal(&cond);

    prosper::dump_guest_sync_trace(path);
    const std::string dump = read_all(path);
    CHECK(!dump.empty(), "the ring dump wrote something");

    // 1. It recorded the signals at all, and attributed them to this cond's slot.
    CHECK(count_occurrences(dump, "kind=signal") > 0, "signals are recorded as kind=signal");

    // 2. It is BOUNDED: a 64-entry ring must not report 100 event lines.
    const size_t lines = count_occurrences(dump, "[sync-trace] seq=");
    CHECK(lines > 0 && lines <= 64,
          "a 64-event ring retains at most 64 events, not every event ever emitted");

    // 3. THE POINT OF THE TEST -- the dump must say it dropped events rather than presenting the
    //    survivors as the whole history. The header carries the true total and the retained window,
    //    so a reader can tell "this is all of it" from "this is the tail of it".
    char header[128];
    std::snprintf(header, sizeof header, "%d events total, showing %d..%d",
                  kEmitted, kEmitted - 64 + 1, kEmitted);
    CHECK(dump.find(header) != std::string::npos,
          "the header reports the true total and the retained window, so drops are visible");

    // 4. The oldest surviving event is the start of that window, not sequence 1. Without this, a
    //    ring that reported an honest header but dumped stale wrapped slots would still pass above.
    CHECK(dump.find("seq=1 ") == std::string::npos,
          "wrapped-away events are gone rather than served stale");
    char oldest[64];
    std::snprintf(oldest, sizeof oldest, "seq=%d ", kEmitted - 64 + 1);
    CHECK(dump.find(oldest) != std::string::npos,
          "the oldest retained event is exactly the start of the reported window");

    // 5. Sequence numbers must be strictly increasing in the printed order, so the tail reads as
    //    history. A ring printed from the physical slot order would fail this.
    size_t at = 0;
    long previous = -1;
    bool ordered = true;
    while ((at = dump.find("[sync-trace] seq=", at)) != std::string::npos) {
        at += std::strlen("[sync-trace] seq=");
        const long sequence = std::strtol(dump.c_str() + at, nullptr, 10);
        if (sequence <= previous) { ordered = false; break; }
        previous = sequence;
    }
    CHECK(ordered, "events are printed oldest-first in strictly increasing sequence order");

    std::remove(path);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
#endif
}
