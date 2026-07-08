// test_wall_clock — the anchored wall-clock sources (hle_kernel_time.cpp, #92).
//
// Previously the wall-clock surfaces were synthetic AND mutually inconsistent: clock_gettime
// ignored the clockid (CLOCK_REALTIME returned uptime-since-1970), gettimeofday the same, while
// time()/sceRtc* sat on a frozen base of 1700000000 (~Nov 2023) — three notions of "now" that
// disagreed by ~54 years, and sceRtcGetCurrentClockLocalTime used gmtime for "local". Now every
// wall surface derives from ONE anchor (host CLOCK_REALTIME paired with the monotonic ns_now()),
// so this verifies: CLOCK_REALTIME ≈ the host's real now, all wall surfaces agree on one epoch,
// the FreeBSD clockid families map correctly, LocalTime is a valid calendar date, and the
// monotonic sources (ptc/tsc/MONOTONIC) still count uptime and never go backwards.
// Deliberately no exact-time asserts — everything is tolerance-based.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static int64_t iabs64(int64_t v) { return v < 0 ? -v : v; }

// SceRtcDateTime (16 bytes): u16 year,month,day,hour,minute,second; u32 microsecond.
struct RtcDateTime { uint16_t year, month, day, hour, minute, second; uint32_t microsecond; };
static_assert(sizeof(RtcDateTime) == 16, "SceRtcDateTime must be 16 bytes");

int main() {
    printf("== test_wall_clock ==\n");
    register_builtin_hle();

    auto clock_gettime_fn = Hle::lookup(nid_hash("sceKernelClockGettime"));
    auto gettimeofday_fn  = Hle::lookup(nid_hash("sceKernelGettimeofday"));
    auto time_fn          = Hle::lookup(nid_hash("time"));
    auto rtc_tick_fn      = Hle::lookup(nid_hash("sceRtcGetCurrentTick"));
    auto rtc_local_fn     = Hle::lookup(nid_hash("sceRtcGetCurrentClockLocalTime"));
    auto rtc_clock_fn     = Hle::lookup(nid_hash("sceRtcGetCurrentClock"));
    auto rtc_utc_fn       = Hle::lookup(nid_hash("sceRtcGetCurrentDateTimeUtc"));
    auto ptc_fn           = Hle::lookup(nid_hash("sceKernelGetProcessTimeCounter"));
    auto tsc_fn           = Hle::lookup(nid_hash("sceKernelReadTsc"));
    CHECK(clock_gettime_fn && gettimeofday_fn && time_fn && rtc_tick_fn && rtc_local_fn &&
          rtc_clock_fn && rtc_utc_fn && ptc_fn && tsc_fn, "all time fns registered");
    if (fails) { printf("== FAIL ==\n"); return 1; }

    // --- CLOCK_REALTIME (FreeBSD id 0) tracks the HOST's real now (± seconds, not ± decades). ---
    int64_t ts[2] = { 0, 0 };
    uint64_t rc = clock_gettime_fn(0, (uint64_t)(uintptr_t)ts, 0, 0, 0, 0);
    int64_t host_now = (int64_t)::time(nullptr);
    CHECK(rc == 0, "clock_gettime(CLOCK_REALTIME) returned 0");
    CHECK(iabs64(ts[0] - host_now) <= 30, "CLOCK_REALTIME within 30s of host now");
    CHECK(ts[1] >= 0 && ts[1] < 1000000000ll, "CLOCK_REALTIME tv_nsec in range");
    // Regression guard for the two historical failure modes: uptime-as-wall (~1970) and the
    // frozen synthetic base 1700000000 (~Nov 2023). Any real run of this test is after 2026.
    CHECK(ts[0] > 1750000000ll, "CLOCK_REALTIME is not the 1970/2023 synthetic base");

    // --- All wall surfaces agree on ONE epoch (each pair within 2 s of live drift). ---
    int64_t tv[2] = { 0, 0 };
    gettimeofday_fn((uint64_t)(uintptr_t)tv, 0, 0, 0, 0, 0);
    int64_t time_s = (int64_t)time_fn(0, 0, 0, 0, 0, 0);
    uint64_t tick = 0;
    rtc_tick_fn((uint64_t)(uintptr_t)&tick, 0, 0, 0, 0, 0);
    int64_t rtc_unix_s = (int64_t)(tick / 1000000ull) - 62135596800ll;   // undo the 0001-01-01 epoch
    CHECK(iabs64(tv[0] - ts[0]) <= 2, "gettimeofday agrees with CLOCK_REALTIME");
    CHECK(iabs64(time_s - ts[0]) <= 2, "time() agrees with CLOCK_REALTIME");
    CHECK(iabs64(rtc_unix_s - ts[0]) <= 2, "RtcGetCurrentTick agrees with CLOCK_REALTIME");

    // --- CLOCK_SECOND (id 13) is the realtime clock at 1 s resolution. ---
    int64_t tsec[2] = { -1, -1 };
    clock_gettime_fn(13, (uint64_t)(uintptr_t)tsec, 0, 0, 0, 0);
    CHECK(iabs64(tsec[0] - ts[0]) <= 2, "CLOCK_SECOND tracks CLOCK_REALTIME");
    CHECK(tsec[1] == 0, "CLOCK_SECOND has 1s resolution (tv_nsec == 0)");

    // --- MONOTONIC/UPTIME family (ids 4/5/7) counts since process start, NOT wall time. ---
    for (int id : { 4, 5, 7 }) {
        int64_t tm2[2] = { -1, -1 };
        clock_gettime_fn((uint64_t)id, (uint64_t)(uintptr_t)tm2, 0, 0, 0, 0);
        char msg[64]; snprintf(msg, sizeof msg, "clockid %d is uptime-scale (< 1h)", id);
        CHECK(tm2[0] >= 0 && tm2[0] < 3600, msg);
    }

    // --- Monotonic sources unaffected: ptc/tsc are uptime-scale and never go backwards. ---
    uint64_t p1 = ptc_fn(0, 0, 0, 0, 0, 0), t1 = tsc_fn(0, 0, 0, 0, 0, 0);
    uint64_t p2 = ptc_fn(0, 0, 0, 0, 0, 0), t2 = tsc_fn(0, 0, 0, 0, 0, 0);
    CHECK(p1 < 3600ull * 1000000000ull, "ProcessTimeCounter is uptime-scale (not wall)");
    CHECK(t1 < 3600ull * 1000000000ull, "ReadTsc is uptime-scale (not wall)");
    CHECK(p2 >= p1 && t2 >= t1, "monotonic sources never go backwards");
    int64_t r2[2] = { 0, 0 };
    clock_gettime_fn(0, (uint64_t)(uintptr_t)r2, 0, 0, 0, 0);
    CHECK(r2[0] > ts[0] || (r2[0] == ts[0] && r2[1] >= ts[1]), "CLOCK_REALTIME never goes backwards");

    // --- LocalTime is a VALID calendar date (the UE4 FDateTime constraint: month/day >= 1). ---
    RtcDateTime lt{};
    rtc_local_fn((uint64_t)(uintptr_t)&lt, 0, 0, 0, 0, 0);
    CHECK(lt.year >= 2026 && lt.year <= 2200, "LocalTime year is current-era");
    CHECK(lt.month >= 1 && lt.month <= 12, "LocalTime month in 1..12");
    CHECK(lt.day >= 1 && lt.day <= 31, "LocalTime day in 1..31");
    CHECK(lt.hour <= 23 && lt.minute <= 59 && lt.second <= 60, "LocalTime h/m/s in range");
    CHECK(lt.microsecond < 1000000u, "LocalTime microsecond in range");

    // --- sceRtcGetCurrentClock applies its tz_minutes argument: +90 min shifts the result. ---
    RtcDateTime c0{}, c90{};
    rtc_clock_fn((uint64_t)(uintptr_t)&c0, 0, 0, 0, 0, 0);                       // tz = 0 (UTC)
    rtc_clock_fn((uint64_t)(uintptr_t)&c90, (uint64_t)(uint32_t)90, 0, 0, 0, 0); // tz = +90 min
    struct tm tm0 {}, tm90 {};
    tm0.tm_year = c0.year - 1900;  tm0.tm_mon = c0.month - 1;  tm0.tm_mday = c0.day;
    tm0.tm_hour = c0.hour;         tm0.tm_min = c0.minute;     tm0.tm_sec = c0.second;
    tm90.tm_year = c90.year - 1900; tm90.tm_mon = c90.month - 1; tm90.tm_mday = c90.day;
    tm90.tm_hour = c90.hour;        tm90.tm_min = c90.minute;    tm90.tm_sec = c90.second;
#ifdef _WIN32
    int64_t s0 = (int64_t)_mkgmtime(&tm0), s90 = (int64_t)_mkgmtime(&tm90);
#else
    int64_t s0 = (int64_t)timegm(&tm0), s90 = (int64_t)timegm(&tm90);
#endif
    CHECK(iabs64((s90 - s0) - 90 * 60) <= 2, "sceRtcGetCurrentClock(+90min) shifts by 90 minutes");

    // --- The UTC datetime agrees with CLOCK_REALTIME (round-trips through the calendar). ---
    RtcDateTime utc{};
    rtc_utc_fn((uint64_t)(uintptr_t)&utc, 0, 0, 0, 0, 0);
    struct tm tmu {};
    tmu.tm_year = utc.year - 1900; tmu.tm_mon = utc.month - 1; tmu.tm_mday = utc.day;
    tmu.tm_hour = utc.hour;        tmu.tm_min = utc.minute;    tmu.tm_sec = utc.second;
#ifdef _WIN32
    int64_t su = (int64_t)_mkgmtime(&tmu);
#else
    int64_t su = (int64_t)timegm(&tmu);
#endif
    CHECK(iabs64(su - ts[0]) <= 2, "DateTimeUtc agrees with CLOCK_REALTIME");

    if (fails) printf("== FAIL (%d) ==\n", fails);
    else       printf("== PASS ==\n");
    return fails ? 1 : 0;
}
