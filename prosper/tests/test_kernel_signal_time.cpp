// test_kernel_signal_time (#190) — the libkernel/libScePosix signal + time-conversion surface that
// PPSA02664 imports and that previously fell to the generic unimplemented stub. Verifies each name
// hashes to the exact import NID the game uses (so the import resolves to our impl), that all five are
// registered, and the real behavior: the UTC<->localtime pair round-trips and fills its out-param,
// and pthread_setcancelstate reports the previous state + rejects a bad state.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_kernel_signal_time ==\n");
    register_builtin_hle();

    // Name -> NID must match the game's imports (#190 table), else the import never reaches our impl.
    struct { const char* name; const char* nid; } syms[] = {
        { "_sigprocmask",                    "6xVpy0Fdq+I" },
        { "_is_signal_return",               "crb5j7mkk1c" },
        { "sceKernelConvertLocaltimeToUtc",  "0NTHN1NKONI" },
        { "sceKernelConvertUtcToLocaltime",  "-o5uEDpN+oY" },
        { "pthread_setcancelstate",          "lZzFeSxPl08" },
    };
    for (auto& s : syms) {
        std::string h = nid_hash(s.name);
        char msg[160]; snprintf(msg, sizeof msg, "%s hashes to its import NID %s (got %s)", s.name, s.nid, h.c_str());
        CHECK(h == s.nid, msg);
        snprintf(msg, sizeof msg, "%s is registered (resolves to a real impl, not the stub)", s.name);
        CHECK(Hle::lookup(s.nid) != nullptr, msg);
    }

    HleFn utc2local = Hle::lookup("-o5uEDpN+oY");
    HleFn local2utc = Hle::lookup("0NTHN1NKONI");
    HleFn setcancel = Hle::lookup("lZzFeSxPl08");
    if (!utc2local || !local2utc || !setcancel) { printf("== FAIL (unresolved) ==\n"); return 1; }

    // UTC -> localtime -> UTC round-trips exactly (tz-independent: whatever the host offset is, the
    // inverse conversion undoes it). Also proves both fill their primary out-param (pre-seeded with a
    // sentinel that must be overwritten). 1e9 = 2001-09-09 UTC, a plain non-DST-edge instant.
    const int64_t UTC = 1000000000;
    int64_t local = (int64_t)0x0BADF00D, back = (int64_t)0x0BADF00D;
    uint64_t r1 = utc2local((uint64_t)UTC, (uint64_t)(uintptr_t)&local, 0, 0, 0, 0);
    CHECK(r1 == 0, "ConvertUtcToLocaltime returns OK");
    CHECK(local != (int64_t)0x0BADF00D, "ConvertUtcToLocaltime WROTE the local-time out-param (was uninitialized before)");
    uint64_t r2 = local2utc((uint64_t)local, 0 /*unk*/, (uint64_t)(uintptr_t)&back, 0, 0, 0);
    CHECK(r2 == 0, "ConvertLocaltimeToUtc returns OK");
    CHECK(back != (int64_t)0x0BADF00D, "ConvertLocaltimeToUtc WROTE the utc out-param (result is at arg index 2)");
    CHECK(back == UTC, "UTC -> localtime -> UTC round-trips exactly (offset applied then undone)");

    // The OrbisTimesec struct out-param (arg 2) is filled: its first field is the converted time.
    int64_t ts_buf[2] = { (int64_t)0x0BADF00D, 0 };
    utc2local((uint64_t)UTC, (uint64_t)(uintptr_t)&local, (uint64_t)(uintptr_t)ts_buf, 0, 0, 0);
    CHECK(ts_buf[0] == local, "ConvertUtcToLocaltime fills OrbisTimesec.t with the converted time");

    // pthread_setcancelstate: valid states return OK and write the previous state (ENABLE(0), the POSIX
    // default — prosper never cancels guest threads, so no per-thread state is tracked); a NULL out-ptr
    // is fine; an out-of-range state is EINVAL. (A host thread_local for real tracking broke the boot —
    // see the impl comment — and its value would be unobservable anyway.)
    int oldst = -1;
    uint64_t c1 = setcancel(1 /*DISABLE*/, (uint64_t)(uintptr_t)&oldst, 0, 0, 0, 0);
    CHECK(c1 == 0 && oldst == 0, "setcancelstate(DISABLE) -> OK, writes previous state ENABLE(0)");
    oldst = -1;
    uint64_t c2 = setcancel(0 /*ENABLE*/, (uint64_t)(uintptr_t)&oldst, 0, 0, 0, 0);
    CHECK(c2 == 0 && oldst == 0, "setcancelstate(ENABLE) -> OK, writes previous state ENABLE(0)");
    uint64_t c2b = setcancel(1 /*DISABLE*/, 0 /*NULL old*/, 0, 0, 0, 0);
    CHECK(c2b == 0, "setcancelstate with NULL old_state -> OK (no write)");
    uint64_t c3 = setcancel(7 /*invalid*/, 0, 0, 0, 0, 0);
    // Plain errno 22, not the 0x80020016 SCE encoding (#1612). This handler is registered ONLY as the
    // POSIX `pthread_setcancelstate`, whose contract returns an errno directly — and every other
    // pthread entry point in the emulator (mutex, rwlock, key, cond, sem) already returns a bare
    // FreeBSD errno from the same shared handlers. A guest comparing the result against EINVAL could
    // never match the encoded form.
    CHECK(c3 == 22ull, "setcancelstate(invalid) -> EINVAL (POSIX contract: a plain errno)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
