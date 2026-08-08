// #2178: a Sony pthread spelling must never hand the guest a BARE FreeBSD errno.
//
// Every other libkernel entry point returns `0x80020000 | errno`. A Sony name registered straight
// onto a fallible POSIX body returns the bare value instead, so the guest gets a number in the wrong
// space -- #1984's shape, the one that killed Oregon Trail until 38619f29. #2158 fixed two
// instances; this file guards the rule rather than the instances.
//
// The assertion is deliberately an INVARIANT over the whole family, not a list of expected values.
// A list would have to be extended by hand every time a function is added, and the failure mode of
// forgetting is silent -- which is exactly how the instances below survived. Anything that returns
// non-zero must have the libkernel prefix; what the errno IS is the POSIX body's business and is
// tested elsewhere.

#include "hle/dispatch.hpp"
#include "hle/nid.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++failures; }
    else std::fprintf(stderr, "ok: %s\n", what.c_str());
}

using prosper::Hle;
using prosper::HleFn;

int main() {
    std::fprintf(stderr, "== test_sce_pthread_encoding ==\n");
    prosper::register_kernel_hle();

    // Each entry is called with arguments chosen to FAIL: a null output slot, an invalid enum, or a
    // bad handle. The point is to reach a return path that carries an errno at all -- a call that
    // succeeds proves nothing about encoding.
    struct Case { const char* name; uint64_t a0, a1, a2; };
    const Case cases[] = {
        {"scePthreadOnce",                0, 0, 0},   // null control / null routine -> EINVAL
        {"scePthreadAttrInit",            0, 0, 0},   // null attr slot
        {"scePthreadAttrSetguardsize",    0, 0, 0},
        {"scePthreadAttrSetdetachstate",  0, 99, 0},  // 99 is not a detach state
        {"scePthreadAttrGetdetachstate",  0, 0, 0},
        {"scePthreadKeyCreate",           0, 0, 0},
        {"scePthreadCreate",              0, 0, 0},
        // The families #2158 and #2359 already fixed, kept here so a regression in ANY of them is
        // caught by this one file rather than by whichever test happened to cover it.
        {"scePthreadMutexUnlock",         0, 0, 0},
        {"scePthreadRwlockattrInit",      0, 0, 0},
        {"scePthreadRwlockattrDestroy",   0, 0, 0},
        {"scePthreadCondDestroy",         0, 0, 0},
    };

    size_t exercised = 0;
    for (const Case& c : cases) {
        HleFn fn = Hle::lookup(prosper::nid_hash(c.name));
        if (!fn) { check(false, std::string(c.name) + " is registered"); continue; }
        const uint64_t rc = fn(c.a0, c.a1, c.a2, 0, 0, 0);
        if (rc == 0) continue;                      // this argument pattern succeeded; says nothing
        ++exercised;
        const bool encoded = (rc & 0xffff0000ull) == 0x80020000ull;
        check(encoded, std::string(c.name) + " returns an ENCODED error (0x8002xxxx), not a bare "
                       "errno -- got 0x" + [&] {
                           char buf[32]; std::snprintf(buf, sizeof buf, "%llx",
                                                       (unsigned long long)rc); return std::string(buf);
                       }());
    }

    // Without this the file would pass vacuously if every call above started returning 0 -- which is
    // a real possibility, since the arguments are chosen to provoke failure and a future change
    // could make one of them legal. A green run that exercised nothing is the #2130 trap, and it is
    // the reason this counter exists rather than a comment saying the arms are meaningful.
    check(exercised >= 6,
          "at least 6 of the calls actually reached an error path (got " +
              std::to_string(exercised) + ") -- otherwise the assertions above are vacuous");

    // The CONTROL, and the half that proves the test can tell the two spellings apart: the POSIX
    // name must still return the bare errno. If encoding had been applied to the shared body instead
    // of the Sony wrapper, every assertion above would pass and this one would fail.
    if (HleFn posix_attr_init = Hle::lookup(prosper::nid_hash("pthread_attr_init"))) {
        const uint64_t rc = posix_attr_init(0, 0, 0, 0, 0, 0);
        check(rc == 0 || (rc & 0xffff0000ull) == 0,
              "control: the POSIX spelling still returns a BARE errno, so encoding was added to the "
              "Sony wrapper rather than to the shared body");
    }

    std::fprintf(stderr, failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}
