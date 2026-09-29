#include "shared/rtt/seed_extent_history.hpp"
#include <cstdio>

using prosper::frontend::SeedExtent;
using prosper::frontend::SeedExtentHistory;

static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}

int main() {
    constexpr SeedExtent hdr{2560, 1440, 97}, alias{3840, 2160, 97};
    constexpr uint64_t target = 0x20893c0000;
    SeedExtentHistory before;
    check(!before.observe(target, hdr, alias) && before.observe(target, alias, hdr),
          "retained #3907 pre-fix opposite extent misses identify a reversal");
    SeedExtentHistory fixed;
    check(!fixed.observe(target, hdr, alias) && !fixed.observe(target, hdr, hdr) &&
          !fixed.observe(target, hdr, alias),
          "fixed masked pass never replaces the HDR entry and produces no reversal");
    check(!fixed.observe(target, hdr, alias), "repeated requests of the same alias are quiet");
    SeedExtentHistory copied = before;
    check(!copied.observe(target + 0x100000, hdr, alias),
          "a resolved entry cannot carry a reversal to a different address");
    SeedExtentHistory matched;
    (void)matched.observe(target, hdr, alias);
    (void)matched.observe(target, alias, alias);
    check(!matched.observe(target, alias, hdr), "a matching seed ends the mismatch sequence");
    SeedExtentHistory format;
    (void)format.observe(target, hdr, alias);
    check(!format.observe(target, alias, {hdr.width, hdr.height, 37}) &&
          !format.observe(target, alias, hdr), "format reinterpretation resets extent history");
    SeedExtentHistory unknown;
    (void)unknown.observe(target, hdr, alias);
    check(!unknown.observe(target, {}, alias) && !unknown.observe(target, alias, hdr),
          "unknown stored extent resets history");
    (void)unknown.observe(target, hdr, alias);
    unknown.reset();
    check(!unknown.observe(target, alias, hdr), "volume or unavailable seed state can reset history");
    std::printf("seed extent history: %d failures\n", failures);
    return failures != 0;
}
