// test_dsbridge_miss_classifier — the sampled-depth bridge must name the reason belonging to the
// entry it actually rejected, not the first failing reason found across every sibling at that
// address.
//
// One address can hold several persistent-DS cache entries: a live surface plus a stale sibling
// retained at a different size. The miss classifier set `extent_bad` from ANY mismatching sibling
// and tested it before every validity reason, so an address holding a correctly-sized entry that
// failed for depth-invalid was filed under `extent` and printed a sentence describing the sibling's
// dimensions. The detail string had the same defect independently — it keyed off `extent_bad` even
// when the classified reason was something else, so the text contradicted the counter it was filed
// under.
//
// Measured on Grand Theft Auto V (PPSA04263): 0x20945c0000 reported
// `extent (T# wants 1024x1536, retained image is 3840x2160)` 1403 times on one route while the cache
// held a 1024x1536 D32_SFLOAT entry at that very address. That mislabel sent an investigation after
// a non-existent extent bug; the true reason is depth-invalid, which is separately falsified ground.
//
// This exercises `classify_ds_bridge_miss` directly rather than through the log. The log path is
// gated on PROSPER_DSBRIDGE_LOG, which is read into a function-local static — arming it from a test
// would trip the cached-env-arming guard AND make the result depend on nothing having called the
// bridge earlier in the process.

#include "render_runner.h"

#include <cstdio>
#include <cstdint>
#include <string>

static int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; } \
} while (0)

namespace {

constexpr uint64_t kAddr = 0x20945c0000ull;   // the address the routed run reported
constexpr uint32_t kWantW = 1024, kWantH = 1536;
constexpr uint32_t kStaleW = 3840, kStaleH = 2160;

prosper::test::PersistentDsKey key_for(uint32_t w, uint32_t h) {
    return prosper::test::PersistentDsKey{
        kAddr, kAddr, 0, 0, 0, w, h, static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT)};
}

std::string reason_of(const prosper::test::DsBridgeMiss& miss) {
    return prosper::test::ds_bridge_miss_detail(miss, kWantW, kWantH);
}

}  // namespace

int main() {
    auto& cache = prosper::test::persistent_ds_cache();

    // --- Arm 1: a right-sized entry failing for depth-invalid, beside a wrong-sized sibling ------
    {
        const auto stale = key_for(kStaleW, kStaleH);
        const auto live = key_for(kWantW, kWantH);
        auto& stale_image = cache[stale];
        auto& live_image = cache[live];
        stale_image.image = (VkImage)(uintptr_t)1;
        stale_image.layout_initialized = true;
        stale_image.depth_valid = true;      // valid, but the wrong size for this request
        live_image.image = (VkImage)(uintptr_t)2;
        live_image.layout_initialized = true;
        live_image.depth_valid = false;      // right size, invalid depth — the REAL reason

        const auto miss = prosper::test::classify_ds_bridge_miss(kAddr, kWantW, kWantH);
        CHECK(miss.matched_plane, "the address is recognised as a plane of a retained surface");
        CHECK(!miss.extent_is_reason,
              "extent is not the reason when the right size is present and failed otherwise");
        const std::string why = reason_of(miss);
        CHECK(why == "depth-invalid",
              ("the reason names the rejected entry, not a wrong-sized sibling (got \"" + why +
               "\")").c_str());
        CHECK(why.find("extent") == std::string::npos,
              "the detail string does not describe an extent mismatch that is not the reason");
        CHECK(!prosper::test::find_persistent_ds_sampled(kAddr, kWantW, kWantH).image,
              "the lookup itself still declines - this fixes the REASON, not the outcome");

        cache.erase(stale);
        cache.erase(live);
    }

    // --- Arm 2 (positive control): nothing at this address is the right size -> extent IS the ----
    // reason. Without this arm the fix is indistinguishable from deleting the extent branch.
    {
        const auto stale = key_for(kStaleW, kStaleH);
        auto& stale_image = cache[stale];
        stale_image.image = (VkImage)(uintptr_t)1;
        stale_image.layout_initialized = true;
        stale_image.depth_valid = true;

        const auto miss = prosper::test::classify_ds_bridge_miss(kAddr, kWantW, kWantH);
        CHECK(miss.matched_plane && miss.extent_is_reason,
              "extent remains the reason when NOTHING at the address is the right size");
        const std::string why = reason_of(miss);
        CHECK(why.rfind("extent", 0) == 0,
              ("the extent reason is reported (got \"" + why + "\")").c_str());
        CHECK(why.find("1024x1536") != std::string::npos &&
                  why.find("3840x2160") != std::string::npos,
              "the extent detail reports both the requested and the retained dimensions");

        cache.erase(stale);
    }

    // --- Arm 3: a right-sized VALID entry is served, so the arms above are not passing merely -----
    // because everything declines.
    {
        const auto stale = key_for(kStaleW, kStaleH);
        const auto live = key_for(kWantW, kWantH);
        auto& stale_image = cache[stale];
        auto& live_image = cache[live];
        stale_image.image = (VkImage)(uintptr_t)1;
        stale_image.layout_initialized = true;
        stale_image.depth_valid = true;
        live_image.image = (VkImage)(uintptr_t)2;
        live_image.layout_initialized = true;
        live_image.depth_valid = true;
        prosper::test::note_persistent_ds_depth_write(live_image, true, true);

        const auto found = prosper::test::find_persistent_ds_sampled(kAddr, kWantW, kWantH);
        CHECK(found.image == &live_image,
              "the right-sized valid entry is selected over the wrong-sized sibling");
        CHECK(found.width == kWantW && found.height == kWantH,
              "the served view reports the retained entry's own extent");
        const auto miss = prosper::test::classify_ds_bridge_miss(kAddr, kWantW, kWantH);
        CHECK(miss.why == nullptr || std::string(miss.why).empty(),
              "a servable address classifies with no miss reason");

        cache.erase(stale);
        cache.erase(live);
    }

    // --- Arm 4: an address the cache has never heard of stays "no entry", not a fabricated -------
    // reason.
    {
        const auto miss = prosper::test::classify_ds_bridge_miss(0x20dead0000ull, kWantW, kWantH);
        CHECK(!miss.matched_plane && miss.why == nullptr,
              "an unknown address matches no plane and is given no reason");
    }

    if (failures) {
        std::fprintf(stderr, "test_dsbridge_miss_classifier: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_dsbridge_miss_classifier: all checks passed\n");
    return 0;
}
