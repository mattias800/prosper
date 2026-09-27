// texture_cache_budget — the persistent texture cache's byte budget from live device headroom (#3873).
//
// Every expected value below is computed by hand from the formula in texture_cache_budget.hpp, not
// by calling the function under test, and each case names the old heap/8 answer beside it so a
// reader sees what changed. The mutation arms this file was checked against are listed in the PR:
// removing the unified cap, the floor, the heap ceiling, the texture subtraction or the max() of the
// two usage figures, or changing the 80% target, each reddens at least one arm here.
#include "gpu/memory/texture_cache_budget.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper::gpu;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "[ok]  " : "[FAIL]", what);
    if (!ok) ++fails;
}

static constexpr uint64_t MiB = 1024ull * 1024ull;
static constexpr uint64_t GiB = 1024ull * MiB;

// 80% of x, to the byte, as the policy rounds it (x / 100 * 80 plus the remainder's share).
static constexpr uint64_t pct80(uint64_t x) { return x / 100 * 80 + x % 100 * 80 / 100; }

static TextureCacheBudgetInputs discrete(uint64_t heap, uint64_t budget, uint64_t usage,
                                         uint64_t held, uint64_t textures) {
    TextureCacheBudgetInputs in;
    in.heap_size = heap;
    in.have_budget = true;
    in.heap_budget = budget;
    in.heap_usage = usage;
    in.prosper_held = held;
    in.texture_bytes = textures;
    in.unified = false;
    return in;
}

int main() {
    // --- the old rule, which is still the fallback --------------------------------------------
    check(texture_cache_heuristic_budget(8 * GiB) == 1 * GiB, "heuristic: 8 GiB heap -> 1 GiB");
    check(texture_cache_heuristic_budget(16 * GiB) == 2 * GiB, "heuristic: 16 GiB heap -> 2 GiB");
    check(texture_cache_heuristic_budget(2 * GiB) == 1 * GiB, "heuristic: small heap floors at 1 GiB");
    check(texture_cache_heuristic_budget(128 * GiB) == 4 * GiB, "heuristic: huge heap caps at 4 GiB");

    // --- missing extension: exactly the old rule ----------------------------------------------
    {
        TextureCacheBudgetInputs in;
        in.heap_size = 16 * GiB;
        in.have_budget = false;
        in.heap_budget = 15 * GiB;  // ignored without the extension
        const auto b = texture_cache_budget(in);
        check(b.bytes == 2 * GiB && b.source == TextureCacheBudgetSource::heuristic,
              "no VK_EXT_memory_budget: 16 GiB discrete keeps the old 2 GiB");
        in.have_budget = true;
        in.heap_budget = 0;
        check(texture_cache_budget(in).bytes == 2 * GiB,
              "a zero heapBudget (driver reported nothing) is treated as no extension");
    }

    // --- 8 GB discrete card -------------------------------------------------------------------
    // budget 7.5 GiB, process usage 2.5 GiB of which 0.5 GiB is textures -> non-texture 2 GiB.
    // target = 0.8 * 7.5 = 6 GiB; live = 6 - 2 = 4 GiB; ceiling 6 GiB.  Old rule: 1 GiB.
    {
        const auto b = texture_cache_budget(
            discrete(8 * GiB, 7680 * MiB, 2560 * MiB, 2 * GiB, 512 * MiB));
        check(b.bytes == 4 * GiB && b.source == TextureCacheBudgetSource::live,
              "8 GB discrete: 4 GiB from live headroom (old rule: 1 GiB)");
        check(b.non_texture == 2 * GiB && b.target == 6 * GiB,
              "8 GB discrete: reports the non-texture usage and the 80% target it used");
    }

    // --- 16 GB discrete card ------------------------------------------------------------------
    // budget 15 GiB -> target 12 GiB; usage 3 GiB incl. 1 GiB textures -> live 10 GiB. Old: 2 GiB.
    {
        const auto b = texture_cache_budget(
            discrete(16 * GiB, 15 * GiB, 3 * GiB, 2560 * MiB, 1 * GiB));
        check(b.bytes == 10 * GiB && b.source == TextureCacheBudgetSource::live,
              "16 GB discrete: 10 GiB from live headroom (old rule: 2 GiB)");
    }
    // An idle 16 GB card: target 12.8 GiB exceeds the 75% ceiling (12 GiB).
    {
        const auto b = texture_cache_budget(discrete(16 * GiB, 16 * GiB, 0, 0, 0));
        check(b.bytes == 12 * GiB && b.source == TextureCacheBudgetSource::ceiling,
              "idle 16 GB discrete: clipped to 75% of the heap");
    }

    // --- which usage figure: the larger of driver usage and prosper's own count -----------------
    {
        // The driver sees 1 GiB prosper cannot (swapchain, driver objects): 8 GiB card, budget 8,
        // usage 4 GiB, prosper counted 3 GiB, no textures -> non-texture 4, target 6.4 -> 2.4 GiB.
        const auto b = texture_cache_budget(discrete(8 * GiB, 8 * GiB, 4 * GiB, 3 * GiB, 0));
        check(b.bytes == pct80(8 * GiB) - 4 * GiB && b.non_texture == 4 * GiB,
              "driver usage above prosper's count is the one subtracted");
        // The driver lags prosper's count: prosper's figure is used.
        const auto c = texture_cache_budget(discrete(8 * GiB, 8 * GiB, 3 * GiB, 4 * GiB, 0));
        check(c.non_texture == 4 * GiB && c.bytes == b.bytes,
              "prosper's count above the driver's usage is the one subtracted");
    }

    // --- stability: a growing cache does not shrink its own budget ----------------------------
    {
        const auto before = texture_cache_budget(
            discrete(8 * GiB, 7680 * MiB, 2560 * MiB, 2 * GiB, 512 * MiB));
        const auto after = texture_cache_budget(
            discrete(8 * GiB, 7680 * MiB, 5 * GiB, 4608 * MiB, 3 * GiB));  // +2.5 GiB of textures
        check(before.bytes == after.bytes && after.bytes == 4 * GiB,
              "texture growth raises usage and texture_bytes equally: budget unchanged");
    }

    // --- a budget that shrinks (another application takes VRAM) --------------------------------
    {
        // Same 8 GB card, the driver's budget drops to 4 GiB: target 3.2, non-texture 2 -> 1.2 GiB.
        const auto b = texture_cache_budget(
            discrete(8 * GiB, 4 * GiB, 2560 * MiB, 2 * GiB, 512 * MiB));
        const uint64_t expected = pct80(4 * GiB) - 2 * GiB;
        check(b.bytes == expected && b.source == TextureCacheBudgetSource::live,
              "shrinking budget: 8 GB card squeezed to 4 GiB -> ~1.2 GiB");
        // Squeezed further to 3 GiB: target 2.4, live 0.4 -> the 1 GiB floor.
        const auto c = texture_cache_budget(
            discrete(8 * GiB, 3 * GiB, 2560 * MiB, 2 * GiB, 512 * MiB));
        check(c.bytes == 1 * GiB && c.source == TextureCacheBudgetSource::floor,
              "shrinking budget: never below the 1 GiB floor");
        // Non-texture usage above the target entirely (saturating subtraction, no wrap).
        const auto d = texture_cache_budget(discrete(8 * GiB, 2 * GiB, 7 * GiB, 7 * GiB, 0));
        check(d.bytes == 1 * GiB && d.source == TextureCacheBudgetSource::floor,
              "usage above the target saturates to the floor instead of wrapping");
    }

    // --- the floor wins over the 75% ceiling on a tiny heap ------------------------------------
    {
        const auto b = texture_cache_budget(discrete(1 * GiB, 1 * GiB, 0, 0, 0));
        check(b.bytes == 1 * GiB && b.source == TextureCacheBudgetSource::floor,
              "1 GiB heap: floor (1 GiB) beats the 768 MiB ceiling, matching the old rule");
    }

    // --- unified memory (APU): never above the old rule ---------------------------------------
    {
        // This project's APU: 43 GiB device-local heap, driver budget ~37 GiB, usage 5 GiB.
        TextureCacheBudgetInputs in = discrete(44196 * MiB, 37993 * MiB, 5 * GiB, 5 * GiB, 0);
        in.unified = true;
        const auto b = texture_cache_budget(in);
        check(b.bytes == 4 * GiB && b.source == TextureCacheBudgetSource::unified,
              "APU with room: capped at the old heuristic (4 GiB), i.e. unchanged behaviour");
        check(b.bytes == texture_cache_heuristic_budget(in.heap_size),
              "APU with room: identical to the pre-#3873 answer");
        // The same heap reported as discrete WOULD balloon: that is what the cap is for.
        in.unified = false;
        check(texture_cache_budget(in).bytes > 16 * GiB,
              "positive control: the same APU figures read as discrete give > 16 GiB");
        // A tight APU: budget 8 GiB, usage 4 GiB -> target 6.4, live 2.4 GiB < heuristic.
        TextureCacheBudgetInputs tight = discrete(44196 * MiB, 8 * GiB, 4 * GiB, 4 * GiB, 0);
        tight.unified = true;
        const auto t = texture_cache_budget(tight);
        check(t.bytes == pct80(8 * GiB) - 4 * GiB && t.source == TextureCacheBudgetSource::live,
              "APU under pressure: live headroom lowers the budget below the heuristic");
    }

    // --- material-change hysteresis -----------------------------------------------------------
    check(!texture_cache_budget_changed_materially(4 * GiB, 4 * GiB + 100 * MiB),
          "hysteresis: +100 MiB on 4 GiB is not material");
    check(texture_cache_budget_changed_materially(4 * GiB, 4 * GiB + 256 * MiB),
          "hysteresis: +256 MiB (1/16) on 4 GiB is material");
    check(!texture_cache_budget_changed_materially(1 * GiB, 1 * GiB + 127 * MiB),
          "hysteresis: +127 MiB on 1 GiB is not material (128 MiB minimum)");
    check(texture_cache_budget_changed_materially(2 * GiB, 1 * GiB),
          "hysteresis: a halving is material");

    // --- the absolute override -----------------------------------------------------------------
    check(resolve_texture_cache_limit(true, 12288 * MiB, 4 * GiB) == 12288 * MiB,
          "PROSPER_BACKEND_TEXTURE_CACHE_MB overrides the policy upward");
    check(resolve_texture_cache_limit(true, 256 * MiB, 10 * GiB) == 256 * MiB,
          "PROSPER_BACKEND_TEXTURE_CACHE_MB overrides the policy downward");
    check(resolve_texture_cache_limit(false, 0, 10 * GiB) == 10 * GiB,
          "no override: the policy's figure is enforced");
    check(resolve_texture_cache_limit(false, 0, 0) == 1 * GiB,
          "no override, device not sized yet: the historical 1 GiB");

    // --- no overflow on absurd heaps -----------------------------------------------------------
    {
        const auto b = texture_cache_budget(discrete(UINT64_MAX, UINT64_MAX, 0, 0, 0));
        // 80% of the budget exceeds 75% of the heap, so the answer is 75% of UINT64_MAX, exactly.
        check(b.bytes == UINT64_MAX / 100 * 75 + UINT64_MAX % 100 * 75 / 100 &&
                  b.source == TextureCacheBudgetSource::ceiling,
              "UINT64_MAX heap and budget: percentages do not overflow");
    }

    std::printf("%s (%d failure%s)\n", fails ? "FAILED" : "PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
