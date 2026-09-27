// test_descriptor_pool_reuse — the bounded VkDescriptorPool free list behind every render pass.
//
// The property under test is that a retired pool is RESET AND HANDED BACK rather than destroyed and
// rebuilt, that a pool in use is never handed to a second caller, that a handed-back pool really is
// empty (its full capacity allocatable again), that a pool too small for a request is never handed
// out for it, and that what is retained is BOUNDED. Rendering correctness is covered by the existing
// render tests, which all allocate their descriptor sets from this list.
//
// Runs twice from CMake: once normally, once with PROSPER_NO_DESCRIPTOR_POOL_REUSE=1. The second
// run is the negative control, and it is a separate PROCESS because the switch is read once into a
// function-local static.
#include "fixtures/render_runner.h"
#include <cstdlib>
#include <set>

using namespace prosper::test;

static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    std::fflush(stdout);
    failures += !ok;
}

int main() {
    const auto& ctx = render_vk_ctx();
    if (!ctx.ok) {
        std::printf("[skip] no Vulkan device\n");
        return 0;
    }
    const bool reuse = std::getenv("PROSPER_NO_DESCRIPTOR_POOL_REUSE") == nullptr;
    std::printf("== test_descriptor_pool_reuse (reuse %s) ==\n", reuse ? "ENABLED" : "DISABLED");
    check(render_descriptor_pool_reuse_enabled() == reuse,
          "the mode under test is the mode the switch reports");

    // One storage-buffer set layout: enough to allocate real sets from each pool, which is how the
    // "a handed-back pool is empty" arm observes the reset.
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_ALL;
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(ctx.dev, &layout_info, nullptr, &layout) != VK_SUCCESS) {
        std::printf("[FAIL] could not create a descriptor set layout\n");
        return 1;
    }
    // Allocates `count` single-buffer sets from `pool`; true only if every allocation succeeded.
    auto allocate_sets = [&](VkDescriptorPool pool, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(ctx.dev, &ai, &set) != VK_SUCCESS) return false;
        }
        return true;
    };

    drain_render_descriptor_pool_cache(ctx.dev);
    const RenderDescriptorPoolCapacity small{4, 4, 1, 1};

    // WARM THE FREE LIST FIRST, for the same reason as the command-pool test: against an empty cache
    // two acquires both take the create path and the ownership arm below would be void.
    RenderDescriptorPoolLease warm = acquire_render_descriptor_pool(ctx.dev, small);
    check(static_cast<bool>(warm), "a pool is acquired");
    check(warm.capacity.covers(small), "the acquired pool covers the requirement");
    // Fill it to its FULL capacity so that, if release forgot to reset, the re-acquired pool would
    // have no room left and the emptiness arm below would fail.
    check(allocate_sets(warm.pool, warm.capacity.sets), "the pool's full set capacity allocates");
    release_render_descriptor_pool(ctx.dev, warm);

    const auto after_warm = render_descriptor_pool_stats();
    RenderDescriptorPoolLease first = acquire_render_descriptor_pool(ctx.dev, small);
    const auto after_first = render_descriptor_pool_stats();
    check(static_cast<bool>(first), "a second acquire after that release succeeds");
    if (reuse) {
        check(after_first.hits == after_warm.hits + 1,
              "the free list really is warm, so the arms below exercise the CACHED path");
        check(first.pool == warm.pool, "a released pool is handed back, not rebuilt");
        // THE RESET ARM. The pool was filled to capacity before release; only a real
        // vkResetDescriptorPool makes all of it allocatable again.
        check(allocate_sets(first.pool, first.capacity.sets),
              "the handed-back pool is EMPTY: its full capacity allocates again");
        check(first.capacity.sets >= 16 && first.capacity.storage_buffers >= 16,
              "capacities are rounded up so neighbouring passes can share a pool");
    } else {
        check(after_first.hits == 0 && after_first.retired == 0,
              "control: with reuse disabled nothing is retired and nothing hits");
        check(first.capacity.sets == small.sets,
              "control: with reuse disabled the pool is sized exactly, as before");
    }

    // EXCLUSIVE OWNERSHIP: `first` holds the entry that was cached, so a cache that failed to remove
    // what it hands out would return it again here.
    RenderDescriptorPoolLease concurrent = acquire_render_descriptor_pool(ctx.dev, small);
    check(concurrent && concurrent.pool != first.pool,
          "a second acquire while the first is still held returns a DIFFERENT pool");

    // COVERAGE: a requirement larger than anything cached must never receive a smaller pool --
    // vkAllocateDescriptorSets would fail with OUT_OF_POOL_MEMORY mid-pass and drop draws.
    release_render_descriptor_pool(ctx.dev, first);
    const RenderDescriptorPoolCapacity large{first.capacity.sets * 4, first.capacity.storage_buffers * 4,
                                             1, 1};
    RenderDescriptorPoolLease big = acquire_render_descriptor_pool(ctx.dev, large);
    check(static_cast<bool>(big), "a larger requirement is served");
    // Handle identity is evidence only when something guarantees the handle was never freed; with
    // reuse off `first` was destroyed and the driver may legitimately hand the same value back.
    if (reuse)
        check(big.pool != first.pool, "a larger requirement is not given the smaller cached pool");
    check(big.capacity.covers(large), "...and the pool it gets covers the requirement");
    check(allocate_sets(big.pool, large.sets), "...which is real: the requested sets allocate");
    // And the other direction: a small request may take the large pool only if it is the smallest
    // that fits. `first` (small) is cached again, so the small request must get it, not `big`.
    release_render_descriptor_pool(ctx.dev, big);
    RenderDescriptorPoolLease again = acquire_render_descriptor_pool(ctx.dev, small);
    if (reuse)
        check(again.pool == first.pool,
              "the SMALLEST covering pool is chosen, so a large pool is not spent on a small pass");
    release_render_descriptor_pool(ctx.dev, again);
    release_render_descriptor_pool(ctx.dev, concurrent);

    // BOUNDED: release well past the limit and require the surplus to be destroyed.
    const size_t limit = render_descriptor_pool_cache_limit();
    std::vector<RenderDescriptorPoolLease> many;
    for (size_t i = 0; i < limit + 4; ++i) {
        RenderDescriptorPoolLease p = acquire_render_descriptor_pool(ctx.dev, small);
        if (p) many.push_back(p);
    }
    check(many.size() == limit + 4, "enough pools for the bound arm were acquired");
    std::set<VkDescriptorPool> distinct;
    for (const auto& l : many) distinct.insert(l.pool);
    check(distinct.size() == many.size(), "every concurrently-held pool is distinct");
    const auto pre_flood = render_descriptor_pool_stats();
    for (const auto& p : many) release_render_descriptor_pool(ctx.dev, p);
    const auto post_flood = render_descriptor_pool_stats();
    if (reuse) {
        check(post_flood.cached <= limit, "the retained set never exceeds the configured limit");
        check(post_flood.destroyed > pre_flood.destroyed,
              "...and the surplus is destroyed rather than accumulated");
    } else {
        check(post_flood.cached == 0, "control: reuse disabled retains nothing at all");
    }

    drain_render_descriptor_pool_cache(ctx.dev);
    check(render_descriptor_pool_stats().cached == 0, "draining empties the list for this device");
    vkDestroyDescriptorSetLayout(ctx.dev, layout, nullptr);
    std::printf("== %s (%d failure%s) ==\n", failures ? "FAIL" : "PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
