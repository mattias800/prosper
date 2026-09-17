// test_command_pool_reuse — the bounded VkCommandPool free list (#3407).
//
// The property under test is not "rendering still works". It is that a released pool is RESET AND
// HANDED BACK rather than destroyed and rebuilt, that a pool in use is never handed to a second
// caller, and that what is retained is BOUNDED. Destroying a command pool makes RADV tear down its
// command streams, free their buffer objects and release each one's GPU virtual-address range —
// 3.79% of the serial critical path on a routed Grand Theft Auto V window — so the reuse is the
// point and every arm here is about the reuse being real and safe rather than about pixels.
//
// Runs twice from CMake: once normally, once with PROSPER_NO_COMMAND_POOL_REUSE=1. The second run
// is the negative control, and it is a separate PROCESS because the switch is read once into a
// function-local static — arming it mid-run would be read by nothing.
#include "fixtures/render_runner.h"
#include <cstdlib>
#include <set>

using namespace prosper::test;

static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}

int main() {
    const auto& ctx = render_vk_ctx();
    if (!ctx.ok) {
        std::printf("[skip] no Vulkan device\n");
        return 0;
    }
    const bool reuse = std::getenv("PROSPER_NO_COMMAND_POOL_REUSE") == nullptr;
    std::printf("== test_command_pool_reuse (reuse %s) ==\n", reuse ? "ENABLED" : "DISABLED");
    check(render_command_pool_reuse_enabled() == reuse,
          "the mode under test is the mode the switch reports");

    // A CONFIGURATION assertion, and labelled as one. Resetting with
    // VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT hands the pool's blocks back to the driver, so the
    // next recording re-allocates its buffer objects and their VA ranges -- the precise cost this
    // change removes, reinstated in full, with every behavioural arm below still green. Measured:
    // that mutation reddens nothing else in this file. No behavioural arm is possible because the
    // difference is invisible to the application; it exists only as driver BO traffic. So this arm
    // pins the bit and says why, rather than leaving the mutation unobserved.
    check(render_command_pool_reset_flags() == 0,
          "pools are reset WITHOUT releasing resources (the reuse is otherwise inert)");

    // A pool has to be usable for real recording after it comes back, not merely non-null. Each
    // arm below records and submits through the pool it was handed, so a pool returned without a
    // successful reset would surface here rather than in a later, unrelated test.
    auto record_and_submit = [&](const RenderCommandPoolLease& lease) {
        VkCommandBuffer cmd = lease.command;   // the lease owns it; allocating another is the bug
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return false;
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(ctx.dev, &fi, nullptr, &fence) != VK_SUCCESS) return false;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        const bool ok = vkQueueSubmit(ctx.queue, 1, &si, fence) == VK_SUCCESS &&
                        vkWaitForFences(ctx.dev, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
        vkDestroyFence(ctx.dev, fence, nullptr);
        return ok;
    };

    const auto before = render_command_pool_stats();
    RenderCommandPoolLease first = acquire_render_command_pool(ctx.dev, ctx.qfi);
    check(static_cast<bool>(first), "a pool and its command buffer are acquired");
    check(record_and_submit(first), "the acquired pool records and submits");

    // EXCLUSIVE OWNERSHIP. Vulkan requires external synchronisation on a command pool, so handing
    // the same one to two live callers would be a use-after-handout even though nothing crashes
    // immediately. Acquire a second WITHOUT releasing the first.
    RenderCommandPoolLease concurrent = acquire_render_command_pool(ctx.dev, ctx.qfi);
    check(concurrent && concurrent.pool != first.pool && concurrent.command != first.command,
          "a second acquire while the first is still held returns a DIFFERENT pool and buffer");
    release_render_command_pool(ctx.dev, ctx.qfi, concurrent);

    // REUSE. Released, then re-acquired: with reuse on this must be the same handle and a hit;
    // with reuse off it must be a fresh pool and no hit. The two directions are what separate this
    // from an arm that would pass against either implementation.
    release_render_command_pool(ctx.dev, ctx.qfi, first);
    const auto after_release = render_command_pool_stats();
    RenderCommandPoolLease second = acquire_render_command_pool(ctx.dev, ctx.qfi);
    const auto after_reacquire = render_command_pool_stats();
    if (reuse) {
        check(second.pool == first.pool, "a released pool is handed back, not rebuilt");
        // THE ARM THAT WOULD HAVE CAUGHT THE REGRESSION. The first version of this change cached
        // the pool alone and let each caller allocate its own buffer. vkResetCommandPool resets
        // every buffer ever allocated from a pool and leaves them allocated, so the pool grew
        // without bound and reset cost grew with it: 8.7 fps -> 3.5 on the routed window, with
        // vk_common_ResetCommandPool at 41.6% of the critical thread. Every other arm in this file
        // stayed green through that. Requiring the SAME buffer handle back is what pins it.
        check(second.command == first.command,
              "the same command buffer comes back, so the pool never accumulates buffers");
        check(after_reacquire.hits == after_release.hits + 1, "...and it is counted as a hit");
        check(after_release.retired > before.retired, "release retires rather than destroys");
    } else {
        check(after_reacquire.hits == 0 && after_release.retired == 0,
              "control: with reuse disabled nothing is retired and nothing hits");
    }
    check(record_and_submit(second), "the reused pool records and submits again");
    // Repeat the cycle: a pool that accumulates one buffer per reuse would hand back a different
    // handle by now even if the first reuse happened to match.
    bool stable = true;
    for (int i = 0; i < 8 && stable; ++i) {
        release_render_command_pool(ctx.dev, ctx.qfi, second);
        second = acquire_render_command_pool(ctx.dev, ctx.qfi);
        stable = second && second.command == first.command && record_and_submit(second);
    }
    check(stable, "eight further acquire/release cycles keep the same pool AND the same buffer");
    release_render_command_pool(ctx.dev, ctx.qfi, second);

    // BOUNDED. The whole hazard of a free list is trading allocator time for unbounded memory, and
    // each retained pool still owns its command-stream allocations. Release well past the limit and
    // require the surplus to be destroyed rather than kept.
    const size_t limit = render_command_pool_cache_limit();
    std::vector<RenderCommandPoolLease> many;
    for (size_t i = 0; i < limit + 4; ++i) {
        RenderCommandPoolLease p = acquire_render_command_pool(ctx.dev, ctx.qfi);
        if (p) many.push_back(p);
    }
    check(many.size() == limit + 4, "enough pools for the bound arm were acquired");
    std::set<VkCommandPool> distinct;
    for (const auto& l : many) distinct.insert(l.pool);
    check(distinct.size() == many.size(), "every concurrently-held pool is distinct");
    const auto pre_flood = render_command_pool_stats();
    for (const auto& p : many) release_render_command_pool(ctx.dev, ctx.qfi, p);
    const auto post_flood = render_command_pool_stats();
    if (reuse) {
        check(post_flood.cached <= limit, "the retained set never exceeds the configured limit");
        check(post_flood.destroyed > pre_flood.destroyed,
              "...and the surplus is destroyed rather than accumulated");
    } else {
        check(post_flood.cached == 0, "control: reuse disabled retains nothing at all");
    }

    // Two mutations this file deliberately does NOT catch, recorded rather than implied:
    //  * ignoring `queue_family` when matching a cached pool. This device exposes one usable family,
    //    so no arm here can distinguish a correct match from a device-only one.
    //  * handing the same cached pool to two live callers. That aborts the process under validation
    //    rather than failing an assertion, so it is detected but not diagnosed.
    if (failures) { std::printf("== FAIL: %d ==\n", failures); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
