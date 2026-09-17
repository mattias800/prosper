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

    // WARM THE FREE LIST FIRST. This is load-bearing for the exclusive-ownership arm below, and the
    // reason is worth stating because the arm looked sound without it: against an EMPTY cache both
    // acquires take the create path, so the arm would assert only that vkCreateCommandPool returns
    // distinct handles and would stay green with the cache's erase (render_runner.h) deleted. It has
    // to be the CACHED path that is asked to hand out each entry at most once.
    RenderCommandPoolLease warm = acquire_render_command_pool(ctx.dev, ctx.qfi);
    check(static_cast<bool>(warm), "a pool and its command buffer are acquired");
    check(record_and_submit(warm), "the acquired pool records and submits");
    release_render_command_pool(ctx.dev, ctx.qfi, warm);

    const auto after_warm = render_command_pool_stats();
    RenderCommandPoolLease first = acquire_render_command_pool(ctx.dev, ctx.qfi);
    const auto after_first = render_command_pool_stats();
    check(static_cast<bool>(first), "a second acquire after that release succeeds");
    if (reuse) {
        // The arm that detects its own invalidity: if this is a miss the free list was not warm and
        // the ownership arm below has silently degraded into the void version described above.
        check(after_first.hits == after_warm.hits + 1,
              "the free list really is warm, so the ownership arm exercises the CACHED path");
    }

    // EXCLUSIVE OWNERSHIP. Vulkan requires external synchronisation on a command pool, so handing
    // the same one to two live callers would be a use-after-handout even though nothing crashes
    // immediately. Acquire a second WITHOUT releasing the first: `first` holds the entry that was
    // cached, so a cache that failed to remove what it hands out would return it again here.
    RenderCommandPoolLease concurrent = acquire_render_command_pool(ctx.dev, ctx.qfi);
    check(concurrent && concurrent.pool != first.pool && concurrent.command != first.command,
          "a second acquire while the first is still held returns a DIFFERENT pool and buffer");

    // REUSE. Released, then re-acquired: with reuse on this must be the same handle and a hit;
    // with reuse off it must be a fresh pool and no hit. The two directions are what separate this
    // from an arm that would pass against either implementation.
    //
    // `concurrent` is deliberately still HELD here rather than released above, so that exactly one
    // entry is in the free list when `second` is acquired. Releasing it first would leave two, and
    // then "the pool that comes back is `first`" would silently depend on the cache being scanned
    // newest-first -- an implementation detail, not the property under test. A correct cache that
    // scanned oldest-first would fail the arm. It is released after the cycles below.
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
        // The identity requirement is the REUSE property, so it must not be asserted against the
        // control, where every acquire builds a fresh pool. It was unconditional once, and passed
        // locally while failing in CI.
        //
        // Do NOT "fix" that by making it unconditional again and blaming one driver. Whether a
        // destroyed handle comes back with the same value is not a property of the no-reuse path at
        // all: it depends on the driver, on whether the validation layer is loaded, AND on the
        // allocation sequence ahead of it. Measured here on this file, reuse off, identity asserted
        // unconditionally, 10 runs per cell, deterministic in every cell -- RADV without the layer
        // FAILS, RADV with the layer PASSES, lavapipe passes either way. Change the sequence above
        // and the cells move. So every green run of the unconditional version was an accident of the
        // allocator, in whichever direction it happened to fall that day.
        //
        // A handle equality is evidence only when something guarantees the handle was never freed.
        // Here that guarantee is the free list, and it exists only when reuse is on.
        stable = second && record_and_submit(second) && (!reuse || second.command == first.command);
    }
    check(stable, reuse
          ? "eight further acquire/release cycles keep the same pool AND the same buffer"
          : "control: eight further cycles still record and submit with reuse disabled");
    release_render_command_pool(ctx.dev, ctx.qfi, second);
    release_render_command_pool(ctx.dev, ctx.qfi, concurrent);   // held since the ownership arm

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

    // What this file does and does not catch. Every line below was produced by APPLYING the mutation
    // and running it, not by reading the code and reasoning about it -- an earlier version of this
    // comment was wrong in both directions, which is the whole reason it is now measured.
    //
    //  * NOT CAUGHT HERE: making vkResetCommandPool a no-op (`if (true)` at its call site). All
    //    arms in this file stay green, because the difference is not observable through the handles
    //    or the counters. It IS caught by Vulkan validation, which is part of this change's merge
    //    gate: the pool is created WITHOUT VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, so
    //    re-recording a buffer that was never reset violates
    //    VUID-vkBeginCommandBuffer-commandBuffer-00050. Measured: 9 such messages from this test
    //    under tools/vkval/vk_validation_scan.py, and zero with the reset restored.
    //  * CAUGHT, and by design: handing the same cached pool to two live callers (deleting the
    //    free list's `erase`). The exclusive-ownership arm above fails on it. Note the run ALSO
    //    aborts later, inside the bound arm, once a duplicated entry is destroyed while another
    //    copy is still cached -- and the abort discards buffered stdout, so a plain run shows no
    //    [FAIL] line at all. Read this test's failures with `stdbuf -o0` before concluding which
    //    arm caught what.
    //  * GENUINELY NOT CAUGHT: ignoring `queue_family` when matching a cached pool. This device
    //    exposes one usable family, so no arm here can distinguish a correct match from a
    //    device-only one.
    if (failures) { std::printf("== FAIL: %d ==\n", failures); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
