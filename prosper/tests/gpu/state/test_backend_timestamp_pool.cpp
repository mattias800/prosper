// BackendSubmissionBatch's timestamp envelope reuses its query pool across batches (#3948 stage 0).
//
// The envelope is armed on every default run, so a create/destroy per batch would put ~500 Vulkan
// query-pool pairs per second on the executor thread it measures. These arms pin the reuse:
//   * a batch that records the envelope and completes returns its pool, and the next batch gets the
//     SAME pool without creating one;
//   * a batch that never reached the queue (discard) also returns it;
//   * the free list is bounded: pools beyond the cap are destroyed, not hoarded.
#include "fixtures/render_runner.h"

#include <cstdio>
#include <vector>

using namespace prosper::test;

namespace {
int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("[FAIL] %s\n", m); ++failures; } \
                        else std::printf("[ok] %s\n", m); } while (0)

struct Recorder {
    VkDevice dev;
    VkCommandPool pool = VK_NULL_HANDLE;
    explicit Recorder(const RenderVkCtx& ctx) : dev(ctx.dev) {
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = ctx.qfi;
        vkCreateCommandPool(dev, &info, nullptr, &pool);
    }
    ~Recorder() { if (pool) vkDestroyCommandPool(dev, pool, nullptr); }
    VkCommandBuffer begin() {
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(dev, &alloc, &cmd);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &begin);
        return cmd;
    }
};

// One batch: envelope around an empty command buffer, then submitted (or discarded). Returns the
// device time the batch measured (negative when no sample).
double run_batch(const RenderVkCtx& ctx, Recorder& rec, bool submit) {
    BackendSubmissionBatch batch;
    VkCommandBuffer cmd = rec.begin();
    batch.begin_gpu_timestamp(ctx.dev, cmd, ctx.timestamp_period_ns, ctx.timestamp_valid_bits);
    batch.end_gpu_timestamp(cmd);
    vkEndCommandBuffer(cmd);
    batch.enqueue(cmd);
    if (!submit) {
        batch.discard();
        return -1.0;
    }
    const BackendSubmissionBatchResult r = batch.submit_and_wait(ctx.dev, ctx.queue, false);
    return r.gpu_timestamp_samples ? r.gpu_device_ms : -1.0;
}
}  // namespace

int main() {
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok || !ctx.timestamp_valid_bits || ctx.timestamp_period_ns <= 0.0) {
        std::printf("SKIP: no Vulkan device with timestamps\n");
        return 77;
    }
    Recorder rec(ctx);
    auto& cache = backend_timestamp_pool_cache();
    const uint64_t created0 = cache.created;
    const double first = run_batch(ctx, rec, true);
    CHECK(first >= 0.0, "a submitted batch with an envelope reads its device time");
    CHECK(cache.created == created0 + 1, "the first envelope creates one pool");
    CHECK(cache.free.size() == 1, "...which the completed batch returns to the free list");
    for (int i = 0; i < 50; ++i) run_batch(ctx, rec, true);
    CHECK(cache.created == created0 + 1 && cache.free.size() == 1,
          "fifty more submitted batches reuse that pool and create none");
    run_batch(ctx, rec, false);
    CHECK(cache.created == created0 + 1 && cache.free.size() == 1,
          "a discarded batch (never queued) returns its pool as well");
    std::vector<VkQueryPool> held;
    for (int i = 0; i < 20; ++i) held.push_back(acquire_backend_timestamp_pool(ctx.dev));
    for (VkQueryPool p : held) return_backend_timestamp_pool(ctx.dev, p);
    CHECK(cache.free.size() == 16, "the free list is bounded at 16; extra pools are destroyed");
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
