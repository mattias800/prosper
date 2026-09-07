#include <vulkan/vulkan.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

static size_t observed_initial_bytes = 0;
static bool reject_cached = false;
static unsigned cache_create_calls = 0;
static VkResult observe_cache_create(VkDevice device, const VkPipelineCacheCreateInfo* info,
                                    const VkAllocationCallbacks* allocator, VkPipelineCache* cache) {
    observed_initial_bytes = info->initialDataSize;
    ++cache_create_calls;
    if (reject_cached && info->initialDataSize) return VK_ERROR_INITIALIZATION_FAILED;
    return vkCreatePipelineCache(device, info, allocator, cache);
}
namespace prosper::test { std::timed_mutex& graphics_driver_cache_mutex(); }
static bool compilation_was_locked = false;
static VkResult observe_pipeline_create(VkDevice device, VkPipelineCache cache, uint32_t count,
        const VkGraphicsPipelineCreateInfo* info, const VkAllocationCallbacks* allocator,
        VkPipeline* pipelines) {
    std::thread observer([] {
        auto& mutex = prosper::test::graphics_driver_cache_mutex();
        compilation_was_locked = !mutex.try_lock();
        if (!compilation_was_locked) mutex.unlock();
    });
    observer.join();
    return vkCreateGraphicsPipelines(device, cache, count, info, allocator, pipelines);
}
#define vkCreateGraphicsPipelines observe_pipeline_create
#define vkCreatePipelineCache observe_cache_create
#include "fixtures/render_runner.h"
#undef vkCreatePipelineCache
#undef vkCreateGraphicsPipelines
#include "fixtures/spirv_triangle.h"

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const bool load = std::strcmp(argv[1], "load") == 0;
    const bool save = std::strcmp(argv[1], "save") == 0;
    reject_cached = std::strcmp(argv[1], "reject") == 0;
    std::vector<uint32_t> vert(std::begin(kTriVertSpv), std::end(kTriVertSpv));
    std::vector<uint32_t> frag(std::begin(kTriFragSpv), std::end(kTriFragSpv));
    auto pixels = prosper::test::render_triangle_rgba(vert, frag, 32, 32);
    if (pixels.size() != 32 * 32 * 4 || pixels[(16 * 32 + 16) * 4] < 200) return 3;
    if (load != (observed_initial_bytes > 0)) {
        std::fprintf(stderr, "initial cache bytes=%zu expected load=%d\n", observed_initial_bytes, load);
        return 4;
    }
    if (reject_cached && cache_create_calls != 2) return 7;
    if (!compilation_was_locked) return 9;
    if (save) {
        // Reproduce app shutdown during a pass: the broad resource lock must not block saving.
        std::atomic<bool> ready{false}, release{false};
        std::thread owner([&] {
            prosper::test::BackendPersistentResourceGuard guard;
            ready.store(true);
            while (!release.load()) std::this_thread::yield();
        });
        while (!ready.load()) std::this_thread::yield();
        const bool saved_during_pass = prosper::test::flush_graphics_pipeline_cache();
        release.store(true);
        owner.join();
        if (!saved_during_pass) return 5;

        // In-flight driver compilation does serialize extraction, with a bounded wait.
        ready.store(false);
        release.store(false);
        std::thread compiler([&] {
            std::lock_guard<std::timed_mutex> lock(prosper::test::graphics_driver_cache_mutex());
            ready.store(true);
            while (!release.load()) std::this_thread::yield();
        });
        while (!ready.load()) std::this_thread::yield();
        const bool saved_while_compiling = prosper::test::flush_graphics_pipeline_cache(
            std::chrono::milliseconds(1));
        release.store(true);
        compiler.join();
        if (saved_while_compiling) return 6;
        if (!prosper::test::flush_graphics_pipeline_cache()) return 8;
    }
    std::fflush(nullptr);
    std::_Exit(0); // destructor-only persistence must not pass this test
}
