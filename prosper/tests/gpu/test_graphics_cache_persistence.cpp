#include <vulkan/vulkan.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

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
#define vkCreatePipelineCache observe_cache_create
#include "fixtures/render_runner.h"
#undef vkCreatePipelineCache
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
    // A held resource domain must skip immediately, not deadlock during _Exit.
    std::atomic<bool> ready{false}, release{false};
    std::thread owner([&] {
        prosper::test::BackendPersistentResourceGuard guard;
        ready.store(true);
        while (!release.load()) std::this_thread::yield();
    });
    while (!ready.load()) std::this_thread::yield();
    const bool saved_while_busy = prosper::test::flush_graphics_pipeline_cache();
    release.store(true);
    owner.join();
    if (saved_while_busy) return 5;
    if (save && !prosper::test::flush_graphics_pipeline_cache()) return 6;
    std::fflush(nullptr);
    std::_Exit(0); // destructor-only persistence must not pass this test
}
