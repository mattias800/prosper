// gpu_resources.cpp — the resource registry behind the front-half↔back-half contract.
//
// This first increment is a real, thread-safe descriptor registry keyed by a stable handle. It is
// deliberately Vulkan-free: it lets the front-half wire the AGC resource-creation path (and unblock
// the eboot+0xba6e08 null-backing fault) against a concrete API immediately, while the back-half
// layers the actual VkImage/VkBuffer creation on top in the next increment. Handles are opaque and
// never recycled within a run, so a stale handle is detectably invalid rather than silently aliased.
#include "gpu_resources.hpp"
#include <deque>
#include <mutex>

namespace prosper::gpu {

namespace {
    std::mutex g_mtx;
    // Slot 0 is reserved so handle 0 stays "invalid"; a handle is simply (index into g_res).
    // deque keeps existing descriptor addresses stable as new resources are appended; callers hold
    // resource_get() pointers after the registry lock is released.
    std::deque<ResourceDesc> g_res = { ResourceDesc{} };
}

ResourceHandle resource_create(const ResourceDesc& desc) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_res.push_back(desc);
    return (ResourceHandle)(g_res.size() - 1);   // >= 1
}

const ResourceDesc* resource_get(ResourceHandle h) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (h == 0 || h >= g_res.size()) return nullptr;
    return &g_res[(size_t)h];
}

bool resource_valid(ResourceHandle h) {
    std::lock_guard<std::mutex> lk(g_mtx);
    return h != 0 && h < g_res.size();
}

size_t resource_count() {
    std::lock_guard<std::mutex> lk(g_mtx);
    return g_res.size() - 1;   // exclude the reserved slot 0
}

void resource_reset() {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_res.clear();
    g_res.push_back(ResourceDesc{});
}

} // namespace prosper::gpu
