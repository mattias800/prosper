// compute_scanout.cpp — see compute_scanout.hpp (#3915).
#include "shared/present/compute_scanout.hpp"
#include "shared/present/present_blit.hpp"
#include "fixtures/render_runner.h"                  // render_vk_ctx()
#include "gpu/diagnostics/gpu_memory_budget_vk.hpp"  // free_device_memory
#include "gpu/diagnostics/memory_placement_log.hpp"  // allocate_gpu_only_memory
#include "gpu/diagnostics/diag_ratelimit.hpp"
#include "gpu/execute/gpu_execute.hpp"               // shared_present_submit_mutex
#include "gpu/present/videoout_present.hpp"
#include "gpu/texture/tile.hpp"                      // videoout_scanout_tile_mode
#include "host/memory/guest_write_watch.hpp"
#include "diagnostics/exit_census.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace prosper::frontend {
namespace {

constexpr VkFormat kMirrorFormat = VK_FORMAT_R8G8B8A8_UNORM;

struct Mirror {
    VkDevice device = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    uint64_t image_registration = 0;   // lineage identity of this allocation
    uint64_t reservation = 0;          // the dispatch currently allowed to commit
    bool committed = false;
    uint32_t tile_mode = 0;
    uint64_t watch_address = 0, watch_bytes = 0;
    prosper::host::GuestWriteWatch watch;
    ProducerSource producer;
};

struct State {
    std::mutex mutex;
    std::unordered_map<uint64_t, Mirror> mirrors;
    uint64_t next_reservation = 0;
    // Run totals for the exit census (all under `mutex`).
    uint64_t reserved = 0, committed = 0, retired = 0, published = 0;
    uint64_t decisions[16] = {};
};

State& state() {
    static State* s = new State();   // never destroyed: the process-lifetime device outlives it
    // One exit line, so a run can say whether the mirror path engaged at all and, when a flip was
    // not published, which present decision refused it -- on a title that never compute-writes a
    // display buffer it prints nothing.
    static const bool census = [] {
        prosper::diagnostics::register_census("PROSPER_NO_COMPUTE_SCANOUT_CENSUS", [] {
            State& st = *s;
            std::lock_guard lock(st.mutex);
            if (!st.reserved && !st.retired) return false;
            std::fprintf(stderr,
                         "[compute-scanout] RUN TOTAL mirrors reserved=%llu committed=%llu "
                         "retired=%llu published=%llu decisions:",
                         (unsigned long long)st.reserved, (unsigned long long)st.committed,
                         (unsigned long long)st.retired, (unsigned long long)st.published);
            for (size_t i = 0; i < 16; ++i)
                if (st.decisions[i])
                    std::fprintf(stderr, " %s=%llu",
                                 compute_scanout_present_name(static_cast<ComputeScanoutPresent>(i)),
                                 (unsigned long long)st.decisions[i]);
            std::fputc('\n', stderr);
            return true;
        });
        return true;
    }();
    (void)census;
    return *s;
}

void destroy_image(Mirror& m) {
    if (m.image) vkDestroyImage(m.device, m.image, nullptr);
    if (m.memory) prosper::gpu::free_device_memory(m.device, m.memory);
    m.image = VK_NULL_HANDLE;
    m.memory = VK_NULL_HANDLE;
    m.width = m.height = 0;
    m.image_registration = 0;
}

bool ensure_image(Mirror& m, VkDevice device, VkPhysicalDevice phys, uint32_t w, uint32_t h) {
    if (m.image && m.device == device && m.width == w && m.height == h) return true;
    // Dispatches and publishes are synchronous (each waits its own fence) and run on the thread
    // that executes ordered GPU work, so no GPU work can still reference the old image here.
    destroy_image(m);
    m.device = device;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = kMirrorFormat;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &ici, nullptr, &m.image) != VK_SUCCESS) {
        m.image = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, m.image, &req);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    if (prosper::gpu::allocate_gpu_only_memory(device, prosper::gpu::GpuOnlyMemoryClass::PresentSlot,
                                               mp, req.memoryTypeBits, mai,
                                               &m.memory) != VK_SUCCESS ||
        vkBindImageMemory(device, m.image, m.memory, 0) != VK_SUCCESS) {
        destroy_image(m);
        return false;
    }
    m.width = w;
    m.height = h;
    m.image_registration = next_producer_identity();
    return true;
}


// PROSPER_COMPUTE_SCANOUT_VERIFY=1: before each GPU publish of a mirror, read it back and compare it
// byte for byte with what the CPU fallback would have presented for the same flip (the guest buffer,
// de-swizzled). A diagnostic, not a production path: it costs a 33 MB readback and a de-swizzle per
// verified flip, which is exactly the work the mirror exists to avoid. Sampled (the first 16 flips,
// then every 64th). The result prints as `[compute-scanout] VERIFY #n: identical|DIFFERENT ...`.
bool verify_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("PROSPER_COMPUTE_SCANOUT_VERIFY");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

bool read_back_mirror(const Mirror& m, std::vector<uint8_t>& out) {
    const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
    if (!ctx.ok || ctx.dev != m.device) return false;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(m.width) * m.height * 4u;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool ok = false;
    do {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(m.device, &bci, nullptr, &buffer) != VK_SUCCESS) break;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(m.device, buffer, &req);
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mp);
        uint32_t type = UINT32_MAX;
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; ++i)
            if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
                type = i;
        if (type == UINT32_MAX) break;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        if (prosper::gpu::allocate_device_memory(m.device, &mai, &memory) != VK_SUCCESS) break;
        if (vkBindBufferMemory(m.device, buffer, memory, 0) != VK_SUCCESS) break;
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = ctx.qfi;
        if (vkCreateCommandPool(m.device, &pci, nullptr, &pool) != VK_SUCCESS) break;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(m.device, &ai, &cb) != VK_SUCCESS) break;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {m.width, m.height, 1};
        vkCmdCopyImageToBuffer(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1,
                               &region);
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host.buffer = buffer;
        host.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                             nullptr, 1, &host, 0, nullptr);
        if (vkEndCommandBuffer(cb) != VK_SUCCESS) break;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(m.device, &fci, nullptr, &fence) != VK_SUCCESS) break;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        VkResult sr;
        {
            std::unique_lock<std::mutex> qlk(prosper::gpu::shared_present_submit_mutex(),
                                             std::defer_lock);
            if (prosper::gpu::shared_present_active()) qlk.lock();
            sr = vkQueueSubmit(ctx.queue, 1, &si, fence);
        }
        if (sr != VK_SUCCESS) break;
        if (vkWaitForFences(m.device, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) != VK_SUCCESS)
            break;
        void* mapped = nullptr;
        if (vkMapMemory(m.device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) break;
        out.assign(static_cast<const uint8_t*>(mapped), static_cast<const uint8_t*>(mapped) + bytes);
        vkUnmapMemory(m.device, memory);
        ok = true;
    } while (false);
    if (fence) vkDestroyFence(m.device, fence, nullptr);
    if (pool) vkDestroyCommandPool(m.device, pool, nullptr);
    if (buffer) vkDestroyBuffer(m.device, buffer, nullptr);
    if (memory) prosper::gpu::free_device_memory(m.device, memory);
    return ok;
}

void verify_against_guest(const Mirror& m, const VideoOutBufferSnapshot& front) {
    static std::atomic<uint64_t> flips{0};
    const uint64_t n = flips.fetch_add(1) + 1;
    if (n > 16 && n % 64) return;
    std::vector<uint8_t> gpu;
    VideoOutLinearRead guest;
    const bool have_gpu = read_back_mirror(m, gpu);
    const bool have_guest = videoout_read_front_linear(guest) &&
                            guest.metadata.address == front.address;
    if (!have_gpu || !have_guest || gpu.size() != guest.pixels.size()) {
        std::fprintf(stderr, "[compute-scanout] VERIFY #%llu: UNAVAILABLE (gpu=%d guest=%d "
                             "sizes %zu/%zu)\n", (unsigned long long)n, (int)have_gpu,
                     (int)have_guest, gpu.size(), guest.pixels.size());
        return;
    }
    size_t differing = 0, first = SIZE_MAX;
    for (size_t i = 0; i < gpu.size(); ++i)
        if (gpu[i] != guest.pixels[i]) { if (first == SIZE_MAX) first = i; ++differing; }
    std::fprintf(stderr, "[compute-scanout] VERIFY #%llu: %s buffer=0x%llx %ux%u bytes=%zu "
                         "differing=%zu first=%lld padded=%d\n",
                 (unsigned long long)n, differing ? "DIFFERENT" : "identical",
                 (unsigned long long)front.address, m.width, m.height, gpu.size(), differing,
                 first == SIZE_MAX ? -1ll : (long long)first, (int)guest.padded_footprint);
}

} // namespace

bool compute_scanout_enabled() {
#ifdef _WIN32
    // No page-protection watches on Windows (guest_write_watch.hpp): a mirror could never be proven
    // current, so allocating and copying one per dispatch would be pure cost.
    return false;
#else
    static const bool on = std::getenv("PROSPER_NO_COMPUTE_SCANOUT_PRESENT") == nullptr;
    return on;
#endif
}

void compute_scanout_retain_registered(const uint64_t* registered, size_t count) {
    State& s = state();
    std::lock_guard lock(s.mutex);
    for (auto it = s.mirrors.begin(); it != s.mirrors.end();) {
        bool live = false;
        for (size_t i = 0; i < count && !live; ++i) live = registered[i] == it->first;
        if (live) { ++it; continue; }
        // Publishes and dispatches wait their own fences, so nothing still uses this image.
        it->second.watch.reset();
        destroy_image(it->second);
        it = s.mirrors.erase(it);
    }
}

bool compute_scanout_device_shared(VkDevice device) {
    const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
    return device && ctx.ok && ctx.dev == device;
}

ComputeScanoutTarget compute_scanout_begin(VkDevice device, uint64_t address, uint32_t width,
                                           uint32_t height) {
    ComputeScanoutTarget out;
    if (!address) return out;
    State& s = state();
    std::lock_guard lock(s.mutex);
    if (!device || !width || !height) {
        auto it = s.mirrors.find(address);
        if (it != s.mirrors.end() && (it->second.committed || it->second.reservation)) {
            it->second.committed = false;
            it->second.reservation = 0;
            ++s.retired;   // only a mirror that could have been published is retired
        }
        return out;
    }
    Mirror& m = s.mirrors[address];
    // This dispatch replaces the bytes at `address`: whatever happens next, the old mirror no
    // longer describes them, and a failed dispatch must not leave it publishable.
    m.committed = false;
    m.reservation = 0;
    const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
    if (!ctx.ok || ctx.dev != device) return out;
    if (!ensure_image(m, device, ctx.phys, width, height)) return out;
    m.reservation = ++s.next_reservation;
    ++s.reserved;
    out.address = address;
    out.reservation = m.reservation;
    out.image = m.image;
    out.width = width;
    out.height = height;
    return out;
}

void compute_scanout_record_copy(VkCommandBuffer command, VkBuffer staging,
                                 const ComputeScanoutTarget& target) {
    if (!command || !staging || !target.valid()) return;
    // staging holds the canonical row-major result, written either by the image-to-buffer copy
    // (TRANSFER) or by the direct-retile shader (COMPUTE).
    VkBufferMemoryBarrier ready{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ready.srcQueueFamilyIndex = ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ready.buffer = staging;
    ready.offset = 0;
    ready.size = static_cast<VkDeviceSize>(target.width) * target.height * 4u;
    // The mirror is overwritten whole, so its old contents are discarded (UNDEFINED). Its previous
    // reader, a present_blit copy, waited its own fence before this command buffer was submitted.
    VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_dst.srcAccessMask = 0;
    to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image = target.image;
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &ready, 1, &to_dst);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {target.width, target.height, 1};
    vkCmdCopyBufferToImage(command, staging, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &region);
    VkImageMemoryBarrier to_src = to_dst;
    to_src.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_src.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_src);
}

void compute_scanout_commit(const ComputeScanoutTarget& target, uint64_t guest_bytes,
                            uint32_t tile_mode, uint64_t source_submit) {
    if (!target.valid() || !guest_bytes) return;
    State& s = state();
    std::lock_guard lock(s.mutex);
    auto it = s.mirrors.find(target.address);
    if (it == s.mirrors.end()) return;
    Mirror& m = it->second;
    if (m.reservation != target.reservation || m.image != target.image) return;  // superseded
    // The writeback that just completed dirtied the watch. Re-arm it now, so from here on any
    // change to these guest bytes -- CPU, GPU, DMA or host -- makes the mirror unpublishable.
    // A fresh registration can start Dirty: a page another watch already tracks joins it in that
    // page's current (possibly disarmed) state. So a new watch is re-armed once as well.
    if (!m.watch || m.watch_address != target.address || m.watch_bytes != guest_bytes ||
        !m.watch.rearm()) {
        m.watch = prosper::host::GuestWriteWatch::create(target.address, guest_bytes);
        m.watch_address = target.address;
        m.watch_bytes = guest_bytes;
        if (m.watch && m.watch.query() != prosper::host::GuestWriteWatchQuery::Unchanged &&
            !m.watch.rearm())
            m.watch.reset();   // cannot be proven current: publish will answer Unwatched
    }
    m.committed = true;
    m.reservation = 0;
    ++s.committed;
    m.tile_mode = tile_mode;
    m.producer = {};
    if (producer_lineage_enabled())
        m.producer = {m.image_registration,
                      {m.image_registration, next_producer_identity(), source_submit}};
}

void compute_scanout_abort(const ComputeScanoutTarget& target) {
    if (!target.valid()) return;
    State& s = state();
    std::lock_guard lock(s.mutex);
    auto it = s.mirrors.find(target.address);
    if (it == s.mirrors.end() || it->second.reservation != target.reservation) return;
    it->second.reservation = 0;
    it->second.committed = false;
}

ComputeScanoutPublishResult compute_scanout_publish(const VideoOutBufferSnapshot& front,
                                                    uint64_t front_flip,
                                                    const ComputeScanoutPresentInputs& renderer) {
    ComputeScanoutPublishResult result;
    State& s = state();
    std::lock_guard lock(s.mutex);
    ComputeScanoutPresentInputs in = renderer;
    in.front_width = front.width;
    in.front_height = front.height;
    in.front_scanout_tile_mode = prosper::gpu::videoout_scanout_tile_mode(front.tiling_mode, 4);
    auto it = s.mirrors.find(front.address);
    Mirror* m = it != s.mirrors.end() && it->second.committed && it->second.image
        ? &it->second : nullptr;
    in.committed = m != nullptr;
    if (m) {
        in.mirror_width = m->width;
        in.mirror_height = m->height;
        in.mirror_tile_mode = m->tile_mode;
        in.watch_state = static_cast<uint8_t>(m->watch.query());
    }
    result.decision = compute_scanout_present_decision(in);
    if (static_cast<size_t>(result.decision) < 16) ++s.decisions[static_cast<size_t>(result.decision)];
    if (result.decision != ComputeScanoutPresent::Publish || !m) return result;
    if (verify_enabled()) verify_against_guest(*m, front);
    // Held under this module's lock so a concurrent begin() cannot recreate the image mid-copy.
    result.published = present_blit_publish(m->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            kMirrorFormat, m->width, m->height, front_flip,
                                            m->producer, &front);
    if (result.published) ++s.published;
    return result;
}

ComputeScanoutTotals compute_scanout_totals() {
    State& s = state();
    std::lock_guard lock(s.mutex);
    return {s.reserved, s.committed, s.retired, s.published};
}

bool compute_scanout_committed(uint64_t address) {
    State& s = state();
    std::lock_guard lock(s.mutex);
    auto it = s.mirrors.find(address);
    return it != s.mirrors.end() && it->second.committed;
}

void compute_scanout_reset_for_test() {
    State& s = state();
    std::lock_guard lock(s.mutex);
    for (auto& [address, m] : s.mirrors) {
        (void)address;
        m.watch.reset();
        destroy_image(m);
    }
    s.mirrors.clear();
}

} // namespace prosper::frontend
