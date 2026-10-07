// test_compute_scanout — GPU present of a display buffer written by a COMPUTE dispatch (#3915).
//
// Sonic Frontiers writes both of its display buffers with a compute storage-image store, so the
// renderer holds no render target at the flipped address and every flip used to go through the CPU
// fallback. This drives the real path end to end on a real device:
//   * a VideoOut buffer registered through the HLE, backed by a dmem-style memfd mapping so the
//     guest write watch can arm over it;
//   * a recompiled compute store that writes a per-texel pattern into that buffer as a storage image,
//     executed by the live compute backend with GPU present active;
//   * compute_scanout_publish, then the present slot read back and compared with the guest bytes the
//     CPU fallback would have presented.
// The store goes through the T# DST_SEL Sonic's display buffers use, (Z,Y,X,W), into the Gen5 display
// format, which is BGRA in guest memory: guest bytes are B,G,R,A and both present paths must show the
// R,G,B the shader computed (#4686: they used to show the guest bytes as RGBA, red and blue swapped).
// Then the arms that must NOT publish: a guest CPU store, a GPU write and a host write after the
// commit (each makes the mirror stale), a tile-mode mismatch, a renderer-owned source, and a dispatch
// with GPU present inactive.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/present/videoout_present.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "host/image/exec_image.hpp"
#include "host/memory/guest_write_watch.hpp"
#include "fixtures/render_runner.h"
#include "shared/live/live_compute.hpp"
#include "shared/live/live_renderer.hpp"
#include "shared/present/compute_scanout.hpp"
#include "shared/present/present_blit.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper;
using namespace prosper::gpu;

namespace {
int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); ++fails; } } while (0)

using Hle8Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t, uint64_t);

constexpr uint32_t W = 32, H = 32;   // one 32x32 workgroup: v0/v1 are the local ids
constexpr size_t kBytes = size_t{W} * H * 4;

// R = x, G = y, B = 0, A = 255 -- the colour the compute store below computes, and so the RGBA a
// present path must show. Distinct per row and column, so a transposed, row-shifted or retiled copy
// cannot match.
uint8_t expected_byte(size_t offset) {
    const size_t texel = offset / 4;
    switch (offset % 4) {
    case 0: return static_cast<uint8_t>(texel % W);
    case 1: return static_cast<uint8_t>(texel / W);
    case 2: return 0;
    default: return 255;
    }
}

// The same texel as it sits in guest memory: the store's DST_SEL (Z,Y,X,W) puts B in byte 0.
uint8_t expected_guest_byte(size_t offset) {
    const size_t component = offset % 4;
    return expected_byte(offset - component +
                         (component == 0   ? 2
                          : component == 2 ? 0
                                           : component));
}

bool read_back_slot(const frontend::GpuScanoutFrame& frame, std::vector<uint8_t>& out) {
    const test::RenderVkCtx& ctx = test::render_vk_ctx();
    const VkDeviceSize bytes = VkDeviceSize{frame.width} * frame.height * 4;
    VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    bool ok = false;
    do {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(ctx.dev, &bci, nullptr, &buffer) != VK_SUCCESS) break;
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(ctx.dev, buffer, &req);
        VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mp);
        uint32_t type = UINT32_MAX;
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; ++i)
            if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
                type = i;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size; mai.memoryTypeIndex = type;
        if (type == UINT32_MAX || vkAllocateMemory(ctx.dev, &mai, nullptr, &memory) != VK_SUCCESS ||
            vkBindBufferMemory(ctx.dev, buffer, memory, 0) != VK_SUCCESS) break;
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = ctx.qfi;
        if (vkCreateCommandPool(ctx.dev, &pci, nullptr, &pool) != VK_SUCCESS) break;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(ctx.dev, &ai, &cb) != VK_SUCCESS) break;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cb, &bi);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {frame.width, frame.height, 1};
        vkCmdCopyImageToBuffer(cb, frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1,
                               &region);
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host.buffer = buffer; host.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                             nullptr, 1, &host, 0, nullptr);
        vkEndCommandBuffer(cb);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(ctx.dev, &fci, nullptr, &fence) != VK_SUCCESS) break;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        if (vkQueueSubmit(ctx.queue, 1, &si, fence) != VK_SUCCESS ||
            vkWaitForFences(ctx.dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) != VK_SUCCESS)
            break;
        void* mapped = nullptr;
        if (vkMapMemory(ctx.dev, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) break;
        out.assign(static_cast<const uint8_t*>(mapped), static_cast<const uint8_t*>(mapped) + bytes);
        vkUnmapMemory(ctx.dev, memory);
        ok = true;
    } while (false);
    if (fence) vkDestroyFence(ctx.dev, fence, nullptr);
    if (pool) vkDestroyCommandPool(ctx.dev, pool, nullptr);
    if (buffer) vkDestroyBuffer(ctx.dev, buffer, nullptr);
    if (memory) vkFreeMemory(ctx.dev, memory, nullptr);
    return ok;
}

// The inputs a renderer with no source of its own supplies at an unscaled present.
frontend::ComputeScanoutPresentInputs guest_only_inputs() {
    frontend::ComputeScanoutPresentInputs in;
    in.present_extent_bytes = kBytes;
    in.display_bytes = kBytes;
    return in;
}
} // namespace

int main() {
#ifdef _WIN32
    std::puts("SKIP: guest write watches are unsupported on Windows; the mirror stays unwatched");
    return 0;
#else
    std::printf("== test_compute_scanout ==\n");
    setenv("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB", "0", 1);
    const test::RenderVkCtx& ctx = test::render_vk_ctx();   // publishes the shared device
    if (!ctx.ok) { std::puts("FAIL: no render device"); return 1; }
    frontend::register_live_renderer(".", false);           // registers the live compute backend

    // ---- A dmem-style mapping for the display buffer, so the guest write watch can arm over it.
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t map_bytes = (kBytes + page - 1) / page * page + page;
    const int fd = memfd_create("prosper-compute-scanout", 0);
    CHECK(fd >= 0 && ftruncate(fd, static_cast<off_t>(map_bytes)) == 0, "memfd backing created");
    auto* guest = static_cast<uint8_t*>(
        mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    CHECK(guest != MAP_FAILED, "display buffer mapped");
    if (fd < 0 || guest == MAP_FAILED) return 1;
    std::memset(guest, 0x9d, map_bytes);
    install_trap_handler();
    host::guest_write_watch_set_fault_onstack(true);
    host::guest_write_watch_notify_direct_mapping_added(reinterpret_cast<uint64_t>(guest), map_bytes,
                                                        0x40000000ull, 0x3 /*CPU_READ|WRITE*/);
    const uint64_t address = reinterpret_cast<uint64_t>(guest);

    // ---- Register it as a LINEAR VideoOut buffer and flip it, exactly as the guest does.
    register_builtin_hle();
    auto open = Hle::lookup(nid_hash("sceVideoOutOpen"));
    auto setba2 = reinterpret_cast<Hle8Fn>(Hle::lookup("PjS5uASwcV8"));   // SetBufferAttribute2
    auto regb2 = Hle::lookup("rKBUtgRrtbk");                              // RegisterBuffers2
    auto flip = Hle::lookup(nid_hash("sceVideoOutSubmitFlip"));
    CHECK(open && setba2 && regb2 && flip, "VideoOut entry points registered");
    if (!(open && setba2 && regb2 && flip)) return 1;
    const uint64_t handle = open(0, 0, 0, 0, 0, 0);
    uint8_t attr[0x50];
    setba2((uint64_t)(uintptr_t)attr, 0x8000000000000000ull, 1 /*LINEAR*/, W, H, 0, 0, 0);
    struct VOB { const void* data; const void* metadata; const void* reserved[2]; };
    VOB buffers[1] = {{guest, nullptr, {nullptr, nullptr}}};
    CHECK(regb2(handle, 0, 0, (uint64_t)(uintptr_t)buffers, 1, (uint64_t)(uintptr_t)attr) == 0,
          "display buffer registered");

    // ---- The compute store: image_store (x/255, y/255, 0, 1) at (x, y) through s[8:15].
    static const uint32_t store_pattern[] = {
        0x7E080300u, 0x7E0A0301u,          // v4 = x, v5 = y
        0x7E000D04u, 0x7E020D05u,          // v0 = float(v4), v1 = float(v5)
        0x100000FFu, std::bit_cast<uint32_t>(1.0f / 255.0f),   // v0 *= 1/255
        0x100202FFu, std::bit_cast<uint32_t>(1.0f / 255.0f),   // v1 *= 1/255
        0x7E040280u, 0x7E0602F2u,          // v2 = 0, v3 = 1.0
        0xF0200F08u, 0x00020004u,          // image_store v[0:3] at v4, v5
        0xBF810000u,
    };
    ShaderResource output{};
    output.cls = ResourceClass::StorageImage; output.format = DataFormat::Unorm8;
    output.num_components = 4; output.binding = 5; output.sgpr_base = 8;
    output.img_dim = 1; output.width = W; output.height = H; output.depth = 1;
    output.gpu_addr = address; output.size = static_cast<uint32_t>(kBytes);
    output.swizzle[0] = 6;
    output.swizzle[1] = 5;
    output.swizzle[2] = 4;
    output.swizzle[3] = 7;
    ShaderResourceTable table; table.resources.push_back(output);
    ComputeShaderConfig config;
    config.user_sgprs.resize(16); config.local_x = W; config.local_y = H; config.local_z = 1;
    config.threads_x = W; config.threads_y = H; config.threads_z = 1; config.tidig_comp_cnt = 1;
    config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Unorm8, 4);
    const auto spirv = recompile_compute(store_pattern, std::size(store_pattern), &table, config);
    CHECK(!spirv.empty(), "pattern store recompiles");
    ComputeItem item;
    item.spirv = spirv; item.resources = std::make_shared<ShaderResourceTable>(table);
    item.launch.threads_x = W; item.launch.threads_y = H; item.launch.threads_z = 1;
    item.launch.local_x = W; item.launch.local_y = H; item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    item.user_sgprs.assign(16, 0);   // the store reads its SGPRs through push constants (#1710)
    item.code_addr = 0x39150001u; item.submit_no = 7;

    auto guest_matches_pattern = [&] {
        for (size_t i = 0; i < kBytes; ++i)
            if (guest[i] != expected_guest_byte(i)) return false;
        return true;
    };
    auto front_after_flip = [&](VideoOutBufferSnapshot& front) {
        return flip(handle, 0, 0, 0, 0, 0) == 0 && videoout_front_snapshot(front) &&
               front.address == address;
    };

    // ---- GPU present inactive (every headless frontend): no mirror is made.
    set_gpu_present_active(false);
    CHECK(frontend::execute_live_compute_items({item}), "dispatch completes with GPU present off");
    CHECK(guest_matches_pattern(), "the dispatch wrote the pattern into guest memory");
    CHECK(!frontend::compute_scanout_committed(address),
          "no mirror is made while GPU present is inactive (headless paths unchanged)");

    // ---- GPU present active: the dispatch leaves a mirror, and the flip publishes it.
    set_gpu_present_active(true);
    std::memset(guest, 0x9d, kBytes);
    CHECK(frontend::execute_live_compute_items({item}), "dispatch completes with GPU present on");
    CHECK(guest_matches_pattern(), "guest bytes carry the pattern after the mirrored dispatch");
    CHECK(frontend::compute_scanout_committed(address),
          "a complete result for a registered display buffer commits a GPU mirror");
    VideoOutBufferSnapshot front;
    CHECK(front_after_flip(front), "flipped the compute-written display buffer");

    {
        frontend::ComputeScanoutPresentInputs renderer = guest_only_inputs();
        renderer.have_selected_pixels = true;
        const auto r = frontend::compute_scanout_publish(front, front.source_flip_seq, renderer);
        CHECK(r.decision == frontend::ComputeScanoutPresent::RendererSource && !r.published,
              "a renderer-selected present source keeps priority over the mirror");
    }
    {
        // A renderer entry at the front address -- here a tombstone, no pixels anywhere -- makes the
        // CPU path keep the previous frame (SkipRendererOwnsTarget). The mirror is the guest bytes,
        // so it must not stand in either, however current it is (#3924 review).
        frontend::ComputeScanoutPresentInputs renderer = guest_only_inputs();
        renderer.renderer_owns_front = true;
        const auto r = frontend::compute_scanout_publish(front, front.source_flip_seq, renderer);
        CHECK(r.decision == frontend::ComputeScanoutPresent::RendererOwnsTarget && !r.published,
              "a renderer tombstone at the front address never publishes the mirror");
        frontend::GpuScanoutFrame none;
        CHECK(!frontend::present_blit_acquire(none), "...and nothing reached a present slot");
    }
    {
        VideoOutBufferSnapshot tiled = front;
        tiled.tiling_mode = 0;   // VideoOut "tiled": de-swizzled, unlike this linear result
        const auto r = frontend::compute_scanout_publish(tiled, front.source_flip_seq,
                                                         guest_only_inputs());
        CHECK(r.decision == frontend::ComputeScanoutPresent::TileMismatch && !r.published,
              "a mirror is never presented where VideoOut would de-swizzle differently");
    }
    {
        VideoOutBufferSnapshot rgba = front;
        rgba.pixel_format = 0x80002200u;   // Gen4 A8B8G8R8: the mirror's BGRA order would be wrong
        const auto r =
            frontend::compute_scanout_publish(rgba, front.source_flip_seq, guest_only_inputs());
        CHECK(r.decision == frontend::ComputeScanoutPresent::FormatMismatch && !r.published,
              "a mirror is never presented for a buffer whose pixel format is not the one it took");
    }
    const auto published = frontend::compute_scanout_publish(front, front.source_flip_seq,
                                                             guest_only_inputs());
    CHECK(published.decision == frontend::ComputeScanoutPresent::Publish && published.published,
          "the flip of a compute-written display buffer publishes through GPU present");
    frontend::GpuScanoutFrame frame;
    std::vector<uint8_t> shown;
    const bool acquired = frontend::present_blit_acquire(frame);
    CHECK(acquired && frame.width == W && frame.height == H && read_back_slot(frame, shown),
          "the published scanout slot is acquirable and readable");
    if (acquired) frontend::present_blit_release(frame.slot);
    VideoOutLinearRead cpu_path;
    CHECK(videoout_read_front_linear(cpu_path), "the CPU fallback's frame is readable");
    CHECK(shown.size() == kBytes && shown == cpu_path.pixels,
          "the GPU-presented frame is byte-identical to what the CPU fallback presents");
    bool pattern = shown.size() == kBytes;
    for (size_t i = 0; pattern && i < kBytes; ++i) pattern = shown[i] == expected_byte(i);
    CHECK(pattern,
          "the GPU-presented frame is the colour the dispatch computed, not its BGRA bytes");
    bool cpu_pattern = cpu_path.pixels.size() == kBytes;
    for (size_t i = 0; cpu_pattern && i < kBytes; ++i)
        cpu_pattern = cpu_path.pixels[i] == expected_byte(i);
    CHECK(cpu_pattern,
          "the CPU fallback also presents the computed colour from the BGRA guest bytes");

    // ---- After the commit, any change to the guest bytes makes the mirror unpublishable.
    auto stale_after = [&](const char* what, auto&& mutate) {
        std::memset(guest, 0x9d, kBytes);
        CHECK(frontend::execute_live_compute_items({item}) &&
                  frontend::compute_scanout_committed(address), "re-dispatch re-commits the mirror");
        VideoOutBufferSnapshot f;
        CHECK(front_after_flip(f), "flipped again");
        const auto fresh = frontend::compute_scanout_publish(f, f.source_flip_seq,
                                                             guest_only_inputs());
        CHECK(fresh.decision == frontend::ComputeScanoutPresent::Publish,
              "a re-armed watch publishes the re-committed mirror");
        frontend::GpuScanoutFrame drained;
        if (frontend::present_blit_acquire(drained)) frontend::present_blit_release(drained.slot);
        mutate();
        VideoOutBufferSnapshot g;
        CHECK(front_after_flip(g), "flipped after the mutation");
        const auto r = frontend::compute_scanout_publish(g, g.source_flip_seq, guest_only_inputs());
        std::printf("  %s -> %s\n", what, frontend::compute_scanout_present_name(r.decision));
        CHECK(r.decision == frontend::ComputeScanoutPresent::Stale && !r.published, what);
    };
    stale_after("a guest CPU store after the commit makes the mirror stale",
                [&] { reinterpret_cast<volatile uint8_t*>(guest)[W * 4 * 3 + 1] = 0x42; });
    stale_after("a GPU/DMA write after the commit makes the mirror stale",
                [&] { host::guest_write_watch_notify_gpu_write(address + kBytes / 2, 16); });
    stale_after("a host write into dmem after the commit makes the mirror stale", [&] {
        host::guest_write_watch_notify_host_write(address + 64, 4);
        std::memset(guest + 64, 0, 4);
        host::guest_write_watch_notify_host_write_done(address + 64, 4);
    });

    // ---- A dispatch that stops being eligible retires the old mirror (Absent, not a stale publish).
    {
        CHECK(frontend::execute_live_compute_items({item}) &&
                  frontend::compute_scanout_committed(address), "mirror committed before retirement");
        ComputeItem partial = item;   // a partial store: not the whole exact result any more
        auto partial_table = std::make_shared<ShaderResourceTable>(table);
        partial_table->resources[0].linear_row_pitch_bytes = W * 4;
        partial.resources = partial_table;
        (void)frontend::execute_live_compute_items({partial});
        CHECK(!frontend::compute_scanout_committed(address),
              "an ineligible write to the display buffer retires its mirror");
    }

    // ---- Retirement is counted only for a mirror that could have been published.
    {
        const auto before = frontend::compute_scanout_totals();
        frontend::compute_scanout_begin(VK_NULL_HANDLE, address + 0x1000000, 0, 0);   // never mirrored
        CHECK(frontend::compute_scanout_totals().retired == before.retired,
              "retiring an address that never had a mirror is not counted");
        frontend::compute_scanout_begin(VK_NULL_HANDLE, address, 0, 0);   // no committed mirror now
        CHECK(frontend::compute_scanout_totals().retired == before.retired,
              "retiring an already-retired mirror is not counted again");
    }

    // ---- A mirror whose address is no longer a registered display buffer is freed.
    CHECK(frontend::execute_live_compute_items({item}) && frontend::compute_scanout_committed(address),
          "mirror committed before unregistration");
    frontend::compute_scanout_retain_registered(nullptr, 0);
    CHECK(!frontend::compute_scanout_committed(address),
          "a mirror for an address no longer registered is freed");

    set_gpu_present_active(false);
    host::guest_write_watch_notify_direct_mapping_removed(address, map_bytes);
    frontend::compute_scanout_reset_for_test();
    frontend::present_blit_reset();
    munmap(guest, map_bytes);
    close(fd);
    std::printf(fails ? "== FAIL (%d) ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
#endif
}
