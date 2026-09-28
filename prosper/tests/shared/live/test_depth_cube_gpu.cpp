// GPU-resident retained depth cube (#3873 plan item 5, tests/fixtures/retained_depth_cube_gpu.h).
//
// Three questions, each with an arm that fails if the answer is wrong:
//   1. EXACTNESS. The GPU gather + compute quantise produces the same bytes as the CPU bridge
//      (read the six faces back, quantise with depth_cube_quantize.hpp) for every texel of a cube
//      seeded with rounding-boundary depths (every (k+0.5)/255 and its float neighbours) and
//      random depths.
//   2. DOMAIN. The conversion alone matches the CPU quantiser for values no depth attachment can
//      hold (NaN, infinities, negatives, > 1, subnormals, random bit patterns).
//   3. ROUTING. Through the live renderer, a fully renderer-owned cube is served by the GPU route
//      (one gather per ordered batch, zero CPU decodes, same pixels), while a mixed cube, a forced
//      fallback and the PROSPER_NO_GPU_DEPTH_CUBE control stay on the CPU bridge with identical output.
// `--control` runs with PROSPER_NO_GPU_DEPTH_CUBE=1 and inverts the routing expectations.
#include "fixtures/retained_depth_cube_gpu.h"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/depth_cube_quantize.hpp"
#include "shared/live/live_renderer.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::frontend;
using namespace prosper::test;

static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}

// One-shot host-visible transfer: `record` fills the command buffer, the buffer is returned mapped.
struct HostBuffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    ~HostBuffer() {
        if (mapped) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
};
static bool make_host_buffer(const RenderVkCtx& ctx, VkDeviceSize bytes, HostBuffer& out) {
    out.device = ctx.dev;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(ctx.dev, &info, nullptr, &out.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.dev, out.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(ctx.phys, &properties);
    const VkMemoryPropertyFlags wanted =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < properties.memoryTypeCount && type == UINT32_MAX; ++i)
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (properties.memoryTypes[i].propertyFlags & wanted) == wanted)
            type = i;
    if (type == UINT32_MAX) return false;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    if (vkAllocateMemory(ctx.dev, &allocation, nullptr, &out.memory) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(ctx.dev, out.buffer, out.memory, 0) != VK_SUCCESS) return false;
    return vkMapMemory(ctx.dev, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped) == VK_SUCCESS;
}
template <class Record>
static bool submit_now(const RenderVkCtx& ctx, Record record) {
    const RenderCommandPoolLease lease = acquire_render_command_pool(ctx.dev, ctx.qfi);
    if (!lease) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(lease.command, &begin);
    record(lease.command);
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(lease.command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(lease.command);
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(ctx.dev, &fence_info, nullptr, &fence);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &lease.command;
    const bool ok = render_locked_queue_submit(ctx.queue, 1, &submit, fence) == VK_SUCCESS &&
        vkWaitForFences(ctx.dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) == VK_SUCCESS;
    vkDestroyFence(ctx.dev, fence, nullptr);
    release_render_command_pool(ctx.dev, ctx.qfi, lease);
    return ok;
}

int main(int argc, char** argv) {
    const bool control = argc == 2 && std::strcmp(argv[1], "--control") == 0;
    if (control) setenv("PROSPER_NO_GPU_DEPTH_CUBE", "1", 1);
    else unsetenv("PROSPER_NO_GPU_DEPTH_CUBE");
    setenv("PROSPER_GPU_REPLAY_DS_SEEDS", "1", 1);
    check(gpu_depth_cube_snapshots_enabled() == !control,
          "the selected retained-cube route is the one this arm asked for");
    prosper::register_builtin_hle();
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    uint64_t guest = 0;
    constexpr uint64_t GuestBytes = 0x400000;
    check(map && unmap && map(reinterpret_cast<uint64_t>(&guest), GuestBytes, 2, 0,
                             reinterpret_cast<uint64_t>("cube-gpu"), 0) == 0 && guest,
          "fixture maps real guest backing");
    if (!guest) return 1;
    std::memset(reinterpret_cast<void*>(guest), 0x55, GuestBytes);

    // The live-renderer fixture: one cube T# (Z16 as GTA V / Outer Wilds program it) sampled at a
    // fixed texel of a selectable face by a recompiled fragment shader.
    constexpr uint32_t W = 64, H = 64;
    const size_t face_bytes = tiled_surface_bytes(W, H, 5, 0, 2);
    const size_t stride = face_bytes;
    check(6 * stride <= GuestBytes, "six guest faces fit the fixture backing");
    auto table = std::make_shared<ShaderResourceTable>();
    ShaderResource source{};
    source.cls = ResourceClass::Texture; source.format = DataFormat::Unorm16;
    source.num_components = 1; source.binding = 4; source.sgpr_base = 8;
    source.img_dim = 3; source.depth = 6; source.width = W; source.height = H;
    source.tile_mode = 5; source.layer_stride_bytes = stride;
    source.gpu_addr = guest; source.size = 6 * stride;
    source.mag_filter = source.min_filter = 0;
    table->resources.push_back(source);
    const uint32_t vs[]{0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u,
        0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u,
        0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u};
    DrawItem draw;
    draw.vs = recompile_vertex(vs, std::size(vs)); draw.vertex_count = 3;
    draw.ps.topology = 3; draw.ps.color_write_mask = 15;
    draw.color0_base = 0x7b000000;
    constexpr uint32_t TW = 8, TH = 8;
    draw.color0_width = TW; draw.color0_height = TH; draw.prt = table;
    register_live_renderer(".", false);
    auto select_face = [&](unsigned face) {
        const uint32_t ps[]{0x7e0002ffu, 0x3fc00000u, 0x7e0202ffu, 0x3fc00000u,
            0x7e0402ffu, std::bit_cast<uint32_t>(float(face)), 0xf09c0f18u, 0x00820000u,
            0xf800080fu, 0x03020100u, 0xbf810000u};
        draw.fs = recompile_fragment(ps, std::size(ps), table.get());
        check(!draw.fs.empty(), "cube face shader recompiles");
    };
    auto center = [&](const std::vector<uint8_t>& image) -> int {
        if (image.size() != TW * TH * 4) return -1;
        const size_t at = (TW * (TH / 2) + TW / 2) * 4;
        if (image[at] != image[at + 1] || image[at] != image[at + 2] || image[at + 3] != 255)
            return -2;
        return image[at];
    };

    // Seed `count` faces; face f's texel i takes value(f, i). Marks each seeded face's depth-write
    // generation the way an ordinary producer draw does, so the renderer-owned cube is eligible.
    auto seed_faces = [&](unsigned count, auto value) {
        BackendPersistentResourceGuard guard;
        invalidate_persistent_ds_guest_write(guest, 6 * stride);
        std::vector<GpuCaptureDsSeed> seeds;
        for (unsigned face = 0; face < count; ++face) {
            GpuCaptureDsSeed seed;
            seed.depth_read_base = seed.depth_write_base = guest;
            seed.width = W; seed.height = H; seed.slice = face;
            seed.format = GpuCaptureDsFormat::D32Float; seed.depth_valid = true;
            seed.depth.resize(W * H * sizeof(float));
            for (size_t i = 0; i < W * H; ++i) {
                const float v = value(face, i);
                std::memcpy(seed.depth.data() + i * sizeof(float), &v, sizeof(v));
            }
            seeds.push_back(std::move(seed));
        }
        std::string error;
        const bool restored = restore_gpu_replay_ds_seeds(seeds, error);
        if (!restored) std::fprintf(stderr, "DS restore: %s\n", error.c_str());
        check(restored, "real Vulkan depth faces restore");
        for (auto& [key, image] : persistent_ds_cache())
            if (key.dr == guest && key.slice < count && image.depth_valid)
                note_persistent_ds_depth_write(image, true, true);
    };

    const RenderVkCtx& ctx = render_vk_ctx();
    check(ctx.ok, "Vulkan renderer context is available");
    if (!ctx.ok) return 1;

    // ---- 1. EXACTNESS: every texel of a boundary-heavy cube, GPU gather vs CPU bridge. ----
    // In-range values only: a D32_SFLOAT attachment holds [0, 1] (Vulkan leaves copies of other
    // values undefined without depth_range_unrestricted), and rasterised depth is clamped there.
    std::vector<float> values;
    for (uint32_t k = 0; k < 255; ++k) {
        const float mid = (k + 0.5f) / 255.0f;
        values.push_back(mid);
        values.push_back(std::nextafter(mid, 0.0f));
        values.push_back(std::nextafter(mid, 1.0f));
    }
    for (float v : {0.0f, 1.0f, std::bit_cast<float>(0x3f010101u), 1e-40f,
                    std::numeric_limits<float>::denorm_min(), 0.49f / 255.0f, 0.5f / 255.0f})
        values.push_back(v);
    uint32_t state = 0x1234567u;
    while (values.size() < 6u * W * H) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        values.push_back(static_cast<float>(state >> 8) / static_cast<float>(1u << 24));
    }
    seed_faces(6, [&](unsigned face, size_t i) { return values[face * W * H + i]; });
    // CPU reference: the bridge's own readback of the stored floats, then its quantiser.
    std::array<std::vector<float>, 6> faces;
    uint32_t slices = 0, present = 0, known = 0;
    std::string error;
    const bool read_ok = read_persistent_ds_cube_depth(guest, W, H, faces, slices, error,
                                                       &present, &known);
    check(read_ok && present == 0x3fu, "CPU bridge reads all six retained faces");
    std::vector<uint8_t> expected(6u * W * H * 4u);
    for (uint32_t face = 0; face < 6u && read_ok; ++face)
        quantize_depth_cube_rgba8_scalar(expected.data() + face * W * H * 4u,
            reinterpret_cast<const uint8_t*>(faces[face].data()), W * H);
    if (!control) {
        BackendSubmissionBatch batch, other;
        std::shared_ptr<PersistentDsDepthCubeGpuImage> snapshot;
        const auto result = copy_persistent_ds_cube_depth_gpu(guest, W, H, batch, snapshot);
        check(result == DepthCubeGpuResult::Ready && snapshot && batch.pending(),
              "GPU gather is recorded into the caller's ordered batch, not submitted");
        check(snapshot && snapshot->servable_to(&batch) && !snapshot->servable_to(nullptr) &&
                  !snapshot->servable_to(&other),
              "GPU cube snapshot serves only consumers in its own batch");
        check(snapshot && persistent_ds_cube_identity_matches(guest, W, H, *snapshot),
              "snapshot records the exact retained identity it gathered");
        const auto submitted = batch.submit_and_wait(ctx.dev, ctx.queue, false);
        check(submitted.submit_result == VK_SUCCESS && submitted.wait_result == VK_SUCCESS &&
                  snapshot && snapshot->valid.load(),
              "owning batch completes and leaves the snapshot valid");
        HostBuffer readback;
        const bool made = make_host_buffer(ctx, expected.size(), readback);
        const bool copied = made && snapshot && submit_now(ctx, [&](VkCommandBuffer command) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = snapshot->image();
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &barrier);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {W, H * 6u, 1};
            vkCmdCopyImageToBuffer(command, snapshot->image(),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &copy);
        });
        check(copied, "GPU snapshot reads back for comparison");
        size_t mismatches = 0;
        if (copied)
            for (size_t i = 0; i < expected.size(); ++i)
                mismatches += static_cast<const uint8_t*>(readback.mapped)[i] != expected[i];
        if (mismatches) std::printf("  %zu of %zu bytes differ\n", mismatches, expected.size());
        check(copied && mismatches == 0,
              "every GPU texel equals the CPU bridge's quantised stack, face-major");
        batch.complete();
        // A later depth write to any one face changes the identity a memo must match.
        {
            BackendPersistentResourceGuard guard;
            for (auto& [key, image] : persistent_ds_cache())
                if (key.dr == guest && key.slice == 3 && image.depth_valid)
                    note_persistent_ds_depth_write(image, true, true);
        }
        check(snapshot && !persistent_ds_cube_identity_matches(guest, W, H, *snapshot),
              "a newer depth write to one face invalidates the snapshot's identity");

        // A batch failing after the gather was enqueued revokes the snapshot.
        BackendSubmissionBatch failed_batch;
        std::shared_ptr<PersistentDsDepthCubeGpuImage> failed;
        check(copy_persistent_ds_cube_depth_gpu(guest, W, H, failed_batch, failed) ==
                  DepthCubeGpuResult::Ready && failed && failed->servable_to(&failed_batch),
              "second snapshot is servable before its batch fails");
        failed_batch.discard();
        check(failed && !failed->valid.load() && !failed->servable_to(&failed_batch),
              "a batch failing after the gather was enqueued revokes the snapshot");
        failed_batch.complete();

        // ---- 2. DOMAIN: the conversion alone, on values no attachment can hold. ----
        const DepthCubeGpuPipeline* pipeline = depth_cube_gpu_pipeline(ctx);
        auto slot = pipeline ? depth_cube_gpu_slot(ctx, *pipeline, W, H) : nullptr;
        check(slot != nullptr, "conversion slot allocates");
        std::vector<float> wild;
        for (float v : {std::numeric_limits<float>::quiet_NaN(),
                        -std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(), -0.0f, -0.25f, -1e-40f, 1.25f,
                        1.0000001f, 3.0e38f, -3.0e38f, 1e-40f})
            wild.push_back(v);
        uint32_t bits_state = 0x7a92b16cu;
        while (wild.size() < 6u * W * H) {
            bits_state ^= bits_state << 13; bits_state ^= bits_state >> 17; bits_state ^= bits_state << 5;
            wild.push_back(std::bit_cast<float>(bits_state));
        }
        std::vector<uint8_t> wild_expected(wild.size() * 4u);
        quantize_depth_cube_rgba8_scalar(wild_expected.data(),
                                         reinterpret_cast<const uint8_t*>(wild.data()), wild.size());
        HostBuffer staging;
        const VkDeviceSize wild_bytes = wild.size() * sizeof(float);
        const bool staged = slot && make_host_buffer(ctx, wild_bytes, staging);
        if (staged) std::memcpy(staging.mapped, wild.data(), wild_bytes);
        const bool converted = staged && submit_now(ctx, [&](VkCommandBuffer command) {
            const VkBufferCopy region{0, 0, wild_bytes};
            vkCmdCopyBuffer(command, staging.buffer, slot->buffer, 1, &region);
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = slot->buffer;
            barrier.size = VK_WHOLE_SIZE;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier,
                                 0, nullptr);
            record_depth_cube_quantize(command, *pipeline, *slot,
                                       static_cast<uint32_t>(wild.size()));
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0,
                                 nullptr);
            vkCmdCopyBuffer(command, slot->buffer, staging.buffer, 1, &region);
        });
        size_t wild_mismatches = 0;
        if (converted)
            wild_mismatches = std::memcmp(staging.mapped, wild_expected.data(), wild_bytes) != 0;
        if (converted && wild_mismatches) {
            size_t n = 0;
            for (size_t i = 0; i < wild.size() && n < 8; ++i) {
                uint32_t got;
                std::memcpy(&got, static_cast<const uint8_t*>(staging.mapped) + i * 4, 4);
                uint32_t want;
                std::memcpy(&want, wild_expected.data() + i * 4, 4);
                if (got != want) {
                    std::printf("  value 0x%08x: gpu 0x%08x cpu 0x%08x\n",
                                std::bit_cast<uint32_t>(wild[i]), got, want);
                    ++n;
                }
            }
        }
        check(converted && wild_mismatches == 0,
              "conversion matches the CPU quantiser for NaN, infinities, negatives, > 1 and "
              "random bit patterns");
    }

    // ---- 3. ROUTING through the live renderer. ----
    auto face_value = [](unsigned face, size_t) { return (face + 1) / 8.0f; };
    const auto copies = [] { return depth_cube_gpu_copy_stats().copies; };
    seed_faces(6, face_value);
    select_face(5);
    reset_texture_decode_scope_stats();
    uint64_t before = copies();
    int pixel = center(render_submit_items({draw}, TW, TH));
    check(pixel == 191, "fully retained cube samples face 5 at 0.75 -> 191");
    check(control ? copies() == before && texture_decode_scope_stats().decodes == 1
                  : copies() == before + 1 && texture_decode_scope_stats().decodes == 0,
          control ? "control: the CPU bridge decodes the cube and records no GPU gather"
                  : "GPU route: one gather, no CPU decode");
    select_face(2);
    before = copies();
    reset_texture_decode_scope_stats();
    pixel = center(render_submit_items({draw, draw}, TW, TH));
    check(pixel == 96, "face 2 at 0.375 -> 96 through the same route");
    check(control || copies() == before + 1,
          "two draws in one submit share one gather (callback-local memo)");
    // A renderer rewrite of one face changes the identity: the next use must re-gather it.
    seed_faces(6, [](unsigned face, size_t) { return face == 2 ? 0.625f : 0.125f; });
    before = copies();
    pixel = center(render_submit_items({draw}, TW, TH));
    check(pixel == 159, "renderer rewrite is observed (0.625 -> 159), never a stale snapshot");
    check(control || copies() == before + 1, "rewritten faces are gathered again");
    // Forced fallback: the CPU bridge takes over with identical output.
    if (!control) {
        const uint64_t fallbacks = depth_cube_gpu_copy_stats().fallbacks;
        seed_faces(6, face_value);
        depth_cube_gpu_force_fallback_once() = true;
        before = copies();
        reset_texture_decode_scope_stats();
        pixel = center(render_submit_items({draw}, TW, TH));
        check(pixel == 96 && copies() == before &&
                  depth_cube_gpu_copy_stats().fallbacks == fallbacks + 1 &&
                  texture_decode_scope_stats().decodes == 1,
              "a GPU fallback hands the same cube to the CPU bridge with identical pixels");
    }
    // A mixed cube (five retained faces, one from guest bytes) stays on the CPU bridge.
    seed_faces(5, face_value);
    select_face(0);
    before = copies();
    pixel = center(render_submit_items({draw}, TW, TH));
    check(pixel == 32 && copies() == before,
          "mixed renderer/guest cube keeps the CPU bridge (face 0 at 0.125 -> 32)");

    unmap(guest, GuestBytes, 0, 0, 0, 0);
    std::printf("depth cube gpu%s: %d failures\n", control ? " (control)" : "", failures);
    return failures ? 1 : 0;
}
