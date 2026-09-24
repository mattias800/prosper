// A production-backend 3D color target must retain one allocation while bounded layered views
// write separate slices. The consumer samples that exact image as a 3D texture, without a guest
// readback/upload. The shader colors and pixel expectations are independent of cache metadata.
#include "fixtures/render_runner.h"
#include "volume_target_spirv.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <vector>

namespace {

constexpr uint32_t kWidth = 64, kHeight = 64, kDepth = 4;
constexpr uint64_t kTarget = 0x764f6c756d650001ull;

std::array<uint8_t, 4> pixel(const std::vector<uint8_t>& frame, uint32_t x) {
    const size_t at = (static_cast<size_t>(kHeight / 2) * kWidth + x) * 4;
    if (frame.size() < at + 4) return {};
    return {frame[at], frame[at + 1], frame[at + 2], frame[at + 3]};
}

bool red(const std::array<uint8_t, 4>& p) {
    return p[0] > 200 && p[1] < 50 && p[2] < 50 && p[3] > 200;
}
bool green(const std::array<uint8_t, 4>& p) {
    return p[0] < 50 && p[1] > 200 && p[2] < 50 && p[3] > 200;
}

} // namespace

int main() {
    using namespace prosper::test;
    const auto& ctx = render_vk_ctx();
    if (!ctx.ok || !ctx.mesh_shader_enabled || !ctx.image_view_2d_on_3d) return 77;
    VkImageFormatProperties format_properties{};
    constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (vkGetPhysicalDeviceImageFormatProperties(
            ctx.phys, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_3D,
            VK_IMAGE_TILING_OPTIMAL, usage, VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT,
            &format_properties) != VK_SUCCESS ||
        ctx.mesh_shader_properties.maxMeshOutputLayers < kDepth) return 77;

    const std::vector<uint32_t> mesh_shader(
        std::begin(prosper::test::volume_fixture::mesh),
        std::end(prosper::test::volume_fixture::mesh));
    const std::vector<uint32_t> red_shader(
        std::begin(prosper::test::volume_fixture::red),
        std::end(prosper::test::volume_fixture::red));
    const std::vector<uint32_t> green_shader(
        std::begin(prosper::test::volume_fixture::green),
        std::end(prosper::test::volume_fixture::green));
    const std::vector<uint32_t> sample_shader(
        std::begin(prosper::test::volume_fixture::sample),
        std::end(prosper::test::volume_fixture::sample));
    #include "../tools/boot_trace/refvs.inc"
    const std::vector<uint32_t> vertex(kRefVs, kRefVs + sizeof(kRefVs) / 4);

    auto produce = [&](uint64_t id, uint32_t depth, uint32_t first, uint32_t count,
                       const std::vector<uint32_t>& fragment,
                       uint32_t width = kWidth, uint32_t height = kHeight) {
        BackendDraw draw;
        draw.mesh_draw = true;
        draw.mesh_groups = {count, 1, 1};
        draw.vs = mesh_shader;
        draw.fs = fragment;
        BackendColorTarget target;
        target.persistent_id = id;
        target.load_existing = false;
        target.readback = false;
        target.format = VK_FORMAT_R8G8B8A8_UNORM;
        target.volume_depth = depth;
        target.volume_first_slice = first;
        target.volume_slice_count = count;
        const auto cpu = render_draws_rgba({draw}, width, height, nullptr, nullptr,
                                           false, &target);
        return cpu.empty() && backend_color_target_stats().writes == 1;
    };
    auto consume = [&](uint64_t id, uint32_t depth) {
        FrameResource resource;
        resource.set = 1;
        resource.binding = 4;
        resource.persistent_render_target_id = id;
        resource.tw = kWidth;
        resource.th = kHeight;
        resource.td = depth;
        resource.img_dim = 2;
        BackendDraw draw;
        draw.vs = vertex;
        draw.fs = sample_shader;
        draw.R = {resource};
        return render_draws_rgba({draw}, kWidth, kHeight);
    };

    if (!produce(kTarget, kDepth, 0, kDepth, red_shader)) {
        std::fprintf(stderr, "complete volume producer did not reach retained path\n");
        return 1;
    }
    auto* retained = find_persistent_volume_target(
        kTarget, kWidth, kHeight, kDepth, VK_FORMAT_R8G8B8A8_UNORM);
    if (!retained || !retained->image || !retained->valid) {
        std::fprintf(stderr, "complete volume was not retained\n");
        return 1;
    }
    const VkImage original_image = retained->image;
    const auto initial = consume(kTarget, kDepth);
    if (backend_color_target_stats().sampled_hits != 1 ||
        !red(pixel(initial, 8)) || !red(pixel(initial, 24)) ||
        !red(pixel(initial, 40)) || !red(pixel(initial, 56))) {
        std::fprintf(stderr, "3D consumer did not sample every red producer slice\n");
        return 1;
    }

    if (!produce(kTarget, kDepth, 2, 2, green_shader)) {
        std::fprintf(stderr, "partial volume producer did not reach retained path\n");
        return 1;
    }
    retained = find_persistent_volume_target(
        kTarget, kWidth, kHeight, kDepth, VK_FORMAT_R8G8B8A8_UNORM);
    const auto partial = consume(kTarget, kDepth);
    if (!retained || retained->image != original_image ||
        backend_color_target_stats().sampled_hits != 1 ||
        !red(pixel(partial, 8)) || !red(pixel(partial, 24)) ||
        !green(pixel(partial, 40)) || !green(pixel(partial, 56))) {
        std::fprintf(stderr, "partial view replaced the volume or changed untouched slices\n");
        return 1;
    }
    inject_render_vk_object_create_failure_once(
        RenderVkObjectCreateSite::VolumeAttachmentView);
    (void)produce(kTarget, kDepth, 0, kDepth, red_shader);
    if (render_vk_object_create_failure_storage() != RenderVkObjectCreateSite::None ||
        find_persistent_volume_target(kTarget, kWidth, kHeight, kDepth,
                                      VK_FORMAT_R8G8B8A8_UNORM)) {
        std::fprintf(stderr, "failed attachment view left prior volume authoritative\n");
        return 1;
    }
    if (!produce(kTarget, kDepth, 0, kDepth, red_shader) ||
        !red(pixel(consume(kTarget, kDepth), 56))) {
        std::fprintf(stderr, "volume did not recover after attachment view failure\n");
        return 1;
    }

    constexpr uint64_t incomplete_id = kTarget + 0x100000u;
    if (!produce(incomplete_id, kDepth, 0, 2, red_shader) ||
        find_persistent_volume_target(incomplete_id, kWidth, kHeight, kDepth,
                                      VK_FORMAT_R8G8B8A8_UNORM)) {
        std::fprintf(stderr, "incomplete volume became sampleable\n");
        return 1;
    }
    if (!produce(incomplete_id, kDepth, 2, 2, green_shader)) return 1;
    const auto completed_ranges = consume(incomplete_id, kDepth);
    if (backend_color_target_stats().sampled_hits != 1 ||
        !red(pixel(completed_ranges, 8)) || !red(pixel(completed_ranges, 24)) ||
        !green(pixel(completed_ranges, 40)) || !green(pixel(completed_ranges, 56))) {
        std::fprintf(stderr, "separately cleared ranges did not preserve the first half\n");
        return 1;
    }
    std::vector<uint8_t> volume_bytes;
    std::string readback_error;
    if (!readback_persistent_color_target(incomplete_id, kWidth, kHeight,
                                          VK_FORMAT_R8G8B8A8_UNORM, volume_bytes,
                                          readback_error, kDepth) ||
        volume_bytes.size() != static_cast<size_t>(kWidth) * kHeight * kDepth * 4) {
        std::fprintf(stderr, "full-volume readback failed: %s\n", readback_error.c_str());
        return 1;
    }
    for (uint32_t z = 0; z < kDepth; ++z) {
        const size_t at = (static_cast<size_t>(z) * kWidth * kHeight +
                           static_cast<size_t>(kHeight / 2) * kWidth + kWidth / 2) * 4;
        const std::array<uint8_t, 4> texel{
            volume_bytes[at], volume_bytes[at + 1], volume_bytes[at + 2], volume_bytes[at + 3]};
        if (z < 2 ? !red(texel) : !green(texel)) {
            std::fprintf(stderr, "full-volume readback lost slice %u\n", z);
            return 1;
        }
    }
    const uint64_t last_slice = kTarget +
        static_cast<uint64_t>(kWidth) * kHeight * 4 * (kDepth - 1);
    if (!invalidate_persistent_color_target_guest_write(last_slice, 4) ||
        find_persistent_volume_target(kTarget, kWidth, kHeight, kDepth,
                                      VK_FORMAT_R8G8B8A8_UNORM)) {
        std::fprintf(stderr, "last-slice guest write did not invalidate entire volume\n");
        return 1;
    }
    if (!produce(kTarget, kDepth, 0, kDepth, red_shader)) return 1;

    // An identical guest address with a 2D registration is a distinct representation. Writing it
    // must revoke the volume rather than let a later 3D binding sample an obsolete allocation.
    BackendDraw ordinary;
    ordinary.vs = vertex;
    ordinary.fs = red_shader;
    BackendColorTarget flat;
    flat.persistent_id = kTarget;
    flat.load_existing = false;
    flat.readback = false;
    flat.format = VK_FORMAT_R8G8B8A8_UNORM;
    (void)render_draws_rgba({ordinary}, kWidth, kHeight, nullptr, nullptr, false, &flat);
    if (backend_color_target_stats().writes != 1 ||
        find_persistent_volume_target(kTarget, kWidth, kHeight, kDepth,
                                      VK_FORMAT_R8G8B8A8_UNORM)) {
        std::fprintf(stderr, "2D alias write left stale volume authority\n");
        return 1;
    }
    // A 2D alias in a nonzero MRT slot must revoke the same whole-volume slice proof.
    // Otherwise a later partial volume write could LOAD slices from before this 2D producer.
    constexpr uint64_t mrt_alias_id = kTarget + 0x300000u;
    if (!produce(mrt_alias_id, kDepth, 0, kDepth, red_shader)) return 1;
    BackendColorTarget mrt_target;
    mrt_target.persistent_id = mrt_alias_id + 0x100000u;
    mrt_target.persistent_id1 = mrt_alias_id;
    mrt_target.format = VK_FORMAT_R8G8B8A8_UNORM;
    mrt_target.format1 = VK_FORMAT_R8G8B8A8_UNORM;
    mrt_target.readback = false;
    mrt_target.readback1 = false;
    BackendMrtOutputs mrt_outputs;
    mrt_outputs.color_count = 2;
    (void)render_draws_rgba({ordinary}, kWidth, kHeight, nullptr, nullptr, false,
                            &mrt_target, nullptr, nullptr, nullptr, nullptr, true,
                            &mrt_outputs, false);
    if (backend_color_target_stats().writes == 0 ||
        find_persistent_volume_target(mrt_alias_id, kWidth, kHeight, kDepth,
                                      VK_FORMAT_R8G8B8A8_UNORM)) {
        std::fprintf(stderr, "MRT1 2D alias left stale volume slices\n");
        return 1;
    }
    // A recorded pass is not a completed producer. Discarding its batch must revoke even a
    // speculative all-slices-valid volume before another draw can bind it.
    constexpr uint64_t discarded_id = incomplete_id + 0x100000u;
    BackendDraw pending_draw;
    pending_draw.mesh_draw = true;
    pending_draw.mesh_groups = {kDepth, 1, 1};
    pending_draw.vs = mesh_shader;
    pending_draw.fs = red_shader;
    BackendColorTarget pending_target;
    pending_target.persistent_id = discarded_id;
    pending_target.format = VK_FORMAT_R8G8B8A8_UNORM;
    pending_target.readback = false;
    pending_target.volume_depth = kDepth;
    pending_target.volume_slice_count = kDepth;
    BackendSubmissionBatch pending;
    (void)render_draws_rgba({pending_draw}, kWidth, kHeight, nullptr, nullptr, false,
                            &pending_target, nullptr, nullptr, nullptr, &pending, false);
    if (!pending.pending() ||
        !find_persistent_volume_target(discarded_id, kWidth, kHeight, kDepth,
                                       VK_FORMAT_R8G8B8A8_UNORM)) {
        std::fprintf(stderr, "pending volume producer never reached speculative state\n");
        return 1;
    }
    pending.discard();
    if (find_persistent_volume_target(discarded_id, kWidth, kHeight, kDepth,
                                      VK_FORMAT_R8G8B8A8_UNORM)) {
        std::fprintf(stderr, "discarded volume remained sampleable\n");
        return 1;
    }
    constexpr uint64_t resized_id = discarded_id + 0x100000u;
    if (!produce(resized_id, kDepth, 0, kDepth, red_shader) ||
        !produce(resized_id, kDepth, 0, kDepth, green_shader, kWidth / 2, kHeight) ||
        find_persistent_volume_target(resized_id, kWidth, kHeight, kDepth,
                                      VK_FORMAT_R8G8B8A8_UNORM) ||
        !find_persistent_volume_target(resized_id, kWidth / 2, kHeight, kDepth,
                                       VK_FORMAT_R8G8B8A8_UNORM)) {
        std::fprintf(stderr, "same-address volume resize left stale shape authority\n");
        return 1;
    }
    // A device with only eight mesh output layers can still retain a 32-slice 3D image through
    // four bounded attachment views. This verifies the backend's range handoff; it does not
    // establish a guest merged-NGG primitive/layer partition.
    constexpr uint32_t large_depth = 32, partition = 8;
    if (ctx.mesh_shader_properties.maxMeshOutputLayers >= partition &&
        format_properties.maxExtent.depth >= large_depth) {
        constexpr uint64_t large_id = kTarget + 0x900000u;
        VkImage large_image = VK_NULL_HANDLE;
        for (uint32_t group = 0; group < large_depth / partition; ++group) {
            if (!produce(large_id, large_depth, group * partition, partition,
                         group & 1u ? green_shader : red_shader)) {
                std::fprintf(stderr, "bounded volume partition %u failed\n", group);
                return 1;
            }
            auto* retained_slice = find_persistent_volume_target(
                large_id, kWidth, kHeight, large_depth, VK_FORMAT_R8G8B8A8_UNORM,
                false);
            if (!retained_slice || !retained_slice->image ||
                (large_image && retained_slice->image != large_image)) {
                std::fprintf(stderr, "partition %u replaced retained image\n", group);
                return 1;
            }
            large_image = retained_slice->image;
            auto* image = find_persistent_volume_target(
                large_id, kWidth, kHeight, large_depth, VK_FORMAT_R8G8B8A8_UNORM);
            if (group != large_depth / partition - 1) {
                if (image) {
                    std::fprintf(stderr, "incomplete 32-slice volume became sampleable\n");
                    return 1;
                }
            } else if (!image || !image->valid || image->image != large_image) {
                std::fprintf(stderr, "completed partitions lost retained image identity\n");
                return 1;
            }
        }
        const auto sampled = consume(large_id, large_depth);
        if (backend_color_target_stats().sampled_hits != 1 ||
            !red(pixel(sampled, 8))) {
            std::fprintf(stderr, "completed 32-slice image was not sampled directly\n");
            return 1;
        }
        volume_bytes.clear();
        if (!readback_persistent_color_target(large_id, kWidth, kHeight,
                                              VK_FORMAT_R8G8B8A8_UNORM, volume_bytes,
                                              readback_error, large_depth) ||
            volume_bytes.size() != static_cast<size_t>(kWidth) * kHeight * large_depth * 4) {
            std::fprintf(stderr, "32-slice volume readback failed: %s\n",
                         readback_error.c_str());
            return 1;
        }
        for (uint32_t z = 0; z < large_depth; ++z) {
            const size_t at = (static_cast<size_t>(z) * kWidth * kHeight +
                               static_cast<size_t>(kHeight / 2) * kWidth + kWidth / 2) * 4;
            const std::array<uint8_t, 4> texel{
                volume_bytes[at], volume_bytes[at + 1],
                volume_bytes[at + 2], volume_bytes[at + 3]};
            if ((z / partition) & 1u ? !green(texel) : !red(texel)) {
                std::fprintf(stderr, "32-slice partition lost slice %u\n", z);
                return 1;
            }
        }
        std::puts("32-slice retained volume passed four eight-layer views");
    }
    std::puts("retained volume writes, partial update, direct 3D sample and invalidation passed");
    return 0;
}
