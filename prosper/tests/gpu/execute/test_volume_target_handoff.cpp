// A production-backend 3D color target must retain one allocation while bounded layered views
// write separate slices. The consumer samples that exact image as a 3D texture, without a guest
// readback/upload. The shader colors and pixel expectations are independent of cache metadata.
#include "fixtures/render_runner.h"
#include "volume_target_spirv.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/execute/layered_volume_target.hpp"
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
        ctx.mesh_shader_properties.maxMeshOutputLayers < 8 ||
        format_properties.maxExtent.depth < 32) return 77;

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
    const std::vector<uint32_t> param_mesh(
        std::begin(prosper::test::volume_fixture::param_mesh),
        std::end(prosper::test::volume_fixture::param_mesh));
    const std::vector<uint32_t> wrong_param_mesh(
        std::begin(prosper::test::volume_fixture::wrong_param_mesh),
        std::end(prosper::test::volume_fixture::wrong_param_mesh));
    const std::vector<uint32_t> param_fragment(
        std::begin(prosper::test::volume_fixture::param_fragment),
        std::end(prosper::test::volume_fixture::param_fragment));
    #include "../tools/boot_trace/refvs.inc"
    const std::vector<uint32_t> vertex(kRefVs, kRefVs + sizeof(kRefVs) / 4);

    auto produce = [&](uint64_t id, uint32_t depth, uint32_t first, uint32_t count,
                       const std::vector<uint32_t>& fragment,
                       uint32_t width = kWidth, uint32_t height = kHeight,
                       const std::vector<uint32_t>* mesh_override = nullptr) {
        BackendDraw draw;
        draw.mesh_draw = true;
        draw.mesh_groups = {count, 1, 1};
        draw.vs = mesh_override ? *mesh_override : mesh_shader;
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

    // The first instruction is byte-identical to Kena's LUT pixel shader:
    // v_bfe_u32 v3, v2, 16, 11. Field 5 consumes v0-v1; field 13 (ANCILLARY)
    // occupies v2. Export its extracted layer as red/8, so four layer outputs
    // stay distinguishable in RGBA8. The disabled control reserves v2 but must
    // retain its old zero value; an unconditional gl_Layer read would fail it.
    const uint32_t ancillary_ps[] = {
        0xd5480003u, 0x022d2102u, // v_bfe_u32 v3, v2, 16, 11
        0x7e000d03u,              // v_cvt_f32_u32 v0, v3
        0x100000ffu, 0x3e000000u, // v_mul_f32 v0, 0.125, v0
        0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800080fu, 0x03020100u, 0xbf810000u,
    };
    prosper::gpu::PixelSystemInputMapping ancillary_inputs{};
    ancillary_inputs.addr = (1u << 5) | (1u << 13);
    ancillary_inputs.ena = ancillary_inputs.addr;
    const auto ancillary_frag = prosper::gpu::recompile_fragment(
        ancillary_ps, std::size(ancillary_ps), nullptr, &ancillary_inputs);
    ancillary_inputs.ena = 1u << 5;
    const auto disabled_frag = prosper::gpu::recompile_fragment(
        ancillary_ps, std::size(ancillary_ps), nullptr, &ancillary_inputs);
    if (ancillary_frag.empty() || disabled_frag.empty()) {
        std::fprintf(stderr, "ancillary pixel-system fixture did not recompile\n");
        return 1;
    }
    VkPhysicalDeviceFeatures supported_features{};
    vkGetPhysicalDeviceFeatures(ctx.phys, &supported_features);
    if (supported_features.geometryShader) {
        auto check_layers = [&](uint64_t id, const std::vector<uint32_t>& fragment,
                                bool enabled) {
            if (!produce(id, kDepth, 0, kDepth, fragment)) return false;
            std::vector<uint8_t> bytes;
            std::string error;
            if (!readback_persistent_color_target(id, kWidth, kHeight,
                    VK_FORMAT_R8G8B8A8_UNORM, bytes, error, kDepth)) return false;
            for (uint32_t layer = 0; layer < kDepth; ++layer) {
                const size_t at = (static_cast<size_t>(layer) * kWidth * kHeight +
                                   static_cast<size_t>(kHeight / 2) * kWidth + kWidth / 2) * 4;
                if (bytes.size() < at + 4) return false;
                const uint32_t expected = enabled ? layer * 32u : 0u;
                const uint32_t actual = bytes[at];
                if (actual + 2u < expected || actual > expected + 2u ||
                    bytes[at + 3] < 250u) return false;
            }
            return true;
        };
        if (!check_layers(kTarget + 0x500000u, ancillary_frag, true) ||
            !check_layers(kTarget + 0x600000u, disabled_frag, false)) {
            std::fprintf(stderr, "ancillary layer input or disabled control rendered incorrectly\n");
            return 1;
        }
        BackendDraw flat_ancillary;
        flat_ancillary.vs = vertex;
        flat_ancillary.fs = ancillary_frag;
        const auto flat_pixels = render_draws_rgba({flat_ancillary}, kWidth, kHeight);
        const size_t flat_at =
            (static_cast<size_t>(kHeight / 2) * kWidth + kWidth / 2) * 4;
        if (flat_pixels.size() < flat_at + 4 || flat_pixels[flat_at] != 0u ||
            flat_pixels[flat_at + 2] != 0u || flat_pixels[flat_at + 3] < 250u) {
            std::fprintf(stderr, "single-layer ancillary draw did not use layer zero\n");
            return 1;
        }
        std::fprintf(stderr, "ancillary layer input, disabled slot, and 2D default passed\n");
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
    {
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
    // Synthesized layered volume pass: a 32-slice 3D volume target produced in a SINGLE pass
    // using kLayeredVolumeVs and kLayeredVolumeGs with 32 instances, bypassing the 8-layer mesh shader limit.
    if (supported_features.geometryShader) {
        constexpr uint64_t layered_id = kTarget + 0xb00000u;
        constexpr uint32_t depth32 = 32;
        BackendDraw draw;
        draw.mesh_draw = false;
        draw.vcount = 3;
        draw.instance_count = depth32;
        draw.vs = std::vector<uint32_t>(std::begin(prosper::gpu::kLayeredVolumeVs),
                                        std::end(prosper::gpu::kLayeredVolumeVs));
        draw.gs = std::vector<uint32_t>(std::begin(prosper::gpu::kLayeredVolumeGs),
                                        std::end(prosper::gpu::kLayeredVolumeGs));
        draw.fs = red_shader;
        BackendColorTarget target;
        target.persistent_id = layered_id;
        target.load_existing = false;
        target.readback = false;
        target.format = VK_FORMAT_R8G8B8A8_UNORM;
        target.volume_depth = depth32;
        target.volume_first_slice = 0;
        target.volume_slice_count = depth32;
        if (const auto cpu = render_draws_rgba({draw}, kWidth, kHeight, nullptr, nullptr,
                                           false, &target);
            !cpu.empty() || backend_color_target_stats().writes != 1) {
            std::fprintf(stderr, "single-pass 32-slice layered volume producer failed to reach retained path\n");
            return 1;
        }
        std::vector<uint8_t> layered_bytes;
        if (std::string read_err;
            !readback_persistent_color_target(layered_id, kWidth, kHeight,
                                              VK_FORMAT_R8G8B8A8_UNORM, layered_bytes,
                                              read_err, depth32) ||
            layered_bytes.size() != static_cast<size_t>(kWidth) * kHeight * depth32 * 4) {
            std::fprintf(stderr, "32-slice layered volume readback failed: %s\n", read_err.c_str());
            return 1;
        }
        for (uint32_t z = 0; z < depth32; ++z) {
            const size_t at = (static_cast<size_t>(z) * kWidth * kHeight +
                               static_cast<size_t>(kHeight / 2) * kWidth + kWidth / 2) * 4;
            const std::array<uint8_t, 4> texel{
                layered_bytes[at], layered_bytes[at + 1],
                layered_bytes[at + 2], layered_bytes[at + 3]};
            if (!red(texel)) {
                std::fprintf(stderr, "32-slice layered volume lost slice %u\n", z);
                return 1;
            }
        }
        if (const auto sampled = consume(layered_id, depth32);
            backend_color_target_stats().sampled_hits != 1 ||
            !red(pixel(sampled, 8)) || !red(pixel(sampled, 24)) ||
            !red(pixel(sampled, 40)) || !red(pixel(sampled, 56))) {
            std::fprintf(stderr, "32-slice layered volume was not sampled directly\n");
            return 1;
        }
        std::puts("single-pass 32-slice layered geometry volume producer passed");
    }
    // The guest producer exports an interpolated PARAM as well as a primitive layer. Constant
    // fragment colors above cannot detect a lost or mislinked PARAM on the MeshEXT handoff.
    {
        constexpr uint64_t param_id = kTarget + 0xa00000u;
        for (uint32_t group = 0; group < large_depth / partition; ++group) {
            if (!produce(param_id, large_depth, group * partition, partition,
                         param_fragment, kWidth, kHeight, &param_mesh)) {
                std::fprintf(stderr, "layered PARAM producer group %u failed\n", group);
                return 1;
            }
        }
        const auto delivered = consume(param_id, large_depth);
        if (backend_color_target_stats().sampled_hits != 1 ||
            pixel(delivered, 24)[0] < 33 || pixel(delivered, 24)[0] > 40 ||
            pixel(delivered, 56)[0] < 106 || pixel(delivered, 56)[0] > 113) {
            std::fprintf(stderr, "layered PARAM image was not sampled directly\n");
            return 1;
        }
        volume_bytes.clear();
        if (!readback_persistent_color_target(param_id, kWidth, kHeight,
                                              VK_FORMAT_R8G8B8A8_UNORM, volume_bytes,
                                              readback_error, large_depth)) {
            std::fprintf(stderr, "layered PARAM readback failed: %s\n",
                         readback_error.c_str());
            return 1;
        }
        auto texel = [&](const std::vector<uint8_t>& bytes, uint32_t z, uint32_t x) {
            const size_t at = (static_cast<size_t>(z) * kWidth * kHeight +
                               static_cast<size_t>(kHeight / 2) * kWidth + x) * 4;
            if (bytes.size() < at + 4) return std::array<uint8_t, 4>{};
            return std::array<uint8_t, 4>{bytes[at], bytes[at + 1],
                                          bytes[at + 2], bytes[at + 3]};
        };
        auto matches = [&](const std::vector<uint8_t>& bytes, uint32_t z) {
            const auto left = texel(bytes, z, 8);
            const auto center = texel(bytes, z, kWidth / 2);
            const auto right = texel(bytes, z, 56);
            const uint32_t expected_red = ((z % partition) * 255u + 3u) / 7u;
            const auto red_near = [&](uint8_t actual) {
                return actual + 3u >= expected_red && actual <= expected_red + 3u;
            };
            return red_near(left[0]) && red_near(center[0]) && red_near(right[0]) &&
                   left[1] < 32 && center[1] > 55 && center[1] < 75 &&
                   right[1] > 96 && center[2] > 55 && center[2] < 75 &&
                   center[3] > 250;
        };
        if (volume_bytes.size() != static_cast<size_t>(kWidth) * kHeight * large_depth * 4)
            return 1;
        for (uint32_t z = 0; z < large_depth; ++z) {
            if (!matches(volume_bytes, z)) {
                const auto center = texel(volume_bytes, z, kWidth / 2);
                std::fprintf(stderr, "layered PARAM slice %u center=%u,%u,%u,%u\n",
                             z, center[0], center[1], center[2], center[3]);
                return 1;
            }
        }
        // Same geometry/layer output, deliberately wrong PARAM. If the positive assertion were
        // accidentally tied only to the render target rather than the fragment input, it would
        // accept this independently rendered image too.
        constexpr uint64_t wrong_id = param_id + 0x100000u;
        if (!produce(wrong_id, partition, 0, partition, param_fragment,
                     kWidth, kHeight, &wrong_param_mesh)) return 1;
        std::vector<uint8_t> wrong_bytes;
        if (!readback_persistent_color_target(wrong_id, kWidth, kHeight,
                                              VK_FORMAT_R8G8B8A8_UNORM, wrong_bytes,
                                              readback_error, partition) ||
            wrong_bytes.size() != static_cast<size_t>(kWidth) * kHeight * partition * 4) {
            std::fprintf(stderr, "wrong PARAM control did not render\n");
            return 1;
        }
        const auto wrong_center = texel(wrong_bytes, partition - 1, kWidth / 2);
        if (wrong_center[0] < 200 || wrong_center[1] > 50 ||
            wrong_center[2] < 200 || wrong_center[3] < 200 ||
            matches(wrong_bytes, partition - 1)) {
            std::fprintf(stderr, "wrong PARAM control passed the layer guard\n");
            return 1;
        }
        std::puts("32-slice mesh PARAM interpolation and wrong-PARAM control passed");
    }
    std::puts("retained volume writes, partial update, direct 3D sample and invalidation passed");
    return 0;
}
