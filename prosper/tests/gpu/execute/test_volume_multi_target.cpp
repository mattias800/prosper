// #4643: a layered colour pass with more than one volume target, through the production backend.
//
// Kena's translucency-lighting injection writes two 64^3 volumes from one layered pass (ambient in
// MRT0, directional in MRT1). The backend used to refuse any volume pass with more than one colour
// target, so every such draw was dropped (backend/volume-multi-target, ~7,700 per run). These arms
// pin the contract that replaced the refusal (fixtures/render_volume_slots.h):
//
//   * two volume slots in one pass: each slot's every slice holds what the layered draw wrote for
//     it, checked per slice and per channel, so a swapped, dropped or aliased slot cannot pass;
//   * per-slot layer routing: a slot's slice range is its own (different first slices per slot),
//     and slices outside a slot's range keep their earlier contents, for slot 1 and for slot 2
//     (slots 2..7 have their own creation and attachment path);
//   * the control: a fragment program that writes only MRT0 fails the slot-1 check, so the check
//     is able to see a slot-1 write that never happened (the mutation this test exists for);
//   * a 2D slot beside a volume slot: refused by name when the framebuffer is layered (Vulkan has
//     one layer count that every attachment must cover), admitted when it has one layer;
//   * the device-free admission rules and the split contract that keeps a volume MRT pass on the
//     GPU across a merged-NGG segment split (CLAUDE.md P1).
#include "fixtures/render_runner.h"
#include "volume_target_spirv.h"

#include "gpu/diagnostics/draw_disposition.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::test;
using prosper::gpu::DrawDrop;

namespace {

constexpr uint32_t kSize = 32, kDepth = 4;
constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr float kClear0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
constexpr float kClear1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

template <size_t N>
std::vector<uint32_t> words(const uint32_t (&array)[N]) {
    return {std::begin(array), std::end(array)};
}

bool device_ready() {
    const auto& ctx = render_vk_ctx();
    if (!ctx.ok || !ctx.mesh_shader_enabled || !ctx.image_view_2d_on_3d ||
        ctx.mesh_shader_properties.maxMeshOutputLayers < kDepth)
        return false;
    VkImageFormatProperties properties{};
    return vkGetPhysicalDeviceImageFormatProperties(
               ctx.phys, kFormat, VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
               VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT, &properties) == VK_SUCCESS &&
           properties.maxExtent.depth >= kDepth;
}

struct SlotView {
    uint64_t id = 0;
    uint32_t depth = 0, first = 0, count = 0;   // depth 0: a 2D slot
};

// One layered mesh draw of `layers` work groups (work group L writes local layer L) into a
// two-target pass. Returns whether the backend recorded a colour write.
bool render_two_targets(const SlotView& slot0, const SlotView& slot1,
                        const std::vector<uint32_t>& mesh, const std::vector<uint32_t>& fragment,
                        uint32_t layers) {
    BackendDraw draw;
    draw.mesh_draw = true;
    draw.mesh_groups = {layers, 1, 1};
    draw.vs = mesh;
    draw.fs = fragment;
    BackendColorTarget target;
    target.persistent_id = slot0.id;
    target.load_existing = false;
    target.readback = false;
    target.format = kFormat;
    target.volume_depth = slot0.depth;
    target.volume_first_slice = slot0.first;
    target.volume_slice_count = slot0.count;
    target.persistent_id1 = slot1.id;
    target.load_existing1 = false;
    target.readback1 = false;
    target.format1 = kFormat;
    target.volume_slots[1] = {slot1.depth, slot1.first, slot1.count, 0};
    BackendMrtOutputs mrt;
    mrt.color_count = 2;
    (void)render_draws_rgba({draw}, kSize, kSize, nullptr, kClear0, false, &target, nullptr,
                            kClear1, nullptr, nullptr, true, &mrt, true);   // as the live renderer
    return backend_color_target_stats().writes == 1;
}

std::array<uint8_t, 4> centre(const std::vector<uint8_t>& bytes, uint32_t slice) {
    const size_t at = ((static_cast<size_t>(slice) * kSize + kSize / 2) * kSize + kSize / 2) * 4;
    if (bytes.size() < at + 4) return {};
    return {bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]};
}

bool near(uint8_t actual, uint32_t expected) {
    return actual + 2u >= expected && actual <= expected + 2u;
}

// The value param_mesh puts in a pixel of local layer L: red L/7, alpha 1 (green and blue are
// barycentric, so they are checked only on slot 1, where the program replaces them).
uint32_t layer_value(uint32_t local_layer) {
    return (local_layer * 255u + 3u) / 7u;
}

::testing::AssertionResult readback(uint64_t id, uint32_t depth, std::vector<uint8_t>& bytes) {
    std::string error;
    if (!readback_persistent_color_target(id, kSize, kSize, kFormat, bytes, error, depth))
        return ::testing::AssertionFailure() << "volume 0x" << std::hex << id << std::dec
                                             << " has no complete retained image: " << error;
    return ::testing::AssertionSuccess();
}

// Slot 0 of param_mrt2_fragment: (L/7, bary, bary, 1).
::testing::AssertionResult slot0_slice(const std::vector<uint8_t>& bytes, uint32_t slice,
                                       uint32_t local_layer) {
    const auto p = centre(bytes, slice);
    if (!near(p[0], layer_value(local_layer)) || p[3] < 250u)
        return ::testing::AssertionFailure()
               << "slot 0 slice " << slice << " = (" << +p[0] << "," << +p[1] << "," << +p[2] << ","
               << +p[3] << "), expected red " << layer_value(local_layer);
    return ::testing::AssertionSuccess();
}

// Slot 1 of param_mrt2_fragment: (0, L/7, 1, 1).
::testing::AssertionResult slot1_slice(const std::vector<uint8_t>& bytes, uint32_t slice,
                                       uint32_t local_layer) {
    const auto p = centre(bytes, slice);
    if (p[0] > 2u || !near(p[1], layer_value(local_layer)) || p[2] < 250u || p[3] < 250u)
        return ::testing::AssertionFailure()
               << "slot 1 slice " << slice << " = (" << +p[0] << "," << +p[1] << "," << +p[2] << ","
               << +p[3] << "), expected (0," << layer_value(local_layer) << ",255,255)";
    return ::testing::AssertionSuccess();
}

bool is_rgba(const std::array<uint8_t, 4>& p, uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
    return near(p[0], r) && near(p[1], g) && near(p[2], b) && near(p[3], a);
}

}   // namespace

// ---- Device-free admission ----------------------------------------------------------------------

TEST(VolumeMultiTarget, ShapeAdmitsMatchingVolumeSlots) {
    BackendColorTarget target;
    target.persistent_id = 0x10;
    target.volume_depth = 64;
    target.volume_slice_count = 64;
    target.persistent_id1 = 0x20;
    target.volume_slots[1] = {64, 0, 64, 0};
    BackendVolumePass pass;
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, nullptr, pass), DrawDrop::Count);
    EXPECT_TRUE(pass.any);
    EXPECT_TRUE(pass.volume(0));
    EXPECT_TRUE(pass.volume(1));
    EXPECT_EQ(pass.layers, 64u);
    // Different bases are each slot's own routing; only the layer count must agree.
    target.volume_first_slice = 32;
    target.volume_slice_count = 32;
    target.volume_slots[1] = {64, 0, 32, 0};
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, nullptr, pass), DrawDrop::Count);
    EXPECT_EQ(pass.layers, 32u);
}

TEST(VolumeMultiTarget, ShapeRefusesWhatOneFramebufferCannotHold) {
    BackendColorTarget target;
    target.persistent_id = 0x10;
    target.volume_depth = 4;
    target.volume_slice_count = 4;
    target.persistent_id1 = 0x20;
    BackendVolumePass pass;
    // A one-layer 2D slot beside a four-layer framebuffer.
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, nullptr, pass),
              DrawDrop::VolumeMixedTarget);
    // ...but with one layer the 2D slot is expressible.
    target.volume_slice_count = 1;
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, nullptr, pass), DrawDrop::Count);
    EXPECT_EQ(pass.layers, 1u);
    EXPECT_FALSE(pass.volume(1));
    // Two volume slots that disagree on the layer count.
    target.volume_slice_count = 4;
    target.volume_slots[1] = {4, 0, 2, 0};
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, nullptr, pass),
              DrawDrop::VolumeMixedTarget);
    // One allocation bound to both attachments.
    target.volume_slots[1] = {4, 0, 4, 0};
    target.persistent_id1 = target.persistent_id;
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, nullptr, pass),
              DrawDrop::VolumeMixedTarget);
    // A volume slot with no identity has no transient fallback.
    target.persistent_id1 = 0;
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, nullptr, pass),
              DrawDrop::VolumeNotPersistent);
    // A CPU seed on any volume slot.
    target.persistent_id1 = 0x20;
    const uint8_t seed = 0;
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, &seed, pass), DrawDrop::VolumeSeeded);
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, &seed, nullptr, pass), DrawDrop::VolumeSeeded);
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, nullptr, pass), DrawDrop::Count);
    // Slots past the pass's colour count are not part of it.
    target.volume_slots[2] = {8, 0, 8, 0};
    EXPECT_EQ(backend_volume_pass_shape(&target, 2, nullptr, nullptr, pass), DrawDrop::Count);
}

TEST(VolumeMultiTarget, SplitCarriesVolumeSlotsWithoutReadback) {
    BackendColorTarget target;
    target.persistent_id = 0x10;
    target.volume_depth = 4;
    target.volume_slice_count = 4;
    target.persistent_id1 = 0x20;
    target.volume_slots[1] = {4, 0, 4, 0};
    target.readback = false;
    target.readback1 = false;
    EXPECT_TRUE(backend_split_carries_on_gpu(&target, 2));
    BackendMrtOutputs mrt;
    mrt.color_count = 2;
    const SplitSegmentContract first = split_segment_contract(&target, &mrt, true, false);
    EXPECT_FALSE(first.target.readback);
    EXPECT_FALSE(first.target.readback1) << "a volume slot is carried by LOAD, not a readback";
    const SplitSegmentContract later = split_segment_contract(&target, &mrt, false, true);
    EXPECT_TRUE(later.target.load_existing);
    EXPECT_TRUE(later.target.load_existing1);
    EXPECT_EQ(later.target.seed_rgba1_slot, nullptr);

    // A 2D slot 1 keeps the established carry: it may fall back to a transient image.
    BackendColorTarget flat = target;
    flat.volume_slots[1] = {};
    EXPECT_FALSE(backend_split_carries_on_gpu(&flat, 2));
    EXPECT_TRUE(split_segment_contract(&flat, &mrt, true, false).target.readback1);
    // One persistent attachment carries on the GPU, as before #4643.
    EXPECT_TRUE(backend_split_carries_on_gpu(&flat, 1));
    flat.persistent_id = 0;
    EXPECT_FALSE(backend_split_carries_on_gpu(&flat, 1));
    EXPECT_FALSE(backend_split_carries_on_gpu(nullptr, 1));
}

// ---- The production backend ---------------------------------------------------------------------

TEST(VolumeMultiTarget, TwoVolumeSlotsHoldEverySliceOfTheirOwnOutput) {
    if (!device_ready()) GTEST_SKIP() << "no mesh-shader device with 2D views of 3D images";
    constexpr uint64_t kAmbient = 0x764d525432000001ull, kDirectional = 0x764d525432000002ull;
    auto& census = prosper::gpu::draw_disposition_census();
    const uint64_t refused =
        census.dropped(DrawDrop::VolumeMixedTarget) + census.dropped(DrawDrop::VolumeMultiTarget);
    ASSERT_TRUE(render_two_targets({kAmbient, kDepth, 0, kDepth}, {kDirectional, kDepth, 0, kDepth},
                                   words(volume_fixture::param_mesh),
                                   words(volume_fixture::param_mrt2_fragment), kDepth));
    EXPECT_EQ(census.dropped(DrawDrop::VolumeMixedTarget) +
                  census.dropped(DrawDrop::VolumeMultiTarget),
              refused);
    std::vector<uint8_t> ambient, directional;
    ASSERT_TRUE(readback(kAmbient, kDepth, ambient));
    ASSERT_TRUE(readback(kDirectional, kDepth, directional));
    for (uint32_t z = 0; z < kDepth; ++z) {
        EXPECT_TRUE(slot0_slice(ambient, z, z));
        EXPECT_TRUE(slot1_slice(directional, z, z));
    }
}

TEST(VolumeMultiTarget, EachSlotRoutesLayersFromItsOwnFirstSlice) {
    if (!device_ready()) GTEST_SKIP() << "no mesh-shader device with 2D views of 3D images";
    constexpr uint64_t kSlot0 = 0x764d525432000011ull, kSlot1 = 0x764d525432000012ull;
    // Establish every slice of both volumes.
    ASSERT_TRUE(render_two_targets({kSlot0, kDepth, 0, kDepth}, {kSlot1, kDepth, 0, kDepth},
                                   words(volume_fixture::param_mesh),
                                   words(volume_fixture::param_mrt2_fragment), kDepth));
    // Two layers: slot 0 writes slices 2-3, slot 1 slices 0-1. The constant varying (magenta on
    // slot 0, full green on slot 1) differs from every value the first pass left.
    ASSERT_TRUE(render_two_targets({kSlot0, kDepth, 2, 2}, {kSlot1, kDepth, 0, 2},
                                   words(volume_fixture::wrong_param_mesh),
                                   words(volume_fixture::param_mrt2_fragment), 2));
    std::vector<uint8_t> slot0, slot1;
    ASSERT_TRUE(readback(kSlot0, kDepth, slot0));
    ASSERT_TRUE(readback(kSlot1, kDepth, slot1));
    EXPECT_TRUE(slot0_slice(slot0, 0, 0)) << "outside slot 0's range: the first pass survives";
    EXPECT_TRUE(slot0_slice(slot0, 1, 1)) << "outside slot 0's range: the first pass survives";
    for (uint32_t z = 2; z < 4; ++z)
        EXPECT_TRUE(is_rgba(centre(slot0, z), 255, 0, 255, 255)) << "slot 0 slice " << z;
    for (uint32_t z = 0; z < 2; ++z)
        EXPECT_TRUE(is_rgba(centre(slot1, z), 0, 255, 255, 255)) << "slot 1 slice " << z;
    EXPECT_TRUE(slot1_slice(slot1, 2, 2)) << "outside slot 1's range: the first pass survives";
    EXPECT_TRUE(slot1_slice(slot1, 3, 3)) << "outside slot 1's range: the first pass survives";
}

// Slots 2..7 take a separate creation and attachment path from slot 1. Three volume slots, each
// routing the same two layers from its own first slice.
TEST(VolumeMultiTarget, ThreeVolumeSlotsIncludingAnExtraSlot) {
    if (!device_ready()) GTEST_SKIP() << "no mesh-shader device with 2D views of 3D images";
    constexpr uint64_t kIds[3] = {0x764d525432000051ull, 0x764d525432000052ull,
                                  0x764d525432000053ull};
    const auto render = [&](uint32_t first0, uint32_t first1, uint32_t first2, uint32_t count,
                            const std::vector<uint32_t>& mesh) {
        BackendDraw draw;
        draw.mesh_draw = true;
        draw.mesh_groups = {count, 1, 1};
        draw.vs = mesh;
        draw.fs = words(volume_fixture::param_mrt3_fragment);
        BackendColorTarget target;
        target.persistent_id = kIds[0];
        target.load_existing = false;
        target.readback = false;
        target.format = kFormat;
        target.volume_depth = kDepth;
        target.volume_first_slice = first0;
        target.volume_slice_count = count;
        target.persistent_id1 = kIds[1];
        target.load_existing1 = false;
        target.readback1 = false;
        target.format1 = kFormat;
        target.volume_slots[1] = {kDepth, first1, count, 0};
        target.persistent_id_slots[2] = kIds[2];
        target.load_existing_slots[2] = false;
        target.readback_slots[2] = false;
        target.volume_slots[2] = {kDepth, first2, count, 0};
        BackendMrtOutputs mrt;
        mrt.color_count = 3;
        (void)render_draws_rgba({draw}, kSize, kSize, nullptr, kClear0, false, &target, nullptr,
                                kClear1, nullptr, nullptr, true, &mrt, true);
        return backend_color_target_stats().retained_slots == 0x7u;
    };
    ASSERT_TRUE(render(0, 0, 0, kDepth, words(volume_fixture::param_mesh)));
    ASSERT_TRUE(render(0, 1, 2, 2, words(volume_fixture::wrong_param_mesh)));
    std::vector<uint8_t> slot0, slot1, slot2;
    ASSERT_TRUE(readback(kIds[0], kDepth, slot0));
    ASSERT_TRUE(readback(kIds[1], kDepth, slot1));
    ASSERT_TRUE(readback(kIds[2], kDepth, slot2));
    for (uint32_t z = 0; z < kDepth; ++z) {
        const uint32_t v = layer_value(z);
        EXPECT_TRUE(z < 2 ? is_rgba(centre(slot0, z), 255, 0, 255, 255)
                          : bool(slot0_slice(slot0, z, z)))
            << "slot 0 slice " << z;
        EXPECT_TRUE(z >= 1 && z < 3 ? is_rgba(centre(slot1, z), 0, 255, 255, 255)
                                    : bool(slot1_slice(slot1, z, z)))
            << "slot 1 slice " << z;
        EXPECT_TRUE(z >= 2 ? is_rgba(centre(slot2, z), 255, 255, 0, 255)
                           : is_rgba(centre(slot2, z), v, 255, 0, 255))
            << "slot 2 slice " << z;
    }
}

// The control that proves the slot-1 check can fail: the same pass with a program that writes MRT0
// only leaves slot 1 at its clear, and the check must say so. Without this arm a backend that
// dropped slot 1 -- the defect #4643 was -- would be indistinguishable from one that drew it.
TEST(VolumeMultiTarget, ControlSlotOneCheckSeesAMissingWrite) {
    if (!device_ready()) GTEST_SKIP() << "no mesh-shader device with 2D views of 3D images";
    constexpr uint64_t kSlot0 = 0x764d525432000021ull, kSlot1 = 0x764d525432000022ull;
    ASSERT_TRUE(render_two_targets({kSlot0, kDepth, 0, kDepth}, {kSlot1, kDepth, 0, kDepth},
                                   words(volume_fixture::param_mesh),
                                   words(volume_fixture::param_fragment), kDepth));
    std::vector<uint8_t> slot0, slot1;
    ASSERT_TRUE(readback(kSlot0, kDepth, slot0));
    ASSERT_TRUE(readback(kSlot1, kDepth, slot1)) << "slot 1 was cleared, so it is complete";
    for (uint32_t z = 0; z < kDepth; ++z) {
        EXPECT_TRUE(slot0_slice(slot0, z, z));
        EXPECT_FALSE(slot1_slice(slot1, z, z)) << "the slot-1 check passed an unwritten slice";
        EXPECT_TRUE(is_rgba(centre(slot1, z), 0, 0, 0, 0)) << "slot 1 holds its clear";
    }
}

TEST(VolumeMultiTarget, LayeredVolumeBesideA2DSlotIsRefusedByName) {
    if (!device_ready()) GTEST_SKIP() << "no mesh-shader device with 2D views of 3D images";
    constexpr uint64_t kVolume = 0x764d525432000031ull, kFlat = 0x764d525432000032ull;
    auto& census = prosper::gpu::draw_disposition_census();
    const uint64_t before = census.dropped(DrawDrop::VolumeMixedTarget);
    EXPECT_FALSE(render_two_targets({kVolume, kDepth, 0, kDepth}, {kFlat, 0, 0, 0},
                                    words(volume_fixture::param_mesh),
                                    words(volume_fixture::param_mrt2_fragment), kDepth));
    EXPECT_EQ(census.dropped(DrawDrop::VolumeMixedTarget) - before, 1u)
        << "the one draw is dropped under volume-mixed-target";
    EXPECT_EQ(find_persistent_volume_target(kVolume, kSize, kSize, kDepth, kFormat), nullptr);
}

TEST(VolumeMultiTarget, OneLayerVolumeBesideA2DSlotRendersBoth) {
    if (!device_ready()) GTEST_SKIP() << "no mesh-shader device with 2D views of 3D images";
    constexpr uint64_t kVolume = 0x764d525432000041ull, kFlat = 0x764d525432000042ull;
    // Complete the volume first, through the single-target path.
    BackendDraw fill;
    fill.mesh_draw = true;
    fill.mesh_groups = {kDepth, 1, 1};
    fill.vs = words(volume_fixture::mesh);
    fill.fs = words(volume_fixture::red);
    BackendColorTarget single;
    single.persistent_id = kVolume;
    single.load_existing = false;
    single.readback = false;
    single.format = kFormat;
    single.volume_depth = kDepth;
    single.volume_slice_count = kDepth;
    (void)render_draws_rgba({fill}, kSize, kSize, nullptr, nullptr, false, &single);
    ASSERT_NE(find_persistent_volume_target(kVolume, kSize, kSize, kDepth, kFormat), nullptr);
    // One slice of the volume (slice 1) and a 2D target in one single-layer pass.
    ASSERT_TRUE(render_two_targets({kVolume, kDepth, 1, 1}, {kFlat, 0, 0, 0},
                                   words(volume_fixture::wrong_param_mesh),
                                   words(volume_fixture::param_mrt2_fragment), 1));
    std::vector<uint8_t> volume, flat;
    ASSERT_TRUE(readback(kVolume, kDepth, volume));
    for (uint32_t z = 0; z < kDepth; ++z)
        EXPECT_TRUE(z == 1 ? is_rgba(centre(volume, z), 255, 0, 255, 255)
                           : is_rgba(centre(volume, z), 255, 0, 0, 255))
            << "volume slice " << z;
    ASSERT_TRUE(readback(kFlat, 0, flat));
    EXPECT_TRUE(is_rgba(centre(flat, 0), 0, 255, 255, 255)) << "the 2D slot";
}
