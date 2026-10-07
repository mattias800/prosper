// Same-TU companion of render_runner.h, inside prosper::test, included before the pass renderer:
// the volume contract of a layered colour pass with any number of colour targets (#4643).
//
// A pass is a VOLUME pass when any active colour slot names a view of a 3D allocation
// (BackendColorTarget's volume_* fields for slot 0, volume_slots[] for the rest). Each such slot
// binds its own 3D image through a 2D-array view of its slice range, and Vulkan routes gl_Layer to
// every attachment relative to that attachment's own base layer -- the routing the guest gets from
// CB_COLORn_VIEW.SLICE_START. A framebuffer has ONE layer count, which every attachment view must
// cover (VUID-VkFramebufferCreateInfo-flags-04535). So:
//
//   - every volume slot of a pass has the same slice count, which is the framebuffer's;
//   - a 2D slot can sit beside volume slots only when that count is 1;
//   - a volume slot is never CPU-seeded and always has its own persistent identity, because there
//     is no transient 3D fallback: it is retained, or the pass is refused.
//
// Anything else is refused by name (volume-mixed-target, volume-seeded, volume-not-persistent)
// rather than drawn into the wrong layers. Before #4643 a volume pass with more than one colour
// target was refused whole (volume-multi-target), which dropped Kena's translucency-lighting
// volume injection: two 64^3 volumes per pass, ambient in slot 0 and directional in slot 1.
#pragma once

using BackendVolumeSlot = BackendColorTarget::VolumeSlot;

// Slot `slot`'s volume view. Slot 0 keeps its historical fields; zero depth is a 2D slot.
inline BackendVolumeSlot backend_color_volume_slot(const BackendColorTarget* target,
                                                   uint32_t slot) {
    if (!target || slot >= prosper::gpu::kColorTargetCount) return {};
    if (slot == 0)
        return {target->volume_depth, target->volume_first_slice, target->volume_slice_count,
                target->volume_guest_bytes};
    return target->volume_slots[slot];
}

// Slot `slot`'s persistent identity, from whichever field carries it.
inline uint64_t backend_color_slot_id(const BackendColorTarget* target, uint32_t slot) {
    if (!target || slot >= prosper::gpu::kColorTargetCount) return 0;
    if (slot == 0) return target->persistent_id;
    if (slot == 1) return target->persistent_id1;
    return target->persistent_id_slots[slot];
}

// One pass's volume shape: every active slot's view and the framebuffer layer count.
struct BackendVolumePass {
    std::array<BackendVolumeSlot, prosper::gpu::kColorTargetCount> slots{};
    uint32_t layers = 1;
    bool any = false;
    bool volume(uint32_t slot) const { return slot < slots.size() && slots[slot].depth != 0; }
};

// Admits a pass's colour slots as one layered framebuffer. Returns DrawDrop::Count when the pass
// can be expressed, else the reason it cannot. `seed0`/`seed1` are the slot-0/1 CPU seeds the pass
// would upload; slots 2+ read theirs from the target.
inline prosper::gpu::DrawDrop backend_volume_pass_shape(const BackendColorTarget* target,
                                                        uint32_t color_count, const uint8_t* seed0,
                                                        const uint8_t* seed1,
                                                        BackendVolumePass& pass) {
    using DD = prosper::gpu::DrawDrop;
    pass = {};
    uint32_t layers = 0;
    bool flat_slot = false;
    color_count = std::min(color_count, prosper::gpu::kColorTargetCount);
    for (uint32_t slot = 0; slot < color_count; ++slot) {
        const BackendVolumeSlot view = backend_color_volume_slot(target, slot);
        pass.slots[slot] = view;
        if (!view.depth) {
            flat_slot = true;
            continue;
        }
        pass.any = true;
        const uint8_t* seed = slot == 0 ? seed0 : slot == 1 ? seed1 : target->seed_slots[slot];
        if (seed) return DD::VolumeSeeded;
        const uint64_t id = backend_color_slot_id(target, slot);
        if (!id) return DD::VolumeNotPersistent;
        for (uint32_t other = 0; other < slot; ++other)
            if (pass.slots[other].depth && backend_color_slot_id(target, other) == id)
                return DD::VolumeMixedTarget;   // one allocation bound to two attachments
        if (layers && view.slice_count != layers) return DD::VolumeMixedTarget;
        layers = view.slice_count;
    }
    if (!pass.any) return DD::Count;
    if (flat_slot && layers != 1u) return DD::VolumeMixedTarget;
    pass.layers = layers;
    return DD::Count;
}

// True when a split of this pass carries every colour slot on the GPU: each later segment LOADs
// the retained image and no segment reads a slot back for the next one to seed from (CLAUDE.md
// P1). One persistent attachment always does. A multi-target pass does when every slot is a volume
// with an identity, because a volume slot is retained or its pass refused -- it has no transient
// fallback that would need its pixels carried through the CPU.
inline bool backend_split_carries_on_gpu(const BackendColorTarget* target, uint32_t colors) {
    if (!target) return false;
    if (colors <= 1u) return target->persistent_id != 0;
    for (uint32_t slot = 0; slot < colors && slot < prosper::gpu::kColorTargetCount; ++slot)
        if (!backend_color_volume_slot(target, slot).depth || !backend_color_slot_id(target, slot))
            return false;
    return true;
}

// Can the device hold this slot's volume as a colour attachment it can also sample and copy?
inline bool backend_volume_slot_fits(VkPhysicalDevice phys, VkFormat format, uint32_t width,
                                     uint32_t height, const BackendVolumeSlot& view) {
    constexpr VkImageUsageFlags usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageFormatProperties properties{};
    return vkGetPhysicalDeviceImageFormatProperties(
               phys, format, VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL, usage,
               VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT, &properties) == VK_SUCCESS &&
           width <= properties.maxExtent.width && height <= properties.maxExtent.height &&
           view.depth <= properties.maxExtent.depth;
}

// A retained slot's image is 3D when its view is a volume; the sampled view follows the image.
inline void backend_volume_image_info(VkImageCreateInfo& info, const BackendVolumeSlot& view) {
    if (!view.depth) return;
    info.imageType = VK_IMAGE_TYPE_3D;
    info.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    info.extent.depth = view.depth;
}

// The per-call attachment view: a 2D array over the slot's slice range of its retained 3D image.
inline VkResult create_volume_attachment_view(VkDevice dev, VkImage image, VkFormat format,
                                              const BackendVolumeSlot& view, VkImageView* out) {
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    info.format = format;
    info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, view.first_slice, view.slice_count};
    return create_render_image_view_checked(dev, info,
                                            RenderVkObjectCreateSite::VolumeAttachmentView, out);
}

// Every slice of the view's range holds a completed version (the LOAD condition).
inline bool volume_view_slices_valid(const PersistentColorTargetImage& image,
                                     const BackendVolumeSlot& view) {
    return image.valid_volume_slices.size() == view.depth &&
           std::all_of(image.valid_volume_slices.begin() + view.first_slice,
                       image.valid_volume_slices.begin() + view.first_slice + view.slice_count,
                       [](uint8_t slice) { return slice != 0; });
}

// Some slice holds content. Without maintenance9 a 2D-array view of a 3D image shares one layout
// across the whole mip, so CLEARing a new range must still transition from the retained image's
// real layout: an UNDEFINED initial layout could discard the other ranges' slices.
inline bool volume_has_valid_slice(const PersistentColorTargetImage* image) {
    return image &&
           std::any_of(image->valid_volume_slices.begin(), image->valid_volume_slices.end(),
                       [](uint8_t slice) { return slice != 0; });
}

// A retained slot's guest footprint changed shape: no earlier slice can stand for the new one.
inline void note_retained_volume_guest_bytes(PersistentColorTargetImage& image, uint64_t bytes) {
    if (image.volume_guest_bytes && bytes && image.volume_guest_bytes != bytes) {
        image.valid = false;
        std::fill(image.valid_volume_slices.begin(), image.valid_volume_slices.end(), 0u);
        invalidate_color_producer(image);
    }
    image.volume_guest_bytes = bytes;
}

// A failed or discarded batch: nothing this pass recorded may stand, in any slice.
inline void invalidate_retained_color(PersistentColorTargetImage* image) {
    image->valid = false;
    std::fill(image->valid_volume_slices.begin(), image->valid_volume_slices.end(), 0u);
}

// The pass's commands are recorded (speculatively valid until the batch fails). CLEAR initializes
// every pixel of the view's slices; a raster draw alone is no whole-slice proof, and LOAD keeps
// the slices an earlier call established. A volume is valid once every slice is.
inline void note_retained_color_recorded(PersistentColorTargetImage& image,
                                         const BackendVolumeSlot& view, bool cleared) {
    if (view.depth) {
        if (cleared)
            std::fill(image.valid_volume_slices.begin() + view.first_slice,
                      image.valid_volume_slices.begin() + view.first_slice + view.slice_count, 1u);
        image.valid =
            std::all_of(image.valid_volume_slices.begin(), image.valid_volume_slices.end(),
                        [](uint8_t slice) { return slice != 0; });
    } else {
        image.valid = true;
    }
    image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// A failed attempt to replace any part of a retained volume cannot leave an earlier version
// authoritative under the same guest identity. Completion/discard callbacks cover work that
// reached a command buffer; this guard covers every earlier refusal and Vulkan create error.
struct VolumeAttemptGuard {
    std::array<uint64_t, prosper::gpu::kColorTargetCount> ids{};
    ~VolumeAttemptGuard() {
        for (const uint64_t id : ids)
            if (id) invalidate_persistent_color_target(id);
    }
    void release() { ids.fill(0); }
};

// The 3D sampled views belong to the retained allocations; their 2D-array attachment views belong
// to this render call, including when the same allocation is reused for another range.
struct VolumeAttachmentViewGuard {
    VkDevice device = VK_NULL_HANDLE;
    std::array<VkImageView, prosper::gpu::kColorTargetCount> views{};
    ~VolumeAttachmentViewGuard() {
        for (const VkImageView view : views)
            if (view) vkDestroyImageView(device, view, nullptr);
    }
    void release() { views.fill(VK_NULL_HANDLE); }
};
