#pragma once

#include "gpu/resources/image_identity.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace prosper::frontend {

// A cube T# sampled through a module-declared `OpTypeImage Dim=Cube`: the view is a real
// VK_IMAGE_VIEW_TYPE_CUBE over a CUBE_COMPATIBLE image (#657, #2790).
//
// Declared HERE, beside the sizing it feeds, and used by the admission gate in live_compute.cpp
// rather than restated there. The two must agree: admitting a shape the sizing does not know about
// allocates staging for one layer and then packs `width * height * depth` texels out of it. That is
// a SIGSEGV in storage_pack_float16x4_f16c, which is how this pairing was found.
//
// Exactly six layers. A cube ARRAY (depth > 6) addresses more than one cube, and which cube a
// sample resolves to is not established by anything here -- those stay skipped and loud.
//
// There is deliberately NO companion predicate for a cube bound as 2D-ARRAY STORAGE. That shape is
// admissible on the binding side, but the storage WRITEBACK cannot publish it: its layered arm is
// `backend_uses_2d_array`, which requires `img_dim == 5`, so a cube would write six faces and
// publish one -- silently partial where the skip is loud. See #3742.
inline bool compute_native_cube_sampled(
    const prosper::gpu::ShaderResource& resource,
    const prosper::gpu::SpirvDescriptorBinding& descriptor) {
    return resource.img_dim == 3 &&
           descriptor.kind != prosper::gpu::SpirvDescriptorKind::StorageImage &&
           descriptor.image_dim == 3u && !descriptor.image_arrayed &&
           !descriptor.image_multisampled && resource.depth == 6u;
}

// Metadata only: this is the shape used by live compute before allocating an image. Validation
// still belongs to the materializer; an invalid descriptor must not become valid by joining a group.
inline prosper::gpu::ComputeImageViewShape compute_image_alias_shape(
    const prosper::gpu::ShaderResource& resource,
    const prosper::gpu::SpirvDescriptorBinding& descriptor) {
    return {
        descriptor.kind == prosper::gpu::SpirvDescriptorKind::StorageImage,
        resource.img_dim == 2 ? resource.depth : 1u,
        (resource.img_dim == 5 && descriptor.image_dim == 1u && descriptor.image_arrayed) ||
                compute_native_cube_sampled(resource, descriptor)
            ? resource.depth : 1u,
    };
}

struct StorageImageAliasGroup {
    size_t owner_index = SIZE_MAX;
    std::vector<uint32_t> bindings;
    bool readable = false;
    bool writable = false;
    bool atomic = false;

    bool write_only() const { return writable && !readable && !atomic; }
    bool read_only() const { return readable && !writable && !atomic; }
};

struct StorageImageAliasPlan {
    bool valid = true;
    const char* decline_reason = nullptr;
    // Non-storage descriptors have no group. Indices refer to the supplied descriptor order.
    std::vector<size_t> group_for_image;
    std::vector<StorageImageAliasGroup> groups;
};

// Seeding must account for EVERY descriptor that will share the canonical image, before that
// image is uploaded or its raw result is reused. This follows the existing late storage-alias identity, not the
// narrower early-fold optimization: storage never imports renderer/depth/value-reuse images, so
// its late same_backing_representation is true. Reflection-format differences must not hide a
// reader here when the late path can still fold it. This plan does not change actual folding.
inline StorageImageAliasPlan plan_storage_image_aliases(
    std::span<const prosper::gpu::SpirvDescriptorBinding> descriptors,
    const prosper::gpu::ShaderResourceTable& resources) {
    StorageImageAliasPlan plan;
    plan.group_for_image.assign(descriptors.size(), SIZE_MAX);
    std::vector<const prosper::gpu::ShaderResource*> resolved(descriptors.size(), nullptr);
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& descriptor = descriptors[i];
        if (descriptor.kind != prosper::gpu::SpirvDescriptorKind::StorageImage) continue;
        const auto* resource = resources.by_binding(descriptor.binding);
        if (!resource) {
            plan.valid = false;
            plan.decline_reason = "storage-alias-resource-missing";
            continue;
        }
        resolved[i] = resource;
        const auto shape = compute_image_alias_shape(*resource, descriptor);
        size_t group_index = 0;
        for (; group_index < plan.groups.size(); ++group_index) {
            const size_t owner = plan.groups[group_index].owner_index;
            if (prosper::gpu::shader_resource_same_view(
                    *resolved[owner], *resource,
                    compute_image_alias_shape(*resolved[owner], descriptors[owner]), shape, true)) {
                // Equal guest bytes and layer counts do not imply equal Vulkan view types.
                // In particular, a one-layer 2D_ARRAY storage view cannot be bound to a
                // sibling's ordinary 2D declaration. Splitting writable aliases into two
                // images would also lose one sibling's stores at whole-image writeback.
                if (resource->img_dim == 1 && resource->depth == 1 &&
                    descriptors[owner].image_dim == 1 && descriptor.image_dim == 1 &&
                    descriptors[owner].image_arrayed != descriptor.image_arrayed) {
                    plan.valid = false;
                    plan.decline_reason = "storage-alias-mixed-array-view";
                    return plan;
                }
                break;
            }
        }
        if (group_index == plan.groups.size()) {
            plan.groups.emplace_back();
            plan.groups.back().owner_index = i;
        }
        plan.group_for_image[i] = group_index;
        auto& group = plan.groups[group_index];
        group.bindings.push_back(descriptor.binding);
        group.readable |= descriptor.readable;
        group.writable |= descriptor.writable;
        group.atomic |= descriptor.atomic_access;
    }
    // Keep a stable member list for whole-group access decisions and exact per-store tracking.
    for (auto& group : plan.groups) std::sort(group.bindings.begin(), group.bindings.end());
    return plan;
}

} // namespace prosper::frontend
