#pragma once
// library_descriptor_budget.hpp — how big the library view's ONE Vulkan descriptor pool has to be, and
// the accounting that keeps a set from being allocated out of a pool that has no room left (#1649).
//
// Pure: no Vulkan, no SDL, no ImGui. That is deliberate — the sizing arithmetic and the exhaustion rule
// are the parts that can be wrong in a way nobody notices until someone owns 250 games, and they are the
// parts a unit test can actually pin down. library_ui.cpp and library_media.cpp own the handles.
//
// WHY A BUDGET AND NOT A RETURN-VALUE CHECK
//
// The obvious guard — call ImGui_ImplVulkan_AddTexture() and check what it returned — is too late. That
// function allocates a descriptor set, ignores the VkResult beyond a diagnostic hook, and then passes the
// set straight into vkUpdateDescriptorSets(). vkAllocateDescriptorSets() is required to write
// VK_NULL_HANDLE into every output slot when it fails, so on an exhausted pool the update is issued with
// dstSet == VK_NULL_HANDLE: a validation error and undefined behaviour that has already happened by the
// time the caller sees the return value. third_party/imgui is vendored verbatim and must not be patched,
// so the only correct place to stop is BEFORE the call — hence acquire_texture_set() below, which never
// invokes the allocator once the pool's sets are spoken for.
//
// THREADING: a DescriptorBudget is not synchronized. Both users are on the UI thread — the media layer's
// threading contract (library_media.hpp) already puts every Vulkan call there.

#include <cstddef>
#include <cstdint>
#include <utility>

namespace prosper::frontend {

// ImGui's Vulkan backend allocates ONE combined-image-sampler set from the same pool for its font atlas,
// and rebuilds it destroy-then-create (imgui_impl_vulkan.cpp's ImGui_ImplVulkan_CreateFontsTexture), so
// exactly one is live at a time. The backend's own IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE is
// 1 for the same reason. prosper never allocates this one, so it is reserved out of the budget rather
// than counted in it.
inline constexpr uint32_t kLibraryImGuiReservedSets = 1;

// Upper bound for the background-art layer (#1630), which shares the pool with the covers:
//   kBackgroundCacheCapacity (3) resident
// + 1 inserted before the LRU evicts, in insert_background()
// + kRetireFrames (3) evicted-but-not-yet-collected, held so an in-flight frame can still sample them
// + 1 slack
// At most one background is evicted per frame (one upload is in flight at a time and it is published in
// poll_upload), and collect_retired() runs at the top of every update(), so the live peak is nearer 6;
// the two spare sets cost nothing and leave room for a change in that cadence. Stated as a literal
// rather than derived from launcher_media.hpp's constants so this header stays free of every other
// dependency; the arithmetic is written out so a change there is checkable by eye.
inline constexpr uint32_t kLibraryBackgroundSets = 8;

// A floor, so a run that starts on an empty or tiny games folder still has room for the font atlas, the
// backgrounds and a handful of covers before the first grow.
inline constexpr uint32_t kLibraryMinPoolSets = 32;

// A ceiling, so a directory full of junk cannot ask the driver for an absurd pool. Past this the covers
// degrade one image at a time through the budget below, which is the same graceful path a title with an
// undecodable icon0.png already takes. 8192 covers would need far more VRAM than any of them can have.
inline constexpr uint32_t kLibraryMaxPoolSets = 8192;

// maxSets (and the matching descriptorCount) for a library of `title_count` titles: one set per cover,
// plus the background allowance, plus ImGui's own, clamped to [kLibraryMinPoolSets, kLibraryMaxPoolSets].
// Saturating, so a nonsense count cannot wrap into a tiny pool.
inline constexpr uint32_t library_descriptor_pool_sets(size_t title_count) {
    const size_t overhead = static_cast<size_t>(kLibraryBackgroundSets) + kLibraryImGuiReservedSets;
    size_t want = title_count > SIZE_MAX - overhead ? SIZE_MAX : title_count + overhead;
    if (want < kLibraryMinPoolSets) want = kLibraryMinPoolSets;
    if (want > kLibraryMaxPoolSets) want = kLibraryMaxPoolSets;
    return static_cast<uint32_t>(want);
}

// How many sets of a pool of `pool_sets` prosper itself may allocate — everything except ImGui's own.
inline constexpr uint32_t library_texture_budget(uint32_t pool_sets) {
    return pool_sets > kLibraryImGuiReservedSets ? pool_sets - kLibraryImGuiReservedSets : 0;
}

// Counts the descriptor sets prosper has taken out of the library's pool. Covers and backgrounds draw
// from the same allowance, which is the whole point: they share one pool, so only one number can tell
// either of them that the next allocation would fail.
class DescriptorBudget {
public:
    // Start over with a new capacity. Used at init and after the pool is regrown; `used` returns to 0
    // because a regrow releases every set prosper held.
    void reset(uint32_t capacity) {
        capacity_ = capacity;
        used_ = 0;
    }

    // Take one slot. False means the pool is full and the caller MUST NOT allocate.
    bool acquire() {
        if (used_ >= capacity_) return false;
        ++used_;
        return true;
    }

    // Give one back. Clamped rather than wrapping: an unbalanced release is a bug, and a wrapped `used_`
    // would turn it into a silently over-allocating pool instead of a merely wrong counter.
    void release() {
        if (used_ > 0) --used_;
    }

    uint32_t capacity() const { return capacity_; }
    uint32_t used() const { return used_; }
    uint32_t available() const { return capacity_ > used_ ? capacity_ - used_ : 0; }

private:
    uint32_t capacity_ = 0;
    uint32_t used_ = 0;
};

// Allocate a descriptor set through `add`, but ONLY if the budget has room — see the header comment for
// why checking `add`'s result instead is too late. Returns `none` without calling `add` on an exhausted
// budget. Templated on the handle type so this header stays Vulkan-free and the rule is testable with a
// fake allocator that records whether it ran.
//
// `add` failing for some other reason (out of device memory, say) still returns `none`; the slot is given
// straight back, so a transient failure does not permanently shrink the pool's usable capacity.
template <class Set, class AddFn>
Set acquire_texture_set(DescriptorBudget& budget, Set none, AddFn&& add) {
    if (!budget.acquire()) return none;
    Set set = std::forward<AddFn>(add)();
    if (set == none) budget.release();
    return set;
}

// The mirror of acquire_texture_set: hand `set` to `remove`, clear it, and return its slot. A null handle
// is a no-op, so this is safe on a half-built resource and safe to call twice.
template <class Set, class RemoveFn>
void release_texture_set(DescriptorBudget& budget, Set& set, Set none, RemoveFn&& remove) {
    if (set == none) return;
    std::forward<RemoveFn>(remove)(set);
    set = none;
    budget.release();
}

} // namespace prosper::frontend
