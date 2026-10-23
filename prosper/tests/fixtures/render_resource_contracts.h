#pragma once
#include "gpu/resources/shader_resources.hpp"
#include "gpu/state/render_state.hpp"
#include <vulkan/vulkan.h>
#include <array>
#include <functional>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <vector>

namespace prosper::test {

struct GpuDetileUpload;

// clang-format off: preserve the byte-identical resource contracts.
struct TexDesc { uint32_t binding; uint32_t w; uint32_t h; const uint8_t* rgba;
                 uint32_t max_aniso_ratio = 0; };   // #275: S# anisotropy ratio (0 = isotropic)

// The storage-buffer subset of FrameResource. The live frontend keeps these in a separate vector:
// a settled Blue Prince submit carries tens of thousands of buffers, and constructing the complete
// texture-capable object (currently 512 bytes) for every one only to leave all image state empty is
// measurable resource-preparation work. Tests and replay inputs may continue putting buffers in
// FrameResource; the backend accepts both representations through the same upload path.
#include "fixtures/fragment_draw_buffer_binding.h"
struct FrameBufferResource {
    uint32_t binding = 0;
    uint32_t set = 0;               // descriptor set: VS resources -> 0, PS resources -> 1 (they must not
                                    // share a set — both stages number bindings from 2, so one set would
                                    // collide binding 2/3 between stages and make the layout invalid).
    std::vector<uint32_t> dwords;   // owned storage-buffer contents (empty -> a 1-dword zero buffer)
    // A callback may point at a fully readable guest/capture range, or at its readable start.
    // The backend consumes and uploads this view synchronously; an explicit submission batch retains
    // only the completed Vulkan upload, never this pointer. Tests/replays may use the same contract
    // when their backing outlives render_draws_rgba().
    const uint32_t* dwords_view = nullptr;
    size_t dwords_view_count = 0;
    // Exact guest resource identity for a storage buffer. The backend may share an immutable upload
    // within one synchronous call only when both this identity and the complete captured bytes match;
    // zero keeps synthetic/replay resources conservatively distinct.
    uint64_t buffer_identity = 0;
    // Explicit live-producer proof: every word of this view is the direct guest VA, unmaterialized.
    // Hosted/capture/owned/padded inputs leave zero; numeric identity alone never grants a watch.
    uint64_t direct_guest_buffer_addr = 0;
    // Backend-owned PS5 GDS storage. Unlike guest/capture buffers this is one persistent,
    // zero-initialized 64 KiB allocation shared across ordered render calls.
    bool is_internal_gds = false;
    // Per-entry payloads for a RUNTIME-SELECTED descriptor array (#2412 stage 5). Empty -- the only
    // state any producer creates today -- means an ordinary single descriptor and every path below
    // behaves exactly as before.
    std::vector<std::vector<uint32_t>> table_entries;
    // Internal immutable device-local transaction plane. No host/guest payload is uploaded for
    // this binding. Its private owner prevents pool reuse until the ordered replay completes.
    std::shared_ptr<const FragmentDrawGpuBuffer> fragment_draw_buffer;
    const uint32_t* buffer_words_data() const {
        return dwords_view && dwords_view_count
            ? dwords_view : (dwords.empty() ? nullptr : dwords.data());
    }
    size_t buffer_word_count() const {
        return dwords_view && dwords_view_count ? dwords_view_count : dwords.size();
    }
    uint32_t descriptor_arity() const {
        return table_entries.empty() ? 1u : static_cast<uint32_t>(table_entries.size());
    }
    uint32_t written_descriptor_count() const {
        return is_internal_gds ? 1u : descriptor_arity();
    }
};

// One resource for the general N-binding path (render_triangle_rgba's `gres`): a storage buffer
// (dwords non-empty, tex_rgba null), combined image sampler, or storage image at `binding`. Lets a
// real game shader that declares several buffers and images have each bound distinctly.
struct FrameResource : FrameBufferResource {
    const uint8_t* tex_rgba = nullptr;   // non-null => a texture; then tw/th are its dimensions
    // Immutable tiled snapshot plus conversion resources; the submission retains
    // this owner through completion. Never interpreted as CPU RGBA pixels.
    std::shared_ptr<GpuDetileUpload> gpu_detile;
    // Readable byte span behind tex_rgba. Zero preserves the historical single-sample fixture
    // contract; multisample plane uploads require an explicit full span and reject before Vulkan.
    size_t tex_byte_size = 0;
    // Optional immutable owner for borrowed frontend pixels. Keeping this beside the raw pointer
    // makes FrameResource copies/moves retain the source through synchronous backend upload setup.
    std::shared_ptr<const std::vector<uint8_t>> tex_rgba_owner;
    // A DCC fast-clear is already a complete image description. Keep it compact until command
    // recording and initialize the full-size Vulkan image with vkCmdClearColorImage instead of
    // allocating, filling, and uploading tens of MiB of identical CPU pixels.
    bool has_uniform_color = false;
    std::array<float, 4> uniform_color{};
    uint32_t tw = 0, th = 0, td = 1;
    // Guest sample count. 2D_MSAA is uploaded as a single-sample 2D-array with one plane/layer per
    // sample; the Vulkan image itself always keeps VK_SAMPLE_COUNT_1_BIT.
    uint32_t sample_count = 1;
    // T#-declared mip-chain length (#1272). 1 (default) = single-level upload, the historical
    // behavior. >1 lets the backend generate a box-filtered chain — bounded by this declared count —
    // for plain-2D RGBA8 sampled textures, so minification stops point-sampling through dense art.
    uint32_t declared_mip_levels = 1;
    // Levels ALREADY present in `tex_rgba`, concatenated level 0 first, each tightly packed at its
    // own extent (backend_texture_bytes). 0/1 = level 0 only, the historical contract. A native BCn
    // texture carries the guest's own chain this way (#3873) because a block format cannot be
    // blit-generated; the backend then copies every level instead of running the blit cascade.
    uint32_t uploaded_mip_levels = 0;
    uint32_t img_dim = 1;             // ShaderResource/MIMG dim (1=2D, 2=3D); depth-1 3D stays 3D
    // #325: the guest T# is one prosper treats as a layered array, as decided by
    // guest_texture_is_uploaded_array(). The view type keys on this IN ADDITION to
    // `sample_count > 1`, which stays for the MSAA plane array -- that shape has no guest_array flag
    // and must keep its own arm. What this adds is the case the count cannot express: the
    // recompiler declares OpTypeImage Arrayed from the same predicate and cannot see how many layers
    // the uploader actually decoded -- a footprint cap, an RTT hit or a DCC fast-clear can all leave
    // the count at 1 -- so a view keyed on the count ALONE would silently become 2D under an Arrayed
    // declaration. A one-layer 2D_ARRAY view is legal and samples layer 0.
    bool guest_array = false;
    // Renderer-owned RTTs keep their native format between producer and consumer. Guest-backed
    // textures still arrive through the existing RGBA8 decoder unless explicitly tagged otherwise.
    VkFormat texture_format = VK_FORMAT_R8G8B8A8_UNORM;
    // StorageImage is a distinct Vulkan descriptor contract: no sampler, STORAGE usage, and GENERAL
    // layout. Keep the flag independent of tex_rgba because both sampled and storage images upload
    // decoded pixels through that pointer.
    bool is_storage_image = false;
    // Reflected OpTypeImage Sampled Type versus the materialized Vulkan view. Storage descriptors
    // without a proven class, or whose frontend could not produce the requested representation,
    // fail before Vulkan command recording instead of relying on undefined format compatibility.
    prosper::gpu::SpirvImageNumericClass storage_image_numeric_class =
        prosper::gpu::SpirvImageNumericClass::Unknown;
    bool storage_image_contract_valid = true;
    // Writable graphics storage images must publish their final linear texels before this backend call
    // returns. The live frontend uses this to restore the guest's linear/tiled layout and notify the
    // normal guest-write invalidation path, so later graphics, sampled aliases, compute, DMA, and CPU
    // consumers all observe the image side effect. An empty callback keeps read-only/test fixtures on
    // the ordinary asynchronous upload path.
    std::function<void(const uint8_t*, size_t)> storage_image_writeback;
    // Sampler state (Texture only). Defaults = LINEAR + clamp-to-edge — the harness's prior fixed
    // sampler — so render tests that build FrameResources directly stay byte-identical. The live path
    // fills these from the decoded S# (shader_resources.hpp). filter: 0=nearest, 1=linear; addr = Gen5
    // SQ_TEX CLAMP enum (0=wrap, 1=mirror, 2=clamp-last-texel, 6/7=border).
    uint32_t mag_filter = 1, min_filter = 1, mip_filter = 0;
    uint32_t addr_uvw[3] = {2, 2, 2};
    // Remaining S# sampler fields (#262). Defaults reproduce the current Vulkan sampler exactly (border
    // transparent-black, LOD 0..0, no bias), so FrameResources built directly by tests are byte-identical.
    uint32_t border_color_type = 0;
    float    min_lod = 0.0f, max_lod = 0.0f, lod_bias = 0.0f;
    // Anisotropy ratio (S# WORD0[11:9]; maxAnisotropy = 1<<ratio). 0 = isotropic (the default) -> the
    // sampler is byte-identical to before, so tests building FrameResources directly are unaffected (#275).
    uint32_t max_aniso_ratio = 0;
    // T# DST_SEL channel swizzle (SQ_SEL per channel: 0=0,1=1,4=R,5=G,6=B,7=A). Default = identity
    // (R,G,B,A) == a no-op VkComponentMapping, so tests that build FrameResources directly are unchanged.
    uint32_t swizzle[4] = {4, 5, 6, 7};
    // Original guest CB_COLOR format when these pixels came from a renderer-owned target. The
    // backend stores color attachments in `backend_color_format()`'s canonical component order;
    // a later T# still describes the guest allocation's component order. Keeping the producer
    // format lets the sampled view compose those two mappings instead of applying the storage-order
    // swap a second time. Undefined means ordinary guest texture / no such canonicalization.
    VkFormat render_target_guest_format = VK_FORMAT_UNDEFINED;
    // Non-zero only when the frontend has revalidated the complete guest source backing and can
    // prove these decoded pixels are the same content version as a prior callback. The Vulkan
    // backend may retain an uploaded image under this ID; zero remains callback-local/transient.
    uint64_t persistent_texture_id = 0;
    // A non-zero version lets mutable, exactly-validated guest textures retain the allocation named
    // above while replacing its contents. Equal versions skip upload; a newer version records one
    // ordered refresh into the existing image. Zero preserves the historical immutable-ID contract.
    uint64_t persistent_texture_version = 0;
    // Non-zero identifies a color target retained by an earlier backend call. When that exact
    // target is available, bind its GPU image directly instead of uploading `tex_rgba` again.
    // `tex_rgba` remains an optional conservative fallback for an invalidated/missing target.
    uint64_t persistent_render_target_id = 0;
    // A guest mip chain is rendered as one independent persistent color target per level because
    // CB_COLOR names a single mip view at a time. A later T# names the complete allocation as one
    // sampled image. The frontend records the renderer-owned level identities here so the backend
    // can assemble them into one Vulkan mip image instead of binding level zero and silently
    // discarding every smaller level. Zero entries are missing levels; the backend derives those
    // from the previous available destination level. The count is the requested chain length.
    std::array<uint64_t, 16> persistent_render_target_mip_ids{};
    uint32_t persistent_render_target_mip_count = 0;
    // Non-zero identifies a persistent depth/stencil surface whose DEPTH plane this resource
    // samples (a shadow map / depth pyramid tap, #1275). The backend binds the retained Vulkan
    // depth image directly — prosper never writes rendered depth back to guest memory, so there
    // is no CPU fallback; an invalidated/missing surface falls through to the guest-byte decode.
    uint64_t persistent_depth_target_id = 0;
    // Exact typed-storage result borrowed from the live compute cache. The image and device handles
    // are opaque here only at the frontend boundary; this Vulkan backend verifies the device,
    // transitions from `borrowed_image_layout` for sampling, restores that layout afterward, and
    // retains the lease until the submission has completed.
    void* borrowed_compute_image = nullptr;
    void* borrowed_compute_device = nullptr;
    uint32_t borrowed_compute_image_layout = 0;
    // A 2D-array compute result cannot be bound directly to the vertically-stacked 2D image expected
    // by the cube lowering. Non-zero asks the backend to copy these source layers into consecutive
    // vertical regions of a sampled image without visiting host/guest memory.
    uint32_t borrowed_compute_vertical_stack_layers = 0;
    std::shared_ptr<void> borrowed_compute_image_lease;
    bool is_texture() const {
        return tex_rgba != nullptr || gpu_detile || has_uniform_color || persistent_render_target_id != 0 ||
               persistent_depth_target_id != 0 || borrowed_compute_image != nullptr;
    }
    // Per-entry payloads for a RUNTIME-SELECTED descriptor array (#2412 stage 5). Empty -- the only
    // state any producer creates today -- means an ordinary single descriptor and every path below
    // behaves exactly as before.
    //
    // Arity is DERIVED from this vector rather than carried in a separate count field, deliberately:
    // two sources of truth for "how many descriptors" is precisely the disagreement that makes a
    // layout declare N while a write supplies M, which is a validation error at bind time and a wrong
    // shader read if validation is off. One vector, one length, no way to disagree.
    // The number of descriptors this binding DECLARES: 1 ordinary, N for a table-indexed array.
    // This is the guest's intent, not what any Vulkan object ends up carrying -- for that, use
    // `written_descriptor_count()` below.
    //
    // This comment used to say the pool, the layout and the write "all three must agree, so all
    // three call this". They must agree; they did not all call this. The pool counts a texture as
    // ONE (`is_texture()` branches to `++storage_images` / `++sampled_images` regardless of arity)
    // and the texture and internal-GDS writes each supply ONE, while the layout took this value for
    // every class -- so the invariant the comment asserted was the one thing not being maintained.
    // #2477.
    // The number of descriptors the WRITE path actually supplies, and therefore what the POOL and the
    // LAYOUT must both declare (#2477). Only the storage-buffer path loops over `table_entries`; the
    // texture / storage-image write and the internal-GDS write each supply exactly one descriptor.
    //
    // Taking `descriptor_arity()` in the layout while the write supplied one produced precisely the
    // layout-declares-N / write-supplies-M disagreement the `table_entries` comment above exists to
    // prevent: the layout would declare N, elements 1..N-1 would never be written, and binding the
    // set would read undefined descriptors. The two sites are ~500 lines apart, so nothing local
    // showed it.
    //
    // This is a fail-visible GUARD, not array support for those classes: it keeps the three sites in
    // agreement and says so loudly if a producer ever populates `table_entries` on a class whose
    // write cannot honour it. Unreachable today -- nothing sets `table_entries` on a texture or the
    // internal GDS buffer -- which is why it is a guard rather than a fix, and why it needs no
    // producer to be worth having. `CONFIDENCE: HIGH` that silent truncation is the wrong direction
    // here: a layout that quietly declares fewer descriptors than the guest asked for is the
    // under-reporting failure, and it is the one nobody files.
    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method): preserve the texture-specific descriptor count.
    uint32_t written_descriptor_count() const {
        return is_texture() || is_internal_gds ? 1u : descriptor_arity();
    }
};

// Optional color-target contract for the live backend. A non-zero ID gives the target a stable
// guest identity across calls. Existing contents are loaded only when both `load_existing` and a
// valid matching image are present. `readback=false` leaves the completed result GPU-resident and
// deliberately returns an empty CPU vector.
struct BackendColorTarget {
    uint64_t persistent_id = 0;
    bool load_existing = true;
    bool readback = true;
    VkFormat format = VK_FORMAT_UNDEFINED;
    // MRT1 has an independent guest identity and lifetime. Keeping its contract beside color0 lets
    // paired passes retain both attachments without changing the established single-target callers.
    uint64_t persistent_id1 = 0;
    bool load_existing1 = true;
    bool readback1 = true;
    VkFormat format1 = VK_FORMAT_UNDEFINED;
    // Slots 2..7. The named fields above predate the complete array and remain the contract for
    // slots 0 and 1; these carry the same two properties for the rest.
    //
    // Without them every slot above 1 was a TRANSIENT image created per backend call and cleared
    // with LOAD_OP_CLEAR, so returning to the same allocation in a later render group erased what an
    // earlier group had drawn there. That was unreachable while the fragment recompiler could only
    // export MRT0/MRT1; the moment it could export five, GTA V's G-buffer -- which accumulates
    // across roughly sixteen pass groups per frame -- kept only the last group's slots 2..4.
    std::array<uint64_t, prosper::gpu::kColorTargetCount> persistent_id_slots{};
    std::array<bool, prosper::gpu::kColorTargetCount> load_existing_slots{};
    // Per-slot twin of `readback`/`readback1`. Defaults to true so every existing caller keeps its
    // pixels; the depth-feedback splitter turns it off for non-final segments, which otherwise pay a
    // full-extent copy per slot per segment purely because `color_count > 2` forces a readback.
    std::array<bool, prosper::gpu::kColorTargetCount> readback_slots{
        true, true, true, true, true, true, true, true};
    // Initial contents for slots 2..7, the per-slot twin of `seed_rgba`/`seed_rgba1`. A depth
    // feedback split renders a logical pass in several physical passes; when the slots are not
    // persistent (the caller passed no identities, e.g. every path that runs with live GPU targets
    // disabled) the ONLY way a later segment sees an earlier one's pixels is to be seeded with them.
    // Without it the final segment created fresh transient attachments and cleared away everything
    // the earlier segments drew into MRT2+.
    std::array<const uint8_t*, prosper::gpu::kColorTargetCount> seed_slots{};
    // Slot 1's seed when the caller receives it through BackendMrtOutputs instead of `out_rgba1`.
    // The live renderer uses that API, so on a split its slot-1 readback landed in
    // `mrt_outputs.colors[1]` while the wrapper only carried a seed forward inside `if (out_rgba1)`
    // -- losing MRT1 on every non-final segment. The claim that "slots 0 and 1 already carry" was
    // true only of the legacy explicit-out_rgba1 API.
    const uint8_t* seed_rgba1_slot = nullptr;
    // Slot 0 may address a bounded view of one persistent 3D allocation. A zero depth keeps the
    // established 2D contract. The allocation identity includes the complete depth; the view
    // range does not, so later draws can fill other slices without replacing the image.
    uint32_t volume_depth = 0;
    uint32_t volume_first_slice = 0;
    uint32_t volume_slice_count = 0;
    uint64_t volume_guest_bytes = 0; // native tiled footprint, supplied by the live producer
    // Slots 1..7 may each address a view of their own persistent 3D allocation, on slot 0's contract
    // above (#4643). Entry 0 is unused. How one pass's slots must agree: render_volume_slots.h.
    struct VolumeSlot {
        uint32_t depth = 0, first_slice = 0, slice_count = 0;
        uint64_t guest_bytes = 0;
    };
    std::array<VolumeSlot, prosper::gpu::kColorTargetCount> volume_slots{};
    // Zero means no complete guest layout bound, independently of native image validity.
    std::array<uint64_t, prosper::gpu::kColorTargetCount> guest_footprint_bytes{};
};

// Optional complete-MRT readback contract. `color_count` is the active Vulkan attachment prefix
// (1..8); slots 1..count-1 receive their native-format bytes after rendering. Slot 0 remains the
// function's return value for compatibility with every existing caller.
struct BackendMrtOutputs {
    uint32_t color_count = 1;
    std::array<std::vector<uint8_t>, prosper::gpu::kColorTargetCount> colors;
};

// The first exact guest-MSAA contract is intentionally narrow. Ordinary resources retain their
// historical implicit byte-span behavior; a 2D_MSAA plane array must prove all four complete R32F
// planes are readable before any Vulkan object or memcpy is attempted.
// Vulkan 1.4 guarantees at least 2048 maxImageArrayLayers. Keep this admission ceiling even
// on devices supporting more layers. It does not prove format/usage-specific creatability or
// allocation success; checking every upload's VkResult is still owed by #3045.
inline constexpr uint32_t kBackendMaxArrayLayers = 2048u;
// clang-format on

bool backend_volume_view_valid(uint32_t depth, uint32_t first, uint32_t count);
bool backend_color_volume_view_valid(const BackendColorTarget& target);
std::array<uint32_t, 4> backend_sampled_component_swizzle(const FrameResource& resource);
bool backend_storage_image_numeric_contract_valid(const FrameResource& resource);
bool backend_texture_array_span_valid(const FrameResource& resource);
bool backend_texture_plane_span_valid(const FrameResource& resource);

} // namespace prosper::test
