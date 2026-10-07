#pragma once
// Private image materialization interface; references retain the draw builder's ownership.
#include "shared/live/submit_renderer/dcc_fast_clear_admission.hpp"
#include "shared/live/submit_renderer/draw_resources.hpp"

namespace prosper::frontend::submit_renderer {

// One image binding may retain the resource, skip only that binding, or reject the draw.
// The status owns no payload: the caller keeps first-rejection accounting and resource-loop control.
enum class ImageDisposition { Keep, Skip, Reject };
struct ImageResourceStatus {
    ImageDisposition disposition;
    DropReason reason = DropReason::Unattributed;
};

// Per-binding locals die in the original resource loop. This context owns no values.
struct ImageBindingContext {
    const prosper::gpu::ShaderResourceTable *& t;
    uint32_t& set;
    uint64_t& shader_identity;
    const prosper::gpu::ShaderResource & r;
    const prosper::gpu::SpirvDescriptorBinding *& reflected_binding;
    prosper::frontend::TextureReferenceCensus *const& texref_census;
    uint64_t& texref_census_key;
    bool& resource_rtt_hit;
    bool& resource_compute_image_hit;
    bool& resource_compute_image_candidate;
    bool& resource_compute_depth_hybrid;
    uint64_t& resource_compute_producer_order;
    uint32_t& resource_compute_depth_overlay_mask;
    bool& resource_local_reuse;
    bool& resource_persistent_hit;
    bool& resource_persistent_submit_reuse;
    bool& resource_persistent_miss;
    bool& resource_persistent_invalidation;
    double& resource_texture_validation_ms;
    size_t& resource_texture_validated_bytes;
    size_t& resource_texture_source_bytes;
    int& resource_texture_submit_query;
    int& resource_texture_watch_query;
    uint32_t& resource_texture_watch_stability;
    bool& resource_texture_exact_validation;
    bool& resource_texture_watch_active;
    bool& resource_texture_watch_disabled;
    bool& resource_texture_watch_only;
    bool& resource_has_live_rtt;
    bool& resource_has_ds_live;
    bool& resource_persistent_candidate;
    size_t& resource_persistent_source_size;
    std::optional<prosper::test::FrameResource>& full_resource;
    const bool& normalized_sampling;
    const bool& writable_storage_image;
    const bool& no_direct_texture_source;
};

size_t consume_image_renderer_mip_chain(DrawResourceContext& ctx, const RendererMipChainLayout& chain, uint32_t width, uint32_t height, VkFormat format);
size_t acquire_image_texstore_slot(DrawResourceContext& ctx);
void clear_image_depth_array_snapshots(DrawResourceContext& ctx);
size_t copy_shader_resource(const prosper::gpu::ShaderResource& r, uint8_t* dst, uint64_t addr, size_t n);
const uint8_t* direct_shader_resource_source(const prosper::gpu::ShaderResource& r, const bool& no_direct_texture_source, uint64_t addr, size_t n);
const uint8_t* stage_tiled_shader_resource(const prosper::gpu::ShaderResource& r, const bool& no_direct_texture_source, prosper::frontend::DecodeScratchPool::Lease& lease, uint64_t addr, size_t n, size_t& got);
size_t copy_shader_dcc_metadata(const prosper::gpu::ShaderResource& r, uint8_t* dst, size_t n);

// PROSPER_DIAG_REF_OUTPUTS materialize_image_resource binding: resource_texture_validation_ms, resource_texture_validated_bytes, resource_texture_submit_query, resource_texture_watch_query, resource_texture_watch_active, resource_texture_watch_disabled, resource_texture_watch_only, resource_texture_watch_stability
ImageResourceStatus materialize_image_resource(DrawResourceContext& ctx, ImageBindingContext& binding, const prosper::gpu::DrawItem& draw, prosper::test::BackendSubmissionBatch* producer_batch);

} // namespace prosper::frontend::submit_renderer
