#include "metadata_kind_correlation.hpp"

namespace prosper::gpu {

CompressionMetadataKind correlate_compression_metadata_kind(
    const MetadataKindRequest& request,
    const std::vector<RetainedDepthCorrelation>& depth,
    const std::vector<RetainedColorCorrelation>& color) {
    // Nothing to correlate. Not an error, and not an invitation to guess from the format.
    if (!request.metadata_addr || !request.resource_addr) return CompressionMetadataKind::Unknown;

    // HTILE: a retained depth surface names this metadata plane, AND one of its aspect bases is the
    // resource being asked about. The second half is what makes a reused metadata address safe: without
    // it, a stale entry whose HTILE address had been recycled would identify an unrelated resource.
    for (const RetainedDepthCorrelation& ds : depth) {
        if (!ds.htile || ds.htile != request.metadata_addr) continue;
        const bool resource_is_an_aspect_of_this_surface =
            (ds.depth_read && ds.depth_read == request.resource_addr) ||
            (ds.depth_write && ds.depth_write == request.resource_addr) ||
            (ds.stencil_read && ds.stencil_read == request.resource_addr) ||
            (ds.stencil_write && ds.stencil_write == request.resource_addr);
        if (resource_is_an_aspect_of_this_surface) return CompressionMetadataKind::Htile;
    }

    // DCC: a retained colour surface AT THIS RESOURCE ADDRESS registered this metadata plane. Also a
    // positive correlation on both halves -- never "it is not depth, therefore it is colour".
    for (const RetainedColorCorrelation& rt : color) {
        if (!rt.resource_addr || rt.resource_addr != request.resource_addr) continue;
        if (rt.dcc_metadata_addr && rt.dcc_metadata_addr == request.metadata_addr)
            return CompressionMetadataKind::Dcc;
    }

    // Uncorrelated. The format is deliberately NOT consulted here: a Float32x1 view is not necessarily
    // depth and a Uint32x1 alias is not necessarily colour, so any format-based fallback is a guess
    // wearing evidence's clothes. Unknown authorizes nothing downstream.
    return CompressionMetadataKind::Unknown;
}

}  // namespace prosper::gpu
