// Establish WHICH kind of compression metadata an address is, by POSITIVE correlation with retained
// renderer state — never by elimination.
//
// The descriptor cannot answer it. T# WORD6[21] is set for depth/stencil and colour alike, with AMD PAL
// putting HTILE's address in the same `meta_data_address` field it puts DCC's, and the T# format is not
// enough either: a Float32x1 view is not necessarily depth, and a Uint32x1 raw alias is not necessarily
// colour.
//
// WHY THIS IS A PURE FUNCTION OVER EXPLICIT STATE. An earlier version lived inside the renderer's
// registration lambda and answered by elimination — HTILE on a metadata-address match, `Unknown` for
// Float32x1, and **DCC for everything else**. Two defects followed, and both are the same mistake:
//   * inventing DCC identity from an absence. An uncorrelated Uint32x1 raw/depth alias was classified
//     DCC, and its all-0xff bytes then authorized reading the base — while the whole point of the
//     source-authority policy is that the kind must be KNOWN before those bytes mean anything.
//   * ignoring resource identity. A stale retained entry whose metadata address had been reused
//     identified HTILE without proving the metadata belonged to THIS resource.
// Both answers are now positive correlations over metadata AND resource identity, and anything
// uncorrelated is `Unknown` — which authorizes nothing downstream, so it is a safe answer.
//
// It is pure so it can be mutation-tested at the site that ships. Testing the renderer's lambda through
// a synthetic callback proved only that the plumbing forwards: mutating the real lambda to answer DCC
// unconditionally left that test green.
#pragma once

#include <cstdint>
#include <vector>

#include "compressed_source_authority.hpp"   // CompressionMetadataKind
#include "gpu_execute.hpp"                   // MetadataKindRequest

namespace prosper::gpu {

// A retained depth/stencil surface, as the renderer holds it: the aspect bases it renders through and
// the HTILE plane that describes them.
struct RetainedDepthCorrelation {
    uint64_t depth_read = 0;
    uint64_t depth_write = 0;
    uint64_t stencil_read = 0;
    uint64_t stencil_write = 0;
    uint64_t htile = 0;
};

// A retained colour surface and the DCC control plane registered for it.
struct RetainedColorCorrelation {
    uint64_t resource_addr = 0;
    uint64_t dcc_metadata_addr = 0;
};

// `Htile` only when a retained depth surface names this metadata address as its HTILE base AND one of
// its aspect bases is this resource. `Dcc` only when a retained colour surface at this resource address
// registered this metadata address. Otherwise `Unknown`.
CompressionMetadataKind correlate_compression_metadata_kind(
    const MetadataKindRequest& request,
    const std::vector<RetainedDepthCorrelation>& depth,
    const std::vector<RetainedColorCorrelation>& color);

}  // namespace prosper::gpu
