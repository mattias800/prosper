#pragma once
#include "gpu/recompiler/raster_quad_collector.hpp"
#include "gpu/recompiler/fragment_resource_packet.hpp"
#include <string>

namespace prosper::gpu {
struct FragmentRasterLaunchCollection {
    FragmentInterpolationLayout parameters;
    RasterQuadCollector shape;
    std::vector<uint32_t> collector, geometry;
    std::string rejection;
};
// Cold code/profile construction from the realizer's exact draw-source observation. The same
// generated PrimitiveId/coefficient GS must be selected for collection. Attachment replay keeps
// the exact original pre-raster modules: unused collector coefficients do not change geometry.
// Returned code/shape owns no original analysis generation and no dynamic draw entry values.
// This is not an attachment, workitem-valid, helper or virtual-composition admission certificate.
FragmentRasterLaunchCollection
compile_fragment_raster_launch_collection(const RasterQuadInputs&, FragmentPacketDeviceContract,
                                          uint32_t max_quads);
} // namespace prosper::gpu
