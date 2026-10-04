#include "gpu/execute/fragment_raster_launch_collection.hpp"
#include "gpu/execute/fragment_raster_launch.hpp"
#include "gpu/recompiler/fragment_quad_parameters.hpp"
#include "gpu/recompiler/fragment_raster_interface.hpp"
#include <bit>

namespace prosper::gpu {
FragmentRasterLaunchCollection
compile_fragment_raster_launch_collection(const RasterQuadInputs& in,
                                          FragmentPacketDeviceContract device, uint32_t max_quads) {
    const auto refuse = [](const std::string& reason) {
        FragmentRasterLaunchCollection result;
        result.rejection = reason;
        return result;
    };
    if (!in.launch_source || !in.launch_source->matches(in))
        return refuse("fragment-raster-launch-producing-observation-unavailable");
    if (!device.device_identity || !device.raster)
        return refuse("fragment-raster-launch-enabled-device-observation-unavailable");
    const auto& raster = *device.raster;
    if ((!in.has_system_inputs && (in.launch.input_ena || in.launch.input_addr)) ||
        !in.launch.input_ena_available || !in.launch.input_addr_available ||
        in.system_inputs.ena != in.launch.input_ena ||
        in.system_inputs.addr != in.launch.input_addr)
        return refuse("fragment-raster-launch-original-system-routing-unavailable");
    if (!in.source_gs->empty() && !in.generated_interpolation_geometry)
        return refuse("fragment-raster-launch-original-geometry-producer-unimplemented");
    const auto complete =
        fragment_quad_parameter_layout(in.launch_source->original_parameters(), in.system_inputs,
                                       raster.max_fragment_input_components);
    if (!complete.rejection.empty()) return refuse(complete.rejection);
    // The collector's genuine PrimitiveId input itself declares Geometry capability even when
    // no extra GS is needed. The enabled device feature is not inferred from a physical limit.
    if (!raster.geometry_shader_enabled)
        return refuse("fragment-raster-launch-enabled-geometry-unavailable");
    const auto vertex = fragment_raster_output_interface(*in.source_vs, 0);
    if (!vertex.available || (vertex.float4_locations & complete.layout.attribute_mask) !=
                                 complete.layout.attribute_mask)
        return refuse("fragment-raster-launch-coefficient-vertex-interface-unavailable");
    if (vertex.components > raster.max_vertex_output_components ||
        vertex.location_end > raster.max_vertex_output_components)
        return refuse("fragment-raster-launch-enabled-vertex-output-budget");
    FragmentRasterLaunchCollection result;
    result.parameters = complete.layout;
    auto collector_inputs = in;
    collector_inputs.launch_source.reset();   // transformed collection module is not the native PS
    collector_inputs.interpolation = complete.layout;
    if (complete.layout.requires_geometry) {
        result.geometry = recompile_interpolation_geometry(complete.layout, false, false,
                                                           in.float_transport, true);
        if (result.geometry.empty())
            return refuse("fragment-raster-launch-coefficient-geometry-unavailable");
        const uint64_t geometry_inputs = 5u + 4u * std::popcount(complete.layout.attribute_mask);
        const uint64_t geometry_input_end =
            4u * uint64_t(std::bit_width(complete.layout.attribute_mask));
        if (geometry_inputs > raster.max_geometry_input_components ||
            geometry_input_end > raster.max_geometry_input_components)
            return refuse("fragment-raster-launch-enabled-geometry-input-budget");
        const auto geometry = fragment_raster_output_interface(result.geometry, 3);
        if (!geometry.available || geometry.output_vertices != 3)
            return refuse("fragment-raster-launch-coefficient-geometry-interface-unavailable");
        if (!raster.max_geometry_shader_invocations || raster.max_geometry_output_vertices < 3 ||
            geometry.components > raster.max_geometry_output_components ||
            geometry.location_end > raster.max_geometry_output_components ||
            geometry.components > raster.max_geometry_total_output_components / 3u)
            return refuse("fragment-raster-launch-enabled-geometry-output-budget");
        collector_inputs.source_gs = std::make_shared<const std::vector<uint32_t>>(result.geometry);
        collector_inputs.generated_interpolation_geometry = true;
    }
    result.collector = build_raster_quad_collector(collector_inputs, max_quads, result.shape);
    if (result.collector.empty()) return refuse(result.shape.rejection);
    const auto fragment = fragment_raster_input_interface(result.collector, 4);
    if (!fragment.available)
        return refuse("fragment-raster-launch-collector-input-interface-unavailable");
    if (fragment.components > raster.max_fragment_input_components ||
        fragment.location_end > raster.max_fragment_input_components)
        return refuse("fragment-raster-launch-enabled-fragment-input-budget");
    return result;
}
}   // namespace prosper::gpu
