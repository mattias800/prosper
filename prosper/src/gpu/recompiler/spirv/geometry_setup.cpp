// Interpolation geometry emission moved from the inline builder.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace prosper::gpu {
std::vector<uint32_t> SpirvCompute::build_interpolation_geometry(
    const FragmentInterpolationLayout& layout, bool capture_geometry_position, bool synthesize_rect,
    bool publish_primitive_id, uint32_t layer_input_location) {
    if ((!layout.requires_geometry && !synthesize_rect) || !layout.valid) return {};
        // The collector's primitive key covers a one-input/one-output triangle, not RectList's
        // two generated children. Native geometry emission is unchanged when this is false.
    if (publish_primitive_id && synthesize_rect) return {};
    // The merged-NGG raster commit (#3135 P3) delivers a primitive's layer as a uint input at
    // `layer_input_location`, equal on all three vertices; this stage writes it to gl_Layer.
    // RectList synthesis never carries one, and the location must not shadow an attribute.
    const bool forward_layer = layer_input_location != FragmentInterpolationLayout::kUnusedLocation;
    if (forward_layer && (synthesize_rect || layer_input_location >= 32u ||
                          (layout.attribute_mask & (1u << layer_input_location))))
        return {};

    t_void = id();
    t_fn = id();
    t_f32 = id();
    t_u32 = id();
    t_i32 = id();
    t_bool = id();
    t_v4f = id();
    // This module declares SPIR-V 1.3 (see finish()), where an OpSelect with a VECTOR result needs
    // a vector condition of the same width -- a scalar condition is only legal from 1.4. The rect
    // path is the only place here selecting between vec4s, so the bvec4 is allocated and declared
    // only for it, and the non-synthesis module gains no instruction and no type from it.
    // (That module is NOT byte-identical to before this change: loading the positions earlier
    // renumbers the ids after it, and the `consumes_parameters` guard below elides a composite
    // that nothing consumed. Both are semantically inert; ids are module-internal.)
    const uint32_t t_v4bool = synthesize_rect ? id() : 0;
    const uint32_t t_input_per_vertex = id(), t_output_per_vertex = id();
    const uint32_t c_three = id();
    const uint32_t t_input_positions = id(), t_input_varyings = id();
    const uint32_t ptr_in_positions = id(), ptr_in_varyings = id();
    const uint32_t ptr_in_v4f = id(), ptr_out_position = id(), ptr_out_v4f = id();
    const uint32_t input_position = id(), output_position = id();
    f_main = id();
    const uint32_t label = id();
    glsl = id();
    const uint32_t primitive_in = publish_primitive_id ? id() : 0;
    const uint32_t primitive_out = publish_primitive_id ? id() : 0;
    const uint32_t primitive_in_ptr = publish_primitive_id ? id() : 0;
    const uint32_t primitive_out_ptr = publish_primitive_id ? id() : 0;
    const uint32_t layer_in = forward_layer ? id() : 0, layer_out = forward_layer ? id() : 0;
    const uint32_t t_layer_inputs = forward_layer ? id() : 0;
    const uint32_t layer_in_ptr = forward_layer ? id() : 0;
    const uint32_t layer_in_element_ptr = forward_layer ? id() : 0;
    const uint32_t layer_out_ptr = forward_layer ? id() : 0;

    std::array<uint32_t, 32> attribute_inputs{}, attribute_outputs{};
    std::array<std::array<uint32_t, 3>, 32> parameter_outputs{};
    std::array<uint32_t, 7> system_outputs{};
    for (uint32_t attr = 0; attr < 32; ++attr) {
        if (!(layout.attribute_mask & (1u << attr))) continue;
        attribute_inputs[attr] = id();
        if ((layout.smooth_mask & (1u << attr)) || synthesize_rect) attribute_outputs[attr] = id();
        for (uint32_t selector = 0; selector < 3; ++selector)
            if (layout.parameter_locations[attr][selector] !=
                FragmentInterpolationLayout::kUnusedLocation)
                parameter_outputs[attr][selector] = id();
    }
    for (uint32_t field = 0; field < 7; ++field)
        if (layout.system_locations[field] != FragmentInterpolationLayout::kUnusedLocation)
            system_outputs[field] = id();

    return geometry_capabilities(
        capture_geometry_position, synthesize_rect, t_input_per_vertex, t_output_per_vertex,
        attribute_inputs, attribute_outputs, parameter_outputs, layout, system_outputs,
        input_position, output_position, publish_primitive_id, primitive_in, primitive_out,
        forward_layer, layer_in, layer_input_location, layer_out, primitive_in_ptr,
        primitive_out_ptr, t_v4bool, t_layer_inputs, layer_in_ptr, layer_in_element_ptr,
        layer_out_ptr, c_three, t_input_positions, t_input_varyings, ptr_in_positions,
        ptr_in_varyings, ptr_in_v4f, ptr_out_position, ptr_out_v4f, label);
}

std::vector<uint32_t> SpirvCompute::geometry_capabilities(
    bool& capture_geometry_position, bool& synthesize_rect, const uint32_t& t_input_per_vertex,
    const uint32_t& t_output_per_vertex, std::array<uint32_t, 32>& attribute_inputs,
    std::array<uint32_t, 32>& attribute_outputs,
    std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
    const FragmentInterpolationLayout& layout, std::array<uint32_t, 7>& system_outputs,
    const uint32_t& input_position, const uint32_t& output_position, bool& publish_primitive_id,
    const uint32_t& primitive_in, const uint32_t& primitive_out, const bool& forward_layer,
    const uint32_t& layer_in, uint32_t& layer_input_location, const uint32_t& layer_out,
    const uint32_t& primitive_in_ptr, const uint32_t& primitive_out_ptr, const uint32_t& t_v4bool,
    const uint32_t& t_layer_inputs, const uint32_t& layer_in_ptr,
    const uint32_t& layer_in_element_ptr, const uint32_t& layer_out_ptr, const uint32_t& c_three,
    const uint32_t& t_input_positions, const uint32_t& t_input_varyings,
    const uint32_t& ptr_in_positions, const uint32_t& ptr_in_varyings, const uint32_t& ptr_in_v4f,
    const uint32_t& ptr_out_position, const uint32_t& ptr_out_v4f, const uint32_t& label) {
    put(caps, Op_Capability, {Cap_Shader});
    declare_float_controls(f_main);
    put(caps, Op_Capability, {Cap_Geometry});
    if (capture_geometry_position) put(caps, Op_Capability, {Cap_TransformFeedback});
    {
        std::vector<uint32_t> operands{glsl};
        pstr(operands, "GLSL.std.450");
        putv(extimp, Op_ExtInstImport, operands);
    }
    put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
    exec_model = Exec_Geometry;
    // Vulkan pipeline rule stage-00715 requires an explicit positive invocation count.
    // One preserves SPIR-V's default: run once per input triangle, for every generated form.
    put(exec, Op_ExecutionMode, {f_main, EM_Invocations, 1});
    put(exec, Op_ExecutionMode, {f_main, EM_Triangles});
    put(exec, Op_ExecutionMode, {f_main, EM_OutputTriangleStrip});
    put(exec, Op_ExecutionMode, {f_main, EM_OutputVertices, synthesize_rect ? 4u : 3u});
    if (capture_geometry_position) put(exec, Op_ExecutionMode, {f_main, EM_Xfb});

    put(deco, Op_MemberDecorate, {t_input_per_vertex, 0, Dec_BuiltIn, BI_Position});
    put(deco, Op_Decorate, {t_input_per_vertex, Dec_Block});
    put(deco, Op_MemberDecorate, {t_output_per_vertex, 0, Dec_BuiltIn, BI_Position});
    put(deco, Op_Decorate, {t_output_per_vertex, Dec_Block});
    if (capture_geometry_position) {
        put(deco, Op_MemberDecorate, {t_output_per_vertex, 0, Dec_XfbBuffer, 0});
        put(deco, Op_MemberDecorate, {t_output_per_vertex, 0, Dec_XfbStride, 16});
        put(deco, Op_MemberDecorate, {t_output_per_vertex, 0, Dec_Offset, 0});
    }
    return geometry_interfaces(
        attribute_inputs, attribute_outputs, parameter_outputs, layout, system_outputs,
        input_position, output_position, publish_primitive_id, primitive_in, primitive_out,
        forward_layer, layer_in, layer_input_location, layer_out, primitive_in_ptr,
        primitive_out_ptr, synthesize_rect, t_v4bool, t_layer_inputs, layer_in_ptr,
        layer_in_element_ptr, layer_out_ptr, t_input_per_vertex, t_output_per_vertex, c_three,
        t_input_positions, t_input_varyings, ptr_in_positions, ptr_in_varyings, ptr_in_v4f,
        ptr_out_position, ptr_out_v4f, label);
}

std::vector<uint32_t> SpirvCompute::geometry_interfaces(
    std::array<uint32_t, 32>& attribute_inputs, std::array<uint32_t, 32>& attribute_outputs,
    std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
    const FragmentInterpolationLayout& layout, std::array<uint32_t, 7>& system_outputs,
    const uint32_t& input_position, const uint32_t& output_position, bool& publish_primitive_id,
    const uint32_t& primitive_in, const uint32_t& primitive_out, const bool& forward_layer,
    const uint32_t& layer_in, uint32_t& layer_input_location, const uint32_t& layer_out,
    const uint32_t& primitive_in_ptr, const uint32_t& primitive_out_ptr, bool& synthesize_rect,
    const uint32_t& t_v4bool, const uint32_t& t_layer_inputs, const uint32_t& layer_in_ptr,
    const uint32_t& layer_in_element_ptr, const uint32_t& layer_out_ptr,
    const uint32_t& t_input_per_vertex, const uint32_t& t_output_per_vertex,
    const uint32_t& c_three, const uint32_t& t_input_positions, const uint32_t& t_input_varyings,
    const uint32_t& ptr_in_positions, const uint32_t& ptr_in_varyings, const uint32_t& ptr_in_v4f,
    const uint32_t& ptr_out_position, const uint32_t& ptr_out_v4f, const uint32_t& label) {
    for (uint32_t attr = 0; attr < 32; ++attr) {
        if (attribute_inputs[attr]) {
            put(deco, Op_Decorate, {attribute_inputs[attr], Dec_Location, attr});
            iface.push_back(attribute_inputs[attr]);
        }
        if (attribute_outputs[attr]) {
            put(deco, Op_Decorate, {attribute_outputs[attr], Dec_Location, attr});
            iface.push_back(attribute_outputs[attr]);
        }
        for (uint32_t selector = 0; selector < 3; ++selector) {
            const uint32_t variable = parameter_outputs[attr][selector];
            if (!variable) continue;
            put(deco, Op_Decorate,
                {variable, Dec_Location, layout.parameter_locations[attr][selector]});
            put(deco, Op_Decorate, {variable, Dec_Flat});
            iface.push_back(variable);
        }
    }
    for (uint32_t field = 0; field < 7; ++field) {
        const uint32_t variable = system_outputs[field];
        if (!variable) continue;
        put(deco, Op_Decorate, {variable, Dec_Location, layout.system_locations[field]});
        if (field == 3 || field >= 4) put(deco, Op_Decorate, {variable, Dec_NoPerspective});
        if (field == 0 || field == 4) put(deco, Op_Decorate, {variable, Dec_Sample});
        if (field == 2 || field == 6) put(deco, Op_Decorate, {variable, Dec_Centroid});
        iface.push_back(variable);
    }
    iface.push_back(input_position);
    iface.push_back(output_position);
    if (publish_primitive_id) {
        put(deco, Op_Decorate, {primitive_in, Dec_BuiltIn, 7});
        put(deco, Op_Decorate, {primitive_out, Dec_BuiltIn, 7});
        iface.push_back(primitive_in);
        iface.push_back(primitive_out);
    }
    if (forward_layer) {
        put(deco, Op_Decorate, {layer_in, Dec_Location, layer_input_location});
        put(deco, Op_Decorate, {layer_out, Dec_BuiltIn, BI_Layer});
        iface.push_back(layer_in);
        iface.push_back(layer_out);
    }

    return geometry_types(
        publish_primitive_id, primitive_in_ptr, primitive_out_ptr, primitive_in, primitive_out,
        synthesize_rect, t_v4bool, forward_layer, t_layer_inputs, layer_in_ptr,
        layer_in_element_ptr, layer_out_ptr, layer_in, layer_out, t_input_per_vertex,
        t_output_per_vertex, c_three, t_input_positions, t_input_varyings, ptr_in_positions,
        ptr_in_varyings, ptr_in_v4f, ptr_out_position, ptr_out_v4f, input_position, output_position,
        attribute_inputs, attribute_outputs, parameter_outputs, system_outputs, label, layout);
}

std::vector<uint32_t> SpirvCompute::geometry_types(
    bool& publish_primitive_id, const uint32_t& primitive_in_ptr, const uint32_t& primitive_out_ptr,
    const uint32_t& primitive_in, const uint32_t& primitive_out, bool& synthesize_rect,
    const uint32_t& t_v4bool, const bool& forward_layer, const uint32_t& t_layer_inputs,
    const uint32_t& layer_in_ptr, const uint32_t& layer_in_element_ptr,
    const uint32_t& layer_out_ptr, const uint32_t& layer_in, const uint32_t& layer_out,
    const uint32_t& t_input_per_vertex, const uint32_t& t_output_per_vertex,
    const uint32_t& c_three, const uint32_t& t_input_positions, const uint32_t& t_input_varyings,
    const uint32_t& ptr_in_positions, const uint32_t& ptr_in_varyings, const uint32_t& ptr_in_v4f,
    const uint32_t& ptr_out_position, const uint32_t& ptr_out_v4f, const uint32_t& input_position,
    const uint32_t& output_position, std::array<uint32_t, 32>& attribute_inputs,
    std::array<uint32_t, 32>& attribute_outputs,
    std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
    std::array<uint32_t, 7>& system_outputs, const uint32_t& label,
    const FragmentInterpolationLayout& layout) {
    put(types, Op_TypeVoid, {t_void});
    put(types, Op_TypeFunction, {t_fn, t_void});
    put(types, Op_TypeFloat, {t_f32, 32});
    put(types, Op_TypeInt, {t_u32, 32, 0});
    put(types, Op_TypeInt, {t_i32, 32, 1});
    put(types, Op_TypeBool, {t_bool});
    if (publish_primitive_id) {
        put(types, Op_TypePointer, {primitive_in_ptr, SC_Input, t_i32});
        put(types, Op_TypePointer, {primitive_out_ptr, SC_Output, t_i32});
        put(types, Op_Variable, {primitive_in_ptr, primitive_in, SC_Input});
        put(types, Op_Variable, {primitive_out_ptr, primitive_out, SC_Output});
    }
    put(types, Op_TypeVector, {t_v4f, t_f32, 4});
    if (synthesize_rect) put(types, Op_TypeVector, {t_v4bool, t_bool, 4});
    if (forward_layer) {
        put(types, Op_TypeArray, {t_layer_inputs, t_u32, uconst(3)});
        put(types, Op_TypePointer, {layer_in_ptr, SC_Input, t_layer_inputs});
        put(types, Op_TypePointer, {layer_in_element_ptr, SC_Input, t_u32});
        put(types, Op_TypePointer, {layer_out_ptr, SC_Output, t_i32});
        put(types, Op_Variable, {layer_in_ptr, layer_in, SC_Input});
        put(types, Op_Variable, {layer_out_ptr, layer_out, SC_Output});
    }
    put(types, Op_TypeStruct, {t_input_per_vertex, t_v4f});
    put(types, Op_TypeStruct, {t_output_per_vertex, t_v4f});
    put(types, Op_Constant, {t_u32, c_three, 3});
    put(types, Op_TypeArray, {t_input_positions, t_input_per_vertex, c_three});
    put(types, Op_TypeArray, {t_input_varyings, t_v4f, c_three});
    put(types, Op_TypePointer, {ptr_in_positions, SC_Input, t_input_positions});
    put(types, Op_TypePointer, {ptr_in_varyings, SC_Input, t_input_varyings});
    put(types, Op_TypePointer, {ptr_in_v4f, SC_Input, t_v4f});
    put(types, Op_TypePointer, {ptr_out_position, SC_Output, t_output_per_vertex});
    put(types, Op_TypePointer, {ptr_out_v4f, SC_Output, t_v4f});
    put(types, Op_Variable, {ptr_in_positions, input_position, SC_Input});
    put(types, Op_Variable, {ptr_out_position, output_position, SC_Output});
    for (uint32_t variable : attribute_inputs)
        if (variable) put(types, Op_Variable, {ptr_in_varyings, variable, SC_Input});
    for (uint32_t variable : attribute_outputs)
        if (variable) put(types, Op_Variable, {ptr_out_v4f, variable, SC_Output});
    for (const auto& selectors : parameter_outputs)
        for (uint32_t variable : selectors)
            if (variable) put(types, Op_Variable, {ptr_out_v4f, variable, SC_Output});
    for (uint32_t variable : system_outputs)
        if (variable) put(types, Op_Variable, {ptr_out_v4f, variable, SC_Output});

    return geometry_inputs(label, publish_primitive_id, primitive_in, forward_layer,
                           layer_in_element_ptr, layer_in, attribute_inputs, ptr_in_v4f,
                           input_position, synthesize_rect, t_v4bool, layout, primitive_out,
                           layer_out, ptr_out_v4f, output_position, attribute_outputs,
                           parameter_outputs, system_outputs);
}

std::vector<uint32_t> SpirvCompute::geometry_inputs(
    const uint32_t& label, bool& publish_primitive_id, const uint32_t& primitive_in,
    const bool& forward_layer, const uint32_t& layer_in_element_ptr, const uint32_t& layer_in,
    std::array<uint32_t, 32>& attribute_inputs, const uint32_t& ptr_in_v4f,
    const uint32_t& input_position, bool& synthesize_rect, const uint32_t& t_v4bool,
    const FragmentInterpolationLayout& layout, const uint32_t& primitive_out,
    const uint32_t& layer_out, const uint32_t& ptr_out_v4f, const uint32_t& output_position,
    std::array<uint32_t, 32>& attribute_outputs,
    std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
    std::array<uint32_t, 7>& system_outputs) {
    put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
    put(code, Op_Label, {label});
    cur_block = label;
    const uint32_t primitive_value = publish_primitive_id ? id() : 0;
    if (publish_primitive_id) put(code, Op_Load, {t_i32, primitive_value, primitive_in});
    uint32_t layer_value = 0;
    if (forward_layer) {
        const uint32_t pointer = id(), bits = id();
        layer_value = id();
        put(code, Op_AccessChain, {layer_in_element_ptr, pointer, layer_in, uconst(0)});
        put(code, Op_Load, {t_u32, bits, pointer});
        put(code, Op_Bitcast, {t_i32, layer_value, bits});
    }

    std::array<std::array<uint32_t, 3>, 32> attribute_values{};
    for (uint32_t attr = 0; attr < 32; ++attr) {
        if (!attribute_inputs[attr]) continue;
        for (uint32_t vertex = 0; vertex < 3; ++vertex) {
            const uint32_t pointer = id();
            put(code, Op_AccessChain,
                {ptr_in_v4f, pointer, attribute_inputs[attr], uconst(vertex)});
            attribute_values[attr][vertex] = id();
            put(code, Op_Load, {t_v4f, attribute_values[attr][vertex], pointer});
        }
    }
    std::array<uint32_t, 3> positions{};
    for (uint32_t vertex = 0; vertex < 3; ++vertex) {
        const uint32_t input_pointer = id();
        put(code, Op_AccessChain,
            {ptr_in_v4f, input_pointer, input_position, uconst(vertex), uconst(0)});
        positions[vertex] = id();
        put(code, Op_Load, {t_v4f, positions[vertex], input_pointer});
    }

    // `rect_shared_is_v0` / `rect_shared_is_v1` name the supplied vertex adjacent to both others
    // (neither true => it is v2). Compared as squared separations, so no square root is needed.
    return geometry_rect_selectors(
        synthesize_rect, positions, t_v4bool, attribute_inputs, attribute_values, layout,
        publish_primitive_id, primitive_out, primitive_value, forward_layer, layer_out, layer_value,
        ptr_out_v4f, output_position, attribute_outputs, parameter_outputs, system_outputs);
}
}   // namespace prosper::gpu
