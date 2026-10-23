// Interpolation geometry emission moved from the inline builder.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace prosper::gpu {
void SpirvCompute::geometry_store_system_values(uint32_t& bary_i, uint32_t& bary_j,
                                                std::array<uint32_t, 7>& system_outputs,
                                                const uint32_t& position) {
    const uint32_t barycentric = id();
    putv(code, Op_CompositeConstruct,
         {t_v4f, barycentric, bary_i, bary_j, fconstf(1.0f), fconstf(1.0f)});
    uint32_t pull_model = 0;
    if (system_outputs[3]) {
        // AMD LLPC LowerInOut::visitEvalIjOffsetSmoothOp consumes I/W,J/W,1/W.
        // Linear interpolation of these vertex planes preserves their denominator;
        // perspective interpolation would divide it a second time. This transports
        // host raster values, not proof of PS5 initial-lane/precision authority.
        const uint32_t w = id(), reciprocal_w = id(), i_over_w = id(), j_over_w = id();
        put(code, Op_CompositeExtract, {t_f32, w, position, 3});
        put(code, Op_FDiv, {t_f32, reciprocal_w, fconstf(1.0f), w});
        put(code, Op_FMul, {t_f32, i_over_w, bary_i, reciprocal_w});
        put(code, Op_FMul, {t_f32, j_over_w, bary_j, reciprocal_w});
        pull_model = id();
        putv(code, Op_CompositeConstruct,
             {t_v4f, pull_model, i_over_w, j_over_w, reciprocal_w, fconstf(1.0f)});
    }
    for (uint32_t field = 0; field < system_outputs.size(); ++field)
        if (system_outputs[field])
            put(code, Op_Store, {system_outputs[field], field == 3 ? pull_model : barycentric});
}

uint32_t SpirvCompute::geometry_slot_value(uint32_t& slot, const std::array<uint32_t, 3>& supplied,
                                           uint32_t& synthesized, bool& synthesize_rect,
                                           uint32_t& rect_shared_is_v1_wide,
                                           uint32_t& rect_shared_is_v0_wide) {
    if (!synthesize_rect) return supplied[slot];
    if (slot == 3) return synthesized;
        // supplied[(shared + slot) % 3] -- a ROTATION, chosen at runtime.
    const uint32_t if_v0 = supplied[(0 + slot) % 3];
    const uint32_t if_v1 = supplied[(1 + slot) % 3];
    const uint32_t if_v2 = supplied[(2 + slot) % 3];
    const uint32_t not_v0 = id();
    put(code, Op_Select, {t_v4f, not_v0, rect_shared_is_v1_wide, if_v1, if_v2});
    const uint32_t chosen = id();
    put(code, Op_Select, {t_v4f, chosen, rect_shared_is_v0_wide, if_v0, not_v0});
    return chosen;
}

std::vector<uint32_t> SpirvCompute::geometry_rect_selectors(
    bool& synthesize_rect, std::array<uint32_t, 3>& positions, const uint32_t& t_v4bool,
    std::array<uint32_t, 32>& attribute_inputs,
    std::array<std::array<uint32_t, 3>, 32>& attribute_values,
    const FragmentInterpolationLayout& layout, bool& publish_primitive_id,
    const uint32_t& primitive_out, const uint32_t& primitive_value, const bool& forward_layer,
    const uint32_t& layer_out, uint32_t& layer_value, const uint32_t& ptr_out_v4f,
    const uint32_t& output_position, std::array<uint32_t, 32>& attribute_outputs,
    std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
    std::array<uint32_t, 7>& system_outputs) {
    uint32_t rect_shared_is_v0 = 0, rect_shared_is_v1 = 0;
    // The same two selectors widened to bvec4, for the vec4 selects (see the t_v4bool note).
    uint32_t rect_shared_is_v0_wide = 0, rect_shared_is_v1_wide = 0;
    if (synthesize_rect) {
        auto squared_separation = [&](uint32_t a, uint32_t b) {
            const uint32_t delta = id();
            put(code, Op_FSub, {t_v4f, delta, a, b});
            const uint32_t squared = id();
            put(code, Op_Dot, {t_f32, squared, delta, delta});
            return squared;
        };
        const uint32_t d01 = squared_separation(positions[0], positions[1]);
        const uint32_t d02 = squared_separation(positions[0], positions[2]);
        const uint32_t d12 = squared_separation(positions[1], positions[2]);
        const uint32_t d12_ge_d01 = id(), d12_ge_d02 = id();
        put(code, Op_FOrdGreaterThanEqual, {t_bool, d12_ge_d01, d12, d01});
        put(code, Op_FOrdGreaterThanEqual, {t_bool, d12_ge_d02, d12, d02});
        rect_shared_is_v0 = id();
        put(code, Op_LogicalAnd, {t_bool, rect_shared_is_v0, d12_ge_d01, d12_ge_d02});
        // Only consulted when v0 is not the shared corner, so d02 vs d01 settles it.
        rect_shared_is_v1 = id();
        put(code, Op_FOrdGreaterThanEqual, {t_bool, rect_shared_is_v1, d02, d01});
        auto broadcast = [&](uint32_t condition) {
            const uint32_t wide = id();
            putv(code, Op_CompositeConstruct,
                 {t_v4bool, wide, condition, condition, condition, condition});
            return wide;
        };
        rect_shared_is_v0_wide = broadcast(rect_shared_is_v0);
        rect_shared_is_v1_wide = broadcast(rect_shared_is_v1);
    }
    // The fourth value of any per-vertex quantity, under the selected shared corner.
    return geometry_parameters(
        rect_shared_is_v1_wide, rect_shared_is_v0_wide, attribute_inputs, attribute_values, layout,
        synthesize_rect, positions, rect_shared_is_v1, rect_shared_is_v0, publish_primitive_id,
        primitive_out, primitive_value, forward_layer, layer_out, layer_value, ptr_out_v4f,
        output_position, attribute_outputs, parameter_outputs, system_outputs);
}

std::vector<uint32_t> SpirvCompute::geometry_parameters(
    uint32_t& rect_shared_is_v1_wide, uint32_t& rect_shared_is_v0_wide,
    std::array<uint32_t, 32>& attribute_inputs,
    std::array<std::array<uint32_t, 3>, 32>& attribute_values,
    const FragmentInterpolationLayout& layout, bool& synthesize_rect,
    std::array<uint32_t, 3>& positions, uint32_t& rect_shared_is_v1, uint32_t& rect_shared_is_v0,
    bool& publish_primitive_id, const uint32_t& primitive_out, const uint32_t& primitive_value,
    const bool& forward_layer, const uint32_t& layer_out, uint32_t& layer_value,
    const uint32_t& ptr_out_v4f, const uint32_t& output_position,
    std::array<uint32_t, 32>& attribute_outputs,
    std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
    std::array<uint32_t, 7>& system_outputs) {
    auto synthesize_fourth = [&](uint32_t v0, uint32_t v1, uint32_t v2) {
        auto complete = [&](uint32_t diag_a, uint32_t diag_b, uint32_t shared) {
            const uint32_t sum = id();
            put(code, Op_FAdd, {t_v4f, sum, diag_a, diag_b});
            const uint32_t corner = id();
            put(code, Op_FSub, {t_v4f, corner, sum, shared});
            return corner;
        };
        const uint32_t shared_v0 = complete(v1, v2, v0);
        const uint32_t shared_v1 = complete(v0, v2, v1);
        const uint32_t shared_v2 = complete(v0, v1, v2);
        const uint32_t not_v0 = id();
        put(code, Op_Select, {t_v4f, not_v0, rect_shared_is_v1_wide, shared_v1, shared_v2});
        const uint32_t chosen = id();
        put(code, Op_Select, {t_v4f, chosen, rect_shared_is_v0_wide, shared_v0, not_v0});
        return chosen;
    };

    std::array<std::array<uint32_t, 3>, 32> parameters{};
    std::array<uint32_t, 32> rect_attribute_values{};
    for (uint32_t attr = 0; attr < 32; ++attr) {
        if (!attribute_inputs[attr]) continue;
        parameters[attr][2] = attribute_values[attr][0];
        if (layout.passthrough_mask & (1u << attr)) {
            parameters[attr][0] = attribute_values[attr][1];
            parameters[attr][1] = attribute_values[attr][2];
        } else {
            parameters[attr][0] = id();
            put(code, Op_FSub,
                {t_v4f, parameters[attr][0], attribute_values[attr][1], attribute_values[attr][0]});
            parameters[attr][1] = id();
            put(code, Op_FSub,
                {t_v4f, parameters[attr][1], attribute_values[attr][2], attribute_values[attr][0]});
        }
        if (synthesize_rect)
            rect_attribute_values[attr] = synthesize_fourth(
                attribute_values[attr][0], attribute_values[attr][1], attribute_values[attr][2]);
    }

    uint32_t rect_position = 0;
    if (synthesize_rect)
        rect_position = synthesize_fourth(positions[0], positions[1], positions[2]);

    // Value emitted in strip slot `slot`. The supplied triple is ROTATED so the shared corner
    // leads -- slot k carries supplied[(shared + k) % 3] -- then the synthesized corner last.
    // Without rect synthesis this is the identity, so the plain path is unchanged.
    return geometry_vertices(synthesize_rect, rect_shared_is_v1_wide, rect_shared_is_v0_wide,
                             rect_shared_is_v1, rect_shared_is_v0, publish_primitive_id,
                             primitive_out, primitive_value, forward_layer, layer_out, layer_value,
                             positions, rect_position, ptr_out_v4f, output_position,
                             attribute_outputs, attribute_values, rect_attribute_values,
                             parameter_outputs, parameters, system_outputs);
}

std::vector<uint32_t> SpirvCompute::geometry_vertices(
    bool& synthesize_rect, uint32_t& rect_shared_is_v1_wide, uint32_t& rect_shared_is_v0_wide,
    uint32_t& rect_shared_is_v1, uint32_t& rect_shared_is_v0, bool& publish_primitive_id,
    const uint32_t& primitive_out, const uint32_t& primitive_value, const bool& forward_layer,
    const uint32_t& layer_out, uint32_t& layer_value, std::array<uint32_t, 3>& positions,
    uint32_t& rect_position, const uint32_t& ptr_out_v4f, const uint32_t& output_position,
    std::array<uint32_t, 32>& attribute_outputs,
    std::array<std::array<uint32_t, 3>, 32>& attribute_values,
    std::array<uint32_t, 32>& rect_attribute_values,
    std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
    std::array<std::array<uint32_t, 3>, 32>& parameters, std::array<uint32_t, 7>& system_outputs) {
    auto slot_of = [&](uint32_t slot, const std::array<uint32_t, 3>& supplied,
                       uint32_t synthesized) -> uint32_t {
        return geometry_slot_value(slot, supplied, synthesized, synthesize_rect,
                                   rect_shared_is_v1_wide, rect_shared_is_v0_wide);
    };
    // A float chosen by the shared corner: `if_v0` / `if_v1` / `if_v2`.
    auto by_shared_corner = [&](float if_v0, float if_v1, float if_v2) {
        const uint32_t not_v0 = id();
        put(code, Op_Select, {t_f32, not_v0, rect_shared_is_v1, fconstf(if_v1), fconstf(if_v2)});
        const uint32_t chosen = id();
        put(code, Op_Select, {t_f32, chosen, rect_shared_is_v0, fconstf(if_v0), not_v0});
        return chosen;
    };

    const uint32_t output_vertices = synthesize_rect ? 4u : 3u;
    for (uint32_t vertex = 0; vertex < output_vertices; ++vertex) {
        if (publish_primitive_id) put(code, Op_Store, {primitive_out, primitive_value});
        if (forward_layer) put(code, Op_Store, {layer_out, layer_value});
        const uint32_t position = slot_of(vertex, positions, rect_position);
        uint32_t output_pointer = id();
        put(code, Op_AccessChain, {ptr_out_v4f, output_pointer, output_position, uconst(0)});
        put(code, Op_Store, {output_pointer, position});

        for (uint32_t attr = 0; attr < 32; ++attr) {
            if (attribute_outputs[attr])
                put(code, Op_Store,
                    {attribute_outputs[attr],
                     slot_of(vertex, attribute_values[attr], rect_attribute_values[attr])});
            for (uint32_t selector = 0; selector < 3; ++selector)
                if (parameter_outputs[attr][selector])
                    put(code, Op_Store,
                        {parameter_outputs[attr][selector], parameters[attr][selector]});
        }
        // The explicit-parameter basis is P0=A0, P10=A1-A0, P20=A2-A0, so a vertex's (i,j) are
        // its coordinates in it. The synthesized corner's depend on which vertex was shared:
        // shared v0 -> A1+A2-A0 = (1,1); shared v1 -> A0+A2-A1 = (-1,1); shared v2 -> (1,-1).
        const bool consumes_parameters =
            std::any_of(system_outputs.begin(), system_outputs.end(),
                        [](uint32_t variable) { return variable != 0; });
        if (consumes_parameters) {
            // Supplied v0/v1/v2 sit at (0,0), (1,0), (0,1); the synthesized corner is the
            // shared corner's opposite, so its coordinates negate the shared axis.
            uint32_t bary_i, bary_j;
            if (synthesize_rect) {
                // [slot][shared corner]. Slot k carries supplied[(shared + k) % 3], whose
                // coordinates are v0=(0,0), v1=(1,0), v2=(0,1); slot 3 is the synthesized corner.
                static constexpr float kI[4][3] = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}, {1, -1, 1}};
                static constexpr float kJ[4][3] = {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, -1}};
                bary_i = by_shared_corner(kI[vertex][0], kI[vertex][1], kI[vertex][2]);
                bary_j = by_shared_corner(kJ[vertex][0], kJ[vertex][1], kJ[vertex][2]);
            } else {
                bary_i = fconstf(vertex == 1 ? 1.0f : 0.0f);
                bary_j = fconstf(vertex == 2 ? 1.0f : 0.0f);
            }
            geometry_store_system_values(bary_i, bary_j, system_outputs, position);
        }
        put(code, Op_EmitVertex, {});
    }
    put(code, Op_EndPrimitive, {});
    return finish();
}
}   // namespace prosper::gpu
