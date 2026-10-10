// ngg_raster_commit.cpp -- see ngg_raster_commit.hpp.
#include "gpu/recompiler/ngg_raster_commit.hpp"

#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "gpu/recompiler/param_ps_routing.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace prosper::gpu {
namespace {

static_assert(kNggRasterDescriptorSet == kNggShellDescriptorSet &&
                  kNggRasterExportBinding == kNggShellExportBinding,
              "the raster commit reads the export buffer where the shell writes it");

// The shell's workgroup bound: four Wave64 or eight Wave32 guest waves.
constexpr uint32_t kMaxLanes = 256;
// The largest layer count a target can have: gfx10's CB_COLOR*_VIEW.SLICE_MAX is 11 bits, and 2048 is
// also RADV's maxFramebufferLayers (Vulkan only guarantees 256, so a device may allow fewer).
constexpr uint32_t kMaxLayerSlices = 2048;
constexpr uint32_t kIndexBits = 0x1ffu;   // one 9-bit PRIM vertex index
constexpr uint32_t kPrimNullBit = 31;

// SPIR-V enumerants the shared builder does not name.
constexpr uint32_t kCapShaderViewportIndexLayerExt = 5254;
constexpr uint32_t kExecutionModeInputLines = 20;
constexpr uint32_t kExecutionModeOutputLineStrip = 28;
constexpr uint32_t kDecorationNonWritable = 24;

// The builder's cbuf keys for the two buffers. They are map keys, not bindings: the builder routes
// keys 2 and 3 to its own constant-buffer variables, so the bindings themselves cannot be keys.
constexpr uint32_t kExportKey = 0x1000u | kNggRasterExportBinding;
constexpr uint32_t kCounterKey = 0x1000u | kNggRasterCounterBinding;

std::vector<uint32_t> fail(std::string* refusal, const char* reason,
                           const std::string& detail = {}) {
    if (refusal) {
        *refusal = std::string("reason=") + reason;
        if (!detail.empty()) *refusal += " " + detail;
    }
    return {};
}

uint32_t vertices_per_primitive(NggOutputTopology topology) {
    switch (topology) {
        case NggOutputTopology::TriangleList: return 3;
        case NggOutputTopology::LineList: return 2;
        default: return 0;
    }
}

// The fragment-input locations the program's PARAM exports reach.
uint32_t routed_output_mask(const NggRasterCommitConfig& config) {
    uint32_t mask = 0;
    for (uint32_t target : config.layout.param_targets)
        for_each_param_ps_location(target - kExpTargetParam0, config.pixel_inputs,
                                   [&](uint32_t location) { mask |= 1u << location; });
    return mask;
}

// The uint storage buffers share one runtime-array block type; each is registered under its cbuf
// key so cbuf_load / cbuf_atomic_rtn address it. Entries are (binding, key, writable).
void declare_storage(SpirvCompute& b,
                     const std::vector<std::tuple<uint32_t, uint32_t, bool>>& bindings) {
    const uint32_t array = b.id(), block = b.id(), pointer = b.id();
    b.t_ptr_sb_u32 = b.id();
    b.put(b.deco, Op_Decorate, {array, Dec_ArrayStride, 4u});
    b.put(b.deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
    b.put(b.deco, Op_Decorate, {block, Dec_Block});
    b.put(b.types, Op_TypeRuntimeArray, {array, b.t_u32});
    b.put(b.types, Op_TypeStruct, {block, array});
    b.put(b.types, Op_TypePointer, {pointer, SC_StorageBuffer, block});
    b.put(b.types, Op_TypePointer, {b.t_ptr_sb_u32, SC_StorageBuffer, b.t_u32});
    for (const auto& [binding, key, writable] : bindings) {
        const uint32_t storage = b.id();
        b.put(b.deco, Op_Decorate, {storage, Dec_DescriptorSet, kNggRasterDescriptorSet});
        b.put(b.deco, Op_Decorate, {storage, Dec_Binding, binding});
        if (!writable) b.put(b.deco, Op_Decorate, {storage, kDecorationNonWritable});
        b.declare_external_storage_buffer(pointer, storage);
        b.cbuf_var.emplace(key, storage);
    }
}

uint32_t select_of(SpirvCompute& b, uint32_t index, const std::array<uint32_t, 3>& values,
                   uint32_t count) {
    uint32_t chosen = values[count - 1];
    for (uint32_t k = count - 1; k-- > 0;)
        chosen = b.sel(b.ucmp(Op_IEqual, index, b.uconst(k)), values[k], chosen);
    return chosen;
}

}   // namespace

NggOutputTopology ngg_output_topology(uint32_t vgt_gs_out_prim_type) {
    switch (vgt_gs_out_prim_type & 0x3fu) {
        case 1: return NggOutputTopology::LineList;
        case 2: return NggOutputTopology::TriangleList;
        default: return NggOutputTopology::Unsupported;   // points, rect list, reserved
    }
}

NggLayerRoute select_ngg_layer_route(const NggLayerRouteQuery& query, std::string* refusal) {
    if (refusal) refusal->clear();
    if (query.interpolation_geometry_required &&
        query.topology != NggOutputTopology::TriangleList) {
        if (refusal) *refusal = "reason=ngg-interpolation-geometry-needs-triangles";
        return NggLayerRoute::None;
    }
    if (!query.layer_from_pos1) return NggLayerRoute::None;
    if (query.interpolation_geometry_required && query.geometry_shader)
        return NggLayerRoute::InterpolationGeometry;
    if (query.shader_output_layer) return NggLayerRoute::ShaderOutputLayer;
    if (query.geometry_shader) return NggLayerRoute::ForwardingGeometry;
    if (refusal)
        *refusal = "reason=ngg-layer-route-unavailable cause=no-shader-output-layer-no-geometry";
    return NggLayerRoute::None;
}

uint32_t ngg_raster_commit_vertex_count(const NggRasterCommitConfig& config, uint32_t blocks) {
    return blocks * config.wave_lanes * config.waves * vertices_per_primitive(config.topology);
}

std::vector<uint32_t> build_ngg_raster_commit_vertex(const NggRasterCommitConfig& config,
                                                     NggRasterCommitInterface* out_interface,
                                                     std::string* refusal) {
    const uint32_t corners = vertices_per_primitive(config.topology);
    if (!corners) return fail(refusal, "ngg-raster-topology-unsupported");
    const NggExportRecordLayout& layout = config.layout;
    if (!config.float_transport.canonical() ||
        (config.wave_lanes != 32u && config.wave_lanes != 64u) || config.waves < 1 ||
        config.waves * config.wave_lanes > kMaxLanes || !layout.words_per_lane ||
        layout.param_targets.size() > kNggMaxParamTargets ||
        layout.param_targets.size() != layout.param_channels.size() || !config.layer_slices ||
        config.layer_slices > kMaxLayerSlices)
        return fail(refusal, "ngg-raster-config");
    for (uint32_t target : layout.param_targets)
        if (target < kExpTargetParam0 || target >= kExpTargetParam0 + 32u ||
            layout.first_param_word == kNggRecordAbsent)
            return fail(refusal, "ngg-raster-config", "cause=param-target");
    // Every word the stage reads must lie inside one record: POS0, POS1 when present, each PARAM.
    // 64-bit arithmetic, so an absurd offset cannot wrap into range.
    const auto fits = [&](uint32_t first_word) {
        return uint64_t{first_word} + 4u <= layout.words_per_lane;
    };
    bool layout_fits = fits(kNggRecordPos0Word) &&
                       (layout.pos1_word == kNggRecordAbsent || fits(layout.pos1_word));
    for (uint32_t k = 0; k < layout.param_targets.size(); ++k)
        layout_fits =
            layout_fits && fits(layout.first_param_word) &&
            uint64_t{layout.first_param_word} + uint64_t{4} * (k + 1u) <= layout.words_per_lane;
    if (!layout_fits) return fail(refusal, "ngg-raster-config", "cause=layout-out-of-record");
    const bool read_layer = config.layer_from_pos1;
    if (read_layer && (layout.pos1_word == kNggRecordAbsent || !(layout.pos1_channels & 4u)))
        return fail(refusal, "ngg-layer-without-pos1");
    // A one-slice target needs no route: the layer is read only to cull (and count) a primitive
    // that names another slice, and nothing writes gl_Layer, which a one-layer framebuffer leaves
    // undefined (Undefined-Value-Layer-Written). More slices need a route (#3135 P7).
    const bool selects_layer = config.layer_select;
    if (selects_layer && (!read_layer || config.route != NggLayerRoute::None))
        return fail(refusal, "ngg-raster-config", "cause=layer-select");
    if (read_layer && config.route == NggLayerRoute::None && config.layer_slices != 1u &&
        !selects_layer)
        return fail(refusal, "ngg-layer-route-unavailable", "cause=route-none");
    if (config.route == NggLayerRoute::InterpolationGeometry && corners != 3)
        return fail(refusal, "ngg-interpolation-geometry-needs-triangles");

    NggRasterCommitInterface published;
    published.vertices_per_primitive = corners;
    published.output_mask = routed_output_mask(config);
    const bool geometry_route =
        read_layer && (config.route == NggLayerRoute::ForwardingGeometry ||
                       config.route == NggLayerRoute::InterpolationGeometry);
    if (geometry_route) {
        const uint32_t taken = published.output_mask | config.reserved_locations;
        if (taken == UINT32_MAX) return fail(refusal, "ngg-raster-layer-location-exhausted");
        published.layer_location = static_cast<uint32_t>(std::countr_one(taken));
    }

    SpirvCompute b;
    b.float_transport = config.float_transport;
    b.begin_vertex();
    std::vector<std::tuple<uint32_t, uint32_t, bool>> bindings = {
        {kNggRasterExportBinding, kExportKey, false}};
    if (config.count_violations) bindings.emplace_back(kNggRasterCounterBinding, kCounterKey, true);
    declare_storage(b, bindings);
    const auto load = [&](uint32_t index) { return b.cbuf_load(index, kExportKey); };

    // Layer output: BuiltIn Layer on the vertex stage, or a uint varying for a geometry stage.
    uint32_t layer_output = 0;
    if (read_layer && config.route != NggLayerRoute::None) {
        const uint32_t pointer = b.id();
        layer_output = b.id();
        if (config.route == NggLayerRoute::ShaderOutputLayer) {
            b.put(b.caps, Op_Capability, {kCapShaderViewportIndexLayerExt});
            std::vector<uint32_t> name;
            b.pstr(name, "SPV_EXT_shader_viewport_index_layer");
            b.putv(b.exts, Op_Extension, name);
            b.put(b.types, Op_TypePointer, {pointer, SC_Output, b.t_i32});
            b.put(b.deco, Op_Decorate, {layer_output, Dec_BuiltIn, BI_Layer});
        } else {
            b.put(b.types, Op_TypePointer, {pointer, SC_Output, b.t_u32});
            b.put(b.deco, Op_Decorate, {layer_output, Dec_Location, published.layer_location});
        }
        b.put(b.types, Op_Variable, {pointer, layer_output, SC_Output});
        b.iface.push_back(layer_output);
    }

    const uint32_t lanes = config.wave_lanes * config.waves;
    const uint32_t vertex = b.load_vertex_index();
    const uint32_t slot = b.ibin(Op_UDiv, vertex, b.uconst(corners));
    const uint32_t corner = b.ibin(Op_UMod, vertex, b.uconst(corners));
    const uint32_t block = b.ibin(Op_UDiv, slot, b.uconst(lanes));
    const uint32_t thread = b.ibin(Op_UMod, slot, b.uconst(lanes));
    const uint32_t header =
        b.ibin(Op_IMul, block, b.uconst(layout.block_words(config.waves, config.wave_lanes)));
    const auto header_word = [&](uint32_t word) {
        return load(b.ibin(Op_IAdd, header, b.uconst(word)));
    };
    // The first word of thread `t`'s record; `t` is clamped into the block.
    const auto record = [&](uint32_t t) {
        const uint32_t safe =
            b.sel(b.ucmp(Op_ULessThan, t, b.uconst(lanes)), t, b.uconst(lanes - 1));
        return b.ibin(Op_IAdd, header,
                      b.ibin(Op_IAdd, b.uconst(kNggSubgroupHeaderWords),
                             b.ibin(Op_IMul, safe, b.uconst(layout.words_per_lane))));
    };
    const auto word_at = [&](uint32_t base, uint32_t word) {
        return load(b.ibin(Op_IAdd, base, b.uconst(word)));
    };

    const uint32_t verts_alloc = header_word(kNggHeaderVertsAlloc);
    const uint32_t prims_alloc = header_word(kNggHeaderPrimsAlloc);
    // An allocation larger than the block's records would lose the primitives that have none, so
    // it invalidates the block (counted) rather than dropping them silently.
    const uint32_t fits_block = b.land(b.ucmp(Op_ULessThanEqual, verts_alloc, b.uconst(lanes)),
                                       b.ucmp(Op_ULessThanEqual, prims_alloc, b.uconst(lanes)));
    const uint32_t block_ok = b.land(
        fits_block,
        b.land(b.ucmp(Op_IEqual, header_word(kNggHeaderAllocRequests), b.uconst(1)),
               b.land(b.ucmp(Op_IEqual, header_word(kNggHeaderStrayRequests), b.uconst(0)),
                      b.ucmp(Op_IEqual, header_word(kNggHeaderLaunchMismatches), b.uconst(0)))));
    const uint32_t in_range = b.ucmp(Op_ULessThan, thread, prims_alloc);
    const uint32_t prim_record = record(thread);
    const uint32_t prim_flags = word_at(prim_record, kNggRecordFlagsWord);
    const uint32_t prim_word = word_at(prim_record, kNggRecordPrimWord);
    const uint32_t prim_written = b.ucmp(
        Op_INotEqual, b.ibin(Op_BitwiseAnd, prim_flags, b.uconst(kNggFlagPrim)), b.uconst(0));
    const uint32_t not_null = b.ucmp(
        Op_IEqual, b.ibin(Op_ShiftRightLogical, prim_word, b.uconst(kPrimNullBit)), b.uconst(0));
    const uint32_t needed = kNggFlagPos0 | (read_layer ? kNggFlagPos1 : 0u);
    std::array<uint32_t, 3> index{}, index_ok{};
    for (uint32_t k = 0; k < corners; ++k) {
        index[k] = b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, prim_word, b.uconst(10u * k)),
                          b.uconst(kIndexBits));
        const uint32_t flags = word_at(record(index[k]), kNggRecordFlagsWord);
        index_ok[k] = b.land(
            b.land(b.ucmp(Op_ULessThan, index[k], verts_alloc),
                   b.ucmp(Op_ULessThan, index[k], b.uconst(lanes))),
            b.ucmp(Op_IEqual, b.ibin(Op_BitwiseAnd, flags, b.uconst(needed)), b.uconst(needed)));
    }
    uint32_t indices_ok = index_ok[0];
    for (uint32_t k = 1; k < corners; ++k) indices_ok = b.land(indices_ok, index_ok[k]);
    const uint32_t connected = b.land(prim_written, indices_ok);
    const uint32_t record_ok =
        config.skip_connectivity_for_test ? b.btrue() : b.land(not_null, connected);
    const uint32_t valid = b.land(block_ok, b.land(in_range, record_ok));

    // Output corner -> guest corner. PROVOKING_VTX_LAST rotates the guest's last vertex to the
    // front, where Vulkan takes flat attributes from.
    const bool rotate = config.provoking_vertex_last && !config.skip_provoking_rotation_for_test;
    const uint32_t guest_corner =
        rotate ? b.ibin(Op_UMod, b.ibin(Op_IAdd, corner, b.uconst(corners - 1)), b.uconst(corners))
               : corner;
    const uint32_t source = record(select_of(b, guest_corner, index, corners));

    uint32_t layer = b.uconst(0), draw = valid, layer_ok = b.btrue();
    // A per-slice replay's layer arrives as the instance index (the run's firstInstance); only the
    // layer-0 replay counts protocol violations, so a culled primitive is counted once.
    const uint32_t selected = selects_layer ? b.load_instance_index() : 0u;
    const uint32_t counting_replay =
        selects_layer ? b.ucmp(Op_IEqual, selected, b.uconst(0)) : b.btrue();
    if (read_layer) {
        const uint32_t provoking =
            config.provoking_vertex_last && !config.layer_from_first_corner_for_test ? corners - 1
                                                                                     : 0u;
        layer = word_at(record(index[provoking]), layout.pos1_word + 2u);
        layer_ok = b.ucmp(Op_ULessThan, layer, b.uconst(config.layer_slices));
        draw = b.land(valid, layer_ok);
        if (selects_layer)   // this pass draws one slice's primitives only
            draw = b.land(draw, b.ucmp(Op_IEqual, layer, selected));
    }

    if (config.count_violations) {
        const uint32_t first_corner =
            b.land(counting_replay, b.ucmp(Op_IEqual, corner, b.uconst(0)));
        const auto count = [&](uint32_t word, uint32_t condition) {
            b.cbuf_atomic_rtn(Op_AtomicIAdd, b.uconst(word), b.uconst(1), kCounterKey, true,
                              b.land(first_corner, condition), b.uconst(0));
        };
        count(kNggViolationInvalidBlocks,
              b.land(b.logical_not(block_ok), b.ucmp(Op_IEqual, thread, b.uconst(0))));
        // A written, non-null primitive whose indices miss, or a slot below prims_alloc that was
        // never written. A written null primitive is the guest culling it, not a violation.
        const uint32_t malformed =
            b.lor(b.logical_not(prim_written), b.land(not_null, b.logical_not(indices_ok)));
        count(kNggViolationConnectivity, b.land(block_ok, b.land(in_range, malformed)));
        if (read_layer) count(kNggViolationLayerCulled, b.land(valid, b.logical_not(layer_ok)));
    }

    // A culled primitive collapses onto one point outside the clip volume (x > w).
    const auto chosen = [&](uint32_t word, uint32_t culled_bits) {
        return b.sel(draw, word_at(source, word), b.uconst(culled_bits));
    };
    const uint32_t outside = std::bit_cast<uint32_t>(2.0f), one = std::bit_cast<uint32_t>(1.0f);
    b.export_position(chosen(kNggRecordPos0Word, outside), chosen(kNggRecordPos0Word + 1, outside),
                      chosen(kNggRecordPos0Word + 2, 0u), chosen(kNggRecordPos0Word + 3, one));
    for (uint32_t ordinal = 0; ordinal < layout.param_targets.size(); ++ordinal) {
        const uint32_t first = layout.param_word(ordinal);
        for_each_param_ps_location(layout.param_targets[ordinal] - kExpTargetParam0,
                                   config.pixel_inputs, [&](uint32_t location) {
                                       b.export_param(
                                           location, chosen(first, 0u), chosen(first + 1u, 0u),
                                           chosen(first + 2u, 0u), chosen(first + 3u, 0u));
                                   });
    }
    if (layer_output) {
        const uint32_t value = b.sel(draw, layer, b.uconst(0));
        if (config.route == NggLayerRoute::ShaderOutputLayer) {
            const uint32_t signed_value = b.id();
            b.put(b.code, Op_Bitcast, {b.t_i32, signed_value, value});
            b.put(b.code, Op_Store, {layer_output, signed_value});
        } else {
            b.put(b.code, Op_Store, {layer_output, value});
        }
    }
    if (out_interface) *out_interface = published;
    return b.finish();
}

std::vector<uint32_t> build_ngg_layer_forward_geometry(const NggRasterCommitInterface& commit,
                                                       FloatTransportConfig float_transport) {
    const uint32_t corners = commit.vertices_per_primitive;
    if ((corners != 2 && corners != 3) || commit.layer_location >= 32u ||
        (commit.output_mask & (1u << commit.layer_location)) || !float_transport.canonical())
        return {};
    SpirvCompute b;
    b.float_transport = float_transport;
    b.t_void = b.id();
    b.t_fn = b.id();
    b.t_f32 = b.id();
    b.t_u32 = b.id();
    b.t_i32 = b.id();
    b.t_bool = b.id();
    b.t_v4f = b.id();
    b.f_main = b.id();
    b.glsl = b.id();
    const uint32_t label = b.id();
    const uint32_t t_in_block = b.id(), t_out_block = b.id(), c_corners = b.id();
    const uint32_t t_in_blocks = b.id(), t_in_v4 = b.id(), t_in_u32 = b.id();
    const uint32_t p_in_blocks = b.id(), p_in_v4_array = b.id(), p_in_u32_array = b.id();
    const uint32_t p_in_v4 = b.id(), p_in_u32 = b.id(), p_out_block = b.id(), p_out_v4 = b.id();
    const uint32_t p_out_i32 = b.id();
    const uint32_t in_position = b.id(), out_position = b.id(), in_layer = b.id();
    const uint32_t out_layer = b.id();
    std::array<uint32_t, 32> inputs{}, outputs{};
    for (uint32_t location = 0; location < 32; ++location)
        if (commit.output_mask & (1u << location)) {
            inputs[location] = b.id();
            outputs[location] = b.id();
        }

    b.put(b.caps, Op_Capability, {Cap_Shader});
    b.put(b.caps, Op_Capability, {Cap_Geometry});
    b.declare_float_controls(b.f_main);
    {
        std::vector<uint32_t> operands{b.glsl};
        b.pstr(operands, "GLSL.std.450");
        b.putv(b.extimp, Op_ExtInstImport, operands);
    }
    b.put(b.mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
    b.exec_model = Exec_Geometry;
    b.put(b.exec, Op_ExecutionMode, {b.f_main, EM_Invocations, 1});
    b.put(b.exec, Op_ExecutionMode,
          {b.f_main, corners == 3 ? uint32_t(EM_Triangles) : kExecutionModeInputLines});
    b.put(b.exec, Op_ExecutionMode,
          {b.f_main,
           corners == 3 ? uint32_t(EM_OutputTriangleStrip) : kExecutionModeOutputLineStrip});
    b.put(b.exec, Op_ExecutionMode, {b.f_main, EM_OutputVertices, corners});

    b.put(b.deco, Op_MemberDecorate, {t_in_block, 0, Dec_BuiltIn, BI_Position});
    b.put(b.deco, Op_Decorate, {t_in_block, Dec_Block});
    b.put(b.deco, Op_MemberDecorate, {t_out_block, 0, Dec_BuiltIn, BI_Position});
    b.put(b.deco, Op_Decorate, {t_out_block, Dec_Block});
    b.put(b.deco, Op_Decorate, {in_layer, Dec_Location, commit.layer_location});
    b.put(b.deco, Op_Decorate, {out_layer, Dec_BuiltIn, BI_Layer});
    b.iface = {in_position, out_position, in_layer, out_layer};
    for (uint32_t location = 0; location < 32; ++location) {
        if (!inputs[location]) continue;
        b.put(b.deco, Op_Decorate, {inputs[location], Dec_Location, location});
        b.put(b.deco, Op_Decorate, {outputs[location], Dec_Location, location});
        b.iface.push_back(inputs[location]);
        b.iface.push_back(outputs[location]);
    }

    b.put(b.types, Op_TypeVoid, {b.t_void});
    b.put(b.types, Op_TypeFunction, {b.t_fn, b.t_void});
    b.put(b.types, Op_TypeFloat, {b.t_f32, 32});
    b.put(b.types, Op_TypeInt, {b.t_u32, 32, 0});
    b.put(b.types, Op_TypeInt, {b.t_i32, 32, 1});
    b.put(b.types, Op_TypeBool, {b.t_bool});
    b.put(b.types, Op_TypeVector, {b.t_v4f, b.t_f32, 4});
    b.put(b.types, Op_TypeStruct, {t_in_block, b.t_v4f});
    b.put(b.types, Op_TypeStruct, {t_out_block, b.t_v4f});
    b.put(b.types, Op_Constant, {b.t_u32, c_corners, corners});
    b.put(b.types, Op_TypeArray, {t_in_blocks, t_in_block, c_corners});
    b.put(b.types, Op_TypeArray, {t_in_v4, b.t_v4f, c_corners});
    b.put(b.types, Op_TypeArray, {t_in_u32, b.t_u32, c_corners});
    b.put(b.types, Op_TypePointer, {p_in_blocks, SC_Input, t_in_blocks});
    b.put(b.types, Op_TypePointer, {p_in_v4_array, SC_Input, t_in_v4});
    b.put(b.types, Op_TypePointer, {p_in_u32_array, SC_Input, t_in_u32});
    b.put(b.types, Op_TypePointer, {p_in_v4, SC_Input, b.t_v4f});
    b.put(b.types, Op_TypePointer, {p_in_u32, SC_Input, b.t_u32});
    b.put(b.types, Op_TypePointer, {p_out_block, SC_Output, t_out_block});
    b.put(b.types, Op_TypePointer, {p_out_v4, SC_Output, b.t_v4f});
    b.put(b.types, Op_TypePointer, {p_out_i32, SC_Output, b.t_i32});
    b.put(b.types, Op_Variable, {p_in_blocks, in_position, SC_Input});
    b.put(b.types, Op_Variable, {p_out_block, out_position, SC_Output});
    b.put(b.types, Op_Variable, {p_in_u32_array, in_layer, SC_Input});
    b.put(b.types, Op_Variable, {p_out_i32, out_layer, SC_Output});
    for (uint32_t location = 0; location < 32; ++location) {
        if (!inputs[location]) continue;
        b.put(b.types, Op_Variable, {p_in_v4_array, inputs[location], SC_Input});
        b.put(b.types, Op_Variable, {p_out_v4, outputs[location], SC_Output});
    }

    b.put(b.code, Op_Function, {b.t_void, b.f_main, FC_None, b.t_fn});
    b.put(b.code, Op_Label, {label});
    b.cur_block = label;
    // Every corner carries the provoking vertex's layer, so vertex 0's is the primitive's.
    const uint32_t layer_pointer = b.id(), layer_bits = b.id(), layer_signed = b.id();
    b.put(b.code, Op_AccessChain, {p_in_u32, layer_pointer, in_layer, b.uconst(0)});
    b.put(b.code, Op_Load, {b.t_u32, layer_bits, layer_pointer});
    b.put(b.code, Op_Bitcast, {b.t_i32, layer_signed, layer_bits});
    for (uint32_t corner = 0; corner < corners; ++corner) {
        const uint32_t position_in = b.id(), position = b.id(), position_out = b.id();
        b.put(b.code, Op_AccessChain,
              {p_in_v4, position_in, in_position, b.uconst(corner), b.uconst(0)});
        b.put(b.code, Op_Load, {b.t_v4f, position, position_in});
        b.put(b.code, Op_AccessChain, {p_out_v4, position_out, out_position, b.uconst(0)});
        b.put(b.code, Op_Store, {position_out, position});
        for (uint32_t location = 0; location < 32; ++location) {
            if (!inputs[location]) continue;
            const uint32_t pointer = b.id(), value = b.id();
            b.put(b.code, Op_AccessChain, {p_in_v4, pointer, inputs[location], b.uconst(corner)});
            b.put(b.code, Op_Load, {b.t_v4f, value, pointer});
            b.put(b.code, Op_Store, {outputs[location], value});
        }
        b.put(b.code, Op_Store, {out_layer, layer_signed});
        b.put(b.code, Op_EmitVertex, {});
    }
    b.put(b.code, Op_EndPrimitive, {});
    return b.finish();
}

}   // namespace prosper::gpu
