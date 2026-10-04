#include "gpu/recompiler/raster_quad_collector.hpp"
#include "gpu/recompiler/original_fragment_producer.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <map>

namespace prosper::gpu {
std::vector<uint32_t> build_raster_quad_collector(const RasterQuadInputs& inputs,
        uint32_t max_quads, RasterQuadCollector& out) {
    out = {};
    auto reject = [&](const char* reason) -> std::vector<uint32_t> { out.rejection = reason; return {}; };
    const bool original_bank_source =
        inputs.original_fragment_producer && inputs.original_fragment_producer->matches(inputs);
    if (!inputs.source_fs ||
        (inputs.source_fs->empty() && !inputs.owned_wave_pending && !original_bank_source) ||
        !inputs.raw_code || inputs.raw_code->empty())
        return reject("quad-collector-producing-source-unavailable");
    if (inputs.owned_wave_pending) {
        std::vector<Rdna2Inst> original;
        rdna2_walk(inputs.raw_code->data(), inputs.raw_code->size(), original);
        if (!inputs.source_fs->empty() || rdna2_raw_wave_wide_certificates(original).empty())
            return reject("quad-collector-owned-wave-intent-invalid");
    }
    if (!inputs.raw_matches_producing_source)
        return reject("quad-collector-producing-source-replaced");
    if (!inputs.interpolation.valid) return reject("quad-collector-interpolation-layout-unavailable");
    if (!inputs.float_transport.explicit_nonfinite32())
        return reject("quad-collector-explicit-transport-unavailable");
    if (!max_quads || max_quads > 4096) return reject("quad-collector-budget-invalid");
    if (inputs.interpolation.requires_geometry && !inputs.generated_interpolation_geometry)
        return reject("quad-collector-parameter-producer-unavailable");
    for (uint32_t attr = 0; attr < 32; ++attr) {
        if (!(inputs.interpolation.attribute_mask & (1u << attr))) continue;
        if (!inputs.has_pixel_inputs || !(inputs.pixel_inputs.valid_mask & (1u << attr)))
            return reject("quad-collector-interpolant-routing-unavailable");
        if ((inputs.pixel_inputs.controls[attr] & 0x3fu) == 0x20u &&
            !(inputs.pixel_inputs.effective_passthrough_mask() & (1u << attr)))
            return reject("quad-collector-default-interpolant-unimplemented");
        if (inputs.interpolation.requires_geometry) {
            for (uint32_t selector = 0; selector < 3; ++selector)
                if (inputs.interpolation.parameter_locations[attr][selector] !=
                    FragmentInterpolationLayout::kUnusedLocation)
                    out.fields.push_back({RasterQuadFieldKind::Parameter, attr, selector, 4});
        } else out.fields.push_back({RasterQuadFieldKind::Interpolant, attr, 0, 4});
    }
    static constexpr uint32_t system_widths[7] = {2, 2, 2, 3, 2, 2, 2};
    for (uint32_t field = 0; field < 7; ++field)
        if (inputs.interpolation.system_locations[field] != FragmentInterpolationLayout::kUnusedLocation) {
            if (!inputs.has_system_inputs || !inputs.launch.input_ena_available ||
                !inputs.launch.input_addr_available)
                return reject("quad-collector-system-routing-unavailable");
            // Sample interpolation needs the separately enabled SampleRateShading contract. This
            // first single-sample producer does not fabricate that enabled feature or substitute
            // center interpolation just because the physical sample count happens to be one.
            if (field == 0 || field == 4) return reject("quad-collector-sample-interpolation-unimplemented");
            out.fields.push_back({RasterQuadFieldKind::SystemInterpolation, field, 0, system_widths[field]});
        }
    out.lane_words = kRasterQuadLaneFixedWords;
    for (const auto& field : out.fields) out.lane_words += field.words;
    out.record_words = out.lane_words * 4;
    out.max_quads = max_quads;
    if (uint64_t(out.record_words) * max_quads * 4 + 16 > (8u << 20))
        return reject("quad-collector-byte-budget-exceeded");

    SpirvCompute b;
    b.float_transport = inputs.float_transport;
    b.fragment_interpolation = &inputs.interpolation;
    b.begin_fragment(nullptr, 0); // No guest EXP or attachment outputs.
    // Khronos SPIR-V enums: GroupNonUniformQuad=68; QuadBroadcast=365.
    b.put(b.caps, Op_Capability, {Cap_GroupNonUniform});
    b.put(b.caps, Op_Capability, {Cap_GroupNonUniformQuad});
    b.put(b.caps, Op_Capability, {Cap_Geometry}); // Fragment PrimitiveId input.
    auto builtin = [&](uint32_t type, uint32_t decoration, bool flat) {
        const uint32_t ptr = b.id(), variable = b.id(), value = b.id();
        b.put(b.types, Op_TypePointer, {ptr, SC_Input, type});
        b.put(b.types, Op_Variable, {ptr, variable, SC_Input});
        b.put(b.deco, Op_Decorate, {variable, Dec_BuiltIn, decoration});
        if (flat) b.put(b.deco, Op_Decorate, {variable, Dec_Flat});
        b.iface.push_back(variable);
        b.put(b.code, Op_Load, {type, value, variable});
        return value;
    };
    const uint32_t helper = b.helper_invocation();
    const uint32_t helper_word = b.sel(helper, b.uconst(1), b.uconst(0));
    const uint32_t facing = builtin(b.t_bool, 17, false); // FrontFacing
    const uint32_t primitive = b.i2u(builtin(b.t_i32, 7, true)); // PrimitiveId
    // Helpers' coverage values are observed but explicitly unavailable to later guest consumers.
    const uint32_t mask_array = b.id(), mask_ptr = b.id(), mask_var = b.id(), mask_element_ptr = b.id();
    b.put(b.types, Op_TypeArray, {mask_array, b.t_i32, b.uconst(1)});
    b.put(b.types, Op_TypePointer, {mask_ptr, SC_Input, mask_array});
    b.put(b.types, Op_TypePointer, {mask_element_ptr, SC_Input, b.t_i32});
    b.put(b.types, Op_Variable, {mask_ptr, mask_var, SC_Input});
    b.put(b.deco, Op_Decorate, {mask_var, Dec_BuiltIn, BI_SampleMask});
    b.iface.push_back(mask_var);
    const uint32_t mask_address = b.id(), mask_value = b.id();
    b.put(b.code, Op_AccessChain, {mask_element_ptr, mask_address, mask_var, b.uconst(0)});
    b.put(b.code, Op_Load, {b.t_i32, mask_value, mask_address});
    std::vector<uint32_t> values{helper_word, b.sel(helper, b.uconst(0), b.uconst(1)),
        b.i2u(mask_value), b.sel(facing, b.uconst(1), b.uconst(0)), primitive};
    for (uint32_t component = 0; component < 4; ++component)
        values.push_back(b.fragcoord_component(component));
    for (const auto& field : out.fields)
        for (uint32_t component = 0; component < field.words; ++component) {
            uint32_t value = field.kind == RasterQuadFieldKind::Interpolant
                ? b.interp_read(field.index, component)
                : field.kind == RasterQuadFieldKind::Parameter
                    ? b.interp_parameter(field.index, component, field.selector)
                    : b.system_interpolation_component(field.index, component);
            if (!value) return reject("quad-collector-input-unavailable");
            values.push_back(value);
        }
    std::array<std::vector<uint32_t>, 4> gathered;
    // All required quad operations precede EVERY election, append and budget branch. Helpers
    // participate in these defined quad operations, not in SSBO stores or whole-subgroup votes.
    for (uint32_t lane = 0; lane < 4; ++lane)
        for (uint32_t value : values) {
            const uint32_t result = b.id();
            b.put(b.code, 365, {b.t_u32, result, b.uconst(Scope_Subgroup), value, b.uconst(lane)});
            gathered[lane].push_back(result);
        }
    uint32_t first = b.uconst(4);
    for (int lane = 3; lane >= 0; --lane)
        first = b.sel(b.ucmp(Op_IEqual, gathered[lane][0], b.uconst(0)), b.uconst(lane), first);
    // This is the HOST quad index, not a guest lane ID. subgroup_local_id() models guest wave
    // identity and would incorrectly stamp an exact Wave64 requirement on this standalone pass.
    const uint32_t own_quad_index = b.ibin(Op_BitwiseAnd,
        builtin(b.t_u32, BI_SubgroupLocalInvocationId, true), b.uconst(3));
    const uint32_t publish = b.land(b.logical_not(helper), b.ucmp(Op_IEqual, own_quad_index, first));

    const uint32_t array = b.id(), block = b.id(), block_ptr = b.id(), word_ptr = b.id(), buffer = b.id();
    b.put(b.types, Op_TypeRuntimeArray, {array, b.t_u32});
    b.put(b.deco, Op_Decorate, {array, Dec_ArrayStride, 4});
    b.put(b.types, Op_TypeStruct, {block, array});
    b.put(b.deco, Op_Decorate, {block, Dec_Block});
    b.put(b.deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
    b.put(b.types, Op_TypePointer, {block_ptr, SC_StorageBuffer, block});
    b.put(b.types, Op_TypePointer, {word_ptr, SC_StorageBuffer, b.t_u32});
    b.put(b.types, Op_Variable, {block_ptr, buffer, SC_StorageBuffer});
    b.put(b.deco, Op_Decorate, {buffer, Dec_DescriptorSet, 1});
    b.put(b.deco, Op_Decorate, {buffer, Dec_Binding, 0});
    auto address = [&](uint32_t offset) {
        const uint32_t result = b.id();
        b.put(b.code, Op_AccessChain, {word_ptr, result, buffer, b.uconst(0), offset});
        return result;
    };
    const uint32_t counter = address(b.uconst(0)), overflow = address(b.uconst(1));
    const uint32_t end = b.id(), append = b.id();
    b.put(b.code, Op_SelectionMerge, {end, 0});
    b.put(b.code, Op_BranchConditional, {publish, append, end});
    b.emit_label(append);
    const uint32_t initial = b.id();
    b.put(b.code, 227, {b.t_u32, initial, counter, b.uconst(Scope_Device), b.uconst(0)}); // AtomicLoad
    const uint32_t loop = b.id(), attempt = b.id(), cont = b.id(), done = b.id();
    const uint32_t current = b.id(), observed = b.id();
    b.emit_branch(loop); b.emit_label(loop);
    b.put(b.code, Op_Phi, {b.t_u32, current, initial, append, observed, cont});
    const uint32_t room = b.ucmp(Op_ULessThan, current, b.uconst(max_quads));
    b.put(b.code, Op_LoopMerge, {done, cont, 0});
    b.put(b.code, Op_BranchConditional, {room, attempt, done});
    b.emit_label(attempt);
    const uint32_t next = b.ibin(Op_IAdd, current, b.uconst(1));
    b.put(b.code, Op_AtomicCompareExchange,
        {b.t_u32, observed, counter, b.uconst(Scope_Device), b.uconst(0), b.uconst(0), next, current});
    const uint32_t won = b.ucmp(Op_IEqual, observed, current);
    // The loop's continue construct is OUTSIDE its loop construct (§2.11 SPIR-V). It cannot
    // also be the merge of a selection headed inside that loop. Keep the CAS body linear and
    // put the retry/exit decision in the single continue/back-edge block, whose two legal exits
    // are the loop header and its merge. Every attempted CAS reaches this block exactly once.
    b.emit_branch(cont); b.emit_label(cont);
    b.put(b.code, Op_BranchConditional, {won, done, loop});
    b.emit_label(done);
    const uint32_t accepted = b.id();
    b.put(b.code, Op_Phi, {b.t_bool, accepted, b.bfalse(), loop, b.btrue(), cont});
    const uint32_t write = b.id(), full = b.id(), finish_append = b.id();
    b.put(b.code, Op_SelectionMerge, {finish_append, 0});
    b.put(b.code, Op_BranchConditional, {accepted, write, full});
    b.emit_label(full);
    const uint32_t ignored = b.id();
    b.put(b.code, 241, {b.t_u32, ignored, overflow, b.uconst(Scope_Device), b.uconst(0), b.uconst(1)});
    b.emit_branch(finish_append);
    b.emit_label(write);
    const uint32_t base = b.ibin(Op_IAdd, b.uconst(kRasterQuadBufferHeaderWords),
                               b.ibin(Op_IMul, current, b.uconst(out.record_words)));
    for (uint32_t lane = 0; lane < 4; ++lane)
        for (uint32_t word = 0; word < out.lane_words; ++word)
            b.put(b.code, Op_Store, {address(b.ibin(Op_IAdd, base,
                b.uconst(lane * out.lane_words + word))), gathered[lane][word]});
    b.emit_branch(finish_append); b.emit_label(finish_append);
    b.emit_branch(end); b.emit_label(end);
    auto spirv = b.finish();
    if (spirv.empty()) return reject("quad-collector-emission-refused");
    return spirv;
}

std::string decode_raster_quad_records(const RasterQuadCollector& collector,
    const uint32_t* words, size_t word_count, uint32_t primitive_count,
    std::vector<std::vector<uint32_t>>& quads) {
    quads.clear();
    const uint64_t capacity = uint64_t(kRasterQuadBufferHeaderWords) +
        uint64_t(collector.record_words) * collector.max_quads;
    if (!collector.rejection.empty() || collector.lane_words < kRasterQuadLaneFixedWords ||
        collector.lane_words > 512 || collector.record_words != collector.lane_words * 4 ||
        !collector.max_quads || collector.max_quads > 4096 || !words ||
        capacity != word_count || words[2] != collector.record_words ||
        words[3] != kRasterQuadMagic || words[0] > collector.max_quads)
        return "quad-collector-output-header-malformed";
    if (words[1]) return "quad-collector-output-overflow";
    std::vector<std::vector<uint32_t>> pending;
    // A geometric key is not a unique Vulkan quad scope: a driver may run one 2x2 quad of a
    // primitive as several helper-backed invocations whose nonhelper lanes are disjoint (llvmpipe
    // does; RADV does not, #4277). Such split scopes are reassembled into ONE record, each lane
    // taken from the invocation that actually ran it as a nonhelper, so the downstream four-lane
    // quad bridge sees the same complete quad domain on every driver. Nothing is invented: a lane
    // that is a helper in every split observation must carry identical words in all of them, or
    // the whole collection is refused. Vulkan also permits repeated fragment invocations, so an
    // OVERLAPPING nonhelper lane stays a conservative supported-domain refusal, not proof of a
    // driver or publisher error.
    struct Seen { uint32_t nonhelper_mask = 0, first_record = 0; size_t pending_index = 0; };
    std::map<std::array<uint32_t, 3>, Seen> keys;
    for (uint32_t q = 0; q < words[0]; ++q) {
        const auto* row = words + kRasterQuadBufferHeaderWords + size_t(q) * collector.record_words;
        uint32_t nonhelper_mask = 0;
        const float x = std::bit_cast<float>(row[5]), y = std::bit_cast<float>(row[6]);
        if (!std::isfinite(x) || !std::isfinite(y)) return "quad-collector-output-topology-malformed";
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const auto* value = row + size_t(lane) * collector.lane_words;
            if (value[0] > 1 || value[1] != !value[0] || value[3] > 1 ||
                value[3] != row[3] || value[4] != row[4] || value[4] >= primitive_count ||
                (!value[0] && value[2] != 1))
                return "quad-collector-output-provenance-malformed";
            // Vulkan Shader Quad scope defines these neighbor relations, not an even framebuffer
            // origin. This producer fixes one-pixel fragments and a single sample; helpers retain
            // their actual extrapolated inputs and their mask word remains explicitly unavailable.
            const float lx = std::bit_cast<float>(value[5]), ly = std::bit_cast<float>(value[6]);
            if (!std::isfinite(lx) || !std::isfinite(ly) ||
                lx != x + float(lane & 1) || ly != y + float(lane >> 1))
                return "quad-collector-output-topology-malformed";
            if (!value[0]) nonhelper_mask |= 1u << lane;
        }
        if (!nonhelper_mask) return "quad-collector-helper-only-publication";
        const auto [previous, first] = keys.try_emplace(
            std::array<uint32_t, 3>{row[4], row[5], row[6]}, Seen{nonhelper_mask, q});
        if (!first) {
            const uint32_t seen_mask = previous->second.nonhelper_mask;
            const bool overlap = (seen_mask & nonhelper_mask) != 0;
            // Preserve bounded actual-wire evidence before a refusal. A CI failure containing only
            // the old duplicate-key reason could not distinguish disjoint helper-backed scopes
            // from overlapping observations or an incorrect primitive identity.
            static std::atomic<uint32_t> collision_reports{0};
            uint32_t report = collision_reports.load(std::memory_order_relaxed);
            while (report < 32 && !collision_reports.compare_exchange_weak(
                    report, report + 1, std::memory_order_relaxed)) {}
            if (report < 32) {
                std::fprintf(stderr,
                    "[raster-quad-collision] primitive=%u origin_bits=%08x,%08x seen_mask=0x%x next_mask=0x%x first_record=%u record=%u decision=%s; host scopes, no guest packing/order authority\n",
                    row[4], row[5], row[6], seen_mask, nonhelper_mask,
                    previous->second.first_record, q, overlap ? "refused-overlap-unproved" : "merged-disjoint");
                if (report < 4) {
                    for (uint32_t record : {previous->second.first_record, q}) {
                        const auto* raw = words + kRasterQuadBufferHeaderWords + size_t(record) * collector.record_words;
                        for (uint32_t lane = 0; lane < 4; ++lane) {
                            std::fprintf(stderr, "[raster-quad-collision-wire] record=%u lane=%u words=", record, lane);
                            for (uint32_t word = 0; word < collector.lane_words; ++word)
                                std::fprintf(stderr, "%s%08x", word ? "," : "", raw[size_t(lane) * collector.lane_words + word]);
                            std::fputc('\n', stderr);
                        }
                    }
                }
            }
            if (overlap) return "quad-collector-overlapping-nonhelper-scopes-unproved";
            auto& merged = pending[previous->second.pending_index];
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const uint32_t bit = 1u << lane;
                const size_t at = size_t(lane) * collector.lane_words;
                if (nonhelper_mask & bit) {
                    std::copy(row + at, row + at + collector.lane_words, merged.begin() + at);
                } else if (!(seen_mask & bit) &&
                           !std::equal(row + at, row + at + collector.lane_words, merged.begin() + at)) {
                    return "quad-collector-split-scope-helper-inputs-disagree";
                }
            }
            previous->second.nonhelper_mask |= nonhelper_mask;
            continue;
        }
        previous->second.pending_index = pending.size();
        pending.emplace_back(row, row + collector.record_words);
    }
    quads = std::move(pending);
    return {};
}
} // namespace prosper::gpu
