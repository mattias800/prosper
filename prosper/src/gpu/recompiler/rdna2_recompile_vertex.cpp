// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"

namespace prosper::gpu {


size_t rdna2_recompile_code_span(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return 0;
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    size_t required = program_dwords;
    std::vector<Rdna2Inst> terminating_cfg = ins;
    size_t terminating_span = program_dwords;
    if (extend_terminating_if_else(code, dwords, terminating_cfg, &terminating_span))
        required = std::max(required, terminating_span);
    // Detection both proves the compiler idiom and bounds every referenced table. Do not retain an
    // arbitrary post-ENDPGM trailer: only bytes that can affect the generated SPIR-V belong in the key.
    (void)detect_pcrel_tables(terminating_cfg, code, dwords, &required);
    const PcrelDispatchInfo dispatch = detect_pcrel_dispatch(ins, code, dwords, program_dwords);
    if (dispatch.valid) required = std::max(required, dispatch.required_dwords);
    return std::min(required, dwords);
}

namespace {


}  // namespace

// Astro Bot's observed NGG wrappers use wave-shared plumbing for vertex allocation/compaction. The
// current Vulkan vertex shell intentionally projects those complete wrappers to one private guest
// lane; applying the projection to arbitrary NGG programs would silently miscompile peer-lane state.
// Keep each exception byte-exact and fail closed for every other wrapper. The hashes are FNV-1a over
// the little-endian instruction bytes through S_ENDPGM, matching the raw hashes in capture
// diagnostics. A proven PC-relative constant-table tail is deliberately excluded from the wrapper
// identity while remaining available to the recompiler.
static bool is_astro_bot_ngg_one_lane_wrapper(const uint32_t* code, size_t dwords) {
    if (!code) return false;
    std::vector<Rdna2Inst> instructions;
    const size_t program_dwords = rdna2_walk(code, dwords, instructions);
    if (program_dwords != 54 && program_dwords != 734 && program_dwords != 749 &&
        program_dwords != 3124 && program_dwords != 3435 && program_dwords != 3455 &&
        program_dwords != 3917)
        return false;
    const uint64_t hash = shader_program_hash(code, program_dwords);
    return (program_dwords == 54 && hash == 0x9e9d8e37bcc70607ull) ||
           (program_dwords == 734 && hash == 0x79eb2b954b07dc8eull) ||
           // The same 734-word culling wrapper is live-linked after its exact 15-word fetch prolog.
           (program_dwords == 749 && hash == 0xb440349937df751eull) ||
           (program_dwords == 3124 && hash == 0x41e6ac616c18d295ull) ||
           (program_dwords == 3435 && hash == 0xfad7a9f486523cfcull) ||
           // The same 3435-word wrapper is live-linked after its exact 20-word fetch prolog.
           // Hashing the complete concatenated program keeps the one-lane projection byte-exact.
           (program_dwords == 3455 && hash == 0x562ce5ad01c4c6e3ull) ||
           (program_dwords == 3917 && hash == 0x7f5f2349e2816f5eull);
}

VertexPrologInfo rdna2_vertex_prolog_info(const uint32_t* code, size_t dwords) {
    VertexPrologInfo result;
    if (!code || !dwords) return result;

    const bool prologlog = getenv("PROSPER_PROLOGLOG") != nullptr;
    uint64_t phash = 0xcbf29ce484222325ull;
    if (prologlog)
        for (size_t i = 0; i < dwords && i < 4096; ++i) phash = (phash ^ code[i]) * 0x100000001b3ull;
    auto prolog_note = [&](const char* what, const Rdna2Inst* at) {
        if (!prologlog) return;
        static std::set<uint64_t> seen; static std::mutex mx;
        std::lock_guard<std::mutex> lk(mx);
        if (!seen.insert(phash).second) return;
        fprintf(stderr, "[prologlog] hash=%016llx dwords=%zu %s pc=%u fmt=%d\n",
                (unsigned long long)phash, dwords, what, at ? at->pc : 0u, at ? (int)at->fmt : -1);
    };
    // Only the prefix before the transfer can belong to a fetch prolog. Track its branch-target
    // bounds while decoding, so neither the prefix nor the unused shader tail needs a vector.
    int64_t min_branch_target = 0;
    int64_t max_branch_target = 0;
    for (size_t pc = 0; pc < dwords;) {
        Rdna2Inst instruction = rdna2_decode_one(code + pc, dwords - pc);
        instruction.pc = static_cast<uint32_t>(pc);
        // A fetch prolog has no architectural output or program termination of its own. Encountering
        // either before the transfer means this is a complete/different shader, not the split ABI.
        if (instruction.is_end || instruction.fmt == Rdna2Format::EXP ||
            instruction.fmt == Rdna2Format::Unknown) {
            prolog_note("BAIL", &instruction);
            return {};
        }
        if (instruction.fmt == Rdna2Format::SOP1 && instruction.opcode == 0x20) {
            // GFX9+ merged-stage fetch prologs receive the continuation PC in reserved s[6:7].
            // Keep this exact pair: an arbitrary indirect jump must not become host fallthrough.
            if (instruction.n_src != 1 || instruction.src[0].kind != OperandKind::SGPR ||
                instruction.src[0].value != 6 || instruction.len_dwords != 1)
                return {};
            prolog_note("TRANSFER", &instruction);
            // Every direct branch must remain inside the retained prefix or land exactly on the
            // transfer (main pc0 after linking), never in discarded padding/data. The linked body
            // recompiler performs the remaining structured-CFG validation.
            if (instruction.pc == 0 || min_branch_target < 0 ||
                max_branch_target > instruction.pc)
                return {};
            result.valid = true;
            result.setpc_pc = instruction.pc;
            result.prefix_dwords = instruction.pc;
            return result;
        }
        if (instruction.fmt == Rdna2Format::SOPP &&
            (instruction.opcode == 0x02 ||
             (instruction.opcode >= 0x04 && instruction.opcode <= 0x09))) {
            const int64_t target = static_cast<int64_t>(instruction.pc) + instruction.len_dwords +
                                   static_cast<int64_t>(instruction.simm16);
            min_branch_target = std::min(min_branch_target, target);
            max_branch_target = std::max(max_branch_target, target);
        }
        if (instruction.len_dwords == 0) break;
        pc += instruction.len_dwords;
    }
    return {};
}

namespace {

// A no-GS NGG program is split into two machine-code allocations by the guest compiler: the
// logical vertex producer writes one compact per-vertex LDS record, then a compiler-generated NGG
// wrapper culls/compacts primitives and exports fields from that record.  Vulkan's vertex stage
// already launches exactly the logical draw vertices and performs primitive assembly itself.  When
// both sides of this ABI can be proven from the machine code, execute only the producer and export
// the same LDS fields directly.  This avoids pretending that Function-private LDS can communicate
// between independent Vulkan vertex invocations.
struct NggPassthroughLayout {
    bool valid = false;
    uint32_t producer_base_vgpr = 0;
    uint32_t producer_base_byte = 0;
    uint32_t record_stride_bytes = 0;
    std::array<int32_t, 4> position = {-1, -1, -1, -1};
    std::array<std::array<int32_t, 4>, 32> params{};
    uint32_t param_mask = 0;

    NggPassthroughLayout() {
        for (auto& param : params) param.fill(-1);
    }
};

struct NggLdsSource {
    bool valid = false;
    uint32_t byte_offset = 0;
    uint32_t stride_bytes = 0;
    uint32_t index_vgpr = 0;
};

// Resolve the terminal wrapper's canonical `stride * exporter + constant` LDS address.  Requiring
// the nearest writer to be this exact u24 MAD shape keeps the optimization fail-closed when a user
// GS or a different compiler layout performs real output computation.
NggLdsSource ngg_terminal_lds_source(const std::vector<Rdna2Inst>& ins, size_t load_index,
                                     uint32_t output_vgpr) {
    const Rdna2Inst& load = ins[load_index];
    uint32_t address_vgpr = 0, component_byte = 0;
    if (load.fmt != Rdna2Format::DS || load.ds_gds) return {};
    if (load.opcode == 0x36u) {                         // ds_read_b32
        if (output_vgpr != static_cast<uint32_t>(load.dst.value)) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        component_byte = load.literal;
    } else if (load.opcode == 0x37u) {                  // ds_read2_b32
        const uint32_t first = static_cast<uint32_t>(load.dst.value);
        if (output_vgpr < first || output_vgpr > first + 1u) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        const uint32_t component = output_vgpr - first;
        component_byte = ((load.literal >> (component * 8u)) & 0xffu) * 4u;
    } else if (load.opcode == 0x76u || load.opcode == 0xfeu || load.opcode == 0xffu) {
        const uint32_t count = load.opcode == 0x76u ? 2u : load.opcode == 0xfeu ? 3u : 4u;
        const uint32_t first = static_cast<uint32_t>(load.dst.value);
        if (output_vgpr < first || output_vgpr >= first + count) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        component_byte = load.literal + (output_vgpr - first) * 4u;
    } else {
        return {};
    }

    for (size_t j = load_index; j-- > 0;) {
        const Rdna2Inst& writer = ins[j];
        if (writer.dst.kind != OperandKind::VGPR ||
            static_cast<uint32_t>(writer.dst.value) != address_vgpr)
            continue;
        // The compacted record address has either of the two canonical compiler forms below:
        //
        //   v_mad_u32_u24 addr, stride, exporter, constant
        //   v_mul_u32_u24 addr, stride, exporter; ds_read ... offset:constant
        //
        // The latter avoids a MAD when the entire constant fits in the DS instruction's immediate.
        // Both prove the same `stride * exporter + constant` identity; accepting only these exact
        // integer-u24 forms keeps arbitrary wrapper address arithmetic fail-closed.
        const bool mad = writer.fmt == Rdna2Format::VOP3 && writer.opcode == 0x143u &&
                         !writer.has_modifier && writer.has_literal;
        const bool mul = writer.fmt == Rdna2Format::VOP2 && writer.opcode == 0x0bu &&
                         !writer.has_modifier && !writer.has_literal;
        if (!mad && !mul) return {};
        // The first two operands may be swapped.
        int stride_src = -1, index_src = -1;
        for (int k = 0; k < 2; ++k) {
            if (writer.src[k].kind == OperandKind::InlineInt && writer.src[k].value > 0)
                stride_src = k;
            else if (writer.src[k].kind == OperandKind::VGPR)
                index_src = k;
        }
        if (stride_src < 0 || index_src < 0 ||
            (mad && writer.src[2].kind != OperandKind::Literal))
            return {};
        const uint32_t stride = static_cast<uint32_t>(writer.src[stride_src].value);
        if ((stride & 3u) || stride < 16u || stride > 4096u ||
            (mad && writer.literal > UINT32_MAX - component_byte))
            return {};
        return {true, (mad ? writer.literal : 0u) + component_byte, stride,
                static_cast<uint32_t>(writer.src[index_src].value)};
    }
    return {};
}

NggLdsSource ngg_find_terminal_output(const std::vector<Rdna2Inst>& ins, size_t export_index,
                                      uint32_t output_vgpr) {
    // The direct output loads sit in the export block.  Stop at the nearest writer; skipping a
    // transform or phi would turn a user GS into a false passthrough.
    for (size_t j = export_index; j-- > 0;) {
        const Rdna2Inst& writer = ins[j];
        if (writer.fmt == Rdna2Format::DS) {
            const NggLdsSource source = ngg_terminal_lds_source(ins, j, output_vgpr);
            if (source.valid) return source;
            const uint32_t first = static_cast<uint32_t>(writer.dst.value);
            const uint32_t count = writer.opcode == 0x37u || writer.opcode == 0x76u ? 2u
                                 : writer.opcode == 0xfeu ? 3u
                                 : writer.opcode == 0xffu ? 4u : 1u;
            if (output_vgpr >= first && output_vgpr < first + count) return {};
        }
        if (writer.dst.kind == OperandKind::VGPR &&
            static_cast<uint32_t>(writer.dst.value) == output_vgpr)
            return {};
    }
    return {};
}

NggPassthroughLayout analyze_ngg_passthrough(const uint32_t* prolog, size_t prefix_dwords,
                                             const uint32_t* main, size_t main_dwords) {
    NggPassthroughLayout out;
    auto reject = [&](const char* reason) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-ngg-passthrough-reject] %s\n", reason);
        return NggPassthroughLayout{};
    };
    std::vector<Rdna2Inst> producer, wrapper;
    rdna2_walk(prolog, prefix_dwords, producer);
    rdna2_walk(main, main_dwords, wrapper);
    if (producer.empty() || wrapper.empty()) return reject("empty producer/wrapper");

    // Prove that every LDS operation in the producer is a non-atomic store into one record base.
    // The private-LDS execution is valid only before the cross-lane wrapper starts reading it.
    bool saw_store = false;
    uint32_t base_vgpr = UINT32_MAX;
    std::set<uint32_t> stored_bytes;
    for (const Rdna2Inst& in : producer) {
        if (in.fmt == Rdna2Format::EXP || in.is_end) return reject("producer terminates or exports");
        if (in.fmt != Rdna2Format::DS) continue;
        if (in.ds_gds || in.src[0].kind != OperandKind::VGPR)
            return reject("producer uses non-LDS DS address");
        const uint32_t base = static_cast<uint32_t>(in.src[0].value);
        if (base_vgpr == UINT32_MAX) base_vgpr = base;
        if (base != base_vgpr) return reject("producer has multiple LDS record bases");
        auto store_byte = [&](uint32_t byte) {
            if ((byte & 3u) == 0u) stored_bytes.insert(byte);
        };
        if (in.opcode == 0x0du) {
            store_byte(in.literal);
        } else if (in.opcode == 0x0eu) {
            store_byte((in.literal & 0xffu) * 4u);
            store_byte(((in.literal >> 8) & 0xffu) * 4u);
        } else if (in.opcode == 0x4du || in.opcode == 0xdeu || in.opcode == 0xdfu) {
            const uint32_t count = in.opcode == 0x4du ? 2u : in.opcode == 0xdeu ? 3u : 4u;
            for (uint32_t k = 0; k < count; ++k) store_byte(in.literal + k * 4u);
        } else {
            return reject("producer has a non-store/cross-lane DS operation");
        }
        saw_store = true;
    }
    if (!saw_store || stored_bytes.empty()) return reject("producer writes no aligned LDS record");

    bool saw_alloc = false, saw_primitive_export = false, saw_position = false;
    uint32_t common_stride = 0, common_index = UINT32_MAX;
    std::vector<uint32_t> absolute_offsets;
    auto accept_source = [&](const NggLdsSource& source, int32_t& destination) -> bool {
        if (!source.valid || (common_stride && source.stride_bytes != common_stride) ||
            (common_index != UINT32_MAX && source.index_vgpr != common_index))
            return false;
        common_stride = source.stride_bytes;
        common_index = source.index_vgpr;
        destination = static_cast<int32_t>(source.byte_offset);
        absolute_offsets.push_back(source.byte_offset);
        return true;
    };
    for (size_t i = 0; i < wrapper.size(); ++i) {
        const Rdna2Inst& in = wrapper[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10u) saw_alloc = true;
        if (in.fmt != Rdna2Format::EXP) continue;
        if (in.exp_target == 20u) { saw_primitive_export = true; continue; }
        if (in.exp_target != 12u && in.exp_target < 32u) continue;
        if (in.exp_compr) return reject("terminal output uses a compressed export");
        if (in.exp_target == 12u) {
            if (saw_position || in.exp_en != 0xfu)
                return reject("POS0 is duplicated or incomplete");
            saw_position = true;
            for (uint32_t component = 0; component < 4; ++component) {
                if (!accept_source(ngg_find_terminal_output(
                                       wrapper, i, static_cast<uint32_t>(in.src[component].value)),
                                   out.position[component]))
                    return reject("POS0 does not directly load the terminal LDS record");
            }
        } else {
            const uint32_t param = in.exp_target - 32u;
            if (param >= out.params.size() || (out.param_mask & (1u << param)))
                return reject("PARAM export is out of range or duplicated");
            out.param_mask |= 1u << param;
            for (uint32_t component = 0; component < 4; ++component) {
                if (!(in.exp_en & (1u << component))) continue;
                if (!accept_source(ngg_find_terminal_output(
                                       wrapper, i, static_cast<uint32_t>(in.src[component].value)),
                                   out.params[param][component]))
                    return reject("PARAM does not directly load the terminal LDS record");
            }
        }
    }
    if (!saw_alloc || !saw_primitive_export || !saw_position || absolute_offsets.empty() ||
        !common_stride || base_vgpr == UINT32_MAX)
        return reject("wrapper lacks the no-GS allocation/export shape");

    const uint32_t wrapper_base = *std::min_element(absolute_offsets.begin(), absolute_offsets.end());
    const uint32_t producer_base = *stored_bytes.begin();
    auto normalize = [&](int32_t& byte) -> bool {
        if (byte < 0) return true;
        const uint32_t absolute = static_cast<uint32_t>(byte);
        if (absolute < wrapper_base) return false;
        const uint32_t relative = absolute - wrapper_base;
        if ((relative & 3u) || relative >= common_stride ||
            producer_base > UINT32_MAX - relative ||
            !stored_bytes.count(producer_base + relative))
            return false;
        byte = static_cast<int32_t>((producer_base + relative) / 4u);
        return true;
    };
    for (int32_t& component : out.position)
        if (!normalize(component)) return reject("POS0 offset does not match the producer record");
    for (auto& param : out.params)
        for (int32_t& component : param)
            if (!normalize(component)) return reject("PARAM offset does not match the producer record");

    out.valid = true;
    out.producer_base_vgpr = base_vgpr;
    out.producer_base_byte = producer_base;
    out.record_stride_bytes = common_stride;
    return out;
}

} // namespace

static std::vector<uint32_t> recompile_vertex_impl(const uint32_t* code, size_t dwords,
                                                   const ShaderResourceTable* rt,
                                                   const PixelInputMapping* pixel_inputs,
                                                   bool capture_position,
                                                   uint32_t virtual_lds_dwords,
                                                   const NggPassthroughLayout* passthrough,
                                                   bool allow_test_ngg_output_gate,
                                                   bool allow_test_ngg_one_lane,
                                                   RecompileDiagnosticContext diagnostic) {
    const uint32_t passthrough_mask =
        pixel_inputs ? pixel_inputs->effective_passthrough_mask() : 0u;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    SpirvCompute b;
    // The vertex stage recorded its rejects at address 0, and record_terminal_reject_reason()
    // early-returns on a zero address -- so `last_terminal_reject_reason()` came back empty for
    // EVERY vertex program and the unconditional skip line could only print `reason=unrecorded`.
    // #3130 fixed exactly this for the fragment stage; the vertex half was left behind.
    b.diagnostic = diagnostic;
    b.capture_position = capture_position;   // geometry-probe: mark gl_Position for xfb capture (gated)
    b.vertex_lds_dwords = std::min(virtual_lds_dwords, 16384u);
    b.vertices_per_instance = rt ? rt->vertices_per_instance : 0u;
    // Shader I/O value tap (PROSPER_SHADER_TAP=pc): redirect the position export to the intermediate VGPR
    // produced at that PC. Applies to the vertex stage only; captured via the geometry probe.
    if (const char* tap = getenv("PROSPER_SHADER_TAP")) b.tap_pc = static_cast<uint32_t>(strtoul(tap, nullptr, 0));
    b.begin_vertex(rt);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // NGG vertex shaders (the exact GS_ALLOC_REQ message present) carry the vertex index in v5, not v0, and wrap
    // the body in wave-packing plumbing (s_sendmsg / exp prim / s_lshr_b64 exec) that lowers to no-ops in
    // our per-invocation model. Detect NGG and bind the index to v5 as well.
    bool ngg = passthrough && passthrough->valid;
    for (const auto& in : ins) { if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10 &&
            in.words[0] == 0xBF900009u) { ngg = true; break; } }
    const bool exact_ngg_projection = ngg && is_astro_bot_ngg_one_lane_wrapper(code, dwords);
    // The generic split-stage path uses private LDS only after the producer/wrapper analyzer has
    // proven that all visible outputs come from one per-vertex record. Wave-sensitive instruction
    // approximations remain restricted to the byte-exact projection above.
    b.ngg_private_lds = exact_ngg_projection || (passthrough && passthrough->valid) ||
                        (ngg && allow_test_ngg_output_gate);
    // The legacy byte-exact NGG projection and its explicit unit-test hook predate the linked-stage
    // ABI's exact LDS allocation. Preserve their proven 16 KiB private scratch contract when no
    // allocation was supplied; generic linked producers still require their real plumbed size.
    if (!b.vertex_lds_dwords && (exact_ngg_projection || allow_test_ngg_output_gate))
        b.vertex_lds_dwords = 4096;
    // Every wave/peer approximation is an exception for the one captured Astro wrapper, not a
    // property of the GS_ALLOC_REQ opcode. Other NGG programs retain only the ordinary merged-stage
    // ABI setup below and fail closed if they reach a lane-sensitive operation.
    b.ngg_one_lane = exact_ngg_projection || (ngg && allow_test_ngg_one_lane);
    // A LO+HI all-ones pair is the compiler's explicit wave64 lane-index construction. Infer the
    // width from that machine-code proof instead of assuming every NGG program is wave64. A low-only
    // producer may be wave32 or may use only half of a wave64 mask, so it remains fail-closed until
    // the graphics wave-size contract is plumbed independently.
    bool logical_mbcnt_lo = false, logical_mbcnt_hi = false, logical_mbcnt_invalid = false;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::VOP3 || (in.opcode != 0x365 && in.opcode != 0x366))
            continue;
        const bool all_ones = in.src[0].kind == OperandKind::InlineInt &&
                              in.src[0].value == -1;
        // Which instruction disqualified the model, not just that something did. The all-ones pair
        // is lowerable on its own, so on a mixed program the reject surfaces at IT and names a pc
        // that is not the cause -- Stray's 0x300f190000 rejects at pc=4 while the general-mask
        // forms that disqualified it are at pc 277-286 (#3135).
        if (!all_ones && b.vertex_general_mask_mbcnt_pc == UINT32_MAX)
            b.vertex_general_mask_mbcnt_pc = in.pc;
        logical_mbcnt_invalid |= !all_ones;
        logical_mbcnt_lo |= all_ones && in.opcode == 0x365;
        logical_mbcnt_hi |= all_ones && in.opcode == 0x366;
    }
    b.ngg_logical_lane = passthrough && passthrough->valid && !logical_mbcnt_invalid &&
                         logical_mbcnt_lo && logical_mbcnt_hi;
    if (b.ngg_logical_lane) b.wave_size = 64;
    b.allow_b32_masks = b.ngg_one_lane;
    uint32_t ngg_output_gate_begin = UINT32_MAX;
    uint32_t ngg_output_gate_end = 0;
    if (ngg) {
        // A terminal compacted-output suffix may reconstruct its values from a shader-embedded
        // constant table. Detect those loads with the same bounded PC-relative proof used by the
        // emitter; an arbitrary external load or any buffer write must not broaden this gate.
        PcrelTables output_tables;
        if (b.ngg_private_lds)
            output_tables = detect_pcrel_tables(ins, code, dwords);
        auto scalar_output_setup = [](const Rdna2Inst& candidate) {
            if (candidate.fmt != Rdna2Format::SOP1 &&
                candidate.fmt != Rdna2Format::SOP2 &&
                candidate.fmt != Rdna2Format::SOPK)
                return false;
            bool wrote_data = false;
            bool safe = !rdna2_instruction_may_change_exec(candidate);
            for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                wrote_data = true;
                safe &= base >= 0 && base + static_cast<int>(width) <= 106;
                safe &= !scalar_write_is_b64_mask(candidate, base);
            });
            return wrote_data && safe;
        };
        // MUBUF-only is DELIBERATE, not an oversight now that `output_tables.mtbuf` exists three
        // lines up (#2859). This gate is a narrowing allow-list for the byte-exact Astro Bot
        // compacted-output wrapper; admitting a typed consumer would widen an NGG output gate with no
        // title evidence behind it. A typed embedded table still FOLDS -- it just does not open this
        // gate, which is the conservative direction.
        auto embedded_output_load = [&](const Rdna2Inst& candidate) {
            if (candidate.fmt != Rdna2Format::MUBUF || candidate.mubuf_lds)
                return false;
            const bool read_only = candidate.opcode <= 0x03u ||
                (candidate.opcode >= 0x0cu && candidate.opcode <= 0x0fu);
            return read_only && output_tables.mubuf.contains(candidate.pc);
        };
        uint32_t end_pc = UINT32_MAX;
        for (const auto& in : ins)
            if (in.is_end) { end_pc = in.pc; break; }
        // An NGG primitive shader finishes by compacting surviving vertices, CMPX-testing whether
        // this lane owns one of those vertices, then exporting POS/PARAM values before S_ENDPGM.
        // Vulkan's vertex shell already represents one surviving guest vertex per invocation, but
        // retaining the condition is still useful when the compacted count is zero. Permit exports
        // under narrowed EXEC only for this mechanically bounded terminal output gate; ordinary
        // vertex CMPX/export programs remain rejected below.
        for (size_t i = 1; i < ins.size(); ++i) {
            const Rdna2Inst& branch = ins[i];
            if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 ||
                branch.simm16 <= 0 || branch_target(branch) != end_pc)
                continue;
            size_t previous = i;
            do { --previous; } while (previous > 0 && sopp_is_noop(ins[previous]));
            if (ins[previous].fmt != Rdna2Format::VOPC ||
                !vopc_is_cmpx(ins[previous].opcode))
                continue;
            // Production accepts data-dependent vertex suppression only for the byte-exact Astro
            // wrapper. The explicit test hook below exercises active/inactive export selection with
            // a tiny shader without turning that shader shape into a runtime allow-list exception.
            if (!b.ngg_private_lds && !allow_test_ngg_output_gate)
                continue;
            bool has_position = false;
            bool output_only = true;
            std::vector<uint32_t> trailing_vcc_branches;
            for (size_t j = i + 1; j < ins.size() && ins[j].pc < end_pc; ++j) {
                const Rdna2Inst& candidate = ins[j];
                if (candidate.fmt == Rdna2Format::EXP) {
                    has_position |= candidate.exp_target == 12;
                    continue;
                }
                // Astro's compacted-output suffix reconstructs the surviving vertex from private
                // LDS or a bounded shader-embedded table immediately before exporting it. Vector,
                // DS, and table-load destination writes are EXEC-predicated by emit_alu; scalar ALU
                // may only build ordinary data/descriptor registers. Admit them only for the
                // byte-exact wrapper (or the explicit test hook); arbitrary NGG shaders never reach
                // this exception, and buffer stores/external reads remain rejected.
                const bool output_rebuild = b.ngg_private_lds || allow_test_ngg_output_gate;
                if (output_rebuild &&
                    (candidate.fmt == Rdna2Format::VOP1 ||
                     candidate.fmt == Rdna2Format::VOP2 ||
                     candidate.fmt == Rdna2Format::VOP3 ||
                     candidate.fmt == Rdna2Format::VOP3P ||
                     (candidate.fmt == Rdna2Format::VOPC &&
                      !vopc_is_cmpx(candidate.opcode)) ||
                     candidate.fmt == Rdna2Format::DS ||
                     scalar_output_setup(candidate) ||
                     embedded_output_load(candidate)))
                    continue;
                if (candidate.fmt == Rdna2Format::SOPC) continue;
                if (sopp_is_noop(candidate)) continue;
                if (candidate.fmt == Rdna2Format::SOPP &&
                    (candidate.opcode == 0x04 || candidate.opcode == 0x05) &&
                    candidate.simm16 > 0 && branch_target(candidate) == end_pc)
                    continue;
                // The 7f5f wrapper exports POS, compares a per-vertex flag into VCC, then conditionally
                // skips only its trailing PARAM exports. Those exports cannot affect position/topology;
                // linearizing the branch merely supplies otherwise-undefined varyings for that path.
                if (has_position && candidate.fmt == Rdna2Format::SOPP &&
                    (candidate.opcode == 0x06 || candidate.opcode == 0x07) &&
                    candidate.simm16 > 0 && branch_target(candidate) == end_pc) {
                    trailing_vcc_branches.push_back(candidate.pc);
                    continue;
                }
                output_only = false;
                break;
            }
            if (has_position && output_only) {
                ngg_output_gate_begin = branch.pc;
                ngg_output_gate_end = end_pc;
                safe_branches.insert(trailing_vcc_branches.begin(), trailing_vcc_branches.end());
                break;
            }
        }
    }
    uint32_t vidx = b.load_vertex_index();
    uint32_t iidx = b.load_instance_index();
    rs.vreg[0] = vidx;                       // Legacy VS ABI: v0 = vertex index
    rs.vreg[3] = iidx;                       // Legacy VS ABI: v3 = instance index
    if (ngg) {
        // Locate the NGG prologue's LDS-to-ES vertex-index handoff without hard-coding a register or
        // program counter: the first MUBUF vaddr must be most recently defined by a scalar DS read.
        // The one-lane backend substitutes BuiltIn VertexIndex at that exact read (see DS lowering).
        for (size_t use = 0; use < ins.size(); ++use) {
            if (ins[use].is_end) break;
            if ((ins[use].fmt != Rdna2Format::MUBUF && ins[use].fmt != Rdna2Format::MTBUF) ||
                ins[use].src[0].kind != OperandKind::VGPR)
                continue;
            const int index_reg = ins[use].src[0].value;
            for (size_t def = use; def-- > 0;) {
                const Rdna2Inst& candidate = ins[def];
                if (candidate.dst.kind != OperandKind::VGPR || candidate.dst.value != index_reg)
                    continue;
                if (candidate.fmt == Rdna2Format::DS && candidate.opcode == 0x36) {
                    b.ngg_vertex_index_read_pc = candidate.pc;
                    b.ngg_vertex_index_value = vidx;
                }
                break;
            }
            break;
        }
        // GFX10's merged GS/ES ABI enters the ES prolog with vertex/instance indices in v5/v8 and
        // merged-wave info in s3: per-wave ES/GS counts [7:0]/[15:8], wave-in-TG [27:24], and
        // TG wave count [31:28]. The host draw has already omitted padding invocations, so expose a
        // full logical wave while deriving the architectural wave ID from the flattened invocation.
        rs.vreg[5] = vidx;
        rs.vreg[8] = iidx;
        if (exact_ngg_projection) {
            rs.sreg[3] = b.uconst(1);

        // The merged NGG wrapper guards its whole counted ES loop with EXECZ. In the single-lane
        // model above that one ES lane is active by construction, so the wave-empty shortcut cannot
        // be taken. Let the ordinary counted-loop lowering consume the loop instead of rejecting the
        // redundant outer guard merely because the original hardware mask was vector-shaped.
        const CountedLoop loop = detect_counted_loop(ins);
        if (loop.found) {
            // NGG culling unrolls several EXEC-predicated LDS blocks inside its counted loop. A
            // larger block may contain a smaller already-safe EXECZ block plus CMPX comparisons,
            // and ends by restoring a VCC-saved mask to EXEC. The generic safe-execz pass deliberately
            // rejects CMPX writes. Here they are exact: the branch immediately follows a CMPX that
            // narrowed EXEC, all effects in the skipped block are EXEC-predicated (further CMPX can
            // only narrow it again), and the common target performs the same EXEC=VCC restore on both
            // paths. Scan inside-out so nested blocks are proven before their parents.
            for (size_t branch_index = ins.size(); branch_index-- > 0;) {
                const Rdna2Inst& branch = ins[branch_index];
                if (branch.pc < loop.header_pc || branch.pc > loop.backedge_pc ||
                    branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 ||
                    branch.simm16 <= 0)
                    continue;
                size_t previous = branch_index;
                while (previous > 0) {
                    --previous;
                    if (!sopp_is_noop(ins[previous])) break;
                }
                if (previous >= branch_index || ins[previous].fmt != Rdna2Format::VOPC ||
                    !vopc_is_cmpx(ins[previous].opcode))
                    continue;
                const uint32_t target_pc = branch_target(branch);
                size_t target_index = ins.size();
                for (size_t i = 0; i < ins.size(); ++i)
                    if (ins[i].pc == target_pc) { target_index = i; break; }
                bool restores_saved_exec = false;
                for (size_t i = target_index; i < ins.size() && i < target_index + 3; ++i) {
                    const Rdna2Inst& candidate = ins[i];
                    if (candidate.fmt == Rdna2Format::SOP1 && candidate.opcode == 0x04 &&
                        candidate.dst.value >= 126 &&
                        (candidate.src[0].value == 106 || candidate.src[0].value == 107)) {
                        restores_saved_exec = true;
                        break;
                    }
                    bool clobbers_saved_mask = rdna2_instruction_may_change_exec(candidate) ||
                        candidate.fmt == Rdna2Format::VOPC;
                    for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                        clobbers_saved_mask |= base < 108 && 106 < base + static_cast<int>(width);
                    });
                    if (clobbers_saved_mask ||
                        (candidate.fmt == Rdna2Format::SOPP && candidate.opcode >= 0x02 &&
                         candidate.opcode <= 0x09 && candidate.opcode != 0x03))
                        break;
                }
                if (!restores_saved_exec)
                    continue;
                bool safe_block = true;
                for (const auto& candidate : ins) {
                    if (candidate.pc <= branch.pc || candidate.pc >= target_pc) continue;
                    const bool predicated_or_masked =
                        candidate.fmt == Rdna2Format::VOP1 ||
                        candidate.fmt == Rdna2Format::VOP2 ||
                        candidate.fmt == Rdna2Format::VOP3 ||
                        candidate.fmt == Rdna2Format::VOP3P ||
                        (candidate.fmt == Rdna2Format::VOPC &&
                         vopc_is_cmpx(candidate.opcode)) ||
                        candidate.fmt == Rdna2Format::MIMG ||
                        candidate.fmt == Rdna2Format::MUBUF ||
                        candidate.fmt == Rdna2Format::MTBUF ||
                        candidate.fmt == Rdna2Format::DS ||
                        candidate.fmt == Rdna2Format::FLAT;
                    const bool nested_safe = candidate.fmt == Rdna2Format::SOPP &&
                        candidate.opcode == 0x08 && safe_branches.count(candidate.pc);
                    if (!predicated_or_masked && !nested_safe && !sopp_is_noop(candidate)) {
                        safe_block = false;
                        break;
                    }
                }
                if (safe_block) safe_branches.insert(branch.pc);
            }
            for (const auto& in : ins) {
                if (in.pc >= loop.header_pc) break;
                if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                    branch_target(in) == loop.exit_pc)
                    safe_branches.insert(in.pc);
            }
        }
        } else {
            const uint32_t wave = b.ibin(
                Op_BitwiseAnd,
                b.ibin(Op_ShiftRightLogical, b.vertex_invocation_id(), b.uconst(6)),
                b.uconst(0xFu));
            rs.sreg[3] = b.ibin(Op_BitwiseOr, b.uconst(0x40004040u),
                                b.ibin(Op_ShiftLeftLogical, wave, b.uconst(24)));
        }
    }
    bool exported = false;
    auto exp_fn = [&](RegState& state, const Rdna2Inst& in) -> bool { // EXP POS0..3 -> gl_Position; PARAM -> varyings
        bool eok = true;   // a Special (wave-mask) source has no data value — reject, don't export 0 (#134)
        // COMPR pos/param exports carry two packed f16x2 pairs, not four f32 fields — reading the
        // VSRCs as full floats would pass packed-half bit patterns as x/y and stale registers as
        // z/w. Never observed in a vertex stage (compilers export positions/params at 32 bits);
        // reject fail-visibly until a live title exercises one.
        if (in.exp_compr) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "vertex compressed export pc=%u target=%u",
                                     in.pc, in.exp_target);
            return false;
        }
        // v_cmpx is now allowed in the vertex shell (allow_exec_update=true below): a divergent block
        // (v_cmpx … s_mov_b64 exec, saved — DOLL's per-vertex lighting/fog attenuation) predicates its
        // VGPR writes like compute. A vertex MUST still export from full EXEC — the compiled shape
        // always restores EXEC before its pos/param exports; if one ever arrives narrowed, reject
        // (fail-visibly) rather than export possibly-inactive-lane values.
        const bool terminal_ngg_output = ngg_output_gate_begin != UINT32_MAX &&
            in.pc > ngg_output_gate_begin && in.pc < ngg_output_gate_end;
        if (state.exec_narrowed && !terminal_ngg_output && (in.exp_target >= 32 ||
            (in.exp_target >= 12 && in.exp_target <= 16))) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "vertex export under narrowed exec pc=%u target=%u",
                                     in.pc, in.exp_target);
            return false;
        }
        if (in.exp_target == 12) {
            // POS0 is the mandatory x/y/z/w position vector. POS1..POS4 carry ancillary position
            // data (clip/cull distances, point size, viewport/layer selection according to the
            // programmed position format) and must never be mistaken for gl_Position merely because
            // an NGG shader emits one before POS0. Until those built-ins are modeled, retain the
            // existing deliberate behavior of ignoring them.
            // A position export must supply all four components (EN=0xF); a partial POS0 is not
            // meaningfully completable in the current model, so reject rather than invent components.
            if (in.exp_en != 0xFu) {
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "partial vertex position export pc=%u en=0x%x",
                                         in.pc, in.exp_en);
                return false;
            }
            uint32_t x = operand_bits(b, state, in, in.src[0], &eok);
            uint32_t y = operand_bits(b, state, in, in.src[1], &eok);
            uint32_t z = operand_bits(b, state, in, in.src[2], &eok);
            uint32_t w = operand_bits(b, state, in, in.src[3], &eok);
            if (terminal_ngg_output && state.exec_narrowed) {
                // The exact compacted-output wrapper emits a contiguous prefix of complete
                // primitives. Map every inactive suffix invocation to one identical clip point;
                // primitives assembled solely from that suffix are therefore degenerate instead of
                // accidentally reusing the last active vertex. Keep the real values on the true path.
                const uint32_t zero = b.uconst(0);
                x = b.sel(state.exec, x, zero);
                y = b.sel(state.exec, y, zero);
                z = b.sel(state.exec, z, zero);
                w = b.sel(state.exec, w, b.uconst(fbits(1.0f)));
            }
            b.export_position(x, y, z, w);
            exported = true;
        } else if (in.exp_target >= 32) {                    // PARAM0.. -> remapped PS input varying
            const uint32_t source = in.exp_target - 32;
            // EN gates which channels the export sends (vec2/vec3 varyings use EN=0x3/0x7):
            // hardware leaves disabled channels unwritten (undefined for the PS). Substitute a
            // deterministic 0.0 for them instead of exporting stale VGPR data.
            uint32_t x = (in.exp_en & 1u) ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0);
            uint32_t y = (in.exp_en & 2u) ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0);
            uint32_t z = (in.exp_en & 4u) ? operand_bits(b, state, in, in.src[2], &eok) : b.uconst(0);
            uint32_t w = (in.exp_en & 8u) ? operand_bits(b, state, in, in.src[3], &eok) : b.uconst(0);
            if (terminal_ngg_output && state.exec_narrowed) {
                const uint32_t zero = b.uconst(0);
                x = b.sel(state.exec, x, zero); y = b.sel(state.exec, y, zero);
                z = b.sel(state.exec, z, zero); w = b.sel(state.exec, w, zero);
            }
            const uint32_t gx = x, gy = y, gz = z, gw = w;
            // #2945: skip a slot the fragment program never reads. See PixelInputMapping::consumes
            // -- SPI_PS_INPUT_CNTL is sticky, so `valid_mask` alone fans one export out to 32
            // locations and overruns maxVertexOutputComponents.
            if (!pixel_inputs || source >= 32 || !(pixel_inputs->valid_mask & (1u << source))) {
                if (!pixel_inputs || pixel_inputs->consumes(source))
                    b.export_param(source, gx, gy, gz, gw);   // absent control retains identity wiring
            }
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    if (!pixel_inputs->consumes(ps_input)) continue;
                    const uint32_t raw_offset = pixel_inputs->controls[ps_input] & 0x3Fu;
                    const uint32_t offset = (passthrough_mask & (1u << ps_input))
                        ? (raw_offset & 0x1fu) : raw_offset;
                    if (offset == source) b.export_param(ps_input, gx, gy, gz, gw);
                }
            }
        }
        if (!eok && getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[vertex-export-reject] pc=%u target=%u unresolved export operand\n",
                         in.pc, in.exp_target);
        return eok;
    };
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                   /*allow_smem*/rt != nullptr, exp_fn, code, dwords)) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-recompile-reject] body or export lowering failed\n");
        return {};
    }
    if (passthrough && passthrough->valid) {
        const auto base_it = rs.vreg.find(static_cast<int>(passthrough->producer_base_vgpr));
        if (base_it == rs.vreg.end() || !b.vertex_lds_dwords) {
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[vertex-ngg-passthrough-reject] missing producer LDS base/allocation\n");
            return {};
        }
        b.declare_lds();
        const uint32_t base_dword = b.ibin(
            Op_ShiftRightLogical, base_it->second, b.uconst(2u));
        auto record_load = [&](int32_t dword) -> uint32_t {
            return b.lds_load(dword == 0
                ? base_dword
                : b.ibin(Op_IAdd, base_dword, b.uconst(static_cast<uint32_t>(dword))));
        };
        b.export_position(record_load(passthrough->position[0]),
                          record_load(passthrough->position[1]),
                          record_load(passthrough->position[2]),
                          record_load(passthrough->position[3]));
        exported = true;

        for (uint32_t source = 0; source < passthrough->params.size(); ++source) {
            if (!(passthrough->param_mask & (1u << source))) continue;
            std::array<uint32_t, 4> value{};
            for (uint32_t component = 0; component < 4; ++component) {
                const int32_t dword = passthrough->params[source][component];
                value[component] = dword >= 0 ? record_load(dword) : b.uconst(0u);
            }
            if (!pixel_inputs || !(pixel_inputs->valid_mask & (1u << source))) {
                if (!pixel_inputs || pixel_inputs->consumes(source))   // #2945
                    b.export_param(source, value[0], value[1], value[2], value[3]);
            }
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    if (!pixel_inputs->consumes(ps_input)) continue;    // #2945
                    if ((pixel_inputs->controls[ps_input] & 0x3fu) == source)
                        b.export_param(ps_input, value[0], value[1], value[2], value[3]);
                }
            }
        }
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[vertex-ngg-passthrough] base=v%u stride=%u params=%08x\n",
                         passthrough->producer_base_vgpr,
                         passthrough->record_stride_bytes, passthrough->param_mask);
    }
    if (!exported) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-recompile-reject] shader emitted no POS0\n");
        return {};
    }
    // OFFSET=0x20 asks the interpolator to synthesize a constant instead of consuming a PARAM
    // export. GFX10 DEFAULT_VAL encodes 0000, 0001, 1110, and 1111. Materialize those outputs in
    // the Vulkan vertex stage, whose fixed-function interface has no equivalent default source.
    if (pixel_inputs) {
        for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
            if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
            if (!pixel_inputs->consumes(ps_input)) continue;            // #2945
            const uint32_t control = pixel_inputs->controls[ps_input];
            if ((passthrough_mask & (1u << ps_input)) ||
                (control & 0x3Fu) != 0x20u) continue;
            const uint32_t one = b.uconst(0x3F800000u), zero = b.uconst(0u);
            const uint32_t default_val = (control >> 8) & 0x3u;
            const bool xyz_one = (default_val & 0x2u) != 0;
            const bool w_one = (default_val & 0x1u) != 0;
            b.export_param(ps_input, xyz_one ? one : zero, xyz_one ? one : zero,
                           xyz_one ? one : zero, w_one ? one : zero);
        }
        // A PS can read a PARAM that this VS never exports (#3416: a position-only guest VS
        // paired with a PS reading attr0). Vulkan requires the matching declaration even when
        // its value is unwritten. Complete only the proven consumed interface; do not invent a
        // store/default or expand sticky controls into 32 outputs. Real exports and explicit
        // DEFAULT_VAL controls above still supply their values through the same variables.
        if (pixel_inputs->consumed_known) {
            for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input)
                if (pixel_inputs->consumes(ps_input)) b.vtx_output(ps_input);
        }
    }
    return b.finish();
}

std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords,
                                       const ShaderResourceTable* rt,
                                       const PixelInputMapping* pixel_inputs,
                                       bool capture_position,
                                       uint32_t virtual_lds_dwords,
                                       RecompileDiagnosticContext diagnostic) {
    return recompile_vertex_impl(code, dwords, rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false, diagnostic);
}

std::vector<uint32_t> recompile_vertex_terminal_ngg_gate_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, true, false,
                                 {RecompileDiagnosticStage::Vertex, 0});
}

std::vector<uint32_t> recompile_vertex_ngg_one_lane_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, false, true,
                                 {RecompileDiagnosticStage::Vertex, 0});
}

std::vector<uint32_t> recompile_vertex_chain(const uint32_t* prolog, size_t prolog_dwords,
                                             const uint32_t* main, size_t main_dwords,
                                             const ShaderResourceTable* rt,
                                             const PixelInputMapping* pixel_inputs,
                                             bool capture_position,
                                             uint32_t virtual_lds_dwords,
                                             RecompileDiagnosticContext diagnostic) {
    const VertexPrologInfo info = rdna2_vertex_prolog_info(prolog, prolog_dwords);
    if (!info.valid || !main || !main_dwords) return {};

    const size_t main_span = rdna2_recompile_code_span(main, main_dwords);
    if (!main_span || info.prefix_dwords > SIZE_MAX - main_span) return {};
    const NggPassthroughLayout passthrough =
        analyze_ngg_passthrough(prolog, info.prefix_dwords, main, main_span);
    if (passthrough.valid) {
        return recompile_vertex_impl(prolog, info.prefix_dwords, rt, pixel_inputs,
                                     capture_position, virtual_lds_dwords, &passthrough, false,
                                     false, diagnostic);
    }
    std::vector<uint32_t> linked;
    linked.reserve(info.prefix_dwords + main_span);
    linked.insert(linked.end(), prolog, prolog + info.prefix_dwords);
    linked.insert(linked.end(), main, main + main_span);
    return recompile_vertex_impl(linked.data(), linked.size(), rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false, diagnostic);
}

} // namespace prosper::gpu
