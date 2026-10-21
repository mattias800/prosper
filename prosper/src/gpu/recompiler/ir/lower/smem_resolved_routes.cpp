// Pure textual relocation: tools/refactor/extract_case_routes.py.
#include "gpu/recompiler/ir/lower/smem_case_routes.hpp"
#include "gpu/recompiler/ir/lower/smem_index_lowering.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <optional>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
// NOLINTBEGIN(concurrency-mt-unsafe): existing diagnostic reads retain their timing.
std::optional<bool> SmemCaseRoutes::resolved(uint32_t n) {
    if (!allow_smem) {
        if (getenv("PROSPER_DBG"))
            fprintf(stderr, "[smem-reject] pc=%u reason=graphics-disabled op=0x%x\n", in.pc,
                    in.opcode);
        ok = false;
        return true;
    }
    // GTA V 0x413ce6000 partitions the active wave by a read-first-lane selector, multiplies
    // that selector by the outer V# stride, then reads one four-dword inner V# at pc153.
    // The front half has proved, for this exact dispatch, that every post-MUL SOFFSET is
    // either the sole reachable in-bounds record (480 bytes, record 4) or wholly outside the
    // 600-byte scalar-buffer range. Materialize the architectural per-dword SMEM result here:
    // selected record words for 480, zero otherwise. The complete-dispatch authority bit is
    // deliberately required in addition to the serialized marker shape.
    const ShaderResource* selected_sbuffer =
        rt && in.opcode == kSmemOpcodeBufferLoadDwordX4 ? rt->by_fetch_pc(in.pc) : nullptr;
    const bool exact_gta5_selected_sbuffer_site =
        in.pc == 153u && in.fmt == Rdna2Format::SMEM && in.len_dwords == 2u &&
        in.dst.kind == OperandKind::SGPR && in.dst.value == 8 &&
        in.src[0].kind == OperandKind::SGPR && in.src[0].value == 4 &&
        in.src[1].kind == OperandKind::Special && in.src[1].value == 106 && in.literal == 8u &&
        in.words[0] == 0xf4280202u && in.words[1] == 0xd4000008u;
    if (selected_sbuffer && is_gta5_selected_sbuffer_marker_candidate(*selected_sbuffer)) {
        if (!b.gta5_selected_sbuffer_dispatch_validated ||
            !is_gta5_selected_sbuffer_descriptor(*selected_sbuffer) ||
            !exact_gta5_selected_sbuffer_site || n != 4u) {
            ok = false;
            return true;
        }
        if (selected_sbuffer->selected_sbuffer_soffset == kGtaSelectedSbufferZeroChainSoffset ||
            selected_sbuffer->selected_sbuffer_soffset == kGtaSelectedSbufferAllOobSoffset ||
            selected_sbuffer->selected_sbuffer_soffset == kGtaSelectedSbufferNullRecord4Soffset) {
            for (uint32_t k = 0; k < n; ++k) {
                rs.sreg[in.dst.value + static_cast<int>(k)] = b.uconst(0u);
                rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
            }
            return true;
        }
        const auto soff = rs.sreg.find(in.src[1].value);
        if (soff == rs.sreg.end()) {
            ok = false;
            return true;
        }
        const uint32_t selected =
            b.ucmp(Op_IEqual, soff->second, b.uconst(selected_sbuffer->selected_sbuffer_soffset));
        for (uint32_t k = 0; k < n; ++k) {
            rs.sreg[in.dst.value + static_cast<int>(k)] = b.sel(
                selected, b.uconst(selected_sbuffer->selected_sbuffer_words[k]), b.uconst(0u));
            rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
        }
        return true;
    }
    // The front half proved all four live SBASE words at this exact scalar-buffer load and
    // proved that its minimum unsigned offset begins beyond the effective scalar bound. The
    // result is therefore exact zero without declaring or touching a dummy storage binding.
    // Check this before SOFFSET tracking: even an unknown runtime offset cannot reduce it.
    const ShaderResource* zero_record_sbuffer =
        rt && in.opcode >= 0x8u ? rt->by_fetch_pc(in.pc) : nullptr;
    if (zero_record_sbuffer && is_zero_record_raw_buffer(*zero_record_sbuffer)) {
        for (uint32_t k = 0; k < n; ++k) {
            rs.sreg[in.dst.value + static_cast<int>(k)] = b.uconst(0);
            rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
        }
        return true;
    }
    if (emit_owned_raw_x2(b, rs, in, rt, ok)) return true;
    // The whole-CFG proof established that this register-offset or aligned-immediate
    // S_LOAD_DWORDX2 feeds only a complete V# (possibly after an exact scalar carry-pair
    // relocation), never ordinary scalar/address or mask state. Every descriptor observation
    // resolves through an exact-PC resource, so no runtime load belongs in the emitted module.
    // Keep this before generic SOFFSET handling: the offset selects front-half provenance,
    // not emitted scalar data.
    if (rt && in.opcode == kSmemOpcodeLoadDwordX2 &&
        rs.smem_x2_descriptor_fragment_loads.contains(in.pc)) {
        for (uint32_t k = 0; k < n; ++k) {
            rs.sreg[in.dst.value + static_cast<int>(k)] = b.uconst(0);
            rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
        }
        return true;
    }
    // The whole-stream proof established that this S_LOAD_DWORDX2 loads a DESCRIPTOR-TABLE
    // POINTER -- its only use anywhere is as the SBASE of another raw scalar load -- so the
    // value it produces is provenance, not scalar data. The front half followed the same
    // pointer out of guest memory and published every descriptor behind it at its exact
    // consumer PC, so zero placeholders represent it exactly. The SRT tag is ERASED rather
    // than propagated: the pointer's own offset names the pointer, not the descriptor a
    // later load fetches through it, and leaving it would resolve that load to the wrong
    // resource. Without this the load reached the constant-buffer path below and rejected
    // the shader for an unresolved cbuf (#3616).
    if (rt && in.opcode == kSmemOpcodeLoadDwordX2 && rs.smem_pointer_loads.contains(in.pc)) {
        for (uint32_t k = 0; k < n; ++k) {
            rs.sreg[in.dst.value + static_cast<int>(k)] = b.uconst(0);
            rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
        }
        return true;
    }
    return std::nullopt;
}
// NOLINTEND(concurrency-mt-unsafe)
} // namespace prosper::gpu::ir
