// FLAT case routes moved verbatim from emit_alu.
#include "gpu/recompiler/ir/lower/flat_case_routes.hpp"
#include "gpu/recompiler/ir/lower/flat_window_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <optional>

namespace prosper::gpu::ir {
std::optional<bool> FlatCaseRoutes::relocated() {
    const IndirectPointerAccessProof* relocated_access =
        b.indirect_pointer_proof ? rdna2_indirect_pointer_access(*b.indirect_pointer_proof, in)
                                 : nullptr;
    if (relocated_access) {
        const bool static_footprint =
            b.indirect_pointer_proof->bound_kind == IndirectPointerBoundKind::StaticFootprint &&
            b.indirect_pointer_proof->guard_kind == IndirectPointerGuardKind::Full64NonZero;
        const bool descriptor_range =
            b.indirect_pointer_proof->bound_kind == IndirectPointerBoundKind::DescriptorRange &&
            b.indirect_pointer_proof->guard_kind == IndirectPointerGuardKind::None;
        const uint64_t directory_end =
            static_cast<uint64_t>(b.indirect_pointer_segment_directory_byte_offset) +
            static_cast<uint64_t>(b.indirect_pointer_segment_count) *
                kIndirectBufferRelocationSegmentBytes;
        const uint64_t record_directory_end =
            static_cast<uint64_t>(b.indirect_pointer_record_directory_byte_offset) +
            static_cast<uint64_t>(b.indirect_pointer_record_count) *
                kIndirectBufferRelocationRecordBytes;
        if (!access.valid || access.store || access.bits != 32u || access.components != 1u ||
            in.flat_segment != 2u || in.flat_lds || in.flat_glc || in.flat_slc || in.flat_dlc ||
            in.dst.kind != OperandKind::VGPR || in.dst.value < 0 ||
            in.src[0].kind != OperandKind::VGPR || in.src[0].value < 0 ||
            static_cast<uint32_t>(in.src[0].value) != relocated_access->address_vgpr ||
            relocated_access->address_vgpr >= 255u || in.src[1].kind != OperandKind::Special ||
            in.src[1].value != 125 || in.literal != relocated_access->immediate_byte_offset ||
            relocated_access->component_bytes != sizeof(uint32_t) ||
            relocated_access->components != 1u || b.indirect_pointer_proof->schema_version != 1u ||
            (!static_footprint && !descriptor_range) || b.indirect_pointer_binding == UINT32_MAX ||
            b.indirect_pointer_segment_directory_byte_offset % sizeof(uint32_t) != 0u ||
            b.indirect_pointer_payload_byte_offset % sizeof(uint32_t) != 0u ||
            b.indirect_pointer_carrier_bytes % sizeof(uint32_t) != 0u ||
            b.indirect_pointer_carrier_bytes < sizeof(uint32_t) ||
            directory_end > b.indirect_pointer_payload_byte_offset ||
            b.indirect_pointer_payload_byte_offset > b.indirect_pointer_carrier_bytes) {
            ok = false;
            return true;
        }
        if (descriptor_range &&
            (!b.indirect_pointer_record_count || !b.indirect_pointer_source_stride ||
             !b.indirect_pointer_source_record_var || !b.indirect_pointer_source_root_lo_var ||
             !b.indirect_pointer_source_root_hi_var ||
             b.indirect_pointer_record_directory_byte_offset % sizeof(uint32_t) != 0u ||
             b.indirect_pointer_record_directory_byte_offset < b.indirect_pointer_source_bytes ||
             record_directory_end != b.indirect_pointer_segment_directory_byte_offset ||
             b.indirect_pointer_proof->source_address_kind !=
                 IndirectBufferRelocationRecord::SourceAddressKind::BufferDescriptorBase48)) {
            ok = false;
            return true;
        }
        const int address_vgpr = static_cast<int>(relocated_access->address_vgpr);
        const uint32_t address_lo = vreg_old(b, rs, address_vgpr);
        const uint32_t address_hi = vreg_old(b, rs, address_vgpr + 1);
        const uint32_t value =
            descriptor_range ? b.relocated_indirect_descriptor_load_dword(
                                   address_lo, address_hi, relocated_access->immediate_byte_offset)
                             : b.relocated_indirect_load_dword(
                                   address_lo, address_hi, relocated_access->immediate_byte_offset);
        const int reg = in.dst.value;
        const uint32_t old_value = vreg_old(b, rs, reg);
        rs.vreg[reg] = value;
        predicate_write(b, rs, reg, old_value);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> FlatCaseRoutes::packed() {
    IndirectBufferShadowAccess packed_access{};
    if (b.indirect_buffer_dispatch_validated &&
        rdna2_gta5_packed_pointer_access(in, packed_access)) {
        if (!access.valid || access.store || access.bits != 32u || in.flat_lds ||
            in.src[0].kind != OperandKind::VGPR || in.src[0].value != 0 ||
            in.src[1].kind != OperandKind::Special || in.src[1].value != 125 ||
            access.components != packed_access.components ||
            b.indirect_buffer_binding == UINT32_MAX || b.indirect_buffer_slot_count == 0u ||
            !b.indirect_buffer_contract_tag || !b.indirect_buffer_header_bytes ||
            !b.indirect_buffer_slot_bytes) {
            ok = false;
            return true;
        }
        const uint32_t address_lo = vreg_old(b, rs, 0);
        const uint32_t address_hi = vreg_old(b, rs, 1);
        const uint32_t relative =
            b.ibin(Op_ISub, address_lo,
                   b.uconst(b.indirect_buffer_source_bytes + b.indirect_buffer_header_bytes));
        const uint32_t aligned =
            b.ucmp(Op_IEqual, b.ibin(Op_UMod, relative, b.uconst(b.indirect_buffer_slot_bytes)),
                   b.uconst(0));
        const uint32_t slot = b.ibin(Op_UDiv, relative, b.uconst(b.indirect_buffer_slot_bytes));
        uint32_t valid = b.ucmp(Op_IEqual, address_hi, b.uconst(b.indirect_buffer_contract_tag));
        valid = b.land(valid, aligned);
        valid = b.land(valid, b.ucmp(Op_ULessThan, slot, b.uconst(b.indirect_buffer_slot_count)));
        for (uint32_t component = 0; component < packed_access.components; ++component) {
            const uint32_t byte_address =
                b.ibin(Op_IAdd, address_lo,
                       b.uconst(packed_access.byte_offset + component * sizeof(uint32_t)));
            // OpSelect is not short-circuiting: select a safe address before OpLoad, then
            // separately force architectural zero for a malformed runtime tag.
            const uint32_t safe_byte = b.sel(valid, byte_address, b.uconst(0));
            const uint32_t index = b.ibin(Op_ShiftRightLogical, safe_byte, b.uconst(2));
            const uint32_t loaded = b.cbuf_load(index, b.indirect_buffer_binding);
            const uint32_t value = b.sel(valid, loaded, b.uconst(0));
            const int reg = in.dst.value + static_cast<int>(component);
            const uint32_t old_value = vreg_old(b, rs, reg);
            rs.vreg[reg] = value;
            predicate_write(b, rs, reg, old_value);
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> FlatCaseRoutes::window() {
    // General (non-scratch) FLAT LOAD from a raw 64-bit guest address (#1171). If the executor
    // resolved this load's base pointer to user SGPRs s[flat_base_sgpr : +1] and bound the
    // containing guest allocation as an SSBO (a ConstantBuffer-class resource keyed by this
    // load's pc), lower it to an indexed read of that window at byte offset (address_lo -
    // base_lo). base_lo is the ORIGINAL push-constant value (the address the executor bound the
    // window at), so the module stays dispatch-independent and correct even if the shader later
    // reuses the base SGPR. Unresolved forms fall through to the scratch/reject path below.
    const ShaderResource* fw = (rt && access.valid && !access.store && !in.flat_lds &&
                                in.src[0].kind == OperandKind::VGPR &&
                                in.src[1].kind == OperandKind::Special && in.src[1].value == 125)
                                   ? rt->by_fetch_pc(in.pc)
                                   : nullptr;
    if (fw && fw->flat_base_sgpr != 0xFFFFFFFFu) {
        const uint32_t addr_lo = val(in.src[0]);   // low dword of the address
        const uint32_t base_lo = b.load_push_constant(fw->flat_base_sgpr);
        // Byte offset in the window: (address - base) mod 2^32, which equals the true offset for
        // any 0 <= offset < window <= 256 MiB (the low-dword subtraction wraps complementarily to
        // the address's own IAdd). A negative offset (address < base) becomes a huge unsigned index
        // -> out-of-window -> robustBufferAccess returns 0 (defined, but a loose-bounds divergence
        // from HW; decode kernels use non-negative offsets).
        const uint32_t byte0 = ir::lower_flat_window_byte_offset(b, addr_lo, base_lo);
        for (uint32_t c = 0; c < access.components; ++c) {
            const auto [addr, idx] = ir::lower_flat_window_component(b, byte0, access, c);
            uint32_t value;
            if (access.bits == 32) {
                value = b.cbuf_load(idx, fw->binding);
            } else {
                // Sub-dword (ubyte/ushort) extract, mirroring the MUBUF raw path: a 16-bit
                // access may begin at byte 3 and straddle two dwords, so join the adjacent words.
                const uint32_t dw0 = b.cbuf_load(idx, fw->binding);
                const uint32_t byte_in_dw = b.ibin(Op_BitwiseAnd, addr, b.uconst(3));
                const uint32_t shift = b.ibin(Op_ShiftLeftLogical, byte_in_dw, b.uconst(3));
                uint32_t joined = b.ibin(Op_ShiftRightLogical, dw0, shift);
                if (access.bits == 16) {
                    const uint32_t dw1 =
                        b.cbuf_load(b.ibin(Op_IAdd, idx, b.uconst(1)), fw->binding);
                    const uint32_t inv_shift =
                        b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
                    const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
                    joined =
                        b.ibin(Op_BitwiseOr, joined,
                               b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
                }
                value = access.sign_extend ? b.bfe_s(joined, b.uconst(0), b.uconst(access.bits))
                                           : b.bfe_u(joined, b.uconst(0), b.uconst(access.bits));
            }
            const int reg = in.dst.value + static_cast<int>(c);
            const uint32_t old_value = vreg_old(b, rs, reg);
            rs.vreg[reg] = value;
            predicate_write(b, rs, reg, old_value);
        }
        return true;
    }
    return std::nullopt;
}

bool FlatCaseRoutes::scratch() {
    // Model the compiler's private spill area only. Each generated shader invocation owns
    // one Function-storage array, so the entry-provided hardware base has no host-visible
    // address to preserve. Other FLAT/GLOBAL forms remain deliberately unsupported.
    if (!access.valid || in.flat_segment != 1u || in.flat_lds || !b.guest_scratch ||
        in.src[0].kind != OperandKind::None || in.src[1].kind != OperandKind::SGPR ||
        in.src[1].value != b.guest_scratch_saddr || rs.sreg_written.count(in.src[1].value)) {
        ok = false;
        return true;
    }
    const int32_t base = static_cast<int32_t>(in.literal);
    const int64_t storage_begin = b.guest_scratch_min_byte;
    const int64_t storage_end = storage_begin + static_cast<int64_t>(b.guest_scratch_dwords) * 4;
    for (uint32_t component = 0; component < access.components; ++component) {
        const int32_t byte_offset = base + static_cast<int32_t>(component * (access.bits / 8u));
        const int64_t access_end = static_cast<int64_t>(byte_offset) + access.bits / 8u;
        if (byte_offset < storage_begin || access_end > storage_end) {
            ok = false;
            return true;
        }
        const int reg = in.dst.value + static_cast<int>(component);
        if (access.store) {
            b.guest_scratch_store_bits(byte_offset, access.bits, vreg_old(b, rs, reg),
                                       rs.exec_narrowed, rs.exec);
        } else {
            const uint32_t old_value = vreg_old(b, rs, reg);
            rs.vreg[reg] = b.guest_scratch_load_bits(byte_offset, access.bits, access.sign_extend);
            predicate_write(b, rs, reg, old_value);
        }
    }
    return true;
}

bool FlatCaseRoutes::lower() {
    if (auto routed = relocated()) return *routed;
    if (auto routed = packed()) return *routed;
    if (auto routed = window()) return *routed;
    return scratch();
}
}   // namespace prosper::gpu::ir
