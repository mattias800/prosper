// Checked relocation by tools/refactor/extract_case_routes.py; original route order is retained.
#include "gpu/recompiler/ir/lower/ds_case_lowering.hpp"
#include "gpu/recompiler/ir/lower/ds_cross_lane_lowering.hpp"
#include "gpu/recompiler/ir/lower/ds_address_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
namespace prosper::gpu::ir {
// NOLINTBEGIN(concurrency-mt-unsafe): existing diagnostic read retains its timing.
bool lower_ds_case(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    if (auto routed = ir::lower_ds_cross_lane(b, rs, in, ok)) return *routed;
    // GDS=1 targets the device-global data share, not workgroup LDS — running it against
    // our LDS model would silently give per-workgroup semantics. Reject — EXCEPT
    // ds_append/ds_consume (0x3e/0x3d): Astro Bot's live counter words carry GDS=1
    // (0xd8fa/0xd8f6 — llvm-mc gfx1030 round-trip: plain append is 0xd8f8), and the
    // existing wave_append model was landed and live-validated against exactly those
    // packets (#554/#580). Kept as a documented per-workgroup approximation of the
    // device-global counter (exact for the exercised dispatch shapes). CONFIDENCE: MED.
    if (b.is_compute && in.ds_gds && in.opcode == 0x36)   // the plain GDS read (#4553)
        return emit_compute_gds_read(b, rs, in);
    if (in.ds_gds && in.opcode != 0x3d && in.opcode != 0x3e &&
        !(b.is_compute && in.opcode == 0x0d)) {
        ok = false;
        return true;
    }
    // ds_write_addtid_b32 (0xb0) / ds_read_addtid_b32 (0xb1). AMD RDNA2 ISA 10.4:
    //   LDS address = LDS_BASE + immediate + M0[15:0] + TID(0..63)*4.
    // TID is the lane within the hardware wave, not the workgroup-linear invocation id.
    // Compute therefore uses real Workgroup LDS and wraps only the lane id at wave_size;
    // M0 supplies a distinct base when a workgroup contains more than one wave. The final
    // address itself does not wrap (RDNA2 LDS allocations explicitly do not wrap).
    // Astro Bot's world-map material kernel uses several such lane arrays before reducing
    // them through explicit DS reads and barriers.
    if (b.is_compute && (in.opcode == 0xb0 || in.opcode == 0xb1)) {
        auto m0 = rs.sreg.find(124);
        if (m0 == rs.sreg.end()) {
            ok = false;
            return true;
        }
        b.declare_lds();
        if (!b.lds_var) {
            ok = false;
            return true;
        }
        const uint32_t base = b.ibin(Op_BitwiseAnd, m0->second, b.uconst(0xffffu));
        const uint32_t tid = b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1u));
        const uint32_t byte_addr = b.ibin(Op_IAdd, b.ibin(Op_IAdd, base, b.uconst(in.literal)),
                                          b.ibin(Op_ShiftLeftLogical, tid, b.uconst(2)));
        const uint32_t idx = b.ibin(Op_ShiftRightLogical, byte_addr, b.uconst(2));
        if (in.opcode == 0xb0) {
            auto value = rs.vreg.find(in.src[1].value);
            b.lds_store(idx, value == rs.vreg.end() ? b.uconst(0) : value->second, rs.exec_narrowed,
                        rs.exec, b.atomicized_lds_store_pcs.contains(in.pc));
        } else {
            const uint32_t old = vreg_old(b, rs, in.dst.value);
            rs.vreg[in.dst.value] = b.lds_load(idx, rs.exec_narrowed, rs.exec, old);
        }
        return true;
    }
    // In a GRAPHICS stage (#273), ADDTID is a compiler spill/reload through per-wave LDS:
    // a per-lane VGPR spill through LDS (addr = M0 + offset + tid*4) — DOLL's title post
    // PSes spill v15 before their accumulation loop and reload it after. Per-invocation the
    // slot is ONE value: track it in rs.lds_addtid keyed by (M0 SSA id, inst offset); the
    // matching read returns the spilled SSA value. A write with UNTRACKED M0 still no-ops
    // (nothing in this model can observe it) but poisons nothing; a read with untracked M0
    // or a never-written slot rejects loudly. VERIFIED(round-trip llvm-mc gfx1010:
    // 0xdac00000/0x00000f00 -> ds_write_addtid_b32 v15; llvm-mc gfx1030:
    // 0xdac40000/0x0f000000 -> ds_read_addtid_b32 v15).
    if (in.opcode == 0xb0 || in.opcode == 0xb1) {
        auto m0 = rs.sreg.find(124);
        if (in.opcode == 0xb0) {
            if (m0 != rs.sreg.end()) record_addtid_spill(b, rs, in, m0->second);
            return true;   // spill: unobservable beyond the tracked slot
        }
        if (m0 == rs.sreg.end()) {
            ok = false;
            return true;
        }
        auto slot = rs.lds_addtid.find(((uint64_t)m0->second << 32) | in.literal);
        if (slot == rs.lds_addtid.end()) {
            ok = false;
            return true;
        }
        uint32_t old = vreg_old(b, rs, in.dst.value);
        rs.vreg[in.dst.value] = slot->second;
        predicate_write(b, rs, in.dst.value, old);
        return true;
    }
    if (in.opcode == 0x3d || in.opcode == 0x3e) {
        auto m0 = rs.sreg.find(124);
        if (!allow_wave || m0 == rs.sreg.end()) {
            ok = false;
            return true;
        }
        const uint32_t idx = ds_append_consume_index(b, m0->second, in.literal, in.ds_gds);
        const uint32_t old = vreg_old(b, rs, in.dst.value);
        if (b.is_fragment && in.ds_gds) {
            rs.vreg[in.dst.value] = b.native_gds_append(idx, rs.exec, in.opcode == 0x3d);
        } else if (b.is_compute) {
            if (!in.ds_gds) b.declare_lds();
            rs.vreg[in.dst.value] = b.wave_append(idx, rs.exec, in.opcode == 0x3d, in.ds_gds);
        } else {
            ok = false;
            return true;
        }
        predicate_write(b, rs, in.dst.value, old);
        return true;
    }
    // Compute uses Workgroup LDS. Only a mechanically proven NGG projection may use the
    // private graphics backing store; every other graphics DS shape remains fail-closed.
    if ((!b.is_compute && !b.ngg_private_lds) ||
        (in.opcode != 0x00 && in.opcode != 0x05 && in.opcode != 0x06 && in.opcode != 0x07 &&
         in.opcode != 0x08 && in.opcode != 0x09 && in.opcode != 0x0a && in.opcode != 0x0b &&
         in.opcode != 0x12 && in.opcode != 0x13 && in.opcode != 0x20 && in.opcode != 0x2d &&
         in.opcode != 0x0d && in.opcode != 0x0e && in.opcode != 0x0f && in.opcode != 0x38 &&
         in.opcode != 0x36 && in.opcode != 0x37 && in.opcode != 0x3d && in.opcode != 0x3e &&
         in.opcode != 0x4d && in.opcode != 0x4e && in.opcode != 0x76 && in.opcode != 0x77 &&
         in.opcode != 0xde && in.opcode != 0xdf && in.opcode != 0xfe && in.opcode != 0xff)) {
        ok = false;
        return true;
    }
    b.declare_lds();
    if (!b.lds_var) {
        ok = false;
        return true;
    }
    auto vread = [&](int r) { return read_vector_for_arbitrary_retry(b, rs, in, r, ok); };
    const bool atomicize_store = b.atomicized_lds_store_pcs.contains(in.pc);
    // The st64 forms multiply their 8-bit offsets by 64, so their CONSTANT index alone
    // reaches 255*64 = 16320 dwords where every other DS form stops at a few hundred. The
    // LDS array is sized from the shader's real allocation when the caller plumbed one and
    // otherwise keeps a 4096-dword default (`rdna2_to_spirv_internal.hpp`,
    // `rdna2_recompile_compute.cpp`), so an unplumbed shader can put a constant index past
    // the end -- an out-of-bounds Workgroup OpAccessChain, which is undefined behaviour and
    // is NOT covered by robustBufferAccess. Fail visibly instead. This can only reject
    // programs that rejected outright before these opcodes were implemented, so it costs
    // nothing that worked, and it is a bound on the constant part only: the base address is
    // a runtime VGPR and was never statically bounded for any DS opcode.
    //
    // The limit MUST be read the same way `declare_lds()` sizes the array --
    // `is_compute ? lds_dwords : vertex_lds_dwords` -- and not from `lds_dwords` alone.
    // `lds_dwords` is only ever assigned on the compute path, so testing it unconditionally
    // would compare an NGG vertex shader's index against an unrelated 4096 default while its
    // array is `vertex_lds_dwords` wide: a guard that is absent on exactly the path this
    // change newly opens. Caught in review, and it is the same shape as the defect the guard
    // exists to stop.
    const uint32_t lds_limit = b.is_compute ? b.lds_dwords : b.vertex_lds_dwords;
    auto lds_const_index_in_range = [&](uint32_t index) {
        if (index < lds_limit) return true;
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[ds-st64-oob] pc=%u op=0x%x const_index=%u limit=%u stage=%s\n",
                         in.pc, in.opcode, index, lds_limit, b.is_compute ? "compute" : "vertex");
        return false;
    };
    if (in.opcode == 0x0e) {                    // ds_write2_b32: two dwords at offset0/offset1
        // AMD RDNA2 ISA 12.13: MEM[ADDR + OFFSET0/1 * 4] = DATA0/1. The packed offsets
        // mirror ds_read2_b32 below; Astro Bot's world-map reduction uses both operations.
        const auto address_pair = ir::lower_ds_write2_b32_address(b, in, std::cref(vread));
        const uint32_t idx0 = address_pair.first, idx1 = address_pair.second;
        b.lds_store(idx0, vread(in.src[1].value), rs.exec_narrowed, rs.exec, atomicize_store);
        // Equal offsets encode one memory access and use DATA0; DATA1 is ignored.
        if ((in.literal & 0xFFu) != ((in.literal >> 8) & 0xFFu))
            b.lds_store(idx1, vread(in.src[2].value), rs.exec_narrowed, rs.exec, atomicize_store);
        return true;
    }
    if (in.opcode == 0x0f) {                    // ds_write2st64_b32: offsets scaled by 64
        // AMD RDNA2 ISA 70648 §12.13: MEM[ADDR + OFFSET0/1 * 4 * 64] = DATA0/1. Identical
        // to ds_write2_b32 (0x0e) above except each 8-bit offset counts 64-dword strides
        // instead of single dwords, which is how a workgroup reduction lays one value per
        // 64-lane wave without the lanes colliding.
        //
        // Measured need (2026-09-17): Sonic Frontiers' Cyber Space stage. Three programs
        // use it and NOTHING else in them is unsupported -- `0x2005712000` (the exposure
        // reduction, 7 uses), `0x2005713000` (the pass whose output the final draws consume,
        // 7 uses) and `0x200581bb00` (22 uses). `tools/shader_inspect` generic coverage over
        // every dumped stage shader in that scene reports `DS op=0xf` as their only gap.
        //
        // The equal-offset guard is the same contract as 0x0e and 0x4e: OFFSET0 == OFFSET1
        // encodes ONE memory access which uses DATA0, so writing DATA1 as well would let the
        // later store win and a subsequent read observe DATA1 where hardware preserves
        // DATA0 -- silent wrong data rather than a fault (#1473).
        const uint32_t off0 = (in.literal & 0xFFu) * 64u;
        const uint32_t off1 = ((in.literal >> 8) & 0xFFu) * 64u;
        if (!lds_const_index_in_range(off0) || !lds_const_index_in_range(off1)) {
            ok = false;
            return true;
        }
        const auto address_pair =
            ir::lower_ds_write2st64_b32_address(b, in, std::cref(vread), off0, off1);
        const uint32_t idx0 = address_pair.first, idx1 = address_pair.second;
        b.lds_store(idx0, vread(in.src[1].value), rs.exec_narrowed, rs.exec, atomicize_store);
        if ((in.literal & 0xFFu) != ((in.literal >> 8) & 0xFFu))
            b.lds_store(idx1, vread(in.src[2].value), rs.exec_narrowed, rs.exec, atomicize_store);
        return true;
    }
    if (in.opcode == 0x4e) {   // ds_write2_b64: two VGPR pairs at offset0/offset1
        const auto address_pair = ir::lower_ds_write2_b64_address(b, in, std::cref(vread));
        const uint32_t idx0 = address_pair.first, idx1 = address_pair.second;
        b.lds_store(idx0, vread(in.src[1].value), rs.exec_narrowed, rs.exec, atomicize_store);
        b.lds_store(ir::lower_ds_store_word_index(b, idx0, 1), vread(in.src[1].value + 1),
                    rs.exec_narrowed, rs.exec, atomicize_store);
        // OFFSET0 == OFFSET1 selects ONE address: the hardware performs a single write and
        // uses DATA0 only (RDNA2 ISA 70648 §10.4.3). Writing DATA1 as well leaves the later
        // store winning, so a subsequent LDS read observes DATA1 where hardware preserves
        // DATA0 -- silent wrong data rather than a fault. The ds_write2_b32 case above
        // already guards this; the 64-bit variant did not (#1473).
        if ((in.literal & 0xFFu) != ((in.literal >> 8) & 0xFFu)) {
            b.lds_store(idx1, vread(in.src[2].value), rs.exec_narrowed, rs.exec, atomicize_store);
            b.lds_store(ir::lower_ds_store_word_index(b, idx1, 1), vread(in.src[2].value + 1),
                        rs.exec_narrowed, rs.exec, atomicize_store);
        }
        return true;
    }
    // NOTE: a SECOND ds_write2_b32 (0x0e) block used to sit here, unreachable because the
    // handler above claims 0x0e first and returns. It was a duplicate carrying the
    // PRE-FIX behaviour -- both stores, no equal-offset guard -- so the hazard was never
    // that it ran. It was that adding any condition to the reachable block (a wave-size
    // check, an encoding refinement) would have silently handed 0x0e to a copy with the
    // #1473 defect still in it, and the tests would have kept passing. Deleted (#1473).
    if (in.opcode == 0x77) {   // ds_read2_b64: two pairs at scaled offsets
        const auto indices = ir::lower_ds_read2_b64_address(b, in, std::cref(vread));
        for (int k = 0; k < 4; ++k) {
            const uint32_t old = vreg_old(b, rs, in.dst.value + k);
            rs.vreg[in.dst.value + k] = b.lds_load(indices[k], rs.exec_narrowed, rs.exec, old);
        }
        return true;
    }
    if (in.opcode == 0x37) {   // ds_read2_b32: two dwords at offset0/offset1
        // AMD RDNA2 ISA 12.13: RETURN_DATA[0/1] = MEM[ADDR + OFFSET0/1 * 4].
        // The two 8-bit offsets share the instruction's 16-bit offset field. Astro's
        // loading compositor uses both adjacent (0,1) and non-zero (16,17) pairs.
        const auto indices = ir::lower_ds_read2_b32_address(b, in, std::cref(vread));
        for (int k = 0; k < 2; ++k) {
            const uint32_t old = vreg_old(b, rs, in.dst.value + k);
            rs.vreg[in.dst.value + k] = b.lds_load(indices[k], rs.exec_narrowed, rs.exec, old);
        }
        return true;
    }
    if (in.opcode == 0x38) {   // ds_read2st64_b32: offsets scaled by 64
        // RETURN_DATA[0/1] = MEM[ADDR + OFFSET0/1 * 4 * 64]. The read sibling of 0x0f above,
        // and implemented with it rather than after it: a kernel that lays values out on
        // 64-dword strides reads them back the same way, so supporting only the write would
        // move the reject one instruction later in the same program. `0x200581bb00` in this
        // scene uses both (22 writes, 2 reads).
        //
        // No equal-offset guard here, deliberately: a read has no write hazard, and
        // ds_read2_b32 (0x37) above loads both slots unconditionally for the same reason.
        const uint32_t roff0 = (in.literal & 0xFFu) * 64u;
        const uint32_t roff1 = ((in.literal >> 8) & 0xFFu) * 64u;
        if (!lds_const_index_in_range(roff0) || !lds_const_index_in_range(roff1)) {
            ok = false;
            return true;
        }
        const auto indices =
            ir::lower_ds_read2st64_b32_address(b, in, std::cref(vread), roff0, roff1);
        for (int k = 0; k < 2; ++k) {
            const uint32_t old = vreg_old(b, rs, in.dst.value + k);
            rs.vreg[in.dst.value + k] = b.lds_load(indices[k], rs.exec_narrowed, rs.exec, old);
        }
        return true;
    }
    if (b.ngg_private_lds && in.opcode == 0x36 && in.pc == b.ngg_vertex_index_read_pc &&
        b.ngg_vertex_index_value) {
        // The merged NGG prologue distributes hardware vertex ids between wave lanes through
        // LDS, then the ES consumes that value as its first MUBUF index. In the one-invocation
        // Vulkan shell there is no peer-lane LDS routing: the equivalent value is precisely
        // BuiltIn VertexIndex. This handoff is identified in recompile_vertex from the first
        // MUBUF's vaddr reaching back to this DS read; all other vertex LDS reads remain real.
        const uint32_t old = vreg_old(b, rs, in.dst.value);
        rs.vreg[in.dst.value] = b.ngg_vertex_index_value;
        predicate_write(b, rs, in.dst.value, old);
        return true;
    }
    const uint32_t idx = ir::lower_ds_byte_address(b, in, std::cref(vread));
    if (in.ds_gds && in.opcode == 0x0d) {
        b.compute_gds_store(idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
        return true;
    }
    if (b.ngg_private_lds && (in.opcode == 0x00 || in.opcode == 0x07 || in.opcode == 0x08 ||
                              in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32 ||
                              in.opcode == 0x20 || in.opcode == 0x2d)) {
        // Cross-lane/atomic NGG LDS effects do not have a proven single-lane reduction yet.
        ok = false;
        return true;
    }
    if (in.opcode == 0x00) {   // ds_add_u32: LDS += DATA0, no VGPR return
        b.lds_atomic(Op_AtomicIAdd, idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
    } else if (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32) {
        // Exact ordinary GFX10 encoding: ADDR + DATA0 only. DATA1 and VDST are reserved for
        // these non-returning forms (llvm-mc gfx1030 emits both as zero); reject a packet
        // carrying either field instead of interpreting an unmodeled SRC2/return variant.
        if ((in.words[1] & 0xffff0000u) != 0u || b.compute_pgm_rsrc1 == UINT32_MAX) {
            ok = false;
            return true;
        }
        b.lds_atomic_fminmax(idx, vread(in.src[1].value), in.opcode == kDsOpcodeMinF32,
                             rs.exec_narrowed, rs.exec);
    } else if (in.opcode >= 0x05 && in.opcode <= 0x0b) {
        // Non-returning 32-bit LDS atomics map directly to SPIR-V atomics. DS_OR_B32
        // (0x0a) occurs in Plucky's first-gameplay compute path; supporting the adjacent
        // signed/unsigned min/max and bitwise family keeps this architectural rather than
        // making the acceptance rule title-specific.
        const uint32_t atomic_op = in.opcode == 0x05   ? Op_AtomicSMin
                                   : in.opcode == 0x06 ? Op_AtomicSMax
                                   : in.opcode == 0x07 ? Op_AtomicUMin
                                   : in.opcode == 0x08 ? Op_AtomicUMax
                                   : in.opcode == 0x09 ? Op_AtomicAnd
                                   : in.opcode == 0x0a ? Op_AtomicOr
                                                       : Op_AtomicXor;
        b.lds_atomic(atomic_op, idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
    } else if (in.opcode == 0x20) {   // ds_add_rtn_u32: VDST = old LDS; LDS += DATA0
        const uint32_t old = vreg_old(b, rs, in.dst.value);
        rs.vreg[in.dst.value] = b.lds_atomic_rtn(Op_AtomicIAdd, idx, vread(in.src[1].value),
                                                 rs.exec_narrowed, rs.exec, old);
        predicate_write(b, rs, in.dst.value, old);
    } else if (in.opcode == 0x2d) {   // ds_wrxchg_rtn_b32: swap DATA0, return old LDS
        const uint32_t old = vreg_old(b, rs, in.dst.value);
        rs.vreg[in.dst.value] = b.lds_atomic_rtn(Op_AtomicExchange, idx, vread(in.src[1].value),
                                                 rs.exec_narrowed, rs.exec, old);
        predicate_write(b, rs, in.dst.value, old);
    } else if (in.opcode == 0x0d) {   // ds_write_b32: LDS[idx] = DATA0
        b.lds_store(idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec, atomicize_store);
    } else if (in.opcode == 0x4d || in.opcode == 0xde || in.opcode == 0xdf) {
        // ds_write_b64 / ds_write_b96 / ds_write_b128. The wide forms consume consecutive
        // DATA0 VGPRs and write consecutive LDS dwords from the ordinary byte address.
        const int dwords = in.opcode == 0x4d ? 2 : in.opcode == 0xde ? 3 : 4;
        for (int k = 0; k < dwords; ++k) {
            const uint32_t at = ir::lower_ds_store_word_index(b, idx, k);
            b.lds_store(at, vread(in.src[1].value + k), rs.exec_narrowed, rs.exec, atomicize_store);
        }
    } else if (in.opcode == 0x76 || in.opcode == 0xfe || in.opcode == 0xff) {
        // ds_read_b64 / ds_read_b96 / ds_read_b128. RDNA2 ISA 12.13 opcodes 118/254/255;
        // all return consecutive dwords beginning at the ordinary LDS byte address.
        const int dwords = in.opcode == 0x76 ? 2 : in.opcode == 0xfe ? 3 : 4;
        for (int k = 0; k < dwords; ++k) {
            const uint32_t old = vreg_old(b, rs, in.dst.value + k);
            const uint32_t at = ir::lower_ds_read_word_index(b, idx, k);
            rs.vreg[in.dst.value + k] = b.lds_load(at, rs.exec_narrowed, rs.exec, old);
        }
    } else {   // ds_read_b32: VDST = LDS[idx]
        uint32_t old = vreg_old(b, rs, in.dst.value);
        rs.vreg[in.dst.value] = b.lds_load(idx, rs.exec_narrowed, rs.exec, old);
    }
    return true;
}
// NOLINTEND(concurrency-mt-unsafe)
}   // namespace prosper::gpu::ir
