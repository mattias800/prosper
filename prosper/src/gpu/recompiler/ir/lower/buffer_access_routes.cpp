// Pure text relocation by tools/refactor/extract_case_routes.py; access expression order is unchanged.
#include "gpu/recompiler/ir/lower/buffer_access_routes.hpp"
#include "gpu/recompiler/ir/lower/buffer_address_lowering.hpp"
#include "gpu/recompiler/ir/lower/buffer_format_values.hpp"
#include "gpu/recompiler/ir/lower/buffer_store_values.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
namespace prosper::gpu::ir {
// NOLINTBEGIN(concurrency-mt-unsafe): existing diagnostic reads retain their timing.
bool BufferAccessRoutes::lower() {
    float norm = normalization_range();
    if (packed && !packed_word && !is_half && !is_uint && !is_sint && norm == 0.0f) {
        if (getenv("PROSPER_DBG"))
            fprintf(stderr, "[mubuf-badfmt] pc=%u fmt=%u comp_bytes=%u stride=%u n=%u\n", in.pc,
                    (unsigned)fmt, comp_bytes, stride, n);
        buf_op.how = "reject-badfmt";
        ok = false;
        return true;
    }
    // Most packed (sub-dword) components below use static fields relative to a DWORD-ALIGNED
    // element base. Float16 LOADS have a general runtime-address path: each component is read
    // from addr+k*2, joining adjacent dwords if the 16-bit field starts at byte 3. Other packed
    // formats still require a proven aligned base; reject them instead of silently dropping the
    // low address bits. Aligned iff: inst offset %4==0; stride %4==0 when idxen; no offen; and
    // SOFFSET is NULL/0. Stores retain their existing stricter paths.
    bool dyn_half = false;   // Float16 components at an arbitrary runtime byte address
    bool dyn_int = false;   // integer sub-dword FORMAT component at a runtime (unaligned) byte addr
    bool dyn_norm = false;   // normalized sub-dword FORMAT component at an arbitrary byte addr
    bool dyn_int_store = false;   // integer sub-dword FORMAT store via race-free atomic clear+set
    if (packed && !raw_subword) {
        bool base_aligned;
        if (folded_vfetch) {
            // The dyn_vfetch address path (below) is exactly gl_VertexIndex*stride and DROPS the
            // shader's inst-offset / offen / SOFFSET — the resolved V# base already folds in this
            // attribute's in-record byte offset. So packed-component alignment depends ONLY on the
            // per-element stride being dword-aligned (vertex records are, and the V# base with
            // them). Requiring soff_zero here (as the general path does) wrongly rejected Unorm8×4
            // packed-color attributes whose fetch carries a register SOFFSET that dyn_vfetch drops
            // anyway — the "mask=0xF draws never render" gap on PPSA01885. CONFIDENCE: HIGH.
            base_aligned = !idxen || (stride & 3u) == 0;
        } else {
            bool soff_zero = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                             (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
            base_aligned =
                ((offset & 3u) == 0) && !offen && (!idxen || (stride & 3u) == 0) && soff_zero;
        }
        // An integer sub-dword FORMAT STORE writes ONE lane's disjoint bit field of the
        // containing dword. A plain masked read-modify-write would race (adjacent lanes' fields
        // share a dword), but atomicAnd(clear field)+atomicOr(set field) COMMUTE across lanes
        // writing DISJOINT fields, so the store is race-free. Requires the field to lie within
        // a single dword: address comp_bytes-aligned, no runtime per-lane byte offset.
        bool soff_zero = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                         (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
        if (is_store && int_subword && !offen && !dyn_vfetch && (offset % comp_bytes) == 0 &&
            (!idxen || (stride % comp_bytes) == 0) && soff_zero)
            dyn_int_store = true;
        if (!base_aligned) {
            // Float16 format loads use the complete runtime byte address for every requested
            // component. This covers both the stride-2 single-half path (#273) and Astro's
            // Float16x4 vertex record with a register SOFFSET. Loads only: packed half stores
            // remain fail-visible until their race-free sub-dword write contract is modeled.
            dyn_half = !packed_word && !is_store && is_half;
            // An integer sub-dword FORMAT load extracts each component at its runtime byte
            // address (join the straddled dwords, then bfe), so it needs no static alignment.
            // Only LOADS: the packed store path still rejects sub-dword ints (they don't pack).
            if (!dyn_half) dyn_int = int_subword && !is_store;
            // UNORM/SNORM 8/16 loads use the same runtime extraction, followed by the format's
            // normalization. Astro's world-map VS fetches SNORM16x3 with a shader-computed
            // SOFFSET; treating it as statically packed rejected the complete map draw.
            if (!dyn_half && !dyn_int)
                dyn_norm = !packed_word && !is_store && norm != 0.0f &&
                           (comp_bytes == 1 || comp_bytes == 2);
            if (!dyn_half && !dyn_int && !dyn_norm && !dyn_int_store) {
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr,
                            "[mubuf-unaligned] pc=%u fmt=%u off=%u offen=%d idxen=%d stride=%u\n",
                            in.pc, (unsigned)fmt, offset, (int)offen, (int)idxen, stride);
                buf_op.how = "reject-unaligned";
                ok = false;
                return true;
            }
        }
    }
    const ir::BufferAddressIds address = ir::lower_buffer_address(
        b, rs, in, ok, {offset, stride, offen, idxen, folded_vfetch, dyn_vfetch, instance_vfetch});
    const uint32_t addr = address.addr, idx = address.idx;
    // Keep the instruction's cache policy at one ordinary-load propagation site. Every raw,
    // typed, packed, and dynamically addressed component below consumes dwords through this
    // wrapper, so adding another format shape cannot accidentally drop GLC/DLC semantics.
    auto load_dword = [&](uint32_t dword_idx) {
        if (indirect_pointer_descriptor_source) {
            // The relocation carrier appends directories and packed segments after the
            // original source table. The guest V# still ends at `source_bytes`; an OOB
            // source-record fetch must return zero rather than accidentally reading the
            // carrier metadata. Clamp the physical load first because OpSelect does not
            // short-circuit it, then select the architectural OOB zero.
            const uint32_t in_source =
                b.ucmp(Op_ULessThan, dword_idx,
                       b.uconst(b.indirect_pointer_source_bytes / sizeof(uint32_t)));
            const uint32_t safe_idx = b.sel(in_source, dword_idx, b.uconst(0));
            const uint32_t value = b.cbuf_load(safe_idx, binding, coherent_load);
            return b.sel(in_source, value, b.uconst(0));
        }
        if (gta5_selected_sbuffer_consumer) {
            const auto soff = rs.sreg.find(106);
            if (soff == rs.sreg.end()) {
                ok = false;
                return b.uconst(0u);
            }
            const uint32_t selected =
                b.ucmp(Op_IEqual, soff->second, b.uconst(b.gta5_selected_sbuffer_soffset));
            // A SPIR-V select does not short-circuit its operands. Redirect rejected
            // partitions to the proven non-empty binding's dword zero before loading, then
            // select the architectural OOB zero, so an arbitrary guest v6 cannot create an
            // out-of-range host access on a path whose V# was actually the zero descriptor.
            const uint32_t safe_idx = b.sel(selected, dword_idx, b.uconst(0u));
            const uint32_t value = b.cbuf_load(safe_idx, binding, coherent_load);
            return b.sel(selected, value, b.uconst(0u));
        }
        return b.cbuf_load(dword_idx, binding, coherent_load);
    };
    if (is_atomic_x2) {
        const int d = in.dst.value;
        const uint32_t old_lo = vreg_old(b, rs, d);
        const uint32_t old_hi = vreg_old(b, rs, d + 1);
        const auto lo_it = rs.vreg.find(d), hi_it = rs.vreg.find(d + 1);
        const uint32_t value_lo = lo_it == rs.vreg.end() ? b.uconst(0) : lo_it->second;
        const uint32_t value_hi = hi_it == rs.vreg.end() ? b.uconst(0) : hi_it->second;
        const uint32_t value = b.u64_from_lohi(value_lo, value_hi);
        // Range-check the original qword record index against this dispatch's concrete V#.
        // Each valid index addresses its natural eight-byte record; larger indices are OOB.
        const uint32_t record_index = val(in.src[0]);
        const uint32_t in_bounds =
            b.ucmp(Op_ULessThan, record_index, b.uconst(atomic_x2_record_count));
        const uint32_t access = rs.exec_narrowed ? b.land(rs.exec, in_bounds) : in_bounds;
        uint32_t fallback = b.uconst64(0);
        if (rs.exec_narrowed)
            fallback = b.sel64(rs.exec, fallback, b.u64_from_lohi(old_lo, old_hi));
        const uint32_t pre =
            b.cbuf_atomic_x2_rtn(atomic_op, record_index, value, binding, access, fallback);
        // GLC=0 preserves BOTH data VGPRs. GLC=1 returns the pre-op qword; active OOB lanes
        // receive zero and inactive EXEC lanes receive their original pair via fallback.
        if (in.mubuf_glc) {
            rs.vreg[d] = b.u64_lo(pre);
            rs.vreg[d + 1] = b.u64_hi(pre);
        }
        return true;
    }
    if (is_atomic) {
        if (is_atomic_fminmax && b.compute_pgm_rsrc1 == UINT32_MAX) {
            ok = false;
            return true;
        }
        const int d = in.dst.value;
        const uint32_t old = vreg_old(b, rs, d);
        const auto it = rs.vreg.find(d);
        const uint32_t value = it == rs.vreg.end() ? b.uconst(0) : it->second;
        const uint32_t pre =
            is_atomic_fminmax
                ? b.cbuf_atomic_fminmax_rtn(idx, value, binding, atomic_fmin, rs.exec_narrowed,
                                            rs.exec, old)
                : b.cbuf_atomic_rtn(atomic_op, idx, value, binding, rs.exec_narrowed, rs.exec, old);
        // ISA 8.1 / Table 98: for atomics GLC means "return pre-op value to VGPR". With
        // GLC=0 hardware leaves VDATA untouched (it still holds the DATA operand) — the
        // unconditional write clobbered it (the exercised Astro packet 0xe0e00004 is GLC=0).
        if (in.mubuf_glc) rs.vreg[d] = pre;
        return true;
    }
    if (is_store) {
        auto vread = [&](int r) {
            auto it = rs.vreg.find(r);
            return it == rs.vreg.end() ? b.uconst(0) : it->second;
        };
        if (coherent_store) b.mark_cbuf_coherent(binding);
        // As with loads, keep Volatile propagation shared by raw and packed ordinary stores.
        auto store_dword = [&](uint32_t dword_idx, uint32_t value) {
            b.cbuf_store(dword_idx, value, binding, rs.exec_narrowed, rs.exec, coherent_store);
        };
        // The DATA FORMAT owns the physical component COUNT of every format store: a wider
        // opcode writes only the components the format has. For MTBUF the format is the
        // instruction's; for MUBUF it is the descriptor's (`fmt_ncomp` above). Table 31 gives
        // MTBUF stores identity routing ("X000, XY00, XYZ0, or XYZW" by that component count)
        // but gives MUBUF format stores the descriptor's DST_SEL. Writing components 0..n-1
        // in order is therefore right for MUBUF only because a routed MUBUF store was
        // rejected above (`reject-dst-sel-store`), so what reaches here is identity.
        // The clamp used to apply to MTBUF only, so buffer_store_format_xyzw through a
        // one-component 32-bit V# wrote four dwords per lane at a four-byte stride: every lane
        // overwrote its three neighbours, and only one element per wave kept its value. That
        // is UE4's typed-buffer clear and Kena's distance-field AO cone buffer. Raw stores
        // leave `fmt_ncomp` at 0 and keep the opcode's width.
        const uint32_t store_n = is_format && fmt_ncomp && fmt_ncomp < n ? fmt_ncomp : n;
        if (dyn_int_store) {
            // Integer sub-dword store: clear then set THIS lane's disjoint field of the containing
            // dword with two atomics. Disjoint fields commute (And clears only this field's bits,
            // Or sets only this field's bits), so no read-modify-write lock is needed; EXEC
            // predication keeps inactive lanes from writing (like cbuf_store). CONFIDENCE: HIGH —
            // this diverges from a hardware byte-enable store ONLY in already-UB situations: two
            // lanes storing to the SAME element OR-merge instead of one-winner, and a racing reader
            // could observe the transient post-And zero. Both require a data race a well-formed
            // shader never has. A straddling field (excluded by the alignment guard above) is the
            // one shape this can't express and stays deferred.
            const uint32_t field_mask = comp_bytes == 2 ? 0xffffu : 0xffu;
            for (uint32_t k = 0; k < store_n; k++) {
                const auto fields = ir::lower_buffer_store_fields(b, in, addr, k, comp_bytes,
                                                                  field_mask, std::cref(vread));
                const uint32_t cidx = fields.index, mask = fields.mask, v = fields.value;
                b.cbuf_atomic_rtn(Op_AtomicAnd, cidx, b.iun(Op_Not, mask), binding,
                                  rs.exec_narrowed, rs.exec, b.uconst(0));
                b.cbuf_atomic_rtn(Op_AtomicOr, cidx, v, binding, rs.exec_narrowed, rs.exec,
                                  b.uconst(0));
            }
            return true;
        }
        // Store the VDATA VGPRs (in.dst..+n-1). Integer sub-dword formats reach the atomic path
        // above when in-dword-provable; a sub-dword integer field that could STRADDLE a dword
        // boundary still rejects rather than mis-store. The packed-word formats are
        // implemented below (#3575).
        if (packed && !packed_word && (is_uint || is_sint)) {
            buf_op.how = "reject-subword-int-store";
            ok = false;
            return true;
        }
        // The identity claim the code below makes is settled rather than assumed: RDNA2 ISA
        // Table 31 gives TBUFFER_STORE_FORMAT_* a DST SEL of "identity", so an MTBUF store
        // really does ignore the descriptor's routing, while BUFFER_STORE_FORMAT_* does not
        // and reaches `reject-dst-sel-store` above when routed (#2869). Everything below is
        // therefore identity routing, written `store_n` components wide (see above).
        if (!packed) {
            // Raw/Float32/Uint32: one dword per component.
            for (uint32_t k = 0; k < store_n; k++) {
                const uint32_t kidx = ir::lower_buffer_store_index(b, idx, k);
                store_dword(kidx, vread(in.dst.value + (int)k));
            }
        } else if (packed_word) {
            // #3575: every component shares ONE dword. This is the exact inverse of the
            // packed_word LOAD in ir::lower_buffer_packed_field. Its field table and the
            // table in ir::build_buffer_store_word are RESTATED independently (#3600).
            // They must be kept in step by hand:
            // a store whose layout disagreed with the load would round-trip wrongly through
            // prosper's own reload, which is the cheapest way for this to be wrong and go
            // unnoticed. If you edit either table, edit both.
            //
            // `comp_bytes` is 0 for these formats (data_format_bytes has no case for them),
            // so the generic packed loop below would compute dwords=0 and silently write
            // NOTHING. That is why this is its own branch rather than a wider `comp_bytes`.
            //
            // The component count comes from the FORMAT, not from the opcode: a
            // buffer_store_format_xyzw through a 3-component 10_11_11 must not write a fourth
            // field, which at k=3 would land back on top of B. Same hazard the format
            // component-count clamp (`store_n`) above addresses, one level down.
            // `packed_10_11_11` and `packed_2_10_10_10` name a BIT LAYOUT, not a channel
            // count. GFX10 names packed formats from the HIGH field down, so 10_11_11 puts
            // 11 bits at [10:0] and 10 at [31:22] -- while 11_11_10 and 10_10_10_2, which
            // `rdna2_buffer_format` does not decode today, are DIFFERENT layouts with the
            // narrow field at the opposite end. prosper's COLOUR-buffer enum canonicalises
            // those pairs as aliases; that is a different register enum and must not be
            // carried over here. Aliasing them into these predicates would make this store
            // pack fields in the wrong order and write a plausible wrong dword -- turning
            // today's fail-visible reject into a silent one.
            const uint32_t packed_word_ncomp = packed_10_11_11 ? 3u : 4u;
            // A PARTIAL packed-word store stays refused, and this is a deliberate stop rather
            // than an oversight. Every field shares one dword, so writing a subset is a
            // read-modify-write of the fields NOT being written -- and whether the hardware
            // preserves those fields or zeroes the whole dword is not something the RDNA2 ISA
            // document settles, nor something any title evidence here establishes. Guessing
            // costs the neighbouring channel silently: the store would succeed and the
            // untouched field would come back zero. A wider opcode than the format (xyzw
            // through a 3-component 10_11_11) is NOT this case -- the fourth component does
            // not exist in the format, so clamping it writes every field that does.
            // CONFIDENCE: HIGH that refusing is right; the semantics are what is unknown.
            if (store_n < packed_word_ncomp) {
                buf_op.how = "reject-packed-word-partial-store";
                ok = false;
                return true;
            }
            const uint32_t acc = ir::lower_buffer_store_packed_word(
                b, in, store_n, packed_word_ncomp, packed_10_11_11, is_uint, is_sint, is_snorm,
                std::cref(vread));
            store_dword(idx, acc);
        } else {
            // Packed UNORM/SNORM/Float16: pack the components tightly into ceil(n*bytes/4) dwords
            // (inverse of the packed load). Each dword ORs together the fields that land in it.
            const uint32_t dwords = (store_n * comp_bytes + 3) / 4;
            for (uint32_t d = 0; d < dwords; d++) {
                const auto packed_dword = ir::lower_buffer_store_packed_dword(
                    b, in, idx, d, store_n, comp_bytes, is_half, is_snorm, norm, std::cref(vread));
                const uint32_t did = packed_dword.index, acc = packed_dword.value;
                store_dword(did, acc);
            }
        }
        return true;
    }
    // Integer-typed format? Its absent-component default for W/A is integer 1, not float 1.0.
    const bool fmt_is_int =
        (fmt == DataFormat::Uint8 || fmt == DataFormat::Uint16 || fmt == DataFormat::Uint32 ||
         fmt == DataFormat::Sint8 || fmt == DataFormat::Sint16 || fmt == DataFormat::Sint32 ||
         fmt == DataFormat::Uint2_10_10_10 || fmt == DataFormat::Sint2_10_10_10);
    for (uint32_t k = 0; k < n; k++) {
        int d = in.dst.value + (int)k;
        uint32_t old = vreg_old(b, rs, d);
        uint32_t value;
        // DST_SEL (#2869): `sk` is the STORED component this RETURNED channel k takes. Every
        // branch below addresses memory by `sk` and writes VGPR `d`, which stays keyed on k,
        // so a permutation costs nothing but an index substitution. `sel_const` is SQ_SEL_0 /
        // SQ_SEL_1, which name a constant instead of a component and read no memory at all.
        // `dst_sel_routed` is false for MTBUF and for every raw op, so `sk == k` there and
        // this whole mechanism is inert -- see the Table 31 partition above.
        const uint32_t sel = dst_sel_routed && k < 4u ? dst_sel[k] : k + 4u;
        const bool sel_const = sel < 4u;
        const uint32_t sk = sel_const ? k : sel - 4u;
        // Format default-fill (#368): a requested component beyond the format's component
        // count is not read from adjacent memory. MUBUF takes the ABSENT-component default
        // from the V# contract (0 for G/B/Z, 1 for A/W). MTBUF's instruction format names the
        // present components directly (X000/XY00/XYZ0/XYZW), so every absent one is zero.
        // NOTE (#2869): this is about which components EXIST, not about how the descriptor
        // routes the ones that do -- DST_SEL is a separate field, applied just above. The
        // absent-component test therefore asks about the ROUTED source `sk`, not about k.
        // MUBUF's absent-W default of 1 predates DST_SEL support (#368) and is kept: a real
        // three-component descriptor spells that with SQ_SEL_1 and now takes the `sel_const`
        // branch instead, while a descriptor prosper could not recover falls back here.
        if (sel_const) {
            // SQ_SEL_0 -> 0; SQ_SEL_1 -> the format's own one, exactly as the absent-component
            // default below spells it: integer 1 for an integer format, float 1.0 otherwise.
            value = b.uconst(sel == 1u ? (fmt_is_int ? 1u : 0x3f800000u) : 0u);
        } else if (is_format && fmt_ncomp && sk >= fmt_ncomp) {
            uint32_t one = fmt_is_int ? 1u : 0x3f800000u;   // integer 1 vs float 1.0 (raw bits)
            value = b.uconst(in.fmt != Rdna2Format::MTBUF && sk == 3 ? one : 0u);
        } else if (one_record_16bit_tail) {
            const uint32_t packed_value =
                b.cbuf_load_zero_padded_tail(binding, one_record_tail_semantic, coherent_load);
            const uint32_t in_bounds = b.ucmp(Op_IEqual, val(in.src[0]), b.uconst(0));
            const uint32_t typed_value =
                one_record_tail_semantic == StorageBufferTailSemantic::Float16
                    ? b.unpack_half(packed_value, 0)
                    : b.bfe_u(packed_value, b.uconst(0), b.uconst(16));
            value = b.sel(in_bounds, typed_value, b.uconst(0));
        } else if (raw_subword) {
            // Raw byte/short loads use their full byte address, unlike typed packed-format
            // loads whose component packing is descriptor-defined. A 16-bit access may begin
            // at byte 3 and straddle two dwords, so join the adjacent words before extracting.
            const uint32_t dw0 = load_dword(idx);
            const auto raw_shift = ir::lower_buffer_raw_shift(b, addr, dw0);
            const uint32_t shift = raw_shift.shift;
            uint32_t joined = raw_shift.joined;
            if (raw_bits == 16) {
                const uint32_t dw1 = load_dword(ir::lower_buffer_next_dword(b, idx));
                joined = ir::lower_buffer_raw_join(b, joined, dw1, shift);
            }
            value = ir::lower_buffer_raw_field(b, joined, raw_bits, raw_signed);
        } else if (!packed) {
            uint32_t kidx = sk ? b.ibin(Op_IAdd, idx, b.uconst(sk)) : idx;
            value = load_dword(kidx);   // raw 32-bit component
        } else if (packed_word) {
            // All requested components share one packed dword. GFX10 names layouts from high
            // field to low field, so 2_10_10_10 is logical R/G/B in bits 0/10/20 and A in 30;
            // 10_11_11 is R/G/B in bits 0/11/22 with widths 11/11/10.
            //
            // ir::lower_buffer_packed_field and ir::build_buffer_store_word keep separate
            // field tables (#3600). Both must agree for a store/load round-trip; if either
            // layout changes, update both.
            uint32_t dw = load_dword(idx);
            value = ir::lower_buffer_packed_field(b, dw, sk, packed_10_11_11, is_uint, is_sint,
                                                  is_snorm);
        } else if (dyn_half) {
            // Float16 component k at byte address addr+k*2. Join the following dword when
            // the field begins at byte 3; masking the inverse shift avoids SPIR-V's undefined
            // shift-by-32 case, and the select discards that word when shift==0.
            const auto location = ir::lower_buffer_half_location(b, addr, sk);
            const uint32_t cidx = location.index, shift = location.shift;
            uint32_t joined = ir::lower_buffer_low_shift(b, load_dword(cidx), shift);
            const uint32_t dw1 = load_dword(ir::lower_buffer_next_dword(b, cidx));
            joined = ir::lower_buffer_half_join(b, joined, dw1, shift);
            value = ir::lower_buffer_half_field(b, joined);
        } else if (dyn_int || dyn_norm) {
            // Integer/normalized sub-dword FORMAT component k at a runtime byte address
            // (addr + k*comp_bytes):
            // shift the loaded dword right by (byteaddr&3)*8, join the next dword when a 16-bit
            // field straddles the boundary, then extend or normalize the low field. Mirrors the
            // raw_subword path but per packed component (stride-1 Uint8, unaligned u16/SNORM16).
            const auto location = ir::lower_buffer_dynamic_location(b, addr, sk, comp_bytes);
            const uint32_t cidx = location.index, shift = location.shift;
            uint32_t joined = ir::lower_buffer_low_shift(b, load_dword(cidx), shift);
            if (comp_bytes == 2) {
                const uint32_t dw1 = load_dword(ir::lower_buffer_next_dword(b, cidx));
                joined = ir::lower_buffer_dynamic_join(b, joined, dw1, shift);
            }
            value = ir::lower_buffer_dynamic_field(b, joined, comp_bytes, dyn_norm, is_snorm, norm,
                                                   is_sint);
        } else {
            // Source component sk lives at byte sk*comp_bytes within the element: pick its
            // dword + field.
            const auto location = ir::lower_buffer_static_location(b, idx, sk, comp_bytes);
            const uint32_t did = location.index, boff = location.bit_offset;
            uint32_t dw = load_dword(did);
            value = ir::lower_buffer_static_field(b, dw, boff, comp_bytes, is_half, is_uint,
                                                  is_sint, is_snorm, norm);
        }
        rs.vreg[d] = value;
        predicate_write(b, rs, d, old);
    }
    return finish();
}
// NOLINTEND(concurrency-mt-unsafe)
}   // namespace prosper::gpu::ir
