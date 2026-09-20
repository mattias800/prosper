// Generated move, not new code: `SpirvCompute`'s method bodies, lifted out of
// gpu/recompiler/rdna2_to_spirv_internal.hpp. outline_methods.py checked each body byte for byte.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

uint32_t SpirvCompute::function_var(uint32_t type, uint32_t& ptr_type) {
        if (!ptr_type) { ptr_type = id(); put(types, Op_TypePointer, {ptr_type, SC_Function, type}); }
        uint32_t var = id();
        std::vector<uint32_t> decl;
        put(decl, Op_Variable, {ptr_type, var, SC_Function});
        code.insert(code.begin() + static_cast<std::ptrdiff_t>(function_var_insert),
                    decl.begin(), decl.end());
        function_var_insert += decl.size();
        return var;
    }

uint32_t SpirvCompute::guest_scratch_load_word(uint32_t index) {
        uint32_t pointer = id();
        putv(code, Op_AccessChain,
             {t_ptr_guest_scratch_u32, pointer, guest_scratch, uconst(index)});
        uint32_t value = id();
        put(code, Op_Load, {t_u32, value, pointer});
        return value;
    }

void SpirvCompute::guest_scratch_store_word(uint32_t index, uint32_t value,
                                  bool predicated, uint32_t pred) {
        auto store = [&]() {
            uint32_t pointer = id();
            putv(code, Op_AccessChain,
                 {t_ptr_guest_scratch_u32, pointer, guest_scratch, uconst(index)});
            put(code, Op_Store, {pointer, value});
        };
        if (!predicated) { store(); return; }
        const uint32_t then_label = id(), merge_label = id();
        emit_selmerge(merge_label);
        emit_condbranch(pred, then_label, merge_label);
        emit_label(then_label);
        store();
        emit_branch(merge_label);
        emit_label(merge_label);
    }

uint32_t SpirvCompute::guest_scratch_load_bits(int32_t byte_offset, uint32_t bits, bool sign_extend) {
        const uint32_t relative = static_cast<uint32_t>(byte_offset - guest_scratch_min_byte);
        const uint32_t index = relative / 4u, shift = (relative & 3u) * 8u;
        uint32_t value = guest_scratch_load_word(index);
        if (shift + bits <= 32u)
            return sign_extend ? bfe_s(value, uconst(shift), uconst(bits))
                               : bfe_u(value, uconst(shift), uconst(bits));
        const uint32_t low_bits = 32u - shift, high_bits = bits - low_bits;
        const uint32_t low = ibin(Op_ShiftRightLogical, value, uconst(shift));
        const uint32_t high_word = guest_scratch_load_word(index + 1u);
        const uint32_t high = bfe_u(high_word, uconst(0), uconst(high_bits));
        const uint32_t joined = ibin(Op_BitwiseOr, low,
                                     ibin(Op_ShiftLeftLogical, high, uconst(low_bits)));
        return sign_extend ? bfe_s(joined, uconst(0), uconst(bits)) : joined;
    }

void SpirvCompute::guest_scratch_store_bits(int32_t byte_offset, uint32_t bits, uint32_t value,
                                  bool predicated, uint32_t pred) {
        const uint32_t relative = static_cast<uint32_t>(byte_offset - guest_scratch_min_byte);
        const uint32_t index = relative / 4u, shift = (relative & 3u) * 8u;
        auto bit_mask = [](uint32_t width) {
            return width == 32u ? 0xffffffffu : ((1u << width) - 1u);
        };
        auto replace = [&](uint32_t word_index, uint32_t dst_shift,
                           uint32_t width, uint32_t source_shift) {
            const uint32_t mask = bit_mask(width) << dst_shift;
            const uint32_t old = guest_scratch_load_word(word_index);
            uint32_t field = source_shift
                ? ibin(Op_ShiftRightLogical, value, uconst(source_shift)) : value;
            if (dst_shift) field = ibin(Op_ShiftLeftLogical, field, uconst(dst_shift));
            field = ibin(Op_BitwiseAnd, field, uconst(mask));
            const uint32_t retained = ibin(Op_BitwiseAnd, old, uconst(~mask));
            guest_scratch_store_word(word_index, ibin(Op_BitwiseOr, retained, field),
                                     predicated, pred);
        };
        if (shift + bits <= 32u) {
            if (bits == 32u && shift == 0u)
                guest_scratch_store_word(index, value, predicated, pred);
            else
                replace(index, shift, bits, 0);
            return;
        }
        const uint32_t low_bits = 32u - shift;
        replace(index, shift, low_bits, 0);
        replace(index + 1u, 0, bits - low_bits, low_bits);
    }

uint32_t SpirvCompute::load_function(uint32_t type, uint32_t var) {
        uint32_t value = id();
        put(code, Op_Load, {type, value, var});
        add_fragment_wave_vote_dependency(value, var);
        return value;
    }

void SpirvCompute::store_function(uint32_t var, uint32_t value) {
        put(code, Op_Store, {var, value});
        // Function storage can join dynamic paths and loops. Keep an explicit dependency edge so
        // a vote emitted later can propagate back through loads already generated by a dispatcher.
        add_fragment_wave_vote_dependency(var, value);
    }

void SpirvCompute::capture_indirect_pointer_descriptor_source(
            uint32_t record_index, uint32_t root_lo, uint32_t descriptor_word1,
            bool predicated, uint32_t predicate) {
        if (!indirect_pointer_source_record_var ||
            !indirect_pointer_source_root_lo_var ||
            !indirect_pointer_source_root_hi_var)
            return;
        uint32_t root_hi = ibin(
            Op_BitwiseAnd, descriptor_word1, uconst(0xffffu));
        if (predicated) {
            const uint32_t old_record = load_function(
                t_u32, indirect_pointer_source_record_var);
            const uint32_t old_root_lo = load_function(
                t_u32, indirect_pointer_source_root_lo_var);
            const uint32_t old_root_hi = load_function(
                t_u32, indirect_pointer_source_root_hi_var);
            record_index = sel(predicate, record_index, old_record);
            root_lo = sel(predicate, root_lo, old_root_lo);
            root_hi = sel(predicate, root_hi, old_root_hi);
        }
        // Preserve one atomic provenance tuple across EXEC masking. The source MUBUF's inactive
        // destination values may have been recycled since an earlier capture; they must not replace
        // only the root while the old record identity survives.
        store_function(indirect_pointer_source_record_var, record_index);
        store_function(indirect_pointer_source_root_lo_var, root_lo);
        store_function(indirect_pointer_source_root_hi_var, root_hi);
    }

uint32_t SpirvCompute::relocated_indirect_carrier_dword(uint32_t selected_byte,
                                               uint32_t valid) {
        // OpSelect does not short-circuit an OpLoad. Select a known in-range carrier address before
        // either load, and avoid index+1 for an aligned dword at the physical end of the binding.
        const uint32_t safe_byte = sel(valid, selected_byte, uconst(0));
        const uint32_t index0 = ibin(Op_ShiftRightLogical, safe_byte, uconst(2));
        const uint32_t shift = ibin(
            Op_ShiftLeftLogical,
            ibin(Op_BitwiseAnd, safe_byte, uconst(3)), uconst(3));
        const uint32_t needs_second = ucmp(Op_INotEqual, shift, uconst(0));
        const uint32_t index1 = sel(
            needs_second, ibin(Op_IAdd, index0, uconst(1)), index0);
        const uint32_t dword0 = cbuf_load(index0, indirect_pointer_binding);
        const uint32_t dword1 = cbuf_load(index1, indirect_pointer_binding);
        const uint32_t lower = ibin(Op_ShiftRightLogical, dword0, shift);
        const uint32_t inverse_shift = ibin(
            Op_BitwiseAnd, ibin(Op_ISub, uconst(32), shift), uconst(31));
        const uint32_t upper = ibin(Op_ShiftLeftLogical, dword1, inverse_shift);
        const uint32_t joined = ibin(
            Op_BitwiseOr, lower, sel(needs_second, upper, uconst(0)));
        return sel(valid, joined, uconst(0));
    }

uint32_t SpirvCompute::relocated_indirect_load_dword(uint32_t address_lo, uint32_t address_hi,
                                            uint32_t immediate_byte_offset) {
        const U64PairAdd access_begin =
            add_u64_pair_u32(address_lo, address_hi, uconst(immediate_byte_offset));
        const U64PairAdd access_end =
            add_u64_pair_u32(access_begin.lo, access_begin.hi, uconst(sizeof(uint32_t)));
        uint32_t selected_byte = uconst(0);
        uint32_t match_count = uconst(0);
        const uint32_t directory_dword =
            indirect_pointer_segment_directory_byte_offset / sizeof(uint32_t);
        constexpr uint32_t kSegmentDwords =
            kIndirectBufferRelocationSegmentBytes / sizeof(uint32_t);
        for (uint32_t segment = 0; segment < indirect_pointer_segment_count; ++segment) {
            const uint32_t entry = directory_dword + segment * kSegmentDwords;
            const uint32_t guest_lo = cbuf_load(
                uconst(entry), indirect_pointer_binding);
            const uint32_t guest_hi = cbuf_load(
                uconst(entry + 1u), indirect_pointer_binding);
            const uint32_t byte_count = cbuf_load(
                uconst(entry + 2u), indirect_pointer_binding);
            const uint32_t packed_byte = cbuf_load(
                uconst(entry + 3u), indirect_pointer_binding);

            const U64PairAdd guest_end =
                add_u64_pair_u32(guest_lo, guest_hi, byte_count);
            uint32_t guest_contains = logical_not(access_begin.overflow);
            guest_contains = land(guest_contains, logical_not(access_end.overflow));
            guest_contains = land(guest_contains, logical_not(guest_end.overflow));
            guest_contains = land(
                guest_contains, ucmp(Op_INotEqual, byte_count, uconst(0)));
            guest_contains = land(
                guest_contains,
                u64_pair_ule(guest_lo, guest_hi, access_begin.lo, access_begin.hi));
            guest_contains = land(
                guest_contains,
                u64_pair_ule(access_end.lo, access_end.hi, guest_end.lo, guest_end.hi));

            // Once containment is true, the interval is at most UINT32_MAX bytes, so the low-word
            // subtraction is the exact residual even when the guest interval crosses 4 GiB.
            const uint32_t residual = ibin(Op_ISub, access_begin.lo, guest_lo);
            const uint32_t candidate = ibin(Op_IAdd, packed_byte, residual);
            const uint32_t candidate_wrapped = ucmp(Op_ULessThan, candidate, packed_byte);
            const uint32_t candidate_end =
                ibin(Op_IAdd, candidate, uconst(sizeof(uint32_t)));
            const uint32_t candidate_end_wrapped =
                ucmp(Op_ULessThan, candidate_end, candidate);
            const uint32_t packed_end = ibin(Op_IAdd, packed_byte, byte_count);
            const uint32_t packed_end_wrapped = ucmp(Op_ULessThan, packed_end, packed_byte);
            uint32_t packed_valid = logical_not(candidate_wrapped);
            packed_valid = land(packed_valid, logical_not(candidate_end_wrapped));
            packed_valid = land(packed_valid, logical_not(packed_end_wrapped));
            packed_valid = land(
                packed_valid,
                ucmp(Op_UGreaterThanEqual, packed_byte,
                     uconst(indirect_pointer_payload_byte_offset)));
            packed_valid = land(
                packed_valid,
                ucmp(Op_ULessThanEqual, packed_end,
                     uconst(indirect_pointer_carrier_bytes)));
            packed_valid = land(
                packed_valid, ucmp(Op_ULessThanEqual, candidate_end, packed_end));

            const uint32_t match = land(guest_contains, packed_valid);
            selected_byte = sel(match, candidate, selected_byte);
            match_count = ibin(
                Op_IAdd, match_count, sel(match, uconst(1), uconst(0)));
        }

        const uint32_t unique = ucmp(Op_IEqual, match_count, uconst(1));
        return relocated_indirect_carrier_dword(selected_byte, unique);
    }

uint32_t SpirvCompute::relocated_indirect_descriptor_load_dword(
            uint32_t address_lo, uint32_t address_hi,
            uint32_t immediate_byte_offset) {
        const U64PairAdd access_begin =
            add_u64_pair_u32(address_lo, address_hi, uconst(immediate_byte_offset));
        const U64PairAdd access_end =
            add_u64_pair_u32(access_begin.lo, access_begin.hi, uconst(sizeof(uint32_t)));
        const uint32_t captured_record = load_function(
            t_u32, indirect_pointer_source_record_var);
        const uint32_t captured_root_lo = load_function(
            t_u32, indirect_pointer_source_root_lo_var);
        const uint32_t captured_root_hi = load_function(
            t_u32, indirect_pointer_source_root_hi_var);

        const uint32_t max_source_index =
            (UINT32_MAX - indirect_pointer_source_pointer_byte_offset) /
            indirect_pointer_source_stride;
        const uint32_t source_index_valid = ucmp(
            Op_ULessThanEqual, captured_record, uconst(max_source_index));
        const uint32_t expected_source_offset = ibin(
            Op_IAdd,
            ibin(Op_IMul, captured_record, uconst(indirect_pointer_source_stride)),
            uconst(indirect_pointer_source_pointer_byte_offset));

        uint32_t selected_byte = uconst(0);
        uint32_t matching_records = uconst(0);
        uint32_t selected_segment_valid = bfalse();
        const uint32_t records_dword =
            indirect_pointer_record_directory_byte_offset / sizeof(uint32_t);
        const uint32_t segments_dword =
            indirect_pointer_segment_directory_byte_offset / sizeof(uint32_t);
        constexpr uint32_t kRecordDwords =
            kIndirectBufferRelocationRecordBytes / sizeof(uint32_t);
        constexpr uint32_t kSegmentDwords =
            kIndirectBufferRelocationSegmentBytes / sizeof(uint32_t);
        for (uint32_t record = 0; record < indirect_pointer_record_count; ++record) {
            const uint32_t entry = records_dword + record * kRecordDwords;
            const uint32_t source_offset = cbuf_load(
                uconst(entry), indirect_pointer_binding);
            const uint32_t segment_index = cbuf_load(
                uconst(entry + 1u), indirect_pointer_binding);
            const uint32_t guest_lo = cbuf_load(
                uconst(entry + 2u), indirect_pointer_binding);
            const uint32_t guest_hi = cbuf_load(
                uconst(entry + 3u), indirect_pointer_binding);
            const uint32_t byte_count = cbuf_load(
                uconst(entry + 4u), indirect_pointer_binding);
            const uint32_t address_kind = cbuf_load(
                uconst(entry + 5u), indirect_pointer_binding);

            const U64PairAdd record_end = add_u64_pair_u32(
                guest_lo, guest_hi, byte_count);
            uint32_t record_contains = logical_not(access_begin.overflow);
            record_contains = land(record_contains, logical_not(access_end.overflow));
            record_contains = land(record_contains, logical_not(record_end.overflow));
            record_contains = land(
                record_contains, ucmp(Op_INotEqual, byte_count, uconst(0)));
            record_contains = land(
                record_contains,
                u64_pair_ule(guest_lo, guest_hi, access_begin.lo, access_begin.hi));
            record_contains = land(
                record_contains,
                u64_pair_ule(access_end.lo, access_end.hi, record_end.lo, record_end.hi));

            uint32_t record_match = source_index_valid;
            record_match = land(
                record_match,
                ucmp(Op_IEqual, source_offset, expected_source_offset));
            record_match = land(
                record_match, ucmp(Op_IEqual, guest_lo, captured_root_lo));
            record_match = land(
                record_match, ucmp(Op_IEqual, guest_hi, captured_root_hi));
            record_match = land(
                record_match,
                ucmp(Op_IEqual, address_kind,
                     uconst(static_cast<uint32_t>(
                         IndirectBufferRelocationRecord::SourceAddressKind::
                             BufferDescriptorBase48))));
            record_match = land(record_match, record_contains);
            matching_records = ibin(
                Op_IAdd, matching_records,
                sel(record_match, uconst(1), uconst(0)));

            const uint32_t segment_index_valid = ucmp(
                Op_ULessThan, segment_index, uconst(indirect_pointer_segment_count));
            const uint32_t safe_segment = sel(
                segment_index_valid, segment_index, uconst(0));
            const uint32_t segment_entry = ibin(
                Op_IAdd, uconst(segments_dword),
                ibin(Op_IMul, safe_segment, uconst(kSegmentDwords)));
            const uint32_t segment_guest_lo = cbuf_load(
                segment_entry, indirect_pointer_binding);
            const uint32_t segment_guest_hi = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(1)),
                indirect_pointer_binding);
            const uint32_t segment_bytes = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(2)),
                indirect_pointer_binding);
            const uint32_t packed_byte = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(3)),
                indirect_pointer_binding);
            const uint32_t reserved_lo = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(4)),
                indirect_pointer_binding);
            const uint32_t reserved_hi = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(5)),
                indirect_pointer_binding);

            const U64PairAdd segment_end = add_u64_pair_u32(
                segment_guest_lo, segment_guest_hi, segment_bytes);
            uint32_t segment_valid = segment_index_valid;
            segment_valid = land(
                segment_valid, ucmp(Op_INotEqual, segment_bytes, uconst(0)));
            segment_valid = land(segment_valid, logical_not(segment_end.overflow));
            segment_valid = land(
                segment_valid,
                u64_pair_ule(segment_guest_lo, segment_guest_hi, guest_lo, guest_hi));
            segment_valid = land(
                segment_valid,
                u64_pair_ule(record_end.lo, record_end.hi,
                             segment_end.lo, segment_end.hi));
            segment_valid = land(
                segment_valid, ucmp(Op_IEqual, reserved_lo, uconst(0)));
            segment_valid = land(
                segment_valid, ucmp(Op_IEqual, reserved_hi, uconst(0)));

            // The record is contained by this segment, so low-word subtraction is the exact byte
            // residual. Validate the packed representation independently before selecting it.
            const uint32_t residual = ibin(
                Op_ISub, access_begin.lo, segment_guest_lo);
            const uint32_t candidate = ibin(Op_IAdd, packed_byte, residual);
            const uint32_t candidate_wrapped = ucmp(
                Op_ULessThan, candidate, packed_byte);
            const uint32_t candidate_end = ibin(
                Op_IAdd, candidate, uconst(sizeof(uint32_t)));
            const uint32_t candidate_end_wrapped = ucmp(
                Op_ULessThan, candidate_end, candidate);
            const uint32_t packed_end = ibin(
                Op_IAdd, packed_byte, segment_bytes);
            const uint32_t packed_end_wrapped = ucmp(
                Op_ULessThan, packed_end, packed_byte);
            segment_valid = land(segment_valid, logical_not(candidate_wrapped));
            segment_valid = land(segment_valid, logical_not(candidate_end_wrapped));
            segment_valid = land(segment_valid, logical_not(packed_end_wrapped));
            segment_valid = land(
                segment_valid,
                ucmp(Op_UGreaterThanEqual, packed_byte,
                     uconst(indirect_pointer_payload_byte_offset)));
            segment_valid = land(
                segment_valid,
                ucmp(Op_ULessThanEqual, packed_end,
                     uconst(indirect_pointer_carrier_bytes)));
            segment_valid = land(
                segment_valid,
                ucmp(Op_ULessThanEqual, candidate_end, packed_end));

            const uint32_t select_record = land(record_match, segment_valid);
            selected_byte = sel(select_record, candidate, selected_byte);
            selected_segment_valid = lor(selected_segment_valid, select_record);
        }

        const uint32_t unique = land(
            ucmp(Op_IEqual, matching_records, uconst(1)), selected_segment_valid);
        return relocated_indirect_carrier_dword(selected_byte, unique);
    }

uint32_t SpirvCompute::cfg_scratch_load(uint32_t idx) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_cfg_u32, p, cfg_scratch, idx});
        uint32_t value = id(); put(code, Op_Load, {t_u32, value, p}); return value;
    }

void SpirvCompute::cfg_scratch_store(uint32_t idx, uint32_t value) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_cfg_u32, p, cfg_scratch, idx});
        put(code, Op_Store, {p, value});
    }

}  // namespace prosper::gpu
