#include "gpu/recompiler/rdna2_native_logical_words.hpp"

namespace prosper::gpu {
namespace {
bool ordinary_pair(const Operand& source) {
    return (source.kind == OperandKind::SGPR && source.value >= 0 && source.value <= 104) ||
           (source.kind == OperandKind::Special && (source.value == 106 || source.value == 107));
}

bool marked_source(const RegState& rs, const Operand& source) {
    if (!ordinary_pair(source)) return false;
    for (int word : {source.value, source.value + 1})
        if (rs.sreg_merge_placeholder.contains(word) || rs.sreg_memory_pattern.contains(word) ||
            rs.unavailable_saved_mask_words.contains(word))
            return true;
    return false;
}

std::array<uint32_t, 2> source_words(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                     const Operand& source) {
    if (source.kind == OperandKind::InlineInt)
        return {b.uconst(static_cast<uint32_t>(source.value)),
                b.uconst(source.value < 0 ? UINT32_MAX : 0u)};
    if (source.kind == OperandKind::Literal && in.has_literal)
        return {b.uconst(in.literal), b.uconst(0)};
    if (!ordinary_pair(source) || marked_source(rs, source)) return {};
    std::array<uint32_t, 2> result{};
    for (int half = 0; half < 2; ++half) {
        const int word = source.value + half;
        if (sreg_word_may_be_fabricated(rs, word)) return {};
        const auto current = rs.sreg.find(word);
        if (current != rs.sreg.end())
            result[half] = current->second;
        else {
            // A mask write ends the launch-input lifetime. Do not resurrect an older input if
            // an incomplete current mask happens to coexist with an entry map spelling.
            if (rs.sreg_bool.contains(source.value) ||
                ((source.value == 106 || source.value == 107) && rs.vcc))
                return {};
            const auto input = rs.sreg_input.find(word);
            if (input == rs.sreg_input.end()) return {};
            result[half] = input->second;
        }
    }
    return result;
}

uint32_t logical_word(SpirvCompute& b, uint32_t opcode, uint32_t a, uint32_t c) {
    const auto complement = [&](uint32_t value) {
        return b.ibin(Op_BitwiseXor, value, b.uconst(UINT32_MAX));
    };
    switch (opcode) {
        case 0x0f: return b.ibin(Op_BitwiseAnd, a, c);
        case 0x11: return b.ibin(Op_BitwiseOr, a, c);
        case 0x13: return b.ibin(Op_BitwiseXor, a, c);
        case 0x15: return b.ibin(Op_BitwiseAnd, a, complement(c));
        case 0x17: return b.ibin(Op_BitwiseOr, a, complement(c));
        case 0x19: return complement(b.ibin(Op_BitwiseAnd, a, c));
        case 0x1b: return complement(b.ibin(Op_BitwiseOr, a, c));
        case 0x1d: return complement(b.ibin(Op_BitwiseXor, a, c));
        default: return 0;
    }
}
}   // namespace

bool is_native_b64_logical(const SpirvCompute& b, const Rdna2Inst& in) {
    return b.is_compute && !b.is_fragment_packet() && b.wave_size == 64 &&
           b.native_subgroup_size == 64 && in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0f &&
           in.opcode <= 0x1d && (in.opcode & 1u) && in.dst.value >= 0 && in.dst.value <= 107;
}

std::array<uint32_t, 2> native_logical_result_words(SpirvCompute& b, const RegState& rs,
                                                    const Rdna2Inst& in, uint32_t result_mask) {
    if (!is_native_b64_logical(b, in)) return {};
    // Read both complete current operands before any destination expires an overlapping source.
    // CONFIDENCE: HIGH -- AMD 70648 section12.1 defines B64 logic on all64 source bits. Retained
    // DATA therefore needs no rendezvous with departed lanes, even in a partial host region.
    const auto a = source_words(b, rs, in, in.src[0]);
    const auto c = source_words(b, rs, in, in.src[1]);
    if (a[0] && a[1] && c[0] && c[1])
        return {logical_word(b, in.opcode, a[0], c[0]), logical_word(b, in.opcode, a[1], c[1])};
    // A Bool-only result is not scalar DATA. The sole ballot fallback requires the original
    // complete producer's participation proof; width, wave_ok and a register number are not it.
    if (!saved_mask_producer_has_full_participation(b, in) || marked_source(rs, in.src[0]) ||
        marked_source(rs, in.src[1]))
        return {};
    return {b.native_wave_ballot_half(result_mask, 0), b.native_wave_ballot_half(result_mask, 1)};
}

void record_native_logical_word_gaps(const SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    if (!is_native_b64_logical(b, in)) return;
    for (int word : {in.dst.value, in.dst.value + 1})
        if (!rs.sreg.contains(word) || sreg_word_may_be_fabricated(rs, word) ||
            rs.sreg_memory_pattern.contains(word))
            rs.unavailable_saved_mask_words.insert(word);
}
}   // namespace prosper::gpu
