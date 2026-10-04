#include "gpu/recompiler/fragment_packet_services.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/fragment_scalar_bank_wire.hpp"

namespace prosper::gpu {
namespace {
// Absolute shared-plane addressing, deliberately NOT load_packet_word's per-wave base. Every
// eager AccessChain is sanitized against the actual descriptor range, independently of tags.
uint32_t bank_word(SpirvCompute& b, const PacketResourceServices& service, uint32_t index) {
    const auto in_range = b.ucmp(Op_ULessThan, index, service.scalar_bank_length);
    const auto safe = b.sel(in_range, index, b.uconst(0));
    const auto pointer = b.id(), value = b.id();
    b.put(b.code, Op_AccessChain,
          {b.t_ptr_sb_u32, pointer, service.scalar_bank_variable, b.uconst(0), safe});
    b.put(b.code, Op_Load, {b.t_u32, value, pointer});
    return b.sel(in_range, value, b.uconst(0));
}
} // namespace
void PacketResourceServices::begin_scalar_bank(SpirvCompute& b) {
    scalar_bank_variable = b.id();
    b.declare_external_storage_buffer(b.t_ptr_sb_struct_u, scalar_bank_variable);
    b.put(b.deco, Op_Decorate, {scalar_bank_variable, Dec_DescriptorSet, 0});
    b.put(b.deco, Op_Decorate, {scalar_bank_variable, Dec_Binding, kFragmentScalarBankBinding});
    b.put(b.deco, Op_Decorate, {scalar_bank_variable, 24}); // NonWritable
    scalar_bank_length = b.id();
    b.put(b.code, 68, {b.t_u32, scalar_bank_length, scalar_bank_variable, 0}); // OpArrayLength
    // A uniform physical-length gate precedes EVERY bank load, including the sanitized index0
    // fallback. A short/empty descriptor skips all guest workers together; the freshly zeroed
    // transaction cannot acquire valid RPK1 status/exports and whole-draw validation refuses it.
    scalar_bank_merge = b.id();
    const auto body = b.id();
    const uint32_t table_words =
        kFragmentScalarBankHeaderWords +
        uint32_t(input.scalar_bank_sites.size()) * kFragmentScalarBankSiteWords;
    const auto available = b.ucmp(Op_UGreaterThanEqual, scalar_bank_length, b.uconst(table_words));
    b.put(b.code, Op_SelectionMerge, {scalar_bank_merge, 0});
    b.put(b.code, Op_BranchConditional, {available, body, scalar_bank_merge});
    b.emit_label(body);
}
void PacketResourceServices::finish_scalar_bank(SpirvCompute& b) {
    if (!scalar_bank_merge) return;
    b.emit_branch(scalar_bank_merge);
    b.emit_label(scalar_bank_merge);
}

int PacketResourceServices::emit_scalar_bank_load(SpirvCompute& b, RegState& state,
                                                  const Rdna2Inst& in) {
    const auto site = std::find_if(input.scalar_bank_sites.begin(), input.scalar_bank_sites.end(),
                                   [&](const auto& candidate) { return candidate.pc == in.pc; });
    if (site == input.scalar_bank_sites.end()) return -1;
    const uint32_t index = static_cast<uint32_t>(site - input.scalar_bank_sites.begin());
    const auto word = [&](uint32_t offset) { return bank_word(b, *this, b.uconst(offset)); };
    const uint32_t table_end =
        kFragmentScalarBankHeaderWords +
        uint32_t(input.scalar_bank_sites.size()) * kFragmentScalarBankSiteWords;
    auto valid = b.ucmp(Op_UGreaterThanEqual, scalar_bank_length, b.uconst(table_end));
    valid =
        b.land(valid, b.ucmp(Op_IEqual, word(ScalarBankMagic), b.uconst(kFragmentScalarBankMagic)));
    valid = b.land(
        valid, b.ucmp(Op_IEqual, word(ScalarBankVersion), b.uconst(kFragmentScalarBankVersion)));
    valid = b.land(valid, b.ucmp(Op_IEqual, word(ScalarBankTotalWords), scalar_bank_length));
    valid = b.land(valid, b.ucmp(Op_IEqual, word(ScalarBankSiteCount),
                                 b.uconst(uint32_t(input.scalar_bank_sites.size()))));
    const auto payload_begin = word(ScalarBankPayloadBegin),
               payload_words = word(ScalarBankPayloadWords);
    valid = b.land(valid, b.ucmp(Op_IEqual, payload_begin, b.uconst(table_end)));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, payload_begin, scalar_bank_length));
    valid = b.land(valid, b.ucmp(Op_IEqual, payload_words,
                                 b.ibin(Op_ISub, scalar_bank_length, payload_begin)));
    valid = b.land(valid, b.ucmp(Op_IEqual, word(ScalarBankReserved0), b.uconst(0)));
    valid = b.land(valid, b.ucmp(Op_IEqual, word(ScalarBankReserved1), b.uconst(0)));
    const uint32_t row = kFragmentScalarBankHeaderWords + index * kFragmentScalarBankSiteWords;
    const auto field = [&](uint32_t offset) { return word(row + offset); };
    valid = b.land(valid, b.ucmp(Op_IEqual, field(ScalarBankPc), b.uconst(in.pc)));
    valid = b.land(valid, b.ucmp(Op_IEqual, field(ScalarBankOpcode), b.uconst(in.opcode)));
    valid = b.land(valid, b.ucmp(Op_IEqual, field(ScalarBankLoadWords), b.uconst(site->words)));
    valid = b.land(valid, b.ucmp(Op_IEqual, field(ScalarBankByteOffset), b.uconst(in.literal)));
    valid = b.land(valid, b.ucmp(Op_IEqual, field(ScalarBankSiteReserved), b.uconst(0)));
    auto equal = b.btrue();
    std::array<uint32_t, 4> runtime;
    for (uint32_t c = 0; c < runtime.size(); ++c) {
        Operand operand;
        operand.kind = OperandKind::SGPR;
        operand.value = in.src[0].value + int(c);
        bool available = true;
        runtime[c] = operand_bits(b, state, in, operand, &available);
        if (!available) return -1;
        equal = b.land(equal, b.ucmp(Op_IEqual, runtime[c], field(ScalarBankDescriptor0 + c)));
    }
    fail(b, b.logical_not(equal), in.pc, FragmentPacketRuntimeFailure::DescriptorMismatch);
    valid = b.land(valid, equal);
    // Runtime bounds derive from genuine runtime raw V#, not a potentially corrupted host table.
    const auto stride = b.ibin(
        Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, runtime[1], b.uconst(16)), b.uconst(0x3fff));
    const auto scale = b.sel(b.ucmp(Op_IEqual, stride, b.uconst(0)), b.uconst(1), stride);
    // Exact two-DWORD product, without adding a shaderInt64 dependency to raw scalar bounds.
    const auto declared_lo = b.ibin(Op_IMul, scale, runtime[2]);
    const auto declared_hi = b.umul_hi(scale, runtime[2]);
    valid = b.land(valid, b.ucmp(Op_IEqual, declared_lo, field(ScalarBankDeclaredLo)));
    valid = b.land(valid, b.ucmp(Op_IEqual, declared_hi, field(ScalarBankDeclaredHi)));
    const uint64_t end = uint64_t(in.literal) + uint64_t(site->words) * 4u;
    valid = b.land(valid, b.u64_pair_ule(b.uconst(uint32_t(end)), b.uconst(uint32_t(end >> 32)),
                                         declared_lo, declared_hi));
    valid =
        b.land(valid, b.ucmp(Op_IEqual, b.ibin(Op_BitwiseAnd, runtime[1], b.uconst(0x80000000u)),
                             b.uconst(0)));
    const auto interval = field(ScalarBankIntervalBase), length = field(ScalarBankIntervalWords);
    const auto offset = field(ScalarBankIntervalByteOffset);
    valid =
        b.land(valid, b.ucmp(Op_IEqual, b.ibin(Op_BitwiseAnd, offset, b.uconst(3)), b.uconst(0)));
    const auto dword_offset = b.ibin(Op_ShiftRightLogical, offset, b.uconst(2));
    valid = b.land(valid, b.ucmp(Op_UGreaterThanEqual, interval, payload_begin));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, interval, scalar_bank_length));
    valid = b.land(
        valid, b.ucmp(Op_ULessThanEqual, length, b.ibin(Op_ISub, scalar_bank_length, interval)));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, dword_offset, length));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, b.uconst(site->words),
                                 b.ibin(Op_ISub, length, dword_offset)));
    fail(b, b.logical_not(valid), in.pc, FragmentPacketRuntimeFailure::ScalarBankInvalid);
    // Sticky failure executes even EXEC=0. Safe zero-index loads are internal refused-packet
    // storage only, never invented successful guest values; all64 workers still rendezvous.
    const auto safe_base = b.sel(valid, b.ibin(Op_IAdd, interval, dword_offset), b.uconst(0));
    for (uint32_t c = 0; c < site->words; ++c) {
        const auto address = b.sel(valid, b.ibin(Op_IAdd, safe_base, b.uconst(c)), b.uconst(0));
        state.sreg[in.dst.value + int(c)] = bank_word(b, *this, address);
    }
    return 1;
}
}   // namespace prosper::gpu
