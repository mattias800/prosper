#include "gpu/recompiler/rdna2_exec_spill_equality.hpp"
#include <algorithm>
#include <optional>

namespace prosper::gpu {
namespace {
using Word = std::optional<ExecSpillWord>;
bool scalar(const Operand& source) {
    return source.kind == OperandKind::SGPR && source.value >= 0 && source.value <= 105;
}
Word word(const ExecSpillEquality& equality, const Operand& source, int offset = 0) {
    if (source.kind == OperandKind::Special && source.value == 126 && offset <= 1)
        return offset == 0 ? ExecSpillWord::Low : ExecSpillWord::High;
    if (source.kind == OperandKind::Special && source.value == 127 && offset == 0)
        return ExecSpillWord::High;
    if (!scalar(source) || source.value + offset > 105) return {};
    const auto found = equality.scalars.find(source.value + offset);
    return found == equality.scalars.end() ? Word{} : found->second;
}
bool lane(const Rdna2Inst& in) {
    return in.n_src >= 2 && !in.has_sdwa && !in.has_dpp && !in.has_modifier &&
           in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 && in.src[1].value < 64;
}
void retire_scalar(ExecSpillEquality& equality, int base, uint32_t width) {
    for (uint32_t offset = 0; offset < width; ++offset)
        equality.scalars.erase(base + static_cast<int>(offset));
}
void retire_vector(ExecSpillEquality& equality, int vector) {
    std::erase_if(equality.slots, [&](const auto& entry) { return entry.first.first == vector; });
}
void publish(ExecSpillEquality& equality, int destination, Word value) {
    if (destination >= 0 && destination <= 105 && value) equality.scalars[destination] = *value;
}
} // namespace

void meet_exec_spill_equality(ExecSpillEquality& into, const ExecSpillEquality& incoming) {
    const auto meet = [](auto& values, const auto& other) {
        std::erase_if(values, [&](const auto& entry) {
            const auto found = other.find(entry.first);
            return found == other.end() || found->second != entry.second;
        });
    };
    meet(into.slots, incoming.slots);
    meet(into.scalars, incoming.scalars);
}

bool advance_exec_spill_equality(ExecSpillEquality& equality, const Rdna2Inst& in,
                                 const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                 const std::vector<int>& vector_writes, int mask_write) {
    // CONFIDENCE: HIGH -- these are architectural word-value equalities, not producer-PC or
    // register-name identities. Their meaning is relative to CURRENT EXEC on every path.
    const bool write_lane = in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361;
    const bool read_lane = in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360;
    const bool move32 = in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32;
    const bool move64 = in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64;
    if (move64 &&
        ((in.dst.value & 1) || (in.n_src && scalar(in.src[0]) && (in.src[0].value & 1)))) {
        // RDNA2 ISA 70648 sections 3.6.3/5.2 requires even SGPR pairs. An unsupported raw
        // index is no proof of its architectural source or destination aliasing.
        equality.slots.clear();
        equality.scalars.clear();
        return false;
    }
    const Word first = in.n_src ? word(equality, in.src[0]) : Word{};
    const Word second = move64 && in.n_src ? word(equality, in.src[0], 1) : Word{};
    Word selected;
    if (read_lane && lane(in) && in.src[0].kind == OperandKind::VGPR) {
        const auto found = equality.slots.find({in.src[0].value, in.src[1].value});
        if (found != equality.slots.end()) selected = found->second;
    }
    const bool unchanged = move64 && in.n_src == 1 && in.dst.value == 126 && scalar(in.src[0]) &&
                           first == ExecSpillWord::Low && second == ExecSpillWord::High;
    bool changes_exec = rdna2_instruction_may_change_exec(in);
    for (const auto& [base, width] : scalar_writes)
        changes_exec |= base <= 127 && base + static_cast<int>(width) > 126;
    changes_exec |= mask_write >= 125 && mask_write <= 127;
    if (changes_exec && !unchanged) {
        equality.slots.clear();
        equality.scalars.clear();
        return false;
    }

    if (write_lane) {
        if (!lane(in))
            retire_vector(equality, in.dst.value);
        else {
            const std::pair<int, int> key{in.dst.value, in.src[1].value};
            equality.slots.erase(key);
            if (first) equality.slots[key] = *first;
        }
    } else if (!read_lane) {
        // Every possible ordinary write retires the slot, including halfword, implicit-old-word
        // and relative writes. A storage writer count never proves a preserved word value.
        for (int vector : vector_writes) retire_vector(equality, vector);
    }
    for (const auto& [base, width] : scalar_writes) retire_scalar(equality, base, width);
    if (mask_write >= 0) retire_scalar(equality, mask_write, 2);
    if (read_lane) publish(equality, in.dst.value, selected);
    // Only actual lane transports originate scalar equality facts. Ordinary copies may relay
    // those already-proven words; assembling untransported raw/Bool words is a separate contract.
    else if (move32 && in.n_src == 1 && scalar(in.src[0]))
        publish(equality, in.dst.value, first);
    else if (move64 && in.n_src == 1 && scalar(in.src[0])) {
        publish(equality, in.dst.value, first);
        publish(equality, in.dst.value + 1, second);
    }
    return unchanged;
}
}   // namespace prosper::gpu
