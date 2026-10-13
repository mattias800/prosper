#include "gpu/recompiler/rdna2_saved_complement_data.hpp"
#include "gpu/recompiler/rdna2_active_vector_words.hpp"

#include <algorithm>

namespace prosper::gpu {
namespace {
using Pairs = std::set<int>;

Pairs intersection(Pairs left, const Pairs& right) {
    std::erase_if(left, [&](int pair) { return !right.contains(pair); });
    return left;
}

bool register_operand(const Operand& operand) {
    return operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special;
}

bool pair_operand(const Operand& operand) {
    return register_operand(operand) && operand.value >= 0 && operand.value <= 124;
}

bool exec_operand(const Operand& operand) {
    return register_operand(operand) && operand.value == 126;
}

bool overlaps(int pair, int base, uint32_t width) {
    return pair <= base + static_cast<int>(width) - 1 && pair + 1 >= base;
}

void retire_parents(SavedComplementData& context,
                    const std::vector<std::pair<int, uint32_t>>& writes, int copied_parent) {
    for (const auto& [base, width] : writes)
        std::erase_if(context.parents, [&](int parent) { return overlaps(parent, base, width); });
    // A complete copy may preserve the old parent value even when it overlaps its source.
    if (copied_parent >= 0) context.parents.insert(copied_parent);
}

SavedComplementData current_context(const ActiveVectorWords& state) {
    const auto& proof = state.complement;
    if (proof.parent < 0 || !proof.parents.contains(proof.parent)) return {};
    Pairs parents{proof.parent};
    const auto aliases = state.saved_exec.aliases.find(proof.parent);
    if (aliases != state.saved_exec.aliases.end())
        parents.insert(aliases->second.begin(), aliases->second.end());
    std::erase_if(parents, [&](int parent) { return !proof.parents.contains(parent); });
    return {std::move(parents), proof.first_words};
}
} // namespace

void meet_saved_complement_data(SavedComplementValues& into,
                                const SavedComplementValues& incoming) {
    for (auto value = into.begin(); value != into.end();) {
        const auto other = incoming.find(value->first);
        if (other == incoming.end()) {
            value = into.erase(value);
            continue;
        }
        value->second.parents =
            intersection(std::move(value->second.parents), other->second.parents);
        value->second.first_words =
            intersection(std::move(value->second.first_words), other->second.first_words);
        if (value->second.parents.empty())
            value = into.erase(value);
        else
            ++value;
    }
}

SavedComplementTransfer
prepare_saved_complement_data(ActiveVectorWords& state, const Rdna2Inst& in,
                              const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                              int mask_write) {
    // CONFIDENCE: HIGH -- B was constructed as P&~A with A within P. Saving that actual B value
    // retains the equation, not a relation inferred from a subset or from a producer PC/SSA ID.
    // Snapshot sources before either carrier or parent generations are overwritten.
    auto& values = state.complement.saved;
    const bool copy =
        in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 && in.n_src == 1;
    const bool saveexec = in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x24 && in.opcode <= 0x2b;
    const bool copy_parent = copy && pair_operand(in.dst) && pair_operand(in.src[0]);
    std::optional<SavedComplementData> source;
    if (saveexec || (copy && exec_operand(in.src[0]))) {
        const auto current = current_context(state);
        if (!current.parents.empty()) source = current;
    } else if (copy && pair_operand(in.src[0])) {
        if (const auto found = values.find(in.src[0].value); found != values.end())
            source = found->second;
    }
    SavedComplementTransfer transfer;
    if ((copy || saveexec) && pair_operand(in.dst)) {
        transfer.destination = in.dst.value;
        transfer.captured = source;
    } else if (copy && exec_operand(in.dst)) {
        transfer.restored = source;
    }

    auto writes = scalar_writes;
    if (mask_write >= 0) writes.emplace_back(mask_write, 2);
    const auto update = [&](SavedComplementData& context) {
        const int copied =
            copy_parent && context.parents.contains(in.src[0].value) ? in.dst.value : -1;
        retire_parents(context, writes, copied);
    };
    for (auto value = values.begin(); value != values.end();) {
        bool destroyed = false;
        for (const auto& [base, width] : writes) destroyed |= overlaps(value->first, base, width);
        update(value->second);
        if (destroyed || value->second.parents.empty())
            value = values.erase(value);
        else
            ++value;
    }
    if (transfer.captured) update(*transfer.captured);
    if (transfer.restored) update(*transfer.restored);
    return transfer;
}

void complete_saved_complement_data(ActiveVectorWords& state, SavedComplementTransfer transfer) {
    auto& proof = state.complement;
    const auto live = [&](SavedComplementData& context) {
        std::erase_if(context.parents, [&](int parent) { return !proof.parents.contains(parent); });
        return !context.parents.empty();
    };
    if (transfer.destination >= 0 && transfer.captured && live(*transfer.captured))
        proof.saved[transfer.destination] = std::move(*transfer.captured);
    if (transfer.restored && live(*transfer.restored)) {
        proof.parent = *transfer.restored->parents.begin();
        proof.first_words = std::move(transfer.restored->first_words);
        // The exact stored equation proves B is within every still-live alias of actual P.
        proof.current_parents.insert(transfer.restored->parents.begin(),
                                     transfer.restored->parents.end());
    }
}

void retire_saved_complement_word(ComplementaryExecData& proof, int word) {
    for (auto& [mask, context] : proof.saved) context.first_words.erase(word);
}
}   // namespace prosper::gpu
