#include "gpu/recompiler/rdna2_saved_exec_data.hpp"
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

bool exec_operand(const Operand& operand) {
    return register_operand(operand) && operand.value == 126;
}

struct Value {
    bool known = false, current = false;
    Pairs aliases;
};

Value value(const SavedExecDataEquality& equality, const Operand& operand) {
    if (exec_operand(operand)) return {true, true, equality.current_equal};
    if (!register_operand(operand)) return {};
    const auto found = equality.aliases.find(operand.value);
    if (found == equality.aliases.end()) return {};
    return {true, equality.current_equal.contains(operand.value), found->second};
}

bool same_value(const Value& left, const Value& right) {
    if (!left.known || !right.known) return false;
    if (left.current && right.current) return true;
    for (int pair : left.aliases)
        if (right.aliases.contains(pair)) return true;
    return false;
}

bool overlaps(int pair, int base, uint32_t width) {
    return pair <= base + static_cast<int>(width) - 1 && pair + 1 >= base;
}

void retire(SavedExecDataEquality& equality, int base, uint32_t width) {
    const auto affected = [&](int pair) { return overlaps(pair, base, width); };
    std::erase_if(equality.current_equal, affected);
    for (auto pair = equality.aliases.begin(); pair != equality.aliases.end();) {
        if (affected(pair->first))
            pair = equality.aliases.erase(pair);
        else {
            std::erase_if(pair->second, affected);
            ++pair;
        }
    }
}

Pairs surviving(const SavedExecDataEquality& equality, Pairs pairs) {
    std::erase_if(pairs, [&](int pair) { return !equality.aliases.contains(pair); });
    return pairs;
}

Pairs publish(SavedExecDataEquality& equality, int destination, const Value& source) {
    if (!source.known || destination < 0 || destination > 124) return {};
    auto aliases = surviving(equality, source.aliases);
    aliases.insert(destination);
    equality.aliases[destination] = aliases;
    for (int pair : aliases) equality.aliases[pair] = aliases;
    return aliases;
}
} // namespace

void meet_saved_exec_data(SavedExecDataEquality& into, const SavedExecDataEquality& incoming) {
    into.current_equal = intersection(std::move(into.current_equal), incoming.current_equal);
    for (auto pair = into.aliases.begin(); pair != into.aliases.end();) {
        const auto other = incoming.aliases.find(pair->first);
        if (other == incoming.aliases.end())
            pair = into.aliases.erase(pair);
        else {
            pair->second = intersection(std::move(pair->second), other->second);
            ++pair;
        }
    }
}

void advance_saved_exec_data(SavedExecDataEquality& equality, const Rdna2Inst& in,
                             const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                             int mask_write) {
    // CONFIDENCE: HIGH -- these are actual B64 value equalities. Sources are snapshotted before
    // overlapping writes; a partial/high-word replacement expires every old pair reference.
    // No producer PC or SSA owner survives as authority for a later loop generation.
    const Value before{true, true, equality.current_equal};
    const Value left = in.n_src > 0 ? value(equality, in.src[0]) : Value{};
    const Value right = in.n_src > 1 ? value(equality, in.src[1]) : Value{};
    const bool copy =
        in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 && in.n_src == 1;
    const bool saveexec = in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x24 && in.opcode <= 0x2b;
    const bool compare =
        in.fmt == Rdna2Format::VOPC || (in.fmt == Rdna2Format::VOP3 && mask_write >= 0);
    const bool identity = in.fmt == Rdna2Format::SOP2 && (in.opcode == 0x0f || in.opcode == 0x11) &&
                          same_value(left, right);
    Value result;
    if (copy || identity)
        result = left;
    else if (mask_write >= 0 &&
             (compare || (in.fmt == Rdna2Format::SOP2 && rdna2_sop2_dest_dwords(in.opcode) == 2)))
        result.known = true;   // a new complete mask value, with no equality to any old value

    bool changes_exec = rdna2_instruction_may_change_exec(in);
    for (const auto& [base, width] : scalar_writes) {
        changes_exec |= overlaps(126, base, width);
        retire(equality, base, width);
    }
    if (mask_write >= 0) {
        changes_exec |= overlaps(126, mask_write, 2);
        retire(equality, mask_write, 2);
    }
    const int destination = saveexec || copy ? in.dst.value : mask_write;
    const Value saved = saveexec ? before : result;
    const auto published = publish(equality, destination, saved);

    if (!changes_exec) {
        if (saved.current) equality.current_equal.insert(published.begin(), published.end());
        return;
    }
    equality.current_equal.clear();
    if (in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)) {
        // CMPX writes the same new comparison bits to VCC and EXEC.
        equality.current_equal = published;
    } else if (saveexec && in.opcode == 0x24 && left.current) {
        // S0 equals old EXEC, so S0 & old EXEC is that exact value.
        equality.current_equal = surviving(equality, before.aliases);
        equality.current_equal.insert(published.begin(), published.end());
    } else if (in.dst.value == 126 && (copy || identity)) {
        equality.current_equal = surviving(equality, result.aliases);
    }
}

void refresh_equal_saved_exec_word(ActiveVectorWords& state, int word) {
    // A CURRENT subset does not suffice: untouched lanes of a wider saved parent remain absent.
    for (int mask : state.saved_exec.current_equal)
        if (const auto found = state.masks.find(mask); found != state.masks.end())
            found->second.insert(word);
}
}   // namespace prosper::gpu
