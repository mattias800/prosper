#include "gpu/recompiler/rdna2_relative_index_values.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {
using Values = std::set<uint32_t>;
using Answer = std::optional<Values>;
struct Definition {
    uint32_t opcode;
    std::vector<uint32_t> arguments;
};

class IndexQuery {
    const SpirvCompute& builder;
    const RegState& state;
    uint32_t owner;
    std::map<uint32_t, Definition> definitions;
    std::map<std::pair<uint32_t, unsigned>, Answer> memo;
    uint32_t work = 0;
    bool valid = true;

    bool insert(Values& values, uint32_t value) {
        if (++work > 512) return false;
        values.insert(value);
        return values.size() <= 32;
    }

    Answer subsets(const Values& masks) {
        Values result;
        for (uint32_t mask : masks) {
            if (std::popcount(mask) > 5) return std::nullopt;
            uint32_t subset = mask;
            for (;;) {
                if (!insert(result, subset)) return std::nullopt;
                if (!subset) break;
                subset = (subset - 1u) & mask;
            }
        }
        return result;
    }

    Answer binary(const Definition& definition, unsigned depth) {
        const auto left = query(definition.arguments[0], depth + 1);
        const auto right = query(definition.arguments[1], depth + 1);
        // A small AND mask bounds any U32 input. Genuineness is the caller's separate proof.
        if (definition.opcode == Op_BitwiseAnd && (!left || !right)) {
            if (left) return subsets(*left);
            if (right) return subsets(*right);
            return std::nullopt;
        }
        if (!left || !right) return std::nullopt;
        Values result;
        for (uint32_t a : *left) {
            for (uint32_t c : *right) {
                if (definition.opcode == Op_ShiftLeftLogical && c >= 32) return std::nullopt;
                const uint32_t value = definition.opcode == Op_BitwiseAnd ? a & c : a << c;
                if (!insert(result, value)) return std::nullopt;
            }
        }
        return result;
    }

    Answer compute(uint32_t value, unsigned depth) {
        uint32_t constant = 0;
        if (builder.uconst_literal(value, &constant)) return Values{constant};
        const auto found = definitions.find(value);
        if (found == definitions.end()) return std::nullopt;
        const auto& definition = found->second;
        if (definition.opcode == Op_Bitcast) return query(definition.arguments[0], depth + 1);
        if (definition.opcode != Op_Select) return binary(definition, depth);
        // Predicated VGPR storage retains inactive bits. Existing immutable EXEC ancestry
        // proves which arm is read in this cohort, without claiming anything about DATA.
        if (exec_within(state, owner, definition.arguments[0]))
            return query(definition.arguments[1], depth + 1);
        const auto left = query(definition.arguments[1], depth + 1);
        const auto right = query(definition.arguments[2], depth + 1);
        if (!left || !right) return std::nullopt;
        Values result = *left;
        for (uint32_t item : *right)
            if (!insert(result, item)) return std::nullopt;
        return result;
    }

public:
    IndexQuery(const SpirvCompute& b, const RegState& rs, uint32_t exec)
        : builder(b), state(rs), owner(exec) {
        for (std::size_t at = 0; at < b.code.size();) {
            const uint32_t width = b.code[at] >> 16;
            if (!width || width > b.code.size() - at) {
                valid = false;
                break;
            }
            const uint32_t opcode = b.code[at] & 65535u;
            const uint32_t expected = opcode == Op_Bitcast  ? 4u
                                      : opcode == Op_Select ? 6u
                                      : opcode == Op_BitwiseAnd || opcode == Op_ShiftLeftLogical
                                          ? 5u
                                          : 0u;
            if (expected && width >= 3 && b.code[at + 1] == b.t_u32) {
                if (width != expected ||
                    !definitions
                         .emplace(
                             b.code[at + 2],
                             Definition{opcode,
                                        {b.code.begin() + static_cast<std::ptrdiff_t>(at + 3),
                                         b.code.begin() + static_cast<std::ptrdiff_t>(at + width)}})
                         .second) {
                    valid = false;
                    break;
                }
            }
            at += width;
        }
    }

    Answer query(uint32_t value, unsigned depth = 0) {
        if (!valid || depth > 24 || ++work > 512) return std::nullopt;
        const auto key = std::pair{value, depth};
        if (const auto cached = memo.find(key); cached != memo.end()) return cached->second;
        auto answer = compute(value, depth);
        memo.emplace(key, answer);
        return answer;
    }
};
}   // namespace

std::optional<std::set<uint32_t>> relative_index_values(const SpirvCompute& b, const RegState& rs,
                                                        uint32_t value, uint32_t exec) {
    if (!exec) return std::nullopt;
    return IndexQuery(b, rs, exec).query(value);
}
}   // namespace prosper::gpu
