// Guest source reads are IR inputs appended at the same expression point as the old val() call.
// A slot map lets eval bind known words and lets emission call the original operand reader then.
#pragma once
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
class TernaryInputs {
public:
    TernaryInputs(Builder& ir, std::vector<uint32_t>& sources) : ir_(ir), sources_(sources) {}
    Value s0() const { return read(0); }
    Value s1() const { return read(1); }
    Value s2() const { return read(2); }

private:
    Value read(uint32_t source) const {
        sources_.push_back(source);
        return ir_.input(Type::U32);
    }
    Builder& ir_;
    std::vector<uint32_t>& sources_;
};
struct TernaryIntegerDef {
    Rdna2Format fmt;
    uint32_t opcode;
    Value (*build)(Builder&, const TernaryInputs&);
};
std::span<const TernaryIntegerDef> vector_ternary_integer_defs();
const TernaryIntegerDef* find_vector_ternary_integer_op(Rdna2Format fmt, uint32_t opcode);
Known eval_ternary_integer_op(const TernaryIntegerDef& def, const std::array<Known, 3>& sources);
using TernarySourceReader = std::function<uint32_t(uint32_t)>;
uint32_t emit_ternary_integer_op(SpirvCompute& b, const TernaryIntegerDef& def,
                                 const TernarySourceReader& read_source);
}   // namespace prosper::gpu::ir
