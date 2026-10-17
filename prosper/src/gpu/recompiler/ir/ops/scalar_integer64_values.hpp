// Pure numeric portions of existing scalar-pair routes. Operand/domain handling stays in lower/.
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
enum class Scalar64Source : uint32_t { S0Lo, S0Hi, S1Lo, S1Hi, Scc };
struct Scalar64Values {
    Value first, second;
};
struct Scalar64Ids {
    uint32_t first = 0, second = 0;
};
struct Scalar64Known {
    Known64 first, second;
};
// Inputs are appended lazily. Emission resolves them at exactly the original expression point,
// including SOPC's high-word reads. The slot map is shared by emission and evaluation.
class Scalar64Inputs {
public:
    Scalar64Inputs(Builder& ir, std::vector<Scalar64Source>& slots, bool high_available)
        : high_available(high_available), ir_(ir), slots_(slots) {}
    Value s0lo() const { return read(Scalar64Source::S0Lo); }
    Value s0hi() const { return read(Scalar64Source::S0Hi); }
    Value s1lo() const { return read(Scalar64Source::S1Lo); }
    Value s1hi() const { return read(Scalar64Source::S1Hi); }
    Value scc() const { return read(Scalar64Source::Scc); }
    const bool high_available;

private:
    Value read(Scalar64Source source) const {
        slots_.push_back(source);
        return ir_.input(source == Scalar64Source::Scc ? Type::Bool : Type::U32);
    }
    Builder& ir_;
    std::vector<Scalar64Source>& slots_;
};
struct ScalarInteger64Def {
    Rdna2Format fmt;
    uint32_t opcode;
    Scalar64Values (*build)(Builder&, const Scalar64Inputs&);
};
std::span<const ScalarInteger64Def> scalar_integer64_value_defs();
const ScalarInteger64Def* find_scalar_integer64_value_op(Rdna2Format fmt, uint32_t opcode);
Scalar64Known eval_scalar_integer64_value(const ScalarInteger64Def& def,
                                          const std::array<Known64, 5>& sources,
                                          bool high_available = true);
using Scalar64Reader = std::function<uint32_t(Scalar64Source)>;
Scalar64Ids emit_scalar_integer64_value(SpirvCompute& b, const ScalarInteger64Def& def,
                                        const Scalar64Reader& read, bool high_available = true);
} // namespace prosper::gpu::ir
