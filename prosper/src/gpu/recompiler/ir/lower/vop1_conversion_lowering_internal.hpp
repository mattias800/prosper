// Original emit_alu locals under their original names, without constructor side effects.
#pragma once
#include <cstdint>
#include <optional>
#include "gpu/recompiler/ir/lower/vop1_conversion_lowering.hpp"
namespace prosper::gpu::ir {
struct Vop1ConversionContext {
    SpirvCompute& b;
    RegState& rs;
    const Rdna2Inst& in;
    uint32_t a;
    uint32_t& d;
    uint32_t old_d;
    bool& ok;
};
class Vop1ConversionLowering : public Vop1ConversionContext {
public:
    std::optional<bool> routes();
    bool values();

private:
    std::optional<bool> route_half_integer();
    std::optional<bool> route_pack_word();
    std::optional<bool> route_integer_subword();
};
}   // namespace prosper::gpu::ir
