// AMD70648 12.8: RPI is floor(S0 + 0.5). The lowerer admits only exact half-step constants.
#include "gpu/recompiler/ir/ops/vector_cvt_rpi_exact_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
const ConversionDef* find_vector_cvt_rpi_exact_e64_op(Rdna2Format fmt, uint32_t opcode) {
    if (fmt != Rdna2Format::VOP3 || opcode != 0x18c) return nullptr;
    const auto* old = find_vector_conversion_op(Rdna2Format::VOP1, 0x0c);
    if (!old) return nullptr;
    // CONFIDENCE: MED: this reuses the old value graph, not guest STATUS/exception emulation.
    static const ConversionDef def{Rdna2Format::VOP3, 0x18c, old->build};
    return &def;
}
} // namespace prosper::gpu::ir
