#include "gpu/recompiler/rdna2_legacy_vcc_writes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
namespace prosper::gpu {
bool legacy_valu_writes_vcc(const Rdna2Inst& in) {
    return (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            in.dst.kind == OperandKind::Special) ||
           (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28u && in.opcode <= 0x2au);
}
} // namespace prosper::gpu
