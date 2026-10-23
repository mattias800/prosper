// Existing full-EXEC-mask definition used by CFG branch proofs.
#include "gpu/recompiler/rdna2_cfg_must_dataflow.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

namespace prosper::gpu {

// `s_mov_b64 exec, -1` is the only accepted full-mask definition. A 32-bit literal written to a
// 64-bit scalar destination is zero-extended, so it activates only the low half and is NOT a full
// wave64 mask; the inline -1 constant is sign-extended and is.
bool exec_write_sets_full_mask(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP1 || in.opcode != 0x04 || in.n_src != 1) return false;
    if ((in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) ||
        in.dst.value != 126)
        return false;
    return in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1;
}

}  // namespace prosper::gpu
