// Pure textual relocation: tools/refactor/extract_case_routes.py.
#include "gpu/recompiler/ir/lower/smem_case_routes.hpp"
#include "gpu/recompiler/ir/lower/smem_index_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <optional>
namespace prosper::gpu::ir {
std::optional<bool> SmemCaseRoutes::pcrel(uint32_t n) {
    // PC-relative scalar embedded table (#1054): this s_buffer_load consumes a descriptor
    // built from s_getpc_b64 and a bounded table carried by the shader blob. Resolve it before
    // the external-resource gate, exactly like the established MUBUF form. The byte offset is
    // still the real tracked scalar value; an untracked SOFFSET remains fail-visible.
    auto pcrel = rs.smem_pcrel_tables.find(in.pc);
    if (pcrel != rs.smem_pcrel_tables.end()) {
        uint32_t soff = 0;
        const bool soff_null = in.src[1].kind == OperandKind::Special && in.src[1].value == 125;
        if (soff_null) {
            soff = b.uconst(0);
        } else if (in.src[1].kind == OperandKind::SGPR ||
                   (in.src[1].kind == OperandKind::Special && in.src[1].value >= 106 &&
                    in.src[1].value <= 123)) {
            auto tracked = rs.sreg.find(in.src[1].value);
            if (tracked == rs.sreg.end()) {
                ok = false;
                return true;
            }
            soff = tracked->second;
        } else if (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0) {
            soff = b.uconst(static_cast<uint32_t>(in.src[1].value));
        } else {
            ok = false;
            return true;
        }
        uint32_t idx = ir::lower_smem_pcrel_index(b, soff, in);
        for (uint32_t k = 0; k < n; ++k) {
            uint32_t kidx = ir::lower_smem_pcrel_component(b, idx, k);
            uint32_t value = ir::lower_smem_constant_word(b, kidx, pcrel->second);
            rs.sreg[in.dst.value + static_cast<int>(k)] = value;
            rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
        }
        return true;
    }
    return std::nullopt;
}
} // namespace prosper::gpu::ir
