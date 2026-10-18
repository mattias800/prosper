// Legacy cursor operations moved by extract_case_routes.py.
#include "gpu/recompiler/ir/cfg/structured_if_cursor.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
namespace prosper::gpu::ir::cfg {
StructuredIfCursor::StructuredIfCursor(const std::vector<Rdna2Inst>&,
                                       const std::vector<ForwardIf>& branches, const RegionPolicy&,
                                       bool)
    : Fs(branches) {}
uint32_t StructuredIfCursor::next_before(uint32_t hi) const {
    const uint32_t next_br = (bi < Fs.size() && Fs[bi].branch_pc < hi) ? Fs[bi].branch_pc : hi;
    return next_br;
}
ForwardIf StructuredIfCursor::take() {
    const ForwardIf F = Fs[bi++];
    return F;
}
uint32_t StructuredIfCursor::next_pc() const {
    const uint32_t next_if = bi < Fs.size() ? Fs[bi].branch_pc : UINT32_MAX;
    return next_if;
}
} // namespace prosper::gpu::ir::cfg
