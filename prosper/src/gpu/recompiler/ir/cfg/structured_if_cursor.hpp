// Structured-emission branch cursor. Register state and all joins remain in emit_body.
#pragma once
#include "gpu/recompiler/ir/cfg/region_tree.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>
namespace prosper::gpu {
struct ForwardIf;
}
namespace prosper::gpu::ir::cfg {
class StructuredIfCursor {
public:
    StructuredIfCursor(const std::vector<Rdna2Inst>& ins, const std::vector<ForwardIf>& branches,
                       const RegionPolicy& policy, bool detector_refused);
    uint32_t next_before(uint32_t hi) const;
    uint32_t next_pc() const;
    ForwardIf take();

private:
    const std::vector<ForwardIf>& Fs;
    size_t bi = 0;   // next unconsumed branch in Fs (pc order; recursion consumes nested ones)
};
} // namespace prosper::gpu::ir::cfg
