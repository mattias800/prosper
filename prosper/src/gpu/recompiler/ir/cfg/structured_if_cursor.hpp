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
    enum class Route { Tree, NoIfs, DetectorRefused, Loops, Unstructured, Mismatch };
    Route route() const { return route_; }
    bool uses_tree() const { return route_ == Route::Tree; }
    StructuredIfCursor(const std::vector<Rdna2Inst>& ins, const std::vector<ForwardIf>& branches,
                       const RegionPolicy& policy, bool detector_refused);
    uint32_t next_before(uint32_t hi) const;
    uint32_t next_pc() const;
    ForwardIf take();

private:
    void restart_tree();
    void advance_tree();
    bool read_tree_if(ForwardIf& result) const;
    const std::vector<ForwardIf>& Fs;
    const std::vector<Rdna2Inst>& ins_;
    GuestCfg cfg_;
    RegionTree tree_;
    std::vector<uint32_t> pending_;
    uint32_t current_ = kNone, end_pc_ = UINT32_MAX;
    size_t steps_ = 0;
    bool malformed_ = false;
    Route route_ = Route::NoIfs;
    size_t bi = 0;   // next unconsumed branch in Fs (pc order; recursion consumes nested ones)
};
} // namespace prosper::gpu::ir::cfg
