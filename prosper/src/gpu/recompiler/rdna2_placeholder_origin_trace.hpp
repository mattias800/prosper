// Typed backward root slice for diagnostics and the ADR 0038 admission boundary.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <vector>
#include <set>
#include <cstdint>
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"

namespace prosper::gpu::detail {
struct PlaceholderOriginTrace {
    const PlaceholderOriginProgram& program;
    ir::cfg::GuestCfg cfg;
    std::vector<std::vector<uint32_t>> predecessors;
    std::vector<bool> full_exec;
    std::set<int> writelane_destinations;
    explicit PlaceholderOriginTrace(const PlaceholderOriginProgram&);
    bool writes(const Rdna2Inst&, PlaceholderWord) const;
    std::set<int> image_addresses(const Rdna2Inst&) const;
    std::vector<PlaceholderWord> sources(const Rdna2Inst&, PlaceholderWord) const;
    PlaceholderRoots entry(PlaceholderWord) const;
    PlaceholderRoots before(uint32_t pc, const std::vector<PlaceholderWord>&,
                            uint32_t work_limit = 1000000) const;
};
} // namespace prosper::gpu::detail
