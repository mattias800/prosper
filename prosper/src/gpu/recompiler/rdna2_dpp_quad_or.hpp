#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <map>
#include <set>
#include <utility>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
// Only the producing ordinary-compute entry may construct this original-program service.
class ComputeDppQuadOrService {
public:
    bool prepare(SpirvCompute&, const RegState&, const std::vector<Rdna2Inst>&,
                 const uint32_t* code, size_t dwords, uint32_t initial_active);
    bool present() const { return !events_.empty(); }
    const char* rejection() const { return rejection_; }
    bool contains(uint32_t pc) const { return events_.contains(pc); }
    void split(std::set<uint32_t>& starts, const std::vector<Rdna2Inst>&) const;
    void begin(SpirvCompute&, uint32_t ptr_bool, uint32_t ptr_u32);
    void reset(SpirvCompute&) const;
    bool publish(SpirvCompute&, RegState&, const Rdna2Inst&) const;
    bool emit_phase(SpirvCompute&, uint32_t value_base, uint32_t metadata_base,
                    const std::map<int, uint32_t>& vgprs,
                    const std::map<std::pair<int, int>, uint32_t>& numeric_lane_aliases,
                    const std::map<std::pair<int, int>, uint32_t>& mask_lane_aliases) const;

private:
    const char* rejection_ = "dpp-quad-or-source-uninitialized";
    std::map<uint32_t, uint32_t> events_;
    std::set<int> destinations_;
    enum Variable { Pending, Active, Source0, Source1, Control, Destination, Event, Count };
    std::array<uint32_t, Count> vars_{};
};
} // namespace prosper::gpu
