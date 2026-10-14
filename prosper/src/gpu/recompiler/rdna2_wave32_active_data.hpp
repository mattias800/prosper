// Lane-wise genuine DATA under CURRENT Wave32 EXEC; this grants no mask or physical words.
#pragma once
#include "gpu/recompiler/rdna2_exec_spill_equality.hpp"
#include <set>
#include <vector>
namespace prosper::gpu {
struct SpirvCompute;
bool tracks_wave32_active_data(const SpirvCompute& builder);
struct Wave32ActiveData {
    std::set<int> current;
    ExecSpillEquality exec_spill;
    bool vcc_produced = false;
    bool control = true;
    bool operator==(const Wave32ActiveData&) const = default;
};
void meet_wave32_active_data(Wave32ActiveData& into, const Wave32ActiveData& incoming);
void advance_wave32_active_data(Wave32ActiveData& data, const Rdna2Inst& instruction,
                                const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                const std::vector<int>& vector_writes, int mask_write,
                                bool genuine_sources);
} // namespace prosper::gpu
