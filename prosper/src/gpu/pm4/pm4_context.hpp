#pragma once
#include <array>
#include <cstdint>

namespace prosper::gpu {
struct GpuState;
struct Pm4Command;

// Ordinary LOAD_* packets install a base for subsequent SET_* shadow writes. Indexed loads
// address independent arrays and do not replace these bases. Gfx/CS SH have separate selectors.
struct Pm4ContextControl {
    uint32_t load = 0, shadow = 0;
    bool load_known = false;
    std::array<uint64_t, 4> bases{};  // Cx, Uc, graphics SH, compute SH
};

void apply_context_control(const Pm4Command& command, GpuState& state);
bool apply_register_ranges(const Pm4Command& command, GpuState& state);
void shadow_direct_registers(const Pm4Command& command, GpuState& state);
bool register_shadow_operand_valid(const Pm4Command& command);
bool execute_register_shadow_write(const Pm4Command& command);
}  // namespace prosper::gpu
