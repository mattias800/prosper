// PM4 memory operations use published GFX10 packet/TC_OP fields. They are command-processor
// effects, never substitutes for cache barriers or draw predication (#4840).
#pragma once

#include "gpu/pm4/pm4_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace prosper::gpu {

struct GpuState;

// Conservative guest destination used by pending queues and checked scalar overlays.
void pm4_memory_effect_span(const Pm4Command& command, uint64_t* address, uint64_t* bytes);

bool integer_atomic_mem_op(uint32_t op);
uint32_t atomic_mem_bytes(uint32_t op);
std::optional<uint64_t> atomic_mem_value(uint32_t op, uint64_t old_value, uint64_t source,
                                         uint64_t compare);
bool execute_atomic_mem(const Pm4Command& command);
bool atomic_mem_operand_valid(const Pm4Command& command);
bool refuse_atomic_mem_producer(const Pm4Command& command);

// Decode and apply incrementally: a false COND_EXEC span need not contain parseable packets.
// Only executed commands are appended to the diagnostic list.
size_t fold_pm4_segment(const uint32_t* buffer, size_t dwords, GpuState& state,
                        std::vector<Pm4Command>& executed, uint8_t queue_origin);

} // namespace prosper::gpu
