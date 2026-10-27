// Observation only: successful compiler admissions, not draws or dynamic lane reads.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace prosper::diagnostics::perf {
struct AlarmFiring;
struct WindowSample;
struct ArbitraryPlaceholderProgram {
    uint64_t compilations = 0, fresh_mask_reads = 0, loop_carry_reads = 0;
};
using ArbitraryPlaceholderInventory = std::map<uint64_t, ArbitraryPlaceholderProgram>;
void note_arbitrary_placeholder_check();
void note_arbitrary_placeholder_admission(uint64_t program, uint64_t fresh, uint64_t loop);
std::string format_arbitrary_placeholder_summary(const ArbitraryPlaceholderInventory&);
void append_arbitrary_placeholder_alarm(std::vector<AlarmFiring>&, const WindowSample&);
} // namespace prosper::diagnostics::perf
