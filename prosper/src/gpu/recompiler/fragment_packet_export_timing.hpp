#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <vector>

namespace prosper::gpu {
// Architectural PS exports read EXEC/VGPRs after bus grant, not necessarily at issue. This
// bounded owned, read-only packet domain may retain private records eagerly only after proving
// their source words/EXEC stable until full WAIT or ordinary END. This is not a timing simulator.
// Call AFTER the complete packet instruction/forward-control/resource inventory has admitted it.
const char* fragment_packet_export_timing_gap(const std::vector<Rdna2Inst>&, uint32_t& pc);
}   // namespace prosper::gpu
