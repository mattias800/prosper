#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace prosper::gpu {

// Original instruction boundaries for conditional terminal tails. Besides NOP/END, the only
// accepted effect is MOV_B64 EXEC,0 followed by one color EXP with EN0 and a real END. These
// facts supply code retention and scalar liveness, never a new emitter branch/entry authority.
struct Rdna2FragmentTerminalTail {
    size_t retained_dwords = 0;
    size_t inspected_dwords = 0;
    bool source_limit_required = false;
    std::vector<Rdna2Inst> instructions;
};

Rdna2FragmentTerminalTail rdna2_fragment_terminal_tail(const uint32_t* code, size_t dwords,
                                                       std::span<const Rdna2Inst> body);

} // namespace prosper::gpu
