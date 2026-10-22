// Existing MUBUF/MTBUF opcode classification; variables are borrowed from the emitter.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <optional>

namespace prosper::gpu::ir {
struct BufferOpcodeCaseRoutes {
    const Rdna2Inst& in;
    uint32_t& n;
    bool& raw_subword;
    uint32_t& raw_bits;
    bool& raw_signed;
    bool& is_format;
    bool& is_store;
    bool& is_atomic;
    uint32_t& atomic_op;
    bool& is_atomic_fminmax;
    bool& atomic_fmin;
    bool& is_atomic_x2;
    bool& ok;
    std::optional<bool> classify();
    void raw_loads();
    void format_and_stores();
    void integer_atomics();
    void float_and_wide_atomics();
};
}  // namespace prosper::gpu::ir
