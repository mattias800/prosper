// AMD 70648 Tables 90/91: DPP16 SRC0 values; EXEC is data, not a mask register here.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include <cstdint>

namespace prosper::gpu::ir {
class Builder;
struct DppRowSource {
    Value word;
    Value in_bounds;
};
// The caller keeps the BC0 write predicate separate from FI0 source zeroing.
DppRowSource dpp_row_shr_source(Builder& ir, Value source, Value active, uint32_t shift,
                                bool bound_control);
Value dpp_quad_source(Builder& ir, Value source, Value active, uint32_t control);
} // namespace prosper::gpu::ir
