// Independent RNE normalized-half evaluation; no NaN result is fabricated.
#pragma once
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
Known evaluate_pack_norm16(uint32_t bits, uint32_t mode);
}
