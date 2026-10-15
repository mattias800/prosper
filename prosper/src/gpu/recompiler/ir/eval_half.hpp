// Binary32 -> binary16, default RTE. Unknown NaN payloads are not invented.
#pragma once
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
Known pack_half_rte(uint32_t bits);
}
