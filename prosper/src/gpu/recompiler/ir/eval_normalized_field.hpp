// Nominal binary32 reference for a finite normalized component; never a guest-fold admission.
#pragma once
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
Known evaluate_normalized_field(uint32_t imm, uint32_t field);
}
