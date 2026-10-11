#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <vector>

namespace prosper::gpu {
// Ordinary READLANE inventory only; no dispatcher or phase admission is granted here.
bool has_ordinary_cfg_readlane(const std::vector<Rdna2Inst>& instructions);
} // namespace prosper::gpu
