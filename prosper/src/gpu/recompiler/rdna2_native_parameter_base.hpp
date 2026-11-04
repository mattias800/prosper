#pragma once

#include "gpu/recompiler/pixel_parameter_launch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::gpu {
struct NativeParameterBasePlan;
struct RegState;
struct SpirvCompute;

// A native interpolation service, never a numerical scalar value. The original entry load
// dominates control, M0 has no other writer or numerical observer, and every later read of the
// source SGPR follows an unconditional complete overwrite of its entry lifetime.
std::shared_ptr<const NativeParameterBasePlan>
native_parameter_base_plan(const PixelParameterLaunch&, const std::vector<Rdna2Inst>&);
// Complete original SOURCE with the already supported bounded terminal trailers. The existing
// plan still proves every entry use; trailer retention supplies no branch or entry-state fact.
std::shared_ptr<const NativeParameterBasePlan>
native_parameter_base_plan(const PixelParameterLaunch&, const uint32_t* code, size_t dwords);
bool native_parameter_base_load(const NativeParameterBasePlan&, const Rdna2Inst&);
// Recursive regions may contain original instructions and synthetic END only. A reused PC or
// an unrelated stream cannot inherit the original program's closed interpolation service.
bool native_parameter_base_region(const NativeParameterBasePlan&, const std::vector<Rdna2Inst>&);
bool native_parameter_base_region(SpirvCompute&, const std::vector<Rdna2Inst>&);
bool emit_native_parameter_base_load(SpirvCompute&, RegState&, const Rdna2Inst&);
} // namespace prosper::gpu
