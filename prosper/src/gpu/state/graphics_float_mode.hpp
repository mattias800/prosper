// Read actual launch-register presence without manufacturing a shader mode from reset defaults.
#pragma once
#include "gpu/recompiler/graphics_float_mode.hpp"
namespace prosper::gpu {
struct GpuState;
enum class GraphicsHardwareStage { Vertex, Geometry, Hull };
GraphicsFloatMode graphics_float_mode(const GpuState&, GraphicsHardwareStage);
GraphicsFloatMode vertex_graphics_float_mode(const GpuState&);
} // namespace prosper::gpu
