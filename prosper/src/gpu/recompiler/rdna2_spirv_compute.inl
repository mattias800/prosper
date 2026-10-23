#ifndef PROSPER_GPU_RECOMPILER_SPIRV_COMPUTE_DECLARATION_SCOPE
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"   // IWYU pragma: export
#else
#ifndef PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_INL
#define PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_INL
#define PROSPER_GPU_RECOMPILER_SPIRV_COMPUTE_MEMBER_SCOPE
struct SpirvCompute {
#include "gpu/recompiler/rdna2_spirv_compute_state.inl" // IWYU pragma: keep
#include "gpu/recompiler/rdna2_spirv_compute_value_ops.inl" // IWYU pragma: keep
#include "gpu/recompiler/rdna2_spirv_compute_resources.inl" // IWYU pragma: keep
#include "gpu/recompiler/rdna2_spirv_compute_workgroup_stages.inl" // IWYU pragma: keep
};
#undef PROSPER_GPU_RECOMPILER_SPIRV_COMPUTE_MEMBER_SCOPE
#endif
#endif
