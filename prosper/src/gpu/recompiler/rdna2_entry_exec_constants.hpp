#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::gpu {

// Analysis alone grants no emitter authority. The opaque root below is constructible only at
// ordinary compute's original entry, never by a phase, body retry, NGG shell or public slice.
std::vector<uint32_t> portable_entry_exec_constant_reads(const std::vector<Rdna2Inst>&,
                                                         const ComputeShaderConfig&);

class PortableEntryExecRoot {
    friend std::vector<uint32_t> recompile_compute(const uint32_t*, size_t,
                                                   const ShaderResourceTable*,
                                                   const ComputeShaderConfig&,
                                                   RecompileDiagnosticContext);
    std::vector<uint32_t> words_, reads_;
    std::vector<Rdna2Inst> original_;
    std::array<uint32_t, 9> domain_{};
    ComputeDispatchLaunchState launch_{};

    PortableEntryExecRoot(const uint32_t*, size_t, const ComputeShaderConfig&);
    PortableEntryExecRoot(const PortableEntryExecRoot&) = delete;
    PortableEntryExecRoot& operator=(const PortableEntryExecRoot&) = delete;

public:
    // Called before any transformation. Repeated original-entry compilation may use the same
    // immutable root, but a decoded edit, raw edit, slice, synthetic END or domain change cannot.
    void specialize(const uint32_t*, size_t, const ComputeShaderConfig&,
                    std::vector<Rdna2Inst>&) const;
};
}   // namespace prosper::gpu
