// Complete private allocations are caller contracts; legacy captures cannot infer them from raw RSRC1.
#include "gpu/capture/serialize/compute_vgpr_codec.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/capture/serialize/capture_codecs.hpp"
#include "gpu/recompiler/rdna2_relative_vgpr_source.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <string>

namespace prosper::gpu {
namespace {
bool valid(const ComputeShaderConfig& c, bool available) {
    return (!c.allocated_vgprs || available) &&
           compute_vgpr_allocation_is_canonical(c.allocated_vgprs, c.compute_pgm_rsrc1,
                                                c.wave_size);
}
}   // namespace
bool write_compute_vgpr_allocations(Writer& w, const GpuCaptureFile& c, std::string& error) {
    const auto write = [&](const auto& record) {
        if (!valid(record.recompile_config, record.recompile_config_available)) return false;
        w.u32(record.recompile_config.allocated_vgprs);
        return true;
    };
    w.u32(static_cast<uint32_t>(c.computes.size()));
    for (const auto& compute : c.computes)
        if (!write(compute)) {
            error = "invalid compute VGPR allocation";
            return false;
        }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        w.u32(static_cast<uint32_t>(diagnostic.stages.size()));
        for (const auto& stage : diagnostic.stages)
            if (!write(stage)) {
                error = "invalid failed-stage VGPR allocation";
                return false;
            }
    }
    return true;
}
bool read_compute_vgpr_allocations(Reader& r, GpuCaptureFile& c, std::string& error) {
    const auto read = [&](auto& record) {
        return r.u32(record.recompile_config.allocated_vgprs) &&
               valid(record.recompile_config, record.recompile_config_available);
    };
    uint32_t count = 0;
    if (!r.u32(count) || count != c.computes.size()) {
        error = "invalid compute VGPR allocation count";
        return false;
    }
    for (auto& compute : c.computes)
        if (!read(compute)) {
            error = "invalid compute VGPR allocation";
            return false;
        }
    if (!r.u32(count) || count != c.failure_diagnostics.size()) {
        error = "invalid diagnostic VGPR allocation count";
        return false;
    }
    for (auto& diagnostic : c.failure_diagnostics) {
        if (!r.u32(count) || count != diagnostic.stages.size()) {
            error = "invalid stage VGPR allocation count";
            return false;
        }
        for (auto& stage : diagnostic.stages)
            if (!read(stage)) {
                error = "invalid failed-stage VGPR allocation";
                return false;
            }
    }
    return true;
}
}   // namespace prosper::gpu
