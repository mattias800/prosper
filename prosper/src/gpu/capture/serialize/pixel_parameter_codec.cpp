#include "gpu/capture/serialize/pixel_parameter_codec.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/capture/serialize/capture_codecs.hpp"

#include <cstdint>

namespace prosper::gpu {
namespace {
bool valid(const PixelSystemInputMapping& inputs, bool has_system_inputs) {
    const auto& launch = inputs.parameter_launch;
    return launch.canonical() && (has_system_inputs || launch == PixelParameterLaunch{}) &&
           (!has_system_inputs || inputs.ena || inputs.addr || launch.available);
}
} // namespace

bool write_pixel_parameter_launches(Writer& w, const GpuCaptureFile& c, std::string& error) {
    const auto write = [&](const auto& record) {
        const auto& launch = record.system_inputs.parameter_launch;
        if (!valid(record.system_inputs, record.has_system_inputs)) return false;
        w.u8(launch.available ? 1u : 0u);
        w.u32(launch.value);
        return true;
    };
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws)
        if (!write(draw)) {
            error = "invalid draw parameter-launch input";
            return false;
        }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics)
        if (!write(diagnostic)) {
            error = "invalid failed-draw parameter-launch input";
            return false;
        }
    return true;
}

bool read_pixel_parameter_launches(Reader& r, GpuCaptureFile& c, std::string& error) {
    const auto read = [&](auto& record) {
        auto& launch = record.system_inputs.parameter_launch;
        uint8_t available = 0;
        if (!r.u8(available) || available > 1 || !r.u32(launch.value)) return false;
        launch.available = available != 0;
        return valid(record.system_inputs, record.has_system_inputs);
    };
    uint32_t count = 0;
    if (!r.u32(count) || count != c.draws.size()) {
        error = "invalid draw parameter-launch count";
        return false;
    }
    for (auto& draw : c.draws)
        if (!read(draw)) {
            error = "invalid draw parameter-launch input";
            return false;
        }
    if (!r.u32(count) || count != c.failure_diagnostics.size()) {
        error = "invalid failed-draw parameter-launch count";
        return false;
    }
    for (auto& diagnostic : c.failure_diagnostics)
        if (!read(diagnostic)) {
            error = "invalid failed-draw parameter-launch input";
            return false;
        }
    return true;
}
} // namespace prosper::gpu
