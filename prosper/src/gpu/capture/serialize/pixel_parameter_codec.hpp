#pragma once

#include <string>

namespace prosper::gpu {
struct Writer;
struct Reader;
struct GpuCaptureFile;
// v81: draw count and parameter-launch inputs, then failed-draw count and inputs.
// An older capture leaves these compiler arguments unavailable, including with raw observations.
bool write_pixel_parameter_launches(Writer&, const GpuCaptureFile&, std::string&);
bool read_pixel_parameter_launches(Reader&, GpuCaptureFile&, std::string&);
} // namespace prosper::gpu
