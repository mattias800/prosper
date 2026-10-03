#pragma once
#include "gpu/recompiler/fragment_draw_capacity.hpp"

namespace prosper::gpu {
// Collection/count/assembly run in the graphics device's ordered submission, never a CPU
// readback. First entry recipe supplies only independently owned per-draw scalar words; scratch
// VGPRs and unavailable masks retain zero availability. New recipes must carry actual authority.
// None of these shaders applies blend or mutates a guest attachment.
std::vector<uint32_t> build_fragment_draw_count(const FragmentDrawCapacity&,
                                                const RasterQuadCollector&);
std::vector<uint32_t> build_fragment_draw_assembly(const FragmentDrawCapacity&,
                                                   const RasterQuadCollector&);
// A bounded device-side transaction. The validator inspects EVERY logical worker before
// publishing word0 of the private commit buffer. Its pixel index stores references into the
// retained collector, never partially published coordinate/value records. Replay is pure
// geometry: this FS has no guest code, side effects or early tests and only transports original
// validated PS exports to the normal fixed-function attachment pipeline.
std::vector<uint32_t> build_fragment_draw_validation(const FragmentDrawCapacity&,
                                                     const RasterQuadCollector&);
std::vector<uint32_t> build_fragment_draw_replay(const FragmentDrawCapacity&,
                                                 const RasterQuadCollector&);
} // namespace prosper::gpu
