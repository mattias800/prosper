// v80 tail: u32 compute count + allocation per compute, then diagnostic/stage counts + allocations.
// Zero is unknown. Each nonzero allocation must match the retained config's RSRC1 and wave size.
#pragma once
#include <string>
namespace prosper::gpu {
struct Writer;
struct Reader;
struct GpuCaptureFile;
bool write_compute_vgpr_allocations(Writer&, const GpuCaptureFile&, std::string&);
bool read_compute_vgpr_allocations(Reader&, GpuCaptureFile&, std::string&);
}   // namespace prosper::gpu
