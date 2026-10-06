// live_rtt_dcc_plane.cpp -- the seam through which a compute pass tells the live renderer about a
// renderer-owned target's DCC control plane (declared in gpu_execute.hpp beside the other live
// renderer callbacks). Kept out of gpu_executor.cpp, which is past its line cap.
#include "gpu/execute/gpu_execute.hpp"

#include <utility>

namespace prosper::gpu {

namespace {
LiveRttDccPlaneRegistrarFn g_live_rtt_dcc_plane_registrar;
}   // namespace

void set_live_rtt_dcc_plane_registrar(LiveRttDccPlaneRegistrarFn fn) {
    g_live_rtt_dcc_plane_registrar = std::move(fn);
}

void register_live_rtt_dcc_plane(uint64_t gpu_addr, uint64_t metadata_addr,
                                 uint64_t metadata_bytes) {
    if (g_live_rtt_dcc_plane_registrar)
        g_live_rtt_dcc_plane_registrar(gpu_addr, metadata_addr, metadata_bytes);
}

}   // namespace prosper::gpu
