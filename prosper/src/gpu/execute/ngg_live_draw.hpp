// ngg_live_draw.hpp -- the live producer of merged ES+GS NGG draws (#3135 phase P5).
//
// realize_draw_item calls realize_ngg_live_draw at exactly one point: where a draw's linked vertex
// chain has just failed to recompile through the ordinary per-vertex path (the vertex-empty failure
// point). So the path is strictly additive: a draw prosper renders today never reaches it, and a
// draw it refuses is dropped exactly as before, now with the refusal named.
//
// The producer runs admission (ngg_draw_admission.hpp), plans the draw and lays it out over
// compiled stages from a bounded cache, then asks the device admission the backend will ask, so a
// description it returns is one the backend runs.
//
// CACHING (CLAUDE.md P3). Compiling the shell and the raster stages is the expensive part. They are
// cached per (compile inputs, W), and the compile inputs are compared EXACTLY: the linked program's
// words, the resource table's compile shape (every field the recompiler reads, never an address or
// host byte -- the partition the ordinary shader cache's key makes, taken conservatively wide), the
// pixel-input mapping, and the admission configuration (user SGPR count, LDS granules, native
// Wave64, output topology, provoking vertex, layer read and slice count, route, violation counting,
// interpolation layout, float transport). The ordinary shader cache's identity cannot stand in:
// that path refuses a chain with owned raw inputs (Kena's) before it assigns one.
// A refused compile is cached too, so a refused program is not recompiled per draw. Compilation
// happens at first use inside realization, which runs on the draw realization workers like every
// other shader compile in prosper; once warm, no draw compiles. The planner and the launch records
// are rebuilt per distinct draw shape, and an assembled description is reused while the shape and
// the push-constant words repeat. Both caches are bounded (least recently used is evicted).
#pragma once

#include "gpu/execute/ngg_draw_admission.hpp"
#include "gpu/execute/ngg_subgroup_draw.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {

struct GpuState;

struct NggLiveDrawInput {
    NggDrawRegisters registers;
    NggDrawFacts facts;
    // SPI_SHADER_USER_DATA_GS_0.. for the program's user-data range; `user_data_complete` is false
    // when any of those registers is absent.
    std::vector<uint32_t> user_data;
    bool user_data_complete = false;
    // The chain as the hardware runs it: the prolog up to its s_setpc, then the main program.
    const uint32_t* prolog = nullptr;
    size_t prolog_prefix_dwords = 0;
    const uint32_t* main = nullptr;
    size_t main_dwords = 0;
    const ShaderResourceTable* resources = nullptr;
    const PixelInputMapping* pixel_inputs = nullptr;
    FragmentInterpolationLayout interpolation;
    FloatTransportConfig float_transport;
    uint64_t program_address = 0;   // diagnostics only
};

struct NggLiveDrawResult {
    std::shared_ptr<const NggSubgroupDraw> draw;
    bool applies = false;   // a merged ES+GS NGG draw: the path was tried
    const char* refusal = nullptr;   // the rule that refused (a static string), null on success
    std::string detail;   // the compiler's full refusal text, when it refused
};

// Reads the registers admission takes from the draw's state.
NggDrawRegisters read_ngg_draw_registers(const GpuState& state, uint32_t primitive_type);
// The user-data words s8.. for `count` user SGPRs; false when one is absent.
bool read_ngg_user_data(const GpuState& state, uint32_t count, std::vector<uint32_t>* words);

NggLiveDrawResult realize_ngg_live_draw(const NggLiveDrawInput& input,
                                        const NggHostCapabilities& host);

struct NggLiveDrawCacheStats {
    uint64_t stage_hits = 0, stage_compiles = 0, stage_evictions = 0;
    uint64_t draw_hits = 0, draw_assemblies = 0;
};
NggLiveDrawCacheStats ngg_live_draw_cache_stats();
void reset_ngg_live_draw_cache_for_test();

}   // namespace prosper::gpu
