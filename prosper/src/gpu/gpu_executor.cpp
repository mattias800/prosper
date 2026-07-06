// gpu_executor.cpp — the live-submit half of the GPU executor (Stage A of docs/GPU_EXECUTOR_DESIGN.md).
//
// Holds the process-wide live render backend and drives it on each AGC submit. This is deliberately the
// ONLY place the executor touches process-global state; execute_gpustate() itself (gpu_execute.hpp) stays
// pure. No Vulkan here — the backend is a std::function injected by whoever owns a device (the runtime
// binary at startup, or a test via render_runner.h), so prosper_core links this without Vulkan.
#include "gpu_execute.hpp"
#include "videoout_present.hpp"   // present_write_frame
#include "agc_shader_layout.hpp"  // AgcShaderHeader + build_shader_resources
#include "pm4_registers.hpp"      // SPI_SHADER_USER_DATA_* offsets
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <vector>

// Look up a registered AGC shader header by its bound code address (hle_agc.cpp). Layout-compatible
// with gpu::AgcShaderHeader (file_header@0, user_data@0x08, code@0x10, type@0x5a).
extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr);

namespace prosper::gpu {
namespace {
LiveRenderFn g_live;   // empty until the runtime/test registers a device-backed renderer

// Read a 32-dword user-data SGPR block from a stage's register file. `base` = the stage's
// SPI_SHADER_USER_DATA_*_0 register offset; absent registers read as 0. 32 (not 16) because NGG merged
// shaders place descriptors in the extended user SGPRs s16..s31 (e.g. vertex buffers at s16/s18).
static constexpr uint32_t kUserSgprs = 32;
void read_user_sgprs(const std::unordered_map<uint32_t, uint32_t>& sh, uint32_t base, uint32_t out[kUserSgprs]) {
    for (uint32_t i = 0; i < kUserSgprs; i++) { auto it = sh.find(base + i); out[i] = it == sh.end() ? 0u : it->second; }
}

// Reassign a built table's bindings to the recompiler+backend convention: constant buffer -> binding 2
// (cbuf slot 0), vertex buffer -> binding 3 (cbuf slot 1 / vertex-fetch storage), textures -> 4,5,...
// (combined image-samplers). The recompiler routes cbuf slot by `binding>=3` and uses `binding`
// directly for image samplers, so these fixed assignments make provenance resolve to the right slot.
void assign_convention_bindings(ShaderResourceTable& t) {
    uint32_t tex_binding = 4;
    for (auto& r : t.resources) {
        switch (r.cls) {
            case ResourceClass::ConstantBuffer: r.binding = 2; break;
            case ResourceClass::VertexBuffer:   r.binding = 3; break;
            case ResourceClass::Texture:
            case ResourceClass::StorageImage:   r.binding = tex_binding++; break;
            default: break;
        }
    }
}
}

std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr, bool is_ps) {
    if (!code_addr) return nullptr;
    const auto* hdr = (const AgcShaderHeader*)prosper_agc_shader_header_for_code(code_addr);
    if (!hdr) return nullptr;
    namespace P = prosper::agc::Pm4;
    const bool log = getenv("PROSPER_GFXLOG") != nullptr;

    // The V#/T# descriptors live in the stage's user-data SGPR block. The pixel stage uses PS user
    // data; the vertex/geometry stage under NGG merges the ES program's descriptors into GS user data.
    // Try the stage's expected base, then the alternates — the exact stage-merge layout varies, so use
    // whichever base actually yields resources.
    uint32_t bases[3];
    if (is_ps) { bases[0] = P::SPI_SHADER_USER_DATA_PS_0; bases[1] = P::SPI_SHADER_USER_DATA_GS_0; bases[2] = P::SPI_SHADER_USER_DATA_VS_0; }
    else       { bases[0] = P::SPI_SHADER_USER_DATA_GS_0; bases[1] = P::SPI_SHADER_USER_DATA_VS_0; bases[2] = P::SPI_SHADER_USER_DATA_PS_0; }

    // PROSPER_RESDUMP: raw dump of the user-data struct + SGPR block per base, so the EUD layout
    // (which sharps have offset_dw>=16, and where the EUD pointer sits) can be read empirically.
    if (getenv("PROSPER_RESDUMP")) {
        const AgcShaderUserData* ud = hdr->user_data;
        fprintf(stderr, "[resdump] %s code=0x%llx type=%u ud=%p\n", is_ps ? "PS" : "VS",
                (unsigned long long)code_addr, hdr->type, (const void*)ud);
        if (ud) {
            fprintf(stderr, "[resdump]   eud_size_dw=%u srt_size_dw=%u direct_count=%u sharp_counts={%u,%u,%u,%u}\n",
                    ud->eud_size_dw, ud->srt_size_dw, ud->direct_resource_count,
                    ud->sharp_resource_count[0], ud->sharp_resource_count[1],
                    ud->sharp_resource_count[2], ud->sharp_resource_count[3]);
            for (int cat = 0; cat < 4; cat++) {
                const AgcShaderSharp* sh = ud->sharp_resource_offset[cat];
                if (!sh || !ud->sharp_resource_count[cat]) continue;
                fprintf(stderr, "[resdump]   sharp[%d] offset_dw:", cat);
                for (uint16_t s = 0; s < ud->sharp_resource_count[cat] && s < 12; s++)
                    fprintf(stderr, " %u%s", sh[s].offset_dw(), sh[s].empty() ? "(empty)" : "");
                fprintf(stderr, "\n");
            }
            if (ud->direct_resource_offset && ud->direct_resource_count) {
                fprintf(stderr, "[resdump]   direct offset_dw:");
                for (uint16_t t2 = 0; t2 < ud->direct_resource_count && t2 < 16; t2++)
                    fprintf(stderr, " [%u]=%u", t2, ud->direct_resource_offset[t2]);
                fprintf(stderr, "\n");
            }
        }
        for (uint32_t base : bases) {
            uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, base, sgprs);
            fprintf(stderr, "[resdump]   sgprs@0x%x:", base);
            for (uint32_t i = 0; i < kUserSgprs; i++) fprintf(stderr, " %08x", sgprs[i]);
            fprintf(stderr, "\n");
        }
        // ALL set sh registers (sorted) — finds where the user-data SGPRs actually landed, including
        // any at unexpected offsets (a wrong indirect-register decode would scatter them).
        fprintf(stderr, "[resdump]   %zu sh regs set; lowest 48 offsets:", st.sh.size());
        std::vector<uint32_t> keys; for (auto& kv : st.sh) if (kv.second) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size() && i < 48; i++) fprintf(stderr, " 0x%x", keys[i]);
        fprintf(stderr, "\n");
    }

    for (uint32_t base : bases) {
        uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, base, sgprs);
        ShaderResourceTable t = build_shader_resources(*hdr, sgprs, kUserSgprs);
        if (t.resources.empty()) continue;
        assign_convention_bindings(t);
        if (log) {
            fprintf(stderr, "[restab] %s code=0x%llx base=0x%x -> %zu resources:\n",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, base, t.resources.size());
            for (auto& r : t.resources)
                fprintf(stderr, "[restab]   cls=%u binding=%u addr=0x%llx size=%u %ux%u fmt=%u stride=%u\n",
                        (unsigned)r.cls, r.binding, (unsigned long long)r.gpu_addr, r.size,
                        r.width, r.height, (unsigned)r.format, r.stride);
        }
        return std::make_shared<ShaderResourceTable>(std::move(t));
    }
    if (log) fprintf(stderr, "[restab] %s code=0x%llx -> no resources in any user-data base\n",
                     is_ps ? "PS" : "VS", (unsigned long long)code_addr);
    return nullptr;
}

void set_submit_renderer(LiveRenderFn fn) { g_live = std::move(fn); }
bool have_submit_renderer()               { return static_cast<bool>(g_live); }

bool execute_and_present(const GpuState& st, uint32_t width, uint32_t height) {
    if (!g_live || st.draws.empty() || !width || !height) return false;
    // Bind the target dimensions and defer to the pure core, which recompiles the shaders from their
    // SHADER_PGM addresses and resolves fixed-function state before calling back into the live renderer.
    std::vector<uint8_t> px = execute_gpustate(st,
        [&](const std::vector<uint32_t>& vs, const std::vector<uint32_t>& fs,
            const ResolvedPipelineState& ps) { return g_live(vs, fs, ps, width, height); });
    if (px.size() != static_cast<size_t>(width) * height * 4) return false;   // recompile/render failed
    present_write_frame(px.data(), width, height);
    return true;
}

} // namespace prosper::gpu
