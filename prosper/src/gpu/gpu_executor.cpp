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
#include "rdna2_decode.hpp"       // rdna2_walk (for the vertex-fetch const-eval)
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_map>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

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

// Async-signal-safe-ish readability probe (guest memory is 1:1-mapped, but a mis-decoded address could
// be unmapped): write() to /dev/null returns EFAULT for an unmapped source, so we can test a guest
// address without risking a nested SIGSEGV on the render/submit thread. Always-true on Windows (the
// const-eval only runs on the live-render path, which is Linux; tests never pass wild addresses).
#ifndef _WIN32
static int g_devnull = -1;
bool guest_readable(uint64_t a, uint32_t n) {
    if (a < 0x1000) return false;
    if (g_devnull < 0) g_devnull = open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (g_devnull < 0) return false;
    return write(g_devnull, (const void*)(uintptr_t)a, n) == (ssize_t)n;
}
#else
bool guest_readable(uint64_t, uint32_t) { return true; }
#endif

// --- Bindless-dynamic vertex-fetch resolution (const-fold the scalar setup) ---------------------------
// This game's NGG vertex shader loads its vertex-buffer V# from a descriptor table at a RUNTIME-computed
// offset (e.g. `s_load_dwordx4 s[8:11], s[24:25], vcc_hi` where `vcc_hi = (s64<<4)&0x1f0` and
// `s64 = *[s26:27]`). The recompiler resolves descriptors by a STATIC provenance key, so it can't match a
// computed offset. We const-fold the wave-uniform scalar setup here: seed the concrete user-data SGPR
// values, interpret the scalar ALU + scalar loads (reading the 1:1-mapped guest memory), and snapshot the
// V# each descriptor load produces AT LOAD TIME (before the shader's later dynamic stride patch). The
// result maps each buffer_load_format's SRSRC SGPR -> its decoded V#, which build_stage_table emits as a
// VertexBuffer keyed by sgpr_base so the recompiler's by_sgpr_base() resolves it. Uniform-scalar only: any
// value that would depend on a VGPR/lane is left unknown (the op's dest becomes unknown), so we never
// fabricate a per-lane-dependent descriptor. CONFIDENCE: MED (covers this game's fetch-shader shape).
std::unordered_map<int, DecodedBufferDescriptor>
resolve_dynamic_fetch(const uint32_t* code, size_t dwords, const uint32_t* user_sgprs, uint32_t nsgpr,
                      uint32_t user_sgpr_base) {
    std::unordered_map<int, DecodedBufferDescriptor> out;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);

    const bool trc = getenv("PROSPER_DYNTRACE") != nullptr;
    std::unordered_map<int, uint32_t> val;                 // known concrete SGPR values (by SHADER SGPR #)
    std::unordered_map<int, std::array<uint32_t, 4>> descr; // SGPR -> the 4-dword V# it holds (load-time snapshot)
    // The SPI loads the user-data block starting at shader SGPR `user_sgpr_base` (s0..s7 are NGG system
    // SGPRs). So user-data block index k lands in shader SGPR (user_sgpr_base + k).
    for (uint32_t i = 0; i < nsgpr; i++) val[(int)(user_sgpr_base + i)] = user_sgprs[i];

    auto known = [&](int r, uint32_t& v) { auto it = val.find(r); if (it == val.end()) return false; v = it->second; return true; };
    // Resolve an ALU source operand to a concrete value (SGPR / inline int / literal / a vcc Special).
    // vcc_lo/hi (106/107) are written by ALU dsts as SGPR 106/107 but read back as Special operands with
    // the same field value, so map them onto the same val[] keys. Other Specials (EXEC/M0/...) stay unknown.
    auto srcval = [&](const Operand& o, uint32_t& v) -> bool {
        switch (o.kind) {
            case OperandKind::SGPR:      return known(o.value, v);
            case OperandKind::InlineInt: v = (uint32_t)o.value; return true;
            case OperandKind::Special:   return (o.value == 106 || o.value == 107) ? known(o.value, v) : false;
            case OperandKind::Literal:   v = 0; return false;   // literal is in in.literal; handled per-op
            default: return false;
        }
    };

    for (const auto& in : ins) {
        if (in.is_end) break;
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                if (in.opcode == 0x03) {                        // s_mov_b32
                    uint32_t v;
                    if (in.src[0].kind == OperandKind::Literal ? (v = in.literal, true) : srcval(in.src[0], v))
                        val[in.dst.value] = v;
                    else val.erase(in.dst.value);
                } else if (in.dst.kind == OperandKind::SGPR) { val.erase(in.dst.value); }
                break;
            case Rdna2Format::SOP2: {
                uint32_t a, c; bool ka, kc;
                ka = (in.src[0].kind == OperandKind::Literal) ? (a = in.literal, true) : srcval(in.src[0], a);
                kc = (in.src[1].kind == OperandKind::Literal) ? (c = in.literal, true) : srcval(in.src[1], c);
                int d = in.dst.value; bool ok = ka && kc; uint32_t r = 0;
                if (ok) switch (in.opcode) {
                    case 0x00: case 0x02: r = a + c; break;                 // s_add_u32 / s_add_i32
                    case 0x01: case 0x03: r = a - c; break;                 // s_sub_u32 / s_sub_i32
                    case 0x0E: r = a & c; break;                            // s_and_b32
                    case 0x10: r = a | c; break;                            // s_or_b32
                    case 0x12: r = a ^ c; break;                            // s_xor_b32
                    case 0x1E: r = a << (c & 31); break;                    // s_lshl_b32
                    case 0x20: r = a >> (c & 31); break;                    // s_lshr_b32
                    case 0x26: r = a * c; break;                            // s_mul_i32
                    case 0x31: r = (a << 4) + c; break;                     // s_lshl4_add_u32
                    case 0x27: { uint32_t off = c & 0x1f, wid = (c >> 16) & 0x7f;   // s_bfe_u32
                                 r = wid == 0 ? 0 : (wid >= 32 ? (a >> off) : ((a >> off) & ((1u << wid) - 1))); break; }
                    default: ok = false; break;                            // SCC-dependent / unmodeled -> unknown
                }
                if (ok) val[d] = r; else val.erase(d);
                break;
            }
            case Rdna2Format::SMEM: {
                // SBASE (src[0]) is a 2-dword pointer (s_load, op<8) or a 4-dword V# (s_buffer_load, op>=8).
                // Address = base + immediate OFFSET (in.literal) + SOFFSET register value (decoded here from
                // words[1][31:25]; the shared decoder doesn't expose it). Dword count from the opcode.
                uint32_t n = 0; bool is_buffer = in.opcode >= 8;
                switch (in.opcode & 7) { case 0: n = 1; break; case 1: n = 2; break; case 2: n = 4; break;
                                         case 3: n = 8; break; case 4: n = 16; break; default: n = 0; }
                int sbase = in.src[0].value, sdst = in.dst.value;
                uint32_t soff_field = (in.words[1] >> 25) & 0x7Fu;         // SOFFSET SGPR (125 = null)
                uint32_t soff_val = 0; bool soff_ok = true;
                if (soff_field < 106) soff_ok = known((int)soff_field, soff_val);          // SGPR
                else if (soff_field == 106 || soff_field == 107) soff_ok = known((int)soff_field, soff_val); // vcc lo/hi
                else soff_val = 0;                                          // null/const-0 soffset
                uint64_t base = 0; bool base_ok;
                if (is_buffer) { uint32_t b0, b1; base_ok = known(sbase, b0) && known(sbase + 1, b1);
                                 base = ((uint64_t)b0 | ((uint64_t)b1 << 32)) & 0xFFFFFFFFFFFFull; }   // V#.Base48
                else { uint32_t p0, p1; base_ok = known(sbase, p0) && known(sbase + 1, p1);
                       base = (uint64_t)p0 | ((uint64_t)p1 << 32); }        // raw pointer
                if (trc) fprintf(stderr, "[dyntrace] SMEM op=0x%x %s sdst=s%d sbase=s%d base=0x%llx base_ok=%d "
                                 "soff_field=%u soff_val=0x%x soff_ok=%d imm=0x%x n=%u\n", in.opcode,
                                 is_buffer ? "bufload" : "load", sdst, sbase, (unsigned long long)base, base_ok,
                                 soff_field, soff_val, soff_ok, in.literal, n);
                if (n == 0 || !base_ok || !soff_ok) { for (uint32_t k = 0; k < n; k++) val.erase(sdst + (int)k); break; }
                uint64_t addr = base + in.literal + soff_val;
                if (!guest_readable(addr, n * 4)) { if (trc) fprintf(stderr, "[dyntrace]   addr 0x%llx unreadable\n", (unsigned long long)addr);
                                                    for (uint32_t k = 0; k < n; k++) val.erase(sdst + (int)k); break; }
                const uint32_t* mem = (const uint32_t*)(uintptr_t)addr;
                for (uint32_t k = 0; k < n; k++) val[sdst + (int)k] = mem[k];
                // A 4-dword load is a V# candidate — snapshot it now (before any later stride patch) so a
                // vertex fetch using these SGPRs resolves to the descriptor as loaded.
                if (n == 4) descr[sdst] = { mem[0], mem[1], mem[2], mem[3] };
                break;
            }
            case Rdna2Format::MUBUF: {
                // buffer_load_format_* (vertex fetch): opcodes 0..3. Resolve the SRSRC (src[1]) SGPR to the
                // V# most-recently loaded into it.
                if (in.opcode <= 3) {
                    int srsrc = in.src[1].value;
                    auto it = descr.find(srsrc);
                    if (trc) fprintf(stderr, "[dyntrace] MUBUF fetch op=0x%x SRSRC=s%d have_descr=%d\n",
                                     in.opcode, srsrc, it != descr.end());
                    if (it != descr.end()) out[srsrc] = decode_buffer_descriptor(it->second.data());
                }
                break;
            }
            default:
                if (in.dst.kind == OperandKind::SGPR) val.erase(in.dst.value);   // unmodeled scalar write -> unknown
                break;
        }
    }
    return out;
}

// Give every resource its OWN descriptor binding, starting at 2 (0/1 reserved). The N-buffer model: the
// shader reads several distinct constant buffers (Unity's per-draw transform, per-frame, …) + vertex
// buffers + textures, and each must land at a separate binding so they don't collapse. The recompiler
// declares a storage buffer (cbuf/vbuf) or image sampler (texture) at each binding and resolves an
// s_buffer_load/image_sample to its resource's binding via provenance (by_sgpr_base / by_srt_offset).
void assign_convention_bindings(ShaderResourceTable& t) {
    uint32_t next = 2;
    for (auto& r : t.resources) r.binding = next++;
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

    // Bindless-dynamic vertex fetch (VS): const-fold the scalar setup to resolve each buffer_load_format's
    // V#, then emit it as a VertexBuffer keyed by its SRSRC SGPR so the recompiler's by_sgpr_base resolves
    // it. Only meaningful for the vertex stage (the PS has no vertex fetch).
    std::unordered_map<int, DecodedBufferDescriptor> dyn_vb;
    if (!is_ps) {
        uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, bases[0], sgprs);
        // NGG merged VS/GS: s0..s7 are system SGPRs, user data starts at s8 (confirmed by matching the
        // shader's s[8:11]/s[24:25] descriptor pointers to the register file at GS_0+offset).
        dyn_vb = resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, 0x4000, sgprs, kUserSgprs, 8);
        if (getenv("PROSPER_GFXLOG") || getenv("PROSPER_RESDUMP")) {
            fprintf(stderr, "[dynvb] VS resolved %zu dynamic vertex-fetch descriptor(s):\n", dyn_vb.size());
            for (auto& kv : dyn_vb) {
                const auto& d = kv.second;
                fprintf(stderr, "[dynvb]   SRSRC s%d -> base=0x%llx stride=%u num_records=%u size=%u fmt=%u nc=%u\n",
                        kv.first, (unsigned long long)d.base, d.stride, d.num_records, d.size_bytes,
                        (unsigned)d.format, d.num_components);
            }
        }
    }

    // NGG VS/GS loads user data at shader s8 (s0..s7 = system SGPRs); PS at s0. The resource sgpr_base
    // (an s_buffer_load/image_sample's SBASE/SRSRC register) is in that shader-SGPR space.
    const uint32_t user_sgpr_base = is_ps ? 0u : 8u;
    for (uint32_t base : bases) {
        uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, base, sgprs);
        ShaderResourceTable t = build_shader_resources(*hdr, sgprs, kUserSgprs, user_sgpr_base);
        // Add the const-fold-resolved dynamic vertex buffers, keyed by their SRSRC SGPR so the
        // recompiler's by_sgpr_base() resolves each buffer_load_format. The V#'s data format is patched
        // at runtime by the fetch shader (so the load-time snapshot reads Unknown) — default to Float32
        // (a raw 32-bit-per-component fetch, correct for float attributes like positions).
        for (auto& kv : dyn_vb) {
            const auto& d = kv.second;
            ShaderResource r;
            r.cls           = ResourceClass::VertexBuffer;
            r.format        = (d.format == DataFormat::Unknown) ? DataFormat::Float32 : d.format;
            r.num_components = d.num_components ? d.num_components : 4;
            r.gpu_addr      = d.base;
            r.size          = d.size_bytes ? d.size_bytes : (d.stride ? d.stride * 4 : 128);
            r.stride        = d.stride;
            r.sgpr_base     = kv.first;           // DIRECT provenance = the fetch's SRSRC SGPR
            r.srt_offset    = 0xFFFFFFFFu;
            t.resources.push_back(r);
        }
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
            const ResolvedPipelineState& ps, const ShaderResourceTable* vrt,
            const ShaderResourceTable* prt, uint32_t vcount) { return g_live(vs, fs, ps, vrt, prt, width, height, vcount); });
    if (px.size() != static_cast<size_t>(width) * height * 4) return false;   // recompile/render failed
    present_write_frame(px.data(), width, height);
    return true;
}

} // namespace prosper::gpu
