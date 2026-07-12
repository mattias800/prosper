// gpu_executor.cpp — the live-submit half of the GPU executor (Stage A of docs/GPU_EXECUTOR_DESIGN.md).
//
// Holds the process-wide live render backend and drives it on each AGC submit. This is deliberately the
// ONLY place the executor touches process-global state; execute_gpustate() itself (gpu_execute.hpp) stays
// pure. No Vulkan here — the backend is a std::function injected by whoever owns a device (the runtime
// binary at startup, or a test via render_runner.h), so prosper_core links this without Vulkan.
#include "gpu_execute.hpp"
#include "gpu_capture.hpp"
#include "videoout_present.hpp"   // present_write_frame
#include "agc_shader_layout.hpp"  // AgcShaderHeader + build_shader_resources
#include "pm4_registers.hpp"      // SPI_SHADER_USER_DATA_* offsets
#include "rdna2_decode.hpp"       // rdna2_walk (for the vertex-fetch const-eval)
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <set>
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

} // namespace (guest_readable below has external linkage — declared in gpu_execute.hpp, shared with
  // the HLE diagnostic probes that chase raw guest pointers)

// PROSPER_DYNTRACE_FAIL support: while true, resolve_dynamic_fetch traces its walk and
// build_stage_table dumps the user-data blocks, regardless of the PROSPER_DYNTRACE/RESDUMP
// envs. Set (and cleared) by realize_draw_item's failure replay — the submit path is serialized
// by the HLE submit mutex, so a plain global is safe there.
bool g_dyntrace_force = false;

// Readability probe (guest memory is 1:1-mapped, but a mis-decoded address could be unmapped),
// so guarded derefs on the render/submit thread don't risk a SIGSEGV. NOTE: /dev/null does NOT
// work for this — the kernel's null_write returns count without ever touching the source buffer,
// so the old probe reported EVERY address >= 0x1000 "readable" and all guards built on it were
// no-ops (verified empirically on this project's WSL kernel). A pipe write actually imports the
// user pages and returns EFAULT for unmapped memory. Readability is page-granular: probe one byte
// in each page the range touches, draining after every write so the pipe can never fill. Always-
// true on Windows (the const-eval only runs on the live-render path, which is Linux; tests never
// pass wild addresses).
#ifndef _WIN32
static int g_probe_pipe[2] = {-1, -1};
// One-time pipe creation via a C++11 magic static (thread-safe). guest_readable is shared with
// the multi-threaded HLE pointer probes: an unguarded `if (fd < 0) pipe2(...)` lazy init let two
// first-callers each pipe2 into the array — a torn pair (write-end of pipe B, read-end of pipe A)
// never drains, fills, and EAGAINs: VALID memory reported unreadable, silently. (PR #61 review.)
static bool probe_pipe_ok() {
    static const bool ok = pipe2(g_probe_pipe, O_CLOEXEC | O_NONBLOCK) == 0;
    return ok;
}
static bool probe_byte(uint64_t a) {
    ssize_t w = write(g_probe_pipe[1], (const void*)(uintptr_t)a, 1);
    if (w == 1) { char c; (void)!read(g_probe_pipe[0], &c, 1); return true; }
    return false;   // EFAULT: unmapped (EAGAIN can't happen — we drain after every byte)
}
bool guest_readable(uint64_t a, uint32_t n) {
    if (a < 0x1000 || n == 0) return false;
    if (a + n < a) return false;   // wrap
    if (!probe_pipe_ok()) return false;
    uint64_t last_page = (a + n - 1) & ~0xfffull;
    for (uint64_t p = a & ~0xfffull; p <= last_page; p += 0x1000)
        if (!probe_byte(p < a ? a : p)) return false;
    return true;
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
// External linkage (DynFetch + declaration in gpu_execute.hpp) so the fold is unit-testable.
std::vector<DynFetch>
resolve_dynamic_fetch(const uint32_t* code, size_t dwords, const uint32_t* user_sgprs, uint32_t nsgpr,
                      uint32_t user_sgpr_base, std::vector<SrtUse>* srt_uses) {
    std::vector<DynFetch> out;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);

    // PROSPER_DYNTRACE traces the whole const-fold walk; PROSPER_DYNTRACE_ADDR=<hex code addr>
    // narrows it to ONE shader (a full run otherwise traces every draw's walk — unusable volume).
    // g_dyntrace_force: set by the PROSPER_DYNTRACE_FAIL failure-replay path (gpu_execute.hpp) so
    // the walk of a shader that just FAILED to recompile is traced without knowing its address.
    bool trc = g_dyntrace_force || getenv("PROSPER_DYNTRACE") != nullptr;
    if (trc && !g_dyntrace_force)
        if (const char* fa = getenv("PROSPER_DYNTRACE_ADDR"))
            trc = strtoull(fa, nullptr, 16) == (uint64_t)(uintptr_t)code;
    std::unordered_map<int, uint32_t> val;                 // known concrete SGPR values (by SHADER SGPR #)
    std::unordered_map<int, std::array<uint32_t, 4>> descr; // SGPR -> the 4-dword V# it holds (load-time snapshot)
    // Descriptor-TABLE provenance (#294): for each snapshotted 4/8-dword s_load, the load's IMMEDIATE
    // byte offset — the recompiler's sreg_srt/by_srt_offset key. 0xFFFFFFFF = not provenance-usable
    // (register-SOFFSET or negative-immediate load, which emit_alu doesn't tag).
    std::unordered_map<int, uint32_t> descr_key;
    std::unordered_map<int, std::array<uint32_t, 8>> descr8;  // SGPR -> 8-dword T# (load-time snapshot)
    std::unordered_map<int, uint32_t> descr8_key;
    // SGPRs overwritten by an s_load since seeding — the seed-V# MUBUF fallback below must not use a
    // stale user-data snapshot once the register was RELOADED from memory (ALU patches deliberately
    // don't count: descriptor snapshots are load-time semantics, pre-patch, like `descr`).
    std::set<int> reloaded;
    int scc = -1;   // tracked SCC (-1 unknown): set by s_cmp_*, consumed by s_cselect (the format patch's tail)
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
                } else if (in.dst.kind == OperandKind::SGPR) {
                    // Not the modeled s_mov_b32 -> the dest is unknown. Erase the PAIR: 64-bit SOP1 ops
                    // (s_mov_b64, s_getpc_b64, s_and/or/xor/not_b64, s_*_saveexec_b64, …) write
                    // S[dst:dst+1], so leaving a stale "known" val[dst+1] let a later instruction fold a
                    // confidently-wrong 64-bit base/offset -> a wrong V#/T# read from the wrong guest
                    // address (#460). Over-erasing dst+1 for a 32-bit SOP1 only loses a fold opportunity
                    // (never fabricates a value) — matching the SOP2 s_bfe_u64 pair-erase.
                    val.erase(in.dst.value);
                    val.erase(in.dst.value + 1);
                }
                // Several SOP1 ops write SCC (s_abs_i32, s_not_b32, s_and_saveexec_*, …). Only the moves
                // (s_mov_b32 0x03 / s_mov_b64 0x04) are known not to — anything else invalidates the
                // tracked SCC, or a later s_cselect folds with a stale compare result.
                if (in.opcode != 0x03 && in.opcode != 0x04) scc = -1;
                break;
            case Rdna2Format::SOP2: {
                uint32_t a, c; bool ka, kc;
                ka = (in.src[0].kind == OperandKind::Literal) ? (a = in.literal, true) : srcval(in.src[0], a);
                kc = (in.src[1].kind == OperandKind::Literal) ? (c = in.literal, true) : srcval(in.src[1], c);
                int d = in.dst.value; bool ok = ka && kc; uint32_t r = 0;
                uint32_t hi64 = 0; bool wrote_pair = false;
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
                    case 0x0A:   // s_cselect_b32: dst = SCC ? src0 : src1 (the vertex-fetch format patch's tail)
                        if (scc < 0) ok = false; else r = scc ? a : c;
                        break;
                    case 0x29: {  // s_bfe_u64: dst[63:0] = bitfield of src0[63:0] (format patch reads a small field)
                        uint32_t off = c & 0x3f, wid = (c >> 16) & 0x7f, ahi = 0;
                        // src0's high dword (RDNA2 ISA 64-bit scalar-operand rules, #155): the next SGPR
                        // of the pair; SIGN-extension of an integer inline constant (-1..-16 read as a
                        // 64-bit operand are all-ones in the high dword); or 0 only for a 32-bit literal
                        // (zero-extended). Inline FLOAT constants read as 64-bit doubles (a different bit
                        // pattern entirely) — srcval() already leaves those unknown, so they never reach
                        // here. Only an SGPR operand may index val[value+1] — a literal's `value` is not
                        // an SGPR number. And an UNTRACKED high dword must not silently fold as 0: if the
                        // field reaches bits >= 32 the result is unknown.
                        bool khi;
                        if (in.src[0].kind == OperandKind::SGPR)           khi = known(in.src[0].value + 1, ahi);
                        else if (in.src[0].kind == OperandKind::InlineInt) { ahi = in.src[0].value < 0 ? 0xFFFFFFFFu : 0u; khi = true; }
                        else                                               { ahi = 0; khi = true; }   // 32-bit literal
                        if (!khi && wid != 0 && off + wid > 32) { ok = false; wrote_pair = true; break; }
                        uint64_t src64 = (uint64_t)a | ((uint64_t)ahi << 32);
                        uint64_t res = wid == 0 ? 0 : (wid >= 64 ? (src64 >> off) : ((src64 >> off) & (((uint64_t)1 << wid) - 1)));
                        r = (uint32_t)res; hi64 = (uint32_t)(res >> 32); wrote_pair = true; break;
                    }
                    default: ok = false; break;                            // SCC-dependent / unmodeled -> unknown
                }
                if (trc)   // unfiltered like the SMEM/MUBUF traces (one shader walk — volume is bounded)
                    fprintf(stderr, "[dyntrace]   SOP2 pc=%u op=0x%x dst=s%d src0=%d(k%d) src1=%d(k%d) ok=%d r=0x%x\n",
                            in.pc, in.opcode, d, in.src[0].value, ka, in.src[1].value, kc, ok, r);
                // Every SOP2 ALU op except s_cselect writes SCC to a value we don't track here — invalidate so a
                // later s_cselect only trusts SCC set by an immediately-preceding s_cmp.
                if (in.opcode != 0x0A) scc = -1;
                if (ok) { val[d] = r; if (wrote_pair) val[d + 1] = hi64; }
                // A 64-bit-dst op (s_bfe_u64) invalidates BOTH dwords even when its sources were
                // unknown (the opcode switch never ran, so wrote_pair may still be false).
                else    { val.erase(d); if (wrote_pair || in.opcode == 0x29) val.erase(d + 1); }
                break;
            }
            case Rdna2Format::SOPC: {   // scalar compare -> SCC (feeds the format patch's s_cselect)
                uint32_t a, c;
                bool ka = (in.src[0].kind == OperandKind::Literal) ? (a = in.literal, true) : srcval(in.src[0], a);
                bool kc = (in.src[1].kind == OperandKind::Literal) ? (c = in.literal, true) : srcval(in.src[1], c);
                if (ka && kc) switch (in.opcode) {
                    case 0x00: scc = (a == c); break;                            // s_cmp_eq_i32
                    case 0x01: scc = (a != c); break;                            // s_cmp_lg_i32
                    case 0x02: scc = ((int32_t)a >  (int32_t)c); break;          // s_cmp_gt_i32
                    case 0x03: scc = ((int32_t)a >= (int32_t)c); break;          // s_cmp_ge_i32
                    case 0x04: scc = ((int32_t)a <  (int32_t)c); break;          // s_cmp_lt_i32
                    case 0x05: scc = ((int32_t)a <= (int32_t)c); break;          // s_cmp_le_i32
                    case 0x06: scc = (a == c); break;                            // s_cmp_eq_u32
                    case 0x07: scc = (a != c); break;                            // s_cmp_lg_u32
                    case 0x08: scc = (a >  c); break;                            // s_cmp_gt_u32
                    case 0x09: scc = (a >= c); break;                            // s_cmp_ge_u32
                    case 0x0A: scc = (a <  c); break;                            // s_cmp_lt_u32
                    case 0x0B: scc = (a <= c); break;                            // s_cmp_le_u32
                    default: scc = -1; break;
                } else scc = -1;
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
                else if (soff_field == 125) soff_val = 0;                   // SGPR_NULL -> const-0 offset (ok)
                else soff_ok = false;                                       // m0(124)/exec(126,127)/reserved:
                                                                            // untracked -> mark UNKNOWN, not 0. The
                                                                            // old `else soff_val=0` claimed ok=true
                                                                            // for these, snapshotting a descriptor
                                                                            // from base+imm+0 instead of +m0/exec
                                                                            // -> wrong V#/T#, silently (#398).
                // Descriptor-table use (#294): an s_buffer_load's SBASE is a V# — if that V# was
                // snapshotted from a table load, report it as a ConstantBuffer use keyed by its load
                // immediate (matching the recompiler's sreg_srt tag). Recorded BEFORE the dest write
                // below (SBASE and SDST ranges may overlap).
                if (srt_uses && is_buffer) {
                    auto dit = descr.find(sbase); auto kit = descr_key.find(sbase);
                    if (dit != descr.end() && kit != descr_key.end() && kit->second != 0xFFFFFFFFu) {
                        SrtUse u; u.kind = 1; u.key = kit->second; u.v4 = dit->second;
                        srt_uses->push_back(u);
                    }
                }
                for (uint32_t k = 0; k < n; k++) reloaded.insert(sdst + (int)k);   // dest now holds memory data
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
                // in.literal is the SIGN-EXTENDED 21-bit immediate (#149) — add it as signed so a
                // negative offset subtracts from the base instead of wrapping to a huge address.
                uint64_t addr = base + (uint64_t)(int64_t)(int32_t)in.literal + soff_val;
                if (!guest_readable(addr, n * 4)) { if (trc) fprintf(stderr, "[dyntrace]   addr 0x%llx unreadable\n", (unsigned long long)addr);
                                                    for (uint32_t k = 0; k < n; k++) val.erase(sdst + (int)k); break; }
                const uint32_t* mem = (const uint32_t*)(uintptr_t)addr;
                for (uint32_t k = 0; k < n; k++) val[sdst + (int)k] = mem[k];
                // Provenance key: the recompiler tags an IMMEDIATE-only descriptor load's dest SGPRs
                // with the load immediate (sreg_srt = in.literal); register-SOFFSET / negative loads
                // are not tagged, so mark those snapshots key-less.
                const bool imm_only = (soff_field == 125) && (int32_t)in.literal >= 0;   // SGPR_NULL soffset
                // A 4-dword load is a V# candidate — snapshot it now (before any later stride patch) so a
                // vertex fetch using these SGPRs resolves to the descriptor as loaded. An 8-dword load is
                // a T# candidate (image_sample SRSRC), snapshotted the same way (#294).
                if (n == 4) { descr[sdst] = { mem[0], mem[1], mem[2], mem[3] };
                              // only s_load (not s_buffer_load) dests get the recompiler's sreg_srt tag
                              descr_key[sdst] = (imm_only && !is_buffer) ? in.literal : 0xFFFFFFFFu; }
                if (n == 8) { descr8[sdst] = { mem[0], mem[1], mem[2], mem[3], mem[4], mem[5], mem[6], mem[7] };
                              descr8_key[sdst] = (imm_only && !is_buffer) ? in.literal : 0xFFFFFFFFu; }
                break;
            }
            case Rdna2Format::MIMG: {
                // Descriptor-table use (#294): an image op's SRSRC (src[1]) is an 8-dword T#; if it was
                // snapshotted from a table load, report it as a Texture use — with the paired SSAMP
                // (src[2]) S# when that 4-dword load also resolved. VGPR-only dest: no SGPR state.
                // Key-less snapshots (register-SOFFSET loads) are reported too (#273): the use carries
                // its instruction pc, which the recompiler resolves via ShaderResource::fetch_pc when
                // the immediate-key model fails or collides.
                if (srt_uses) {
                    auto tit = descr8.find(in.src[1].value); auto kit = descr8_key.find(in.src[1].value);
                    if (trc) fprintf(stderr, "[dyntrace] MIMG pc=%u srsrc=s%d ssamp=s%d have_t8=%d key=0x%x t8[0]=0x%x\n",
                                     in.pc, in.src[1].value, in.src[2].value, tit != descr8.end(),
                                     kit != descr8_key.end() ? kit->second : 0xEEEEEEEEu,
                                     tit != descr8.end() ? tit->second[0] : 0u);
                    if (tit != descr8.end()) {
                        SrtUse u; u.kind = 0; u.t8 = tit->second;
                        u.key = kit != descr8_key.end() ? kit->second : 0xFFFFFFFFu;
                        u.use_pc = in.pc;
                        auto sit = descr.find(in.src[2].value);
                        if (sit != descr.end()) { u.has_samp = true; u.s4 = sit->second; }
                        srt_uses->push_back(u);
                    }
                }
                break;
            }
            case Rdna2Format::MUBUF: {
                // Descriptor-table V# use (#273 — the title post-chain PSes' per-lane structured-buffer
                // fetches): a buffer_load_format_* / buffer_load_dword* whose SRSRC V# was s_loaded from
                // a keyed table slot. Report it as a kind-1 (buffer) use so the consuming instruction
                // resolves via its sreg_srt tag -> by_srt_offset — the same key model the s_buffer_loads
                // use. (The VS vertex-fetch path below is unchanged: by_fetch_pc still wins there.)
                if (srt_uses && (in.opcode <= 3 || (in.opcode >= 0x0C && in.opcode <= 0x0F))) {
                    int srsrc = in.src[1].value;
                    auto dit = descr.find(srsrc); auto kit = descr_key.find(srsrc);
                    if (dit != descr.end()) {
                        // Key-less snapshots (an s_buffer_load-fetched V# — a structured-buffer
                        // descriptor stored INSIDE a constant buffer, DOLL's title post PSes) carry
                        // the consuming instruction's pc instead; the recompiler resolves those via
                        // ShaderResource::fetch_pc with the faithful (non-vertex-index) address path.
                        SrtUse u; u.kind = 1; u.v4 = dit->second;
                        u.key = kit != descr_key.end() ? kit->second : 0xFFFFFFFFu;
                        u.use_pc = in.pc;
                        srt_uses->push_back(u);
                    }
                }
                // buffer_load_format_* (vertex fetch): opcodes 0..3. Resolve the SRSRC (src[1]) SGPR to the
                // V# most-recently loaded into it.
                if (in.opcode <= 3) {
                    int srsrc = in.src[1].value;
                    // Prefer the FETCH-TIME V#: the fetch shader patches the descriptor's format field (v[3])
                    // between load and fetch — so read the CURRENT SGPR values (which the interpreter has
                    // tracked through the patch, incl. the s_cselect tail) to get the real data format (e.g.
                    // UNORM8 for a packed vertex color, vs the load-time Unknown). Fall back to the load-time
                    // snapshot if the patched dwords aren't fully known.
                    uint32_t vv[4]; bool k0 = known(srsrc, vv[0]), k1 = known(srsrc + 1, vv[1]),
                                         k2 = known(srsrc + 2, vv[2]), k3 = known(srsrc + 3, vv[3]);
                    bool patched = k0 && k1 && k2 && k3;
                    auto it = descr.find(srsrc);
                    // Fold the fetch's CONSTANT byte offset into the emitted V# base (#273 item 1, the
                    // "solid banner" bug): the recompiler's per-fetch (by_fetch_pc) address model is
                    // exactly gl_VertexIndex*stride from the resolved base — it assumes the attribute's
                    // in-record byte offset is already IN the base. Unity-style fetch shaders satisfy
                    // that by patching each attribute's V# base; DOLL's UE4 Slate VS instead uses ONE
                    // un-patched V# and carries each attribute's offset in the MUBUF SOFFSET register
                    // (+ the 12-bit inst offset). Without the fold, all four Slate attributes (pos, uv,
                    // material-uv, color) read the position bytes -> the loading-banner widget rendered
                    // as a solid bar. The walk knows the SOFFSET value (it computed it from the attr-spec
                    // words), so add soffset+inst_offset to the descriptor base; an UNKNOWN soffset keeps
                    // the un-offset base (previous behavior). CONFIDENCE: HIGH (fetch-time values traced
                    // live; Messenger's fetches carry SOFFSET=0 so they are byte-identical).
                    uint32_t soff = 0; bool soff_known = true;
                    if (in.src[2].kind == OperandKind::Special && in.src[2].value == 125)     // SGPR_NULL -> 0
                        soff = 0;
                    else if (!srcval(in.src[2], soff))
                        soff_known = false;                          // real but untracked SOFFSET (#398) — see below
                    const uint32_t inst_off = in.literal & 0xFFFu;
                    const uint32_t fetch_off = soff + inst_off;
                    auto with_off = [&](DecodedBufferDescriptor d) {
                        d.base += fetch_off;
                        // size_bytes stays num_records*stride: the hardware bound is INDEX < num_records
                        // (record granularity), so from the offset base the last record's attribute still
                        // lies within (num_records-1)*stride + fetch_off + attr bytes — trimming the size
                        // by fetch_off cut the LAST vertex's attribute off the upload (guarded reads made
                        // it zeros -> a collapsed final vertex).
                        return d;
                    };
                    if (trc) fprintf(stderr, "[dyntrace] MUBUF fetch pc=%u op=0x%x SRSRC=s%d patched=%d (k=%d%d%d%d v3=0x%x) have_descr=%d off=+0x%x soff_known=%d\n",
                                     in.pc, in.opcode, srsrc, patched, k0, k1, k2, k3, k3 ? vv[3] : 0, it != descr.end(), fetch_off, (int)soff_known);
                    // A real (non-NULL) SOFFSET the fold cannot resolve would silently collapse fetch_off's
                    // in-record component to 0 — every attribute reads base+inst_off (the "solid banner"
                    // collapse this fold was written to fix) or a wrong descriptor address. Leave the fetch
                    // UNRESOLVED (a loud recompile-coverage miss) rather than fabricating offset 0 (#398).
                    if (!soff_known) {
                        if (trc) fprintf(stderr, "[dyntrace]   MUBUF pc=%u SOFFSET untracked -> fetch left unresolved (not folded to 0)\n", in.pc);
                        break;
                    }
                    // Per-fetch: record THIS fetch's live V# (the SRSRC SGPR is reloaded with a different V#
                    // per vertex attribute — position, uv, color…). Keyed by the fetch's pc so the recompiler
                    // resolves each buffer_load_format to the descriptor as loaded at that instruction.
                    if (patched)          out.push_back({ in.pc, srsrc, with_off(decode_buffer_descriptor(vv)), vv[3] });
                    else if (it != descr.end())
                        out.push_back({ in.pc, srsrc, with_off(decode_buffer_descriptor(it->second.data())), it->second[3] });
                    else if (srsrc >= (int)user_sgpr_base && srsrc + 4 <= (int)(user_sgpr_base + nsgpr) &&
                             !reloaded.count(srsrc) && !reloaded.count(srsrc + 1) &&
                             !reloaded.count(srsrc + 2) && !reloaded.count(srsrc + 3)) {
                        // SEED fallback (#294): the SRSRC V# was placed directly in the user-data SGPRs
                        // by the driver (never s_loaded — so no `descr` snapshot) and the shader's
                        // stride/format patch left the CURRENT dwords partially unknown (its s_cselect
                        // condition reads an NGG system SGPR we don't model). Use the SEED values — the
                        // same load-time/pre-patch semantics as the `descr` fallback above. Refused if
                        // any of the 4 SGPRs was RELOADED from memory since seeding (a stale seed then
                        // no longer describes the register). DOLL's scene-geometry VS fetches resolve
                        // through exactly this path. CONFIDENCE: MED (patch-ignoring, like `descr`).
                        const uint32_t sv[4] = { user_sgprs[srsrc - (int)user_sgpr_base],
                                                 user_sgprs[srsrc - (int)user_sgpr_base + 1],
                                                 user_sgprs[srsrc - (int)user_sgpr_base + 2],
                                                 user_sgprs[srsrc - (int)user_sgpr_base + 3] };
                        DecodedBufferDescriptor d = decode_buffer_descriptor(sv);
                        // Plausibility: only emit a real-looking V# (mirrors the direct-resource guard).
                        if (d.base > 0x10000 && d.size_bytes != 0 && d.size_bytes <= 0x10000000u) {
                            if (trc) fprintf(stderr, "[dyntrace]   MUBUF pc=%u seed-V# fallback SRSRC=s%d base=0x%llx\n",
                                             in.pc, srsrc, (unsigned long long)d.base);
                            out.push_back({ in.pc, srsrc, with_off(d), sv[3], /*from_seed=*/true });
                        }
                    }
                }
                break;
            }
            case Rdna2Format::SOPK:
                // s_cmpk_* / s_addk_i32 write SCC (only s_movk/s_version/s_cmovk/s_mulk don't); this
                // interpreter doesn't model SOPK, so ANY SOPK conservatively invalidates the tracked SCC —
                // a stale SCC consumed by a later s_cselect would fabricate a confidently-wrong V# patch.
                scc = -1;
                if (in.dst.kind == OperandKind::SGPR) val.erase(in.dst.value);
                break;
            default:
                // Remaining formats (SOPP, VALU, memory, …) don't write SCC, so the tracked SCC survives.
                if (in.dst.kind == OperandKind::SGPR) val.erase(in.dst.value);   // unmodeled scalar write -> unknown
                break;
        }
    }
    return out;
}

namespace { constexpr uint32_t kPsBindingBase = 32; }

// Assign each resource its OWN descriptor binding, starting at `first` (0/1 reserved). The N-buffer
// model: the shader reads several distinct constant buffers (Unity's per-draw transform, per-frame,
// …) + vertex buffers + textures, and each must land at a separate binding so they don't collapse.
// The recompiler declares a storage buffer (cbuf/vbuf) or image sampler (texture) at each binding and
// resolves an s_buffer_load/image_sample to its resource's binding via provenance.
//
// The VS and PS tables are bound together in ONE descriptor set by the live renderer, so the two
// stages' binding ranges MUST be disjoint (VS keeps 2.., PS starts at kPsBindingBase). Within a
// stage, constant/vertex BUFFERS are assigned first (from `first`), then TEXTURES / storage images —
// but never on binding 2 or 3, which the recompiler's declare_cbufs always occupies with its two
// hardwired storage-buffer cbufs (v_cbuf/v_cbuf1). A shader whose FIRST resource is a texture used to
// land it on binding 2, declaring BOTH a combined image sampler AND that storage buffer at one
// binding (two descriptor types -> layout-creation failure, the draw disappears) (#157). Buffers-
// first keeps the common cbufs-first shaders' bindings byte-identical (cbufs 2/3, textures 4+).
// External linkage (declared in gpu_execute.hpp) so the binding policy is unit-testable.
void assign_convention_bindings(ShaderResourceTable& t, uint32_t first) {
    uint32_t next = first;
    for (auto& r : t.resources)
        if (r.cls == ResourceClass::ConstantBuffer || r.cls == ResourceClass::VertexBuffer)
            r.binding = next++;
    uint32_t tex_next = next > first + 2 ? next : first + 2;   // reserve the two hardwired cbuf slots
    for (auto& r : t.resources)
        if (r.cls != ResourceClass::ConstantBuffer && r.cls != ResourceClass::VertexBuffer)
            r.binding = tex_next++;
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

    // Per-shader user-data RANGE: the shader blob's "specials" block declares which DWORD range of
    // the stage's USER_DATA register block holds this shader's SGPR-visible user data
    // (user_data_range_start/end — the range SetSource programs; e.g. DOLL's UE4 Slate VS declares
    // [0,8) matching its 8-dword {V#, ptr, ptr} block). Header sharp/direct offsets are relative to
    // range_start, so seed the SGPR block from USER_DATA_<stage>_<range_start>. Every shader
    // observed live (DOLL + Messenger) declares start=0, so this is currently behavior-identical —
    // LATENT support for a start!=0 shader, guarded back to 0 on insane metadata.
    // CONFIDENCE: LOW on start!=0 semantics (no live example yet); zero risk for start==0.
    uint32_t range_start = 0;
    if (hdr->specials && guest_readable((uint64_t)(uintptr_t)hdr->specials, sizeof(AgcShaderSpecials))) {
        const uint32_t s = hdr->specials->user_data_range_start;
        const uint32_t e = hdr->specials->user_data_range_end;
        if (s < kUserSgprs && e > s && e <= 2 * kUserSgprs) range_start = s;
        if (log && range_start) {   // once per shader: non-zero ranges are the rare/interesting case
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second)
                fprintf(stderr, "[agc] %s 0x%llx user_data_range=[%u,%u) -> seeding from USER_DATA_%u\n",
                        is_ps ? "PS" : "VS", (unsigned long long)code_addr, s, e, range_start);
        }
    }

    // PROSPER_RESDUMP: raw dump of the user-data struct + SGPR block per base, so the EUD layout
    // (which sharps have offset_dw>=16, and where the EUD pointer sits) can be read empirically.
    bool resdump = getenv("PROSPER_RESDUMP") != nullptr;
    if (resdump)   // PROSPER_RESDUMP_ADDR=<hex code addr>: narrow the dump to one shader
        if (const char* fa = getenv("PROSPER_RESDUMP_ADDR"))
            resdump = strtoull(fa, nullptr, 16) == code_addr;
    if (g_dyntrace_force) resdump = true;   // failure replay: always dump the failing stage's blocks
    if (resdump) {
        const AgcShaderUserData* ud = hdr->user_data;
        fprintf(stderr, "[resdump] %s code=0x%llx type=%u ud=%p range_start=%u (end=%u)\n",
                is_ps ? "PS" : "VS", (unsigned long long)code_addr, hdr->type, (const void*)ud,
                range_start, hdr->specials ? (uint32_t)hdr->specials->user_data_range_end : 0u);
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
    // The SAME const-fold also recovers descriptor-TABLE uses for BOTH stages (#294): UE4 shaders
    // s_load their T#/S#/V# descriptors from a table pointer in the user-data SGPRs and consume them
    // via image_sample / s_buffer_load — srt_uses reports each with its load-immediate key, which
    // becomes the resource's srt_offset (the recompiler's by_srt_offset provenance).
    std::vector<DynFetch> dyn_vb;
    std::vector<SrtUse> srt_uses;
    if (is_ps) {
        uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, bases[0] + range_start, sgprs);
        resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, 0x4000, sgprs, kUserSgprs, 0, &srt_uses);
    } else {
        uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, bases[0] + range_start, sgprs);
        // NGG merged VS/GS: s0..s7 are system SGPRs, user data starts at s8 (confirmed by matching the
        // shader's s[8:11]/s[24:25] descriptor pointers to the register file at GS_0+offset).
        dyn_vb = resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, 0x4000, sgprs, kUserSgprs, 8, &srt_uses);
        if (getenv("PROSPER_GFXLOG") || getenv("PROSPER_RESDUMP")) {
            fprintf(stderr, "[dynvb] VS resolved %zu dynamic vertex-fetch descriptor(s):\n", dyn_vb.size());
            for (auto& kv : dyn_vb) {
                const auto& d = kv.desc;
                fprintf(stderr, "[dynvb]   pc=%u SRSRC s%d -> base=0x%llx stride=%u num_records=%u size=%u fmt=%u nc=%u\n",
                        kv.fetch_pc, kv.srsrc, (unsigned long long)d.base, d.stride, d.num_records, d.size_bytes,
                        (unsigned)d.format, d.num_components);
                if (guest_readable(d.base, 64)) {   // raw dwords of vertex 0 (to read packed formats) + floats
                    const uint32_t* u = (const uint32_t*)(uintptr_t)d.base;
                    fprintf(stderr, "[dynvb]     v3fmt=0x%x  raw v0: %08x %08x %08x %08x\n",
                            kv.desc_v3, u[0], u[1], u[2], u[3]);
                    // Print the first 3 vertex records (the DrawIndexAuto count=3 triangle) as floats, so the
                    // actual on/off-screen span can be computed offline. Each record is `stride` bytes.
                    for (int rec = 0; rec < 3 && d.stride; rec++) {
                        uint64_t a = d.base + (uint64_t)rec * d.stride;
                        if (!guest_readable(a, 16)) break;
                        const float* f = (const float*)(uintptr_t)a;
                        fprintf(stderr, "[dynvb]     rec%d: %.3f %.3f %.3f %.3f\n", rec, f[0], f[1], f[2], f[3]);
                    }
                }
            }
        }
    }

    // NGG VS/GS loads user data at shader s8 (s0..s7 = system SGPRs); PS at s0. The resource sgpr_base
    // (an s_buffer_load/image_sample's SBASE/SRSRC register) is in that shader-SGPR space.
    const uint32_t user_sgpr_base = is_ps ? 0u : 8u;
    for (uint32_t base : bases) {
        uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, base + range_start, sgprs);
        ShaderResourceTable t = build_shader_resources(*hdr, sgprs, kUserSgprs, user_sgpr_base);
        // Add the const-fold-resolved dynamic vertex buffers, keyed by their SRSRC SGPR so the
        // recompiler's by_sgpr_base() resolves each buffer_load_format. The V#'s data format is patched
        // at runtime by the fetch shader (so the load-time snapshot reads Unknown) — default to Float32
        // (a raw 32-bit-per-component fetch, correct for float attributes like positions).
        for (auto& kv : dyn_vb) {
            const auto& d = kv.desc;
            // A SEED-fallback entry must not shadow a metadata-described DIRECT vertex buffer at the
            // same SGPRs (see DynFetch::from_seed): the direct resource resolves the fetch through
            // the faithful address path, which is the correct model for a single un-patched V#.
            if (kv.from_seed) {
                bool direct_exists = false;
                for (const auto& r0 : t.resources)
                    if (r0.cls == ResourceClass::VertexBuffer && r0.sgpr_base == (uint32_t)kv.srsrc)
                        { direct_exists = true; break; }
                if (direct_exists) continue;
            }
            ShaderResource r;
            r.cls           = ResourceClass::VertexBuffer;
            r.format        = (d.format == DataFormat::Unknown) ? DataFormat::Float32 : d.format;
            r.num_components = d.num_components ? d.num_components : 4;
            r.gpu_addr      = d.base;
            r.size          = d.size_bytes ? d.size_bytes : (d.stride ? d.stride * 4 : 128);
            r.stride        = d.stride;
            r.sgpr_base     = kv.srsrc;           // DIRECT provenance = the fetch's SRSRC SGPR (fallback)
            r.fetch_pc      = kv.fetch_pc;        // PER-FETCH provenance = the exact fetch instruction
            r.srt_offset    = 0xFFFFFFFFu;
            t.resources.push_back(r);
        }
        // Descriptor-TABLE resources (#294): one ShaderResource per distinct table use, keyed by the
        // s_load immediate (srt_offset) so the recompiler's sreg_srt/by_srt_offset provenance resolves
        // the consuming image_sample / s_buffer_load. Never shadow an existing resource at the same
        // srt_offset (the EUD-sharp path may already have emitted it — first match wins in
        // by_srt_offset, and two DIFFERENT tables reusing one immediate would be ambiguous anyway).
        {
            std::set<uint64_t> srt_seen;
            for (const auto& u : srt_uses) {
                // Dedupe: a KEYED cbuf use per key (the s_buffer_load resolves by key); texture and
                // key-less buffer uses per CONSUMING INSTRUCTION (#273 — several image ops may share
                // one key, or have none; a key-less V# fetch resolves by its pc).
                // Distinct namespaces: pc keys must never collide with byte-offset keys.
                uint64_t dk = (u.kind == 0 || u.key == 0xFFFFFFFFu)
                                  ? (0x8000000000000000ull | ((uint64_t)(uint32_t)u.kind << 32) | u.use_pc)
                                  : ((uint64_t)(uint32_t)u.kind << 32) | u.key;
                if (!srt_seen.insert(dk).second) continue;
                bool clash = u.key == 0xFFFFFFFFu;       // key-less: never resolvable by srt_offset
                if (!clash)
                    for (const auto& r0 : t.resources) if (r0.srt_offset == u.key) { clash = true; break; }
                if (u.kind == 1) {                       // constant buffer / structured-buffer V#
                    DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
                    if (d.base <= 0x10000 || d.size_bytes == 0 || d.size_bytes > 0x10000000u) continue;
                    // A keyed use whose key already resolves keeps the existing resource; a key-less
                    // (or key-clashed) use still needs a pc-provenance entry — piggyback the pc onto
                    // an existing resource describing the SAME buffer, else create one (#273).
                    if (clash) {
                        bool piggybacked = false;
                        for (auto& r0 : t.resources)
                            if ((r0.cls == ResourceClass::ConstantBuffer || r0.cls == ResourceClass::VertexBuffer) &&
                                r0.gpu_addr == d.base && r0.size == d.size_bytes) {
                                if (r0.fetch_pc == 0xFFFFFFFFu && r0.cls == ResourceClass::ConstantBuffer)
                                    r0.fetch_pc = u.use_pc;
                                piggybacked = r0.fetch_pc == u.use_pc || r0.cls == ResourceClass::VertexBuffer;
                                if (piggybacked) break;
                            }
                        if (piggybacked) continue;
                    }
                    ShaderResource r;
                    r.cls = ResourceClass::ConstantBuffer;
                    r.format = d.format; r.num_components = d.num_components ? d.num_components : 1;
                    r.gpu_addr = d.base; r.size = d.size_bytes; r.stride = d.stride;
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
                    if (clash) r.fetch_pc = u.use_pc;    // pc-only provenance (key-less/collided V#)
                    if (log) fprintf(stderr, "[srt] %s cbuf key=0x%x pc=%u base=0x%llx size=%u\n", is_ps ? "PS" : "VS",
                                     u.key, u.use_pc, (unsigned long long)d.base, d.size_bytes);
                    t.resources.push_back(r);
                } else {                                  // texture (T# [+ paired S#])
                    DecodedImageDescriptor d = decode_image_descriptor(u.t8.data());
                    if (g_dyntrace_force)
                        fprintf(stderr, "[dynfail] tex use pc=%u key=0x%x base=0x%llx %ux%u fmt=%u tile=%u\n",
                                u.use_pc, u.key, (unsigned long long)d.base, d.width, d.height,
                                d.format, d.tile_mode);
                    if (d.base == 0 || d.width == 0 || d.height == 0 ||
                        d.width > 16384 || d.height > 16384) continue;       // garbage/degenerate T#
                    // A previous use already produced a resource for this SAME surface (same T# base +
                    // extent): don't duplicate the binding/upload — give it this use's pc provenance
                    // if it has none yet (#273). If it already carries a DIFFERENT use's pc, fall
                    // through and create a second resource for this pc (fetch_pc holds one pc; a
                    // sample whose pc has no mapping would stay unresolved).
                    {
                        bool mapped = false;
                        for (auto& r0 : t.resources)
                            if (r0.cls == ResourceClass::Texture && r0.gpu_addr == d.base &&
                                r0.width == d.width && r0.height == d.height) {
                                if (r0.fetch_pc == 0xFFFFFFFFu) { r0.fetch_pc = u.use_pc; mapped = true; break; }
                                if (r0.fetch_pc == u.use_pc)    { mapped = true; break; }
                            }
                        if (mapped) continue;
                    }
                    Gen5ImageFormatInfo fi;
                    if (!gen5_image_format(d.format, &fi)) {
                        // Same policy as build_shader_resources: the normal per-target renderer can
                        // bind this as RGBA8 for RTT injection; legacy single-target mode skips it.
                        static const bool rtt_bind = getenv("PROSPER_RTT") != nullptr ||
                                                     getenv("PROSPER_RTT_PERTARGET") != nullptr;
                        if (!rtt_bind) continue;
                        fi.format = DataFormat::Unorm8; fi.num_components = 4; fi.bytes_per_block = 4;
                        fi.block_width = fi.block_height = 1; fi.srgb = false; fi.snorm = false;
                    }
                    const bool is_bcn = fi.block_width > 1;
                    if (is_bcn && fi.snorm) continue;   // signed BCn (SNORM / BC6H SF16): decode not wired
                    ShaderResource r;
                    r.cls = ResourceClass::Texture;
                    r.format = fi.format; r.num_components = fi.num_components;
                    r.gpu_addr = d.base; r.width = d.width; r.height = d.height;
                    r.tile_mode = d.tile_mode; r.srgb = fi.srgb;
                    // T# TYPE -> MIMG dim (GFX10: 9=2D, 10=3D, 11=CUBE, 13=2D_ARRAY); a cube
                    // uploads as six vertically-stacked faces (#273 — see agc_shader_layout).
                    r.img_dim = d.type == 11 ? 3u : d.type == 10 ? 2u : d.type == 13 ? 5u : 1u;
                    r.swizzle[0] = d.dst_sel[0]; r.swizzle[1] = d.dst_sel[1];
                    r.swizzle[2] = d.dst_sel[2]; r.swizzle[3] = d.dst_sel[3];
                    r.size = is_bcn ? (((d.width + 3) / 4) * ((d.height + 3) / 4) * fi.bytes_per_block)
                                    : (d.width * d.height * fi.bytes_per_block);
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;   // ambiguous/absent key: pc-only provenance
                    r.fetch_pc   = u.use_pc;                       // per-instruction provenance (#273)
                    if (u.has_samp) {                     // paired S# (same SQ_IMG_SAMP decode as the sharp path)
                        const uint32_t* sm = u.s4.data();
                        r.mag_filter  = ((sm[2] >> 20) & 0x3u) ? 1u : 0u;
                        r.min_filter  = ((sm[2] >> 22) & 0x3u) ? 1u : 0u;
                        r.mip_filter  = ((sm[2] >> 26) & 0x3u) ? 1u : 0u;
                        r.addr_uvw[0] = (sm[0] >> 0) & 0x7u;
                        r.addr_uvw[1] = (sm[0] >> 3) & 0x7u;
                        r.addr_uvw[2] = (sm[0] >> 6) & 0x7u;
                        r.max_aniso_ratio    = (sm[0] >> 9)  & 0x7u;
                        r.depth_compare_func = (sm[0] >> 12) & 0x7u;
                        r.unnormalized       = (sm[0] >> 15) & 0x1u;
                        r.min_lod            = (float)( sm[1]        & 0xFFFu) / 256.0f;
                        r.max_lod            = (float)((sm[1] >> 12) & 0xFFFu) / 256.0f;
                        int32_t bias14       = (int32_t)(sm[2] & 0x3FFFu);
                        if (bias14 & 0x2000) bias14 -= 0x4000;
                        r.lod_bias           = (float)bias14 / 256.0f;
                        r.border_color_type  = (sm[3] >> 30) & 0x3u;
                    }
                    if (log) fprintf(stderr, "[srt] %s tex key=0x%x %ux%u fmt=%u base=0x%llx tile=%u samp=%d\n",
                                     is_ps ? "PS" : "VS", u.key, d.width, d.height, d.format,
                                     (unsigned long long)d.base, d.tile_mode, (int)u.has_samp);
                    t.resources.push_back(r);
                }
            }
        }
        if (t.resources.empty()) continue;
        assign_convention_bindings(t, is_ps ? kPsBindingBase : 2u);
        if (log) {
            fprintf(stderr, "[restab] %s code=0x%llx base=0x%x -> %zu resources:\n",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, base, t.resources.size());
            for (auto& r : t.resources) {
                fprintf(stderr, "[restab]   cls=%u binding=%u addr=0x%llx size=%u %ux%u fmt=%u stride=%u\n",
                        (unsigned)r.cls, r.binding, (unsigned long long)r.gpu_addr, r.size,
                        r.width, r.height, (unsigned)r.format, r.stride);
                if (r.cls == ResourceClass::ConstantBuffer && guest_readable(r.gpu_addr, 32)) {
                    const float* f = (const float*)(uintptr_t)r.gpu_addr;
                    fprintf(stderr, "[restab]     cbuf@0 floats: %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f\n",
                            f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
                    if (r.size >= 64 && guest_readable(r.gpu_addr, 64))
                        fprintf(stderr, "[restab]     cbuf@0 8..15:   %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f\n",
                                f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
                    if (r.size >= 0x150 && guest_readable(r.gpu_addr + 0x110, 64)) {
                        const float* g = (const float*)(uintptr_t)(r.gpu_addr + 0x110);
                        // Full 4x4 projection matrix at 0x110 (16 floats). Row 3's w element (g[15]) must be
                        // ~1.0 for clip.w to be nonzero — a zero here collapses the perspective divide.
                        fprintf(stderr, "[restab]     mtx@0x110 r0: %.4f %.4f %.4f %.4f\n", g[0], g[1], g[2], g[3]);
                        fprintf(stderr, "[restab]     mtx@0x110 r1: %.4f %.4f %.4f %.4f\n", g[4], g[5], g[6], g[7]);
                        fprintf(stderr, "[restab]     mtx@0x110 r2: %.4f %.4f %.4f %.4f\n", g[8], g[9], g[10], g[11]);
                        fprintf(stderr, "[restab]     mtx@0x110 r3: %.4f %.4f %.4f %.4f  <- g[15]=clip.w src\n", g[12], g[13], g[14], g[15]);
                    }
                }
            }
        }
        return std::make_shared<ShaderResourceTable>(std::move(t));
    }
    if (log) fprintf(stderr, "[restab] %s code=0x%llx -> no resources in any user-data base\n",
                     is_ps ? "PS" : "VS", (unsigned long long)code_addr);
    return nullptr;
}

bool validate_runtime_descriptor_contract(const char* stage_name,
                                           const std::vector<uint32_t>& spirv,
                                           const ShaderResourceTable* runtime,
                                           uint32_t expected_set,
                                           SpirvShaderStage expected_stage) {
    const char* mode = getenv("PROSPER_DESCRIPTOR_VALIDATE");
    if (!mode || !*mode || !strcmp(mode, "off") || !strcmp(mode, "0")) return true;

    DescriptorValidationReport report = validate_spirv_descriptor_interface(
        spirv, runtime, expected_set, expected_stage, true);
    const bool verbose = !strcmp(mode, "all");
    if (!report.issues.empty() || verbose) {
        uint64_t hash = 1469598103934665603ull;
        for (uint32_t word : spirv) { hash ^= word; hash *= 1099511628211ull; }
        static std::set<uint64_t> logged;
        uint64_t key = hash ^ (static_cast<uint64_t>(expected_set) << 56);
        auto mix = [&](uint64_t value) { key ^= value; key *= 1099511628211ull; };
        for (const auto& issue : report.issues) {
            mix(static_cast<uint32_t>(issue.code)); mix(issue.binding); mix(issue.set);
            mix(static_cast<uint32_t>(issue.actual));
            // Contract errors with different proven ranges are distinct. Warning-only unused
            // resources are not: their guest address/size can change every draw and must not flood
            // a long diagnostic run with the same module/binding warning.
            if (issue.error) { mix(issue.required_bytes); mix(issue.available_bytes); }
        }
        if (logged.insert(key).second) {
            fprintf(stderr, "[descriptor] %s module=%016llx used=%zu runtime=%zu result=%s mode=%s\n",
                    stage_name, (unsigned long long)hash, report.descriptors.size(),
                    runtime ? runtime->resources.size() : 0, report.ok() ? "accept" : "reject", mode);
            for (const auto& d : report.descriptors)
                fprintf(stderr, "[descriptor]   set=%u binding=%u type=%s required=%llu%s\n",
                        d.set, d.binding, spirv_descriptor_kind_name(d.kind),
                        (unsigned long long)d.required_bytes, d.dynamic_access ? "+dynamic" : "");
            for (const auto& issue : report.issues)
                fprintf(stderr, "[descriptor]   %s: %s set=%u binding=%u expected=%s actual=%s "
                                "required=%llu available=%llu\n",
                        issue.error ? "ERROR" : "warn", descriptor_issue_name(issue.code),
                        issue.set, issue.binding, spirv_descriptor_kind_name(issue.expected),
                        spirv_descriptor_kind_name(issue.actual),
                        (unsigned long long)issue.required_bytes,
                        (unsigned long long)issue.available_bytes);
            if (runtime) for (const auto& r : runtime->resources)
                fprintf(stderr, "[descriptor]   runtime binding=%u cls=%u addr=0x%llx size=%u "
                                "stride=%u fmt=%u comps=%u srt=0x%x sgpr=%u pc=%u\n",
                        r.binding, (unsigned)r.cls, (unsigned long long)r.gpu_addr, r.size,
                        r.stride, (unsigned)r.format, r.num_components,
                        r.srt_offset, r.sgpr_base, r.fetch_pc);
        }
    }
    return strcmp(mode, "strict") != 0 || report.ok();
}

ComputeLaunchDimensions resolve_compute_launch(const GpuState::Dispatch& d) {
    namespace P = prosper::agc::Pm4;
    ComputeLaunchDimensions out;
    out.threads_x = d.threads_x;
    out.threads_y = d.threads_y;
    out.threads_z = d.threads_z;
    const GpuState* ds = d.state.get();
    auto reg = [&](uint32_t off) {
        if (!ds) {
            return 0u;
        }
        auto it = ds->sh.find(off);
        return it == ds->sh.end() ? 0u : it->second;
    };
    out.local_x = reg(P::COMPUTE_NUM_THREAD_X);
    out.local_y = reg(P::COMPUTE_NUM_THREAD_Y);
    out.local_z = reg(P::COMPUTE_NUM_THREAD_Z);
    if (!out.local_x) out.local_x = 1;
    if (!out.local_y) out.local_y = 1;
    if (!out.local_z) out.local_z = 1;
    auto groups = [](uint32_t threads, uint32_t local) {
        return threads ? 1u + (threads - 1u) / local : 0u;
    };
    out.groups_x = groups(out.threads_x, out.local_x);
    out.groups_y = groups(out.threads_y, out.local_y);
    out.groups_z = groups(out.threads_z, out.local_z);
    return out;
}

void diagnose_compute_dispatches(const GpuState& st, uint64_t submit_no) {
    const char* enabled = getenv("PROSPER_COMPUTELOG");
    const char* dim_env = getenv("PROSPER_COMPUTELOG_DIM");
    if ((!enabled || !*enabled) && (!dim_env || !*dim_env)) return;

    uint32_t want_w = 0, want_h = 0;
    if (dim_env && *dim_env && sscanf(dim_env, "%ux%u", &want_w, &want_h) != 2) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[compute] invalid PROSPER_COMPUTELOG_DIM='%s' (expected WxH)\n", dim_env);
        }
        want_w = want_h = 0;
    }

    namespace P = prosper::agc::Pm4;
    auto rd = [](const std::unordered_map<uint32_t, uint32_t>& regs, uint32_t off) {
        auto it = regs.find(off); return it == regs.end() ? 0u : it->second;
    };
    auto pgm_addr = [&](const GpuState& ds) {
        uint32_t lo = rd(ds.sh, P::COMPUTE_PGM_LO), hi = rd(ds.sh, P::COMPUTE_PGM_HI);
        return (static_cast<uint64_t>(lo) << 8) | (static_cast<uint64_t>(hi & 0xffu) << 40);
    };

    size_t matched = 0;
    for (size_t i = 0; i < st.dispatches.size(); ++i) {
        const auto& d = st.dispatches[i];
        const GpuState& ds = d.state ? *d.state : st;
        const ComputeLaunchDimensions launch = resolve_compute_launch(d);
        const uint64_t code_addr = pgm_addr(ds);
        const auto* hdr = static_cast<const AgcShaderHeader*>(prosper_agc_shader_header_for_code(code_addr));

        uint32_t range_start = 0;
        if (hdr && hdr->specials && guest_readable((uint64_t)(uintptr_t)hdr->specials,
                                                   sizeof(AgcShaderSpecials))) {
            uint32_t s = hdr->specials->user_data_range_start;
            uint32_t e = hdr->specials->user_data_range_end;
            if (s < kUserSgprs && e > s && e <= 2 * kUserSgprs) range_start = s;
        }

        ShaderResourceTable table;
        uint32_t sgprs[kUserSgprs] = {};
        if (hdr) {
            read_user_sgprs(ds.sh, P::COMPUTE_USER_DATA_0 + range_start, sgprs);
            table = build_shader_resources(*hdr, sgprs, kUserSgprs, 0);
            assign_convention_bindings(table, 2);
        }

        // A compute shader can carry only an inline direct type-1 V# and no sharp descriptors. Dump
        // its metadata and bound SGPRs once per program if resource decoding still returns empty.
        // This turns the next unsupported layout into a reproducible decode problem instead of
        // another blind `resources=0` investigation.
        if (hdr && table.resources.empty() && enabled && *enabled) {
            static std::set<uint64_t> logged_empty;
            if (logged_empty.insert(code_addr).second) {
                const AgcShaderUserData* ud = hdr->user_data;
                fprintf(stderr, "[compute] empty-resource metadata code=0x%llx type=%u ud=%p",
                        (unsigned long long)code_addr, hdr->type, (const void*)ud);
                if (ud) {
                    fprintf(stderr, " eud=%u srt=%u direct_count=%u sharp={%u,%u,%u,%u}",
                            ud->eud_size_dw, ud->srt_size_dw, ud->direct_resource_count,
                            ud->sharp_resource_count[0], ud->sharp_resource_count[1],
                            ud->sharp_resource_count[2], ud->sharp_resource_count[3]);
                }
                fprintf(stderr, "\n[compute]   user_sgprs:");
                for (uint32_t s = 0; s < kUserSgprs; ++s) fprintf(stderr, " %08x", sgprs[s]);
                fprintf(stderr, "\n");

                if (ud && ud->direct_resource_offset && ud->direct_resource_count) {
                    fprintf(stderr, "[compute]   direct offsets:");
                    for (uint16_t t = 0; t < ud->direct_resource_count && t < 16; ++t)
                        fprintf(stderr, " [%u]=%u", t, ud->direct_resource_offset[t]);
                    fprintf(stderr, "\n");
                    const uint32_t reg = ud->direct_resource_count > 1 ? ud->direct_resource_offset[1] : 0xffffu;
                    if (reg != 0xffffu && reg + 4 <= kUserSgprs) {
                        const DecodedBufferDescriptor d = decode_buffer_descriptor(&sgprs[reg]);
                        fprintf(stderr, "[compute]   type1 V# reg=%u base=0x%llx stride=%u records=%u "
                                        "size=%u fmt=%u comps=%u\n",
                                reg, (unsigned long long)d.base, d.stride, d.num_records, d.size_bytes,
                                (unsigned)d.format, d.num_components);
                    }
                }
            }
        }

        bool dim_match = !want_w || !want_h;
        if (!dim_match) {
            for (const auto& r : table.resources)
                if (r.width == want_w && r.height == want_h) { dim_match = true; break; }
        }
        if (!dim_match) continue;
        matched++;

        uint64_t code_hash = 1469598103934665603ull;
        if (code_addr && guest_readable(code_addr, 4096)) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(code_addr));
            for (size_t n = 0; n < 4096; ++n) { code_hash ^= p[n]; code_hash *= 1099511628211ull; }
        } else {
            code_hash = 0;
        }
        fprintf(stderr,
                "[compute] submit=%llu dispatch=%zu threads=%ux%ux%u local=%ux%ux%u "
                "groups=%ux%ux%u modifier=0x%llx "
                "code=0x%llx hash4k=%016llx header=%s resources=%zu\n",
                (unsigned long long)submit_no, i,
                launch.threads_x, launch.threads_y, launch.threads_z,
                launch.local_x, launch.local_y, launch.local_z,
                launch.groups_x, launch.groups_y, launch.groups_z,
                (unsigned long long)d.modifier, (unsigned long long)code_addr,
                (unsigned long long)code_hash, hdr ? "yes" : "no", table.resources.size());
        for (const auto& r : table.resources) {
            fprintf(stderr,
                    "[compute]   cls=%u binding=%u addr=0x%llx size=%u dims=%ux%u "
                    "fmt=%u comps=%u tile=%u sgpr=%u srt=0x%x\n",
                    (unsigned)r.cls, r.binding, (unsigned long long)r.gpu_addr, r.size,
                    r.width, r.height, (unsigned)r.format, r.num_components, r.tile_mode,
                    r.sgpr_base, r.srt_offset);
        }
    }

    if (want_w && want_h && !st.dispatches.empty() && matched == 0 && enabled && enabled[0] == 'a') {
        fprintf(stderr, "[compute] submit=%llu dispatches=%zu: no resource matched %ux%u\n",
                (unsigned long long)submit_no, st.dispatches.size(), want_w, want_h);
    }
}

void diagnose_resource_provenance(const GpuState& st, uint64_t submit_no) {
    const char* dim_env = getenv("PROSPER_PROVENANCE_DIM");
    if (!dim_env || !*dim_env) return;

    uint32_t want_w = 0, want_h = 0;
    if (sscanf(dim_env, "%ux%u", &want_w, &want_h) != 2 || !want_w || !want_h) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[provenance] invalid PROSPER_PROVENANCE_DIM='%s' (expected WxH)\n", dim_env);
        }
        return;
    }
    const size_t min_draws = [] {
        const char* e = getenv("PROSPER_PROVENANCE_MIN_DRAWS");
        return e ? static_cast<size_t>(strtoull(e, nullptr, 0)) : size_t{0};
    }();

    struct ColorWrite {
        uint64_t submit = 0;
        uint64_t draw_submit = 0;
        size_t draw = 0;
        uint64_t vs = 0, ps = 0;
        uint32_t width = 0, height = 0;
        GpuState::Draw draw_record{};
    };
    static std::unordered_map<uint64_t, ColorWrite> last_color_write;
    static uint64_t draw_submit_ordinal = 0;
    const uint64_t this_draw_submit = st.draws.empty() ? draw_submit_ordinal : draw_submit_ordinal++;

    const bool inspect_consumers = st.draws.size() >= min_draws;
    for (size_t i = 0; i < st.draws.size(); ++i) {
        const GpuState& ds = st.draws[i].state ? *st.draws[i].state : st;
        const RenderState rs = extract_render_state(ds);

        // Query before recording this draw's target: a feedback draw should resolve to the preceding
        // writer, not identify itself as its own producer.
        if (inspect_consumers && rs.ps_addr) {
            auto prt = build_stage_table(ds, rs.ps_addr, true);
            if (prt) for (const auto& r : prt->resources) {
                if (r.width != want_w || r.height != want_h) continue;
                auto it = last_color_write.find(r.gpu_addr);
                if (it == last_color_write.end()) {
                    fprintf(stderr,
                            "[provenance] consumer submit=%llu draw=%zu ps=0x%llx samples "
                            "addr=0x%llx dims=%ux%u draw_submit=%llu: no prior color-target write\n",
                            (unsigned long long)submit_no, i, (unsigned long long)rs.ps_addr,
                            (unsigned long long)r.gpu_addr, r.width, r.height,
                            (unsigned long long)this_draw_submit);
                } else {
                    const ColorWrite& w = it->second;
                    fprintf(stderr,
                            "[provenance] consumer submit=%llu draw=%zu ps=0x%llx samples "
                            "addr=0x%llx dims=%ux%u draw_submit=%llu: last color write submit=%llu "
                            "draw_submit=%llu draw=%zu target_extent=%ux%u "
                            "vs=0x%llx ps=0x%llx\n",
                            (unsigned long long)submit_no, i, (unsigned long long)rs.ps_addr,
                            (unsigned long long)r.gpu_addr, r.width, r.height,
                            (unsigned long long)this_draw_submit, (unsigned long long)w.submit,
                            (unsigned long long)w.draw_submit, w.draw, w.width, w.height,
                            (unsigned long long)w.vs, (unsigned long long)w.ps);
                    static std::set<uint64_t> probed;
                    if (probed.insert(r.gpu_addr).second && w.draw_record.state) {
                        DrawItem producer;
                        bool realized = realize_draw_item(*w.draw_record.state, &w.draw_record,
                                                         w.draw_record.index_count, 0x10000, true, producer);
                        fprintf(stderr,
                                "[provenance] producer-realize addr=0x%llx result=%s extent=%ux%u "
                                "items-target=0x%llx\n",
                                (unsigned long long)r.gpu_addr, realized ? "success" : "dropped",
                                producer.color0_width, producer.color0_height,
                                (unsigned long long)producer.color0_base);
                    }
                }
            }
        }

        if (rs.color0_base)
            last_color_write[rs.color0_base] = {
                submit_no, this_draw_submit, i, rs.es_addr, rs.ps_addr,
                rs.color0_width, rs.color0_height, st.draws[i]
            };
    }
}

void set_submit_renderer(LiveRenderFn fn) { g_live = std::move(fn); }
bool have_submit_renderer()               { return static_cast<bool>(g_live); }
std::vector<uint8_t> render_submit_items(const std::vector<DrawItem>& items,
                                         uint32_t width, uint32_t height) {
    return g_live ? g_live(items, width, height) : std::vector<uint8_t>{};
}

bool execute_and_present(const GpuState& st, uint32_t width, uint32_t height) {
    if (!g_live || st.draws.empty() || !width || !height) return false;
    // Bind the target dimensions and defer to the pure core, which recompiles the shaders from their
    // SHADER_PGM addresses and resolves fixed-function state before calling back into the live renderer.
    // Scale the guest viewport to our framebuffer: `width`/`height` are the render target (reduced by
    // PROSPER_RENDER_SCALE), while the guest programs its viewport in full present-resolution pixels — so
    // without this a 1/N render shows only the bottom-left 1/N of the frame.
    uint32_t fw = present_width(), fh = present_height();
    float sx = fw ? (float)width  / (float)fw : 1.0f;
    float sy = fh ? (float)height / (float)fh : 1.0f;
    std::vector<uint8_t> px = execute_gpustate(st,
        [&](const std::vector<DrawItem>& items) {
            auto pending = begin_requested_gpu_capture(items, width, height);
            std::vector<uint8_t> rendered = g_live(items, width, height);
            if (pending) {
                std::string error;
                if (!finish_requested_gpu_capture(std::move(pending), rendered, error))
                    std::fprintf(stderr, "[gpucap] write failed: %s\n", error.c_str());
            }
            return rendered;
        },
        0x10000, sx, sy);
    if (px.size() != static_cast<size_t>(width) * height * 4) return false;   // recompile/render failed
    present_write_frame(px.data(), width, height);
    return true;
}

} // namespace prosper::gpu
