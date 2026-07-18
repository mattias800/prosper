// rdna2_to_spirv.hpp — first real RDNA2 -> SPIR-V translation (recompiler stage 3).
//
// Scope for this stage: a straight-line float VALU kernel. We model VGPRs as SSA float values,
// translate a decoded RDNA2 instruction stream (rdna2_decode.hpp) op-by-op into SPIR-V float ops,
// and wrap it in a compute shader so it is verifiable by execution (test_rdna2_to_spirv runs it and
// asserts the numbers against the RDNA2 semantics — verification layer 4). Control flow, the EXEC
// mask, memory instructions, and integer/scalar ALU come in later stages; unsupported opcodes make
// recompile_valu return an empty vector so the caller can tell it wasn't handled.
//
// I/O convention (so the kernel is testable): the compute shader loads `num_inputs` consecutive
// floats per invocation from storage buffer 0 into v0..v(num_inputs-1)  (v[k] = a[gid*num_inputs+k]),
// runs the translated ALU ops, then stores v[out_vgpr] to storage buffer 1 (b[gid]).
#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace prosper::gpu {

struct ShaderResourceTable;   // resource-binding contract (shader_resources.hpp); optional to recompile_valu
struct Rdna2Inst;

// A narrowly-proven compiler-generated scalar jump table. The shader loads a uniform selector from a
// direct constant buffer, bounds it, scales it by the 64-bit table-entry size, loads a PC-relative
// target, and reaches it with s_setpc_b64. Arbitrary indirect control flow remains unsupported: this
// metadata is returned only when the complete bounded idiom and every table target can be proven.
struct PcrelDispatchInfo {
    bool valid = false;
    uint32_t selector_sgpr_base = 0;
    uint32_t selector_byte_offset = 0;
    int32_t selector_addend = 0;
    uint32_t selector_max = 0;
    uint32_t setpc_pc = 0;
    uint32_t merge_pc = 0;
    size_t required_dwords = 0;
    std::vector<uint32_t> target_pcs;
    std::vector<uint32_t> setup_pcs;
};

PcrelDispatchInfo rdna2_pcrel_dispatch_info(const uint32_t* code, size_t dwords);

// Retain only the selected arm of a proven compiler-generated PC-relative dispatch. The fragment
// recompiler and the live descriptor fold must specialize the same instruction stream: otherwise
// descriptor loads in omitted alternatives can contaminate the selected arm's resource provenance.
bool rdna2_specialize_pcrel_dispatch(std::vector<Rdna2Inst>& instructions,
                                     const PcrelDispatchInfo& info,
                                     uint32_t selected_target);

// Return the portion of a raw shader blob that participates in recompilation. This normally ends at
// S_ENDPGM, but compiler-generated PC-relative lookup tables may live immediately after the program and
// must remain part of an owning/cache copy. The result never exceeds `dwords`.
size_t rdna2_recompile_code_span(const uint32_t* code, size_t dwords);

// Fixed-function PS interpolant wiring captured from SPI_PS_INPUT_CNTL_0..31. A valid entry's
// OFFSET selects the vertex PARAM export feeding that logical PS input; OFFSET=0x20 selects the
// four hardware default vectors encoded by DEFAULT_VAL instead. Keeping this separate from
// RenderState makes the recompiler independently unit-testable.
struct PixelInputMapping {
    std::array<uint32_t, 32> controls{};
    uint32_t valid_mask = 0;

    bool operator==(const PixelInputMapping&) const = default;
};

// Fixed-function per-pixel VGPR loads selected by SPI_PS_INPUT_ENA / SPI_PS_INPUT_ADDR. ENA says
// which values hardware computes and loads; ADDR also reserves VGPR slots for disabled values, so it
// participates in the destination-register mapping. The fragment shell currently materializes the
// four floating-point position terms through SPIR-V FragCoord; the remaining enabled terms still
// reserve their documented slots for faithful position placement.
struct PixelSystemInputMapping {
    uint32_t ena = 0;
    uint32_t addr = 0;

    bool operator==(const PixelSystemInputMapping&) const = default;
};

// Portable lowering contract for GFX10's explicit pixel-interpolation parameters. AMD hardware can
// expose P0/P10/P20 directly to v_interp_mov; Vulkan only has an equivalent fragment extension on a
// subset of devices. When `requires_geometry` is true, the renderer inserts the generated geometry
// stage returned by recompile_interpolation_geometry(). It passes ordinary smooth attributes through
// and publishes P0, P10=(P1-P0), P20=(P2-P0), plus any requested barycentric system inputs, as a
// packed interface understood by recompile_fragment(). Locations are capped at Vulkan's portable
// 128-component fragment-input minimum (32 vec4 locations); `valid == false` keeps overflow visible.
struct FragmentInterpolationLayout {
    static constexpr uint32_t kUnusedLocation = UINT32_MAX;
    std::array<std::array<uint32_t, 3>, 32> parameter_locations{}; // [attr][0=P10,1=P20,2=P0]
    std::array<uint32_t, 7> system_locations{};                    // PS system fields 0..6
    uint32_t attribute_mask = 0;                                  // attributes consumed by VINTRP
    uint32_t smooth_mask = 0;                                     // attributes consumed by P1/P2
    bool requires_geometry = false;
    bool valid = true;

    FragmentInterpolationLayout();
};

FragmentInterpolationLayout fragment_interpolation_layout(
    const uint32_t* code, size_t dwords,
    const PixelSystemInputMapping* system_inputs = nullptr);

// Generate the descriptor-free triangle geometry stage described above. Returns {} when no fallback
// is required or the packed interface is invalid. Triangle lists, strips, fans, and the RectList
// triangle-strip lowering all feed Vulkan's `Triangles` geometry input primitive.
std::vector<uint32_t> recompile_interpolation_geometry(
    const FragmentInterpolationLayout& layout);

// Translate a straight-line float-VALU RDNA2 stream to a compute-shader SPIR-V module.
// Returns {} if the stream contains an opcode/format this stage does not yet handle. An optional
// ShaderResourceTable routes SMEM constant-buffer loads to distinct bindings via descriptor provenance.
// `lds_bytes` is the shader's real per-workgroup LDS allocation (COMPUTE_PGM_RSRC2.LDS_SIZE): the
// emitted Workgroup array is sized to it, clamped to the RDNA2 64 KB max (#130). 0 = the 16 KB
// default (also the safe cap for the common maxComputeSharedMemorySize of 32 KB).
std::vector<uint32_t> recompile_valu(const uint32_t* code, size_t dwords,
                                     uint32_t num_inputs, uint32_t out_vgpr,
                                     const ShaderResourceTable* rt = nullptr, uint32_t lds_bytes = 0);

// Register and launch state for a real compute program. User SGPR values are supplied as one
// push-constant dword per register; enabled system SGPRs follow them in hardware order. TIDIG_COMP_CNT
// controls whether local IDs seed v0 only (0), v0-v1 (1), or v0-v2 (2+).
struct ComputeShaderConfig {
    std::vector<uint32_t> user_sgprs;
    uint32_t local_x = 64, local_y = 1, local_z = 1;
    // Vulkan can only launch complete workgroups, while AGC thread-dimension dispatches may end in a
    // partial workgroup. When enabled, the generated entry point suppresses invocations outside this
    // exact global extent before they can execute guest shader instructions.
    bool exact_thread_extent = false;
    uint32_t threads_x = 0, threads_y = 0, threads_z = 0;
    uint32_t wave_size = 64; // COMPUTE_DISPATCH_INITIATOR.CS_W32_EN selects 32; otherwise 64.
    uint32_t tidig_comp_cnt = 0;
    bool tgid_x_en = false, tgid_y_en = false, tgid_z_en = false;
    bool tg_size_en = false;
    uint32_t lds_bytes = 0;
};

// Translate a game compute program without the synthetic binding-0 input / binding-1 output used by
// recompile_valu. Memory effects come only from the program's actual resource operations.
std::vector<uint32_t> recompile_compute(const uint32_t* code, size_t dwords,
                                        const ShaderResourceTable* rt,
                                        const ComputeShaderConfig& config);

// Recompile a pixel/fragment shader to a fragment SPIR-V module: run the VALU, and on EXP to MRT0/1
// write vec4(src0..3) to the matching color output. NULL-only shaders retain discard/EXEC effects and
// intentionally expose no color output. Returns {} if unsupported / no implemented export.
// An optional ShaderResourceTable enables memory ops (SMEM/MUBUF) with resolved bindings.
std::vector<uint32_t> recompile_fragment(const uint32_t* code, size_t dwords,
                                         const ShaderResourceTable* rt = nullptr,
                                         const PixelSystemInputMapping* system_inputs = nullptr,
                                         uint32_t pcrel_dispatch_target = UINT32_MAX,
                                         const FragmentInterpolationLayout* interpolation = nullptr);

// Packed RGBA component-enable nibbles for the first realized color export to MRT0/MRT1. EXP.EN is
// an attachment write mask, not a request to source disabled VGPRs; callers intersect this with the
// fixed-function CB_TARGET_MASK/CB_SHADER_MASK before creating the Vulkan pipeline. This preserves
// destination channels disabled by a partial export exactly as RDNA2 does.
uint32_t fragment_color_export_mask(const uint32_t* code, size_t dwords);

// Recompile a vertex shader to a vertex SPIR-V module: v0 = gl_VertexIndex, run the VALU, and on EXP
// to a POS target write vec4(src0..3) to gl_Position. Returns {} if unsupported / no position export.
// An optional ShaderResourceTable enables vertex fetch (buffer_load_format_*) + constant loads.
std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords,
                                       const ShaderResourceTable* rt = nullptr,
                                       const PixelInputMapping* pixel_inputs = nullptr);

// How much of a shader the recompiler currently covers (per-instruction), without requiring the
// stream to be a complete vertex/fragment. `alu` = instructions emit_alu handles (VALU/scalar/
// control-flow); `exports` = EXP (handled by the stage recompilers); `unsupported` = not yet handled
// (memory ops / unknown), with the first such (format, opcode) recorded. A data-driven coverage metric.
struct RecompileCoverage {
    uint32_t total = 0, alu = 0, exports = 0, unsupported = 0;
    // Instructions the recompiler handles GIVEN the right context (a resource table for MIMG /
    // buffer_load_format, or a fragment stage for VINTRP) but that recompile_coverage — which runs
    // table-less on a compute shell — cannot exercise. Counting them apart from `unsupported` gives an
    // honest "recompilable in context" number; the table-less `alu`/`unsupported` split understates it.
    uint32_t table_dependent = 0;
    int      first_bad_fmt = -1;   // Rdna2Format of the first TRULY-unsupported instruction (-1 if none)
    uint32_t first_bad_op  = 0;
    uint32_t first_bad_pc  = 0xFFFFFFFFu; // dword offset of that instruction (-1 if none)
};
RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords);

} // namespace prosper::gpu
