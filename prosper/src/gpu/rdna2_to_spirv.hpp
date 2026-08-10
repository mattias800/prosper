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

inline constexpr uint32_t kComputeInternalGdsBinding = 127;

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

// A compiler-generated vertex-fetch prolog can be installed as the hardware entry point while the
// main vertex/NGG program remains a separate allocation. The prolog loads attributes into VGPRs and
// tail-transfers through s_setpc_b64. This descriptor is returned only for the bounded terminal
// `s_setpc_b64 s[6:7]` ABI shape with no export or program end in the prefix; arbitrary indirect
// control flow remains unsupported.
struct VertexPrologInfo {
    bool valid = false;
    uint32_t setpc_pc = 0;
    size_t prefix_dwords = 0; // executable prolog words before s_setpc_b64
};

VertexPrologInfo rdna2_vertex_prolog_info(const uint32_t* code, size_t dwords);

PcrelDispatchInfo rdna2_pcrel_dispatch_info(const uint32_t* code, size_t dwords);

// Retain only the selected arm of a proven compiler-generated PC-relative dispatch. The fragment
// recompiler and the live descriptor fold must specialize the same instruction stream: otherwise
// descriptor loads in omitted alternatives can contaminate the selected arm's resource provenance.
bool rdna2_specialize_pcrel_dispatch(std::vector<Rdna2Inst>& instructions,
                                     const PcrelDispatchInfo& info,
                                     uint32_t selected_target);

// Prove scalar conditional branches whose SCC input depends only on shader-embedded integer
// constants, then remove instructions unreachable from the shader entry. The proof deliberately
// excludes entry/user SGPRs, memory loads, wave masks, and values crossing another basic-block
// boundary, so specialization depends only on shader bytes already present in the compile key.
// Returns the number of conditional branches specialized.
size_t rdna2_specialize_shader_constant_branches(
    std::vector<Rdna2Inst>& instructions);

// A dispatch-scoped proven-null BVH can make an exact Wave32 no-hit exit unconditional and, for the
// fully proven empty-stack idiom, remove an unseeded traversal cycle. Prune that resource-dependent
// path before shader-byte-only branch folding; the compute resource builder and translator call this
// same helper so dead instruction-scoped descriptors cannot diverge. The resource table's null marker
// and fetch PC are part of the module cache key. Returns the number of proven exits specialized.
size_t rdna2_specialize_proven_null_bvh_paths(
    std::vector<Rdna2Inst>& instructions, const ShaderResourceTable* resources,
    uint32_t wave_size);

// Return the portion of a raw shader blob that participates in recompilation. This normally ends at
// S_ENDPGM, but compiler-generated PC-relative lookup tables may live immediately after the program and
// must remain part of an owning/cache copy. The result never exceeds `dwords`.
size_t rdna2_recompile_code_span(const uint32_t* code, size_t dwords);

// Fixed-function PS interpolant wiring captured from SPI_PS_INPUT_CNTL_0..31. A valid entry's
// OFFSET selects the vertex PARAM export feeding that logical PS input; OFFSET=0x20 selects the
// four hardware default vectors encoded by DEFAULT_VAL instead. Keeping this separate from
// RenderState makes the recompiler independently unit-testable. `passthrough_mask` disambiguates the
// custom PARAM0 encoding; other custom PARAM slots can also be recovered from their register control.
struct PixelInputMapping {
    std::array<uint32_t, 32> controls{};
    uint32_t valid_mask = 0;
    uint32_t passthrough_mask = 0;

    uint32_t effective_passthrough_mask() const {
        uint32_t mask = passthrough_mask;
        for (uint32_t input = 0; input < controls.size(); ++input) {
            if (!(valid_mask & (1u << input))) continue;
            const uint32_t control = controls[input];
            // OFFSET bit 5 + FLAT_SHADE is the register form emitted for explicit parameter-cache
            // pass-through. A nonzero low OFFSET distinguishes it from the constant-default form;
            // metadata's explicit mask covers the otherwise ambiguous PARAM0 encoding.
            if ((control & 0x420u) == 0x420u && (control & 0x1fu) != 0u)
                mask |= 1u << input;
        }
        return mask & valid_mask;
    }

    uint32_t ambiguous_passthrough_mask() const {
        uint32_t mask = 0;
        for (uint32_t input = 0; input < controls.size(); ++input) {
            if (!(valid_mask & (1u << input))) continue;
            const uint32_t control = controls[input];
            if ((control & 0x420u) == 0x420u && (control & 0x1fu) == 0u)
                mask |= 1u << input;
        }
        return mask;
    }

    void merge_compatible_passthrough(
            const std::array<uint32_t, 32>& metadata_controls,
            uint32_t metadata_valid_mask, uint32_t metadata_passthrough_mask) {
        uint32_t candidates = valid_mask & metadata_valid_mask & metadata_passthrough_mask;
        for (uint32_t input = 0; input < controls.size(); ++input) {
            const uint32_t bit = 1u << input;
            if (!(candidates & bit)) continue;
            // OFFSET and FLAT_SHADE must describe the same producer PARAM. Ignore unrelated control
            // fields so metadata can disambiguate custom PARAM0 without overriding register wiring.
            if ((controls[input] & 0x43fu) == (metadata_controls[input] & 0x43fu))
                passthrough_mask |= bit;
        }
    }

    bool operator==(const PixelInputMapping&) const = default;
};

// Reconcile the register-backed linkage used by a live draw with richer AGC shader metadata. The
// registers remain authoritative whenever present; metadata only supplies the custom PARAM0 bit that
// SPI_PS_INPUT_CNTL cannot distinguish from a flat default input.
inline PixelInputMapping resolve_pixel_input_mapping(
        PixelInputMapping registered, const PixelInputMapping& metadata,
        bool* resolved_from_metadata = nullptr) {
    const bool use_metadata = !registered.valid_mask && metadata.valid_mask;
    if (use_metadata) {
        registered = metadata;
    } else if (metadata.passthrough_mask) {
        registered.merge_compatible_passthrough(
            metadata.controls, metadata.valid_mask, metadata.passthrough_mask);
    }
    if (resolved_from_metadata) *resolved_from_metadata = use_metadata;
    return registered;
}

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
// packed interface understood by recompile_fragment(). Custom/pass-through attributes instead expose
// the three raw vertex values that the guest shader explicitly interpolates. Locations are capped at
// Vulkan's portable 128-component fragment-input minimum (32 vec4 locations); `valid == false` keeps
// overflow visible.
struct FragmentInterpolationLayout {
    static constexpr uint32_t kUnusedLocation = UINT32_MAX;
    std::array<std::array<uint32_t, 3>, 32> parameter_locations{}; // [attr][0=P10,1=P20,2=P0]
    std::array<uint32_t, 7> system_locations{};                    // PS system fields 0..6
    uint32_t attribute_mask = 0;                                  // attributes consumed by VINTRP
    uint32_t smooth_mask = 0;                                     // attributes consumed by P1/P2
    uint32_t passthrough_mask = 0;                                // explicit inputs exposing raw vertex values
    bool requires_geometry = false;
    bool valid = true;

    FragmentInterpolationLayout();
};

FragmentInterpolationLayout fragment_interpolation_layout(
    const uint32_t* code, size_t dwords,
    const PixelSystemInputMapping* system_inputs = nullptr,
    const PixelInputMapping* pixel_inputs = nullptr);

// Generate the descriptor-free triangle geometry stage described above. Returns {} when no fallback
// is required or the packed interface is invalid. Triangle lists, strips, fans, and the RectList
// triangle-strip lowering all feed Vulkan's `Triangles` geometry input primitive. The optional
// capture flag decorates this final pre-rasterization stage for the geometry diagnostic only.
std::vector<uint32_t> recompile_interpolation_geometry(
    const FragmentInterpolationLayout& layout, bool capture_position = false);

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
    // Non-zero only when the live backend can REQUIRE this exact Vulkan subgroup size and full
    // compute subgroups. The shell remaps guest local coordinates into subgroup lane order, so
    // complex guest-wave CFGs may replace portable workgroup-scratch vote/scan emulation with native
    // subgroup operations without assuming Vulkan's LocalInvocationIndex ordering.
    uint32_t native_subgroup_size = 0;
    // Per-format VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT support published by the device-owning
    // frontend. Offline callers default to the portable raw path; live execution supplies the exact
    // physical-device mask so unsupported typed formats compile to the raw uvec4 fallback.
    uint32_t native_storage_format_support = 0;
    // Use an exact typed R32_UINT storage image for R11G11B10 when the device cannot store that
    // packed float format natively. The generated shader packs/unpacks the guest word in SPIR-V,
    // avoiding the portable but much wider RGBA32_UINT CPU-conversion interchange format.
    bool packed_r11_storage = true;
};

// Translate a game compute program without the synthetic binding-0 input / binding-1 output used by
// recompile_valu. Memory effects come only from the program's actual resource operations.
std::vector<uint32_t> recompile_compute(const uint32_t* code, size_t dwords,
                                        const ShaderResourceTable* rt,
                                        const ComputeShaderConfig& config);

// True when the program contains enough proven wave work to justify an exact multi-wave subgroup
// shell: either the canonical low/high MBCNT pair, or at least four VCC/EXEC branches accepted by the
// structured compute emitter in a shader containing only top-level guest workgroup barriers. Portable
// compute lowers those operations through synchronized workgroup scratch; native compute uses subgroup
// scans/votes. This deliberately conservative structural signal is independent of title and address.
bool compute_shader_prefers_native_multiwave(const uint32_t* code, size_t dwords);
bool compute_shader_prefers_native_multiwave(const std::vector<Rdna2Inst>& instructions,
                                             const uint32_t* code, size_t dwords);

// DPP/permlane operations are native subgroup shuffles: quad permutations need four contiguous
// lanes, row operations need 16, and PERMLANEX16 needs a paired 32-lane row. The backend rejects
// (rather than miscompiles) a module on a narrower host.
// Zero means that the module does not use a native compute-subgroup operation.
uint32_t compute_spirv_min_subgroup_size(const std::vector<uint32_t>& spirv);

// Recompile a pixel/fragment shader to a fragment SPIR-V module: run the VALU, and on EXP to MRT0/1
// write vec4(src0..3) to the matching color output. NULL-only shaders retain discard/EXEC effects and
// intentionally expose no color output. Returns {} if unsupported / no implemented export.
// An optional ShaderResourceTable enables memory ops (SMEM/MUBUF/MTBUF) with resolved bindings.
std::vector<uint32_t> recompile_fragment(const uint32_t* code, size_t dwords,
                                         const ShaderResourceTable* rt = nullptr,
                                         const PixelSystemInputMapping* system_inputs = nullptr,
                                         uint32_t pcrel_dispatch_target = UINT32_MAX,
                                         const FragmentInterpolationLayout* interpolation = nullptr,
                                         bool wave32 = false);

// Test hook for the low-half EXEC/VCC mask path. Production fragment compilation supplies the same
// mode from SPI_PS_IN_CONTROL.PS_W32_EN.
std::vector<uint32_t> recompile_fragment_wave32_for_test(
    const uint32_t* code, size_t dwords);
// Test hook for the byte-exact legacy capture exception: it must select one coherent Wave32
// contract, not Wave32 masks inside a Wave64 native subgroup.
uint32_t fragment_effective_wave_size_for_test(uint32_t requested_wave_size,
                                               size_t program_dwords,
                                               uint64_t program_hash);

// Recompiled fragment wave operations use native Vulkan subgroup instructions and therefore require
// an exact guest-wave subgroup. Returns zero for ordinary modules and 32/64 for that contract.
uint32_t fragment_spirv_required_subgroup_size(const std::vector<uint32_t>& spirv);
// WHY a fragment module requires its exact guest-wave width (#2147). The width alone is not
// actionable: a shader needing 64 for lane IDENTITY can never run at 32, because
// SubgroupLocalInvocationId IS the guest lane id and 32 lanes silently renumber it. One needing
// 64 only for a branch-guard vote may be width-agnostic, since two 32-lane votes union to the
// same executed-pixel set. Those have opposite prospects under a wave32 lowering, and the skip
// diagnostic could not tell them apart: kFragmentSubgroup* below are CAPABILITY bits, and the
// lane-id path declares only base GroupNonUniform, so it reported required-ops=0.
//
// These name the emitter path that raised the contract. A module may set several.
inline constexpr uint32_t kFragmentWaveReasonLaneId     = 1u << 0;  // lane id from SubgroupLocalInvocationId
inline constexpr uint32_t kFragmentWaveReasonWaveAny    = 1u << 1;  // OpGroupNonUniformAny (vote)
inline constexpr uint32_t kFragmentWaveReasonDppRow16   = 1u << 2;  // DPP row, needs >= 16
inline constexpr uint32_t kFragmentWaveReasonPermLane32 = 1u << 3;  // PERMLANEX16, needs >= 32
inline constexpr uint32_t kFragmentWaveReasonReadLane64 = 1u << 4;  // V_READLANE_B32 across a wave64
inline constexpr uint32_t kFragmentWaveReasonShuffle    = 1u << 5;  // lane-addressed shuffle
// OpGroupNonUniformBallot. Separate from WaveAny because the two have OPPOSITE prospects under a
// wave32 lowering, which is the same distinction #2147 drew between lane-id and vote and the same
// reason it drew it (#2441). A ballot's bits become guest scalar DATA -- half a mask reported as
// whole is silently wrong -- so its width requirement can never be relaxed, whereas a vote's may
// be. Reporting both as "wave-any" made the recoverable and unrecoverable cases one number: on
// PPSA04263 that number was 68 of 112 skipped fragment shaders, with no way to ask how it divides.
inline constexpr uint32_t kFragmentWaveReasonWaveBallot = 1u << 6;  // OpGroupNonUniformBallot (reduce)

// Reasons recorded by the emitter, or UINT32_MAX when the module carries no reason marker at
// all (built, cached or captured before #2147). Absent must not read as none.
uint32_t fragment_spirv_required_subgroup_reasons(const std::vector<uint32_t>& spirv);

inline constexpr uint32_t kFragmentSubgroupVote = 1u << 0;
inline constexpr uint32_t kFragmentSubgroupArithmetic = 1u << 1;
inline constexpr uint32_t kFragmentSubgroupShuffle = 1u << 2;
inline constexpr bool fragment_subgroup_features_supported(uint32_t required,
                                                           uint32_t supported) {
    return (required & ~supported) == 0;
}
// Capability requirements encoded in a fragment module, independent of the host graphics API's bit
// values. The backend maps these to VkSubgroupFeatureFlagBits before accepting a wave64 pipeline.
uint32_t fragment_spirv_required_subgroup_features(const std::vector<uint32_t>& spirv);
bool fragment_spirv_uses_internal_gds(const std::vector<uint32_t>& spirv);

// Packed RGBA component-enable nibbles for the first realized color export to MRT0/MRT1. EXP.EN is
// an attachment write mask, not a request to source disabled VGPRs; callers intersect this with the
// fixed-function CB_TARGET_MASK/CB_SHADER_MASK before creating the Vulkan pipeline. This preserves
// destination channels disabled by a partial export exactly as RDNA2 does.
uint32_t fragment_color_export_mask(const uint32_t* code, size_t dwords);

// Recompile a vertex shader to a vertex SPIR-V module: v0 = gl_VertexIndex, run the VALU, and on EXP
// to a POS target write vec4(src0..3) to gl_Position. Returns {} if unsupported / no position export.
// An optional ShaderResourceTable enables vertex fetch (buffer_load_format_*) + constant loads.
// `capture_position` (geometry-probe, gated): decorate gl_Position for VK_EXT_transform_feedback capture
// so the host can read back this draw's post-transform clip-space vertices. Inert to the computation.
// `virtual_lds_dwords` is the exact SPI_SHADER_PGM_RSRC2_GS allocation for an NGG program.  The
// portable vertex shell represents one live guest lane, so that lane's LDS is private to the host
// invocation; zero keeps graphics DS operations fail-closed.
std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords,
                                       const ShaderResourceTable* rt = nullptr,
                                       const PixelInputMapping* pixel_inputs = nullptr,
                                       bool capture_position = false,
                                       uint32_t virtual_lds_dwords = 0);

// Recompile a separately-installed vertex-fetch prolog and its main shader as one architectural
// program. The validated tail transfer is replaced by fallthrough, preserving every SGPR/VGPR value
// produced by the prolog. Resource fetch_pc values for the main program must already be shifted by
// VertexPrologInfo::prefix_dwords in `rt`.
std::vector<uint32_t> recompile_vertex_chain(const uint32_t* prolog, size_t prolog_dwords,
                                             const uint32_t* main, size_t main_dwords,
                                             const ShaderResourceTable* rt = nullptr,
                                             const PixelInputMapping* pixel_inputs = nullptr,
                                             bool capture_position = false,
                                             uint32_t virtual_lds_dwords = 0);

// Test hook: allow a tiny synthetic NGG shader to reach the terminal EXEC-gated export lowering so
// Vulkan tests can execute both active and inactive outcomes. Production recompile_vertex keeps that
// exception restricted to the byte-exact observed Astro wrapper.
std::vector<uint32_t> recompile_vertex_terminal_ngg_gate_for_test(
    const uint32_t* code, size_t dwords);

// Test hook: exercise operations whose semantics depend on the exact Astro one-lane NGG projection.
// Production recompile_vertex only enables that projection for byte-exact observed wrappers.
std::vector<uint32_t> recompile_vertex_ngg_one_lane_for_test(
    const uint32_t* code, size_t dwords);

// How much of a shader the recompiler currently covers (per-instruction), without requiring the
// stream to be a complete vertex/fragment. `alu` = instructions emit_alu handles (VALU/scalar/
// control-flow); `exports` = EXP (handled by the stage recompilers); `unsupported` = not yet handled
// (memory ops / unknown), with the first such (format, opcode) recorded. A data-driven coverage metric.
struct RecompileCoverage {
    uint32_t total = 0, alu = 0, exports = 0, unsupported = 0;
    // Instructions the recompiler handles GIVEN the right context (a resource table for MIMG /
    // MUBUF / MTBUF, or a fragment stage for VINTRP) but that recompile_coverage — which runs
    // table-less on a compute shell — cannot exercise. Counting them apart from `unsupported` gives an
    // honest "recompilable in context" number; the table-less `alu`/`unsupported` split understates it.
    uint32_t table_dependent = 0;
    int      first_bad_fmt = -1;   // Rdna2Format of the first TRULY-unsupported instruction (-1 if none)
    uint32_t first_bad_op  = 0;
    uint32_t first_bad_pc  = 0xFFFFFFFFu; // dword offset of that instruction (-1 if none)
};
RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords);

// A general (non-scratch) FLAT load resolved to a base user-SGPR pointer pair (#1171). The 64-bit
// address VGPR pair of a `base + offset` flat access is `v[vaddr_lo_reg : vaddr_lo_reg+1]`, and the
// base pointer's low/high dwords live in consecutive user SGPRs `s[base_sgpr : base_sgpr+1]`. With the
// base identified, the executor can bind the containing guest allocation as an SSBO and the emitter can
// lower the load to an indexed read.
struct FlatLoadInfo {
    uint32_t load_pc = 0;        // dword offset of the flat_load in the stream
    uint32_t vaddr_lo_reg = 0;   // VADDR VGPR; the 64-bit address pair is v[vaddr_lo_reg : +1]
    uint32_t dst_reg = 0;        // first destination VGPR
    uint32_t bits = 0;           // access width in bits: 8 / 16 / 32
    uint32_t components = 0;     // dword count (x1/x2/x3/x4)
    bool     sign_extend = false;
    int32_t  base_sgpr = -1;     // resolved base-pointer low-dword user-SGPR (high dword is +1); -1 = unresolved
};
struct FlatLoadAnalysis {
    bool any = false;            // at least one general (non-scratch) FLAT memory op is present
    bool all_resolved = true;    // every general FLAT op is a resolvable load (false => keep rejecting)
    std::vector<FlatLoadInfo> loads;
};
// Identify general FLAT loads (segment != scratch) and resolve each to a base user-SGPR pointer pair,
// so a windowed-SSBO lowering can replace today's fail-visible reject (#1171). Fail-visible by design:
// any store/atomic/LDS/global-with-saddr form, or an address that does not reduce to
// `user_sgpr_pair + offset`, leaves `all_resolved=false` and the caller keeps rejecting the shader.
FlatLoadAnalysis analyze_flat_loads(const uint32_t* code, size_t dwords, uint32_t user_sgpr_count);

// Test hook (#1183): the set of forward s_cbranch_execz pcs the recompiler classifies as safe to
// linearize (drop the branch, run the block under per-lane EXEC). A loop-EXIT execz — one enclosed by a
// backward branch that jumps to at-or-before it — must NOT appear here, so detect_divergent_loops claims
// it and emit_divloop reconstructs the structured loop; a plain guard-to-end/if execz still does.
std::vector<uint32_t> safe_execz_branches_for_test(const uint32_t* code, size_t dwords);
std::vector<uint32_t> mask_test_branches_for_test(const uint32_t* code, size_t dwords,
                                                  bool wave32);

} // namespace prosper::gpu
