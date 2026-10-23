#ifndef PROSPER_GPU_RECOMPILER_SPIRV_COMPUTE_MEMBER_SCOPE
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"   // IWYU pragma: export
#else
#ifndef PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_WORKGROUP_STAGES_INL
#define PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_WORKGROUP_STAGES_INL
// Class-scope fragment: preserve the mechanically moved member bytes.
// clang-format off
    // LDS (Local Data Share) — a workgroup-shared u32 array for compute ds_read/ds_write. NGG shaders
    // are lowered as one independent Vulkan vertex invocation, so their LDS becomes Function-private:
    // the single modeled guest lane can observe its own writes, with no cross-invocation races or an
    // illegal Workgroup variable in the Vertex execution model. Declared on first use and sized to
    // `lds_dwords`. RDNA2's per-workgroup LDS MAX is 64 KB (16384 dwords); the real
    // per-shader allocation is COMPUTE_PGM_RSRC2.LDS_SIZE. `lds_dwords` defaults to 4096 (16 KB) — a
    // shader whose real allocation exceeds that would ds_read/write past the array (OOB Workgroup
    // access, UB — NOT covered by robustBufferAccess), so recompile_valu raises it from the plumbed
    // size when known (#130). Kept at 16 KB by default because it must also stay within the target
    // device's VkPhysicalDeviceLimits::maxComputeSharedMemorySize (e.g. llvmpipe = 32 KB), so we can't
    // just declare the full 64 KB unconditionally.
    uint32_t lds_dwords = 4096;
    // NGG is lowered through a portable one-live-lane-per-Vulkan-invocation vertex shell. Its LDS
    // allocation therefore becomes per-invocation Function memory. The exact allocation comes from
    // RSRC2_GS and remains disabled unless the caller supplied that hardware state.
    uint32_t vertex_lds_dwords = 0;
    uint32_t vertices_per_instance = 0;
    uint32_t lds_var = 0, t_ptr_lds_u32 = 0;
    void declare_lds();
    uint32_t lds_load(uint32_t idx);
    // EXEC-predicated LDS load. The inactive arm must not merely discard a loaded value: its
    // address VGPR retains the old, potentially arbitrary bits when the instruction is masked off,
    // so even forming an AccessChain/OpLoad there can access outside the Workgroup array or race an
    // active lane. Keep the memory access inside the active selection and join the architectural
    // old destination through OpPhi, matching the returning-atomic helper below.
    uint32_t lds_load(uint32_t idx, bool predicated, uint32_t pred, uint32_t fallback);
    // Store to LDS[idx]; EXEC-predicated (conditional store) under a narrowed mask, like cbuf_store.
    void lds_store(uint32_t idx, uint32_t value, bool predicated, uint32_t pred,
                   bool atomicize = false);
    // Unsigned LDS atomic RMW. The old value is intentionally discarded; the non-returning RDNA DS
    // form exposes only the memory effect. Workgroup scope + WorkgroupMemory AcquireRelease matches
    // the storage class and the barrier helper used around cooperating accesses.
    void lds_atomic(uint32_t op, uint32_t idx, uint32_t value, bool predicated, uint32_t pred);
    // RDNA2 DS_MIN/MAX_F32 quiet and propagate signaling NaNs before numeric selection; a lone quiet
    // NaN yields the number. FP32_DENORM controls whether subnormal inputs compare at their true
    // value or as signed zero, while signed-zero ties choose -0 for min / +0 for max.
    // SPIR-V 1.3 has no core floating atomic min/max, while integer AtomicS/UMin orders the raw bits
    // incorrectly. Implement the architectural operation as a u32 compare-exchange loop instead.
    // The ordering key is monotonic for every non-NaN binary32 bit pattern and therefore does not
    // depend on the host driver's denormal mode. AMD's both-qNaN selection returns the first operand;
    // the resident `*ptr` value is that first operand for an atomic min/max update.
    void lds_atomic_fminmax(uint32_t idx, uint32_t value, bool is_min,
                            bool predicated, uint32_t pred);
    // Returning LDS atomic RMW. Inactive lanes must neither touch LDS nor observe an undefined
    // atomic result, so the predicated form joins the old destination fallback through OpPhi.
    uint32_t lds_atomic_rtn(uint32_t op, uint32_t idx, uint32_t value,
                            bool predicated, uint32_t pred, uint32_t fallback);
    // s_barrier: workgroup execution + memory barrier (OpControlBarrier).
    void barrier();

    // Dispatcher-only scratch. Unlike guest LDS, this is private recompiler state: one flag slot per
    // padded hardware-wave lane, one result per wave, and one whole-workgroup liveness result.
    uint32_t cfg_scratch = 0, t_ptr_cfg_u32 = 0, cfg_scratch_dwords = 0;
    bool declare_cfg_scratch(uint32_t dwords);
    uint32_t cfg_scratch_load(uint32_t idx);
    void cfg_scratch_store(uint32_t idx, uint32_t value);

    // --- Wave model (cross-lane ops via workgroup scratch). A dedicated LDS scratch array (separate
    // from ds_read/write's lds_var) holds each lane's contribution so cross-lane reductions work.
    // Inactive lanes still EXECUTE (exec is a per-lane predication bool), so all 64 reach the barriers —
    // valid only at wave-uniform points (the caller must not emit these inside divergent control flow). ---
    uint32_t v_localid = 0, localid = 0, lds_wave = 0, t_ptr_wg_u32b = 0;
    void declare_wave_lds();
    // Exact guest-wave vote at a workgroup-uniform site. Vulkan subgroup width is implementation-
    // defined and need not match the guest's 32/64-lane wave, so publish every lane through shared
    // scratch and reduce only lanes with the same guest-wave index. One lane per guest wave performs
    // the reduction and publishes it in a separate result slot.  Besides avoiding an assumption about
    // Vulkan subgroup membership, this makes the amount of LDS work linear in the workgroup size:
    // the old per-invocation reduction made a 256-thread group perform 65,536 loads for every vote.
    // The first barrier publishes lane bits and the second publishes wave results. The caller proves
    // the site is outside wave-divergent structured control flow.
    uint32_t guest_wave_any(uint32_t active_bool);
    uint32_t guest_wave_popcount(uint32_t active_bool);
    // V_READLANE_B32 at a workgroup-uniform site. Publish every invocation's source value and
    // address the selected lane within this invocation's own guest wave. This is exact at any
    // host subgroup width; unlike a native subgroup shuffle it does not require a Wave64 guest
    // wave to fit inside one Vulkan subgroup. V_READLANE ignores EXEC, so publication is
    // unconditional. Missing lanes in a partial final wave read as zero.
    uint32_t guest_wave_readlane(uint32_t source_value, uint32_t selector);
    // v_mbcnt_lo/hi: count active lanes below this one. active_bool = this lane's mask bit (EXEC); acc =
    // src1 accumulator (bits); `lo` selects the [0,32) half (lo) or [32,64) half (hi). Combined lo→hi over
    // a wave = the lane's compaction index among active lanes. Populate LDS[lane]=active, barrier, then an
    // unrolled prefix-count over the half; trailing barrier so a following mbcnt can safely re-populate.
    uint32_t mbcnt(uint32_t active_bool, uint32_t acc_bits, bool lo);
    // DS_APPEND/DS_CONSUME: one atomic add/subtract per emulated hardware wave, changing the counter
    // by popcount(EXEC), with the old value broadcast to every lane. GDS operations target the
    // backend-owned persistent device buffer; ordinary operations target workgroup LDS. Every
    // Vulkan invocation participates in the barriers;
    // the per-lane EXEC bool only contributes 0/1 to the reduction. A partial final hardware wave
    // (including a one-thread workgroup) contains only the launched lanes; every scratch read remains
    // statically in-bounds and absent hardware lanes contribute zero.
    uint32_t wave_append(uint32_t index, uint32_t active_bool, bool consume = false,
                         bool gds = false);
    uint32_t native_wave_append(uint32_t lds_idx, uint32_t active_bool,
                                uint32_t consume);
    uint32_t b_iadd(uint32_t a, uint32_t b_) { uint32_t r = id(); put(code, Op_IAdd, {t_u32, r, a, b_}); return r; }
    void validate_cbuf_table_contracts(const ShaderResourceTable*& rt);
    void declare_cbuf_block_types(uint32_t& t_rta_u, uint32_t& t_struct_u);
    void declare_additional_cbufs(const ShaderResourceTable*& rt, uint32_t& t_struct_u);
    // Declare the two scalar-memory constant/vertex buffers (bindings 2 & 3) that SMEM / buffer_load_
    // format_* read. Called by every shell (compute/vertex/fragment) so cbuf_load works in each.
    // Requires t_u32 to already be declared. Unused by shaders without memory ops.
    void declare_cbufs(const ShaderResourceTable* rt = nullptr);
    // Store a VGPR (bits) as one float per invocation: b[gid.x] (stride 1), independent of input stride.
    void     store_output(uint32_t bits);
    // Guest merged-geometry execution probe: retain each enabled EXP word in a fixed per-lane
    // uint32 record. This is not renderer publication or a substitute for primitive assembly.
    void     store_output_word(uint32_t bits, uint32_t word_stride, uint32_t word_offset,
                               uint32_t exec_bool = 0);
    // EXEC-predicated store: lanes with exec=false keep the output slot's prior value. Modeled with
    // OpSelect (no control flow needed) — load old, pick new-vs-old by the per-lane exec bool, store.
    // Correct for the straight-line "conditional write / discard" pattern (v_cmpx narrows EXEC).
    void     store_output_pred(uint32_t bits, uint32_t exec_bool);
    uint32_t btrue() { uint32_t r = id(); put(types, Op_ConstantTrue, {t_bool, r}); return r; }
    uint32_t load_push_constant(uint32_t index);
    // Logical AND / OR of two bools (EXEC narrowing / saveexec).
    uint32_t land(uint32_t a, uint32_t b_);
    uint32_t lor(uint32_t a, uint32_t b_);

    void begin_compute_layout(uint32_t& local_x, uint32_t& local_y, uint32_t& local_z,
                              uint32_t& t_rta, uint32_t& t_struct, uint32_t& push_constant_dwords);
    void begin_compute_invocations(uint32_t& lbl, uint32_t& local_x, uint32_t& local_y,
                                   uint32_t& local_z);
    void begin(uint32_t input_stride, const ShaderResourceTable* rt = nullptr,
               uint32_t local_x = 64, uint32_t local_y = 1, uint32_t local_z = 1,
               uint32_t hardware_wave_size = 64, uint32_t push_constant_dwords = 0,
               bool raw_word_output = false, bool raw_word_input = false);
    // AGC thread-dimension mode can request a non-multiple of the shader's local size. Vulkan still
    // launches the final complete workgroup, so make its excess invocations branch directly to the
    // function merge instead of executing guest instructions (and, in particular, memory stores).
    uint32_t invocation_within_extent(uint32_t threads_x, uint32_t threads_y,
                                      uint32_t threads_z);
    void guard_invocation_extent(uint32_t threads_x, uint32_t threads_y, uint32_t threads_z);
    // --- Fragment-shader shell: vec4 outputs for the MRT0..MRT7 exports. ---
    //
    // This array WAS sized 2, and that single number was the whole of GTA V's missing 3D world.
    // Its G-buffer pass exports five render targets
    // (c0=albedo c1=... c2=... c3=... c4=..., tmask=0x000fffff smask=0x0003ffff), and every part of
    // the pipeline below the shader already carried eight: the render state decodes all eight slots,
    // `active_color_count` scans all eight, and the backend's render pass, framebuffer and blend
    // state are all generic over `color_count`. Only the fragment recompiler stopped at two -- so
    // three of the five attachments were dropped, the deferred lighting pass sampled buffers nothing
    // had written, and the world rendered into a G-buffer that was then lit by nothing.
    //
    // The rest of this shell was already written generically against `v_color.size()`; the size was
    // the only thing holding it to two.
    uint32_t t_v4f = 0;
    std::array<uint32_t, kFragmentColorOutputs> v_color{};
    // #4703: the numeric class of each colour output, set before begin_fragment(). An integer
    // attachment needs a uvec4/ivec4 output; Float is the historical vec4 declaration.
    std::array<FragmentOutputClass, kFragmentColorOutputs> color_output_class{};
    uint32_t t_v4i_cache = 0;
    uint32_t t_v4i() {
        if (!t_v4i_cache) {
            t_v4i_cache = id();
            put(types, Op_TypeVector, {t_v4i_cache, t_i32, 4});
        }
        return t_v4i_cache;
    }
    void begin_fragment(const ShaderResourceTable* rt = nullptr, uint32_t color_mask = 1u);
    // Declare each enabled colour output variable with its class's pointer type (#4703).
    void declare_color_outputs(uint32_t t_ptr_out_float);
    // Write a vec4(r,g,b,a) (bit-operands) to the matching fragment color output. An integer
    // output (color_output_class) receives the operand bits unchanged, as uvec4 or ivec4.
    void export_color(uint32_t mrt, uint32_t r, uint32_t g, uint32_t bl, uint32_t a);
    // Fragment depth export (EXP target 8 = MRTZ) — a lazily declared BuiltIn FragDepth output.
    // Writing FragDepth requires ExecutionMode DepthReplacing (fixed-function Z is replaced by the
    // shader's exported value, matching the hardware's shader-Z path). Silently DROPPING the
    // target-8 export left occlusion to interpolated Z — wrong for any depth-writing shader.
    uint32_t v_fragdepth = 0;
    void export_depth(uint32_t z_bits);

    // Fragment sample-coverage export (EXP target 8 / VSRC2). Vulkan exposes this as the first
    // element of a BuiltIn SampleMask output array, matching RDNA's 32-bit sample-mask payload.
    uint32_t v_sample_mask = 0;
    uint32_t t_sample_mask = 0;
    void export_sample_mask(uint32_t mask_bits);

    // --- Vertex-shader shell: draw inputs + gl_Position (member 0 of a gl_PerVertex Block). ---
    uint32_t v_vid = 0, v_iid = 0, v_pos = 0, t_ptr_out_v4f = 0;
    void begin_vertex(const ShaderResourceTable* rt = nullptr);
    // Load gl_VertexIndex as raw bits (VGPR v0 for a vertex shader).
    uint32_t load_vertex_index() {
        if (ngg_workgroup_shell && ngg_vertex_index_value) return ngg_vertex_index_value;
        uint32_t r = id(); put(code, Op_Load, {t_i32, r, v_vid}); return i2u(r);
    }
    uint32_t load_instance_index() {
        if (ngg_workgroup_shell && ngg_instance_index_value) return ngg_instance_index_value;
        uint32_t r = id(); put(code, Op_Load, {t_i32, r, v_iid}); return i2u(r);
    }
    // Hardware packs consecutive vertex/instance invocations into merged guest waves. VertexIndex
    // repeats for each instance, so flatten it with the captured per-instance vertex count before
    // deriving lane and wave IDs. Standalone fixtures retain their historical vertex-only ID.
    uint32_t vertex_invocation_id();
    uint32_t guest_lane_id();
    // Write vec4(x,y,z,w) (bit-operands) to gl_Position (EXP POS0).
    void export_position(uint32_t x, uint32_t y, uint32_t z, uint32_t w);

    // --- Interpolated I/O varyings: VS EXP PARAM_n (output) <-> FS v_interp attribute (input) ---
    std::unordered_map<uint32_t, uint32_t> in_varying, out_varying;
    uint32_t t_ptr_in_v4f = 0;
    const FragmentInterpolationLayout* fragment_interpolation = nullptr;
    // Attributes read via v_interp_mov (a raw per-vertex / provoking-vertex value, NOT rasterizer-
    // interpolated) — their FS Input varying is decorated Flat so the driver delivers the provoking
    // vertex value instead of a smooth blend of the three (#152). Populated by recompile_fragment's
    // pre-scan; an attribute read via BOTH v_interp_mov (flat) and v_interp_p2 (smooth) is rejected
    // there (a varying can't be both), so this set never contradicts a smooth read.
    std::unordered_set<uint32_t> flat_attrs;
    // FS: an Input vec4 at Location=attr, rasterizer-interpolated (or Flat if flat_attrs says so, OR
    // if the guest's own PS_INPUT_CNTL declared FLAT_SHADE for this slot -- #3051. Both sources feed
    // the same SPIR-V property and are unioned: flat_attrs recognises flat-ness from the shader's own
    // choice of VINTRP opcode (v_interp_mov), while fragment_interpolation->flat_mask recognises it
    // from the guest's declared semantic, which still applies to an ordinary v_interp_p1/p2 read).
    uint32_t frag_input(uint32_t attr);
    // Read component `chan` (0..3) of interpolated attribute `attr` -> float bits (v_interp_p2 / mov).
    uint32_t interp_read(uint32_t attr, uint32_t chan);
    // AMD v_interp_mov selects one of the triangle coefficients directly: P10, P20, or P0. On the
    // portable geometry fallback each selected coefficient has a packed Flat vec4 input. The legacy
    // P0-only path remains the ordinary Flat attribute at Location=attr and needs no extra stage.
    uint32_t interp_parameter(uint32_t attr, uint32_t chan, uint32_t selector);
    uint32_t system_interpolation_component(uint32_t field, uint32_t component);
    // FS: gl_FragCoord (BuiltIn 15), lazily declared — used by the DPP quad_perm lowering.
    uint32_t v_fragcoord = 0;
    uint32_t fragcoord_var();
    uint32_t fragcoord_component(uint32_t component);
    // The guest's POS_{X,Y,Z,W}_FLOAT pixel inputs (component 0..3). X, Y and Z are FragCoord's.
    // GFX10 loads clip-space w into POS_W_FLOAT -- SV_Position.w, the view depth -- while SPIR-V's
    // FragCoord.w is 1/w. The guest reads the VGPR as a depth with no reciprocal of its own: UE4's
    // dithered near-camera fade computes saturate((w - 100) / 50) from it, and given 1/w it faded
    // Kena's foliage out (KENA_STATUS.md). Mesa's AMD drivers agree (main @ 1375dc60642a; the
    // installed 26.1.4 predates the mechanism): radv_shader.c:72 and radeonsi's
    // gfx/si_gfx_screen.c:344 set nir_frag_coord_use_w_rcp; nir_builder.c:809 then builds
    // FragCoord.w as frcp(load_frag_coord_w_rcp); and ac_nir_lower_intrinsics_to_args.c:205 reads
    // that intrinsic from frag_pos[3], the POS_W VGPR -- so the register holds clip-space w.
    // CONFIDENCE: HIGH (guest disassembly and constants, and Mesa's lowering).
    uint32_t guest_pixel_position_component(uint32_t component) {
        const uint32_t value = fragcoord_component(component);
        return component == 3 ? fbin(Op_FDiv, uconst(0x3f800000u), value) : value;
    }
    // SPI_PS_INPUT_ENA.ANCILLARY_ENA places the rasterized primitive's render-target
    // array index in bits 26:16 of its one VGPR. Other ancillary fields remain zero.
    uint32_t v_front_facing = 0;
    uint32_t fragment_front_face_word(FragmentFrontFaceEncoding encoding);
    uint32_t fragment_ancillary_layer_bits();
    // DPP16 quad_perm (#273 — DOLL's manual ddx/ddy in its sharpen/AA PSs): the value of `x` at quad
    // lane t = QP[my_lane] is reconstructed as x + (tx-px)·dPdx(x) + (ty-py)·dPdy(x), where (px,py) =
    // (int(gl_FragCoord.xy) & 1) is this invocation's quad position and (tx,ty) = (t&1, t>>1). Exact
    // for values linear within the 2x2 quad — the same assumption hardware derivatives make (the
    // idiom's whole purpose IS ddx/ddy: e.g. `v_sub_f32_dpp v4, v2, v2@[0,0,2,2] qp[1,1,3,3]` =
    // v2@(1,py) - v2@(0,py) = dPdxFine). Fragment-only. CONFIDENCE: MED (render-tested vs a known
    // varying gradient).
    uint32_t dpp_quad(uint32_t x_bits, uint32_t ctrl);
    // VS: an Output vec4 at Location=loc (EXP PARAM_loc); uses t_ptr_out_v4f from begin_vertex.
    uint32_t vtx_output(uint32_t loc);
    void export_param(uint32_t loc, uint32_t x, uint32_t y, uint32_t z, uint32_t w);
    void geometry_store_system_values(uint32_t& bary_i, uint32_t& bary_j,
                                      std::array<uint32_t, 7>& system_outputs,
                                      const uint32_t& position);
    uint32_t geometry_slot_value(uint32_t& slot, const std::array<uint32_t, 3>& supplied,
                                 uint32_t& synthesized, bool& synthesize_rect,
                                 uint32_t& rect_shared_is_v1_wide,
                                 uint32_t& rect_shared_is_v0_wide);
    std::vector<uint32_t> geometry_capabilities(
        bool& capture_geometry_position, bool& synthesize_rect, const uint32_t& t_input_per_vertex,
        const uint32_t& t_output_per_vertex, std::array<uint32_t, 32>& attribute_inputs,
        std::array<uint32_t, 32>& attribute_outputs,
        std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
        const FragmentInterpolationLayout& layout, std::array<uint32_t, 7>& system_outputs,
        const uint32_t& input_position, const uint32_t& output_position, bool& publish_primitive_id,
        const uint32_t& primitive_in, const uint32_t& primitive_out, const bool& forward_layer,
        const uint32_t& layer_in, uint32_t& layer_input_location, const uint32_t& layer_out,
        const uint32_t& primitive_in_ptr, const uint32_t& primitive_out_ptr,
        const uint32_t& t_v4bool, const uint32_t& t_layer_inputs, const uint32_t& layer_in_ptr,
        const uint32_t& layer_in_element_ptr, const uint32_t& layer_out_ptr,
        const uint32_t& c_three, const uint32_t& t_input_positions,
        const uint32_t& t_input_varyings, const uint32_t& ptr_in_positions,
        const uint32_t& ptr_in_varyings, const uint32_t& ptr_in_v4f,
        const uint32_t& ptr_out_position, const uint32_t& ptr_out_v4f, const uint32_t& label);
    std::vector<uint32_t> geometry_interfaces(
        std::array<uint32_t, 32>& attribute_inputs, std::array<uint32_t, 32>& attribute_outputs,
        std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
        const FragmentInterpolationLayout& layout, std::array<uint32_t, 7>& system_outputs,
        const uint32_t& input_position, const uint32_t& output_position, bool& publish_primitive_id,
        const uint32_t& primitive_in, const uint32_t& primitive_out, const bool& forward_layer,
        const uint32_t& layer_in, uint32_t& layer_input_location, const uint32_t& layer_out,
        const uint32_t& primitive_in_ptr, const uint32_t& primitive_out_ptr, bool& synthesize_rect,
        const uint32_t& t_v4bool, const uint32_t& t_layer_inputs, const uint32_t& layer_in_ptr,
        const uint32_t& layer_in_element_ptr, const uint32_t& layer_out_ptr,
        const uint32_t& t_input_per_vertex, const uint32_t& t_output_per_vertex,
        const uint32_t& c_three, const uint32_t& t_input_positions,
        const uint32_t& t_input_varyings, const uint32_t& ptr_in_positions,
        const uint32_t& ptr_in_varyings, const uint32_t& ptr_in_v4f,
        const uint32_t& ptr_out_position, const uint32_t& ptr_out_v4f, const uint32_t& label);
    std::vector<uint32_t> geometry_types(
        bool& publish_primitive_id, const uint32_t& primitive_in_ptr,
        const uint32_t& primitive_out_ptr, const uint32_t& primitive_in,
        const uint32_t& primitive_out, bool& synthesize_rect, const uint32_t& t_v4bool,
        const bool& forward_layer, const uint32_t& t_layer_inputs, const uint32_t& layer_in_ptr,
        const uint32_t& layer_in_element_ptr, const uint32_t& layer_out_ptr,
        const uint32_t& layer_in, const uint32_t& layer_out, const uint32_t& t_input_per_vertex,
        const uint32_t& t_output_per_vertex, const uint32_t& c_three,
        const uint32_t& t_input_positions, const uint32_t& t_input_varyings,
        const uint32_t& ptr_in_positions, const uint32_t& ptr_in_varyings,
        const uint32_t& ptr_in_v4f, const uint32_t& ptr_out_position, const uint32_t& ptr_out_v4f,
        const uint32_t& input_position, const uint32_t& output_position,
        std::array<uint32_t, 32>& attribute_inputs, std::array<uint32_t, 32>& attribute_outputs,
        std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
        std::array<uint32_t, 7>& system_outputs, const uint32_t& label,
        const FragmentInterpolationLayout& layout);
    std::vector<uint32_t>
    geometry_inputs(const uint32_t& label, bool& publish_primitive_id, const uint32_t& primitive_in,
                    const bool& forward_layer, const uint32_t& layer_in_element_ptr,
                    const uint32_t& layer_in, std::array<uint32_t, 32>& attribute_inputs,
                    const uint32_t& ptr_in_v4f, const uint32_t& input_position,
                    bool& synthesize_rect, const uint32_t& t_v4bool,
                    const FragmentInterpolationLayout& layout, const uint32_t& primitive_out,
                    const uint32_t& layer_out, const uint32_t& ptr_out_v4f,
                    const uint32_t& output_position, std::array<uint32_t, 32>& attribute_outputs,
                    std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
                    std::array<uint32_t, 7>& system_outputs);
    std::vector<uint32_t> geometry_rect_selectors(
        bool& synthesize_rect, std::array<uint32_t, 3>& positions, const uint32_t& t_v4bool,
        std::array<uint32_t, 32>& attribute_inputs,
        std::array<std::array<uint32_t, 3>, 32>& attribute_values,
        const FragmentInterpolationLayout& layout, bool& publish_primitive_id,
        const uint32_t& primitive_out, const uint32_t& primitive_value, const bool& forward_layer,
        const uint32_t& layer_out, uint32_t& layer_value, const uint32_t& ptr_out_v4f,
        const uint32_t& output_position, std::array<uint32_t, 32>& attribute_outputs,
        std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
        std::array<uint32_t, 7>& system_outputs);
    std::vector<uint32_t>
    geometry_parameters(uint32_t& rect_shared_is_v1_wide, uint32_t& rect_shared_is_v0_wide,
                        std::array<uint32_t, 32>& attribute_inputs,
                        std::array<std::array<uint32_t, 3>, 32>& attribute_values,
                        const FragmentInterpolationLayout& layout, bool& synthesize_rect,
                        std::array<uint32_t, 3>& positions, uint32_t& rect_shared_is_v1,
                        uint32_t& rect_shared_is_v0, bool& publish_primitive_id,
                        const uint32_t& primitive_out, const uint32_t& primitive_value,
                        const bool& forward_layer, const uint32_t& layer_out, uint32_t& layer_value,
                        const uint32_t& ptr_out_v4f, const uint32_t& output_position,
                        std::array<uint32_t, 32>& attribute_outputs,
                        std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
                        std::array<uint32_t, 7>& system_outputs);
    std::vector<uint32_t> geometry_vertices(
        bool& synthesize_rect, uint32_t& rect_shared_is_v1_wide, uint32_t& rect_shared_is_v0_wide,
        uint32_t& rect_shared_is_v1, uint32_t& rect_shared_is_v0, bool& publish_primitive_id,
        const uint32_t& primitive_out, const uint32_t& primitive_value, const bool& forward_layer,
        const uint32_t& layer_out, uint32_t& layer_value, std::array<uint32_t, 3>& positions,
        uint32_t& rect_position, const uint32_t& ptr_out_v4f, const uint32_t& output_position,
        std::array<uint32_t, 32>& attribute_outputs,
        std::array<std::array<uint32_t, 3>, 32>& attribute_values,
        std::array<uint32_t, 32>& rect_attribute_values,
        std::array<std::array<uint32_t, 3>, 32>& parameter_outputs,
        std::array<std::array<uint32_t, 3>, 32>& parameters,
        std::array<uint32_t, 7>& system_outputs);

    // Descriptor-free geometry pass-through used when a fragment program asks for AMD's explicit
    // P0/P10/P20 vertex parameters on a Vulkan device without a barycentric/vertex-parameter
    // extension. Input assembly has already decomposed lists/strips/fans into triangles here. A
    // three-vertex PS5 RectList whose attributes come from a vertex buffer also comes through this
    // stage: GFX10 synthesizes its fourth corner after vertex shading, while Vulkan has no RectList
    // topology. Emitting P0,P1,P2,P3 as a strip preserves the hardware rectangle.
    //
    // WHICH corner is missing is decided per draw, not by a fixed index. Three supplied corners of a
    // rectangle are one SHARED corner (adjacent to both others) plus that corner's diagonal pair; the
    // synthesized vertex is the shared corner's opposite, `diag_a + diag_b - shared`. This used to be
    // hardcoded to `P1 + P2 - P0`, i.e. "the shared corner is always P0". Darksiders II's video blit
    // supplies its rectangle in the other order -- P0=bottom-right, P1=top-left, P2=top-right, so the
    // shared corner is P2 and the missing one is bottom-left. Under the hardcoded rule that draw
    // synthesized (-1,3) instead of (-1,-1): off-screen, so its half of the strip never rasterized and
    // the FMV rendered as the single upper-right triangle of #3507.
    //
    // The shared corner is identified geometrically, from the shaded positions: of the three pairwise
    // separations the largest is the diagonal, and the vertex not on it is the shared corner. For a
    // rectangle this is exact (the diagonal strictly exceeds either side), it needs no convention
    // guess, and it AGREES with the old formula precisely when the old formula was right -- when P0 is
    // the shared corner it selects `P1 + P2 - P0` -- so no draw that renders correctly today changes.
    // Every varying and the explicit-parameter barycentrics follow the same selected index.
    //
    // CONFIDENCE: MED on the choice of discriminator, HIGH that it beats the fixed index it replaces.
    // "Longest separation is the diagonal" is exact for a rectangle and for any parallelogram whose
    // shared corner is not acute enough to make a side exceed a diagonal; a sufficiently SHEARED
    // parallelogram breaks it, and the rule would then complete the wrong corner. Every RectList
    // observed so far is an axis-aligned screen-space rectangle, where it cannot happen. Separation is
    // measured on the full 4-component clip position, which is exact when w is constant across the
    // three vertices -- true of every screen-space blit, and the same assumption the hardware's own
    // affine completion makes. A title that submits a genuinely sheared or perspective-divided
    // RectList would need the real hardware rule, which needs a title to demonstrate it.
    //
    // The EMISSION ORDER is the other half, and getting only the corner right is not enough. A strip's
    // two triangles share its middle edge, so that edge must be the rectangle's DIAGONAL -- the shared
    // corner leads and the synthesized one comes last. Emitting the supplied order when the shared
    // corner is not v0 gives two triangles that share a SIDE instead: they overlap on one half and
    // leave the wedge between the two diagonals uncovered, which on Darksiders II's blit is a black
    // triangle with its apex at the screen centre.
    //
    // The order is a ROTATION of the supplied triple -- supplied[(shared + slot) % 3] -- and it has to
    // be, because a strip's facing is its FIRST triangle's and the guest's cull state is passed
    // straight through (`render_state.cpp`, no RectList exemption). "Shared first, then the other two
    // in ascending order" also puts the diagonal in the middle, but for shared == v1 it is the odd
    // permutation (v1,v0,v2): it reverses the facing, so a culled draw would render NOTHING where it
    // previously rendered half. Rotations are even, so facing is preserved for all three. When the
    // shared corner is v0 the rotation is the identity, so nothing that renders correctly today
    // changes. Vulkan flips the winding test on a strip's odd triangle, so both halves agree.
    std::vector<uint32_t> build_interpolation_geometry(
        const FragmentInterpolationLayout& layout, bool capture_geometry_position,
        bool synthesize_rect = false, bool publish_primitive_id = false,
        uint32_t layer_input_location = FragmentInterpolationLayout::kUnusedLocation);

    std::vector<uint32_t> finish();
// clang-format on
#endif
#endif
