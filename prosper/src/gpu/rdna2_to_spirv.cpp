// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
#include "rdna2_to_spirv.hpp"
#include "diagnostic_selectors.hpp"
#include "rdna2_decode.hpp"
#include "shader_resources.hpp"
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

FragmentInterpolationLayout::FragmentInterpolationLayout() {
    for (auto& locations : parameter_locations) locations.fill(kUnusedLocation);
    system_locations.fill(kUnusedLocation);
}

namespace {

enum : uint32_t {
    Op_ExtInstImport=11, Op_ExtInst=12, Op_MemoryModel=14, Op_EntryPoint=15, Op_ExecutionMode=16,
    Op_Capability=17, Op_TypeVoid=19, Op_TypeBool=20, Op_TypeInt=21, Op_TypeFloat=22, Op_TypeVector=23,
    Op_TypeRuntimeArray=29, Op_TypeStruct=30, Op_TypePointer=32, Op_TypeFunction=33,
    Op_ConstantTrue=41, Op_ConstantFalse=42, Op_Constant=43, Op_Function=54, Op_FunctionEnd=56, Op_Variable=59,
    Op_LogicalOr=166, Op_LogicalAnd=167, Op_LogicalNot=168, Op_Select=169, Op_FOrdEqual=180, Op_FOrdNotEqual=182, Op_FOrdLessThan=184, Op_FOrdGreaterThan=186,
    Op_FOrdLessThanEqual=188, Op_FOrdGreaterThanEqual=190,
    Op_FUnordEqual=181, Op_FUnordNotEqual=183, Op_FUnordLessThan=185, Op_FUnordGreaterThan=187,   // NaN-inclusive ("n"-prefix) compares
    Op_FUnordLessThanEqual=189, Op_FUnordGreaterThanEqual=191,
    Op_Load=61, Op_Store=62, Op_AccessChain=65, Op_Decorate=71, Op_MemberDecorate=72,
    Op_ConvertFToU=109, Op_ConvertFToS=110, Op_ConvertSToF=111, Op_ConvertUToF=112, Op_Bitcast=124,
    Op_CompositeConstruct=80, Op_CompositeExtract=81, Op_IAdd=128, Op_FAdd=129, Op_ISub=130, Op_FSub=131, Op_IMul=132, Op_FMul=133,
    Op_UDiv=134, Op_UMod=137,
    Op_UMulExtended=151, Op_SMulExtended=152,   // {lo,hi} struct results (for mul_hi)
    Op_FDiv=136, Op_SMod=139, Op_IEqual=170, Op_INotEqual=171, Op_UGreaterThan=172, Op_SGreaterThan=173,
    Op_UGreaterThanEqual=174, Op_SGreaterThanEqual=175, Op_ULessThan=176, Op_SLessThan=177,
    Op_ULessThanEqual=178, Op_SLessThanEqual=179,
    Op_ShiftRightLogical=194, Op_ShiftRightArithmetic=195, Op_ShiftLeftLogical=196, Op_BitwiseOr=197,
    Op_BitwiseXor=198, Op_BitwiseAnd=199, Op_Not=200, Op_BitFieldSExtract=202, Op_BitFieldUExtract=203,
    Op_BitReverse=204, Op_BitCount=205, Op_UConvert=113,
    Op_TypeImage=25, Op_TypeSampledImage=27, Op_ImageTexelPointer=60, Op_SampledImage=86,
    Op_ImageSampleImplicitLod=87, Op_ImageSampleExplicitLod=88,
    Op_ImageSampleDrefImplicitLod=89, Op_ImageSampleDrefExplicitLod=90,
    Op_ImageFetch=95, Op_ImageGather=96, Op_Image=100,
    Op_ImageRead=98, Op_ImageWrite=99, Op_ImageQuerySizeLod=103, Op_ImageQuerySize=104,
    Op_ImageQueryLevels=106,
    Op_TypeArray=28, Op_ControlBarrier=224, Op_AtomicExchange=229, Op_AtomicIAdd=234,
    Op_AtomicISub=235,
    Op_AtomicSMin=236, Op_AtomicUMin=237, Op_AtomicSMax=238, Op_AtomicUMax=239,
    Op_AtomicAnd=240, Op_AtomicOr=241, Op_AtomicXor=242,
    Op_DPdx=207, Op_DPdy=208,   // screen-space derivatives (Fragment; plain Shader capability)
    Op_Phi=245, Op_LoopMerge=246,
    Op_SelectionMerge=247, Op_Label=248, Op_Branch=249, Op_BranchConditional=250, Op_Switch=251,
    Op_EmitVertex=218, Op_EndPrimitive=219,
    Op_Kill=252, Op_Return=253, Op_ModuleProcessed=330, Op_GroupNonUniformAny=335,
    Op_GroupNonUniformShuffle=345,
    Op_GroupNonUniformIAdd=349,
};
// GLSL.std.450 extended-instruction numbers.
enum : uint32_t { Glsl_FAbs=4, Glsl_RoundEven=2, Glsl_Trunc=3, Glsl_Floor=8, Glsl_Ceil=9, Glsl_Fract=10, Glsl_Sin=13, Glsl_Cos=14,
                  Glsl_Exp2=29, Glsl_Log2=30,
                  Glsl_Sqrt=31, Glsl_InverseSqrt=32, Glsl_FMin=37, Glsl_UMin=38, Glsl_SMin=39, Glsl_FMax=40,
                  Glsl_UMax=41, Glsl_SMax=42, Glsl_PackHalf2x16=58, Glsl_UnpackHalf2x16=62,
                  Glsl_NMin=79, Glsl_NMax=80 };   // NaN-aware min/max: one-NaN operand -> the other operand
enum : uint32_t {
    Cap_Shader=1, Cap_Geometry=2, Cap_Int64=11, Cap_GroupNonUniform=61, Cap_GroupNonUniformVote=62,
    Cap_GroupNonUniformArithmetic=63, Cap_GroupNonUniformShuffle=65, Cap_GroupNonUniformQuad=68,
    Cap_TransformFeedback=53,   // VK_EXT_transform_feedback (geometry-probe capture of gl_Position, gated)
    Addr_Logical=0, Mem_GLSL450=1, Exec_Vertex=0, Exec_Geometry=3, Exec_Fragment=4, Exec_GLCompute=5,
    EM_OriginUpperLeft=7, EM_DepthReplacing=12, EM_LocalSize=17, EM_Triangles=22,
    EM_OutputVertices=26, EM_OutputTriangleStrip=29, EM_Xfb=11,   // transform-feedback execution mode
    SC_Input=1, SC_UniformConstant=0, SC_Output=3, SC_Function=7, SC_PushConstant=9,
    SC_Image=11, SC_StorageBuffer=12, FC_None=0,
    Dim_1D=0, Dim_2D=1, Dim_3D=2,   // SPIR-V Dim. (2D coincides with the SQ_RSRC 2D dim value, but distinct.)
    Cap_Sampled1D=43, Cap_Image1D=44,   // Dim=1D needs Sampled1D; a 1D STORAGE image (read/write) also needs Image1D
    Cap_StorageImageMultisample=27,      // MS=1 storage image (read/write a multisampled image)
    Cap_ImageMSArray=48,                 // MS=1 AND Arrayed=1 image (2D_MSAA_ARRAY)
    Cap_StorageImageReadWithoutFormat=55, Cap_StorageImageWriteWithoutFormat=56,  // for Format=Unknown storage images
    Cap_ImageGatherExtended=25,          // dynamic (non-const) Offset image operand on OpImageGather
    Cap_ImageQuery=50,                   // OpImageQuerySizeLod (the sample_*_o texel->UV offset fold)
    ImgOp_Offset=0x10,                   // ImageOperands bit: dynamic texel offset (needs ImageGatherExtended)
    Img_Sampled_Storage=2,   // OpTypeImage "Sampled" operand: 2 = used WITHOUT a sampler (read/write storage image)
    ImgFmt_Unknown=0,        // OpTypeImage "Image Format": Unknown (runtime view format; needs the caps above)
    ImgFmt_R32ui=kSpirvImageFormatR32ui, // exact uint32 storage image format required by image atomics
    ImgOp_Bias=1, ImgOp_Lod=2, ImgOp_Grad=4, ImgOp_Sample=0x40,   // ImageOperands bits.
    SC_Workgroup=4, Scope_Device=1, Scope_Workgroup=2, Scope_Subgroup=3,
    MemSem_UniformAcqRel=0x48, MemSem_WGAcqRel=0x108,   // storage-buffer/LDS AcquireRelease semantics
    MemSem_ImageAcqRel=0x808,                            // AcquireRelease | ImageMemory
    Dec_Block=2, Dec_ArrayStride=6, Dec_BuiltIn=11, Dec_NoPerspective=13, Dec_Flat=14,
    Dec_Centroid=16, Dec_Sample=17, Dec_Location=30, Dec_Binding=33,
    Dec_DescriptorSet=34, Dec_Offset=35, Dec_XfbBuffer=36, Dec_XfbStride=37,
    BI_Position=0, BI_FragCoord=15, BI_FragDepth=22, BI_HelperInvocation=23,
    BI_WorkgroupId=26, BI_LocalInvocationId=27,
    BI_GlobalInvocationId=28, BI_SubgroupId=40, BI_SubgroupLocalInvocationId=41,
    BI_VertexIndex=42, BI_InstanceIndex=43,
    GroupOp_Reduce=0, GroupOp_ExclusiveScan=2,
};

uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct FlatAccessInfo {
    bool valid = false;
    bool store = false;
    bool sign_extend = false;
    uint32_t bits = 0;
    uint32_t components = 0;
    uint32_t bytes() const { return (bits / 8u) * components; }
};

FlatAccessInfo flat_access_info(uint32_t opcode) {
    switch (opcode) {
        case 0x08: return {true, false, false, 8, 1};   // *_load_ubyte
        case 0x09: return {true, false, true,  8, 1};   // *_load_sbyte
        case 0x0a: return {true, false, false, 16, 1};  // *_load_ushort
        case 0x0b: return {true, false, true,  16, 1};  // *_load_sshort
        case 0x0c: return {true, false, false, 32, 1};  // *_load_dword
        case 0x0d: return {true, false, false, 32, 2};
        case 0x0e: return {true, false, false, 32, 4};
        case 0x0f: return {true, false, false, 32, 3};
        case 0x18: return {true, true,  false, 8, 1};   // *_store_byte
        case 0x1a: return {true, true,  false, 16, 1};  // *_store_short
        case 0x1c: return {true, true,  false, 32, 1};  // *_store_dword
        case 0x1d: return {true, true,  false, 32, 2};
        case 0x1e: return {true, true,  false, 32, 4};
        case 0x1f: return {true, true,  false, 32, 3};
        default: return {};
    }
}

struct StaticScratchLayout {
    bool valid = true;
    bool used = false;
    int32_t min_byte = 0;
    uint32_t dwords = 0;
    int32_t saddr = -1;
};

StaticScratchLayout analyze_static_scratch(const std::vector<Rdna2Inst>& ins) {
    StaticScratchLayout out;
    int32_t min_byte = INT32_MAX, max_byte = INT32_MIN;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::FLAT) continue;
        const FlatAccessInfo access = flat_access_info(in.opcode);
        // The common compiler spill form is a static byte offset from one entry-provided scratch
        // base: `scratch_* ..., off, sN`. Dynamic VADDR, multiple/rewritten bases, D16-high forms,
        // atomics, and arbitrary flat/global pointers stay fail-closed.
        if (in.flat_segment != 1u || in.flat_lds || !access.valid ||
            in.src[0].kind != OperandKind::None ||
            in.src[1].kind != OperandKind::SGPR) {
            out.valid = false;
            return out;
        }
        if (out.saddr < 0) out.saddr = in.src[1].value;
        else if (out.saddr != in.src[1].value) {
            out.valid = false;
            return out;
        }
        const int32_t begin = static_cast<int32_t>(in.literal);
        const int32_t end = begin + static_cast<int32_t>(access.bytes());
        min_byte = std::min(min_byte, begin);
        max_byte = std::max(max_byte, end);
        out.used = true;
    }
    if (!out.used) return out;
    auto floor4 = [](int32_t value) {
        return value >= 0 ? (value / 4) * 4 : -(((-value + 3) / 4) * 4);
    };
    auto ceil4 = [](int32_t value) {
        return value >= 0 ? ((value + 3) / 4) * 4 : -((-value / 4) * 4);
    };
    const int32_t aligned_min = floor4(min_byte);
    const int32_t aligned_max = ceil4(max_byte);
    // The signed 12-bit instruction offset spans 4096 bytes, and a vector access at the positive
    // edge can extend by another 16 bytes. Keep a modest ceiling while accepting that full range.
    if (aligned_max <= aligned_min || aligned_max - aligned_min > 8192) {
        out.valid = false;
        return out;
    }
    out.min_byte = aligned_min;
    out.dwords = static_cast<uint32_t>((aligned_max - aligned_min) / 4);
    return out;
}

// The f16 bit pattern an inline float constant supplies in a 16-bit operand position (ISA Table 10
// lists per-width encodings: "0.5 ... half: 0x3800" etc.). Only 1/(2*pi) (code 248, 0x3118) differs
// from rounding the f32 value — the f32 table entry 0.15915494 would round to a different last bit
// than the documented operand, so 16-bit consumers must use these bits, not the f32 constant.
uint32_t inline_float_f16_bits(int code) {
    switch (code) {
        case 240: return 0x3800u; case 241: return 0xB800u;   // +/-0.5
        case 242: return 0x3C00u; case 243: return 0xBC00u;   // +/-1.0
        case 244: return 0x4000u; case 245: return 0xC000u;   // +/-2.0
        case 246: return 0x4400u; case 247: return 0xC400u;   // +/-4.0
        case 248: return 0x3118u;                             // 1/(2*pi)
        default:  return 0;
    }
}

// A compute-shader SPIR-V builder specialized for "load N floats -> compute over SSA floats ->
// store 1 float", with helpers the VALU translator drives.
struct SpirvCompute {
    std::vector<uint32_t> caps, extimp, mem, entry, exec, debug, deco, types, code;
    std::unordered_map<uint32_t, uint32_t> fconst_cache, uconst_cache;
    uint32_t next_id = 1;
    uint32_t stride = 1;
    // fixed ids (set in begin()):
    uint32_t t_void=0, t_fn=0, t_f32=0, t_u32=0, t_i32=0, t_v3u=0, t_bool=0, t_ptr_sb_f32=0;
    uint32_t v_gid=0, v_groupid=0, v_in=0, v_out=0, gidx=0, f_main=0, glsl=0, bconst_false=0;
    uint32_t globalid_comp[3] = {0, 0, 0}, groupid[3] = {0, 0, 0}, localid_comp[3] = {0, 0, 0};
    uint32_t invocation_guard_merge = 0;
    uint32_t v_push_constants = 0, t_ptr_push_u32 = 0;
    uint32_t linear_localid = 0, local_count = 64, wave_size = 64;
    uint32_t v_cbuf=0, v_cbuf1=0, t_ptr_sb_u32=0;   // scalar-memory constant buffers (bindings 2 and 3)
    uint32_t t_ptr_sb_struct_u=0;                    // shared runtime-u32 Block pointer type
    uint32_t t_ptr_img_u32=0;                       // OpImageTexelPointer result for R32_UINT atomics
    uint32_t guest_scratch=0, t_ptr_guest_scratch_u32=0;
    int32_t guest_scratch_min_byte=0, guest_scratch_saddr=-1;
    uint32_t guest_scratch_dwords=0;
    std::map<uint32_t, uint32_t> cbuf_var;          // binding -> storage-buffer var (N-buffer model; 2/3 map to v_cbuf/v_cbuf1)
    bool     is_fragment=0;                          // true in the fragment shell (gates VINTRP interp)
    bool     is_vertex=0;                            // true in every vertex shell
    bool     allow_b32_masks=0;                      // proven Wave32 or byte-exact graphics exception
    bool     ngg_one_lane=0;                         // exact GS_ALLOC_REQ wrapper: one guest lane/invocation
    bool     ngg_logical_lane=0;                     // proven wave64 no-GS producer uses flattened guest lane
    bool     ngg_private_lds=0;                      // exact captured wrapper whose LDS projection is known
    uint32_t ngg_vertex_index_read_pc = UINT32_MAX;   // NGG wave/LDS prologue handoff -> host VertexIndex
    uint32_t ngg_vertex_index_value = 0;
    bool     is_compute=0;                            // true in the compute shell (gates LDS / s_barrier)
    bool     uses_barrier=0;                          // guest or synthesized workgroup barrier emitted
    bool     declared_subgroup=0, declared_subgroup_vote=0, declared_subgroup_arithmetic=0;
    // When non-zero, the backend promises to create this compute pipeline with an exact required
    // subgroup size equal to the PS5 wave. Native votes/scans are then architecture-exact.
    uint32_t native_subgroup_size=0;
    uint32_t native_storage_format_support=0;
    bool packed_r11_storage=true;
    uint32_t compute_min_subgroup_size=0;             // non-semantic backend contract (4/16/32/64)
    uint32_t fragment_required_subgroup_size=0;       // exact guest-wave contract (32 or 64)
    uint32_t v_subgroupid=0, v_subgroup_localid=0, t_ptr_in_u32=0;
    uint32_t v_helper_invocation=0, t_ptr_in_bool=0;
    uint32_t v_internal_gds=0, t_ptr_gds_u32=0;
    // Descriptor set for this stage's resources. VS and PS share ONE Vulkan pipeline, so they must NOT
    // reuse binding numbers within one set (both stages number their cbuf/texture from binding 2 -> a
    // set-0 collision made the descriptor layout invalid, corrupting the VS's reads -> degenerate
    // geometry). Each stage owns its own set: VS=set 0, PS=set 1 (mirrors the PS5's per-stage resource
    // tables). Set in begin_fragment(); the host binds one descriptor set per stage.
    uint32_t desc_set=0;
    size_t function_var_insert = 0;                   // first-function-block OpVariable insertion point
    uint32_t exec_model=0;                           // deferred EntryPoint (emitted in finish() so lazily-
    std::vector<uint32_t> iface;                     // declared I/O varyings can join the interface list)
    // Geometry-probe (PROSPER_GEOM_PROBE): when set before begin_vertex(), decorate gl_Position for
    // transform-feedback capture. Inert to the shader's computation; only marks the output for readback.
    bool capture_position = false;
    // Shader I/O value tap (PROSPER_SHADER_TAP=pc): after the instruction at `tap_pc`, snapshot its
    // destination VGPR (and the next 3) as a vec4; the vertex position export is then redirected to it,
    // so the geometry-probe capture reports the intermediate value AT that PC (e.g. a MUBUF fetch result)
    // instead of gl_Position. Inert unless the env var is set. Values are bitcast-as-float in the output.
    uint32_t tap_pc = 0xFFFFFFFFu;
    uint32_t tap_vec = 0;   // 0 = no tap captured; else the vec4 SPIR-V id export_position emits
    void set_tap(uint32_t a, uint32_t bb, uint32_t c, uint32_t d) {
        uint32_t v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(a), bcf(bb), bcf(c), bcf(d)});
        tap_vec = v;
    }

    uint32_t id() { return next_id++; }
    static void put(std::vector<uint32_t>& s, uint32_t op, std::initializer_list<uint32_t> o) {
        s.push_back(((uint32_t)(o.size() + 1) << 16) | op); for (uint32_t x : o) s.push_back(x);
    }
    static void putv(std::vector<uint32_t>& s, uint32_t op, const std::vector<uint32_t>& o) {
        s.push_back(((uint32_t)(o.size() + 1) << 16) | op); s.insert(s.end(), o.begin(), o.end());
    }
    void pstr(std::vector<uint32_t>& v, const char* s) {
        size_t len = std::strlen(s);
        for (size_t i = 0; i <= len; i += 4) { uint32_t w = 0;
            for (size_t k = 0; k < 4; k++) { size_t j = i + k; if (j <= len) w |= (uint32_t)(uint8_t)s[j] << (8*k); }
            v.push_back(w); }
    }
    uint32_t fconst(float f) {
        uint32_t b = fbits(f); auto it = fconst_cache.find(b); if (it != fconst_cache.end()) return it->second;
        uint32_t c = id(); put(types, Op_Constant, {t_f32, c, b}); fconst_cache[b] = c; return c;
    }
    uint32_t uconst(uint32_t v) {
        auto it = uconst_cache.find(v); if (it != uconst_cache.end()) return it->second;
        uint32_t c = id(); put(types, Op_Constant, {t_u32, c, v}); uconst_cache[v] = c; return c;
    }
    // VGPRs are modeled as raw 32-bit VALUES (uint). Float ops bitcast their operands uint->float and
    // bitcast the result back to uint; integer ops operate on the bits directly. This matches the
    // hardware's untyped VGPRs and lets float and integer instructions share the same register file.
    uint32_t bcf(uint32_t u) { uint32_t r = id(); put(code, Op_Bitcast, {t_f32, r, u}); return r; }   // bits -> float
    uint32_t bcu(uint32_t f) { uint32_t r = id(); put(code, Op_Bitcast, {t_u32, r, f}); return r; }   // float -> bits
    uint32_t fconstf(float f) { uint32_t b = fbits(f); auto it = fconst_cache.find(b); if (it != fconst_cache.end()) return it->second;
        uint32_t c = id(); put(types, Op_Constant, {t_f32, c, b}); fconst_cache[b] = c; return c; }
    // Float binary op on bit-operands -> bit-result.
    uint32_t fbin(uint32_t op, uint32_t a, uint32_t b) { uint32_t rf = id(); put(code, op, {t_f32, rf, bcf(a), bcf(b)}); return bcu(rf); }
    // Integer binary op on bit-operands -> bit-result.
    uint32_t ibin(uint32_t op, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, op, {t_u32, r, a, b}); return r; }
    // Integer unary op on a bit-operand -> bit-result (e.g. OpNot).
    uint32_t iun(uint32_t op, uint32_t a) { uint32_t r = id(); put(code, op, {t_u32, r, a}); return r; }
    // GLSL.std.450 uint ext-instruction (UMin/UMax) on bit-operands -> bit-result (no bitcast).
    uint32_t uext2(uint32_t inst, uint32_t a, uint32_t b) { uint32_t r = id(); putv(code, Op_ExtInst, {t_u32, r, glsl, inst, a, b}); return r; }
    // GLSL.std.450 float ext-instructions on bit-operands -> bit-result.
    uint32_t fext1(uint32_t inst, uint32_t a) { uint32_t r = id(); putv(code, Op_ExtInst, {t_f32, r, glsl, inst, bcf(a)}); return bcu(r); }
    uint32_t fext2(uint32_t inst, uint32_t a, uint32_t b) { uint32_t r = id(); putv(code, Op_ExtInst, {t_f32, r, glsl, inst, bcf(a), bcf(b)}); return bcu(r); }
    uint32_t frcp(uint32_t a) { uint32_t rf = id(); put(code, Op_FDiv, {t_f32, rf, fconstf(1.0f), bcf(a)}); return bcu(rf); }
    // The NEG source modifier is a floating-point negation = raw sign-bit toggle (ISA sec 6.2:
    // "Negate input"). Lowering it as 0.0f - x lost the sign of zero (FSub(+0,+0) = +0 where
    // hardware yields -0, observable via v_rcp -> wrong-signed Inf, packed sign bits, min/max +-0
    // ordering) and did not deterministically flip NaN signs. A bit-XOR is exact for every input
    // (+-0, NaN, denormals) on every driver, unlike OpFNegate which lacks that guarantee without
    // SignedZeroInfNanPreserve. The unpacked-f16 paths run in the f32 domain, so this applies there too.
    uint32_t fneg(uint32_t a) { return ibin(Op_BitwiseXor, a, uconst(0x80000000u)); }
    // CLAMP-modifier saturate to [0,1] with the hardware's DX10_CLAMP NaN rule: a NaN result clamps
    // to 0 (graphics shaders run with MODE.DX10_CLAMP set). NMax(NaN, 0) = 0 then NMin(0, 1) = 0
    // gives that for free; the previous FMin/FMax chain left a NaN result driver-defined.
    uint32_t clamp01(uint32_t x) {
        return fext2(Glsl_NMin, fext2(Glsl_NMax, x, bcu(fconstf(0.0f))), bcu(fconstf(1.0f)));
    }
    uint32_t cvt_u2f(uint32_t u) { uint32_t rf = id(); put(code, Op_ConvertUToF, {t_f32, rf, u}); return bcu(rf); }   // uint -> float bits
    // v_cvt_u32_f32 SATURATES: NaN -> 0, negative -> 0, >= 2^32 -> 0xFFFFFFFF. A bare OpConvertFToU
    // has an undefined result out of range (#135), so clamp first. FMin/FMax are themselves
    // NaN-undefined, so NaN is routed to 0 explicitly. 4294967040.0f (0x4F7FFFFF) is the largest
    // float < 2^32 — the next float IS 2^32, so the clamp is exact for every in-range input;
    // inputs >= 2^32 (incl. +inf) select UINT_MAX.
    uint32_t cvt_f2u(uint32_t bits) {
        uint32_t f = bcf(bits);
        uint32_t nan = id();  put(code, Op_FUnordNotEqual, {t_bool, nan, f, f});   // true iff NaN
        uint32_t safe = id(); put(code, Op_Select, {t_f32, safe, nan, fconstf(0.0f), f});
        uint32_t lo = id();   putv(code, Op_ExtInst, {t_f32, lo, glsl, Glsl_FMax, safe, fconstf(0.0f)});
        uint32_t cl = id();   putv(code, Op_ExtInst, {t_f32, cl, glsl, Glsl_FMin, lo, fconstf(4294967040.0f)});
        uint32_t r = id();    put(code, Op_ConvertFToU, {t_u32, r, cl});
        uint32_t big = id();  put(code, Op_FOrdGreaterThanEqual, {t_bool, big, f, fconstf(4294967296.0f)});
        return sel(big, uconst(0xFFFFFFFFu), r);
    }
    // Float ordered compare on bit-operands -> bool (for VCC). select() picks bits by a bool condition.
    uint32_t fcmp(uint32_t cmpop, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, cmpop, {t_bool, r, bcf(a), bcf(b)}); return r; }
    uint32_t bfalse() { if (!bconst_false) { bconst_false = id(); put(types, Op_ConstantFalse, {t_bool, bconst_false}); } return bconst_false; }
    uint32_t sel(uint32_t cond, uint32_t tval, uint32_t fval) { uint32_t r = id(); put(code, Op_Select, {t_u32, r, cond, tval, fval}); return r; }
    uint32_t bsel(uint32_t cond, uint32_t tval, uint32_t fval) { uint32_t r = id(); put(code, Op_Select, {t_bool, r, cond, tval, fval}); return r; }  // bool-domain select (wave masks)

    // --- Structured control-flow primitives (for counted-loop reconstruction) ---
    uint32_t cur_block = 0;   // label id of the block currently being emitted (predecessor for OpPhi)
    void emit_label(uint32_t l)          { put(code, Op_Label, {l}); cur_block = l; }
    void emit_branch(uint32_t t)         { put(code, Op_Branch, {t}); }
    void emit_condbranch(uint32_t c, uint32_t t, uint32_t f) { put(code, Op_BranchConditional, {c, t, f}); }
    void emit_loopmerge(uint32_t m, uint32_t c)              { put(code, Op_LoopMerge, {m, c, 0}); }
    // OpPhi with two predecessors; the back-edge (value,label) is patched later via patch_phi(). Returns
    // the phi result id and sets `patch_off` to the code index of the placeholder back-edge value word.
    uint32_t emit_phi2(uint32_t type, uint32_t v0, uint32_t l0, size_t& patch_off) {
        uint32_t r = id();
        put(code, Op_Phi, {type, r, v0, l0, 0u, 0u});
        patch_off = code.size() - 2;   // the trailing {v1, l1} placeholders
        return r;
    }
    void patch_phi(size_t patch_off, uint32_t v1, uint32_t l1) { code[patch_off] = v1; code[patch_off + 1] = l1; }
    void emit_selmerge(uint32_t m) { put(code, Op_SelectionMerge, {m, 0}); }   // structured if (before condbranch)
    void emit_switch(uint32_t selector, uint32_t fallback,
                     const std::vector<std::pair<uint32_t, uint32_t>>& cases) {
        std::vector<uint32_t> operands{selector, fallback};
        for (const auto& c : cases) { operands.push_back(c.first); operands.push_back(c.second); }
        putv(code, Op_Switch, operands);
    }
    uint32_t function_var(uint32_t type, uint32_t& ptr_type) {
        if (!ptr_type) { ptr_type = id(); put(types, Op_TypePointer, {ptr_type, SC_Function, type}); }
        uint32_t var = id();
        std::vector<uint32_t> decl;
        put(decl, Op_Variable, {ptr_type, var, SC_Function});
        code.insert(code.begin() + static_cast<std::ptrdiff_t>(function_var_insert),
                    decl.begin(), decl.end());
        function_var_insert += decl.size();
        return var;
    }
    void declare_guest_scratch(const StaticScratchLayout& layout) {
        if (!layout.valid || !layout.used || !layout.dwords || guest_scratch) return;
        guest_scratch_min_byte = layout.min_byte;
        guest_scratch_saddr = layout.saddr;
        guest_scratch_dwords = layout.dwords;
        const uint32_t array = id();
        put(types, Op_TypeArray, {array, t_u32, uconst(layout.dwords)});
        uint32_t ptr_array = 0;
        guest_scratch = function_var(array, ptr_array);
        t_ptr_guest_scratch_u32 = id();
        put(types, Op_TypePointer, {t_ptr_guest_scratch_u32, SC_Function, t_u32});
    }
    uint32_t guest_scratch_load_word(uint32_t index) {
        uint32_t pointer = id();
        putv(code, Op_AccessChain,
             {t_ptr_guest_scratch_u32, pointer, guest_scratch, uconst(index)});
        uint32_t value = id();
        put(code, Op_Load, {t_u32, value, pointer});
        return value;
    }
    void guest_scratch_store_word(uint32_t index, uint32_t value,
                                  bool predicated, uint32_t pred) {
        auto store = [&]() {
            uint32_t pointer = id();
            putv(code, Op_AccessChain,
                 {t_ptr_guest_scratch_u32, pointer, guest_scratch, uconst(index)});
            put(code, Op_Store, {pointer, value});
        };
        if (!predicated) { store(); return; }
        const uint32_t then_label = id(), merge_label = id();
        emit_selmerge(merge_label);
        emit_condbranch(pred, then_label, merge_label);
        emit_label(then_label);
        store();
        emit_branch(merge_label);
        emit_label(merge_label);
    }
    uint32_t guest_scratch_load_bits(int32_t byte_offset, uint32_t bits, bool sign_extend) {
        const uint32_t relative = static_cast<uint32_t>(byte_offset - guest_scratch_min_byte);
        const uint32_t index = relative / 4u, shift = (relative & 3u) * 8u;
        uint32_t value = guest_scratch_load_word(index);
        if (shift + bits <= 32u)
            return sign_extend ? bfe_s(value, uconst(shift), uconst(bits))
                               : bfe_u(value, uconst(shift), uconst(bits));
        const uint32_t low_bits = 32u - shift, high_bits = bits - low_bits;
        const uint32_t low = ibin(Op_ShiftRightLogical, value, uconst(shift));
        const uint32_t high_word = guest_scratch_load_word(index + 1u);
        const uint32_t high = bfe_u(high_word, uconst(0), uconst(high_bits));
        const uint32_t joined = ibin(Op_BitwiseOr, low,
                                     ibin(Op_ShiftLeftLogical, high, uconst(low_bits)));
        return sign_extend ? bfe_s(joined, uconst(0), uconst(bits)) : joined;
    }
    void guest_scratch_store_bits(int32_t byte_offset, uint32_t bits, uint32_t value,
                                  bool predicated, uint32_t pred) {
        const uint32_t relative = static_cast<uint32_t>(byte_offset - guest_scratch_min_byte);
        const uint32_t index = relative / 4u, shift = (relative & 3u) * 8u;
        auto bit_mask = [](uint32_t width) {
            return width == 32u ? 0xffffffffu : ((1u << width) - 1u);
        };
        auto replace = [&](uint32_t word_index, uint32_t dst_shift,
                           uint32_t width, uint32_t source_shift) {
            const uint32_t mask = bit_mask(width) << dst_shift;
            const uint32_t old = guest_scratch_load_word(word_index);
            uint32_t field = source_shift
                ? ibin(Op_ShiftRightLogical, value, uconst(source_shift)) : value;
            if (dst_shift) field = ibin(Op_ShiftLeftLogical, field, uconst(dst_shift));
            field = ibin(Op_BitwiseAnd, field, uconst(mask));
            const uint32_t retained = ibin(Op_BitwiseAnd, old, uconst(~mask));
            guest_scratch_store_word(word_index, ibin(Op_BitwiseOr, retained, field),
                                     predicated, pred);
        };
        if (shift + bits <= 32u) {
            if (bits == 32u && shift == 0u)
                guest_scratch_store_word(index, value, predicated, pred);
            else
                replace(index, shift, bits, 0);
            return;
        }
        const uint32_t low_bits = 32u - shift;
        replace(index, shift, low_bits, 0);
        replace(index + 1u, 0, bits - low_bits, low_bits);
    }
    uint32_t load_function(uint32_t type, uint32_t var) {
        uint32_t value = id(); put(code, Op_Load, {type, value, var}); return value;
    }
    void store_function(uint32_t var, uint32_t value) { put(code, Op_Store, {var, value}); }
    uint32_t logical_not(uint32_t value) {
        uint32_t result = id(); put(code, Op_LogicalNot, {t_bool, result, value}); return result;
    }
    uint32_t subgroup_local_id() {
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!v_subgroup_localid) {
            if (!t_ptr_in_u32) {
                t_ptr_in_u32 = id();
                put(types, Op_TypePointer, {t_ptr_in_u32, SC_Input, t_u32});
            }
            v_subgroup_localid = id();
            put(types, Op_Variable, {t_ptr_in_u32, v_subgroup_localid, SC_Input});
            put(deco, Op_Decorate,
                {v_subgroup_localid, Dec_BuiltIn, BI_SubgroupLocalInvocationId});
            put(deco, Op_Decorate, {v_subgroup_localid, Dec_Flat});
            iface.push_back(v_subgroup_localid);
        }
        // Fragment lane ids model the guest RDNA wave, not the implementation's default subgroup.
        // Record that exact-width contract as non-semantic module metadata in finish().
        if (is_fragment) fragment_required_subgroup_size = wave_size;
        uint32_t lane = id();
        put(code, Op_Load, {t_u32, lane, v_subgroup_localid});
        return lane;
    }
    // Exact scalar wave vote for fragment control flow and mask reductions. A guest scalar branch
    // observes one 64-bit EXEC/VCC value for the complete hardware wave; branching directly on this
    // invocation's bool would instead create 64 independent pixel branches. Mark the module with the
    // same arithmetic capability used by fragment MBCNT so the backend enforces subgroup size 64,
    // and declare Vote for OpGroupNonUniformAny itself.
    uint32_t fragment_wave_any(uint32_t active_bit) {
        if (!is_fragment) return 0;
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_vote) {
            put(caps, Op_Capability, {Cap_GroupNonUniformVote});
            declared_subgroup_vote = true;
        }
        fragment_required_subgroup_size = wave_size;
        uint32_t result = id();
        put(code, Op_GroupNonUniformAny,
            {t_bool, result, uconst(Scope_Subgroup), active_bit});
        return result;
    }
    uint32_t subgroup_id() {
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!v_subgroupid) {
            if (!t_ptr_in_u32) {
                t_ptr_in_u32 = id();
                put(types, Op_TypePointer, {t_ptr_in_u32, SC_Input, t_u32});
            }
            v_subgroupid = id();
            put(types, Op_Variable, {t_ptr_in_u32, v_subgroupid, SC_Input});
            put(deco, Op_Decorate, {v_subgroupid, Dec_BuiltIn, BI_SubgroupId});
            put(deco, Op_Decorate, {v_subgroupid, Dec_Flat});
            iface.push_back(v_subgroupid);
        }
        uint32_t subgroup = id();
        put(code, Op_Load, {t_u32, subgroup, v_subgroupid});
        return subgroup;
    }
    uint32_t fragment_mbcnt(uint32_t mask_bit, uint32_t acc_bits, bool lo) {
        if (!is_fragment) return 0;
        const uint32_t lane = subgroup_local_id();
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t in_half = lo
            ? ucmp(Op_ULessThan, lane, uconst(32))
            : ucmp(Op_UGreaterThanEqual, lane, uconst(32));
        // Helper invocations execute subgroup operations under WQM but are not guest lanes and
        // cannot own GDS allocation slots. Excluding them here keeps MBCNT's prefix exactly aligned
        // with fragment GDS append/consume's non-helper population.
        const uint32_t guest_lane = land(mask_bit, logical_not(helper_invocation()));
        const uint32_t selected = sel(land(guest_lane, in_half), uconst(1), uconst(0));
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, selected});
        return ibin(Op_IAdd, acc_bits, prefix);
    }
    uint32_t native_wave_any(uint32_t value) {
        if (!native_subgroup_size) return 0;
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_vote) {
            put(caps, Op_Capability, {Cap_GroupNonUniformVote});
            declared_subgroup_vote = true;
        }
        uint32_t result = id();
        put(code, Op_GroupNonUniformAny,
            {t_bool, result, uconst(Scope_Subgroup), value});
        return result;
    }
    uint32_t native_compute_mbcnt(uint32_t mask_bit, uint32_t acc_bits, uint32_t lo) {
        if (!native_subgroup_size) return 0;
        const uint32_t lane = subgroup_local_id();
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t in_half = bsel(
            lo, ucmp(Op_ULessThan, lane, uconst(32)),
            ucmp(Op_UGreaterThanEqual, lane, uconst(32)));
        const uint32_t selected = sel(land(mask_bit, in_half), uconst(1), uconst(0));
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, selected});
        return ibin(Op_IAdd, acc_bits, prefix);
    }
    uint32_t helper_invocation() {
        if (!v_helper_invocation) {
            t_ptr_in_bool = id();
            put(types, Op_TypePointer, {t_ptr_in_bool, SC_Input, t_bool});
            v_helper_invocation = id();
            put(types, Op_Variable, {t_ptr_in_bool, v_helper_invocation, SC_Input});
            put(deco, Op_Decorate,
                {v_helper_invocation, Dec_BuiltIn, BI_HelperInvocation});
            iface.push_back(v_helper_invocation);
        }
        uint32_t result = id();
        put(code, Op_Load, {t_bool, result, v_helper_invocation});
        return result;
    }
    void declare_internal_gds(uint32_t set = 1, uint32_t binding = 0) {
        if (v_internal_gds) return;
        uint32_t runtime_array = id(), block = id(), block_ptr = id();
        put(deco, Op_Decorate, {runtime_array, Dec_ArrayStride, 4});
        put(deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
        put(deco, Op_Decorate, {block, Dec_Block});
        put(types, Op_TypeRuntimeArray, {runtime_array, t_u32});
        put(types, Op_TypeStruct, {block, runtime_array});
        put(types, Op_TypePointer, {block_ptr, SC_StorageBuffer, block});
        t_ptr_gds_u32 = id();
        put(types, Op_TypePointer, {t_ptr_gds_u32, SC_StorageBuffer, t_u32});
        v_internal_gds = id();
        put(types, Op_Variable, {block_ptr, v_internal_gds, SC_StorageBuffer});
        put(deco, Op_Decorate, {v_internal_gds, Dec_DescriptorSet, set});
        put(deco, Op_Decorate, {v_internal_gds, Dec_Binding, binding});
    }
    void compute_gds_store(uint32_t index, uint32_t value, bool predicated, uint32_t pred) {
        declare_internal_gds(0, kComputeInternalGdsBinding);
        auto emit = [&]() {
            uint32_t pointer = id();
            putv(code, Op_AccessChain,
                 {t_ptr_gds_u32, pointer, v_internal_gds, uconst(0), index});
            put(code, Op_Store, {pointer, value});
        };
        if (!predicated) { emit(); return; }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        emit();
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }
    uint32_t compute_gds_atomic_rtn(uint32_t op, uint32_t index, uint32_t value) {
        declare_internal_gds(0, kComputeInternalGdsBinding);
        uint32_t pointer = id();
        putv(code, Op_AccessChain,
             {t_ptr_gds_u32, pointer, v_internal_gds, uconst(0), index});
        uint32_t result = id();
        put(code, op,
            {t_u32, result, pointer, uconst(Scope_Device),
             uconst(MemSem_UniformAcqRel), value});
        return result;
    }
    // Native-subgroup GDS append/consume is one device-global atomic per hardware wave. Fragment
    // helper invocations participate in subgroup operations but cannot consume guest counter slots;
    // compute has no helper lanes. Callers use this only when one host subgroup is one guest wave.
    uint32_t native_gds_append(uint32_t index, uint32_t exec_bit, bool consume) {
        declare_internal_gds(is_compute ? 0 : 1,
                             is_compute ? kComputeInternalGdsBinding : 0);
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t active_bit = is_fragment
            ? land(exec_bit, logical_not(helper_invocation())) : exec_bit;
        const uint32_t contribution = sel(active_bit, uconst(1), uconst(0));
        uint32_t count = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, count, uconst(Scope_Subgroup), GroupOp_Reduce, contribution});
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, contribution});
        const uint32_t elected = land(active_bit, ucmp(Op_IEqual, prefix, uconst(0)));
        const uint32_t entry = cur_block, leader = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {elected, leader, merge});
        put(code, Op_Label, {leader}); cur_block = leader;
        uint32_t pointer = id();
        putv(code, Op_AccessChain,
             {t_ptr_gds_u32, pointer, v_internal_gds, uconst(0), index});
        uint32_t leader_old = id();
        put(code, consume ? Op_AtomicISub : Op_AtomicIAdd,
            {t_u32, leader_old, pointer, uconst(Scope_Device),
             uconst(MemSem_UniformAcqRel), count});
        const uint32_t leader_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        const uint32_t local_old = emit_phi_2way(
            t_u32, leader_old, leader_end, uconst(0), entry);
        uint32_t old = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, old, uconst(Scope_Subgroup), GroupOp_Reduce, local_old});
        return old;
    }
    bool declared_subgroup_shuffle = false;
    void mark_subgroup_min16() {
        // DPP rows contain 16 contiguous lanes. Keep width metadata independent of SPIR-V
        // capabilities: declaring an unused operation merely as a marker over-requires the host and
        // confuses shaders that genuinely use that capability for unrelated work.
        if (is_fragment) fragment_required_subgroup_size = wave_size;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 16u);
    }
    void mark_subgroup_min32() {
        // PERMLANEX16 crosses a pair of 16-lane rows.
        if (is_fragment) fragment_required_subgroup_size = wave_size;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 32u);
    }
    void mark_subgroup_min64() {
        // V_READLANE_B32 may address every lane of a wave64.
        if (is_fragment) fragment_required_subgroup_size = wave_size;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 64u);
    }
    uint32_t subgroup_shuffle(uint32_t value, uint32_t lane) {
        // Every supported native shuffle at least addresses an architectural quad. Wider row/wave
        // operations raise this contract before calling the common helper.
        if (is_fragment) fragment_required_subgroup_size = wave_size;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 4u);
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_shuffle) {
            put(caps, Op_Capability, {Cap_GroupNonUniformShuffle});
            declared_subgroup_shuffle = true;
        }
        uint32_t result = id();
        put(code, Op_GroupNonUniformShuffle,
            {t_u32, result, uconst(Scope_Subgroup), value, lane});
        return result;
    }
    uint32_t subgroup_row_shr(uint32_t value, uint32_t active, uint32_t amount,
                              uint32_t* valid_lane = nullptr) {
        mark_subgroup_min16();
        const uint32_t lane = subgroup_local_id();
        const uint32_t row_lane = ibin(Op_BitwiseAnd, lane, uconst(15));
        const uint32_t in_bounds = ucmp(Op_UGreaterThanEqual, row_lane, uconst(amount));
        // Keep the shuffle index valid even for row-leading lanes. FI=0 also makes an EXEC-inactive
        // source invalid. The caller uses `valid` to preserve VDST for BOUND_CTRL=0; returning VALUE
        // here is only a safe placeholder for the disabled lane, not its architectural result.
        const uint32_t source_lane = sel(in_bounds,
            ibin(Op_ISub, lane, uconst(amount)), lane);
        const uint32_t shifted = subgroup_shuffle(value, source_lane);
        const uint32_t source_active = subgroup_shuffle(
            sel(active, uconst(1), uconst(0)), source_lane);
        const uint32_t valid = land(in_bounds,
            ucmp(Op_INotEqual, source_active, uconst(0)));
        if (valid_lane) *valid_lane = valid;
        return sel(valid, shifted, value);
    }
    uint32_t subgroup_quad_permute(uint32_t value, uint32_t ctrl) {
        const uint32_t lane = subgroup_local_id();
        const uint32_t quad_lane = ibin(Op_BitwiseAnd, lane, uconst(3));
        uint32_t selected = uconst(ctrl & 3u);
        for (uint32_t output_lane = 1; output_lane < 4; ++output_lane)
            selected = sel(ucmp(Op_IEqual, quad_lane, uconst(output_lane)),
                           uconst((ctrl >> (2u * output_lane)) & 3u), selected);
        const uint32_t source_lane = ibin(
            Op_BitwiseOr, ibin(Op_BitwiseAnd, lane, uconst(~3u)), selected);
        return subgroup_shuffle(value, source_lane);
    }
    bool ds_swizzle_source_lane(uint32_t offset, uint32_t* source_lane) {
        // RDNA2 ISA 12.13.1: the two basic DS_SWIZZLE modes precede rotate/FFT. The quad form
        // (10xx...) carries four two-bit selectors; the group32 form carries AND/OR/XOR masks.
        // Rotate and FFT remain fail-closed until their distinct mappings are implemented.
        if (!source_lane || offset >= 0xc000u) return false;
        const uint32_t lane = subgroup_local_id();
        if (offset & 0x8000u) {
            const uint32_t quad_lane = ibin(Op_BitwiseAnd, lane, uconst(3));
            uint32_t selected = uconst(offset & 3u);
            for (uint32_t output_lane = 1; output_lane < 4; ++output_lane)
                selected = sel(ucmp(Op_IEqual, quad_lane, uconst(output_lane)),
                               uconst((offset >> (2u * output_lane)) & 3u), selected);
            *source_lane = ibin(
                Op_BitwiseOr, ibin(Op_BitwiseAnd, lane, uconst(~3u)), selected);
            return true;
        }
        mark_subgroup_min32();
        const uint32_t and_mask = uconst(offset & 0x1fu);
        const uint32_t or_mask = uconst((offset >> 5) & 0x1fu);
        const uint32_t xor_mask = uconst((offset >> 10) & 0x1fu);
        const uint32_t lane_in_group = ibin(Op_BitwiseAnd, lane, uconst(0x1fu));
        const uint32_t selected = ibin(
            Op_BitwiseXor,
            ibin(Op_BitwiseOr, ibin(Op_BitwiseAnd, lane_in_group, and_mask), or_mask),
            xor_mask);
        *source_lane = ibin(
            Op_BitwiseOr, ibin(Op_BitwiseAnd, lane, uconst(~0x1fu)), selected);
        return true;
    }
    uint32_t subgroup_permlane16(uint32_t value, uint32_t selectors_lo,
                                uint32_t selectors_hi, bool across_rows,
                                uint32_t* source_lane_out = nullptr) {
        const uint32_t lane = subgroup_local_id();
        const uint32_t row_lane = ibin(Op_BitwiseAnd, lane, uconst(15));
        const uint32_t selector_word = sel(
            ucmp(Op_ULessThan, row_lane, uconst(8)), selectors_lo, selectors_hi);
        const uint32_t selector_index = ibin(Op_BitwiseAnd, row_lane, uconst(7));
        const uint32_t selector_shift = ibin(Op_ShiftLeftLogical, selector_index, uconst(2));
        const uint32_t selected = ibin(
            Op_BitwiseAnd,
            ibin(Op_ShiftRightLogical, selector_word, selector_shift), uconst(15));
        uint32_t row_base = ibin(Op_BitwiseAnd, lane, uconst(~15u));
        if (across_rows) row_base = ibin(Op_BitwiseXor, row_base, uconst(16));
        const uint32_t source_lane = ibin(Op_BitwiseOr, row_base, selected);
        if (source_lane_out) *source_lane_out = source_lane;
        return subgroup_shuffle(value, source_lane);
    }
    // Fragment discard: OpKill any lane where `alive` is false, survivors fall through to `mergeL`. Used to
    // lower an EXP done under a narrowed EXEC (an alpha-test / WQM discard — the surviving lanes are the
    // ones that pass the kill test) to a real per-invocation discard. OpKill is a block terminator, so the
    // kill block has no back-edge to the merge; the merge's sole predecessor is the conditional branch.
    void discard_unless(uint32_t alive) {
        uint32_t killL = id(), mergeL = id();
        put(code, Op_SelectionMerge, {mergeL, 0});
        put(code, Op_BranchConditional, {alive, mergeL, killL});
        emit_label(killL); put(code, Op_Kill, {});
        emit_label(mergeL);
    }
    // OpPhi with two fully-known predecessors (both edges' values available now) — for an if/merge join.
    uint32_t emit_phi_2way(uint32_t type, uint32_t va, uint32_t la, uint32_t vb, uint32_t lb) {
        uint32_t r = id(); put(code, Op_Phi, {type, r, va, la, vb, lb}); return r;
    }
    // Signed-int helpers: bits are bitcast to i32, the op runs, and the i32 result is bitcast to bits.
    uint32_t bcs(uint32_t u) { uint32_t r = id(); put(code, Op_Bitcast, {t_i32, r, u}); return r; }   // bits -> i32
    uint32_t i2u(uint32_t i) { uint32_t r = id(); put(code, Op_Bitcast, {t_u32, r, i}); return r; }   // i32 -> bits
    // High 32 bits of a 32x32 multiply (v_/s_mul_hi_*), via the {lo,hi} struct of OpU/SMulExtended.
    uint32_t t_u32pair = 0, t_i32pair = 0;
    uint32_t umul_hi(uint32_t a, uint32_t b_) {
        if (!t_u32pair) { t_u32pair = id(); put(types, Op_TypeStruct, {t_u32pair, t_u32, t_u32}); }
        uint32_t r = id(); put(code, Op_UMulExtended, {t_u32pair, r, a, b_});
        uint32_t hi = id(); put(code, Op_CompositeExtract, {t_u32, hi, r, 1}); return hi;
    }
    uint32_t smul_hi(uint32_t a, uint32_t b_) {
        if (!t_i32pair) { t_i32pair = id(); put(types, Op_TypeStruct, {t_i32pair, t_i32, t_i32}); }
        uint32_t r = id(); put(code, Op_SMulExtended, {t_i32pair, r, bcs(a), bcs(b_)});
        uint32_t hi = id(); put(code, Op_CompositeExtract, {t_i32, hi, r, 1}); return i2u(hi);
    }
    uint32_t sbin(uint32_t op, uint32_t a, uint32_t b) { uint32_t ri = id(); put(code, op, {t_i32, ri, bcs(a), bcs(b)}); return i2u(ri); }
    uint32_t sext2(uint32_t inst, uint32_t a, uint32_t b) { uint32_t ri = id(); putv(code, Op_ExtInst, {t_i32, ri, glsl, inst, bcs(a), bcs(b)}); return i2u(ri); }
    // v_cvt_i32_f32 SATURATES: NaN -> 0, clamp to [INT_MIN, INT_MAX] (#135). Do not emit
    // OpConvertFToS: SPIR-V leaves invalid conversions undefined, and Metal also produces a
    // backend-specific result near the signed limits even after a float clamp (#686). Convert a
    // bounded absolute magnitude through the working unsigned path, then restore two's-complement
    // sign. 2^31 is representable by u32, so every conversion is defined on all backends.
    uint32_t cvt_f2i(uint32_t bits) {
        uint32_t f = bcf(bits);
        uint32_t nan = id();  put(code, Op_FUnordNotEqual, {t_bool, nan, f, f});   // true iff NaN
        uint32_t safe = id(); put(code, Op_Select, {t_f32, safe, nan, fconstf(0.0f), f});
        uint32_t magnitude = id();
        putv(code, Op_ExtInst, {t_f32, magnitude, glsl, Glsl_FAbs, safe});
        uint32_t bounded = id();
        putv(code, Op_ExtInst,
             {t_f32, bounded, glsl, Glsl_FMin, magnitude, fconstf(2147483648.0f)});
        uint32_t unsigned_magnitude = id();
        put(code, Op_ConvertFToU, {t_u32, unsigned_magnitude, bounded});
        uint32_t negative = id();
        put(code, Op_FOrdLessThan, {t_bool, negative, f, fconstf(0.0f)});
        uint32_t negated = ibin(Op_ISub, uconst(0), unsigned_magnitude);
        uint32_t signed_bits = sel(negative, negated, unsigned_magnitude);
        uint32_t big = id();  put(code, Op_FOrdGreaterThanEqual, {t_bool, big, f, fconstf(2147483648.0f)});
        return sel(big, uconst(0x7FFFFFFFu), signed_bits);
    }
    uint32_t cvt_i2f(uint32_t bits) { uint32_t rf = id(); put(code, Op_ConvertSToF, {t_f32, rf, bcs(bits)}); return bcu(rf); }
    // Integer compares -> bool. scmp treats operands as signed, ucmp as unsigned.
    uint32_t scmp(uint32_t op, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, op, {t_bool, r, bcs(a), bcs(b)}); return r; }
    uint32_t ucmp(uint32_t op, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, op, {t_bool, r, a, b}); return r; }
    // Bitfield extract (base, offset, count) -> bits. Unsigned and signed variants.
    // Clamp so Offset+Count <= 32 (#455). SPIR-V OpBitField*Extract is UNDEFINED when off+cnt exceeds
    // the 32-bit operand width, but s_bfe's 7-bit width field (0..127) and v_bfe's off+cnt (up to 62)
    // can exceed it. RDNA2 reads bits past the source MSB as 0 (unsigned) / sign (signed), i.e. the
    // effective count is min(cnt, 32-off) — reproduce that instead of emitting UB (the constant-offset
    // callers already satisfy off+cnt<=32, so the umin/isub fold to no-ops there).
    uint32_t bfe_clamp_cnt(uint32_t offc, uint32_t cnt) { return uext2(Glsl_UMin, cnt, ibin(Op_ISub, uconst(32), offc)); }
    uint32_t bfe_u(uint32_t base, uint32_t off, uint32_t cnt) { uint32_t offc = uext2(Glsl_UMin, off, uconst(32)); uint32_t r = id(); put(code, Op_BitFieldUExtract, {t_u32, r, base, offc, bfe_clamp_cnt(offc, cnt)}); return r; }
    uint32_t bfe_s(uint32_t base, uint32_t off, uint32_t cnt) { uint32_t offc = uext2(Glsl_UMin, off, uconst(32)); uint32_t ri = id(); put(code, Op_BitFieldSExtract, {t_i32, ri, bcs(base), offc, bfe_clamp_cnt(offc, cnt)}); return i2u(ri); }
    // Lazily declared 64-bit uint (+ Int64 capability), for s_bfe_u64 etc. that operate on SGPR pairs.
    uint32_t t_u64_cache = 0; bool declared_int64 = false;
    uint32_t t_u64() { if (!t_u64_cache) { if (!declared_int64) { put(caps, Op_Capability, {Cap_Int64}); declared_int64 = true; }
                       t_u64_cache = id(); put(types, Op_TypeInt, {t_u64_cache, 64, 0}); } return t_u64_cache; }
    // A 64-bit uint constant needs two value words. Shift amounts MUST be u64 here: a u32 shift operand on
    // a u64 base is mishandled by some drivers (llvmpipe), silently dropping the high half.
    uint32_t uconst64(uint64_t v) {
        // For values that fit in 32 bits (all current uses are shift amounts of 32), materialize the
        // u64 by widening a 32-bit constant in the body rather than emitting a 2-word 64-bit OpConstant
        // literal. Two reasons, both satisfied by this form: (1) MoltenVK's bundled SPIRV-Cross
        // mis-parses 64-bit OpConstant literals (reads the high value word as a zero Id -> "Cannot
        // resolve expression type", failing the whole shader) — a u32->u64 OpUConvert has no 2-word
        // literal; (2) the result is still u64-typed, so llvmpipe's requirement that a u64 shift amount
        // be u64 (a u32 shift operand drops the high half there) is preserved. Semantically identical on
        // every driver. A genuine >32-bit constant still needs the 2-word form (not currently emitted).
        if (v <= 0xffffffffull) { uint32_t r = id(); put(code, Op_UConvert, {t_u64(), r, uconst((uint32_t)v)}); return r; }
        uint32_t c = id(); put(types, Op_Constant, {t_u64(), c, (uint32_t)v, (uint32_t)(v >> 32)}); return c;
    }
    uint32_t u64_from_lohi(uint32_t lo, uint32_t hi) {   // (u64)hi<<32 | (u64)lo  — combine an SGPR pair
        uint32_t l = id(); put(code, Op_UConvert, {t_u64(), l, lo});
        uint32_t h = id(); put(code, Op_UConvert, {t_u64(), h, hi});
        uint32_t hs = id(); put(code, Op_ShiftLeftLogical, {t_u64(), hs, h, uconst64(32)});
        uint32_t r = id(); put(code, Op_BitwiseOr, {t_u64(), r, l, hs}); return r;
    }
    uint32_t u64_shift(uint32_t op, uint32_t value, uint32_t amount) {
        // Keep the shift operand u64-typed for the same cross-driver reason as uconst64 above.
        uint32_t shift = id(); put(code, Op_UConvert, {t_u64(), shift, amount});
        uint32_t result = id(); put(code, op, {t_u64(), result, value, shift}); return result;
    }
    uint32_t bfe_u64(uint32_t base64, uint32_t off, uint32_t cnt) {   // 64-bit unsigned bitfield extract
        // res = (base << (64-off-cnt)) >> (64-cnt), all logical u64 (portable — OpBitFieldUExtract on a
        // 64-bit base isn't reliably supported, e.g. llvmpipe returns 0). The 7-bit width field
        // legally encodes 0 and values past the register end; SPIR-V shifts >= 64 are undefined
        // VALUES, so clamp like the 32-bit helper (#455): effective cnt = min(cnt, 64-off) (bits
        // past bit 63 read as 0), and a zero width selects the architectural result 0 explicitly.
        // Requires off <= 63 (both callers mask offset to [5:0]).
        uint32_t cnt_c = uext2(Glsl_UMin, cnt, ibin(Op_ISub, uconst(64), off));
        uint32_t total = ibin(Op_IAdd, off, cnt_c);
        uint32_t lsh32 = ibin(Op_ISub, uconst(64), total);
        uint32_t rsh32 = ibin(Op_ISub, uconst(64), cnt_c);
        uint32_t lsh = id(); put(code, Op_UConvert, {t_u64(), lsh, lsh32});
        uint32_t rsh = id(); put(code, Op_UConvert, {t_u64(), rsh, rsh32});
        uint32_t sl  = id(); put(code, Op_ShiftLeftLogical,  {t_u64(), sl, base64, lsh});
        uint32_t r   = id(); put(code, Op_ShiftRightLogical, {t_u64(), r, sl, rsh});
        uint32_t nz  = ucmp(Op_INotEqual, cnt, uconst(0));
        uint32_t rs  = id(); put(code, Op_Select, {t_u64(), rs, nz, r, uconst64(0)}); return rs; }
    uint32_t u64_lo(uint32_t v64) { uint32_t r = id(); put(code, Op_UConvert, {t_u32, r, v64}); return r; }  // truncate low 32
    uint32_t u64_hi(uint32_t v64) { uint32_t s = id(); put(code, Op_ShiftRightLogical, {t_u64(), s, v64, uconst64(32)});
        uint32_t r = id(); put(code, Op_UConvert, {t_u32, r, s}); return r; }
    uint32_t u64_bit(uint32_t v64, uint32_t bit) {
        uint32_t shift = id(); put(code, Op_UConvert, {t_u64(), shift, bit});
        uint32_t shifted = id(); put(code, Op_ShiftRightLogical, {t_u64(), shifted, v64, shift});
        return ucmp(Op_INotEqual,
                    ibin(Op_BitwiseAnd, u64_lo(shifted), uconst(1)), uconst(0));
    }
    // Lazily declared 2-float vector type (types are emitted as a block before code, so on-demand is safe).
    uint32_t t_v2f_cache = 0;
    uint32_t t_v2f() { if (!t_v2f_cache) { t_v2f_cache = id(); put(types, Op_TypeVector, {t_v2f_cache, t_f32, 2}); } return t_v2f_cache; }
    // pack two f32 (raw VGPR bits) into a dword of two f16 halves (src0->low, src1->high). Uses SPIR-V
    // PackHalf2x16, which is round-to-nearest-even — correct for the RTE store path (pack_half_lo).
    uint32_t pack_half2x16(uint32_t a, uint32_t b) {
        uint32_t vec = id(); put(code, Op_CompositeConstruct, {t_v2f(), vec, bcf(a), bcf(b)});
        uint32_t r = id(); putv(code, Op_ExtInst, {t_u32, r, glsl, Glsl_PackHalf2x16, vec}); return r;
    }
    // v_cvt_pkrtz_f16_f32 is round-toward-ZERO. PackHalf2x16 is round-to-nearest-even — within range that
    // differs by <=1 ULP (accepted), but at the f16 OVERFLOW boundary RTE yields +/-Inf where RTZ clamps
    // to the max finite f16 (+/-65504). Clamp each source to [-65504, 65504] before packing so an HDR
    // value above the f16 range becomes 65504 (matching RTZ's saturate) instead of an Inf/NaN that then
    // propagates through blending/compositing (#452). The RTE store path (pack_half_lo) is unaffected.
    uint32_t pack_half2x16_rtz(uint32_t a, uint32_t b) {
        const uint32_t hi = bcu(fconstf(65504.0f)), lo = bcu(fconstf(-65504.0f));
        uint32_t ca = fext2(Glsl_FMax, fext2(Glsl_FMin, a, hi), lo);
        uint32_t cb = fext2(Glsl_FMax, fext2(Glsl_FMin, b, hi), lo);
        return pack_half2x16(ca, cb);
    }
    // Vertex-attribute unpacking (buffer_load_format_* with a packed data format). Each returns the
    // component as raw float VGPR bits, matching how the hardware presents a format load to the shader.
    //   unpack_norm: extract a `bits`-wide field at `bit_off` of `dword`, convert to float, divide by
    //   `norm` (255/127/65535/32767) — the UNORM/SNORM normalization; SNORM is clamped to >= -1.0.
    uint32_t unpack_norm(uint32_t dword, uint32_t bit_off, uint32_t bits, bool is_signed, float norm) {
        uint32_t fbits_ = is_signed ? cvt_i2f(bfe_s(dword, uconst(bit_off), uconst(bits)))
                                    : cvt_u2f(bfe_u(dword, uconst(bit_off), uconst(bits)));
        uint32_t v = fbin(Op_FDiv, fbits_, bcu(fconstf(norm)));
        if (is_signed) v = fext2(Glsl_FMax, v, bcu(fconstf(-1.0f)));
        return v;
    }
    // unpack_half: extract one of the two f16 halves packed in `dword` (which=0 low, 1 high) -> float bits.
    uint32_t unpack_half(uint32_t dword, uint32_t which) {
        uint32_t vec = id(); putv(code, Op_ExtInst, {t_v2f(), vec, glsl, Glsl_UnpackHalf2x16, dword});
        uint32_t f = id(); put(code, Op_CompositeExtract, {t_f32, f, vec, which}); return bcu(f);
    }
    // GFX10's packed 10/11-bit vertex-float fields use binary16's five-bit exponent and a shortened
    // mantissa, without a sign bit. Widening the complete field left by 4 (11-bit) or 5 (10-bit)
    // produces the exact low-half binary16 encoding, including subnormals, infinity, and NaN.
    uint32_t unpack_ufloat(uint32_t dword, uint32_t bit_off, uint32_t bits) {
        uint32_t raw = bfe_u(dword, uconst(bit_off), uconst(bits));
        uint32_t half = ibin(Op_ShiftLeftLogical, raw, uconst(bits == 11 ? 4u : 5u));
        return unpack_half(half, 0);
    }
    // Round a 24-bit significand right by a dynamic amount using round-to-nearest-even. SPIR-V
    // shifts by the word width are undefined, so clamp the executed shift and select zero for the
    // large-shift case (the significand is then strictly below halfway).
    uint32_t round_shift_even_u32(uint32_t value, uint32_t shift) {
        const uint32_t safe_shift = uext2(Glsl_UMin, shift, uconst(31));
        const uint32_t nonzero_shift = uext2(Glsl_UMax, safe_shift, uconst(1));
        const uint32_t rounded = ibin(Op_ShiftRightLogical, value, nonzero_shift);
        const uint32_t one_shifted = ibin(Op_ShiftLeftLogical, uconst(1), nonzero_shift);
        const uint32_t remainder = ibin(
            Op_BitwiseAnd, value, ibin(Op_ISub, one_shifted, uconst(1)));
        const uint32_t halfway_shift = ibin(Op_ISub, nonzero_shift, uconst(1));
        const uint32_t halfway = ibin(Op_ShiftLeftLogical, uconst(1), halfway_shift);
        const uint32_t above = ucmp(Op_UGreaterThan, remainder, halfway);
        const uint32_t tie = land(
            ucmp(Op_IEqual, remainder, halfway),
            ucmp(Op_INotEqual, ibin(Op_BitwiseAnd, rounded, uconst(1)), uconst(0)));
        const uint32_t incremented = ibin(Op_IAdd, rounded, sel(lor(above, tie), uconst(1), uconst(0)));
        const uint32_t finite = sel(ucmp(Op_IEqual, shift, uconst(0)), value, incremented);
        return sel(ucmp(Op_UGreaterThanEqual, shift, uconst(32)), uconst(0), finite);
    }
    // Exact inverse of unpack_ufloat for R11G11B10 stores. This mirrors float_to_f11/f10 without
    // going through binary16, which would double-round values near the shortened-mantissa ties.
    uint32_t pack_ufloat(uint32_t bits, uint32_t mantissa_bits) {
        const uint32_t exponent = bfe_u(bits, uconst(23), uconst(8));
        const uint32_t mantissa = ibin(Op_BitwiseAnd, bits, uconst(0x7fffff));
        const uint32_t sign = ibin(Op_ShiftRightLogical, bits, uconst(31));
        const uint32_t exponent_bits = uconst(0x1fu << mantissa_bits);
        const uint32_t mantissa_mask = uconst((1u << mantissa_bits) - 1u);

        const uint32_t payload_shifted = ibin(
            Op_ShiftRightLogical, mantissa, uconst(23u - mantissa_bits));
        const uint32_t payload = sel(
            ucmp(Op_IEqual, payload_shifted, uconst(0)), uconst(1), payload_shifted);
        const uint32_t special = sel(
            ucmp(Op_IEqual, mantissa, uconst(0)), exponent_bits,
            ibin(Op_BitwiseOr, exponent_bits, payload));

        const uint32_t significand = ibin(Op_BitwiseOr, uconst(0x800000), mantissa);
        const uint32_t sub_shift = ibin(
            Op_ISub, uconst(136u - mantissa_bits), exponent);
        const uint32_t sub_rounded = round_shift_even_u32(significand, sub_shift);
        const uint32_t subnormal = sel(
            ucmp(Op_UGreaterThanEqual, sub_rounded, uconst(1u << mantissa_bits)),
            uconst(1u << mantissa_bits), sub_rounded);

        const uint32_t normal_rounded = round_shift_even_u32(
            significand, uconst(23u - mantissa_bits));
        const uint32_t carry = ucmp(
            Op_IEqual, normal_rounded, uconst(1u << (mantissa_bits + 1u)));
        const uint32_t target_exponent = ibin(Op_ISub, exponent, uconst(112));
        const uint32_t carried_exponent = ibin(
            Op_IAdd, target_exponent, sel(carry, uconst(1), uconst(0)));
        const uint32_t carried_mantissa = sel(
            carry, uconst(0), ibin(Op_BitwiseAnd, normal_rounded, mantissa_mask));
        const uint32_t normal_finite = ibin(
            Op_BitwiseOr,
            ibin(Op_ShiftLeftLogical, carried_exponent, uconst(mantissa_bits)),
            carried_mantissa);
        const uint32_t normal = sel(
            ucmp(Op_UGreaterThanEqual, carried_exponent, uconst(31)),
            exponent_bits, normal_finite);

        const uint32_t finite = sel(
            ucmp(Op_ULessThanEqual, exponent, uconst(112)), subnormal,
            sel(ucmp(Op_UGreaterThanEqual, exponent, uconst(143)), exponent_bits, normal));
        const uint32_t invalid = lor(
            ucmp(Op_INotEqual, sign, uconst(0)), ucmp(Op_IEqual, exponent, uconst(0)));
        const uint32_t ordinary = sel(invalid, uconst(0), finite);
        // Infinity and NaN are handled before the sign clamp, matching the guest conversion.
        return sel(ucmp(Op_IEqual, exponent, uconst(0xff)), special, ordinary);
    }
    // Inverse of unpack_norm: pack a float VGPR (bits) into a `bits`-wide UNORM/SNORM integer field:
    // clamp to [0,1] (unsigned) or [-1,1] (signed), scale by `norm`, round-to-nearest-even, mask to width.
    uint32_t pack_norm(uint32_t fbits, uint32_t bits, bool is_signed, float norm) {
        uint32_t lo = bcu(fconstf(is_signed ? -1.0f : 0.0f)), hi = bcu(fconstf(1.0f));
        uint32_t clamped = fext2(Glsl_FMax, fext2(Glsl_FMin, fbits, hi), lo);
        uint32_t scaled  = fbin(Op_FMul, clamped, bcu(fconstf(norm)));
        uint32_t rounded = fext1(Glsl_RoundEven, scaled);
        uint32_t ival    = is_signed ? cvt_f2i(rounded) : cvt_f2u(rounded);
        uint32_t mask    = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
        return ibin(Op_BitwiseAnd, ival, uconst(mask));
    }
    // Pack a float into the low 16 bits as an f16 (inverse of unpack_half low half).
    uint32_t pack_half_lo(uint32_t fbits) {
        return ibin(Op_BitwiseAnd, pack_half2x16(fbits, bcu(fconstf(0.0f))), uconst(0xFFFFu));
    }

    // Combined image+sampler support (MIMG image_sample/image_load). Float and unsigned-integer
    // textures need distinct SPIR-V image types even at the same dimension: normalized/float T#s
    // return float bits, while UINT T#s return raw integer channel values. Each texture is still a
    // COMBINED_IMAGE_SAMPLER UniformConstant at its binding.
    std::unordered_map<uint32_t, uint32_t> tex_var;       // binding -> combined-sampler OpVariable id
    std::unordered_map<uint32_t, uint32_t> tex_simg_type; // (Dim | UINT flag) -> OpTypeSampledImage id
    std::unordered_map<uint32_t, uint32_t> tex_img_type;  // same key -> its OpTypeImage id
    std::unordered_map<uint32_t, uint32_t> tex_binding_simg;
    std::unordered_map<uint32_t, uint32_t> tex_binding_img;
    std::unordered_map<uint32_t, bool> tex_binding_uint;
    std::unordered_map<uint32_t, uint32_t> tex_binding_key;
    uint32_t t_v3f_cache = 0;
    uint32_t t_v3f() { if (!t_v3f_cache) { t_v3f_cache = id(); put(types, Op_TypeVector, {t_v3f_cache, t_f32, 3}); } return t_v3f_cache; }
    uint32_t t_v3i_cache = 0;
    uint32_t t_v3i() { if (!t_v3i_cache) { t_v3i_cache = id(); put(types, Op_TypeVector, {t_v3i_cache, t_i32, 3}); } return t_v3i_cache; }
    static uint32_t sampled_image_key(uint32_t dim, bool is_uint, bool arrayed, bool depth) {
        return dim | (is_uint ? 0x100u : 0u) | (arrayed ? 0x200u : 0u) |
               (depth ? 0x400u : 0u);
    }
    uint32_t sampled_image_type(uint32_t dim, bool is_uint = false,
                                bool arrayed = false, bool depth = false) {
        const uint32_t key = sampled_image_key(dim, is_uint, arrayed, depth);
        auto it = tex_simg_type.find(key); if (it != tex_simg_type.end()) return it->second;
        uint32_t ti = id(); put(types, Op_TypeImage, {ti, is_uint ? t_u32 : t_f32,
                                                      dim, depth ? 1u : 0u, arrayed ? 1u : 0u,
                                                      0, 1, 0});  // sampled, Sampled=1
        uint32_t si = id(); put(types, Op_TypeSampledImage, {si, ti});
        tex_simg_type[key] = si; tex_img_type[key] = ti; return si;
    }
    // Declare (idempotently) a combined image+sampler of SPIR-V `dim` at descriptor-set 0, `binding`.
    bool declare_texture(uint32_t binding, uint32_t dim = Dim_2D, bool is_uint = false,
                         bool arrayed = false, bool depth = false) {
        const uint32_t key = sampled_image_key(dim, is_uint, arrayed, depth);
        uint32_t simg = sampled_image_type(dim, is_uint, arrayed, depth);
        if (tex_var.count(binding)) return tex_binding_key[binding] == key;
        uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_UniformConstant, simg});
        uint32_t v = id();     put(types, Op_Variable,    {t_ptr, v, SC_UniformConstant});
        put(deco, Op_Decorate, {v, Dec_DescriptorSet, desc_set});
        put(deco, Op_Decorate, {v, Dec_Binding, binding});
        tex_var[binding] = v;
        tex_binding_simg[binding] = simg;
        tex_binding_img[binding] = tex_img_type[key];
        tex_binding_uint[binding] = is_uint;
        tex_binding_key[binding] = key;
        return true;
    }
    uint32_t texture_vec4(uint32_t binding) {
        return tex_binding_uint[binding] ? t_v4u() : t_v4f;
    }
    void unpack_texture_result(uint32_t binding, uint32_t result, uint32_t out[4]) {
        const bool is_uint = tex_binding_uint[binding];
        const uint32_t scalar_type = is_uint ? t_u32 : t_f32;
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {scalar_type, e, result, c});
            out[c] = is_uint ? e : bcu(e);
        }
    }
    // image_sample 2D: sample the combined sampler at `binding` with (u,v) float-BITS coords; fills
    // out[0..3] with the RGBA result components as raw VGPR bits. Implicit LOD is only legal in the
    // Fragment execution model (#151) — the compute/vertex shells have no derivatives, so there we
    // sample at explicit LOD 0 (what a non-pixel-shader image_sample resolves to without gradients).
    void image_sample_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id();
        if (is_fragment) put(code, Op_ImageSampleImplicitLod, {texture_vec4(binding), res, si, coord});
        else             put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        unpack_texture_result(binding, res, out);
    }
    // image_sample_d 2D: the guest supplies explicit normalized-coordinate derivatives in the
    // modifier-first vaddr slots. Preserve both vectors with the SPIR-V Grad image operand; using
    // implicit fragment-quad derivatives selects the wrong mip at seams and for uniform coordinates.
    void image_sample_grad_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                              uint32_t dsdx_bits, uint32_t dtdx_bits,
                              uint32_t dsdy_bits, uint32_t dtdy_bits, uint32_t out[4]) {
        uint32_t si = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct,
                                   {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t grad_x = id(); put(code, Op_CompositeConstruct,
                                    {t_v2f(), grad_x, bcf(dsdx_bits), bcf(dtdx_bits)});
        uint32_t grad_y = id(); put(code, Op_CompositeConstruct,
                                    {t_v2f(), grad_y, bcf(dsdy_bits), bcf(dtdy_bits)});
        uint32_t res = id(); put(code, Op_ImageSampleExplicitLod,
                                 {texture_vec4(binding), res, si, coord,
                                  ImgOp_Grad, grad_x, grad_y});
        unpack_texture_result(binding, res, out);
    }
    // image_sample 3D: (u,v,w) float-BITS coords -> RGBA. Uses the Dim_3D sampled image; same
    // implicit-LOD-only-in-Fragment rule as image_sample_2d.
    void image_sample_3d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t w_bits, uint32_t out[4]) {
        uint32_t simg  = tex_binding_simg[binding];
        uint32_t si    = id(); put(code, Op_Load, {simg, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v3f(), coord, bcf(u_bits), bcf(v_bits), bcf(w_bits)});
        uint32_t res   = id();
        if (is_fragment) put(code, Op_ImageSampleImplicitLod, {texture_vec4(binding), res, si, coord});
        else             put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        unpack_texture_result(binding, res, out);
    }
    // image_sample_l / _lz: sample with an EXPLICIT LOD (lod_bits float). Stage-agnostic (no derivatives).
    void image_sample_lod_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t lod_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id(); put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, bcf(lod_bits)});
        unpack_texture_result(binding, res, out);
    }
    // A real 2D-array explicit-LOD sample. The layer is the third float coordinate; keeping it in
    // SPIR-V makes reflection require a matching 2D-array view instead of silently sampling layer 0.
    void image_sample_lod_2d_array(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                   uint32_t layer_bits, uint32_t lod_bits, uint32_t out[4]) {
        uint32_t si = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct,
                                   {t_v3f(), coord, bcf(u_bits), bcf(v_bits), bcf(layer_bits)});
        uint32_t res = id(); put(code, Op_ImageSampleExplicitLod,
                                 {texture_vec4(binding), res, si, coord,
                                  ImgOp_Lod, bcf(lod_bits)});
        unpack_texture_result(binding, res, out);
    }
    // image_sample_lz from a 3D texture: explicit LOD (usually 0) on a (u,v,w) coord. Stage-agnostic.
    void image_sample_lod_3d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t w_bits,
                             uint32_t lod_bits, uint32_t out[4]) {
        uint32_t simg  = tex_binding_simg[binding];
        uint32_t si    = id(); put(code, Op_Load, {simg, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v3f(), coord, bcf(u_bits), bcf(v_bits), bcf(w_bits)});
        uint32_t res   = id(); put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, bcf(lod_bits)});
        unpack_texture_result(binding, res, out);
    }
    // Compare one sampled/fetched depth value against DREF and return raw float bits containing 0/1.
    uint32_t image_dref_compare_bits(uint32_t depth, uint32_t dref_bits, uint32_t compare_func) {
        if (compare_func == 0u) return bcu(fconstf(0.0f));   // NEVER
        if (compare_func == 7u) return bcu(fconstf(1.0f));   // ALWAYS
        uint32_t op;
        switch (compare_func) {
            case 1u: op = Op_FOrdLessThan;         break;
            case 2u: op = Op_FOrdEqual;            break;
            case 3u: op = Op_FOrdLessThanEqual;    break;
            case 4u: op = Op_FOrdGreaterThan;      break;
            case 5u: op = Op_FOrdNotEqual;         break;
            default: op = Op_FOrdGreaterThanEqual; break;   // 6 GEQUAL
        }
        uint32_t cmp = id(); put(code, op, {t_bool, cmp, bcf(dref_bits), depth});
        uint32_t result = id();
        put(code, Op_Select, {t_f32, result, cmp, fconstf(1.0f), fconstf(0.0f)});
        return bcu(result);
    }
    // IMAGE_SAMPLE_C_LZ on a plain 2D texture (or the backend's base-slice 2D_ARRAY fallback),
    // lowered as a MANUAL depth compare. The hardware path
    // (compareEnable sampler over a depth-format view) needs backend machinery the color-texture path
    // lacks (see the render-runner S# note). NEAREST retains the exact single sampled tap. LINEAR uses
    // four level-0 fetches, compares each independently, then bilinearly weights the booleans — the
    // compare-before-filter order required for 2x2 PCF. Vulkan compare semantics: result =
    // (reference OP stored); the SQ S# compare enum
    // (WORD0 [14:12]) matches VkCompareOp order — 0 NEVER, 1 LESS, 2 EQUAL, 3 LEQUAL, 4 GREATER,
    // 5 NOTEQUAL, 6 GEQUAL, 7 ALWAYS. CONFIDENCE: MED (standard AMD sampler enum ordering; Blue
    // Prince renders visually-correct shadows with it — the live A/B on #1271).
    void image_sample_dref_manual_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                     uint32_t dref_bits, uint32_t compare_func,
                                     bool linear_filter, uint32_t addr_u, uint32_t addr_v,
                                     uint32_t border_color_type, uint32_t out[4],
                                     bool arrayed = false, uint32_t layer_bits = 0) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        if (linear_filter) {
            if (!declared_image_query) {
                put(caps, Op_Capability, {Cap_ImageQuery});
                declared_image_query = true;
            }
            uint32_t img = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
            uint32_t size = id(); put(code, Op_ImageQuerySizeLod,
                                      {arrayed ? t_v3i() : t_v2i(), size, img, uconst(0)});
            uint32_t w_i = id(); put(code, Op_CompositeExtract, {t_i32, w_i, size, 0});
            uint32_t h_i = id(); put(code, Op_CompositeExtract, {t_i32, h_i, size, 1});
            const uint32_t w = i2u(w_i), h = i2u(h_i);

            // Normalized linear filtering addresses texel centers at n + 0.5. Convert to that
            // footprint, then address each tap with the decoded S#. Match the backend's current
            // Gen5 contract (#1400): 0=repeat, 1=mirrored repeat, 2..5=clamp-to-edge, 6..7=border.
            // The finer mirror-once/half-border distinctions in 3..5 remain deliberately folded by
            // both paths until the shared sampler decoder grows them.
            const uint32_t half = bcu(fconstf(0.5f));
            const uint32_t x = fbin(Op_FSub, fbin(Op_FMul, u_bits, cvt_i2f(w)), half);
            const uint32_t y = fbin(Op_FSub, fbin(Op_FMul, v_bits, cvt_i2f(h)), half);
            const uint32_t fx = fext1(Glsl_Fract, x), fy = fext1(Glsl_Fract, y);
            const uint32_t bx = cvt_f2i(fext1(Glsl_Floor, x));
            const uint32_t by = cvt_f2i(fext1(Glsl_Floor, y));
            auto address_coord = [&](uint32_t value, uint32_t extent, uint32_t mode,
                                     uint32_t& outside_border) {
                outside_border = bfalse();
                if (mode == 0u) return sbin(Op_SMod, value, extent);
                if (mode == 1u) {
                    const uint32_t period = sbin(Op_IMul, extent, uconst(2));
                    const uint32_t wrapped = sbin(Op_SMod, value, period);
                    const uint32_t reflected = sbin(
                        Op_ISub, sbin(Op_ISub, period, uconst(1)), wrapped);
                    return sel(scmp(Op_SGreaterThanEqual, wrapped, extent), reflected, wrapped);
                }
                const uint32_t maximum = sbin(Op_ISub, extent, uconst(1));
                if (mode >= 6u) {
                    outside_border = lor(scmp(Op_SLessThan, value, uconst(0)),
                                         scmp(Op_SGreaterThanEqual, value, extent));
                }
                return sext2(Glsl_SMin, sext2(Glsl_SMax, value, uconst(0)), maximum);
            };
            uint32_t x0_border, x1_border, y0_border, y1_border;
            const uint32_t x0 = address_coord(bx, w, addr_u, x0_border);
            const uint32_t y0 = address_coord(by, h, addr_v, y0_border);
            const uint32_t x1 = address_coord(
                sbin(Op_IAdd, bx, uconst(1)), w, addr_u, x1_border);
            const uint32_t y1 = address_coord(
                sbin(Op_IAdd, by, uconst(1)), h, addr_v, y1_border);

            const bool uses_border = addr_u >= 6u || addr_v >= 6u;
            uint32_t border_compare = 0;
            if (uses_border) {
                // The sampled depth is R: transparent/opaque black and unsupported custom-register
                // border colors all contribute 0, while opaque white contributes 1. This mirrors the
                // backend sampler's custom-border fallback.
                const float border_depth = border_color_type == 2u ? 1.0f : 0.0f;
                border_compare = image_dref_compare_bits(
                    fconstf(border_depth), dref_bits, compare_func);
            }

            auto compare_fetch = [&](uint32_t tx, uint32_t ty, uint32_t outside_border) {
                uint32_t coord = id();
                if (arrayed)
                    put(code, Op_CompositeConstruct,
                        {t_v3u_fetch(), coord, tx, ty, i2u(cvt_f2i(bcf(layer_bits)))});
                else
                    put(code, Op_CompositeConstruct, {t_v2u(), coord, tx, ty});
                uint32_t texel = id();
                put(code, Op_ImageFetch,
                    {t_v4f, texel, img, coord, ImgOp_Lod, uconst(0)});
                uint32_t depth = id();
                put(code, Op_CompositeExtract, {t_f32, depth, texel, 0});
                const uint32_t fetched = image_dref_compare_bits(
                    depth, dref_bits, compare_func);
                return uses_border ? sel(outside_border, border_compare, fetched) : fetched;
            };
            auto tap_border = [&](uint32_t x_border, uint32_t y_border) {
                return uses_border ? lor(x_border, y_border) : bfalse();
            };
            const uint32_t c00 = compare_fetch(x0, y0, tap_border(x0_border, y0_border));
            const uint32_t c10 = compare_fetch(x1, y0, tap_border(x1_border, y0_border));
            const uint32_t c01 = compare_fetch(x0, y1, tap_border(x0_border, y1_border));
            const uint32_t c11 = compare_fetch(x1, y1, tap_border(x1_border, y1_border));
            auto lerp = [&](uint32_t a, uint32_t b, uint32_t t) {
                return fbin(Op_FAdd, a, fbin(Op_FMul, fbin(Op_FSub, b, a), t));
            };
            const uint32_t bits = lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy);
            out[0] = out[1] = out[2] = out[3] = bits;
            return;
        }

        uint32_t coord = id();
        if (arrayed)
            put(code, Op_CompositeConstruct,
                {t_v3f(), coord, bcf(u_bits), bcf(v_bits), bcf(layer_bits)});
        else
            put(code, Op_CompositeConstruct,
                {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id(); put(code, Op_ImageSampleExplicitLod,
                                   {t_v4f, res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        uint32_t depth = id(); put(code, Op_CompositeExtract, {t_f32, depth, res, 0});
        const uint32_t bits = image_dref_compare_bits(depth, dref_bits, compare_func);
        out[0] = out[1] = out[2] = out[3] = bits;
    }
    // image_sample_b 2D: implicit-LOD sample with an LOD BIAS (bias_bits float). Bias only means
    // anything with implicit LOD (fragment derivatives); outside the fragment stage the op resolves
    // like the other samples there — explicit LOD 0 (bias dropped, matching image_sample_2d's rule).
    void image_sample_bias_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t bias_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id();
        if (is_fragment) put(code, Op_ImageSampleImplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Bias, bcf(bias_bits)});
        else             put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        unpack_texture_result(binding, res, out);
    }
    // image_gather4_lz 2D: OpImageGather of component `comp` (0..3) — the 2x2 footprint's four texels
    // of one channel, in the DX/GL gather order ((0,1),(1,1),(1,0),(0,0)), which the AMD gather4
    // result order matches. Gather always samples the base level (== the _lz behavior). out[0..3] =
    // the four gathered values as raw bits.
    void image_gather_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t comp, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id(); put(code, Op_ImageGather, {texture_vec4(binding), res, si, coord, uconst(comp)});
        unpack_texture_result(binding, res, out);
    }
    // image_gather4_lz_o 2D: gather with the MIMG _o per-pixel OFFSET operand. The offset VGPR packs
    // signed 6-bit TEXEL offsets (x = bits[5:0], y = bits[13:8] — AMD RDNA2 ISA "offset" packing, the
    // same fields image_sample_*_o uses). SPIR-V's dynamic Offset image operand requires the
    // ImageGatherExtended capability (the offset VGPR is not a compile-time constant in our SSA model,
    // even though shaders typically v_mov an immediate into it).
    bool declared_gather_ext = false;
    uint32_t t_v2i_cache = 0;
    uint32_t t_v2i() { if (!t_v2i_cache) { t_v2i_cache = id(); put(types, Op_TypeVector, {t_v2i_cache, t_i32, 2}); } return t_v2i_cache; }
    void image_gather_offset_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t comp,
                                uint32_t off_bits, uint32_t out[4]) {
        if (!declared_gather_ext) { put(caps, Op_Capability, {Cap_ImageGatherExtended}); declared_gather_ext = true; }
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        // signed 6-bit texel offsets. NOTE: bfe_s takes SPIR-V IDs — raw integers here (the #296
        // original) emitted OpBitFieldSExtract with invalid operand IDs; never caught live because
        // the only gather4_lz_o user (DOLL's FXAA PS) still rejected upstream on its execz region.
        uint32_t ox    = bfe_s(off_bits, uconst(0), uconst(6)), oy = bfe_s(off_bits, uconst(8), uconst(6));
        uint32_t offv  = id(); put(code, Op_CompositeConstruct, {t_v2i(), offv, bcs(ox), bcs(oy)});
        uint32_t res   = id(); put(code, Op_ImageGather, {texture_vec4(binding), res, si, coord, uconst(comp), ImgOp_Offset, offv});
        unpack_texture_result(binding, res, out);
    }
    // image_sample_lz_o 2D: explicit-LOD-0 sample with the MIMG _o packed texel offset (x = bits[5:0],
    // y = bits[13:8], signed 6-bit — same packing as gather4_lz_o). Vulkan forbids the dynamic Offset
    // image operand on OpImageSample* (it is gather-only), so fold the texel offset into the
    // NORMALIZED coordinate instead: u' = u + ox/width, v' = v + oy/height (exact — the hardware adds
    // the integer offset to the unnormalized coordinate before filtering, and at LOD 0
    // (u + ox/W)*W == u*W + ox). The level-0 size comes from OpImageQuerySizeLod (Cap ImageQuery).
    bool declared_image_query = false;
    void image_sample_lz_offset_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                   uint32_t off_bits, uint32_t out[4]) {
        if (!declared_image_query) { put(caps, Op_Capability, {Cap_ImageQuery}); declared_image_query = true; }
        uint32_t si   = id(); put(code, Op_Load,  {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t img  = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
        uint32_t size = id(); put(code, Op_ImageQuerySizeLod, {t_v2i(), size, img, uconst(0)});
        uint32_t w_i  = id(); put(code, Op_CompositeExtract, {t_i32, w_i, size, 0});
        uint32_t h_i  = id(); put(code, Op_CompositeExtract, {t_i32, h_i, size, 1});
        uint32_t ox = bfe_s(off_bits, uconst(0), uconst(6)), oy = bfe_s(off_bits, uconst(8), uconst(6));   // signed 6-bit texel offsets
        uint32_t du = fbin(Op_FDiv, cvt_i2f(ox), cvt_i2f(i2u(w_i)));
        uint32_t dv = fbin(Op_FDiv, cvt_i2f(oy), cvt_i2f(i2u(h_i)));
        image_sample_lod_2d(binding, fbin(Op_FAdd, u_bits, du), fbin(Op_FAdd, v_bits, dv), uconst(0), out);
    }
    // image_get_resinfo: query a sampled image's dimensions at the integer LOD carried in VADDR.
    // RDNA returns {width,height,depth-or-layers,mip-levels}; absent spatial axes are 1. The result is
    // integer data in ordinary VGPRs, so the signed SPIR-V query result is bitcast back to raw u32.
    void image_get_resinfo(uint32_t binding, uint32_t dim, uint32_t lod_bits, uint32_t out[4]) {
        if (!declared_image_query) { put(caps, Op_Capability, {Cap_ImageQuery}); declared_image_query = true; }
        const uint32_t simg = tex_binding_simg[binding];
        const uint32_t image_type = tex_binding_img[binding];
        uint32_t si = id();  put(code, Op_Load,  {simg, si, tex_var[binding]});
        uint32_t img = id(); put(code, Op_Image, {image_type, img, si});
        out[0] = out[1] = out[2] = uconst(1);
        if (dim == Dim_1D) {
            uint32_t size = id(); put(code, Op_ImageQuerySizeLod, {t_i32, size, img, bcs(lod_bits)});
            out[0] = i2u(size);
        } else {
            const uint32_t size_type = dim == Dim_2D ? t_v2i() : t_v3i();
            uint32_t size = id(); put(code, Op_ImageQuerySizeLod, {size_type, size, img, bcs(lod_bits)});
            const uint32_t components = dim == Dim_2D ? 2u : 3u;
            for (uint32_t c = 0; c < components; c++) {
                uint32_t value = id(); put(code, Op_CompositeExtract, {t_i32, value, size, c});
                out[c] = i2u(value);
            }
        }
        uint32_t levels = id(); put(code, Op_ImageQueryLevels, {t_i32, levels, img});
        out[3] = i2u(levels);
    }
    // 2-component uint vector (integer texel coordinates for OpImageFetch).
    uint32_t t_v2u_cache = 0;
    uint32_t t_v2u() { if (!t_v2u_cache) { t_v2u_cache = id(); put(types, Op_TypeVector, {t_v2u_cache, t_u32, 2}); } return t_v2u_cache; }
    // image_load 2D (image_load): texelFetch the image at the combined sampler's `binding` with INTEGER
    // (x,y) coords (raw VGPR bits). OpImage strips the sampler; OpImageFetch at explicit LOD 0.
    void image_fetch_2d(uint32_t binding, uint32_t x_bits, uint32_t y_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load,  {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2u(), coord, x_bits, y_bits});
        uint32_t res   = id(); put(code, Op_ImageFetch, {texture_vec4(binding), res, img, coord, ImgOp_Lod, uconst(0)});
        unpack_texture_result(binding, res, out);
    }

    // image_load from a 3D texture (integer texel fetch through the combined sampler — DOLL's
    // color-grade LUT, #273): OpImage strips the sampler; OpImageFetch with (x,y,z) integer coords.
    uint32_t t_v3u_cache2 = 0;
    uint32_t t_v3u_fetch() {
        if (t_v3u) return t_v3u;   // compute/vertex shells already declare a uvec3 for built-ins
        if (!t_v3u_cache2) { t_v3u_cache2 = id(); put(types, Op_TypeVector, {t_v3u_cache2, t_u32, 3}); }
        return t_v3u_cache2;
    }
    void image_fetch_3d(uint32_t binding, uint32_t x_bits, uint32_t y_bits, uint32_t z_bits, uint32_t out[4]) {
        uint32_t simg  = tex_binding_simg[binding];
        uint32_t t_img3 = tex_binding_img[binding];   // OpImage's result type must be the pair's Image type
        uint32_t si    = id(); put(code, Op_Load,  {simg, si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {t_img3, img, si});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v3u_fetch(), coord, x_bits, y_bits, z_bits});
        uint32_t res   = id(); put(code, Op_ImageFetch, {texture_vec4(binding), res, img, coord, ImgOp_Lod, uconst(0)});
        unpack_texture_result(binding, res, out);
    }

    // --- STORAGE images (MIMG image_load / image_store WITHOUT a sampler; compute copy/blit) ---
    // Integer/packed formats use a UINT-sampled OpTypeImage. The portable host path converts between
    // raw VGPR channels and guest texels; the exact R11G11B10 fallback instead packs one R32ui word
    // in the shader. Four-channel UNORM8/FLOAT16/FLOAT32 use a FLOAT-sampled image instead:
    // Vulkan then performs exactly the descriptor's normalized/float load-store conversion and the
    // host can stage the guest texels at their native width. VGPRs remain raw u32 bits; float image
    // components are bitcast at this boundary. Both paths use Format=Unknown and therefore require
    // the read/write-without-format capabilities.
    uint32_t t_v4u_cache = 0;
    uint32_t t_v4u() { if (!t_v4u_cache) { t_v4u_cache = id(); put(types, Op_TypeVector, {t_v4u_cache, t_u32, 4}); } return t_v4u_cache; }
    uint32_t t_v3u_c = 0;   // integer coordinate vector type (uvec3); 2D reuses the shared t_v2u()
    std::unordered_map<uint32_t, uint32_t> stg_img_type;   // dimension/type/format key -> OpTypeImage
    std::unordered_map<uint32_t, uint32_t> stg_img_binding_type; // binding -> declared OpTypeImage
    std::unordered_map<uint32_t, uint32_t> stg_img_var;    // binding -> storage-image OpVariable id
    std::unordered_map<uint32_t, bool> stg_img_float;      // binding -> float (rather than uint) texels
    std::unordered_map<uint32_t, uint32_t> stg_img_format; // binding -> SPIR-V Image Format
    std::unordered_map<uint32_t, bool> stg_img_packed_r11; // binding -> R11G11B10 packed in R32ui
    bool declared_read_wo_fmt = false, declared_write_wo_fmt = false, declared_sampled1d = false, declared_ms = false, declared_msarray = false;
    static uint32_t stg_key(uint32_t dim, bool arrayed, bool ms, bool float_texel,
                            uint32_t image_format) {
        return dim | (arrayed ? 0x100u : 0u) | (ms ? 0x200u : 0u) |
               (float_texel ? 0x400u : 0u) | (image_format << 11);
    }
    // Declare (idempotently) a storage image of SPIR-V `dim` (arrayed = layer in the coord; ms =
    // multisampled) at set 0, `binding`. Float and uint sampled types are keyed separately.
    void declare_storage_image(uint32_t binding, uint32_t dim, bool arrayed = false,
                               bool ms = false, bool float_texel = false,
                               uint32_t image_format = ImgFmt_Unknown,
                               bool packed_r11 = false) {
        if (dim == Dim_1D && !declared_sampled1d) {   // SPIR-V: Dim=1D needs Sampled1D; storage 1D also needs Image1D
            put(caps, Op_Capability, {Cap_Sampled1D});
            put(caps, Op_Capability, {Cap_Image1D});
            declared_sampled1d = true;
        }
        if (ms && !declared_ms) { put(caps, Op_Capability, {Cap_StorageImageMultisample}); declared_ms = true; }
        if (ms && arrayed && !declared_msarray) { put(caps, Op_Capability, {Cap_ImageMSArray}); declared_msarray = true; }
        uint32_t key = stg_key(dim, arrayed, ms, float_texel, image_format);
        if (!stg_img_type.count(key)) {
            uint32_t ti = id();
            put(types, Op_TypeImage, {ti, float_texel ? t_f32 : t_u32, dim, 0,
                                      arrayed ? 1u : 0u, ms ? 1u : 0u,
                                      Img_Sampled_Storage, image_format});
            stg_img_type[key] = ti;
        }
        if (stg_img_var.count(binding)) return;
        uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_UniformConstant, stg_img_type[key]});
        uint32_t v = id();     put(types, Op_Variable,    {t_ptr, v, SC_UniformConstant});
        put(deco, Op_Decorate, {v, Dec_DescriptorSet, desc_set});
        put(deco, Op_Decorate, {v, Dec_Binding, binding});
        stg_img_binding_type[binding] = stg_img_type[key];
        stg_img_var[binding] = v;
        stg_img_float[binding] = float_texel;
        stg_img_format[binding] = image_format;
        stg_img_packed_r11[binding] = packed_r11;
    }
    // Build the integer coordinate operand from `n` raw-bit VGPR coords (u32 texel indices, incl. the
    // array layer as the last component for arrayed images). n=1 -> scalar u32; 2 -> uvec2; 3 -> uvec3.
    uint32_t stg_coord(uint32_t n, const uint32_t* c) {
        if (n == 1) return c[0];
        // 2D reuses the shared uvec2 helper (also used by texelFetch) so a shader mixing a texture
        // image_load and a 2-coord storage image doesn't emit a duplicate OpTypeVector %uint 2.
        if (n == 2) { uint32_t v = id(); put(code, Op_CompositeConstruct, {t_v2u(), v, c[0], c[1]}); return v; }
        // uvec3: reuse the compute shell's uvec3 (t_v3u, declared for gl_GlobalInvocationID) if present —
        // a second OpTypeVector %uint 3 would be an illegal duplicate non-aggregate type.
        if (!t_v3u_c) {
            if (t_v3u) t_v3u_c = t_v3u;
            else { t_v3u_c = id(); put(types, Op_TypeVector, {t_v3u_c, t_u32, 3}); }
        }
        uint32_t v = id(); put(code, Op_CompositeConstruct, {t_v3u_c, v, c[0], c[1], c[2]}); return v;
    }
    // image_load: OpImageRead the storage image at `binding` (dim gives the coord count); fills out[0..3]
    // with the RGBA texel components as raw VGPR bits (uint sampled type -> no bitcast needed).
    // OOB CONTRACT: like a buffer load (which relies on robustBufferAccess), the read is issued for ALL
    // invocations — including EXEC-inactive lanes whose coordinate may be out of range (e.g. a grid-tail
    // bounds check narrowed EXEC). The loaded value for such a lane is discarded by write-back predication
    // (predicate_write) at the call site, so it is harmless — PROVIDED the device enables IMAGE ROBUSTNESS
    // (robustImageAccess / VK_EXT_image_robustness: OOB image reads return 0), the image analogue of the
    // robustBufferAccess this recompiler already depends on for buffer loads. The runtime must enable it
    // (the test harness does). The store side is instead EXEC-predicated (no robust "harmless" OOB write).
    void image_read(uint32_t binding, uint32_t dim, bool arrayed, uint32_t ncoord, const uint32_t* coords,
                    uint32_t out[4], bool ms = false, uint32_t sample = 0) {
        if (stg_img_format[binding] == ImgFmt_Unknown && !declared_read_wo_fmt) {
            put(caps, Op_Capability, {Cap_StorageImageReadWithoutFormat});
            declared_read_wo_fmt = true;
        }
        const bool float_texel = stg_img_float[binding];
        const uint32_t vector_type = float_texel ? t_v4f : t_v4u();
        uint32_t img   = id(); put(code, Op_Load,
                                   {stg_img_binding_type[binding], img,
                                    stg_img_var[binding]});
        uint32_t coord = stg_coord(ncoord, coords);
        uint32_t res   = id();
        if (ms) put(code, Op_ImageRead, {vector_type, res, img, coord, ImgOp_Sample, sample});
        else    put(code, Op_ImageRead, {vector_type, res, img, coord});
        if (stg_img_packed_r11[binding]) {
            uint32_t packed = id();
            put(code, Op_CompositeExtract, {t_u32, packed, res, 0});
            out[0] = unpack_ufloat(packed, 0, 11);
            out[1] = unpack_ufloat(packed, 11, 11);
            out[2] = unpack_ufloat(packed, 22, 10);
            out[3] = bcu(fconstf(1.0f));
            return;
        }
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id();
            put(code, Op_CompositeExtract, {float_texel ? t_f32 : t_u32, e, res, c});
            out[c] = float_texel ? bcu(e) : e;
        }
    }
    // image_store: OpImageWrite raw-bit VGPR components vals[0..3] as a uvec4 texel to the storage image.
    // When `predicated` (narrowed EXEC), the write is wrapped in a selection merge on `pred` (the per-lane
    // EXEC bool) so inactive lanes do not write — a real conditional store, like cbuf_store. (Image OOB is
    // not covered by robustBufferAccess, so guarding matters: it also skips a lane's write when EXEC is off
    // e.g. a grid-tail bounds check.)
    void image_write(uint32_t binding, uint32_t dim, bool arrayed, uint32_t ncoord, const uint32_t* coords,
                     const uint32_t vals[4], bool predicated = false, uint32_t pred = 0) {
        if (stg_img_format[binding] == ImgFmt_Unknown && !declared_write_wo_fmt) {
            put(caps, Op_Capability, {Cap_StorageImageWriteWithoutFormat});
            declared_write_wo_fmt = true;
        }
        const bool float_texel = stg_img_float[binding];
        uint32_t img   = id(); put(code, Op_Load,
                                   {stg_img_binding_type[binding], img,
                                    stg_img_var[binding]});
        uint32_t coord = stg_coord(ncoord, coords);
        uint32_t texel = id();
        if (stg_img_packed_r11[binding]) {
            const uint32_t r = pack_ufloat(vals[0], 6);
            const uint32_t g = ibin(Op_ShiftLeftLogical, pack_ufloat(vals[1], 6), uconst(11));
            const uint32_t b = ibin(Op_ShiftLeftLogical, pack_ufloat(vals[2], 5), uconst(22));
            const uint32_t packed = ibin(Op_BitwiseOr, ibin(Op_BitwiseOr, r, g), b);
            put(code, Op_CompositeConstruct,
                {t_v4u(), texel, packed, uconst(0), uconst(0), uconst(0)});
        } else if (float_texel)
            put(code, Op_CompositeConstruct,
                {t_v4f, texel, bcf(vals[0]), bcf(vals[1]), bcf(vals[2]), bcf(vals[3])});
        else
            put(code, Op_CompositeConstruct,
                {t_v4u(), texel, vals[0], vals[1], vals[2], vals[3]});
        if (!predicated) { put(code, Op_ImageWrite, {img, coord, texel}); return; }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        put(code, Op_ImageWrite, {img, coord, texel});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }

    // An integer atomic on an R32_UINT 2D storage image. OpImageTexelPointer consumes the descriptor
    // variable directly (not a loaded image object) and yields an Image-storage pointer suitable for
    // the SPIR-V atomic. Vulkan leaves an out-of-bounds image atomic undefined (robust image access
    // does not cover atomics), and RADV can spend seconds in such an atomic before resetting the GPU.
    // Query the bound view and put BOTH the texel pointer and atomic behind an explicit bounds/EXEC
    // branch. A skipped lane leaves VDATA unchanged, matching the no-operation EXEC fallback.
    uint32_t image_atomic_u32(uint16_t opcode, uint32_t binding, uint32_t ncoord,
                              const uint32_t* coords, uint32_t value, bool predicated,
                              uint32_t pred, uint32_t fallback) {
        // The decoder/resource gate currently accepts only non-arrayed 2D R32_UINT atomics.
        if (ncoord != 2) return fallback;
        if (!t_ptr_img_u32) {
            t_ptr_img_u32 = id();
            put(types, Op_TypePointer, {t_ptr_img_u32, SC_Image, t_u32});
        }
        if (!declared_image_query) {
            put(caps, Op_Capability, {Cap_ImageQuery});
            declared_image_query = true;
        }
        const uint32_t image = id();
        put(code, Op_Load, {stg_img_binding_type[binding], image, stg_img_var[binding]});
        const uint32_t size = id();
        put(code, Op_ImageQuerySize, {t_v2i(), size, image});
        const uint32_t width_i = id(), height_i = id();
        put(code, Op_CompositeExtract, {t_i32, width_i, size, 0});
        put(code, Op_CompositeExtract, {t_i32, height_i, size, 1});
        uint32_t in_bounds = ucmp(Op_ULessThan, coords[0], i2u(width_i));
        in_bounds = land(in_bounds, ucmp(Op_ULessThan, coords[1], i2u(height_i)));
        const uint32_t active = predicated ? land(pred, in_bounds) : in_bounds;
        auto emit = [&]() {
            const uint32_t coord = stg_coord(ncoord, coords);
            const uint32_t pointer = id();
            put(code, Op_ImageTexelPointer,
                {t_ptr_img_u32, pointer, stg_img_var[binding], coord, uconst(0)});
            const uint32_t result = id();
            put(code, opcode,
                {t_u32, result, pointer, uconst(Scope_Device),
                 uconst(MemSem_ImageAcqRel), value});
            return result;
        };
        const uint32_t entry = cur_block;
        const uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {active, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = emit();
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u32, result, then_end, fallback, entry);
    }

    // buffer element pointer: base[ gid.x*stride + k ]
    uint32_t elem_ptr(uint32_t bufvar, uint32_t k) {
        uint32_t idx = gidx;
        if (stride != 1) { uint32_t m = id(); put(code, Op_IMul, {t_u32, m, gidx, uconst(stride)}); idx = m; }
        if (k != 0) { uint32_t a = id(); put(code, Op_IAdd, {t_u32, a, idx, uconst(k)}); idx = a; }
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_f32, p, bufvar, uconst(0), idx}); return p;
    }
    // Load one float from the input buffer and return it as raw bits (VGPR value).
    uint32_t load_input(uint32_t k) { uint32_t p = elem_ptr(v_in, k); uint32_t r = id(); put(code, Op_Load, {t_f32, r, p}); return bcu(r); }
    // The storage-buffer variable for a descriptor `binding`. N-buffer model: each distinct constant/
    // vertex buffer the shader reads is bound at its own descriptor binding (the executor assigns them),
    // so multiple constant buffers (e.g. Unity's per-draw transform vs per-frame) don't collapse onto one.
    // Bindings 2/3 keep mapping to v_cbuf/v_cbuf1 for the compute-shell + legacy 2-slot callers; any other
    // binding uses cbuf_var[] (declared by declare_cbufs from the resource table). Falls back to v_cbuf.
    uint32_t buf_for_binding(uint32_t binding) {
        if (binding == 2) return v_cbuf;
        if (binding == 3) return v_cbuf1;
        auto it = cbuf_var.find(binding); return it != cbuf_var.end() ? it->second : v_cbuf;
    }
    // Load one dword (raw bits) from the constant/vertex buffer at descriptor `binding` at dword index
    // `idx` (SMEM). The 1-arg form keeps the legacy slot convention (0 -> binding 2, 1 -> binding 3).
    uint32_t cbuf_load(uint32_t idx, uint32_t binding = 2) {
        uint32_t buf = buf_for_binding(binding);
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_u32, p, buf, uconst(0), idx});
        uint32_t r = id(); put(code, Op_Load, {t_u32, r, p}); return r;
    }
    // Store one dword `value` to the buffer at descriptor `binding` at dword index `idx` (MUBUF store).
    // When `predicated`, the store is wrapped in a selection merge on `pred` (the per-lane EXEC bool) so
    // inactive lanes do not write — a real conditional store, not a select of a loaded old value.
    void cbuf_store(uint32_t idx, uint32_t value, uint32_t binding, bool predicated, uint32_t pred) {
        uint32_t buf = buf_for_binding(binding);
        if (!predicated) {
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_u32, p, buf, uconst(0), idx});
            put(code, Op_Store, {p, value}); return;
        }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_u32, p, buf, uconst(0), idx});
        put(code, Op_Store, {p, value});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }
    // Returning atomic RMW on a descriptor-backed storage buffer. BUFFER_ATOMIC_* writes memory and
    // returns the pre-operation value in VDATA. Inactive EXEC lanes neither access the buffer nor
    // clobber VDATA, hence the predicated path joins the old destination through OpPhi.
    uint32_t cbuf_atomic_rtn(uint32_t op, uint32_t idx, uint32_t value, uint32_t binding,
                             bool predicated, uint32_t pred, uint32_t fallback) {
        const uint32_t buf = buf_for_binding(binding);
        auto emit = [&]() {
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_u32, p, buf, uconst(0), idx});
            uint32_t result = id();
            put(code, op, {t_u32, result, p, uconst(Scope_Device),
                           uconst(MemSem_UniformAcqRel), value});
            return result;
        };
        if (!predicated) return emit();
        const uint32_t entry = cur_block;
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = emit();
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u32, result, then_end, fallback, entry);
    }
    // RADV currently hangs/reset-poisons the device on Astro Bot's compute R32_UINT image atomic.
    // Compute lowers that exact 2D resource through a detiled storage-buffer view instead. The live
    // backend recognizes the reflected atomic buffer over a StorageImage resource, detiles before
    // dispatch, and tiles the result back afterwards. Graphics retains the native image-atomic path.
    bool declare_compute_atomic_image_buffer(uint32_t binding) {
        if (!is_compute || !t_ptr_sb_struct_u || cbuf_var.count(binding)) return false;
        const uint32_t variable = id();
        put(deco, Op_Decorate, {variable, Dec_DescriptorSet, desc_set});
        put(deco, Op_Decorate, {variable, Dec_Binding, binding});
        put(types, Op_Variable, {t_ptr_sb_struct_u, variable, SC_StorageBuffer});
        cbuf_var[binding] = variable;
        return true;
    }
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
    void declare_lds() {
        if (lds_var) return;
        const uint32_t dwords = is_compute ? lds_dwords : vertex_lds_dwords;
        if (!dwords) return;
        uint32_t len = uconst(dwords);
        uint32_t t_arr = id();        put(types, Op_TypeArray, {t_arr, t_u32, len});
        if (is_vertex) {
            // Vertex-stage LDS is only legal for the exact NGG wrapper selected by recompile_vertex.
            // All other vertex DS shapes reject before reaching this declaration.
            uint32_t t_ptr_fn_arr = 0;
            lds_var = function_var(t_arr, t_ptr_fn_arr);
            t_ptr_lds_u32 = id();
            put(types, Op_TypePointer, {t_ptr_lds_u32, SC_Function, t_u32});
        } else {
            uint32_t t_ptr_wg_arr = id(); put(types, Op_TypePointer, {t_ptr_wg_arr, SC_Workgroup, t_arr});
            lds_var = id();               put(types, Op_Variable, {t_ptr_wg_arr, lds_var, SC_Workgroup});
            t_ptr_lds_u32 = id();
            put(types, Op_TypePointer, {t_ptr_lds_u32, SC_Workgroup, t_u32});
        }
    }
    uint32_t lds_load(uint32_t idx) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
        uint32_t r = id(); put(code, Op_Load, {t_u32, r, p}); return r;
    }
    // Store to LDS[idx]; EXEC-predicated (conditional store) under a narrowed mask, like cbuf_store.
    void lds_store(uint32_t idx, uint32_t value, bool predicated, uint32_t pred) {
        if (!predicated) {
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
            put(code, Op_Store, {p, value}); return;
        }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
        put(code, Op_Store, {p, value});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }
    // Unsigned LDS atomic RMW. The old value is intentionally discarded; the non-returning RDNA DS
    // form exposes only the memory effect. Workgroup scope + WorkgroupMemory AcquireRelease matches
    // the storage class and the barrier helper used around cooperating accesses.
    void lds_atomic(uint32_t op, uint32_t idx, uint32_t value, bool predicated, uint32_t pred) {
        auto emit = [&]() {
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
            uint32_t result = id();
            put(code, op, {t_u32, result, p, uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel), value});
        };
        if (!predicated) { emit(); return; }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        emit();
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }
    // Returning LDS atomic RMW. Inactive lanes must neither touch LDS nor observe an undefined
    // atomic result, so the predicated form joins the old destination fallback through OpPhi.
    uint32_t lds_atomic_rtn(uint32_t op, uint32_t idx, uint32_t value,
                            bool predicated, uint32_t pred, uint32_t fallback) {
        auto emit = [&]() {
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
            uint32_t result = id();
            put(code, op, {t_u32, result, p, uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel), value});
            return result;
        };
        if (!predicated) return emit();
        const uint32_t entry = cur_block;
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = emit();
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u32, result, then_end, fallback, entry);
    }
    // s_barrier: workgroup execution + memory barrier (OpControlBarrier).
    void barrier() {
        if (ngg_private_lds) return; // exact one-lane wrapper has no peer requiring synchronization
        uses_barrier = true;
        put(code, Op_ControlBarrier, {uconst(Scope_Workgroup), uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel)});
    }

    // Dispatcher-only scratch. Unlike guest LDS, this is private recompiler state: one flag slot per
    // padded hardware-wave lane, one result per wave, and one whole-workgroup liveness result.
    uint32_t cfg_scratch = 0, t_ptr_cfg_u32 = 0;
    void declare_cfg_scratch(uint32_t dwords) {
        if (cfg_scratch) return;
        uint32_t t_arr = id(); put(types, Op_TypeArray, {t_arr, t_u32, uconst(dwords)});
        uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_Workgroup, t_arr});
        cfg_scratch = id(); put(types, Op_Variable, {t_ptr, cfg_scratch, SC_Workgroup});
        t_ptr_cfg_u32 = id(); put(types, Op_TypePointer, {t_ptr_cfg_u32, SC_Workgroup, t_u32});
    }
    uint32_t cfg_scratch_load(uint32_t idx) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_cfg_u32, p, cfg_scratch, idx});
        uint32_t value = id(); put(code, Op_Load, {t_u32, value, p}); return value;
    }
    void cfg_scratch_store(uint32_t idx, uint32_t value) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_cfg_u32, p, cfg_scratch, idx});
        put(code, Op_Store, {p, value});
    }

    // --- Wave model (cross-lane ops via workgroup scratch). A dedicated LDS scratch array (separate
    // from ds_read/write's lds_var) holds each lane's contribution so cross-lane reductions work.
    // Inactive lanes still EXECUTE (exec is a per-lane predication bool), so all 64 reach the barriers —
    // valid only at wave-uniform points (the caller must not emit these inside divergent control flow). ---
    uint32_t v_localid = 0, localid = 0, lds_wave = 0, t_ptr_wg_u32b = 0;
    void declare_wave_lds() {
        if (lds_wave) return;
        // Keep one result slot per guest wave separate from the per-lane publication area.  A
        // completed vote can then be consumed while a faster invocation starts publishing the next
        // one, without needing a third workgroup barrier solely to protect the result.
        const uint32_t wave_count = (local_count + wave_size - 1) / wave_size;
        uint32_t t_arr = id();
        put(types, Op_TypeArray,
            {t_arr, t_u32, uconst(std::max(1u, local_count + wave_count))});
        uint32_t t_ptr = id();  put(types, Op_TypePointer, {t_ptr, SC_Workgroup, t_arr});
        lds_wave = id();        put(types, Op_Variable, {t_ptr, lds_wave, SC_Workgroup});
        t_ptr_wg_u32b = id();   put(types, Op_TypePointer, {t_ptr_wg_u32b, SC_Workgroup, t_u32});
    }
    // Exact guest-wave vote at a workgroup-uniform site. Vulkan subgroup width is implementation-
    // defined and need not match the guest's 32/64-lane wave, so publish every lane through shared
    // scratch and reduce only lanes with the same guest-wave index. One lane per guest wave performs
    // the reduction and publishes it in a separate result slot.  Besides avoiding an assumption about
    // Vulkan subgroup membership, this makes the amount of LDS work linear in the workgroup size:
    // the old per-invocation reduction made a 256-thread group perform 65,536 loads for every vote.
    // The first barrier publishes lane bits and the second publishes wave results. The caller proves
    // the site is outside wave-divergent structured control flow.
    uint32_t guest_wave_any(uint32_t active_bool) {
        declare_wave_lds();
        const uint32_t zero = uconst(0);
        const uint32_t bit = sel(active_bool, uconst(1), uconst(0));
        uint32_t p = id();
        putv(code, Op_AccessChain, {t_ptr_wg_u32b, p, lds_wave, linear_localid});
        put(code, Op_Store, {p, bit});
        barrier();

        const uint32_t wave_shift = wave_size == 32 ? 5u : 6u;
        const uint32_t wave_index = ibin(
            Op_ShiftRightLogical, linear_localid, uconst(wave_shift));
        const uint32_t wave_base = ibin(
            Op_ShiftLeftLogical, wave_index, uconst(wave_shift));
        const uint32_t lane = ibin(
            Op_BitwiseAnd, linear_localid, uconst(wave_size - 1));
        const uint32_t leader = id(), reduced = id();
        const uint32_t is_leader = ucmp(Op_IEqual, lane, zero);
        emit_selmerge(reduced);
        emit_condbranch(is_leader, leader, reduced);
        emit_label(leader);
        uint32_t any_word = zero;
        for (uint32_t i = 0; i < wave_size; ++i) {
            const uint32_t idx = ibin(Op_IAdd, wave_base, uconst(i));
            const uint32_t valid = ucmp(Op_ULessThan, idx, uconst(local_count));
            // Keep the memory access in-bounds for a partial final guest wave. The selected value is
            // ignored when idx names a padded lane, but Vulkan must never observe an OOB OpLoad.
            const uint32_t safe_idx = sel(valid, idx, zero);
            uint32_t q = id();
            putv(code, Op_AccessChain, {t_ptr_wg_u32b, q, lds_wave, safe_idx});
            uint32_t value = id();
            put(code, Op_Load, {t_u32, value, q});
            any_word = ibin(Op_BitwiseOr, any_word, sel(valid, value, zero));
        }
        const uint32_t result_index = ibin(Op_IAdd, uconst(local_count), wave_index);
        uint32_t result_ptr = id();
        putv(code, Op_AccessChain,
             {t_ptr_wg_u32b, result_ptr, lds_wave, result_index});
        put(code, Op_Store, {result_ptr, any_word});
        emit_branch(reduced);
        emit_label(reduced);
        barrier();
        const uint32_t result_index_all = ibin(
            Op_IAdd, uconst(local_count), wave_index);
        uint32_t result_ptr_all = id();
        putv(code, Op_AccessChain,
             {t_ptr_wg_u32b, result_ptr_all, lds_wave, result_index_all});
        uint32_t result = id();
        put(code, Op_Load, {t_u32, result, result_ptr_all});
        return ucmp(Op_INotEqual, result, zero);
    }
    // v_mbcnt_lo/hi: count active lanes below this one. active_bool = this lane's mask bit (EXEC); acc =
    // src1 accumulator (bits); `lo` selects the [0,32) half (lo) or [32,64) half (hi). Combined lo→hi over
    // a wave = the lane's compaction index among active lanes. Populate LDS[lane]=active, barrier, then an
    // unrolled prefix-count over the half; trailing barrier so a following mbcnt can safely re-populate.
    uint32_t mbcnt(uint32_t active_bool, uint32_t acc_bits, bool lo) {
        declare_wave_lds();
        uint32_t bit = sel(active_bool, uconst(1), uconst(0));
        { uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32b, p, lds_wave, linear_localid}); put(code, Op_Store, {p, bit}); }
        barrier();
        const uint32_t lane = ibin(Op_BitwiseAnd, linear_localid, uconst(wave_size - 1));
        const uint32_t wave_index = ibin(Op_ShiftRightLogical, linear_localid,
                                          uconst(wave_size == 32 ? 5u : 6u));
        uint32_t sum = uconst(0);
        const uint32_t first = lo ? 0u : 32u;
        const uint32_t last = std::min(wave_size, lo ? 32u : 64u);
        for (uint32_t i = 0; i < local_count; i++) {
            const uint32_t candidate_lane = i % wave_size;
            if (candidate_lane < first || candidate_lane >= last) continue;
            uint32_t cond = land(ucmp(Op_IEqual, wave_index, uconst(i / wave_size)),
                                 ucmp(Op_ULessThan, uconst(candidate_lane), lane));
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32b, p, lds_wave, uconst(i)});
            uint32_t v = id(); put(code, Op_Load, {t_u32, v, p});
            sum = b_iadd(sum, sel(cond, v, uconst(0)));
        }
        barrier();
        return b_iadd(acc_bits, sum);
    }
    // DS_APPEND/DS_CONSUME: one atomic add/subtract per emulated hardware wave, changing the counter
    // by popcount(EXEC), with the old value broadcast to every lane. GDS operations target the
    // backend-owned persistent device buffer; ordinary operations target workgroup LDS. Every
    // Vulkan invocation participates in the barriers;
    // the per-lane EXEC bool only contributes 0/1 to the reduction. A partial final hardware wave
    // (including a one-thread workgroup) contains only the launched lanes; every scratch read remains
    // statically in-bounds and absent hardware lanes contribute zero.
    uint32_t wave_append(uint32_t index, uint32_t active_bool, bool consume = false,
                         bool gds = false) {
        declare_wave_lds();
        const uint32_t active = sel(active_bool, uconst(1), uconst(0));
        uint32_t p = id(); putv(code, Op_AccessChain,
            {t_ptr_wg_u32b, p, lds_wave, linear_localid});
        put(code, Op_Store, {p, active});
        barrier();

        const uint32_t wave_shift = wave_size == 32 ? 5u : 6u;
        const uint32_t lane = ibin(Op_BitwiseAnd, linear_localid, uconst(wave_size - 1));
        const uint32_t wave_base = ibin(Op_ShiftLeftLogical,
            ibin(Op_ShiftRightLogical, linear_localid, uconst(wave_shift)), uconst(wave_shift));
        const uint32_t wave_index = ibin(Op_ShiftRightLogical, linear_localid, uconst(wave_shift));
        uint32_t count = uconst(0);
        for (uint32_t i = 0; i < local_count; ++i) {
            uint32_t q = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32b, q, lds_wave, uconst(i)});
            uint32_t v = id(); put(code, Op_Load, {t_u32, v, q});
            const uint32_t same_wave = ucmp(Op_IEqual, wave_index, uconst(i / wave_size));
            count = b_iadd(count, sel(same_wave, v, uconst(0)));
        }

        const uint32_t is_leader = ucmp(Op_IEqual, lane, uconst(0));
        const uint32_t leader = id(), reduced = id();
        emit_selmerge(reduced);
        emit_condbranch(is_leader, leader, reduced);
        emit_label(leader);
        const uint32_t old = gds
            ? compute_gds_atomic_rtn(consume ? Op_AtomicISub : Op_AtomicIAdd, index, count)
            : lds_atomic_rtn(consume ? Op_AtomicISub : Op_AtomicIAdd, index, count,
                             false, btrue(), uconst(0));
        uint32_t result_slot = id(); putv(code, Op_AccessChain,
            {t_ptr_wg_u32b, result_slot, lds_wave, wave_base});
        put(code, Op_Store, {result_slot, old});
        emit_branch(reduced);
        emit_label(reduced);
        barrier();
        uint32_t result_ptr = id(); putv(code, Op_AccessChain,
            {t_ptr_wg_u32b, result_ptr, lds_wave, wave_base});
        uint32_t result = id(); put(code, Op_Load, {t_u32, result, result_ptr});
        barrier();
        return result;
    }
    uint32_t native_wave_append(uint32_t lds_idx, uint32_t active_bool,
                                uint32_t consume) {
        if (!native_subgroup_size) return 0;
        (void)subgroup_local_id();
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t contribution = sel(active_bool, uconst(1), uconst(0));
        uint32_t count = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, count, uconst(Scope_Subgroup), GroupOp_Reduce, contribution});
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, contribution});
        const uint32_t elected = land(active_bool, ucmp(Op_IEqual, prefix, uconst(0)));
        const uint32_t entry = cur_block, leader = id(), merge = id();
        emit_selmerge(merge);
        emit_condbranch(elected, leader, merge);
        emit_label(leader);
        const uint32_t delta = sel(consume, ibin(Op_ISub, uconst(0), count), count);
        const uint32_t leader_old = lds_atomic_rtn(
            Op_AtomicIAdd, lds_idx, delta, false, btrue(), uconst(0));
        const uint32_t leader_end = cur_block;
        emit_branch(merge);
        emit_label(merge);
        const uint32_t local_old = emit_phi_2way(
            t_u32, leader_old, leader_end, uconst(0), entry);
        uint32_t old = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, old, uconst(Scope_Subgroup), GroupOp_Reduce, local_old});
        return old;
    }
    uint32_t b_iadd(uint32_t a, uint32_t b_) { uint32_t r = id(); put(code, Op_IAdd, {t_u32, r, a, b_}); return r; }
    // Declare the two scalar-memory constant/vertex buffers (bindings 2 & 3) that SMEM / buffer_load_
    // format_* read. Called by every shell (compute/vertex/fragment) so cbuf_load works in each.
    // Requires t_u32 to already be declared. Unused by shaders without memory ops.
    void declare_cbufs(const ShaderResourceTable* rt = nullptr) {
        uint32_t t_rta_u = id(), t_struct_u = id();
        t_ptr_sb_struct_u = id();
        v_cbuf = id(); v_cbuf1 = id(); t_ptr_sb_u32 = id();
        put(deco, Op_Decorate, {t_rta_u, Dec_ArrayStride, 4});
        put(deco, Op_MemberDecorate, {t_struct_u, 0, Dec_Offset, 0});
        put(deco, Op_Decorate, {t_struct_u, Dec_Block});
        put(deco, Op_Decorate, {v_cbuf,  Dec_DescriptorSet, desc_set}); put(deco, Op_Decorate, {v_cbuf,  Dec_Binding, 2});
        put(deco, Op_Decorate, {v_cbuf1, Dec_DescriptorSet, desc_set}); put(deco, Op_Decorate, {v_cbuf1, Dec_Binding, 3});
        put(types, Op_TypeRuntimeArray, {t_rta_u, t_u32});
        put(types, Op_TypeStruct, {t_struct_u, t_rta_u});
        put(types, Op_TypePointer, {t_ptr_sb_struct_u, SC_StorageBuffer, t_struct_u});
        put(types, Op_Variable, {t_ptr_sb_struct_u, v_cbuf,  SC_StorageBuffer});
        put(types, Op_Variable, {t_ptr_sb_struct_u, v_cbuf1, SC_StorageBuffer});
        put(types, Op_TypePointer, {t_ptr_sb_u32, SC_StorageBuffer, t_u32});
        cbuf_var[2] = v_cbuf; cbuf_var[3] = v_cbuf1;
        // N-buffer model: declare an additional storage buffer for each distinct constant/vertex buffer
        // binding the resource table uses beyond 2/3, so the shader's several buffers stay distinct. All
        // share the same Block struct type (a runtime array of u32); only the binding decoration differs.
        if (rt) {
            std::set<uint32_t> seen{2, 3};
            for (const auto& r : rt->resources) {
                if (r.cls != ResourceClass::ConstantBuffer && r.cls != ResourceClass::VertexBuffer) continue;
                if (!seen.insert(r.binding).second) continue;
                uint32_t v = id();
                put(deco, Op_Decorate, {v, Dec_DescriptorSet, desc_set}); put(deco, Op_Decorate, {v, Dec_Binding, r.binding});
                put(types, Op_Variable, {t_ptr_sb_struct_u, v, SC_StorageBuffer});
                cbuf_var[r.binding] = v;
            }
        }
    }
    // Store a VGPR (bits) as one float per invocation: b[gid.x] (stride 1), independent of input stride.
    void     store_output(uint32_t bits) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_f32, p, v_out, uconst(0), gidx});
        put(code, Op_Store, {p, bcf(bits)});
    }
    // EXEC-predicated store: lanes with exec=false keep the output slot's prior value. Modeled with
    // OpSelect (no control flow needed) — load old, pick new-vs-old by the per-lane exec bool, store.
    // Correct for the straight-line "conditional write / discard" pattern (v_cmpx narrows EXEC).
    void     store_output_pred(uint32_t bits, uint32_t exec_bool) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_f32, p, v_out, uconst(0), gidx});
        uint32_t old = id(); put(code, Op_Load, {t_f32, old, p});
        uint32_t sel = id(); put(code, Op_Select, {t_f32, sel, exec_bool, bcf(bits), old});
        put(code, Op_Store, {p, sel});
    }
    uint32_t btrue() { uint32_t r = id(); put(types, Op_ConstantTrue, {t_bool, r}); return r; }
    uint32_t load_push_constant(uint32_t index) {
        uint32_t pointer = id();
        put(code, Op_AccessChain,
            {t_ptr_push_u32, pointer, v_push_constants, uconst(0), uconst(index)});
        uint32_t value = id();
        put(code, Op_Load, {t_u32, value, pointer});
        return value;
    }
    // Logical AND / OR of two bools (EXEC narrowing / saveexec).
    uint32_t land(uint32_t a, uint32_t b_) { uint32_t r = id(); put(code, Op_LogicalAnd, {t_bool, r, a, b_}); return r; }
    uint32_t lor(uint32_t a, uint32_t b_)  { uint32_t r = id(); put(code, Op_LogicalOr,  {t_bool, r, a, b_}); return r; }

    void begin(uint32_t input_stride, const ShaderResourceTable* rt = nullptr,
               uint32_t local_x = 64, uint32_t local_y = 1, uint32_t local_z = 1,
               uint32_t hardware_wave_size = 64, uint32_t push_constant_dwords = 0) {
        stride = input_stride;
        local_count = local_x * local_y * local_z;
        wave_size = hardware_wave_size;
        t_void = id(); t_fn = id(); t_f32 = id(); t_u32 = id(); t_i32 = id(); t_v3u = id(); t_bool = id();
        t_v4f = id();   // vec4<float>: needed by the sampled-texture path (image_sample) in a compute shader
        uint32_t t_ptr_in_v3u = id(); v_gid = id(); v_groupid = id(); v_localid = id();
        uint32_t t_rta = id(), t_struct = id(), t_ptr_sb_struct = id();
        v_in = id(); v_out = id(); t_ptr_sb_f32 = id();
        f_main = id(); uint32_t lbl = id(); glsl = id();

        put(caps, Op_Capability, {Cap_Shader});
        { std::vector<uint32_t> o{glsl}; pstr(o, "GLSL.std.450"); putv(extimp, Op_ExtInstImport, o); }
        put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
        is_compute = true;
        exec_model = Exec_GLCompute; iface = {v_gid, v_groupid, v_localid};   // EntryPoint deferred to finish()
        // REQUIRE_FULL_SUBGROUPS constrains LocalSize X, not merely the total workgroup size. The
        // native shell never consumes Vulkan's LocalInvocationId: it reconstructs the original guest
        // x/y/z coordinates below from the exact subgroup lane order. Declare the equivalent Vulkan
        // workgroup flattened on X so 8x8 and other multidimensional guest waves satisfy that rule.
        const uint32_t vulkan_local_x = native_subgroup_size ? local_count : local_x;
        const uint32_t vulkan_local_y = native_subgroup_size ? 1u : local_y;
        const uint32_t vulkan_local_z = native_subgroup_size ? 1u : local_z;
        put(exec, Op_ExecutionMode,
            {f_main, EM_LocalSize, vulkan_local_x, vulkan_local_y, vulkan_local_z});
        put(deco, Op_Decorate, {v_gid, Dec_BuiltIn, BI_GlobalInvocationId});
        put(deco, Op_Decorate, {v_groupid, Dec_BuiltIn, BI_WorkgroupId});
        put(deco, Op_Decorate, {v_localid, Dec_BuiltIn, BI_LocalInvocationId});   // wave lane index (.x)
        put(deco, Op_Decorate, {t_rta, Dec_ArrayStride, 4});
        put(deco, Op_MemberDecorate, {t_struct, 0, Dec_Offset, 0});
        put(deco, Op_Decorate, {t_struct, Dec_Block});
        put(deco, Op_Decorate, {v_in, Dec_DescriptorSet, 0});  put(deco, Op_Decorate, {v_in, Dec_Binding, 0});
        put(deco, Op_Decorate, {v_out, Dec_DescriptorSet, 0}); put(deco, Op_Decorate, {v_out, Dec_Binding, 1});
        put(types, Op_TypeVoid, {t_void});
        put(types, Op_TypeFunction, {t_fn, t_void});
        put(types, Op_TypeFloat, {t_f32, 32});
        put(types, Op_TypeInt, {t_u32, 32, 0});
        put(types, Op_TypeInt, {t_i32, 32, 1});
        put(types, Op_TypeVector, {t_v3u, t_u32, 3});
        put(types, Op_TypeVector, {t_v4f, t_f32, 4});
        put(types, Op_TypeBool, {t_bool});
        if (push_constant_dwords) {
            const uint32_t count = id();
            put(types, Op_Constant, {t_u32, count, push_constant_dwords});
            const uint32_t array = id();
            put(types, Op_TypeArray, {array, t_u32, count});
            put(deco, Op_Decorate, {array, Dec_ArrayStride, 4});
            const uint32_t block = id();
            put(types, Op_TypeStruct, {block, array});
            put(deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
            put(deco, Op_Decorate, {block, Dec_Block});
            const uint32_t block_pointer = id();
            put(types, Op_TypePointer, {block_pointer, SC_PushConstant, block});
            t_ptr_push_u32 = id();
            put(types, Op_TypePointer, {t_ptr_push_u32, SC_PushConstant, t_u32});
            v_push_constants = id();
            put(types, Op_Variable, {block_pointer, v_push_constants, SC_PushConstant});
        }
        put(types, Op_TypePointer, {t_ptr_in_v3u, SC_Input, t_v3u});
        put(types, Op_Variable, {t_ptr_in_v3u, v_gid, SC_Input});
        put(types, Op_Variable, {t_ptr_in_v3u, v_groupid, SC_Input});
        put(types, Op_Variable, {t_ptr_in_v3u, v_localid, SC_Input});
        put(types, Op_TypeRuntimeArray, {t_rta, t_f32});
        put(types, Op_TypeStruct, {t_struct, t_rta});
        put(types, Op_TypePointer, {t_ptr_sb_struct, SC_StorageBuffer, t_struct});
        put(types, Op_Variable, {t_ptr_sb_struct, v_in, SC_StorageBuffer});
        put(types, Op_Variable, {t_ptr_sb_struct, v_out, SC_StorageBuffer});
        put(types, Op_TypePointer, {t_ptr_sb_f32, SC_StorageBuffer, t_f32});
        declare_cbufs(rt); // scalar-memory buffers, including table-assigned bindings beyond 2/3
        put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
        put(code, Op_Label, {lbl}); cur_block = lbl;
        function_var_insert = code.size();
        uint32_t ldg = id(); put(code, Op_Load, {t_v3u, ldg, v_groupid});
        for (uint32_t c = 0; c < 3; c++) {
            groupid[c] = id();
            put(code, Op_CompositeExtract, {t_u32, groupid[c], ldg, c});
        }
        if (native_subgroup_size) {
            // Vulkan does not promise that LocalInvocationIndex follows subgroup lane order.  A
            // full, exact-size subgroup does promise that SubgroupId/SubgroupLocalInvocationId
            // identify every workgroup invocation once, so assign guest coordinates in that order.
            // This makes one Vulkan subgroup exactly one consecutive RDNA wave without depending on
            // the implementation's otherwise-unspecified local-invocation mapping.
            linear_localid = ibin(Op_IAdd,
                ibin(Op_IMul, subgroup_id(), uconst(native_subgroup_size)),
                subgroup_local_id());
            localid_comp[0] = ibin(Op_UMod, linear_localid, uconst(local_x));
            const uint32_t linear_yz = ibin(Op_UDiv, linear_localid, uconst(local_x));
            localid_comp[1] = ibin(Op_UMod, linear_yz, uconst(local_y));
            localid_comp[2] = ibin(Op_UDiv, linear_yz, uconst(local_y));
            const uint32_t local_size[3] = {local_x, local_y, local_z};
            for (uint32_t c = 0; c < 3; ++c)
                globalid_comp[c] = ibin(Op_IAdd,
                    ibin(Op_IMul, groupid[c], uconst(local_size[c])), localid_comp[c]);
        } else {
            uint32_t ld = id(); put(code, Op_Load, {t_v3u, ld, v_gid});
            for (uint32_t c = 0; c < 3; c++) {
                globalid_comp[c] = id();
                put(code, Op_CompositeExtract, {t_u32, globalid_comp[c], ld, c});
            }
            uint32_t ldl = id(); put(code, Op_Load, {t_v3u, ldl, v_localid});
            for (uint32_t c = 0; c < 3; c++) {
                localid_comp[c] = id();
                put(code, Op_CompositeExtract, {t_u32, localid_comp[c], ldl, c});
            }
            // Vulkan and RDNA both linearize X fastest, then Y, then Z.
            uint32_t yz = ibin(Op_IMul, localid_comp[2], uconst(local_y));
            yz = ibin(Op_IAdd, yz, localid_comp[1]);
            linear_localid = ibin(
                Op_IAdd, ibin(Op_IMul, yz, uconst(local_x)), localid_comp[0]);
        }
        gidx = globalid_comp[0];
        localid = localid_comp[0];
    }
    // AGC thread-dimension mode can request a non-multiple of the shader's local size. Vulkan still
    // launches the final complete workgroup, so make its excess invocations branch directly to the
    // function merge instead of executing guest instructions (and, in particular, memory stores).
    void guard_invocation_extent(uint32_t threads_x, uint32_t threads_y, uint32_t threads_z) {
        uint32_t within = ucmp(Op_ULessThan, globalid_comp[0], uconst(threads_x));
        within = land(within, ucmp(Op_ULessThan, globalid_comp[1], uconst(threads_y)));
        within = land(within, ucmp(Op_ULessThan, globalid_comp[2], uconst(threads_z)));
        const uint32_t active = id();
        invocation_guard_merge = id();
        emit_selmerge(invocation_guard_merge);
        emit_condbranch(within, active, invocation_guard_merge);
        emit_label(active);
    }
    // --- Fragment-shader shell: vec4 outputs for the implemented MRT0/MRT1 exports. ---
    uint32_t t_v4f = 0;
    std::array<uint32_t, 2> v_color{};
    void begin_fragment(const ShaderResourceTable* rt = nullptr, uint32_t color_mask = 1u) {
        bool with_cbufs = rt != nullptr;
        t_void = id(); t_fn = id(); t_f32 = id(); t_u32 = id(); t_i32 = id(); t_bool = id();
        t_v4f = id(); uint32_t t_ptr_out = id();
        for (uint32_t mrt = 0; mrt < v_color.size(); ++mrt)
            if (color_mask & (1u << mrt)) v_color[mrt] = id();
        f_main = id(); uint32_t lbl = id(); glsl = id();
        put(caps, Op_Capability, {Cap_Shader});
        { std::vector<uint32_t> o{glsl}; pstr(o, "GLSL.std.450"); putv(extimp, Op_ExtInstImport, o); }
        put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
        is_fragment = true;
        desc_set = 1;   // PS resources live in descriptor set 1 (VS owns set 0) — no cross-stage binding collision
        exec_model = Exec_Fragment;
        for (uint32_t output : v_color) if (output) iface.push_back(output); // EntryPoint deferred
        put(exec, Op_ExecutionMode, {f_main, EM_OriginUpperLeft});
        for (uint32_t mrt = 0; mrt < v_color.size(); ++mrt)
            if (v_color[mrt]) put(deco, Op_Decorate, {v_color[mrt], Dec_Location, mrt});
        put(types, Op_TypeVoid, {t_void});
        put(types, Op_TypeFunction, {t_fn, t_void});
        put(types, Op_TypeFloat, {t_f32, 32});
        put(types, Op_TypeInt, {t_u32, 32, 0});
        put(types, Op_TypeInt, {t_i32, 32, 1});
        put(types, Op_TypeBool, {t_bool});
        put(types, Op_TypeVector, {t_v4f, t_f32, 4});
        put(types, Op_TypePointer, {t_ptr_out, SC_Output, t_v4f});
        for (uint32_t output : v_color)
            if (output) put(types, Op_Variable, {t_ptr_out, output, SC_Output});
        if (with_cbufs) declare_cbufs(rt);   // only when the shader has memory ops (keeps no-op renders binding-free)
        put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
        put(code, Op_Label, {lbl}); cur_block = lbl;
        function_var_insert = code.size();
    }
    // Write a vec4(r,g,b,a) (bit-operands) to the matching fragment color output.
    void export_color(uint32_t mrt, uint32_t r, uint32_t g, uint32_t bl, uint32_t a) {
        if (mrt >= v_color.size() || !v_color[mrt]) return;
        // Fragment I/O tap (PROSPER_FS_TAP): if an intermediate was snapshotted at the tapped PC, store THAT
        // as the MRT0 colour instead of the shader's real colour, so the rendered frame visualises the value.
        uint32_t v;
        if (tap_vec && mrt == 0) { v = tap_vec; }
        else { v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(r), bcf(g), bcf(bl), bcf(a)}); }
        put(code, Op_Store, {v_color[mrt], v});
    }
    // Fragment depth export (EXP target 8 = MRTZ) — a lazily declared BuiltIn FragDepth output.
    // Writing FragDepth requires ExecutionMode DepthReplacing (fixed-function Z is replaced by the
    // shader's exported value, matching the hardware's shader-Z path). Silently DROPPING the
    // target-8 export left occlusion to interpolated Z — wrong for any depth-writing shader.
    uint32_t v_fragdepth = 0;
    void export_depth(uint32_t z_bits) {
        if (!v_fragdepth) {
            uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_Output, t_f32});
            v_fragdepth = id(); put(types, Op_Variable, {t_ptr, v_fragdepth, SC_Output});
            put(deco, Op_Decorate, {v_fragdepth, Dec_BuiltIn, BI_FragDepth});
            put(exec, Op_ExecutionMode, {f_main, EM_DepthReplacing});
            iface.push_back(v_fragdepth);
        }
        put(code, Op_Store, {v_fragdepth, bcf(z_bits)});
    }

    // --- Vertex-shader shell: draw inputs + gl_Position (member 0 of a gl_PerVertex Block). ---
    uint32_t v_vid = 0, v_iid = 0, v_pos = 0, t_ptr_out_v4f = 0;
    void begin_vertex(const ShaderResourceTable* rt = nullptr) { bool with_cbufs = rt != nullptr;
        t_void = id(); t_fn = id(); t_f32 = id(); t_u32 = id(); t_i32 = id(); t_bool = id();
        t_v4f = id();
        uint32_t t_ptr_in_i32 = id(); v_vid = id(); v_iid = id();
        uint32_t t_pv = id(), t_ptr_out_pv = id(); v_pos = id(); t_ptr_out_v4f = id();
        f_main = id(); uint32_t lbl = id(); glsl = id();
        put(caps, Op_Capability, {Cap_Shader});
        { std::vector<uint32_t> o{glsl}; pstr(o, "GLSL.std.450"); putv(extimp, Op_ExtInstImport, o); }
        put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
        is_vertex = true;
        exec_model = Exec_Vertex; iface = {v_vid, v_iid, v_pos};   // EntryPoint deferred to finish()
        put(deco, Op_Decorate, {v_vid, Dec_BuiltIn, BI_VertexIndex});
        put(deco, Op_Decorate, {v_iid, Dec_BuiltIn, BI_InstanceIndex});
        put(deco, Op_MemberDecorate, {t_pv, 0, Dec_BuiltIn, BI_Position});   // gl_PerVertex.gl_Position
        put(deco, Op_Decorate, {t_pv, Dec_Block});
        // Geometry-probe: mark gl_Position for transform-feedback capture (xfb buffer 0, one vec4/vertex).
        // These decorations do not change the shader's computation; the host reads back the captured
        // clip-space positions for the probed draw. Only emitted when explicitly requested.
        if (capture_position) {
            put(caps, Op_Capability, {Cap_TransformFeedback});
            put(exec, Op_ExecutionMode, {f_main, EM_Xfb});
            put(deco, Op_MemberDecorate, {t_pv, 0, Dec_XfbBuffer, 0});
            put(deco, Op_MemberDecorate, {t_pv, 0, Dec_XfbStride, 16});
            put(deco, Op_MemberDecorate, {t_pv, 0, Dec_Offset, 0});
        }
        put(types, Op_TypeVoid, {t_void});
        put(types, Op_TypeFunction, {t_fn, t_void});
        put(types, Op_TypeFloat, {t_f32, 32});
        put(types, Op_TypeInt, {t_u32, 32, 0});
        put(types, Op_TypeInt, {t_i32, 32, 1});
        put(types, Op_TypeBool, {t_bool});
        put(types, Op_TypeVector, {t_v4f, t_f32, 4});
        put(types, Op_TypePointer, {t_ptr_in_i32, SC_Input, t_i32});
        put(types, Op_Variable, {t_ptr_in_i32, v_vid, SC_Input});
        put(types, Op_Variable, {t_ptr_in_i32, v_iid, SC_Input});
        put(types, Op_TypeStruct, {t_pv, t_v4f});
        put(types, Op_TypePointer, {t_ptr_out_pv, SC_Output, t_pv});
        put(types, Op_Variable, {t_ptr_out_pv, v_pos, SC_Output});
        put(types, Op_TypePointer, {t_ptr_out_v4f, SC_Output, t_v4f});
        if (with_cbufs) declare_cbufs(rt);   // vertex fetch (buffer_load_format_*) reads these
        put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
        put(code, Op_Label, {lbl}); cur_block = lbl;
        function_var_insert = code.size();
    }
    // Load gl_VertexIndex as raw bits (VGPR v0 for a vertex shader).
    uint32_t load_vertex_index() { uint32_t r = id(); put(code, Op_Load, {t_i32, r, v_vid}); return i2u(r); }
    uint32_t load_instance_index() { uint32_t r = id(); put(code, Op_Load, {t_i32, r, v_iid}); return i2u(r); }
    // Hardware packs consecutive vertex/instance invocations into merged guest waves. VertexIndex
    // repeats for each instance, so flatten it with the captured per-instance vertex count before
    // deriving lane and wave IDs. Standalone fixtures retain their historical vertex-only ID.
    uint32_t vertex_invocation_id() {
        const uint32_t vertex = load_vertex_index();
        if (!vertices_per_instance) return vertex;
        return ibin(Op_IAdd,
                    ibin(Op_IMul, load_instance_index(), uconst(vertices_per_instance)), vertex);
    }
    uint32_t guest_lane_id() {
        if (is_fragment) return subgroup_local_id();
        if (is_compute) return linear_localid;
        return ibin(Op_BitwiseAnd, vertex_invocation_id(), uconst(wave_size - 1));
    }
    // Write vec4(x,y,z,w) (bit-operands) to gl_Position (EXP POS0).
    void export_position(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
        // PROSPER_FORCE_W: diagnostic — force the clip-space w to 1.0. Some shaders' factored MVP
        // multiply leaves w at 0 under our (still-incomplete) descriptor decode, collapsing the
        // perspective divide; forcing w=1 reveals whether the x/y are otherwise on-screen.
        if (getenv("PROSPER_FORCE_W")) w = uconst(0x3f800000u);   // raw bits of 1.0f (bcf bitcasts to float)
        // Shader I/O tap: if an intermediate was snapshotted at PROSPER_SHADER_TAP's PC, export THAT as
        // gl_Position instead of the real clip position, so the geometry-probe capture reads it back.
        uint32_t v;
        if (tap_vec) { v = tap_vec; }
        else { v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(x), bcf(y), bcf(z), bcf(w)}); }
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_out_v4f, p, v_pos, uconst(0)});
        put(code, Op_Store, {p, v});
    }

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
    // FS: an Input vec4 at Location=attr, rasterizer-interpolated (or Flat if flat_attrs says so).
    uint32_t frag_input(uint32_t attr) {
        auto it = in_varying.find(attr); if (it != in_varying.end()) return it->second;
        if (!t_ptr_in_v4f) { t_ptr_in_v4f = id(); put(types, Op_TypePointer, {t_ptr_in_v4f, SC_Input, t_v4f}); }
        uint32_t v = id(); put(types, Op_Variable, {t_ptr_in_v4f, v, SC_Input});
        put(deco, Op_Decorate, {v, Dec_Location, attr});
        if (flat_attrs.count(attr)) put(deco, Op_Decorate, {v, Dec_Flat});   // un-interpolated (v_interp_mov)
        in_varying[attr] = v; iface.push_back(v); return v;
    }
    // Read component `chan` (0..3) of interpolated attribute `attr` -> float bits (v_interp_p2 / mov).
    uint32_t interp_read(uint32_t attr, uint32_t chan) {
        uint32_t v = frag_input(attr);
        uint32_t vec = id(); put(code, Op_Load, {t_v4f, vec, v});
        uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, vec, chan}); return bcu(e);
    }
    // AMD v_interp_mov selects one of the triangle coefficients directly: P10, P20, or P0. On the
    // portable geometry fallback each selected coefficient has a packed Flat vec4 input. The legacy
    // P0-only path remains the ordinary Flat attribute at Location=attr and needs no extra stage.
    uint32_t interp_parameter(uint32_t attr, uint32_t chan, uint32_t selector) {
        if (!fragment_interpolation || !fragment_interpolation->requires_geometry)
            return selector == 2 ? interp_read(attr, chan) : 0;
        if (attr >= fragment_interpolation->parameter_locations.size() || selector >= 3) return 0;
        const uint32_t location = fragment_interpolation->parameter_locations[attr][selector];
        if (location == FragmentInterpolationLayout::kUnusedLocation) return 0;
        auto it = in_varying.find(0x10000u | location);
        uint32_t variable = 0;
        if (it != in_varying.end()) variable = it->second;
        else {
            if (!t_ptr_in_v4f) {
                t_ptr_in_v4f = id();
                put(types, Op_TypePointer, {t_ptr_in_v4f, SC_Input, t_v4f});
            }
            variable = id(); put(types, Op_Variable, {t_ptr_in_v4f, variable, SC_Input});
            put(deco, Op_Decorate, {variable, Dec_Location, location});
            put(deco, Op_Decorate, {variable, Dec_Flat});
            in_varying[0x10000u | location] = variable; iface.push_back(variable);
        }
        uint32_t vec = id(); put(code, Op_Load, {t_v4f, vec, variable});
        uint32_t element = id(); put(code, Op_CompositeExtract, {t_f32, element, vec, chan});
        return bcu(element);
    }
    uint32_t system_interpolation_component(uint32_t field, uint32_t component) {
        if (!fragment_interpolation || !fragment_interpolation->requires_geometry || field >= 7)
            return 0;
        const uint32_t location = fragment_interpolation->system_locations[field];
        if (location == FragmentInterpolationLayout::kUnusedLocation) return 0;
        auto it = in_varying.find(0x20000u | location);
        uint32_t variable = 0;
        if (it != in_varying.end()) variable = it->second;
        else {
            if (!t_ptr_in_v4f) {
                t_ptr_in_v4f = id();
                put(types, Op_TypePointer, {t_ptr_in_v4f, SC_Input, t_v4f});
            }
            variable = id(); put(types, Op_Variable, {t_ptr_in_v4f, variable, SC_Input});
            put(deco, Op_Decorate, {variable, Dec_Location, location});
            if (field >= 4) put(deco, Op_Decorate, {variable, Dec_NoPerspective});
            if (field == 0 || field == 4) put(deco, Op_Decorate, {variable, Dec_Sample});
            if (field == 2 || field == 6) put(deco, Op_Decorate, {variable, Dec_Centroid});
            in_varying[0x20000u | location] = variable; iface.push_back(variable);
        }
        uint32_t vec = id(); put(code, Op_Load, {t_v4f, vec, variable});
        uint32_t element = id(); put(code, Op_CompositeExtract, {t_f32, element, vec, component});
        return bcu(element);
    }
    // FS: gl_FragCoord (BuiltIn 15), lazily declared — used by the DPP quad_perm lowering.
    uint32_t v_fragcoord = 0;
    uint32_t fragcoord_var() {
        if (v_fragcoord) return v_fragcoord;
        if (!t_ptr_in_v4f) { t_ptr_in_v4f = id(); put(types, Op_TypePointer, {t_ptr_in_v4f, SC_Input, t_v4f}); }
        v_fragcoord = id(); put(types, Op_Variable, {t_ptr_in_v4f, v_fragcoord, SC_Input});
        put(deco, Op_Decorate, {v_fragcoord, Dec_BuiltIn, BI_FragCoord});
        iface.push_back(v_fragcoord); return v_fragcoord;
    }
    uint32_t fragcoord_component(uint32_t component) {
        uint32_t value = id(); put(code, Op_Load, {t_v4f, value, fragcoord_var()});
        uint32_t scalar = id(); put(code, Op_CompositeExtract, {t_f32, scalar, value, component});
        return bcu(scalar);
    }
    // DPP16 quad_perm (#273 — DOLL's manual ddx/ddy in its sharpen/AA PSs): the value of `x` at quad
    // lane t = QP[my_lane] is reconstructed as x + (tx-px)·dPdx(x) + (ty-py)·dPdy(x), where (px,py) =
    // (int(gl_FragCoord.xy) & 1) is this invocation's quad position and (tx,ty) = (t&1, t>>1). Exact
    // for values linear within the 2x2 quad — the same assumption hardware derivatives make (the
    // idiom's whole purpose IS ddx/ddy: e.g. `v_sub_f32_dpp v4, v2, v2@[0,0,2,2] qp[1,1,3,3]` =
    // v2@(1,py) - v2@(0,py) = dPdxFine). Fragment-only. CONFIDENCE: MED (render-tested vs a known
    // varying gradient).
    uint32_t dpp_quad(uint32_t x_bits, uint32_t ctrl) {
        uint32_t fc = id(); put(code, Op_Load, {t_v4f, fc, fragcoord_var()});
        uint32_t fx = id(); put(code, Op_CompositeExtract, {t_f32, fx, fc, 0});
        uint32_t fy = id(); put(code, Op_CompositeExtract, {t_f32, fy, fc, 1});
        uint32_t pxu = id(); put(code, Op_ConvertFToU, {t_u32, pxu, fx});
        uint32_t pyu = id(); put(code, Op_ConvertFToU, {t_u32, pyu, fy});
        uint32_t px = ibin(Op_BitwiseAnd, pxu, uconst(1)), py = ibin(Op_BitwiseAnd, pyu, uconst(1));
        uint32_t lane = ibin(Op_IAdd, px, ibin(Op_ShiftLeftLogical, py, uconst(1)));
        uint32_t tsel = uconst((uint32_t)(ctrl & 3u));                 // QP[0] default; select QP[lane]
        for (uint32_t k = 1; k < 4; k++)
            tsel = sel(ucmp(Op_IEqual, lane, uconst(k)), uconst((ctrl >> (2 * k)) & 3u), tsel);
        uint32_t tx = ibin(Op_BitwiseAnd, tsel, uconst(1)), ty = ibin(Op_ShiftRightLogical, tsel, uconst(1));
        uint32_t xf = bcf(x_bits);
        uint32_t dx = id(); put(code, Op_DPdx, {t_f32, dx, xf});
        uint32_t dy = id(); put(code, Op_DPdy, {t_f32, dy, xf});
        // (tx-px) and (ty-py) as floats (each in {-1,0,1}).
        uint32_t dtx = id(); put(code, Op_ConvertSToF, {t_f32, dtx, bcs(ibin(Op_ISub, tx, px))});
        uint32_t dty = id(); put(code, Op_ConvertSToF, {t_f32, dty, bcs(ibin(Op_ISub, ty, py))});
        uint32_t r0 = id(); put(code, Op_FMul, {t_f32, r0, dtx, dx});
        uint32_t r1 = id(); put(code, Op_FMul, {t_f32, r1, dty, dy});
        uint32_t s0 = id(); put(code, Op_FAdd, {t_f32, s0, xf, r0});
        uint32_t s1 = id(); put(code, Op_FAdd, {t_f32, s1, s0, r1});
        return bcu(s1);
    }
    // VS: an Output vec4 at Location=loc (EXP PARAM_loc); uses t_ptr_out_v4f from begin_vertex.
    uint32_t vtx_output(uint32_t loc) {
        auto it = out_varying.find(loc); if (it != out_varying.end()) return it->second;
        uint32_t v = id(); put(types, Op_Variable, {t_ptr_out_v4f, v, SC_Output});
        put(deco, Op_Decorate, {v, Dec_Location, loc});
        out_varying[loc] = v; iface.push_back(v); return v;
    }
    void export_param(uint32_t loc, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
        uint32_t v = vtx_output(loc);
        uint32_t vec = id(); putv(code, Op_CompositeConstruct, {t_v4f, vec, bcf(x), bcf(y), bcf(z), bcf(w)});
        put(code, Op_Store, {v, vec});
    }

    // Descriptor-free geometry pass-through used when a fragment program asks for AMD's explicit
    // P0/P10/P20 vertex parameters on a Vulkan device without a barycentric/vertex-parameter
    // extension. Input assembly has already decomposed lists/strips/fans into triangles here.
    std::vector<uint32_t> build_interpolation_geometry(
            const FragmentInterpolationLayout& layout, bool capture_geometry_position) {
        if (!layout.requires_geometry || !layout.valid) return {};

        t_void = id(); t_fn = id(); t_f32 = id(); t_u32 = id(); t_i32 = id(); t_bool = id();
        t_v4f = id();
        const uint32_t t_input_per_vertex = id(), t_output_per_vertex = id();
        const uint32_t c_three = id();
        const uint32_t t_input_positions = id(), t_input_varyings = id();
        const uint32_t ptr_in_positions = id(), ptr_in_varyings = id();
        const uint32_t ptr_in_v4f = id(), ptr_out_position = id(), ptr_out_v4f = id();
        const uint32_t input_position = id(), output_position = id();
        f_main = id(); const uint32_t label = id(); glsl = id();

        std::array<uint32_t, 32> attribute_inputs{}, attribute_outputs{};
        std::array<std::array<uint32_t, 3>, 32> parameter_outputs{};
        std::array<uint32_t, 7> system_outputs{};
        for (uint32_t attr = 0; attr < 32; ++attr) {
            if (!(layout.attribute_mask & (1u << attr))) continue;
            attribute_inputs[attr] = id();
            if (layout.smooth_mask & (1u << attr)) attribute_outputs[attr] = id();
            for (uint32_t selector = 0; selector < 3; ++selector)
                if (layout.parameter_locations[attr][selector] !=
                    FragmentInterpolationLayout::kUnusedLocation)
                    parameter_outputs[attr][selector] = id();
        }
        for (uint32_t field = 0; field < 7; ++field)
            if (layout.system_locations[field] != FragmentInterpolationLayout::kUnusedLocation)
                system_outputs[field] = id();

        put(caps, Op_Capability, {Cap_Shader});
        put(caps, Op_Capability, {Cap_Geometry});
        if (capture_geometry_position) put(caps, Op_Capability, {Cap_TransformFeedback});
        { std::vector<uint32_t> operands{glsl}; pstr(operands, "GLSL.std.450");
          putv(extimp, Op_ExtInstImport, operands); }
        put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
        exec_model = Exec_Geometry;
        put(exec, Op_ExecutionMode, {f_main, EM_Triangles});
        put(exec, Op_ExecutionMode, {f_main, EM_OutputTriangleStrip});
        put(exec, Op_ExecutionMode, {f_main, EM_OutputVertices, 3});
        if (capture_geometry_position) put(exec, Op_ExecutionMode, {f_main, EM_Xfb});

        put(deco, Op_MemberDecorate, {t_input_per_vertex, 0, Dec_BuiltIn, BI_Position});
        put(deco, Op_Decorate, {t_input_per_vertex, Dec_Block});
        put(deco, Op_MemberDecorate, {t_output_per_vertex, 0, Dec_BuiltIn, BI_Position});
        put(deco, Op_Decorate, {t_output_per_vertex, Dec_Block});
        if (capture_geometry_position) {
            put(deco, Op_MemberDecorate, {t_output_per_vertex, 0, Dec_XfbBuffer, 0});
            put(deco, Op_MemberDecorate, {t_output_per_vertex, 0, Dec_XfbStride, 16});
            put(deco, Op_MemberDecorate, {t_output_per_vertex, 0, Dec_Offset, 0});
        }
        for (uint32_t attr = 0; attr < 32; ++attr) {
            if (attribute_inputs[attr]) {
                put(deco, Op_Decorate, {attribute_inputs[attr], Dec_Location, attr});
                iface.push_back(attribute_inputs[attr]);
            }
            if (attribute_outputs[attr]) {
                put(deco, Op_Decorate, {attribute_outputs[attr], Dec_Location, attr});
                iface.push_back(attribute_outputs[attr]);
            }
            for (uint32_t selector = 0; selector < 3; ++selector) {
                const uint32_t variable = parameter_outputs[attr][selector];
                if (!variable) continue;
                put(deco, Op_Decorate,
                    {variable, Dec_Location, layout.parameter_locations[attr][selector]});
                put(deco, Op_Decorate, {variable, Dec_Flat});
                iface.push_back(variable);
            }
        }
        for (uint32_t field = 0; field < 7; ++field) {
            const uint32_t variable = system_outputs[field];
            if (!variable) continue;
            put(deco, Op_Decorate, {variable, Dec_Location, layout.system_locations[field]});
            if (field >= 4) put(deco, Op_Decorate, {variable, Dec_NoPerspective});
            if (field == 0 || field == 4) put(deco, Op_Decorate, {variable, Dec_Sample});
            if (field == 2 || field == 6) put(deco, Op_Decorate, {variable, Dec_Centroid});
            iface.push_back(variable);
        }
        iface.push_back(input_position); iface.push_back(output_position);

        put(types, Op_TypeVoid, {t_void});
        put(types, Op_TypeFunction, {t_fn, t_void});
        put(types, Op_TypeFloat, {t_f32, 32});
        put(types, Op_TypeInt, {t_u32, 32, 0});
        put(types, Op_TypeInt, {t_i32, 32, 1});
        put(types, Op_TypeBool, {t_bool});
        put(types, Op_TypeVector, {t_v4f, t_f32, 4});
        put(types, Op_TypeStruct, {t_input_per_vertex, t_v4f});
        put(types, Op_TypeStruct, {t_output_per_vertex, t_v4f});
        put(types, Op_Constant, {t_u32, c_three, 3});
        put(types, Op_TypeArray, {t_input_positions, t_input_per_vertex, c_three});
        put(types, Op_TypeArray, {t_input_varyings, t_v4f, c_three});
        put(types, Op_TypePointer, {ptr_in_positions, SC_Input, t_input_positions});
        put(types, Op_TypePointer, {ptr_in_varyings, SC_Input, t_input_varyings});
        put(types, Op_TypePointer, {ptr_in_v4f, SC_Input, t_v4f});
        put(types, Op_TypePointer, {ptr_out_position, SC_Output, t_output_per_vertex});
        put(types, Op_TypePointer, {ptr_out_v4f, SC_Output, t_v4f});
        put(types, Op_Variable, {ptr_in_positions, input_position, SC_Input});
        put(types, Op_Variable, {ptr_out_position, output_position, SC_Output});
        for (uint32_t variable : attribute_inputs)
            if (variable) put(types, Op_Variable, {ptr_in_varyings, variable, SC_Input});
        for (uint32_t variable : attribute_outputs)
            if (variable) put(types, Op_Variable, {ptr_out_v4f, variable, SC_Output});
        for (const auto& selectors : parameter_outputs)
            for (uint32_t variable : selectors)
                if (variable) put(types, Op_Variable, {ptr_out_v4f, variable, SC_Output});
        for (uint32_t variable : system_outputs)
            if (variable) put(types, Op_Variable, {ptr_out_v4f, variable, SC_Output});

        put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
        put(code, Op_Label, {label}); cur_block = label;

        std::array<std::array<uint32_t, 3>, 32> attribute_values{};
        for (uint32_t attr = 0; attr < 32; ++attr) {
            if (!attribute_inputs[attr]) continue;
            for (uint32_t vertex = 0; vertex < 3; ++vertex) {
                const uint32_t pointer = id();
                put(code, Op_AccessChain,
                    {ptr_in_v4f, pointer, attribute_inputs[attr], uconst(vertex)});
                attribute_values[attr][vertex] = id();
                put(code, Op_Load, {t_v4f, attribute_values[attr][vertex], pointer});
            }
        }
        std::array<std::array<uint32_t, 3>, 32> parameters{};
        for (uint32_t attr = 0; attr < 32; ++attr) {
            if (!attribute_inputs[attr]) continue;
            parameters[attr][2] = attribute_values[attr][0];
            if (layout.passthrough_mask & (1u << attr)) {
                parameters[attr][0] = attribute_values[attr][1];
                parameters[attr][1] = attribute_values[attr][2];
            } else {
                parameters[attr][0] = id();
                put(code, Op_FSub, {t_v4f, parameters[attr][0],
                                    attribute_values[attr][1], attribute_values[attr][0]});
                parameters[attr][1] = id();
                put(code, Op_FSub, {t_v4f, parameters[attr][1],
                                    attribute_values[attr][2], attribute_values[attr][0]});
            }
        }

        for (uint32_t vertex = 0; vertex < 3; ++vertex) {
            uint32_t input_pointer = id();
            put(code, Op_AccessChain,
                {ptr_in_v4f, input_pointer, input_position, uconst(vertex), uconst(0)});
            uint32_t position = id(); put(code, Op_Load, {t_v4f, position, input_pointer});
            uint32_t output_pointer = id();
            put(code, Op_AccessChain,
                {ptr_out_v4f, output_pointer, output_position, uconst(0)});
            put(code, Op_Store, {output_pointer, position});

            for (uint32_t attr = 0; attr < 32; ++attr) {
                if (attribute_outputs[attr])
                    put(code, Op_Store,
                        {attribute_outputs[attr], attribute_values[attr][vertex]});
                for (uint32_t selector = 0; selector < 3; ++selector)
                    if (parameter_outputs[attr][selector])
                        put(code, Op_Store,
                            {parameter_outputs[attr][selector], parameters[attr][selector]});
            }
            const float i = vertex == 1 ? 1.0f : 0.0f;
            const float j = vertex == 2 ? 1.0f : 0.0f;
            const uint32_t barycentric = id();
            putv(code, Op_CompositeConstruct,
                 {t_v4f, barycentric, fconstf(i), fconstf(j), fconstf(1.0f), fconstf(1.0f)});
            for (uint32_t variable : system_outputs)
                if (variable) put(code, Op_Store, {variable, barycentric});
            put(code, Op_EmitVertex, {});
        }
        put(code, Op_EndPrimitive, {});
        return finish();
    }

    std::vector<uint32_t> finish() {
        if (invocation_guard_merge) {
            emit_branch(invocation_guard_merge);
            emit_label(invocation_guard_merge);
        }
        put(code, Op_Return, {}); put(code, Op_FunctionEnd, {});
        // EntryPoint is emitted here (not in begin_*) so lazily-declared Input/Output varyings — added
        // to `iface` as v_interp / EXP PARAM are encountered — appear in the interface list (SPIR-V 1.3).
        { std::vector<uint32_t> o{exec_model, f_main}; pstr(o, "main");
          for (uint32_t v : iface) o.push_back(v); putv(entry, Op_EntryPoint, o); }
        if (is_compute && compute_min_subgroup_size) {
            char marker[64];
            std::snprintf(marker, sizeof marker, "Prosper.ComputeSubgroupMin=%u",
                          compute_min_subgroup_size);
            std::vector<uint32_t> words;
            pstr(words, marker);
            putv(debug, Op_ModuleProcessed, words);
        }
        if (is_fragment && fragment_required_subgroup_size) {
            char marker[64];
            std::snprintf(marker, sizeof marker, "Prosper.FragmentSubgroupSize=%u",
                          fragment_required_subgroup_size);
            std::vector<uint32_t> words;
            pstr(words, marker);
            putv(debug, Op_ModuleProcessed, words);
        }
        std::vector<uint32_t> m{0x07230203u, 0x00010300u, 0u, next_id, 0u};
        for (auto* s : {&caps, &extimp, &mem, &entry, &exec, &debug, &deco, &types, &code})
            m.insert(m.end(), s->begin(), s->end());
        return m;
    }
};

}  // namespace

// Extend the narrow compiler shape
//
//   s_cbranch_scc* ELSE
//   THEN...
//   s_endpgm
// ELSE:
//   ELSE...
//   s_endpgm
//
// into the existing structured-if/else input. The first s_endpgm is represented as a synthetic
// s_branch to the second end, which is semantically exact: it skips the other terminating arm, while
// the shared SPIR-V shell may still converge solely to publish outputs/return. Keep this deliberately
// conservative: one scalar conditional, adjacent straight-line else arm, and a real terminating end.
// Immediate-end/nop-end tails remain the established early-out shape and are not rewritten.
static bool extend_terminating_if_else(const uint32_t* code, size_t dwords,
                                       std::vector<Rdna2Inst>& instructions,
                                       size_t* required_dwords = nullptr,
                                       uint32_t* synthetic_branch_pc = nullptr) {
    if (synthetic_branch_pc) *synthetic_branch_pc = UINT32_MAX;
    if (!code || instructions.empty()) return false;
    auto first_end = std::find_if(instructions.begin(), instructions.end(),
                                  [](const Rdna2Inst& in) { return in.is_end; });
    if (first_end == instructions.end()) return false;

    const Rdna2Inst* branch = nullptr;
    for (auto it = instructions.begin(); it != first_end; ++it) {
        if (it->fmt != Rdna2Format::SOPP) continue;
        if (it->opcode == 0x04 || it->opcode == 0x05) {
            if (branch) return false;
            branch = &*it;
        } else if (it->opcode >= 0x02 && it->opcode <= 0x09 && it->opcode != 0x03) {
            return false;
        }
    }
    if (!branch || branch->simm16 <= 0) return false;
    const int64_t target64 = static_cast<int64_t>(branch->pc) + branch->len_dwords + branch->simm16;
    if (target64 < 0 || static_cast<uint64_t>(target64) >= dwords) return false;
    const uint32_t target = static_cast<uint32_t>(target64);
    if (target != first_end->pc + first_end->len_dwords) return false;

    std::vector<Rdna2Inst> tail;
    const size_t tail_dwords = rdna2_walk(code + target, dwords - target, tail);
    if (tail.empty() || !tail.back().is_end) return false;
    bool has_real_else = false;
    for (auto it = tail.begin(); it != tail.end() - 1; ++it) {
        if (it->fmt == Rdna2Format::SOPP) {
            if (it->opcode == 0x00) continue; // padding is harmless but not a real else arm
            if (it->opcode >= 0x02 && it->opcode <= 0x09 && it->opcode != 0x03) return false;
        }
        has_real_else = true;
    }
    if (!has_real_else) return false;

    for (auto& in : tail) in.pc += target;
    const uint32_t merge_pc = tail.back().pc;
    const uint32_t skip_dwords = merge_pc - (first_end->pc + first_end->len_dwords);
    if (!skip_dwords || skip_dwords > static_cast<uint32_t>(INT16_MAX)) return false;
    first_end->fmt = Rdna2Format::SOPP;
    first_end->opcode = 0x02; // s_branch merge_pc: terminates the lexical then arm
    first_end->simm16 = static_cast<int32_t>(skip_dwords);
    first_end->words[0] = 0xbf820000u | (skip_dwords & 0xffffu);
    first_end->is_end = false;
    if (synthetic_branch_pc) *synthetic_branch_pc = first_end->pc;
    instructions.insert(instructions.end(), tail.begin(), tail.end());
    if (required_dwords) *required_dwords = target + tail_dwords;
    return true;
}

FragmentInterpolationLayout fragment_interpolation_layout(
        const uint32_t* code, size_t dwords,
        const PixelSystemInputMapping* system_inputs,
        const PixelInputMapping* pixel_inputs) {
    FragmentInterpolationLayout layout;
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    std::array<uint8_t, 32> selectors{};
    uint32_t highest_attribute = 0;
    bool has_attribute = false;
    for (const auto& instruction : instructions) {
        if (instruction.is_end) break;
        if (instruction.fmt != Rdna2Format::VINTRP || instruction.vintrp_attr >= 32) continue;
        const uint32_t attr = instruction.vintrp_attr;
        layout.attribute_mask |= 1u << attr;
        highest_attribute = std::max(highest_attribute, attr);
        has_attribute = true;
        if (instruction.opcode == 0 || instruction.opcode == 1)
            layout.smooth_mask |= 1u << attr;
        else if (instruction.opcode == 2 && instruction.src[0].value < 3)
            selectors[attr] |= static_cast<uint8_t>(1u << instruction.src[0].value);
    }
    if (pixel_inputs)
        layout.passthrough_mask =
            pixel_inputs->effective_passthrough_mask() & layout.attribute_mask;

    // P10/P20 have no ordinary Vulkan varying equivalent. P0 can retain the cheap Flat-input path
    // when it is the attribute's only interpolation mode; mixed P0+smooth needs the geometry copy too.
    for (uint32_t attr = 0; attr < 32; ++attr) {
        if (selectors[attr] & 0x3u) layout.requires_geometry = true;
        if ((selectors[attr] & 0x4u) && (layout.smooth_mask & (1u << attr)))
            layout.requires_geometry = true;
    }
    if (!layout.requires_geometry) return layout;

    uint32_t location = has_attribute ? highest_attribute + 1 : 0;
    for (uint32_t attr = 0; attr < 32; ++attr) {
        for (uint32_t selector = 0; selector < 3; ++selector) {
            if (!(selectors[attr] & (1u << selector))) continue;
            if (location >= 32) { layout.valid = false; return layout; }
            layout.parameter_locations[attr][selector] = location++;
        }
    }
    if (system_inputs) {
        for (uint32_t field = 0; field < 7; ++field) {
            const uint32_t bit = 1u << field;
            if (!(system_inputs->addr & bit) || !(system_inputs->ena & bit)) continue;
            if (location >= 32) { layout.valid = false; return layout; }
            layout.system_locations[field] = location++;
        }
    }
    return layout;
}

std::vector<uint32_t> recompile_interpolation_geometry(
        const FragmentInterpolationLayout& layout, bool capture_position) {
    SpirvCompute builder;
    return builder.build_interpolation_geometry(layout, capture_position);
}

// Machine state during recompilation: the VGPR and SGPR files (VGPR/SGPR number -> current SSA bits
// id) and VCC (current bool condition). VGPRs and SGPRs are separate register files; VALU/EXP source
// operands may reference either (SGPR is a valid ALU operand), so both are resolved by operand_bits.
struct RegState {
    std::unordered_map<int, uint32_t> vreg, sreg;
    int max_vgpr = 0;          // highest statically referenced VGPR in this shader
    // Immutable raw user-data words for DIRECT descriptors. They stay absent from `sreg` so
    // descriptor provenance can still distinguish driver input from a shader overwrite, while
    // scalar ALU may read their real bits (Astro copies V#.word2 into M0 as an LDS base).
    std::unordered_map<int, uint32_t> sreg_input;
    // SGPRs definitely written by shader instructions along at least one path to this point. This
    // provenance is independent of `sreg`: an unrepresentable write deliberately erases its SSA
    // value, but must still invalidate an entry-time direct descriptor stored in that register.
    std::unordered_set<int> sreg_written;
    std::unordered_map<int, uint32_t> sreg_bool;   // SGPR (pairs) holding a saved per-lane mask (bool id)
    std::unordered_map<int, bool> sreg_bool_narrowed;  // was EXEC narrowed when this mask was saved? (restores it)
    // Wave32 saves occupy exactly one physical SGPR. Track those lifetimes separately so a later
    // scalar-data write can invalidate the bool without also clobbering an unrelated neighbor.
    std::unordered_set<int> sreg_bool_b32;
    std::unordered_map<int, uint32_t> sreg_srt;    // SGPR holding a descriptor -> its user_data/SRT byte offset
                                                   // (descriptor provenance: s_load_dwordx4 tags, s_buffer_load resolves)
    uint32_t vcc = 0;
    uint32_t scc = 0;          // scalar condition code (bool); set by s_cmp_*/SCC-writing SOP2, read by s_cselect
    uint32_t exec = 0;         // per-lane execution mask (bool); v_cmpx narrows it, output store honors it
    bool exec_narrowed = false; // true once EXEC is narrowed below all-lanes-on (so VGPR writes predicate)
    // PC-relative EMBEDDED TABLES (#273): load pc -> the table's dwords, resolved by
    // detect_pcrel_tables (an s_getpc_b64-derived V# whose num_records is a known constant). The
    // shader BLOB carries the table; the recompiler folds it into a compile-time constant lookup.
    std::unordered_map<uint32_t, std::vector<uint32_t>> mubuf_pcrel_tables;
    std::unordered_map<uint32_t, std::vector<uint32_t>> smem_pcrel_tables;
    // SCALAR-SPILL VGPR (#273 — DOLL's big post PS): the compiler packs excess wave-uniform scalars
    // into one VGPR's lanes via `v_writelane_b32 vN, sX, <const lane>` and reads them back with
    // `v_readlane_b32 sY, vN, <const lane>`. Per-invocation each (vgpr, lane) slot is just a named
    // scalar: vgpr -> lane -> SSA id. A vgpr used as a spill array must never be read as ordinary
    // per-lane data — operand_bits rejects that (fail-visible), keeping the model honest.
    std::unordered_map<int, std::unordered_map<int, uint32_t>> vgpr_lane_slots;
    // EXEC/VCC/saved-mask spills use the same fixed lanes but remain bool-domain values.
    std::unordered_map<int, std::unordered_map<int, uint32_t>> vgpr_lane_mask_slots;
    // An ordinary VALU write ends a spill-array lifetime. Keep a tombstone so a later static
    // v_readlane cannot reinterpret the recycled per-lane VGPR as a generic subgroup shuffle.
    // A new v_writelane starts a fresh spill lifetime and clears the tombstone.
    std::unordered_set<int> invalidated_vgpr_lane_slots;
    // LDS ADDTID per-lane spill slots (#273 — DOLL's title post PSes): `s_movk m0, K;
    // ds_write_addtid_b32 vN` spills THIS lane's vN to LDS[M0 + offset + tid*4], and the matching
    // `ds_read_addtid_b32` reloads it. Per-invocation the slot is one value, keyed by the M0 SSA id
    // + the instruction offset (uconst is interned, so equal constant M0s share one id). A read of a
    // never-written slot rejects (fail-visible).
    std::unordered_map<uint64_t, uint32_t> lds_addtid;
};

// Predicate a just-computed VGPR write against EXEC: under a narrowed mask, inactive lanes keep their
// prior value. A no-op when EXEC is full (the straight-line common case), so nothing is perturbed.
inline void predicate_write(SpirvCompute& b, RegState& rs, int idx, uint32_t old_val) {
    if (rs.exec_narrowed) rs.vreg[idx] = b.sel(rs.exec, rs.vreg[idx], old_val);
    // A VGPR can be recycled after serving as a v_writelane scalar-spill array. Any ordinary
    // per-lane write starts a new register lifetime, so later ALU/EXP reads must see that value
    // rather than rejecting it as a stale cross-lane spill. Blasphemous 2 does exactly this after
    // an image_sample overwrites the shader's early scalar-spill v11 (#652).
    if (rs.vgpr_lane_slots.count(idx) || rs.vgpr_lane_mask_slots.count(idx))
        rs.invalidated_vgpr_lane_slots.insert(idx);
    rs.vgpr_lane_slots.erase(idx);
    rs.vgpr_lane_mask_slots.erase(idx);
}
inline uint32_t vreg_old(SpirvCompute& b, RegState& rs, int idx) {
    auto it = rs.vreg.find(idx); return it == rs.vreg.end() ? b.uconst(0) : it->second;
}

inline bool sreg_range_written(const RegState& rs, int base, uint32_t words) {
    for (uint32_t word = 0; word < words; ++word)
        if (rs.sreg_written.count(base + static_cast<int>(word))) return true;
    return false;
}

// An SRT tag describes the complete hardware descriptor, not merely its base SGPR. Accept it only
// while every word still carries the same provenance; any partial overwrite makes the descriptor
// unrepresentable even when the base word itself was untouched.
inline bool sreg_srt_range_tag(const RegState& rs, int base, uint32_t words, uint32_t& tag) {
    auto first = rs.sreg_srt.find(base);
    if (first == rs.sreg_srt.end()) return false;
    tag = first->second;
    for (uint32_t word = 1; word < words; ++word) {
        auto it = rs.sreg_srt.find(base + static_cast<int>(word));
        if (it == rs.sreg_srt.end() || it->second != tag) return false;
    }
    return true;
}

// A scalar inline integer used by a B64 mask operation is sign-extended to 64 bits.  Return the bit
// belonging to this emulated hardware lane without ever issuing an undefined >=32 SPIR-V shift.
// Astro's reduction tails use 15, 3, and 1 for the final 4/2/1 active lanes.
uint32_t inline_int_mask_bit(SpirvCompute& b, int value) {
    if (value == -1) return b.btrue();
    if (value == 0) return b.bfalse();
    // A Vulkan vertex invocation is the single guest lane retained by the NGG approximation. It is
    // lane zero, so an inline B64 mask contributes exactly its low bit; vertex modules do not declare
    // LocalInvocationIndex, and trying to use that compute-only ID emitted invalid SPIR-V ID zero.
    if (b.ngg_one_lane)
        return (static_cast<uint32_t>(value) & 1u) ? b.btrue() : b.bfalse();
    if (b.is_vertex) return 0; // an ordinary VS has no proven lane identity for a partial wave mask
    const uint32_t lane_id = b.is_fragment ? b.subgroup_local_id() : b.linear_localid;
    const uint32_t lane = b.ibin(Op_BitwiseAnd, lane_id,
                                  b.uconst(b.wave_size - 1));
    const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
    const uint32_t high = b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
    const uint32_t word = b.sel(high, b.uconst(value < 0 ? UINT32_MAX : 0u),
                                b.uconst(static_cast<uint32_t>(value)));
    return b.ucmp(Op_INotEqual,
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, word, bit), b.uconst(1)),
        b.uconst(0));
}

// V_MBCNT_HI positions its 32-bit S0 at lanes 32..63 (ISA ops 869/870: HI tests S0 bit i against
// ThreadMask[32+i]) — S0 is an independent 32-bit value, NOT the high dword of a 64-bit-extended
// operand. Lanes < 32 never contribute to the HI window, so their bit is 0.
uint32_t inline_int_mask_bit_hi(SpirvCompute& b, int value) {
    if (value == 0) return b.bfalse();
    if (b.ngg_one_lane) return b.bfalse(); // the modeled NGG lane is lane zero, never in the HI window
    if (b.is_vertex) return 0;
    const uint32_t lane_id = b.is_fragment ? b.subgroup_local_id() : b.linear_localid;
    const uint32_t lane = b.ibin(Op_BitwiseAnd, lane_id, b.uconst(b.wave_size - 1));
    const uint32_t high = b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
    const uint32_t bit  = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));   // lane-32 for lanes >= 32
    const uint32_t isset = b.ucmp(Op_INotEqual,
        b.ibin(Op_BitwiseAnd,
               b.ibin(Op_ShiftRightLogical, b.uconst(static_cast<uint32_t>(value)), bit),
               b.uconst(1)),
        b.uconst(0));
    return b.bsel(high, isset, b.bfalse());
}

// Per-invocation bit of a 64-bit mask consumed by V_MBCNT. The HI instruction names the odd SGPR
// of an aligned scalar pair (for example s7 for s[6:7]), while our bool-domain mask is keyed by the
// pair's low register. Accept the exact key first for unusual compiler allocations, then the aligned
// low half. VCC/EXEC are represented directly rather than through the scalar-data register file —
// but only in the canonical pairing (LO with the low half, HI with the high half); a cross-pairing
// (e.g. v_mbcnt_lo with exec_hi) reads OTHER lanes' bits, which the per-invocation model cannot
// represent, so it returns the 0 reject sentinel.
uint32_t mbcnt_source_bit(SpirvCompute& b, const RegState& rs, const Operand& source,
                          bool high_half) {
    if (source.value == 126 || source.value == 127)
        return (source.value == (high_half ? 127 : 126)) ? rs.exec : 0;
    if (source.value == 106 || source.value == 107)
        return (source.value == (high_half ? 107 : 106)) ? rs.vcc : 0;
    if (source.kind == OperandKind::InlineInt)
        return high_half ? inline_int_mask_bit_hi(b, source.value)
                         : inline_int_mask_bit(b, source.value);
    if (source.kind != OperandKind::SGPR) return 0;
    auto found = rs.sreg_bool.find(source.value);
    if (found != rs.sreg_bool.end()) return found->second;
    if (high_half && source.value > 0) {
        found = rs.sreg_bool.find(source.value - 1);
        if (found != rs.sreg_bool.end()) return found->second;
    }
    return 0;
}

bool sopp_is_noop(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOPP) return false;
    switch (in.opcode) {
        case 0x00:   // s_nop
        case 0x0c:   // s_waitcnt
        case 0x20:   // s_inst_prefetch
        case 0x21:   // s_clause
        case 0x22:   // s_wait_idle
            // NOTE: s_waitcnt_vscnt is NOT SOPP on gfx10 — it is SOPK opcode 0x17
            // (round-trip: llvm-mc gfx1010 encodes it 0xBBFD0000). Handled in the SOPK case.
            return true;
        default:
            return false;
    }
}

bool vopc_is_cmpx(uint32_t opcode);   // defined below (register-lifetime helpers)

// Is SGPR `R` provably DEAD at pc `target` — i.e. redefined before any read on the fall-through, so a
// write to it inside a divergent (execz) block linearized before `target` cannot be observed by later
// code? Sound/conservative: we only reason across the simple wave-uniform ALU formats whose SGPR source
// operands we can fully enumerate (at most a single reg or a 64-bit pair). At the first memory / control-
// flow / interp / export / unknown instruction (whose reads of R we can't bound — e.g. a wide T#/V#
// descriptor source) we give up and report NOT-dead. Checking value ∈ {R, R-1} covers both a 32-bit read
// of R and a 64-bit pair whose low half is R-1. (RE-TAG: divergent-block scalar liveness.)
inline bool sopk_writes_scalar_data(uint32_t opcode) {
    // GFX10 SOPK encodes both read-only compares/waits/register-mode operations and genuine SGPR
    // destinations in the same SDST field. Keep the distinction central so CFG/provenance scans do
    // not mistake s_cmpk's source register for a redefinition.
    return opcode == 0x00 || opcode == 0x02 || opcode == 0x0F ||
           opcode == 0x10 || opcode == 0x12;
}

inline bool sgpr_dead_at_merge(const std::vector<Rdna2Inst>& ins, uint32_t target, int R) {
    // Prove liveness over the actual scalar CFG, not just lexical order. UE4 commonly places another
    // forward EXECZ after an if/merge and only then overwrites the scratch SGPR/VCC value. A linear
    // scan used to reject that safe shape merely because it encountered the second branch. Explore
    // BOTH successors of every recognized conditional branch; a read on either path fails the proof,
    // while a redefinition/end terminates only that path. The visited set also bounds back-edges.
    std::unordered_map<uint32_t, size_t> pc_to_index;
    for (size_t i = 0; i < ins.size(); ++i) pc_to_index.emplace(ins[i].pc, i);
    const auto start = pc_to_index.find(target);
    if (start == pc_to_index.end()) return false;
    std::vector<size_t> pending{start->second};
    std::vector<uint8_t> visited(ins.size(), 0);
    while (!pending.empty()) {
        size_t index = pending.back();
        pending.pop_back();
        if (index >= ins.size() || visited[index]) continue;
        visited[index] = 1;
        const auto& in = ins[index];
        if (in.is_end) continue;                           // this path ends without a read
        // VCC (R = 106/107) has IMPLICIT readers the operand scan can't see: the e32 VOP2 carry /
        // cndmask ops (0x01, 0x28-0x2A) and the vccz/vccnz branches. Flag those as reads up front so
        // the redef checks below are sound for VCC too (which the s_mov carve-out excludes, but the
        // SMEM carve-out needs — compilers love `s_buffer_load_dword vcc_lo, …` scratch loads).
        if (R == 106 || R == 107) {
            if (in.fmt == Rdna2Format::VOP2 &&
                (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2A))) return false;
            if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07)) return false;
        }
        switch (in.fmt) {
            // SOPK is intentionally EXCLUDED: several SOPK ops (s_addk/s_mulk/s_cmovk/s_cmpk) READ or
            // read-modify-write their "dst" via the implicit SIMM16, but decode with n_src==0 — so the
            // read-scan below can't see the read and the dst-match would falsely report a redefinition.
            case Rdna2Format::SOPP:
                // Hint/sync SOPPs read and write nothing — scan through them (s_waitcnt is ubiquitous
                // after the merge). A barrier is also scalar-register-transparent; it changes only
                // workgroup execution order. For branches, enqueue every possible successor. Unknown
                // scalar control flow remains fail-closed.
                if (sopp_is_noop(in) || in.opcode == 0x0a) break;
                if (in.opcode == 0x02 ||
                    (in.opcode >= 0x04 && in.opcode <= 0x09)) {
                    const int64_t branch_pc = static_cast<int64_t>(in.pc) +
                        in.len_dwords + in.simm16;
                    const auto branch = branch_pc >= 0
                        ? pc_to_index.find(static_cast<uint32_t>(branch_pc)) : pc_to_index.end();
                    if (branch == pc_to_index.end()) return false;
                    pending.push_back(branch->second);
                    if (in.opcode != 0x02) {
                        if (index + 1 >= ins.size()) return false;
                        pending.push_back(index + 1);
                    }
                    continue;
                }
                return false;
            case Rdna2Format::SMEM: {
                // s_load/s_buffer_load: reads the SBASE pair (src[0], 2 regs) + SOFFSET (src[1]);
                // redefines N consecutive dst SGPRs. Anything not a plain load -> bail.
                uint32_t n = 0;
                switch (in.opcode) {
                    case 0x0: case 0x8: n = 1;  break;   case 0x1: case 0x9: n = 2;  break;
                    case 0x2: case 0xA: n = 4;  break;   case 0x3: case 0xB: n = 8;  break;
                    case 0x4: case 0xC: n = 16; break;
                    default: return false;
                }
                if (in.src[0].value == R || in.src[0].value + 1 == R) return false;         // SBASE read
                if ((in.src[1].kind == OperandKind::SGPR || in.src[1].kind == OperandKind::Special) &&
                    in.src[1].value == R) return false;                                     // SOFFSET read
                if (R >= in.dst.value && R < in.dst.value + (int)n) continue;  // redefined: this path is dead
                break;
            }
            case Rdna2Format::SOP1: case Rdna2Format::SOP2: case Rdna2Format::SOPC:
            case Rdna2Format::VOP1: case Rdna2Format::VOP2: case Rdna2Format::VOP3: case Rdna2Format::VOPC:
            case Rdna2Format::DS:    // DS operands are VGPR/M0 only; it cannot read an ordinary SGPR/VCC
            case Rdna2Format::EXP:   // EXP data sources are all VGPRs — it can never read an SGPR
                // s_bitset{0,1}_b32 reads its destination before replacing that same word; the
                // decoded source is only the bit index, so the generic operand walk cannot see it.
                if (in.fmt == Rdna2Format::SOP1 &&
                    (in.opcode == 0x1c || in.opcode == 0x1d) &&
                    in.dst.value == R)
                    return false;
                for (int k = 0; k < in.n_src; k++) {
                    if (in.src[k].kind != OperandKind::SGPR &&
                        in.src[k].kind != OperandKind::Special) continue;
                    if (in.src[k].value == R) return false;              // direct read before redef -> live
                    // Vector ALU operands are single dwords even when the scalar field names VCC_LO.
                    // Do not mistake that low-half read for a read of VCC_HI: a later VOPC can kill the
                    // complete physical pair before any real B64 consumer. Scalar ALU operands retain
                    // the conservative pair interpretation because their opcode-specific B32/B64 width
                    // is not represented in Operand. VOP3 is conservative too: cndmask's third operand
                    // can be an explicit mask pair. The ordinary VOP1/VOP2/VOPC scalar inputs here are
                    // dwords (implicit VOP2 VCC-mask readers were handled above). This distinction proves
                    // UE4's temporary VCC_HI descriptor word dead without weakening mask lifetimes.
                    const bool dword_vector_alu = in.fmt == Rdna2Format::VOP1 ||
                        in.fmt == Rdna2Format::VOP2 || in.fmt == Rdna2Format::VOPC;
                    if (!dword_vector_alu && in.src[k].value == R - 1) return false;
                }
                // A VOP3B carry-out (sdst) is a 64-bit mask write — reads none of R beyond its sources.
                if (in.sdst.kind == OperandKind::SGPR &&
                    (in.sdst.value == R || in.sdst.value + 1 == R)) continue;   // redefined (pair)
                // A non-X VOPC e32 compare writes the whole VCC pair — kills both halves. V_CMPX
                // writes EXEC ONLY on RDNA2 (ISA 12.9: "EXEC[threadId] = ..."; no VCC/SDST dest),
                // so it must NOT count as a VCC kill — miscounting it let safe_execz linearize a
                // block whose vcc scratch-write hardware would have skipped.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                    (in.dst.value == 106 || in.dst.value == 107) && (R == 106 || R == 107))
                    continue;
                // A redefinition kills R for a plain numbered SGPR dst (s0..s105) — and now also for
                // VCC_LO/HI (106/107), whose implicit readers are enumerated above. EXEC/M0
                // (124/126/127) still can't be proven dead (implicit reads everywhere). A cmpx
                // SDWAB's decoded SGPR dst is excluded for the same reason as above (EXEC only).
                if (in.dst.kind == OperandKind::SGPR && in.dst.value == R && R <= 107 &&
                    !(in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)))
                    continue;
                break;
            case Rdna2Format::MUBUF:
            case Rdna2Format::MTBUF:
                // Buffer packets fully decode their scalar inputs: SRSRC is the four-dword
                // descriptor at src[1], and SOFFSET is the single src[2] operand.  Treating every
                // buffer access as an unknowable SGPR read made an otherwise-dead VCC_HI scratch
                // word look live across Astro Bot's world-map setup pass.  Keep the proof exact by
                // checking the complete descriptor range and explicit offset before scanning on.
                if (in.src[1].kind != OperandKind::SGPR) return false;
                if (R >= in.src[1].value && R < in.src[1].value + 4) return false;
                if ((in.src[2].kind == OperandKind::SGPR ||
                     in.src[2].kind == OperandKind::Special) && in.src[2].value == R)
                    return false;
                break;
            default:
                return false;   // memory/branch/interp/unknown: can't bound reads of R -> assume live
        }
        if (index + 1 < ins.size()) pending.push_back(index + 1);
    }
    return true;   // every reachable path hit a redefinition/end without a read
}

std::unordered_set<uint32_t> safe_execz_branches(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> safe;
    // pc of the terminating s_endpgm — a forward execz whose target is here skips straight to the end.
    uint32_t end_pc = 0; bool have_end = false;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; have_end = true; break; }
    for (const auto& br : ins) {
        if (br.fmt != Rdna2Format::SOPP || br.opcode != 0x08 || br.simm16 <= 0) continue;
        const uint32_t target = br.pc + br.len_dwords + (uint32_t)br.simm16;
        bool ok = target > br.pc;
        bool target_found = false;
        for (const auto& in : ins) {
            if (in.pc == target) { target_found = true; break; }
        }
        if (!target_found) ok = false;
        // LOOP EXIT (#1183): if a backward branch that jumps to at-or-before this execz sits inside the
        // block [br, target), then this execz is a data-dependent loop's EXIT, not an if/guard-to-end —
        // even when its target happens to be s_endpgm. Linearizing it here would strand the loop's
        // back-edge as a straight-line reject; instead leave it OUT of `safe` so detect_divergent_loops
        // claims it and emit_divloop reconstructs the structured loop. Back-edges are the shapes that
        // detector recognizes: unconditional s_branch (0x02) or s_cbranch_execnz (0x09).
        bool is_loop_exit = false;
        for (const auto& bb : ins) {
            if (bb.fmt != Rdna2Format::SOPP || (bb.opcode != 0x02 && bb.opcode != 0x09)) continue;
            if (bb.pc <= br.pc || bb.pc >= target) continue;                 // must sit inside this block
            const int64_t bt = static_cast<int64_t>(bb.pc) + bb.len_dwords + bb.simm16;   // signed target
            if (bt <= static_cast<int64_t>(br.pc)) { is_loop_exit = true; break; }
        }
        if (is_loop_exit) continue;
        // Two shapes are safe to linearize (drop the branch, run the block under per-lane EXEC):
        //  * IF/ENDIF rejoining live code (target < end): only EXEC-predicated VGPR writes are safe,
        //    since scalar/VCC/memory writes past the merge would be observed by later code.
        //  * GUARD-TO-END (target == s_endpgm): the block's scalar/SGPR/VCC writes are DEAD (nothing
        //    runs after s_endpgm) and wave-uniform, memory loads are fault-free (robustBufferAccess) into
        //    predicated VGPRs, and memory stores are EXEC-predicated (conditional store). So anything is
        //    safe EXCEPT an EXP export (not EXEC-predicated → would export from inactive lanes).
        const bool guard_to_end = have_end && target >= end_pc;
        for (const auto& in : ins) {
            if (in.pc <= br.pc || in.pc >= target || in.is_end) continue;
            if (in.fmt == Rdna2Format::EXP) { ok = false; break; }   // exports are never EXEC-masked
            if (guard_to_end) continue;                              // dead-at-end / predicated / fault-free
            // IF/ENDIF rejoining live code: safe to linearize iff every write is EXEC-predicated. VGPR ALU
            // (VOP1/2/3) predicates its writes; memory ops (MIMG/MUBUF/DS) predicate their VGPR loads (and
            // are fault-free under robust access) and EXEC-predicate their stores. SGPR/VCC writes (SOP*/
            // VOPC/SMEM) are NOT predicated and would be observed past the merge -> still unsafe.
            // CARVE-OUTS: three op groups inside the "safe" VALU formats have UNPREDICATED scalar side
            // effects (emit_alu writes rs.sreg/rs.vcc/rs.sreg_bool for them, and predicate_write only
            // covers VGPRs) — on hardware the skipped block would have preserved VCC/the SGPR:
            //   VOP1 0x02 v_readfirstlane_b32 (writes an SGPR), VOP2 0x28-0x2A carry ops (write VCC),
            //   VOP3B 0x128-0x12A (write the carry-out SGPR pair/VCC), and VOP3B v_mad_u64_u32
            //   0x176 (its 65th-bit carry mask also lands in VCC/an SGPR pair unpredicated).
            const bool scalar_side_effect =
                (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02) ||
                (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2A) ||
                (in.fmt == Rdna2Format::VOP3 &&
                 ((in.opcode >= 0x128 && in.opcode <= 0x12A) || in.opcode == 0x176));
            if (!scalar_side_effect &&
                (in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 || in.fmt == Rdna2Format::VOP3 ||
                 in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::MUBUF ||
                 in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::DS ||
                 in.fmt == Rdna2Format::FLAT ||
                 sopp_is_noop(in))) {
                continue;
            }
            if (scalar_side_effect) { ok = false; break; }
            // A pure scalar move (s_mov_b32/b64: no SCC/VCC/memory side effect) whose destination SGPR is
            // DEAD at the merge is also safe to linearize: the unconditional write is overwritten before any
            // later read, so masked-off lanes never observe it. (The tonemap/sRGB divergent-ifs load a scalar
            // constant used only within the block and reset it right after — see shader 033.)
            if (in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x03 || in.opcode == 0x04) &&
                in.dst.kind == OperandKind::SGPR) {
                const bool b64 = (in.opcode == 0x04);
                // dst must be a plain SGPR (s0..s105). SOP1 destinations also decode EXEC/VCC/M0
                // (106/107/124/126/127) as SGPR-kind, but a move into those has wave-wide side effects read
                // implicitly (not via an SGPR operand) — they can never be proven dead, so exclude them.
                const int hi = in.dst.value + (b64 ? 1 : 0);
                if (hi <= 105 && sgpr_dead_at_merge(ins, target, in.dst.value) &&
                    (!b64 || sgpr_dead_at_merge(ins, target, in.dst.value + 1))) continue;
            }
            // A scalar LOAD (s_load/s_buffer_load) inside the block is likewise safe iff every SGPR it
            // writes is DEAD at the merge: the load itself is wave-uniform and fault-free, and a result
            // overwritten before any read is never observed by post-merge code. This is the divergent
            // lighting/fog block's `s_buffer_load_dwordx2 vcc, …` scratch-load shape (DOLL VS, #273);
            // sgpr_dead_at_merge understands VCC's implicit readers, so vcc-targeted loads qualify.
            if (in.fmt == Rdna2Format::SMEM) {
                uint32_t n = 0;
                switch (in.opcode) {
                    case 0x0: case 0x8: n = 1;  break;   case 0x1: case 0x9: n = 2;  break;
                    case 0x2: case 0xA: n = 4;  break;   case 0x3: case 0xB: n = 8;  break;
                    case 0x4: case 0xC: n = 16; break;   default: break;
                }
                if (n) {
                    bool all_dead = true;
                    for (uint32_t k = 0; k < n && all_dead; k++)
                        all_dead = sgpr_dead_at_merge(ins, target, in.dst.value + (int)k);
                    if (all_dead) continue;
                }
            }
            ok = false;
            break;
        }
        if (ok) safe.insert(br.pc);
    }
    return safe;
}

bool vopc_is_cmpx(uint32_t opcode);

namespace {
uint32_t scalar_write_width(const Rdna2Inst& in);
}

// FRAGMENT alpha-test / clip() discard via a SCALAR BRANCH. A per-lane condition (v_cmp -> VCC) is folded
// into a saved-EXEC survivor mask by a 64-bit wave-mask op (s_and/s_andn2_b64 sDST,sDST,vcc — which on HW
// sets SCC = "any lane survives"), then `s_cbranch_scc0 <fwd>` skips the shading when NO lane survives; the
// block then narrows EXEC (s_wqm exec, sDST) and shades, and the export lowers to an OpKill of the failed
// lanes. Per-invocation the wave early-out is a pure optimization — running the block for a lane that will
// be OpKill'd at export is harmless — so the branch is safe to LINEARIZE (drop it, run the block straight-
// line) exactly like a forward s_cbranch_execz. Recognize it by the mask op IMMEDIATELY preceding the
// branch (hints ignored). Returns the pc of each such branch. This is the shape of every Unity cutout /
// text draw; rejecting the branch dropped all of them (The Messenger's missing cutscene text, #102).
std::unordered_set<uint32_t> mask_test_branches(const std::vector<Rdna2Inst>& ins,
                                                bool allow_b32_masks = false) {
    std::unordered_set<uint32_t> out;
    std::unordered_set<uint32_t> b32_mask_writer_pcs;
    std::unordered_set<uint32_t> block_entries;
    size_t active_count = 0;
    while (active_count < ins.size() && !ins[active_count].is_end) ++active_count;
    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < active_count; ++i) index_by_pc.emplace(ins[i].pc, i);
    auto is_scalar_branch = [](const Rdna2Inst& candidate) {
        return candidate.fmt == Rdna2Format::SOPP &&
            (candidate.opcode == 0x02 ||
             (candidate.opcode >= 0x04 && candidate.opcode <= 0x09));
    };
    auto branch_target_pc = [](const Rdna2Inst& candidate) -> int64_t {
        return static_cast<int64_t>(candidate.pc) + candidate.len_dwords + candidate.simm16;
    };
    for (size_t i = 0; i < active_count; ++i) {
        if (!is_scalar_branch(ins[i])) continue;
        const int64_t target = branch_target_pc(ins[i]);
        if (target >= 0 && target <= UINT32_MAX)
            block_entries.insert(static_cast<uint32_t>(target));
    }

    // Transfer one instruction's one-word Wave32-mask provenance. The domain is a MUST fact: an
    // SGPR is present only when every path reaching this instruction proves that it contains a
    // complete Wave32 lane mask. This preserves masks through a conditional fall-through (the live
    // Astro alpha-test preamble) while intersecting away a compare performed on only one predecessor.
    auto transfer_b32_masks = [&](const Rdna2Inst& in,
                                  const std::unordered_set<int>& incoming,
                                  bool* mask_writer) {
        std::unordered_set<int> masks = incoming;
        auto tracked_mask_source = [&](const Operand& source) {
            return source.value == 126 ||
                   ((source.kind == OperandKind::SGPR ||
                     source.kind == OperandKind::Special) &&
                    incoming.contains(source.value));
        };
        bool writes_b32_mask = false;
        int mask_dst = -1;
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
            mask_dst = in.dst.kind == OperandKind::SGPR ? in.dst.value : 106;
            writes_b32_mask = true;
        } else if (in.fmt == Rdna2Format::SOP1 &&
                   (in.opcode == 0x03 || in.opcode == 0x09 ||
                    in.opcode == 0x3c || in.opcode == 0x40 || in.opcode == 0x44)) {
            writes_b32_mask = tracked_mask_source(in.src[0]) && in.dst.value != 127;
            mask_dst = in.dst.value;
        } else if (in.fmt == Rdna2Format::SOP2 &&
                   (in.opcode == 0x0a ||
                    (in.opcode >= 0x0e && in.opcode <= 0x1c &&
                     (in.opcode & 1u) == 0))) {
            const bool real_mask_source = tracked_mask_source(in.src[0]) ||
                                          tracked_mask_source(in.src[1]);
            auto representable = [&](const Operand& source) {
                return tracked_mask_source(source) || source.kind == OperandKind::InlineInt;
            };
            writes_b32_mask = real_mask_source && representable(in.src[0]) &&
                              representable(in.src[1]) && in.dst.value != 127;
            mask_dst = in.dst.value;
        } else if (in.fmt == Rdna2Format::VOP3 &&
                   in.opcode >= 0x128 && in.opcode <= 0x12a &&
                   in.sdst.kind == OperandKind::SGPR) {
            writes_b32_mask = tracked_mask_source(in.src[2]);
            mask_dst = in.sdst.value;
        }

        auto erase_written_words = [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                masks.erase(base + static_cast<int>(word));
        };
        const uint32_t dst_width = scalar_write_width(in);
        if (dst_width) erase_written_words(in.dst.value, dst_width);
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)
            erase_written_words(in.dst.value, 1);
        if (in.fmt == Rdna2Format::VOP3 && in.sdst.kind == OperandKind::SGPR)
            erase_written_words(in.sdst.value,
                in.opcode >= 0x128 && in.opcode <= 0x12a ? 1u : 2u);
        if (writes_b32_mask && mask_dst >= 0 && mask_dst != 126)
            masks.insert(mask_dst);
        if (mask_writer) *mask_writer = writes_b32_mask && mask_dst >= 0 && mask_dst != 126;
        return masks;
    };

    if (allow_b32_masks && active_count) {
        std::vector<std::unordered_set<int>> incoming(active_count);
        std::vector<bool> reachable(active_count, false);
        std::vector<size_t> worklist{0};
        reachable[0] = true;
        auto merge_into = [&](size_t successor, const std::unordered_set<int>& masks) {
            if (!reachable[successor]) {
                incoming[successor] = masks;
                reachable[successor] = true;
                worklist.push_back(successor);
                return;
            }
            std::unordered_set<int> intersection;
            for (int reg : incoming[successor])
                if (masks.contains(reg)) intersection.insert(reg);
            if (intersection != incoming[successor]) {
                incoming[successor] = std::move(intersection);
                worklist.push_back(successor);
            }
        };
        for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
            const size_t i = worklist[cursor];
            const auto outgoing = transfer_b32_masks(ins[i], incoming[i], nullptr);
            auto merge_pc = [&](int64_t pc) {
                if (pc < 0 || pc > UINT32_MAX) return;
                const auto found = index_by_pc.find(static_cast<uint32_t>(pc));
                if (found != index_by_pc.end()) merge_into(found->second, outgoing);
            };
            if (is_scalar_branch(ins[i])) {
                merge_pc(branch_target_pc(ins[i]));
                if (ins[i].opcode != 0x02 && i + 1 < active_count)
                    merge_into(i + 1, outgoing);
            } else if (i + 1 < active_count) {
                merge_into(i + 1, outgoing);
            }
        }
        for (size_t i = 0; i < active_count; ++i) {
            if (!reachable[i]) continue;
            bool writes_mask = false;
            (void)transfer_b32_masks(ins[i], incoming[i], &writes_mask);
            if (writes_mask) b32_mask_writer_pcs.insert(ins[i].pc);
        }
    }

    const Rdna2Inst* prev = nullptr;
    for (const auto& in : ins) {
        if (block_entries.contains(in.pc)) prev = nullptr;
        if (in.is_end) break;
        if (sopp_is_noop(in)) continue;                         // hints don't break the mask->branch pairing
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04 || in.opcode == 0x05) && in.simm16 > 0) {
            // scc0/scc1 FORWARD branch whose SCC was set by a 64-bit wave-mask op: SOP2 s_and_b64(0x0f) /
            // s_or_b64(0x11) / s_xor_b64(0x13) / s_andn2_b64(0x15), or SOP1 s_and/or_saveexec_b64 (0x24/0x25),
            // writing a plain SGPR-pair kill mask. (A branch on a v_cmp/s_cmp SCC is a REAL uniform-if and is
            // NOT matched — prev would be a SOPC/ALU, not a mask op.)
            if (prev) {
                const bool b64_mask_sop2 = prev->fmt == Rdna2Format::SOP2 &&
                    (prev->opcode == 0x0f || prev->opcode == 0x11 ||
                     prev->opcode == 0x13 || prev->opcode == 0x15);
                // The same compiler alpha-test shape uses the one-word logical family when
                // SPI_PS_IN_CONTROL proves Wave32. The B32 result is then the complete wave mask,
                // and SCC is the same whole-wave early-out vote as for the B64 form.
                const bool b32_mask_sop2 = allow_b32_masks && prev->fmt == Rdna2Format::SOP2 &&
                    (prev->opcode == 0x0e || prev->opcode == 0x10 ||
                     prev->opcode == 0x12 || prev->opcode == 0x14) &&
                    b32_mask_writer_pcs.contains(prev->pc);
                const bool mask_sop2 = b64_mask_sop2 || b32_mask_sop2;
                bool mask_saveexec = prev->fmt == Rdna2Format::SOP1 &&
                                     (prev->opcode == 0x24 || prev->opcode == 0x25 ||
                                      prev->opcode == 0x37);
                // The kill mask may live in a plain SGPR pair (s0..s105) OR in VCC itself — DOLL's
                // alpha-cull PS does `s_andn2_b64 vcc, exec, vcc; s_cbranch_scc0 <null-export>` then
                // `s_mov_b64 exec, vcc; export`. The SOP2 dst field decodes VCC_LO as SGPR 106, and
                // emit_alu's mask ops route a 106/107 dst to rs.vcc, so the same linearization holds
                // (the branch is a whole-wave early-out; per-invocation the export's OpKill covers it).
                if ((mask_sop2 || mask_saveexec) && prev->dst.kind == OperandKind::SGPR && prev->dst.value <= 106)
                    out.insert(in.pc);
            }
        }

        prev = is_scalar_branch(in) ? nullptr : &in;
    }
    return out;
}

// Some GFX10 pixel shaders leave scheduled 64-bit mask operations whose VCC/SCC results feed only
// other dead mask operations and are overwritten before an observable read. Astro Bot's SSAO shader does this with
// `s_and_b64 vcc, s[0:1], vcc`, where s[0:1] is also a live T# descriptor; attempting to reinterpret
// the descriptor bits as a per-lane mask is both impossible in descriptor-backed SPIR-V and pointless.
// Elide only the mechanically proven dead form: the shader has no SCC consumer anywhere, and CFG
// liveness proves the VCC pair cannot reach a non-mask read before redefinition. This deliberately
// does not become a general scalar-pair-to-wave-mask fallback.
std::unordered_set<uint32_t> dead_wave_mask_writes(const std::vector<Rdna2Inst>& ins) {
    for (const auto& in : ins) {
        const bool reads_scc =
            (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04 || in.opcode == 0x05)) ||
            (in.fmt == Rdna2Format::SOP2 &&
             (in.opcode == 0x04 || in.opcode == 0x05 ||
              in.opcode == 0x0a || in.opcode == 0x0b)) ||
            (in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x05 || in.opcode == 0x06)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x02);
        if (reads_scc) return {};
    }
    auto is_mask = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOP2 &&
            (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 ||
             in.opcode == 0x15 || in.opcode == 0x17 || in.opcode == 0x19 ||
             in.opcode == 0x1b || in.opcode == 0x1d) &&
            (in.dst.value == 106 || in.dst.value == 107);
    };
    const auto is_cmpx = [](uint32_t op) {
        return (op >= 0x10 && op <= 0x1f) || (op >= 0x90 && op <= 0x9f) ||
               (op >= 0xd0 && op <= 0xdf);
    };
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i) by_pc[ins[i].pc] = i;
    std::vector<std::vector<size_t>> succ(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end) continue;
        const bool branch = in.fmt == Rdna2Format::SOPP &&
                            in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03;
        if (!branch || in.opcode != 0x02) {
            if (i + 1 < ins.size()) succ[i].push_back(i + 1);
        }
        if (branch) {
            const uint32_t target = in.pc + in.len_dwords +
                                    static_cast<uint32_t>(static_cast<int32_t>(in.simm16));
            auto it = by_pc.find(target);
            if (it == by_pc.end()) return {}; // malformed/unbounded CFG: make no dead-write claim
            succ[i].push_back(it->second);
        }
    }
    auto uses = [&](const Rdna2Inst& in) -> uint8_t {
        uint8_t bits = 0;
        for (uint8_t k = 0; k < in.n_src; ++k) {
            if (in.src[k].kind != OperandKind::SGPR &&
                in.src[k].kind != OperandKind::Special) continue;
            if (in.src[k].value == 106) bits |= 1;
            if (in.src[k].value == 107) bits |= 2;
        }
        if (is_mask(in) && bits) bits = 3; // every modeled mask logical reads a full B64 pair
        if (in.fmt == Rdna2Format::VOP2 &&
            (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a))) bits |= 3;
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07)) bits |= 3;
        return bits;
    };
    auto defs = [&](const Rdna2Inst& in) -> uint8_t {
        if (is_mask(in)) return 3;
        if (in.fmt == Rdna2Format::VOPC && !is_cmpx(in.opcode) &&
            (in.dst.value == 106 || in.dst.value == 107)) return 3;
        if (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2a)
            return 3;
        if (in.fmt == Rdna2Format::VOP3 &&
            (in.sdst.value == 106 || in.sdst.value == 107)) return 3;
        if (in.fmt == Rdna2Format::SOP1) {
            if (in.dst.value == 106) return in.opcode == 0x04 ? 3 : 1;
            if (in.dst.value == 107) return 2;
        }
        if (in.fmt == Rdna2Format::SMEM) {
            uint32_t n = 0;
            switch (in.opcode) {
                case 0x0: case 0x8: n=1; break; case 0x1: case 0x9: n=2; break;
                case 0x2: case 0xa: n=4; break; case 0x3: case 0xb: n=8; break;
                case 0x4: case 0xc: n=16; break; default: break;
            }
            uint8_t bits = 0;
            if (n && in.dst.value <= 106 && 106 < in.dst.value + static_cast<int>(n)) bits |= 1;
            if (n && in.dst.value <= 107 && 107 < in.dst.value + static_cast<int>(n)) bits |= 2;
            return bits;
        }
        return 0;
    };
    // Least-fixed-point dataflow rooted only in observable (non-candidate) VCC reads. A mask
    // candidate propagates liveness to its B64 input only when its output is itself live; this also
    // removes dead self-dependent mask chains inside loops without mistaking the cycle for a use.
    std::vector<uint8_t> live_in(ins.size(), 0), live_out(ins.size(), 0);
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t ri = ins.size(); ri-- > 0;) {
            uint8_t out = 0;
            for (size_t s : succ[ri]) out |= live_in[s];
            const uint8_t def = defs(ins[ri]), use = uses(ins[ri]);
            const uint8_t propagated_use = is_mask(ins[ri]) && !(out & def) ? 0 : use;
            const uint8_t in = propagated_use | (out & static_cast<uint8_t>(~def));
            if (out != live_out[ri] || in != live_in[ri]) {
                live_out[ri] = out; live_in[ri] = in; changed = true;
            }
        }
    }
    std::unordered_set<uint32_t> dead;
    for (size_t i = 0; i < ins.size(); ++i)
        if (is_mask(ins[i]) && !(live_out[i] & 3)) dead.insert(ins[i].pc);
    return dead;
}

// WATERFALL loops (#273 — DOLL's skinned scene VS): the readfirstlane-uniformize idiom
//   L: v_readfirstlane s4, vIDX ; v_cmpx_eq_u32 s4, vIDX ; s_mov m0, s4 ; v_movrels …
//      s_andn2_b64 REMAIN, REMAIN, exec ; s_mov_b64 exec, REMAIN ; s_cbranch_scc1 L
// iterates once per DISTINCT per-lane index, with EXEC selecting the matching lanes. In the
// per-invocation model each lane's iteration IS its own: readfirstlane returns MY value, the cmpx
// keeps me active, and the andn2 leaves MY remaining-mask empty — so the loop body runs EXACTLY
// ONCE per invocation and the backward branch (whose SCC = "any lane remains", a cross-lane
// reduction) is never taken. A BACKWARD s_cbranch_scc0/scc1 whose SCC was produced by a 64-bit
// wave-mask op is therefore safe to DROP (linearize), exactly like the forward kill-mask branches.
// The SCC producer may sit a few instructions back (the shape interposes `s_mov_b64 exec, REMAIN`,
// which does not write SCC) — walk back over non-SCC-writing instructions to find it.
// CONFIDENCE: MED — exec-diff kernel T19 locks the once-through semantics.
std::unordered_set<uint32_t> waterfall_branches(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> out;
    for (size_t i = 0; i < ins.size(); i++) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || (in.opcode != 0x04 && in.opcode != 0x05) || in.simm16 >= 0)
            continue;
        // Walk back to the most recent SCC-writing instruction, skipping ops known not to touch SCC
        // (scalar moves, all vector ALU incl. VOPC — it writes VCC/EXEC, not SCC — memory, hints).
        for (size_t j = i; j-- > 0;) {
            const Rdna2Inst& p = ins[j];
            bool skip_ok =
                (p.fmt == Rdna2Format::SOP1 && (p.opcode == 0x03 || p.opcode == 0x04)) ||   // s_mov_b32/b64
                p.fmt == Rdna2Format::VOP1 || p.fmt == Rdna2Format::VOP2 || p.fmt == Rdna2Format::VOP3 ||
                p.fmt == Rdna2Format::VOPC || p.fmt == Rdna2Format::MIMG ||
                p.fmt == Rdna2Format::MUBUF || p.fmt == Rdna2Format::MTBUF ||
                p.fmt == Rdna2Format::FLAT ||
                sopp_is_noop(p);
            if (skip_ok) continue;
            // The SCC producer: a 64-bit wave-mask logical op (s_and/or/xor/andn2_b64, SCC = result!=0).
            if (p.fmt == Rdna2Format::SOP2 &&
                (p.opcode == 0x0f || p.opcode == 0x11 || p.opcode == 0x13 || p.opcode == 0x15) &&
                p.dst.kind == OperandKind::SGPR && p.dst.value <= 106)
                out.insert(in.pc);
            break;   // any other SCC producer (s_cmp etc.): a REAL loop condition — not a waterfall
        }
    }
    return out;
}

// A recognized COUNTED loop (the game's MSAA-resolve / accumulation shape). Two accepted shapes:
//   (a) TOP-tested: a single backward UNCONDITIONAL s_branch (the back-edge) to a header, with
//       exactly one forward s_cbranch_scc0/scc1 exit inside the body — `for (...) {...}`.
//   (b) BOTTOM-tested (do-while): the back-edge is ITSELF a backward s_cbranch_scc0/scc1; the loop
//       continues while its SCC condition holds and exits by falling through — `do {...} while(...)`.
//       This is what the compiler emits for a fixed-trip texture-accumulation loop whose whole body
//       (including the s_cmp that sets the exit SCC) runs before the test. Alex Kidd's (PPSA02664)
//       light/blur-accumulation pixel shaders use it, and rejecting it dropped their draws (#320).
// Both are lowered by ONE emitter: shape (b) is modeled as exit_branch_pc == backedge_pc, so the
// emitter emits the entire body as the "condition region" (runs every iteration incl. the exiting
// one — exactly do-while), tests SCC at the bottom, and its "body region" (exit_branch+1..backedge)
// is empty. Anything more complex (nested loops, VCC/EXECNZ exits, multiple back-edges, mid-loop
// s_branch) is rejected — the recompiler then falls back to the straight-line path (as before).
struct CountedLoop {
    bool found = false;
    uint32_t header_pc = 0;       // loop header (target of the back-edge; condition eval starts here)
    uint32_t exit_branch_pc = 0;  // the s_cbranch_scc0/scc1 exit test (== backedge_pc for a do-while)
    uint32_t backedge_pc = 0;     // the backward branch (unconditional s_branch, or the scc back-edge)
    uint32_t exit_pc = 0;         // first pc after the loop (== backedge_pc + its length)
    bool exit_on_scc0 = true;     // exit-test polarity: true ⇔ the loop continues while SCC==1 (exits on SCC==0)
};
inline uint32_t branch_target(const Rdna2Inst& in) { return in.pc + in.len_dwords + (uint32_t)(int32_t)in.simm16; }

CountedLoop detect_counted_loop(const std::vector<Rdna2Inst>& ins) {
    CountedLoop L;
    // The single backward branch that closes the loop. A conditional scc0/scc1 back-edge is a
    // bottom-tested do-while; an unconditional s_branch is a top-tested loop.
    // A backward s_cbranch_scc0/scc1 whose SCC is a 64-bit wave-mask reduction is a WATERFALL
    // (once-through per invocation), NOT a loop back-edge — its cross-lane SCC has no representable
    // per-lane value, and every emit path already consumes it via safe_branches/the linearizer.
    // Waterfalls are excluded from the back-edge count entirely: SELECTING one would feed the loop
    // emitter a poisoned SCC exit and mis-lower the divergent body (T19), and merely COUNTING one
    // would reject a counted loop that legitimately coexists with a waterfall elsewhere in the same
    // shader — a previously-recompilable combination (kernel 44d locks it).
    const std::unordered_set<uint32_t> wf = waterfall_branches(ins);
    const Rdna2Inst* back = nullptr; int nback = 0;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || in.simm16 >= 0) continue;
        if (in.opcode == 0x02 ||
            ((in.opcode == 0x04 || in.opcode == 0x05) && !wf.count(in.pc))) { back = &in; nback++; }
    }
    if (nback != 1) return L;                       // 0 -> no loop; >1 -> nested/multiple (unhandled)
    const bool do_while = (back->opcode != 0x02);   // conditional back-edge => bottom-tested
    const uint32_t header = branch_target(*back);
    bool header_ok = false;
    for (const auto& in : ins) if (in.pc == header && in.pc < back->pc) { header_ok = true; break; }
    if (!header_ok) return L;
    const uint32_t exit_pc = back->pc + back->len_dwords;
    const Rdna2Inst* exitbr = nullptr; int nexit = 0;
    for (const auto& in : ins) {                    // scan the loop body [header, back-edge]
        if (in.pc < header || in.pc > back->pc || in.fmt != Rdna2Format::SOPP) continue;
        if (&in == back) continue;                  // the back-edge is classified above, not here
        switch (in.opcode) {
            case 0x04: case 0x05:                   // s_cbranch_scc0 / scc1 — must be the single exit
                if (branch_target(in) != exit_pc) return L;
                exitbr = &in; nexit++; break;
            case 0x02:                              // any other s_branch -> reject
                return L;
            case 0x06: case 0x07: case 0x09:        // vcc / execnz branches -> reject
                return L;
            case 0x08: default: break;              // execz (forward, predication-handled) / hints ok
        }
    }
    if (do_while) {
        // Bottom-tested: the conditional back-edge IS the exit test; no separate forward exit is
        // allowed (a do-while with an inner uniform-if stays unsupported and rejects, conservatively).
        // The back-edge is TAKEN (loops) while its SCC condition holds. exit_on_scc0 drives the
        // emitter's loop_cond = (exit_on_scc0 ? SCC : !SCC), i.e. "continue when this is true": an
        // s_cbranch_scc1 back-edge continues while SCC==1 (exit_on_scc0=true); scc0 continues while
        // SCC==0 (exit_on_scc0=false).
        if (nexit != 0) return L;
        L.found = true; L.header_pc = header; L.exit_branch_pc = back->pc; L.backedge_pc = back->pc;
        L.exit_pc = exit_pc; L.exit_on_scc0 = (back->opcode == 0x05);
        return L;
    }
    if (nexit != 1) return L;
    L.found = true; L.header_pc = header; L.exit_branch_pc = exitbr->pc; L.backedge_pc = back->pc;
    L.exit_pc = exit_pc; L.exit_on_scc0 = (exitbr->opcode == 0x04);
    return L;
}

// DIVERGENT EXEC/VCC-EXIT LOOPS (#273/#615 — post-process and light-accumulation PSes, the title-composite
// content producers). The compiled shape:
//     header: <exec recompute: v_cmpx_.. counter,bound  |  v_cmp..;s_andn2_b64 exec,exec,vcc>
//             s_cbranch_execz EXIT                      ; leave when no lane remains
//     body:   ... (nested forward-execz if regions, saveexec/restore, scalar counter++) ...
//             s_branch header                           ; backward unconditional back-edge
//     EXIT:   s_mov_b64 exec, <saved>
// Fragment semantics are exact: a forced 64-lane native subgroup vote keeps the complete guest wave
// in the loop until every EXEC/VCC bit clears. Cleared lanes remain synchronization participants while
// EXEC predicates their vector writes, so scalar state advances exactly as on RDNA2. Vertex and the
// narrow structured-compute path retain the older per-invocation lowering and its uniformity guards.
//
// A second flavor (DOLL's scalar-indexed unroll): the back-edge is a backward s_cbranch_EXECNZ and
// the header IS the execz exit (empty condition region); the body may carry an extra forward
// vccz/execz BREAK to the exit. A break lowers as a plain forward IF over the remainder of the body
// (skip-to-backedge): the broken lane's EXEC bit is already clear (the compiled break condition
// mirrors the exec recompute), so the next header check exits it. Fragment wave64 additionally
// supports a VCCZ break with an unconditional back-edge by carrying the uniform continue vote to the
// loop latch and branching the complete wave directly to the loop merge.
struct DivLoop {
    enum class Condition : uint8_t { Exec, Vcc };
    uint32_t header_pc = 0;        // back-edge target; condition region = [header_pc, exit_branch_pc)
    uint32_t exit_branch_pc = 0;   // canonical forward execz/vccz branch whose target is exit_pc
    uint32_t backedge_pc = 0;      // backward s_branch (unconditional) or s_cbranch_execnz
    uint32_t exit_pc = 0;          // backedge_pc + its length (first pc after the loop)
    std::vector<uint32_t> break_pcs;   // extra forward vccz/execz -> exit_pc (lowered as body ifs)
    bool direct_exec_breaks = false;   // unconditional back-edge: an interior execz exits directly
    bool direct_wave_breaks = false;   // fragment wave64: an interior vccz exits the complete wave
    Condition condition = Condition::Exec;
};

// Number of consecutive VGPRs an instruction writes, starting at dst. This is intentionally an
// over-approximation for memory operations: treating a store's VDATA as clobbered can only make the
// uniformity proof below reject, while missing a multi-register load could make it accept stale data.
uint32_t vgpr_write_count(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::VGPR) return 0;
    switch (in.fmt) {
        case Rdna2Format::VOP1:
            return in.opcode == 0x02 ? 0 : 1;                   // v_readfirstlane writes an SGPR
        case Rdna2Format::VOP2: case Rdna2Format::VOP3P:
            return 1;
        case Rdna2Format::VOP3:
            // 64-bit VGPR results: v_mad_u64_u32 (0x176), v_mad_i64_i32 (0x177), v_div_scale_f64
            // (0x16E) — ISA opcodes 374/375/366 (the latter two are unimplemented in emit_alu but
            // must count correctly the day they land).
            return (in.opcode == 0x176 || in.opcode == 0x177 || in.opcode == 0x16E) ? 2 : 1;
        case Rdna2Format::DS:
            // Keep in sync with the DS opcodes the emitter accepts: the rtn atomics ds_add_rtn_u32
            // (0x20) / ds_wrxchg_rtn_b32 (0x2d) write VDST, ds_read_b96/b128 (0xfe/0xff) write 3/4
            // VGPRs — returning 0 for them hid their definitions from loop-header phi collection
            // and from the #615 uniformity proof's defining-write scan.
            if (in.opcode == 0x35 || in.opcode == 0x36 || in.opcode == 0x3d || in.opcode == 0x3e || in.opcode == 0xb1 ||
                in.opcode == 0x20 || in.opcode == 0x2d) return 1;
            if (in.opcode == 0x37 || in.opcode == 0x76) return 2;
            if (in.opcode == 0xfe) return 3;
            if (in.opcode == 0x77 || in.opcode == 0xff) return 4;
            return 0;
        case Rdna2Format::MUBUF:
            switch (in.opcode) {
                case 0x1: case 0x5: case 0xD: return 2;
                case 0x2: case 0x6: case 0xF: return 3;
                case 0x3: case 0x7: case 0xE: return 4;
                default: return 1;
            }
        case Rdna2Format::MTBUF:
            if (in.opcode >= 4u) return 0; // stores read VDATA; they do not write it
            return in.opcode + 1u;
        case Rdna2Format::FLAT: {
            const FlatAccessInfo access = flat_access_info(in.opcode);
            return access.valid && !access.store ? access.components : 0;
        }
        case Rdna2Format::MIMG: {
            if (in.opcode == 0x47 || in.opcode == 0x57) return 4;  // gather4 always returns four texels
            uint32_t n = 0;
            for (uint32_t c = 0; c < 4; ++c) n += (in.mimg_dmask >> c) & 1u;
            return n ? n : 4;                                     // unknown/empty mask: reject conservatively
        }
        case Rdna2Format::VINTRP:
            return 1;   // v_interp_p1/p2/mov write VDST (per-lane interpolated data — inherently divergent)
        default:
            return 0;
    }
}

bool vopc_is_cmpx(uint32_t opcode) {
    return (opcode >= 0x10 && opcode <= 0x1f) ||
           (opcode >= 0x90 && opcode <= 0x9f) ||
           (opcode >= 0xd0 && opcode <= 0xdf);
}

bool instruction_may_change_exec(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)) return true;
    // saveexec writes EXEC implicitly while its explicit destination receives the previous mask.
    if (in.fmt == Rdna2Format::SOP1 &&
        ((in.opcode >= 0x24 && in.opcode <= 0x2b) ||
         in.opcode == 0x37 || in.opcode == 0x38 ||
         in.opcode == 0x3c || in.opcode == 0x40 || in.opcode == 0x44))
        return true;
    auto is_exec = [](const Operand& operand) {
        return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
               (operand.value == 126 || operand.value == 127);
    };
    return is_exec(in.dst) || is_exec(in.sdst);
}

// A scalar VCCZ branch tests whether ANY active lane set VCC, not this lane's bit. Stages without
// fragment's forced wave64 vote may lower it as structured control only when the VCC producer is
// provably uniform across the wave. Dead Cells' light loops use the narrow compiler shape
//   v_cvt_i32_f32 vBOUND, sBOUND; ...; v_cmp_* vcc, sCOUNTER, vBOUND; s_cbranch_vccz EXIT
// so every active lane makes the same comparison. Unity also derives the bound through a short
// lane-local VALU chain fed only by scalar sources. Trace that chain backwards: uniform inputs stay
// uniform through ordinary VOP1/VOP2 arithmetic, while lane permutations, VCC-dependent selects,
// carry chains, interpolants, and unresolved VGPR inputs remain rejected.
// True when `in` provably cannot overwrite VCC — used by vcc_exit_is_wave_uniform's compare-finding
// walk (#590) to look past instructions scheduled between the VOPC and its consuming branch (real
// UE4 compute kernels hoist ~15 unrelated ALU ops in between). Anything not provably safe stops the
// walk conservatively: VOPC itself (the implicit dst), the VOP2 carry-chain ops (v_*_co_ci_u32,
// 0x28-0x2a), VOP3 windows that may carry a scalar dest (VOPC-as-VOP3 sdst at 0x000-0x13f, VOP3B
// carry-out / v_readlane at 0x300+ — the decoder does not expose their sdst), and any scalar or
// SMEM write whose dest range could cover SGPR 106/107 (the shaders use s106/s107 as scratch).
static bool cannot_write_vcc(const Rdna2Inst& in) {
    auto sgpr_dst_misses_vcc = [&](uint32_t n_dwords) {
        if (in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) return true;
        const int lo = in.dst.value, hi = in.dst.value + (int)n_dwords - 1;
        return hi < 106 || lo > 107;
    };
    switch (in.fmt) {
        case Rdna2Format::VOP1:
        case Rdna2Format::VINTRP:
        case Rdna2Format::MUBUF: case Rdna2Format::MTBUF: case Rdna2Format::MIMG:
        case Rdna2Format::DS:    case Rdna2Format::FLAT:
        case Rdna2Format::EXP:                                     // VGPR/memory dsts only
        case Rdna2Format::SOPP:                                    // branches/hints/waitcnt: no dst
        case Rdna2Format::SOPC:  return true;                      // writes SCC only
        case Rdna2Format::VOP2:  return in.opcode < 0x28 || in.opcode > 0x2a;
        case Rdna2Format::VOP3P: return true;                      // packed VALU: VGPR dst, no carry
        case Rdna2Format::VOP3:
            // gfx10 v_writelane_b32 writes only its VGPR lane slot. v_readlane_b32 writes the
            // scalar destination encoded in the decoder's dst field, so it is VCC-preserving as
            // long as that destination does not overlap s106:s107. UE4 schedules long scalar-spill
            // sequences of these two ops between a uniform VOPC and its s_cbranch_vccz.
            if (in.opcode == 0x361) return true;
            if (in.opcode == 0x360) return sgpr_dst_misses_vcc(1);
            if (in.opcode == 0x176)
                return in.sdst.value != 106 && in.sdst.value != 107; // v_mad_u64_u32 carry-out
            // The remaining VOP3B ops inside the plain-VALU window — v_div_scale_f32/f64 (0x16D/
            // 0x16E) and v_mad_i64_i32 (0x177) — write a scalar carry/flag SDST that may be VCC;
            // their VGPR dst would otherwise satisfy sgpr_dst_misses_vcc. Stop the walk for them.
            if (in.opcode == 0x16D || in.opcode == 0x16E || in.opcode == 0x177) return false;
            return in.opcode >= 0x140 && in.opcode < 0x300 &&
                   sgpr_dst_misses_vcc(1);                         // plain-VALU window only
        case Rdna2Format::SOP1: case Rdna2Format::SOP2: case Rdna2Format::SOPK:
            return sgpr_dst_misses_vcc(2);                         // conservative 64-bit pair
        case Rdna2Format::SMEM: {
            uint32_t n = 0;
            switch (in.opcode & 7) { case 0: n = 1; break; case 1: n = 2; break; case 2: n = 4; break;
                                     case 3: n = 8; break; case 4: n = 16; break; default: return false; }
            return sgpr_dst_misses_vcc(n);
        }
        default: return false;                                     // VOPC / unknown: stops the walk
    }
}

bool vcc_exit_is_wave_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
    const Rdna2Inst* compare = nullptr;
    for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
        if (it->pc >= branch_pc) continue;
        if (sopp_is_noop(*it)) continue;
        if (it->fmt == Rdna2Format::VOPC) { compare = &*it; break; }
        if (cannot_write_vcc(*it)) continue;   // (#590) look past provably-VCC-preserving instrs —
        break;                                 // VCC's value is frozen at the compare; only a real
                                               // rewrite between compare and branch invalidates it
    }
    if (!compare || compare->fmt != Rdna2Format::VOPC || vopc_is_cmpx(compare->opcode)) return false;

    std::function<bool(const Operand&, uint32_t)> uniform_operand_at;
    uniform_operand_at = [&](const Operand& operand, uint32_t use_pc) -> bool {
        if (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special ||
            operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat ||
            operand.kind == OperandKind::Literal)
            return true; // scalar-register sources (including VCC halves) broadcast one wave value
        if (operand.kind != OperandKind::VGPR) return false;
        for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
            if (it->pc >= use_pc) continue;
            const uint32_t writes = vgpr_write_count(*it);
            if (!writes || operand.value < it->dst.value ||
                operand.value >= it->dst.value + (int32_t)writes) continue;
            if (it->dst.value != operand.value || it->has_dpp) return false;
            // The write makes every currently active lane uniform. Keep the original conservative
            // EXEC rule: any later mask change could expose a lane that retained an older value.
            for (const auto& between : ins)
                if (between.pc > it->pc && between.pc < compare->pc &&
                    instruction_may_change_exec(between)) return false;

            bool pure_lane_alu = false;
            if (it->fmt == Rdna2Format::VOP1) {
                pure_lane_alu = it->n_src == 1;
            } else if (it->fmt == Rdna2Format::VOP2) {
                // v_cndmask consumes VCC; the carry family consumes/produces VCC. Neither preserves
                // scalar-source uniformity as a simple lane-local expression.
                pure_lane_alu = it->n_src == 2 && it->opcode != 0x01 &&
                    (it->opcode < 0x28 || it->opcode > 0x2a);
            }
            if (!pure_lane_alu) return false;
            for (uint32_t i = 0; i < it->n_src; ++i)
                if (!uniform_operand_at(it->src[i], it->pc)) return false;
            return true;
        }
        return false;
    };
    for (uint32_t i = 0; i < compare->n_src; ++i)
        if (!uniform_operand_at(compare->src[i], compare->pc)) return false;
    return compare->n_src != 0;
}

// Returns the loops in header-pc order, or {} when any backward branch doesn't fit the shape (the
// caller then rejects the stream loudly, exactly as before this feature). `safe` carries the
// linearized branches (waterfalls etc.) which are not loop back-edges.
std::vector<DivLoop> detect_divergent_loops(const std::vector<Rdna2Inst>& ins,
                                            const std::unordered_set<uint32_t>& safe,
                                            bool exact_fragment_wave_breaks = false) {
    std::vector<DivLoop> out;
    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    // Pass 1: each backward s_branch / s_cbranch_execnz is a candidate back-edge.
    std::vector<bool> backedge_execnz;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        if ((in.opcode != 0x02 && in.opcode != 0x09) || in.simm16 >= 0) continue;
        if (safe.count(in.pc)) continue;
        DivLoop L;
        L.header_pc = branch_target(in);
        L.backedge_pc = in.pc;
        L.exit_pc = in.pc + in.len_dwords;
        if (L.header_pc >= L.backedge_pc || L.exit_pc > end_pc) return {};
        bool hdr_ok = false;
        for (const auto& h : ins) { if (h.pc == L.header_pc) { hdr_ok = true; break; } if (h.pc > L.header_pc) break; }
        if (!hdr_ok) return {};
        out.push_back(L);
        backedge_execnz.push_back(in.opcode == 0x09);
    }
    if (out.empty()) return out;
    // Loops may be strictly DISJOINT (sequential) or PROPERLY NESTED (#590 — DOLL's post-process
    // kernel iterates an inner table loop inside an outer row loop). Pass 1 collected loops in
    // BACK-EDGE pc order (an inner back-edge precedes its outer's); re-sort by header so the list is
    // in emission order — emit_structured consumes loops by header pc, and emitting an outer body
    // recursively reaches the inner loop first because its header lies inside that range. Then
    // validate every pair as either disjoint-in-order or properly nested; partial overlap (an
    // unstructured region) still rejects loudly.
    {
        std::vector<size_t> order(out.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t c) { return out[a].header_pc < out[c].header_pc; });
        std::vector<DivLoop> sorted_loops; std::vector<bool> sorted_execnz;
        sorted_loops.reserve(out.size()); sorted_execnz.reserve(out.size());
        for (size_t i : order) { sorted_loops.push_back(out[i]); sorted_execnz.push_back(backedge_execnz[i]); }
        out.swap(sorted_loops); backedge_execnz.swap(sorted_execnz);
    }
    for (size_t i = 0; i < out.size(); i++)
        for (size_t j = i + 1; j < out.size(); j++) {
            const DivLoop& A = out[i]; const DivLoop& B = out[j];   // A.header_pc <= B.header_pc
            if (A.header_pc == B.header_pc) return {}; // shared header: not modeled
            const bool disjoint = B.header_pc >= A.exit_pc;
            const bool nested   = B.exit_pc <= A.backedge_pc;       // B entirely inside A's body
            if (!disjoint && !nested) return {};      // partial overlap: unstructured
        }
    // Pass 2: validate each loop's interior branches and find the canonical exit + breaks. A branch
    // inside a NESTED child loop belongs to the child (which validates itself in its own pass-2
    // iteration) — skip it here, including the child's backward back-edge, which is exactly the
    // "second back-edge inside" that used to reject nesting outright.
    auto inside_nested_child = [&out](const DivLoop& L, uint32_t pc) {
        for (const auto& C : out)
            if (C.header_pc > L.header_pc && C.exit_pc <= L.backedge_pc &&
                pc >= C.header_pc && pc <= C.backedge_pc) return true;
        return false;
    };
    for (size_t li = 0; li < out.size(); li++) {
        DivLoop& L = out[li];
        const bool execnz = backedge_execnz[li];
        for (const auto& in : ins) {
            if (in.is_end || in.pc >= L.backedge_pc) break;
            if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
            if (inside_nested_child(L, in.pc)) continue;
            switch (in.opcode) {
                case 0x02: case 0x04: case 0x05: case 0x06: case 0x07: case 0x08: case 0x09: break;
                default: continue;   // hints
            }
            if (in.simm16 < 0) return {};         // second back-edge inside -> nested loop
            if (safe.count(in.pc)) continue;                    // linearized (kill-mask / safe-execz)
            uint32_t tgt = branch_target(in);
            if (in.opcode == 0x02) {                            // forward s_branch: an else-arm terminator
                if (tgt > L.backedge_pc) return {}; // may not leave the body
                continue;                                       // (validated by detect_forward_ifs)
            }
            if (tgt > L.exit_pc) return {};       // conditional jumping past the loop
            if (tgt == L.exit_pc) {                             // an exit test
                if (!L.exit_branch_pc) {                        // first one = the canonical exit
                    if (in.opcode == 0x08) {                    // execz -> EXIT
                        L.condition = DivLoop::Condition::Exec;
                    } else if (in.opcode == 0x06 && !execnz &&
                               (exact_fragment_wave_breaks ||
                                vcc_exit_is_wave_uniform(ins, in.pc))) {
                        // Fragment reduces the real VCC mask through a forced wave64 vote. Other
                        // structured stages require the compare to be proven uniform first.
                        L.condition = DivLoop::Condition::Vcc;
                    } else {
                        return {};
                    }
                    L.exit_branch_pc = in.pc;
                } else {                                        // later ones = breaks
                    if (in.opcode != 0x06 && in.opcode != 0x08) return {}; // vccz/execz only
                    // With an execnz back-edge, a cleared lane skips to the next header check. With
                    // an unconditional back-edge that could re-enable EXEC, only an EXECZ branch is
                    // exact: the emitter sends the cleared lane directly to the loop merge instead.
                    // A VCCZ break does not itself clear EXEC and remains unsupported for that shape.
                    if (!execnz && in.opcode != 0x08) {
                        // A VCCZ break does not clear per-lane EXEC, so the old lane-local loop
                        // approximation could not carry it to the merge. The fragment shell now
                        // reduces VCC over an enforced wave64 subgroup and can branch the complete
                        // guest wave directly at the loop latch. Other stages remain conservative.
                        if (!exact_fragment_wave_breaks || in.opcode != 0x06) return {};
                        L.direct_wave_breaks = true;
                    } else if (!execnz) {
                        L.direct_exec_breaks = true;
                    }
                    L.break_pcs.push_back(in.pc);
                }
            }
            // (tgt <= backedge_pc: an interior forward if — validated by detect_forward_ifs.)
        }
        if (!L.exit_branch_pc) return {};         // no exit test: not this shape
        // The canonical exit must be the FIRST branch in the loop, so the condition region
        // [header, exit_branch) is branch-free (it is emitted straight-line).
        for (const auto& in : ins) {
            if (in.pc >= L.exit_branch_pc) break;
            if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
            if (in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03 && !safe.count(in.pc))
                return {};
        }
        // The execnz flavor's unconditional-continue lowering requires the header check to
        // immediately re-test EXEC (empty condition region) — see the shape comment.
        if (execnz && L.exit_branch_pc != L.header_pc) return {};
    }
    // A nested child must lie entirely within its parent's BODY: after the parent's canonical exit
    // test (the condition region [header, exit_branch) stays branch-free) and before its back-edge.
    for (size_t i = 0; i < out.size(); i++)
        for (size_t j = i + 1; j < out.size(); j++)
            if (out[j].exit_pc <= out[i].backedge_pc &&        // nested per the classification above
                out[j].header_pc <= out[i].exit_branch_pc) return {};
    // Pass 3: no branch from OUTSIDE a loop may target its interior (an unstructured entry edge).
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        if (in.opcode < 0x02 || in.opcode > 0x09 || in.opcode == 0x03) continue;
        uint32_t tgt = branch_target(in);
        for (const auto& L : out) {
            const bool inside_br = in.pc >= L.header_pc && in.pc <= L.backedge_pc;
            const bool inside_tgt = tgt > L.header_pc && tgt < L.exit_pc;
            if (!inside_br && inside_tgt) return {};
        }
    }
    return out;
}

// A single structured uniform IF: exactly ONE control-flow branch in the whole stream, and it is a
// FORWARD s_cbranch_scc0/scc1 (a scalar-uniform conditional — all lanes take the same path). The
// conditional block is [branch_pc+1, target_pc). Anything more (any s_branch/vcc/execz/execnz branch,
// a backward branch/loop, or a target outside the stream) is rejected → falls back to straight-line
// (which then rejects the scc branch, as before). Deliberately conservative: covers the common single
// forward uniform-if without a general structurizer.
struct ForwardIf {
    bool found = false;
    uint32_t branch_pc = 0, target_pc = 0;
    bool on_scc0 = true;   // s_cbranch_scc0/vccz/execz (branch taken == skip block when flag==0) vs scc1/vccnz
    bool on_vcc  = false;  // condition register: false = SCC, true = VCC (both are wave-uniform branches)
    bool early_out = false;// branch target was clamped to end_pc: the conditional block ENDS the shader
    // Divergent-region if (#273): the branch is a forward s_cbranch_execz over an EXEC-narrowed block
    // (v_cmpx … s_cbranch_execz — DOLL's FXAA PS). Fragment reduces EXEC across an enforced wave64;
    // other structured stages use the guarded per-invocation model. EXEC is phi'd at the merge.
    bool on_exec = false;
    // IF/ELSE (#273): the then-arm [branch_pc+1, sb_pc) is terminated by `s_branch merge_pc` at sb_pc
    // (the instruction immediately before target_pc); the else-arm starts at target_pc and runs to
    // merge_pc (or to the enclosing region's end when merge_pc is the shared OUTER merge — DOLL's
    // color-grade if/else-if cascade, where every arm's s_branch jumps to the outermost merge).
    bool has_else = false;
    uint32_t sb_pc = 0, merge_pc = 0;
};
// allow_vcc: also accept a forward s_cbranch_vccz/vccnz. Fragment reduces the per-lane bool through an
// enforced native wave64 subgroup; vertex retains the guarded per-invocation representation. Compute
// routes accepted VCC/EXEC branches through the CFG dispatcher, whose workgroup scratch reduction spans
// the configured 32/64-lane guest wave independently of the implementation-defined host subgroup width.
// code/dwords: the raw stream, so a branch target past the first s_endpgm can be decoded and
// verified to be a genuine early-out (see below) instead of blanket-clamped.
ForwardIf detect_forward_if(const std::vector<Rdna2Inst>& ins, bool allow_vcc,
                            const uint32_t* code, size_t dwords,
                            const std::unordered_set<uint32_t>* skip = nullptr) {
    ForwardIf F; const Rdna2Inst* br = nullptr; int nbranch = 0; uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        switch (in.opcode) {
            case 0x02: case 0x08: case 0x09:                         // s_branch / execz / execnz -> reject
                return F;
            case 0x06: case 0x07:                                    // vccz / vccnz
                if (!allow_vcc) return F;                            // compute: needs a wave-uniform VCC test
                br = &in; nbranch++; break;
            case 0x04: case 0x05:                                    // scc0 / scc1 (SCC is scalar/wave-uniform)
                if (skip && skip->count(in.pc)) break;               // alpha-test kill-mask branch: handled by
                                                                     // straight-line linearization, not a struct-if
                br = &in; nbranch++; break;
            default: break;                                          // hints (nop/waitcnt/…) are fine
        }
    }
    if (nbranch != 1 || !br) return F;
    uint32_t tgt = branch_target(*br);
    // Early-out: a forward branch PAST s_endpgm skips the rest of the shader — but only if execution
    // at the target immediately terminates. Decode from tgt and require the first non-s_nop
    // instruction to be s_endpgm (the compiled early-out shape `s_cbranch L; work; s_endpgm;
    // L: s_endpgm`). Anything else is REAL reachable code (an else-block) that the old blanket clamp
    // silently discarded — valid SPIR-V, wrong semantics (#129) — so reject instead (the caller's
    // straight-line fallback also rejects the branch, loudly). When verified, clamp the merge to
    // end_pc so the conditional block is [branch_pc+1, end_pc) and s_endpgm is emitted after the merge.
    bool early = false;
    if (tgt > end_pc && tgt != UINT32_MAX) {
        if (!code || tgt >= dwords) return F;    // target outside the decode window: can't verify
        std::vector<Rdna2Inst> tail;
        rdna2_walk(code + tgt, dwords - tgt, tail);
        bool ends_immediately = false;
        for (const auto& ti : tail) {
            if (ti.is_end) { ends_immediately = true; break; }
            if (ti.fmt == Rdna2Format::SOPP && ti.opcode == 0x00) continue;   // s_nop padding
            break;                               // real instruction at the target -> not an early-out
        }
        if (!ends_immediately) return F;
        tgt = end_pc; early = true;
    }
    if (tgt <= br->pc || tgt > end_pc) return F;                     // must be forward, within the stream
    F.found = true; F.branch_pc = br->pc; F.target_pc = tgt; F.early_out = (early || tgt == end_pc);
    F.on_scc0 = (br->opcode == 0x04 || br->opcode == 0x06);         // scc0/vccz: skip block when flag==0
    F.on_vcc  = (br->opcode == 0x06 || br->opcode == 0x07);
    return F;
}

// MULTIPLE structured uniform IFs — the generalization of detect_forward_if to N forward
// s_cbranch_scc*/vcc* branches that form a properly-NESTED (or disjoint sequential) region tree.
// Real shaders routinely carry several sequential uniform ifs (DOLL's color-grade PS: two
// `s_cbranch_vccz` blocks) or a nested guard (`vccz L1; …; vccz L2` with L2 < L1). Same per-branch
// validity rules as detect_forward_if (forward target inside the stream, or a VERIFIED early-out
// clamped to s_endpgm; scc branches in `skip` are alpha-test kill-masks handled by linearization);
// plus a structural check: blocks must nest or be disjoint — a partial overlap (branch into the
// middle of another block) is not an if-tree and rejects. Returns branches in pc order; empty =
// no/unsupported control flow (the caller falls back to straight-line, which rejects loudly at the
// branch). Compute execz uses a subgroup vote; execnz and unclaimed s_branch still reject wholesale.
// CONFIDENCE: HIGH on the structure (guarded by the phi machinery shared with the single-if path).
std::vector<ForwardIf> detect_forward_ifs(const std::vector<Rdna2Inst>& ins, bool allow_vcc,
                                          const uint32_t* code, size_t dwords,
                                          const std::unordered_set<uint32_t>* skip = nullptr,
                                          const std::vector<DivLoop>* loops = nullptr,
                                          bool* rejected = nullptr,
                                          bool compute_wave_branches = false) {
    std::vector<ForwardIf> out;
    if (rejected) *rejected = false;
    auto reject = [&]() -> std::vector<ForwardIf> { if (rejected) *rejected = true; return {}; };
    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    // Loop-claimed pcs (#273): back-edges + canonical exits are consumed by the loop emitter; a break
    // lowers as a forward if over the remainder of its loop's body (target clamped to the back-edge).
    auto loop_backedge = [&](uint32_t pc) { if (loops) for (const auto& L : *loops) if (L.backedge_pc == pc) return true; return false; };
    auto loop_exit     = [&](uint32_t pc) { if (loops) for (const auto& L : *loops) if (L.exit_branch_pc == pc) return true; return false; };
    auto loop_break    = [&](uint32_t pc) -> const DivLoop* {
        if (loops) for (const auto& L : *loops)
            for (uint32_t bp : L.break_pcs) if (bp == pc) return &L;
        return nullptr;
    };
    // Pass 1: collect the conditional branches (else-arm s_branch terminators are claimed in pass 2).
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        // Set when a compute vccz/vccnz if is accepted via the #590 uniformity proof — its region is
        // then required to be barrier/LDS/cross-lane free (checked after the target/merge is known).
        bool compute_uniform_vcc = false;
        switch (in.opcode) {
            case 0x02: continue;                                     // s_branch: validated in pass 2
            case 0x09:                                               // execnz: only as a loop back-edge
                if (loop_backedge(in.pc)) continue;
                if (getenv("PROSPER_DBG"))
                    std::fprintf(stderr, "[compute-struct-reject] unclaimed execnz pc=%u\n", in.pc);
                return reject();
            case 0x08:                                               // execz: safe-linearized predication
                if (skip && skip->count(in.pc)) continue;            // branch -> emit_alu no-ops it; else a
                if (loop_exit(in.pc)) continue;                      // loop exit test: the loop emitter's
                if (!allow_vcc && !compute_wave_branches) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr, "[compute-struct-reject] unsupported execz pc=%u\n", in.pc);
                    return reject();
                }
                break;                                               // compute lowers the wave-empty test with
                                                                     // OpGroupNonUniformAny(EXEC)
            case 0x06: case 0x07:                                    // vccz / vccnz
                if (loop_exit(in.pc)) continue;                      // canonical loop condition: loop emitter owns it
                if (!allow_vcc) {
                    // Compute VCC is a per-lane bool in the SPIR-V shell. The scalar branch consumes
                    // the architectural whole-wave VCCZ/VCCNZ flag. Accept it for the exact guest-wave
                    // CFG dispatcher; emit_body must never lower it through a native subgroup vote.
                    if (!compute_wave_branches) {
                        if (getenv("PROSPER_DBG"))
                            std::fprintf(stderr, "[compute-struct-reject] unsupported vcc branch pc=%u\n", in.pc);
                        return reject();
                    }
                    compute_uniform_vcc = true;
                }
                break;
            case 0x04: case 0x05:                                    // scc0 / scc1
                if (skip && skip->count(in.pc)) continue;            // kill-mask branch: linearized, not an if
                break;
            default: continue;                                       // hints
        }
        // A loop BREAK (validated vccz/execz -> exit_pc inside an execnz-back-edge loop): a plain
        // forward if skipping the REST of the body — the exec-governed continue then exits the lane.
        // A compute-uniform-accepted vcc branch can never be a break (compute loops require the
        // s_branch back-edge Vcc shape, which has no breaks) — reject the combination defensively.
        if (compute_uniform_vcc && loop_break(in.pc)) {
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr, "[compute-struct-reject] uniform vcc loop break pc=%u\n", in.pc);
            return reject();
        }
        if (const DivLoop* L = loop_break(in.pc)) {
            ForwardIf F;
            F.found = true; F.branch_pc = in.pc; F.target_pc = L->backedge_pc;
            F.on_scc0 = true;                                        // vccz/execz: skip block when flag==0
            F.on_vcc  = (in.opcode == 0x06);
            F.on_exec = (in.opcode == 0x08);
            out.push_back(F);
            continue;
        }
        uint32_t tgt = branch_target(in);
        bool early = false;
        if (tgt > end_pc && tgt != UINT32_MAX) {                     // possible early-out past s_endpgm:
            if (!code || tgt >= dwords) {
                if (getenv("PROSPER_DBG"))
                    std::fprintf(stderr, "[compute-struct-reject] target out of range pc=%u target=%u\n", in.pc, tgt);
                return reject();                                     // verify the target terminates (see
            }
            std::vector<Rdna2Inst> tail;                             // detect_forward_if for the rationale)
            rdna2_walk(code + tgt, dwords - tgt, tail);
            bool ends_immediately = false;
            for (const auto& ti : tail) {
                if (ti.is_end) { ends_immediately = true; break; }
                if (ti.fmt == Rdna2Format::SOPP && ti.opcode == 0x00) continue;
                break;
            }
            if (!ends_immediately) {
                if (getenv("PROSPER_DBG"))
                    std::fprintf(stderr, "[compute-struct-reject] nonterminal early target pc=%u target=%u\n", in.pc, tgt);
                return reject();
            }
            tgt = end_pc; early = true;
        }
        if (tgt <= in.pc || tgt > end_pc) {
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr, "[compute-struct-reject] nonforward target pc=%u target=%u\n", in.pc, tgt);
            return reject();                                         // must be forward, within the stream
        }
        ForwardIf F;
        F.found = true; F.branch_pc = in.pc; F.target_pc = tgt; F.early_out = (early || tgt == end_pc);
        F.on_scc0 = (in.opcode == 0x04 || in.opcode == 0x06 || in.opcode == 0x08);
        F.on_vcc  = (in.opcode == 0x06 || in.opcode == 0x07);
        F.on_exec = (in.opcode == 0x08);
        // IF/ELSE detection: the instruction immediately preceding target_pc is a FORWARD s_branch —
        // the then-arm's terminator jumping to the merge; the else-arm is [target_pc, merge). (Skip
        // for early-outs: the arm ends the shader.)
        if (!early && tgt != end_pc) {
            for (const auto& sb : ins) {
                if (sb.pc <= in.pc || sb.pc >= tgt) continue;
                if (sb.fmt != Rdna2Format::SOPP || sb.opcode != 0x02) continue;
                if (sb.pc + sb.len_dwords != tgt) continue;          // must be the arm's LAST instruction
                if (loop_backedge(sb.pc)) continue;                  // a loop's back-edge, not an else-jump
                if (sb.simm16 <= 0) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-struct-reject] backward else pc=%u branch=%u target=%u\n",
                                     in.pc, sb.pc, tgt);
                    return reject();                                 // backward else-jump: a loop, not an if
                }
                uint32_t lm = branch_target(sb);
                if (lm <= tgt || lm > end_pc) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-struct-reject] invalid else merge pc=%u branch=%u "
                                     "target=%u merge=%u\n",
                                     in.pc, sb.pc, tgt, lm);
                    return reject();                                 // merge must be forward, in-stream
                }
                F.has_else = true; F.sb_pc = sb.pc; F.merge_pc = lm;
                break;
            }
        }
        if (!allow_vcc && compute_wave_branches && F.on_exec) {
            // A compute execz condition is subgroup-uniform, so live scalar writes and LDS are safe
            // inside the structured arm. A workgroup barrier is not: different waves may take
            // different arms. Scalar writes into VCC_LO/HI also have an implicit mask-domain side
            // effect which the scalar scratch model cannot reconstruct; accept those only when the
            // mask is provably overwritten before any post-merge implicit read. The captured UE4
            // kernel writes VCC_LO as an integer scratch in the arm and overwrites VCC with a VOPC at
            // the merge, which this deliberately narrow proof accepts.
            const uint32_t region_end = F.has_else ? F.merge_pc : F.target_pc;
            for (const auto& r : ins) {
                if (r.is_end || r.pc >= region_end) break;
                if (r.pc <= in.pc) continue;
                if (r.fmt == Rdna2Format::SOPP && r.opcode == 0x0a) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-struct-reject] execz pc=%u contains barrier pc=%u\n",
                                     in.pc, r.pc);
                    return reject();
                }

                uint32_t scalar_width = 0;
                if (r.fmt == Rdna2Format::SOP1) scalar_width = r.opcode == 0x04 ? 2u : 1u;
                else if (r.fmt == Rdna2Format::SOP2) scalar_width = 1;
                else if (r.fmt == Rdna2Format::SOPK && sopk_writes_scalar_data(r.opcode)) scalar_width = 1;
                else if (r.fmt == Rdna2Format::SMEM) {
                    switch (r.opcode) {
                        case 0x0: case 0x8: scalar_width = 1; break;
                        case 0x1: case 0x9: scalar_width = 2; break;
                        case 0x2: case 0xA: scalar_width = 4; break;
                        case 0x3: case 0xB: scalar_width = 8; break;
                        case 0x4: case 0xC: scalar_width = 16; break;
                        default: break;
                    }
                }
                if (!scalar_width ||
                    (r.dst.kind != OperandKind::SGPR && r.dst.kind != OperandKind::Special)) continue;
                for (int vcc_half = 106; vcc_half <= 107; ++vcc_half) {
                    if (vcc_half < r.dst.value || vcc_half >= r.dst.value + (int)scalar_width) continue;
                    if (!sgpr_dead_at_merge(ins, region_end, vcc_half)) {
                        if (getenv("PROSPER_DBG"))
                            std::fprintf(stderr,
                                         "[compute-struct-reject] execz pc=%u scalar write pc=%u "
                                         "leaves vcc-half s%d live at merge pc=%u\n",
                                         in.pc, r.pc, vcc_half, region_end);
                        return reject();
                    }
                }
            }
        }
        if (compute_uniform_vcc) {
            // (#590) the uniformity proof is per-WAVE: reject when the guarded region contains a
            // barrier / LDS / cross-lane op (mirroring the compute loop-body guard — a barrier under
            // control flow whose condition could differ across the workgroup's waves would be
            // workgroup-divergent, UB).
            const uint32_t region_end = F.has_else ? F.merge_pc : F.target_pc;
            for (const auto& r : ins) {
                if (r.is_end || r.pc >= region_end) break;
                if (r.pc <= in.pc) continue;
                if (r.fmt == Rdna2Format::DS ||
                    (r.fmt == Rdna2Format::SOPP && r.opcode == 0x0a) ||
                    (r.fmt == Rdna2Format::VOP3 &&
                     (r.opcode == 0x365 || r.opcode == 0x366))) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-struct-reject] vcc branch pc=%u contains "
                                     "wave/workgroup op pc=%u fmt=%d op=0x%x\n",
                                     in.pc, r.pc, static_cast<int>(r.fmt), r.opcode);
                    return reject();
                }
            }
        }
        out.push_back(F);
    }
    // Pass 2: every s_branch in the stream must be a claimed else-arm terminator or a loop back-edge —
    // any other unconditional branch is control flow this structurizer doesn't model.
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x02) continue;
        if (loop_backedge(in.pc)) continue;
        bool claimed = false;
        for (const auto& F : out) if (F.has_else && F.sb_pc == in.pc) { claimed = true; break; }
        if (!claimed) {
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[compute-struct-reject] unclaimed s_branch pc=%u target=%u\n",
                             in.pc, branch_target(in));
            return reject();
        }
    }
    // Structural nesting check (branches are already in pc order): each region must lie entirely
    // inside the innermost still-open region, or start after it closed. A region spans to its merge
    // (if/else) or target (plain if); a LOOP is an opaque region [header_pc, exit_pc) that must nest
    // strictly. EXCEPTION (the shared-outer-merge cascade): an if/else whose merge_pc escapes the
    // innermost open region is accepted here and re-validated during emission (emit_structured
    // requires the escaping merge to equal the enclosing merge chain's continuation, i.e. no
    // instructions are skipped) — a violation rejects the shader there, fail-visible.
    std::vector<uint32_t> open;
    {
        size_t fi = 0, li = 0;
        const size_t nl = loops ? loops->size() : 0;
        while (fi < out.size() || li < nl) {
            const bool take_loop = li < nl &&
                (fi >= out.size() || (*loops)[li].header_pc < out[fi].branch_pc);
            const uint32_t start = take_loop ? (*loops)[li].header_pc : out[fi].branch_pc;
            uint32_t span_end    = take_loop ? (*loops)[li].exit_pc
                                 : (out[fi].has_else ? out[fi].merge_pc : out[fi].target_pc);
            const bool clampable = !take_loop && out[fi].has_else;
            while (!open.empty() && open.back() <= start) open.pop_back();
            if (!open.empty() && span_end > open.back()) {
                if (!clampable) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-struct-reject] overlapping region start=%u end=%u "
                                     "parent-end=%u\n",
                                     start, span_end, open.back());
                    return reject();                     // plain if / loop escaping its region: not a tree
                }
                span_end = open.back();                  // cascade if/else: clamped to the enclosing
            }                                            // region for nesting purposes (see above)
            open.push_back(span_end);
            if (take_loop) li++; else fi++;
        }
    }
    return out;
}

namespace {

uint32_t scalar_write_width(const Rdna2Inst& in) {
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            if (in.opcode == 0x20) return 0; // s_setpc_b64 reads its decoded "dst" field.
            switch (in.opcode) {
                case 0x04: case 0x08: case 0x0a: case 0x1f:
                case 0x24: case 0x25: case 0x26: case 0x27:
                case 0x28: case 0x29: case 0x2a: case 0x2b:
                case 0x37: case 0x38:
                    return 2; // B64 data/mask writes
                default: return 1;
            }
        case Rdna2Format::SOP2:
            switch (in.opcode) {
                case 0x0b: case 0x0f: case 0x11: case 0x13: case 0x15:
                case 0x17: case 0x19: case 0x1b: case 0x1d:
                case 0x1f: case 0x21: case 0x25: case 0x29:
                    return 2; // B64 data/mask writes
                default: return 1;
            }
        case Rdna2Format::SOPK: return sopk_writes_scalar_data(in.opcode) ? 1u : 0u;
        case Rdna2Format::SMEM:
            switch (in.opcode) {
                case 0x0: case 0x8: return 1;
                case 0x1: case 0x9: return 2;
                case 0x2: case 0xa: return 4;
                case 0x3: case 0xb: return 8;
                case 0x4: case 0xc: return 16;
                default: return 0;
            }
        case Rdna2Format::VOP1: return in.opcode == 0x02 ? 1u : 0u; // v_readfirstlane
        case Rdna2Format::VOP3: return in.opcode == 0x360 ? 1u : 0u; // v_readlane
        default: return 0;
    }
}

// Visit every explicit scalar-register write performed by one supported instruction. Most scalar
// instructions use `dst`, but VOPC SDWAB/e64 compares can write an SGPR mask pair through `dst`, and
// VOP3B arithmetic writes its carry mask through the independent `sdst` field. Keeping this as the
// single writer inventory prevents provenance/data-flow users from silently missing secondary dsts.
template <typename Visitor>
void for_each_scalar_write(const Rdna2Inst& in, Visitor&& visit,
                           bool wave32_one_word_masks = false) {
    const uint32_t width = scalar_write_width(in);
    if (width) visit(in.dst.value, width);
    if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
        in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)
        visit(in.dst.value, wave32_one_word_masks ? 1u : 2u);
    if (in.fmt == Rdna2Format::VOP3 && in.sdst.kind == OperandKind::SGPR)
        visit(in.sdst.value, wave32_one_word_masks &&
                              in.opcode >= 0x128 && in.opcode <= 0x12a ? 1u : 2u);
}

// True when this explicit scalar destination is written in the per-lane B64 mask domain.  Keep
// this classification independent of the post-emission SSA maps: folded/data writers such as
// s_getpc_b64 intentionally leave no scalar value behind, so absence from `sreg` cannot identify a
// mask write.  VOP3B's SDST is a mask/carry pair even though the instruction's primary VDST is data.
bool scalar_write_is_b64_mask(const Rdna2Inst& in, int base) {
    if (in.fmt == Rdna2Format::SOP1 && in.dst.value == base) {
        switch (in.opcode) {
            case 0x04: case 0x08: case 0x0a:
            case 0x24: case 0x25: case 0x26: case 0x27:
            case 0x28: case 0x29: case 0x2a: case 0x2b:
            case 0x37: case 0x38:
                return true;
            default: break;
        }
    }
    if (in.fmt == Rdna2Format::SOP2 && in.dst.value == base) {
        switch (in.opcode) {
            case 0x0b: case 0x0f: case 0x11: case 0x13: case 0x15:
            case 0x17: case 0x19: case 0x1b: case 0x1d: case 0x25:
                return true;
            default: break;
        }
    }
    if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
        in.dst.kind == OperandKind::SGPR && in.dst.value == base && base <= 105)
        return true;
    return in.fmt == Rdna2Format::VOP3 && in.sdst.kind == OperandKind::SGPR &&
           in.sdst.value == base;
}

void record_scalar_write(RegState& rs, const Rdna2Inst& in) {
    // VOPC/VOP3 mask destinations live in sreg_bool, but they still overwrite the physical SGPR
    // pair. Drop any scalar-data value or SRT descriptor tag left by that pair's earlier lifetime;
    // keeping either would let a later descriptor use observe the pre-overwrite value.
    auto invalidate_mask_pair = [&](int base) {
        for (int word = 0; word < 2; ++word) {
            rs.sreg.erase(base + word);
            rs.sreg_srt.erase(base + word);
        }
    };
    if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
        in.dst.kind == OperandKind::SGPR && in.dst.value <= 105) {
        if (rs.sreg_bool_b32.contains(in.dst.value)) {
            rs.sreg.erase(in.dst.value);
            rs.sreg_srt.erase(in.dst.value);
        } else {
            invalidate_mask_pair(in.dst.value);
        }
    }
    const bool vop3_b32_mask = in.fmt == Rdna2Format::VOP3 &&
        in.sdst.kind == OperandKind::SGPR &&
        rs.sreg_bool_b32.contains(in.sdst.value);
    if (in.fmt == Rdna2Format::VOP3 && in.sdst.kind == OperandKind::SGPR) {
        if (vop3_b32_mask) {
            rs.sreg.erase(in.sdst.value);
            rs.sreg_srt.erase(in.sdst.value);
        } else {
            invalidate_mask_pair(in.sdst.value);
        }
    }

    const bool vopc_b32_write = in.fmt == Rdna2Format::VOPC &&
        !vopc_is_cmpx(in.opcode) && rs.sreg_bool_b32.contains(in.dst.value);
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        // emit_alu has already materialized the new lifetime. Classify mask writers from the
        // instruction itself rather than inferring them from `sreg`: s_getpc_b64's folded form
        // deliberately erases both data SSA words, but hardware still overwrites the pair. End every
        // overlapping one-word Wave32 alias on all non-mask writes. A B64 mask replacement retains
                // its newly-emitted bool-domain value while dropping the obsolete B32 width marker.
        const bool sop2_b32_mask_domain = in.fmt == Rdna2Format::SOP2 &&
            (in.opcode == 0x0a ||
             (in.opcode >= 0x0e && in.opcode <= 0x1c && (in.opcode & 1u) == 0)) &&
            (in.dst.value == 126 ||
             in.src[0].value == 126 || in.src[1].value == 126 ||
             ((in.src[0].kind == OperandKind::SGPR ||
               in.src[0].kind == OperandKind::Special) &&
              rs.sreg_bool_b32.contains(in.src[0].value)) ||
             ((in.src[1].kind == OperandKind::SGPR ||
               in.src[1].kind == OperandKind::Special) &&
              rs.sreg_bool_b32.contains(in.src[1].value)));
        const bool vopc_b32_mask = vopc_b32_write && base == in.dst.value;
        const bool vop3_b32_mask_write = vop3_b32_mask && base == in.sdst.value;
        const uint32_t effective_width = (vopc_b32_mask || vop3_b32_mask_write) ? 1u : width;
        const bool writes_b32_mask = (effective_width == 1 || vopc_b32_mask) &&
            !rs.sreg.contains(base) &&
            rs.sreg_bool_b32.contains(base) &&
            ((in.fmt == Rdna2Format::SOP1 &&
              (in.opcode == 0x03 || in.opcode == 0x07 || in.opcode == 0x09 ||
               in.opcode == 0x3c || in.opcode == 0x40 || in.opcode == 0x44)) ||
             sop2_b32_mask_domain || vopc_b32_mask || vop3_b32_mask_write);
        const bool writes_b64_mask = effective_width == 2 && !vopc_b32_mask &&
            scalar_write_is_b64_mask(in, base);
        for (uint32_t word = 0; word < effective_width; ++word) {
            const int reg = base + static_cast<int>(word);
            if (!rs.sreg_bool_b32.contains(reg) || (writes_b32_mask && reg == base))
                continue;
            rs.sreg_bool_b32.erase(reg);
            if (!writes_b64_mask || reg != base) {
                rs.sreg_bool.erase(reg);
                rs.sreg_bool_narrowed.erase(reg);
                if (reg == 106) rs.vcc = 0;
            }
        }
        if (!writes_b32_mask) rs.sreg_bool_b32.erase(base);
        for (uint32_t word = 0; word < effective_width; ++word) {
            const int reg = base + static_cast<int>(word);
            rs.sreg_written.insert(reg);
            rs.sreg_input.erase(reg);
        }
    }, vopc_b32_write || vop3_b32_mask);
}

// Scalar registers that MAY be overwritten while a loop executes. This is deliberately separate
// from loop_written_regs: mask-pair destinations overwrite physical SGPRs (and therefore descriptor
// provenance) but their values live in sreg_bool rather than the scalar-data SSA domain.
void loop_scalar_may_writes(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                            std::set<int>& sregs) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                sregs.insert(base + static_cast<int>(word));
        });
    }
}

void invalidate_loop_descriptor_provenance(RegState& rs, const std::set<int>& sregs) {
    for (int reg : sregs) {
        rs.sreg_written.insert(reg);
        rs.sreg_input.erase(reg);
        rs.sreg_srt.erase(reg);
    }
}

// Complex CFG dispatch and the narrow loop structurizers persist B64 mask values, but not the
// separate one-word-validity state required by Wave32 aliases. Conservatively find any B32 mask
// copy that the region could create. The source set is deliberately path-insensitive: a pair made a
// mask on any path may reach a later copy on another dispatcher edge, and rejecting an impossible
// ordering is safer than silently restoring a stale Boolean.
bool has_unpersisted_b32_mask_lifetime(const std::vector<Rdna2Inst>& ins,
                                      uint32_t lo, uint32_t hi,
                                      const RegState& entry) {
    std::set<int> possible_mask_sources{106, 126}; // VCC_LO and EXEC_LO
    for (const auto& kv : entry.sreg_bool) possible_mask_sources.insert(kv.first);
    for (int reg : entry.sreg_bool_b32) possible_mask_sources.insert(reg);
    for (const auto& in : ins) {
        if (in.is_end || in.pc < lo || in.pc >= hi) continue;
        for_each_scalar_write(in, [&](int base, uint32_t) {
            if (scalar_write_is_b64_mask(in, base)) possible_mask_sources.insert(base);
        });
    }
    for (const auto& in : ins) {
        if (in.is_end || in.pc < lo || in.pc >= hi || in.fmt != Rdna2Format::SOP1)
            continue;
        if (in.opcode == 0x09) return true; // s_wqm_b32 creates/consumes the same width state
        if (in.opcode != 0x03) continue;
        const bool register_source = in.src[0].kind == OperandKind::SGPR ||
                                     in.src[0].kind == OperandKind::Special;
        if ((register_source && possible_mask_sources.contains(in.src[0].value)) ||
            in.dst.value == 126)
            return true;
    }
    return false;
}

} // namespace

int shader_max_vgpr(const std::vector<Rdna2Inst>& ins) {
    int highest = 0;
    for (const auto& in : ins) {
        if (in.is_end) break;
        for (uint32_t source = 0; source < in.n_src; ++source)
            if (in.src[source].kind == OperandKind::VGPR)
                highest = std::max(highest, in.src[source].value);
        const uint32_t writes = vgpr_write_count(in);
        if (writes)
            highest = std::max(highest, in.dst.value + static_cast<int>(writes) - 1);
    }
    return highest;
}

// Registers WRITTEN in the pc range [lo, hi): candidates for an OpPhi at the loop header. Over-
// approximation is safe (an extra phi for a non-carried value merges equal values). Mirrors emit_alu's
// write targets, INCLUDING multi-register writes (MIMG dmask -> N consecutive VGPRs, SMEM -> N SGPRs) so
// no genuinely-carried register is missed (a missing phi = an undominated use = invalid SPIR-V).
void loop_written_regs(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                       std::set<int>& vregs, std::set<int>& sregs) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        switch (in.fmt) {
            case Rdna2Format::VOP1:
                if (in.opcode == 0x02) sregs.insert(in.dst.value);        // v_readfirstlane -> SGPR
                else if (in.opcode == 0x42)                              // v_movreld: any observable
                    for (int reg = in.dst.value; reg <= shader_max_vgpr(ins); ++reg)
                        vregs.insert(reg);                               // VDST+M0 target
                else vregs.insert(in.dst.value); break;
            case Rdna2Format::VOP2: case Rdna2Format::VOP3P:
                vregs.insert(in.dst.value); break;
            case Rdna2Format::VOP3:
                if (in.opcode == 0x360) sregs.insert(in.dst.value);       // v_readlane -> SGPR
                else {
                    for (uint32_t k = 0; k < vgpr_write_count(in); ++k)
                        vregs.insert(in.dst.value + (int)k);
                }
                break;                                                    // (writelane: slots, not SSA)
            case Rdna2Format::DS:
                for (uint32_t k = 0; k < vgpr_write_count(in); ++k)
                    vregs.insert(in.dst.value + (int)k);
                break;
            case Rdna2Format::MUBUF: case Rdna2Format::MTBUF: case Rdna2Format::MIMG:
            case Rdna2Format::FLAT:
                for (uint32_t k = 0; k < vgpr_write_count(in); ++k)
                    vregs.insert(in.dst.value + (int)k);
                break;
            case Rdna2Format::SOP1:
                sregs.insert(in.dst.value); break;
            case Rdna2Format::SOPK:
                if (sopk_writes_scalar_data(in.opcode)) sregs.insert(in.dst.value);
                break;
            case Rdna2Format::SOP2:
                // s_lshr_b64 -> EXEC is modeled only in the per-lane mask domain. It does not
                // produce scalar SGPR data, so carrying a scalar value through a loop/if merge is
                // both unnecessary and semantically wrong.
                if (in.opcode != 0x21 || (in.dst.value != 126 && in.dst.value != 127)) {
                    sregs.insert(in.dst.value);
                    if (in.opcode == 0x1f || in.opcode == 0x21)
                        sregs.insert(in.dst.value + 1);
                }
                break;
            case Rdna2Format::SMEM: {                                      // s_load/s_buffer_load: N consecutive SGPRs
                uint32_t n = 1; switch (in.opcode) { case 0x1: case 0x9: n=2; break; case 0x2: case 0xA: n=4; break;
                    case 0x3: case 0xB: n=8; break; case 0x4: case 0xC: n=16; break; }
                for (uint32_t k = 0; k < n; k++) sregs.insert(in.dst.value + (int)k); break;
            }
            default: break;                          // VOPC/SOPC write VCC/SCC — handled by their own phis
        }
    }
}

// Resolve an operand to its raw 32-bit value (bits). Float ops bitcast these to float.
// `ok`: cleared when the operand's VALUE is not representable — a Special operand read as ALU DATA
// (VCC/EXEC live as per-lane bools in rs.vcc/rs.exec; their 32-bit wave-mask value does not exist
// in the per-invocation model, and untracked M0/ttmp aren't modeled). SGPR_NULL (field 125) is 0;
// SCC (field 253) is a scalar 0/1 and therefore is representable exactly. Previously other
// untracked Specials silently read as 0 and the shader computed
// garbage (#134); now it rejects, matching the SDWA/DPP reject-rather-than-miscompute discipline.
uint32_t operand_bits(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, const Operand& o, bool* ok = nullptr) {
    switch (o.kind) {
        case OperandKind::VGPR: {
            // A scalar-spill vgpr (v_writelane slots) has no per-lane data value — reject, don't read 0.
            if (rs.vgpr_lane_slots.count(o.value) || rs.vgpr_lane_mask_slots.count(o.value)) {
                if (ok) *ok = false; return b.uconst(0);
            }
            auto it = rs.vreg.find(o.value); return it == rs.vreg.end() ? b.uconst(0) : it->second; }
        case OperandKind::SGPR: {
            auto it = rs.sreg.find(o.value);
            if (it != rs.sreg.end()) return it->second;
            auto input = rs.sreg_input.find(o.value);
            return input == rs.sreg_input.end() ? b.uconst(0) : input->second;
        }
        case OperandKind::InlineInt:   return b.uconst((uint32_t)o.value);
        case OperandKind::InlineFloat: return b.uconst(fbits(inline_float_value((uint32_t)o.value)));
        case OperandKind::Literal:     return b.uconst(in.literal);
        case OperandKind::Special:
            if (o.value == 125) return b.uconst(0);   // SGPR_NULL: the one Special whose data value IS 0
            if (o.value == 253) {                     // SCC scalar source
                // rs.scc == 0 marks SCC poisoned by a 64-bit mask op (its SCC is a cross-lane
                // reduction this model cannot form) — reject rather than read a stale condition.
                if (!rs.scc) { if (ok) *ok = false; return b.uconst(0); }
                return b.sel(rs.scc, b.uconst(1), b.uconst(0));
            }
            // VCC_LO/HI (106/107), ttmp0..15 (108..123) and M0 (124) double as plain scalar SCRATCH
            // in compiled code — the NGG preamble does `s_bfe_u32 vcc_lo, s3, ...` then reads vcc
            // back as data, and DOLL's skinned VS round-trips M0 (`s_mov m0, s4 … s_mov s36, m0`,
            // #273). A scalar write lands in rs.sreg[o.value] (the DST field decodes as SGPR); read
            // it back from there. Only an UNTRACKED read (a VOPC-produced mask, EXEC, SCC, or a
            // never-written scratch) has no representable per-invocation data value.
            if (o.value >= 106 && o.value <= 124) {
                auto it = rs.sreg.find(o.value);
                if (it != rs.sreg.end()) return it->second;
            }
            // One Vulkan vertex invocation models one live lane of a virtual guest wave (lane 0),
            // which is also the contract used by vertex MBCNT and BFE-to-mask lowering. A compiler
            // generated NGG prolog may temporarily consume VCC_LO as scalar DATA after a VOPC; in
            // this shell its complete representable dword is therefore exactly {bit0=vcc}. VCC_HI
            // is zero. Compute/fragment keep their real multi-lane masks and must not take this path.
            if (!b.is_compute && !b.is_fragment && (o.value == 106 || o.value == 107))
                return o.value == 106 ? b.sel(rs.vcc, b.uconst(1), b.uconst(0)) : b.uconst(0);
            if (ok) *ok = false;
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[scalar-data-reject] pc=%u special=s%d tracked=%d mask=%d\n",
                             in.pc, o.value, rs.sreg.contains(o.value),
                             rs.sreg_bool.contains(o.value));
            return b.uconst(0);
        default:
            if (ok) *ok = false;
            return b.uconst(0);
    }
}

// A 64-bit mask write architecturally overwrites the destination SGPR pair ("D = ..." in every
// ISA mask-op description): any tracked data-domain SSA value or descriptor-provenance tag for
// those registers is stale afterwards. Erase both so a later data read / SRSRC resolution cannot
// alias the pre-mask-op value (the mask itself lives in sreg_bool / rs.vcc / rs.exec).
inline void mask_write_clobbers_pair(RegState& rs, int dst) {
    rs.sreg.erase(dst); rs.sreg.erase(dst + 1);
    rs.sreg_srt.erase(dst); rs.sreg_srt.erase(dst + 1);
}

// Emit one ALU instruction (VOP1/2/C/3 or SOP1/2) into `b`, updating `rs`. Returns true if `in` is an
// ALU format handled here; sets ok=false if it is an ALU op this stage doesn't support yet. Non-ALU
// formats (EXP/memory/...) return false so the stage-specific caller can handle them.
bool emit_alu(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_exec_update,
              const std::unordered_set<uint32_t>* safe_execz = nullptr, bool allow_smem = false,
              const ShaderResourceTable* rt = nullptr, bool allow_wave = false) {
    auto& vreg = rs.vreg; uint32_t& vcc = rs.vcc;
    auto val = [&](const Operand& o) { return operand_bits(b, rs, in, o, &ok); };
    // SDWA/DPP forms carry a sub-dword select or cross-lane control word we don't model. The decoder
    // flags them (and gets their length right); reject here rather than compute with a wrong operand.
    if (in.has_modifier) { ok = false; return true; }
    switch (in.fmt) {
        case Rdna2Format::SOP1: {
            if ((b.is_compute || b.is_fragment) && b.wave_size == 64 &&
                in.opcode == 0x03 && (in.dst.value == 126 || in.dst.value == 127) &&
                (in.src[0].value == 106 || in.src[0].value == 107) &&
                (in.dst.value - 126) == (in.src[0].value - 106)) {
                // Wave64 can update one EXEC dword without touching the other. Preserve that
                // distinction per invocation for the common VCC_LO->EXEC_LO / VCC_HI->EXEC_HI
                // copy. A cross-half copy would need another guest lane's predicate and remains
                // unsupported; the matching-half form maps directly to this lane's VCC bit.
                const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
                const uint32_t in_half = in.dst.value == 126
                    ? b.ucmp(Op_ULessThan, lane, b.uconst(32))
                    : b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
                rs.exec = b.bsel(in_half, rs.vcc, rs.exec);
                rs.exec_narrowed = true;
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if ((b.is_compute || b.is_fragment) && b.wave_size == 64 &&
                in.opcode == 0x03 && (in.dst.value == 126 || in.dst.value == 127) &&
                in.src[0].kind == OperandKind::InlineInt) {
                // The companion immediate form restores or clears one EXEC dword. Select the
                // addressed 32-lane half and preserve the other half exactly; this covers the
                // compiler's `s_mov_b32 exec_lo, -1` reconvergence after a low-half mask.
                const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
                const uint32_t in_half = in.dst.value == 126
                    ? b.ucmp(Op_ULessThan, lane, b.uconst(32))
                    : b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
                const uint32_t value = in.dst.value == 126
                    ? inline_int_mask_bit(b, in.src[0].value)
                    : inline_int_mask_bit_hi(b, in.src[0].value);
                rs.exec = b.bsel(in_half, value, rs.exec);
                rs.exec_narrowed = true;
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            // Wave32 shaders use the low 32-bit halves of EXEC/VCC for the same save/copy/restore
            // idioms that wave64 shaders express with s_mov_b64.  A wave mask is one bool in this
            // per-invocation model, so preserve that bool domain when either source is an
            // unambiguous low-half mask or EXEC_LO is the destination.  VCC_LO remains available as
            // ordinary scalar scratch when its source is ordinary data, matching the data path
            // below.  High-half moves remain unsupported: without the guest wave mode they may name
            // another lane rather than this invocation's bit.
            if (b.allow_b32_masks && in.opcode == 0x03) {   // s_mov_b32 (mask-domain form)
                const auto data_value_present = [&](int reg) {
                    return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
                };
                const bool src_exec_lo = in.src[0].value == 126;
                const bool src_vcc_lo = in.src[0].value == 106 &&
                    !data_value_present(106);
                const auto saved = (in.src[0].kind == OperandKind::SGPR ||
                                    in.src[0].kind == OperandKind::Special)
                    ? rs.sreg_bool.find(in.src[0].value) : rs.sreg_bool.end();
                const bool src_saved_mask = saved != rs.sreg_bool.end() &&
                    rs.sreg_bool_b32.contains(in.src[0].value) &&
                    !data_value_present(in.src[0].value);
                const bool dst_exec_lo = in.dst.value == 126;
                const bool mask_move = src_exec_lo || src_vcc_lo || src_saved_mask ||
                    (dst_exec_lo && in.src[0].kind == OperandKind::InlineInt);
                if (mask_move) {
                    uint32_t mask = 0;
                    bool narrowed = true;
                    if (src_exec_lo) {
                        mask = rs.exec;
                        narrowed = rs.exec_narrowed;
                    } else if (src_vcc_lo) {
                        mask = rs.vcc;
                        auto state = rs.sreg_bool_narrowed.find(106);
                        narrowed = state == rs.sreg_bool_narrowed.end() || state->second;
                    } else if (src_saved_mask) {
                        mask = saved->second;
                        auto state = rs.sreg_bool_narrowed.find(in.src[0].value);
                        narrowed = state == rs.sreg_bool_narrowed.end() || state->second;
                    } else if (in.src[0].value == -1) {
                        mask = b.btrue();
                        narrowed = false;
                    } else if (in.src[0].value == 0) {
                        mask = b.bfalse();
                    }
                    if (!mask || in.dst.value == 127) {
                        ok = false;
                        return true;
                    }

                    if (dst_exec_lo) {
                        rs.exec = mask;
                        rs.exec_narrowed = narrowed;
                    } else if (in.dst.value == 106) {
                        rs.vcc = mask;
                        rs.sreg_bool[106] = mask;
                        rs.sreg_bool_narrowed[106] = narrowed;
                        rs.sreg_bool_b32.insert(106);
                    } else {
                        rs.sreg_bool[in.dst.value] = mask;
                        rs.sreg_bool_narrowed[in.dst.value] = narrowed;
                        rs.sreg_bool_b32.insert(in.dst.value);
                    }
                    // A B32 move overwrites only the addressed physical word.  Remove stale scalar
                    // data/descriptor provenance for that word while leaving its neighbor intact.
                    rs.sreg.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    return true;
                }
            }
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
                (in.opcode == 0x3c || in.opcode == 0x40 || in.opcode == 0x44)) {
                // s_and_saveexec_b32 / s_orn2_saveexec_b32 / s_andn1_saveexec_b32
                // Save the previous EXEC_LO into one physical SGPR, then narrow EXEC_LO by the
                // one-word source mask. Astro uses VCC_HI as the saved-mask destination and restores
                // it later with s_mov_b32. The ANDN1 form selects the complementary old-EXEC arm.
                // Both halves are independent complete values in Wave32.
                auto source_mask = [&]() -> uint32_t {
                    if (in.src[0].value == 126) return rs.exec;
                    if ((in.src[0].kind == OperandKind::SGPR ||
                         in.src[0].kind == OperandKind::Special) &&
                        rs.sreg_bool_b32.contains(in.src[0].value)) {
                        auto it = rs.sreg_bool.find(in.src[0].value);
                        if (it != rs.sreg_bool.end()) return it->second;
                        if (in.src[0].value == 106) return rs.vcc;
                    }
                    if (in.src[0].kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, in.src[0].value);
                    return 0;
                };
                const uint32_t mask = source_mask();
                if (!mask || in.dst.value == 126 || in.dst.value == 127) {
                    ok = false; return true;
                }
                const uint32_t old_exec = rs.exec;
                rs.sreg_bool[in.dst.value] = old_exec;
                rs.sreg_bool_narrowed[in.dst.value] = rs.exec_narrowed;
                rs.sreg_bool_b32.insert(in.dst.value);
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                if (in.dst.value == 106) rs.vcc = old_exec;
                rs.exec = in.opcode == 0x3c ? b.land(old_exec, mask)
                        : in.opcode == 0x40 ? b.lor(mask, b.logical_not(old_exec))
                                            : b.land(old_exec, b.logical_not(mask));
                rs.exec_narrowed = true;
                rs.scc = b.is_fragment ? b.fragment_wave_any(rs.exec) : 0;
                return true;
            }
            if (b.allow_b32_masks && in.opcode == 0x09) {   // s_wqm_b32
                // Wave32 uses the low half of EXEC for the same whole-quad-mode idiom as
                // s_wqm_b64.  In the per-invocation fragment model helper lanes are implicit, so
                // widening is an identity on the tracked bool.  Keep this in the mask domain and
                // reject either high half; treating EXEC_LO as scalar data drops Astro's material
                // fragments before their first sample.
                const auto data_value_present = [&](int reg) {
                    return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
                };
                const bool src_exec_lo = in.src[0].value == 126;
                const bool src_vcc_lo = in.src[0].value == 106 &&
                    !data_value_present(106);
                const auto saved = in.src[0].kind == OperandKind::SGPR
                    ? rs.sreg_bool.find(in.src[0].value) : rs.sreg_bool.end();
                const bool src_saved_mask = saved != rs.sreg_bool.end() &&
                    !data_value_present(in.src[0].value);
                uint32_t mask = 0;
                if (src_exec_lo) mask = rs.exec;
                else if (src_vcc_lo) mask = rs.vcc;
                else if (src_saved_mask) mask = saved->second;
                else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1)
                    mask = b.btrue();
                else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == 0)
                    mask = b.bfalse();
                if (!mask || in.dst.value == 127 || in.dst.value == 107) {
                    ok = false;
                    return true;
                }

                // ISA SCC is a reduction over the whole resulting wave mask.  That value cannot be
                // recovered from one invocation, so poison it exactly like s_wqm_b64 does.
                rs.scc = 0;
                if (in.dst.value == 126) {
                    if (src_exec_lo) { /* exec <- wqm(exec): tracked identity */ }
                    else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                        rs.exec = b.btrue();
                        rs.exec_narrowed = false;
                    } else {
                        rs.exec = mask;
                        rs.exec_narrowed = true;
                    }
                } else if (in.dst.value == 106) {
                    rs.vcc = mask;
                    rs.sreg_bool[106] = mask;
                    rs.sreg_bool_narrowed[106] = true;
                    rs.sreg_bool_b32.insert(106);
                } else {
                    rs.sreg_bool[in.dst.value] = mask;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    rs.sreg_bool_b32.insert(in.dst.value);
                }
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
                in.opcode == 0x07) {   // s_not_b32 (mask-domain form)
                // Wave32 shader compilers also invert a saved one-word mask in an ordinary SGPR
                // before combining it back into VCC. Astro's world-map kernel does exactly this:
                //   s_mov_b32 s20, exec_lo
                //   ...
                //   s_not_b32 vcc_lo, s20
                //   s_nor_b32 vcc_lo, vcc_lo, vcc_hi
                // The old special case below only recognized `s_not_b32 vcc_lo,vcc_lo`; routing the
                // saved-SGPR form through scalar DATA either rejected (there are no uint bits for a
                // per-lane mask) or left a stale VCC lifetime at a dispatcher boundary. Accept only
                // an unambiguous complete Wave32 mask source with no competing scalar-data value.
                auto data_value_present = [&](int reg) {
                    return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
                };
                uint32_t source_mask = 0;
                if (in.src[0].value == 126) {
                    source_mask = rs.exec;
                } else if ((in.src[0].kind == OperandKind::SGPR ||
                            in.src[0].kind == OperandKind::Special) &&
                           rs.sreg_bool_b32.contains(in.src[0].value) &&
                           !data_value_present(in.src[0].value)) {
                    auto found = rs.sreg_bool.find(in.src[0].value);
                    if (found != rs.sreg_bool.end()) source_mask = found->second;
                    else if (in.src[0].value == 106) source_mask = rs.vcc;
                }
                if (source_mask && in.dst.value != 127) {
                    const uint32_t result = b.logical_not(source_mask);
                    if (in.dst.value == 126) {
                        rs.exec = result;
                        rs.exec_narrowed = true;
                    } else {
                        rs.sreg_bool[in.dst.value] = result;
                        // Complementing an arbitrary mask may activate any previously-clear lane;
                        // it is not a proof that the result is the full wave.
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        rs.sreg_bool_b32.insert(in.dst.value);
                        if (in.dst.value == 106) rs.vcc = result;
                    }
                    rs.sreg.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    rs.scc = 0; // SCC=(complete written dword != 0) needs a guest-wave reduction.
                    return true;
                }
            }
            if (in.opcode == 0x10) {   // s_bcnt1_i32_b64: popcount the complete scalar pair
                // A VOPC may write an arbitrary SGPR pair as a wave mask.  In the portable vertex
                // shell that pair contains the one represented guest lane, so its exact population
                // count is 0/1.  Ordinary data pairs retain full uint semantics and use two native
                // OpBitCount operations.  Multi-lane mask reductions remain fail-closed here; their
                // compute/fragment lowering needs a real wave reduction rather than scalar data.
                uint32_t result = 0;
                auto mask = rs.sreg_bool.find(in.src[0].value);
                if (!b.is_compute && !b.is_fragment && b.ngg_one_lane &&
                    mask != rs.sreg_bool.end()) {
                    result = b.sel(mask->second, b.uconst(1), b.uconst(0));
                } else {
                    // A Boolean-domain pair is a whole wave mask, not two ordinary scalar dwords.
                    // Only the byte-exact one-lane NGG projection above can reduce it locally.
                    if (mask != rs.sreg_bool.end()) { ok = false; return true; }
                    auto scalar_half = [&](int reg, uint32_t& value) {
                        auto current = rs.sreg.find(reg);
                        if (current != rs.sreg.end()) { value = current->second; return true; }
                        auto input = rs.sreg_input.find(reg);
                        if (input != rs.sreg_input.end()) { value = input->second; return true; }
                        return false;
                    };
                    uint32_t lo = 0, hi = 0;
                    if (in.src[0].kind == OperandKind::SGPR) {
                        if (!scalar_half(in.src[0].value, lo) ||
                            !scalar_half(in.src[0].value + 1, hi)) {
                            ok = false; return true;
                        }
                    } else if (in.src[0].kind == OperandKind::InlineInt) {
                        lo = b.uconst(static_cast<uint32_t>(in.src[0].value));
                        hi = b.uconst(in.src[0].value < 0 ? UINT32_MAX : 0u);
                    } else if (in.src[0].kind == OperandKind::Literal) {
                        lo = b.uconst(in.literal); hi = b.uconst(0);
                    } else {
                        ok = false; return true;
                    }
                    result = b.ibin(Op_IAdd, b.iun(Op_BitCount, lo), b.iun(Op_BitCount, hi));
                }
                rs.sreg[in.dst.value] = result;
                rs.sreg_srt.erase(in.dst.value);
                // A B32 write to VCC_LO also replaces virtual lane zero's architectural mask bit.
                // VCC_HI cannot affect that lane.  Other stages keep their multi-lane masks out of
                // the scalar domain and therefore never take this update.
                if (!b.is_compute && !b.is_fragment && in.dst.value == 106) {
                    const uint32_t bit = b.ibin(Op_BitwiseAnd, result, b.uconst(1));
                    rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                    rs.sreg_bool_narrowed[106] = true;
                }
                return true;
            }
            // 64-bit per-lane MASK ops (EXEC / VCC / saved masks). In our per-invocation model a wave
            // mask is a single bool for this lane. EXEC=SGPR 126/127, VCC=106/107; a saved mask lives
            // in sreg_bool. These implement divergent control flow (if/endif via saveexec + restore).
            if (in.opcode == 0x04 || in.opcode == 0x08 || in.opcode == 0x0a ||
                (in.opcode >= 0x24 && in.opcode <= 0x2b) ||
                in.opcode == 0x37 || in.opcode == 0x38) {
                // ISA: every op here EXCEPT s_mov_b64 writes SCC=(result!=0) — a cross-lane
                // reduction the per-invocation model cannot form. POISON the tracked SCC (SSA id 0)
                // so a later consumer (s_cselect/s_addc/SCC-source/scc-branch emission) rejects
                // fail-visibly instead of silently evaluating the value an OLDER s_cmp produced.
                // Any real SCC writer re-arms it. Branches the mask_test/waterfall linearizers
                // claim never read rs.scc, so the exercised adjacent shapes are unaffected.
                if (in.opcode != 0x04) rs.scc = 0;
                auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
                auto src_mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 106 || o.value == 107) return rs.vcc;    // VCC
                    if (o.value == 126 || o.value == 127) return rs.exec;   // EXEC
                    if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second; }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    return 0;   // not a recognizable mask
                };
                if (in.opcode == 0x0a) {                    // s_wqm_b64: whole-quad-mode mask widen
                    // WQM widens a lane mask to whole 2x2 quads so derivative/sample helper lanes stay
                    // active. In our per-invocation scalar SPIR-V model each lane is one bool and helper
                    // lanes are implicit in the fragment stage, so WQM-widening a mask is the IDENTITY.
                    // VERIFIED(round-trip llvm-mc gfx1030): SOP1 op 0x0a. Common PS preamble around
                    // image_sample (real game shaders 26-29,39), where it is `s_wqm_b64 exec,exec`.
                    // exec_narrowed handling mirrors s_mov_b64 (0x04): the exec<-exec self case is a true
                    // no-op (leave exec AND its narrowed flag untouched — so a later forward s_cbranch_execz
                    // is not spuriously rejected); replacing exec with a *different* mask may narrow it, so
                    // set exec_narrowed conservatively (else inactive-lane writes would escape predication).
                    // SCC = (mask != 0) is a CROSS-lane reduction our per-lane model can't form —
                    // rs.scc was POISONED above so later SCC consumers reject instead of misreading.
                    uint32_t m = src_mask(in.src[0]);
                    if (!m) { ok = false; return true; }
                    if (is_exec(in.dst)) {
                        if (is_exec(in.src[0])) { /* exec <- wqm(exec): identity; exec & narrowed unchanged */ }
                        else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                            rs.exec = b.btrue(); rs.exec_narrowed = false;      // exec = all lanes on
                        } else { rs.exec = m; rs.exec_narrowed = true; }        // replaced by a (maybe narrowed) mask
                    } else { rs.sreg_bool[in.dst.value] = m; rs.sreg_bool_narrowed[in.dst.value] = true;
                             mask_write_clobbers_pair(rs, in.dst.value); }  // conservative: WQM widens
                    return true;
                }
                // Narrowed-state carried alongside a saved mask: restoring EXEC from a mask that was saved
                // while EXEC was all-on must clear exec_narrowed (else it stays stuck true past an if/endif
                // — which e.g. breaks a loop that must re-enter the header with full EXEC).
                auto saved_narrowed = [&](const Operand& o) -> bool {
                    // Key by operand VALUE (VCC=106/107 and saved SGPR pairs alike; VCC decodes as Special,
                    // not SGPR, so don't gate on kind). The flag is kept in sync with the mask at EVERY
                    // writer (v_cmp/saveexec/s_mov/s_cselect all update it), so a lookup is accurate.
                    // Unknown provenance -> conservatively narrowed (over-narrowing is a safe no-op).
                    auto it = rs.sreg_bool_narrowed.find(o.value);
                    return it != rs.sreg_bool_narrowed.end() ? it->second : true;
                };
                if (in.opcode == 0x08) {                    // s_not_b64
                    const uint32_t m = src_mask(in.src[0]);
                    if (!m) { ok = false; return true; }
                    const uint32_t inverted = b.logical_not(m);
                    if (is_exec(in.dst)) {
                        rs.exec = inverted;
                        rs.exec_narrowed = true;
                    } else if (in.dst.value == 106 || in.dst.value == 107) {
                        rs.vcc = inverted;
                        rs.sreg_bool[in.dst.value] = inverted;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        mask_write_clobbers_pair(rs, in.dst.value);
                    } else {
                        rs.sreg_bool[in.dst.value] = inverted;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        mask_write_clobbers_pair(rs, in.dst.value);
                    }
                    return true;
                }
                if (in.opcode == 0x04) {                    // s_mov_b64
                    if (is_exec(in.dst)) {                  // set/restore EXEC
                        if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                            rs.exec = b.btrue(); rs.exec_narrowed = false;   // exec = all lanes on
                        } else { uint32_t m = src_mask(in.src[0]); if (!m) ok = false;
                                 else { rs.exec = m; rs.exec_narrowed = saved_narrowed(in.src[0]); } }
                    } else {                                // s_mov_b64 sDST, <mask-or-data> : save a mask / copy a pair
                        uint32_t m = src_mask(in.src[0]);
                        if (m) { rs.sreg_bool[in.dst.value] = m;
                                 rs.sreg_bool_narrowed[in.dst.value] = is_exec(in.src[0]) ? rs.exec_narrowed : saved_narrowed(in.src[0]);
                                 // A move INTO VCC is a VCC write (DOLL's scalar-indexed unroll does
                                 // `s_mov_b64 vcc, s[4:5]` before its vccz break, #273): the branch reads
                                 // rs.vcc, so it must be updated too — sreg_bool[106] alone left the
                                 // branch on a stale VCC. (The SOP2 mask ops already special-case 106.)
                                 if (in.dst.value == 106) rs.vcc = m; }
                        // s_mov_b64 ALSO moves a plain 64-bit VALUE (a descriptor pair, a scratch pair,
                        // an inline constant) — compilers use it to shuffle T#/V# halves and constants,
                        // not just wave masks. Copy the data SSA + descriptor provenance (sreg_srt) for
                        // both halves so a later ALU read / buffer op resolves; an untracked source half
                        // clears the stale dest entry rather than aliasing old data. Mask save (above)
                        // and data copy coexist: reads pick their own domain. CONFIDENCE: HIGH.
                        bool data_copied = false;
                        if (in.src[0].kind == OperandKind::SGPR ||
                            (in.src[0].kind == OperandKind::Special && in.src[0].value >= 106 && in.src[0].value <= 123)) {
                            for (int k = 0; k < 2; k++) {
                                int s = in.src[0].value + k, dr = in.dst.value + k;
                                auto it = rs.sreg.find(s);
                                if (it != rs.sreg.end()) rs.sreg[dr] = it->second; else rs.sreg.erase(dr);
                                auto jt = rs.sreg_srt.find(s);
                                if (jt != rs.sreg_srt.end()) rs.sreg_srt[dr] = jt->second; else rs.sreg_srt.erase(dr);
                            }
                            data_copied = true;
                        } else if (in.src[0].kind == OperandKind::InlineInt) {   // 64-bit sign-extended constant
                            rs.sreg[in.dst.value]     = b.uconst((uint32_t)in.src[0].value);
                            rs.sreg[in.dst.value + 1] = b.uconst(in.src[0].value < 0 ? 0xFFFFFFFFu : 0u);
                            rs.sreg_srt.erase(in.dst.value); rs.sreg_srt.erase(in.dst.value + 1);
                            data_copied = true;
                        } else if (in.src[0].kind == OperandKind::Literal) {
                            // 32-bit literal on a B64 move: zero-extended (the f64 high-placement rule is
                            // for double operands only; integer/untyped b64 literals zero-extend).
                            // CONFIDENCE: MED (matches llvm-mc's value validation for s_mov_b64 literals).
                            rs.sreg[in.dst.value]     = b.uconst(in.literal);
                            rs.sreg[in.dst.value + 1] = b.uconst(0);
                            rs.sreg_srt.erase(in.dst.value); rs.sreg_srt.erase(in.dst.value + 1);
                            data_copied = true;
                        }
                        if (data_copied && !m) {   // data-only move: drop any stale saved-mask alias
                            rs.sreg_bool.erase(in.dst.value); rs.sreg_bool_narrowed.erase(in.dst.value);
                        }
                        // Mask-only source (EXEC/a saved bool with no data half): the pair was still
                        // architecturally overwritten — stale data/descriptor entries must not survive.
                        if (m && !data_copied) mask_write_clobbers_pair(rs, in.dst.value);
                        if (!m && !data_copied) ok = false;
                    }
                } else {                                    // s_*_saveexec_b64 sDST, src
                    // All ten SAVEEXEC forms read OLD EXEC and S0 before writing either destination.
                    // Resolve S0 first because D may itself be VCC (Plucky's exact
                    // `s_orn2_saveexec_b64 vcc, exec`); publishing the saved mask early would change
                    // a VCC source into OLD EXEC and silently compute a different operation.
                    const uint32_t old_exec = rs.exec;
                    const bool old_exec_narrowed = rs.exec_narrowed;
                    uint32_t m = src_mask(in.src[0]);
                    if (!m) { ok = false; return true; }
                    rs.sreg_bool[in.dst.value] = old_exec;  // D = OLD_EXEC
                    rs.sreg_bool_narrowed[in.dst.value] = old_exec_narrowed;
                    if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = old_exec;
                    mask_write_clobbers_pair(rs, in.dst.value);
                    const uint32_t ne = b.logical_not(old_exec);
                    const uint32_t nm = b.logical_not(m);
                    const uint32_t xor_em = b.bsel(old_exec, nm, m);
                    rs.exec = in.opcode == 0x24 ? b.land(old_exec, m)       // AND
                            : in.opcode == 0x25 ? b.lor(old_exec, m)        // OR
                            : in.opcode == 0x26 ? xor_em                    // XOR
                            : in.opcode == 0x27 ? b.land(m, ne)             // ANDN2: S0 & ~EXEC
                            : in.opcode == 0x28 ? b.lor(m, ne)              // ORN2: S0 | ~EXEC
                            : in.opcode == 0x29 ? b.lor(ne, nm)             // NAND
                            : in.opcode == 0x2a ? b.land(ne, nm)            // NOR
                            : in.opcode == 0x2b ? b.logical_not(xor_em)     // XNOR
                            : in.opcode == 0x37 ? b.land(old_exec, nm)      // GFX10 ANDN1_SAVEEXEC
                                                : b.lor(old_exec, nm);      // GFX10 ORN1_SAVEEXEC
                    // Every non-trivial mask expression may be narrower than full EXEC. Two self
                    // identities are exactly all-on and must clear the flag or a later forward
                    // EXECZ guard is conservatively (and incorrectly) rejected.
                    const bool source_is_exec = in.src[0].value == 126 || in.src[0].value == 127;
                    rs.exec_narrowed = !((in.opcode == 0x28 || in.opcode == 0x2b) && source_is_exec);
                }
                return true;
            }
            if ((b.is_compute || b.is_fragment) && in.opcode == 0x07 &&
                (in.dst.value == 106 || in.dst.value == 107) &&
                in.src[0].value == in.dst.value &&
                !rs.sreg.contains(in.src[0].value)) {
                // s_not_b32 on one physical VCC half updates only those 32 guest lanes. Large
                // Wave64 compute kernels use `v_cmp ... vcc; s_not_b32 vcc_lo,vcc_lo;
                // s_mov_b64 exec,vcc` to select the complement for lanes 0..31 while retaining the
                // original predicate for lanes 32..63. The per-invocation Bool already represents
                // this lane's VCC bit, so select the inversion only in the addressed half.
                uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(),
                                       b.uconst(b.wave_size - 1));
                const uint32_t in_half = in.dst.value == 106
                    ? b.ucmp(Op_ULessThan, lane, b.uconst(32))
                    : b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
                rs.vcc = b.bsel(in_half, b.logical_not(rs.vcc), rs.vcc);
                rs.sreg_bool[in.dst.value] = rs.vcc;
                rs.sreg_bool_narrowed[in.dst.value] = true;
                if (b.allow_b32_masks) rs.sreg_bool_b32.insert(in.dst.value);
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                rs.scc = 0; // SCC=(complete written dword != 0) needs a guest-half reduction.
                return true;
            }
            if (in.opcode == 0x1f) {   // s_getpc_b64
                // Accepted ONLY when the pcrel pre-pass FOLDED an embedded-table load from this
                // shader — the pair then only feeds that folded chain. Otherwise the PC would flow
                // into unmodeled address math: keep rejecting.
                if (rs.mubuf_pcrel_tables.empty() && rs.smem_pcrel_tables.empty()) {
                    ok = false; return true;
                }
                for (int k = 0; k < 2; k++) {
                    rs.sreg.erase(in.dst.value + k); rs.sreg_srt.erase(in.dst.value + k);
                }
                return true;
            }
            if (in.opcode == 0x10 && b.ngg_one_lane) { // s_bcnt1_i32_b64
                // NGG's final primitive packing counts active bits in a saved wave mask. A Vulkan
                // vertex invocation models one guest lane, so the population count of that lane's
                // Boolean mask is exactly 0 or 1. Keep the result in the ordinary scalar-data domain
                // for the following integer packing arithmetic.
                uint32_t mask = 0;
                if (in.src[0].value == 106 || in.src[0].value == 107) mask = rs.vcc;
                else if (in.src[0].value == 126 || in.src[0].value == 127) mask = rs.exec;
                else if (in.src[0].kind == OperandKind::SGPR) {
                    auto it = rs.sreg_bool.find(in.src[0].value);
                    if (it != rs.sreg_bool.end()) mask = it->second;
                }
                if (!mask) { ok = false; return true; }
                rs.sreg[in.dst.value] = b.sel(mask, b.uconst(1), b.uconst(0));
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if (in.opcode == 0x14 && b.ngg_one_lane) { // s_ff1_i32_b64
                // RDNA2 returns the bit index of the first set bit, or -1 for an empty mask.
                // The exact Astro NGG projection represents guest lane zero only, so a tracked
                // mask has either that bit set (result 0) or no bits set (result 0xffffffff).
                uint32_t mask = 0;
                if (in.src[0].value == 106 || in.src[0].value == 107) mask = rs.vcc;
                else if (in.src[0].value == 126 || in.src[0].value == 127) mask = rs.exec;
                else if (in.src[0].kind == OperandKind::SGPR) {
                    auto it = rs.sreg_bool.find(in.src[0].value);
                    if (it != rs.sreg_bool.end()) mask = it->second;
                }
                if (!mask) { ok = false; return true; }
                rs.sreg[in.dst.value] = b.sel(mask, b.uconst(0), b.uconst(0xffffffffu));
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
                return true;
            }
            if (in.opcode == 0x1c || in.opcode == 0x1d) {
                // s_bitset{0,1}_b32 is an in-place scalar read/modify/write. The encoded source is
                // the bit index while SDST supplies both the old value and destination. Astro's
                // world-map kernel uses `s_bitset1_b32 s5,31` in a resource-table address path.
                if (in.dst.value == 126 || in.dst.value == 127) {
                    ok = false; return true;
                }
                uint32_t old_value = 0;
                auto current = rs.sreg.find(in.dst.value);
                if (current != rs.sreg.end()) {
                    old_value = current->second;
                } else {
                    auto input = rs.sreg_input.find(in.dst.value);
                    if (input == rs.sreg_input.end()) { ok = false; return true; }
                    old_value = input->second;
                }
                const uint32_t bit = b.ibin(
                    Op_BitwiseAnd, val(in.src[0]), b.uconst(31));
                if (!ok) return true;
                const uint32_t mask = b.ibin(
                    Op_ShiftLeftLogical, b.uconst(1), bit);
                rs.sreg[in.dst.value] = in.opcode == 0x1c
                    ? b.ibin(Op_BitwiseAnd, old_value, b.iun(Op_Not, mask))
                    : b.ibin(Op_BitwiseOr, old_value, mask);
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
                rs.sreg_bool_b32.erase(in.dst.value);
                return true;
            }
            // A 32-bit scalar DATA write into an EXEC half would leave the live per-lane mask
            // (rs.exec) stale — hardware updates EXEC (and EXECZ) immediately. No exercised title
            // writes EXEC halves via b32 scalar ops (wave64 compilers use the b64 forms), so
            // reject rather than model it. VCC_LO/HI (106/107) and M0 stay accepted as the
            // documented data-scratch round-trip (NGG preamble s_bfe_u32 vcc_lo, DOLL M0 moves).
            if (in.dst.value == 126 || in.dst.value == 127) { ok = false; return true; }
            uint32_t a = val(in.src[0]); uint32_t& d = rs.sreg[in.dst.value];
            // An ordinary scalar-data write starts a new lifetime for this physical word.  Do not
            // let an earlier Wave32 mask save alias that new value in later mask-domain moves.
            rs.sreg_bool.erase(in.dst.value);
            rs.sreg_bool_narrowed.erase(in.dst.value);
            rs.sreg_bool_b32.erase(in.dst.value);
            switch (in.opcode) {
                case 0x03: {                                // s_mov_b32
                    d = a;
                    // Descriptor provenance is part of the scalar value. Shader compilers commonly
                    // load several V#s into separate SGPR ranges and copy the selected four words into
                    // one reused SRSRC range before each MUBUF. Keeping that range's OLD tag makes all
                    // subsequent buffer loads resolve to the first descriptor (DOLL scene VS: the four
                    // transform rows became one row, degenerating every triangle).
                    auto tag = rs.sreg_srt.find(in.src[0].value);
                    if ((in.src[0].kind == OperandKind::SGPR ||
                         (in.src[0].kind == OperandKind::Special && in.src[0].value >= 106 && in.src[0].value <= 123)) &&
                        tag != rs.sreg_srt.end())
                        rs.sreg_srt[in.dst.value] = tag->second;
                    else
                        rs.sreg_srt.erase(in.dst.value);
                    break;
                }
                case 0x07:                                  // s_not_b32 changes the descriptor word
                    // ISA op 7: "D = ~S0; SCC = (D != 0)". (s_brev_b32 below has NO SCC write.)
                    d = b.iun(Op_Not, a); rs.sreg_srt.erase(in.dst.value);
                    rs.scc = b.ucmp(Op_INotEqual, d, b.uconst(0)); break;
                case 0x0b:                                  // s_brev_b32
                    d = b.iun(Op_BitReverse, a); rs.sreg_srt.erase(in.dst.value); break;
                case 0x34: {                                // s_abs_i32
                    // Two's-complement absolute value.  OpISub deliberately preserves the ISA's
                    // INT_MIN -> INT_MIN wraparound; SCC is set iff the resulting bits are nonzero.
                    const uint32_t negative = b.scmp(Op_SLessThan, a, b.uconst(0));
                    d = b.sel(negative, b.ibin(Op_ISub, b.uconst(0), a), a);
                    rs.scc = b.ucmp(Op_INotEqual, d, b.uconst(0));
                    rs.sreg_srt.erase(in.dst.value);
                    break;
                }
                default: ok = false;
            }
            return true;
        }
        case Rdna2Format::SOP2: {
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
                in.opcode == 0x0a &&
                (in.dst.value == 126 ||
                 in.src[0].value == 126 || in.src[1].value == 126 ||
                 (((in.src[0].kind == OperandKind::SGPR ||
                    in.src[0].kind == OperandKind::Special) &&
                   rs.sreg_bool_b32.contains(in.src[0].value))) ||
                 (((in.src[1].kind == OperandKind::SGPR ||
                    in.src[1].kind == OperandKind::Special) &&
                   rs.sreg_bool_b32.contains(in.src[1].value))))) {
                // Wave32 s_cselect_b32 can select complete one-word wave masks. Astro's material
                // setup uses `s_cselect_b32 vcc_hi, exec_lo, 0` after a scalar mode comparison,
                // then consumes that VCC_HI mask with VOP3 cndmask instructions. Keep the select in
                // the Bool domain; routing either operand through scalar bits would lose lane state.
                auto mask = [&](const Operand& source) -> uint32_t {
                    if (source.value == 126) return rs.exec;
                    if ((source.value == 106 || source.value == 107) &&
                        rs.sreg_bool_b32.contains(source.value)) {
                        auto it = rs.sreg_bool.find(source.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                        if (source.value == 106) return rs.vcc;
                    }
                    if (source.kind == OperandKind::SGPR &&
                        rs.sreg_bool_b32.contains(source.value)) {
                        auto it = rs.sreg_bool.find(source.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                    }
                    if (source.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, source.value);
                    return 0;
                };
                const uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (!m0 || !m1 || !rs.scc || in.dst.value == 127) {
                    ok = false; return true;
                }
                const uint32_t result = b.bsel(rs.scc, m0, m1);
                if (in.dst.value == 126) {
                    rs.exec = result;
                    rs.exec_narrowed = true;
                } else {
                    rs.sreg_bool[in.dst.value] = result;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    rs.sreg_bool_b32.insert(in.dst.value);
                    rs.sreg.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    if (in.dst.value == 106) rs.vcc = result;
                }
                return true;
            }
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
                (in.opcode == 0x0e || in.opcode == 0x10 || in.opcode == 0x12 ||
                 in.opcode == 0x14 || in.opcode == 0x16 || in.opcode == 0x18 ||
                 in.opcode == 0x1a || in.opcode == 0x1c) &&
                (in.dst.value == 126 ||
                 in.src[0].value == 126 || in.src[1].value == 126 ||
                 ((in.src[0].kind == OperandKind::SGPR ||
                   in.src[0].kind == OperandKind::Special) &&
                  rs.sreg_bool_b32.contains(in.src[0].value)) ||
                 ((in.src[1].kind == OperandKind::SGPR ||
                   in.src[1].kind == OperandKind::Special) &&
                  rs.sreg_bool_b32.contains(in.src[1].value)))) {
                // Wave32 B32 logical operations are one-word wave-mask operations, parallel to the
                // B64 family immediately below. The live Astro material uses
                //   s_andn2_b32 s64, s64, vcc_hi
                // after an explicit VOPC write to VCC_HI. Resolve each operand in the per-lane Bool
                // domain; treating VCC_HI as scalar bits is unrepresentable in this model.
                auto mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 126) return rs.exec;
                    if ((o.value == 106 || o.value == 107) &&
                        rs.sreg_bool_b32.contains(o.value)) {
                        auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                        if (o.value == 106) return rs.vcc;
                    }
                    if (o.kind == OperandKind::SGPR && rs.sreg_bool_b32.contains(o.value)) {
                        auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                    }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    return 0;
                };
                const uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (!m0 || !m1 || in.dst.value == 127) { ok = false; return true; }
                const uint32_t n0 = b.logical_not(m0), n1 = b.logical_not(m1);
                const uint32_t x = b.bsel(m0, n1, m1);
                const uint32_t r = in.opcode == 0x0e ? b.land(m0, m1)
                                 : in.opcode == 0x10 ? b.lor(m0, m1)
                                 : in.opcode == 0x12 ? x
                                 : in.opcode == 0x14 ? b.land(m0, n1)
                                 : in.opcode == 0x16 ? b.lor(m0, n1)
                                 : in.opcode == 0x18 ? b.lor(n0, n1)
                                 : in.opcode == 0x1a ? b.land(n0, n1)
                                 : b.logical_not(x);
                // SCC=(result!=0) is a guest-wave reduction. A fragment with proven Wave32 can
                // request one exact 32-lane Vulkan subgroup and vote here; compute retains the
                // existing poison unless its dispatcher supplies a synchronized reduction.
                rs.scc = b.is_fragment ? b.fragment_wave_any(r) : 0;
                if (in.dst.value == 126) {
                    rs.exec = r;
                    rs.exec_narrowed = true;
                } else {
                    rs.sreg_bool[in.dst.value] = r;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    rs.sreg_bool_b32.insert(in.dst.value);
                    rs.sreg.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    if (in.dst.value == 106) rs.vcc = r;
                }
                return true;
            }
            if (in.opcode >= 0x32 && in.opcode <= 0x34) {
                // GFX10 scalar halfword pack family.  LL selects src0.lo/src1.lo, LH selects
                // src0.lo/src1.hi, and HH selects src0.hi/src1.hi.  These are pure scalar DATA
                // operations (no SCC write); NGG uses LL to pack two wave population counts that
                // temporarily live in VCC_LO/HI.
                const uint32_t a = val(in.src[0]), c = val(in.src[1]);
                if (!ok) return true;
                const uint32_t lo = in.opcode == 0x34
                    ? b.ibin(Op_ShiftRightLogical, a, b.uconst(16))
                    : b.ibin(Op_BitwiseAnd, a, b.uconst(0xffff));
                const uint32_t hi = in.opcode == 0x32
                    ? b.ibin(Op_ShiftLeftLogical,
                             b.ibin(Op_BitwiseAnd, c, b.uconst(0xffff)), b.uconst(16))
                    : b.ibin(Op_BitwiseAnd, c, b.uconst(0xffff0000));
                const uint32_t result = b.ibin(Op_BitwiseOr, lo, hi);
                rs.sreg[in.dst.value] = result;
                rs.sreg_srt.erase(in.dst.value);
                if (!b.is_compute && !b.is_fragment && in.dst.value == 106) {
                    const uint32_t bit = b.ibin(Op_BitwiseAnd, result, b.uconst(1));
                    rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                    rs.sreg_bool_narrowed[106] = true;
                }
                return true;
            }
            if (in.opcode == 0x25) {   // s_bfm_b64: D=((1ULL<<(S0&63))-1)<<(S1&63)
                if (b.is_vertex && !b.ngg_one_lane) { ok = false; return true; }
                // Wave masks are represented as one bool per SPIR-V invocation.  Constructing the
                // architectural 64-bit integer and then splitting it would lose that domain, so test
                // the current lane directly: bits [offset, offset+width) are set, truncated at bit 63.
                // This is also exact for width==0 (the ISA expression produces an empty mask).
                const uint32_t width = b.ibin(Op_BitwiseAnd, val(in.src[0]), b.uconst(63));
                const uint32_t offset = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(63));
                // Vertex NGG is the same one-lane approximation as the inline-mask helpers: its
                // represented guest lane is lane zero and it has no compute LocalInvocationIndex.
                const uint32_t lane = b.ngg_one_lane ? b.uconst(0) :
                    b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1));
                const uint32_t at_or_after = b.ucmp(Op_UGreaterThanEqual, lane, offset);
                const uint32_t within_width = b.ucmp(
                    Op_ULessThan, b.ibin(Op_ISub, lane, offset), width);
                const uint32_t r = b.land(at_or_after, within_width);
                if (in.dst.value == 126 || in.dst.value == 127) {
                    rs.exec = r;
                    rs.exec_narrowed = true;
                } else if (in.dst.value == 106 || in.dst.value == 107) {
                    rs.vcc = r;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    mask_write_clobbers_pair(rs, in.dst.value);
                } else {
                    rs.sreg_bool[in.dst.value] = r;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    mask_write_clobbers_pair(rs, in.dst.value);
                }
                return true;
            }
            if (in.opcode == 0x0b) {   // s_cselect_b64: 64-bit MASK dst = SCC ? src0 : src1 (mask domain)
                // Operands/dest are wave masks (EXEC/VCC/saved/inline), NOT uint bits — resolve like the
                // SOP1 mask ops and select in the bool domain. (s_cselect_b32 0x0a stays in the uint path.)
                auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
                auto mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 106 || o.value == 107) return rs.vcc;
                    if (o.value == 126 || o.value == 127) return rs.exec;
                    if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second; }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    return 0;   // not a recognizable mask
                };
                uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (!m0 || !m1 || !rs.scc) { ok = false; return true; }   // !rs.scc: SCC poisoned by a mask op
                uint32_t r = b.bsel(rs.scc, m0, m1);
                if (is_exec(in.dst)) { rs.exec = r; rs.exec_narrowed = true; }
                else if (in.dst.value == 106 || in.dst.value == 107) { rs.vcc = r; rs.sreg_bool_narrowed[in.dst.value] = true;
                                                                       mask_write_clobbers_pair(rs, in.dst.value); }
                else { rs.sreg_bool[in.dst.value] = r; rs.sreg_bool_narrowed[in.dst.value] = true;   // conservative flag
                       mask_write_clobbers_pair(rs, in.dst.value); }
                return true;
            }
            if (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 || in.opcode == 0x15 ||
                in.opcode == 0x17 || in.opcode == 0x19 || in.opcode == 0x1b || in.opcode == 0x1d) {
                // 64-bit wave-mask LOGICAL ops on per-lane bools: AND/OR/XOR/ANDN2 plus the
                // complementary ORN2/NAND/NOR/XNOR family. Used for lane-mask arithmetic around
                // divergent control flow / ballot. SCC=(result!=0) is a cross-lane reduction we can't
                // form per-lane — POISON rs.scc so a later consumer rejects instead of silently
                // reading an older s_cmp's value (the adjacent scc-branch shapes are claimed by the
                // mask_test/waterfall linearizers and never read rs.scc). Same mask-resolution as
                // s_cselect_b64.
                auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
                auto mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 106 || o.value == 107) return rs.vcc;
                    if (o.value == 126 || o.value == 127) return rs.exec;
                    if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second; }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    return 0;
                };
                uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (!m0 || !m1) { ok = false; return true; }
                rs.scc = 0;   // poison: hardware SCC=(result!=0) is unrepresentable per-lane
                const uint32_t n0 = b.logical_not(m0), n1 = b.logical_not(m1);
                const uint32_t x = b.bsel(m0, n1, m1);               // xor = m0 ? !m1 : m1
                uint32_t r = in.opcode == 0x0f ? b.land(m0, m1)      // and
                           : in.opcode == 0x11 ? b.lor(m0, m1)       // or
                           : in.opcode == 0x13 ? x                    // xor
                           : in.opcode == 0x15 ? b.land(m0, n1)      // andn2
                           : in.opcode == 0x17 ? b.lor(m0, n1)       // orn2
                           : in.opcode == 0x19 ? b.lor(n0, n1)       // nand
                           : in.opcode == 0x1b ? b.land(n0, n1)      // nor
                           : b.logical_not(x);                        // xnor
                if (is_exec(in.dst)) { rs.exec = r; rs.exec_narrowed = true; }
                else if (in.dst.value == 106 || in.dst.value == 107) { rs.vcc = r; rs.sreg_bool_narrowed[in.dst.value] = true;
                                                                       mask_write_clobbers_pair(rs, in.dst.value); }
                else { rs.sreg_bool[in.dst.value] = r; rs.sreg_bool_narrowed[in.dst.value] = true;
                       mask_write_clobbers_pair(rs, in.dst.value); }
                return true;
            }
            // The NGG wave-packing s_lshr_b64 form (dst = EXEC) sets EXEC to the count of active
            // vertices/primitives in the wave. A per-invocation SPIR-V shader has no wave to pack,
            // so leave EXEC full. Handle that special form before the ordinary scalar-pair shift.
            if (in.opcode == 0x21) {
                if (in.dst.value == 126 || in.dst.value == 127) {
                    rs.scc = 0; // poison: hardware SCC=(result!=0) over the packed mask is cross-lane
                    return true;
                }
            }
            if (in.opcode == 0x1f || in.opcode == 0x21) {
                // Ordinary s_lshl/lshr_b64 operates on a complete scalar-data pair. Generated
                // compute shaders frequently borrow VCC_LO/HI for 64-bit address arithmetic; that
                // lifetime coexists with VCC's per-lane predicate view, so retain both exactly.
                auto scalar_word = [&](int reg, uint32_t& value) {
                    auto current = rs.sreg.find(reg);
                    if (current != rs.sreg.end()) { value = current->second; return true; }
                    auto input = rs.sreg_input.find(reg);
                    if (input != rs.sreg_input.end()) { value = input->second; return true; }
                    return false;
                };
                uint32_t source_lo = 0, source_hi = 0;
                const Operand& source = in.src[0];
                if (source.kind == OperandKind::SGPR ||
                    (source.kind == OperandKind::Special &&
                     source.value >= 106 && source.value < 124)) {
                    if (!scalar_word(source.value, source_lo) ||
                        !scalar_word(source.value + 1, source_hi)) {
                        ok = false; return true;
                    }
                } else if (source.kind == OperandKind::InlineInt) {
                    source_lo = b.uconst(static_cast<uint32_t>(source.value));
                    source_hi = b.uconst(source.value < 0 ? UINT32_MAX : 0u);
                } else if (source.kind == OperandKind::Literal) {
                    source_lo = b.uconst(in.literal);
                    source_hi = b.uconst(0);
                } else {
                    ok = false; return true;
                }
                const uint32_t amount = b.ibin(
                    Op_BitwiseAnd, val(in.src[1]), b.uconst(63));
                if (!ok) return true;
                const uint32_t result = b.u64_shift(
                    in.opcode == 0x1f ? Op_ShiftLeftLogical : Op_ShiftRightLogical,
                    b.u64_from_lohi(source_lo, source_hi), amount);
                const uint32_t lo = b.u64_lo(result), hi = b.u64_hi(result);
                rs.scc = b.ucmp(
                    Op_INotEqual, b.ibin(Op_BitwiseOr, lo, hi), b.uconst(0));
                const bool writes_exec = in.dst.value == 126 || in.dst.value == 127;
                const bool writes_vcc = in.dst.value == 106 || in.dst.value == 107;
                if (writes_exec || writes_vcc) {
                    uint32_t lane = b.ibin(
                        Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1));
                    const uint32_t mask_bit = b.u64_bit(result, lane);
                    if (writes_exec) {
                        rs.exec = mask_bit;
                        rs.exec_narrowed = true;
                        rs.sreg.erase(in.dst.value);
                        rs.sreg.erase(in.dst.value + 1);
                    } else {
                        rs.vcc = mask_bit;
                        rs.sreg_bool[in.dst.value] = mask_bit;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        rs.sreg[in.dst.value] = lo;
                        rs.sreg[in.dst.value + 1] = hi;
                    }
                } else {
                    rs.sreg[in.dst.value] = lo;
                    rs.sreg[in.dst.value + 1] = hi;
                    rs.sreg_bool.erase(in.dst.value);
                    rs.sreg_bool.erase(in.dst.value + 1);
                    rs.sreg_bool_narrowed.erase(in.dst.value);
                    rs.sreg_bool_narrowed.erase(in.dst.value + 1);
                }
                rs.sreg_bool_b32.erase(in.dst.value);
                rs.sreg_bool_b32.erase(in.dst.value + 1);
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value + 1);
                return true;
            }
            if (in.opcode == 0x29) {   // s_bfe_u64
                // BFE's sources are scalar DATA even when its destination is the architectural
                // EXEC/VCC mask pair. Keep the complete uniform source pair long enough to extract
                // the bit belonging to this emulated lane; treating the destination as ordinary
                // SGPR data leaves rs.exec/rs.vcc stale, while rejecting EXEC drops valid NGG
                // merged-stage prologues. A Vulkan vertex invocation represents the one live guest
                // lane for that vertex; compute/fragment shells retain their real wave lane.
                auto high = [&]() -> uint32_t {
                    const Operand& source = in.src[0];
                    if (source.kind == OperandKind::SGPR) {
                        auto value = rs.sreg.find(source.value + 1);
                        if (value != rs.sreg.end()) return value->second;
                        auto input = rs.sreg_input.find(source.value + 1);
                        return input == rs.sreg_input.end() ? b.uconst(0) : input->second;
                    }
                    if (source.kind == OperandKind::Special &&
                        source.value >= 106 && source.value < 124) {
                        auto value = rs.sreg.find(source.value + 1);
                        if (value != rs.sreg.end()) return value->second;
                        ok = false; return b.uconst(0);
                    }
                    if (source.kind == OperandKind::InlineInt)
                        return b.uconst(source.value < 0 ? UINT32_MAX : 0u);
                    if (source.kind == OperandKind::Literal) return b.uconst(0);
                    ok = false; return b.uconst(0);
                };
                const uint32_t source_lo = val(in.src[0]);
                const uint32_t source_hi = high();
                const uint32_t control = val(in.src[1]);
                if (!ok) return true;
                const uint32_t offset = b.ibin(Op_BitwiseAnd, control, b.uconst(0x3f));
                const uint32_t width = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, control, b.uconst(16)),
                    b.uconst(0x7f));
                const uint32_t result = b.bfe_u64(
                    b.u64_from_lohi(source_lo, source_hi), offset, width);
                const bool writes_exec = in.dst.value == 126 || in.dst.value == 127;
                const bool writes_vcc = in.dst.value == 106 || in.dst.value == 107;
                if (writes_exec || writes_vcc) {
                    uint32_t lane = b.guest_lane_id();
                    lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(b.wave_size - 1));
                    const uint32_t mask_bit = b.u64_bit(result, lane);
                    rs.scc = 0;   // poison: SCC=(complete mask != 0) is a guest-wave reduction
                    if (writes_exec) {
                        rs.exec = mask_bit;
                        rs.exec_narrowed = true;
                    } else {
                        rs.vcc = mask_bit;
                        rs.sreg_bool[in.dst.value] = mask_bit;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                    }
                    mask_write_clobbers_pair(rs, in.dst.value);
                } else {
                    const uint32_t lo = b.u64_lo(result), hi = b.u64_hi(result);
                    rs.sreg[in.dst.value] = lo;
                    rs.sreg[in.dst.value + 1] = hi;
                    rs.scc = b.ucmp(Op_INotEqual,
                                    b.ibin(Op_BitwiseOr, lo, hi), b.uconst(0));
                    rs.sreg_srt.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value + 1);
                }
                return true;
            }
            if ((in.dst.value == 106 || in.dst.value == 107) &&
                in.opcode >= 0x0e && in.opcode <= 0x1c && (in.opcode & 1u) == 0) {
                // 32-bit logical writes to VCC_LO/HI bridge scalar DATA masks into the architectural
                // lane mask. NGG prologs use this to merge a scalar bitset with a VOPC result before
                // a v_cndmask. We cannot materialize the complete VCC dword in a per-invocation
                // module, but its bit for this guest lane is exact. Preserve the untouched VCC half;
                // unlike a B64 mask write, a B32 write changes only LO or HI.
                auto logical_u32 = [&](uint32_t a, uint32_t c) {
                    return in.opcode == 0x0e ? b.ibin(Op_BitwiseAnd, a, c)
                         : in.opcode == 0x10 ? b.ibin(Op_BitwiseOr, a, c)
                         : in.opcode == 0x12 ? b.ibin(Op_BitwiseXor, a, c)
                         : in.opcode == 0x14 ? b.ibin(Op_BitwiseAnd, a, b.iun(Op_Not, c))
                         : in.opcode == 0x16 ? b.ibin(Op_BitwiseOr, a, b.iun(Op_Not, c))
                         : in.opcode == 0x18 ? b.iun(Op_Not, b.ibin(Op_BitwiseAnd, a, c))
                         : in.opcode == 0x1a ? b.iun(Op_Not, b.ibin(Op_BitwiseOr, a, c))
                         : b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c));
                };
                auto has_scalar_data = [&](const Operand& source) {
                    switch (source.kind) {
                        case OperandKind::SGPR:
                        case OperandKind::InlineInt:
                        case OperandKind::InlineFloat:
                        case OperandKind::Literal:
                            return true;
                        case OperandKind::Special:
                            if (source.value == 125) return true;       // SGPR_NULL
                            if (source.value == 253) return rs.scc != 0; // SCC as scalar 0/1
                            return source.value >= 106 && source.value <= 124 &&
                                   rs.sreg.count(source.value) != 0;
                        default:
                            return false;
                    }
                };
                if ((!b.is_fragment && !b.is_compute) ||
                    (has_scalar_data(in.src[0]) && has_scalar_data(in.src[1]))) {
                    // The vertex shell is a complete one-lane virtual wave, so its scalar VCC
                    // dwords are representable: LO starts as {bit0=vcc}, HI as zero, and later B32
                    // operations may use either as ordinary scratch. Fragment/compute shaders also
                    // use VCC_LO/HI as ordinary scalar scratch; when BOTH inputs have complete
                    // scalar-data representations, retain that full dword rather than collapsing it
                    // into a per-lane mask. The captured Plucky Squire tonemap shader, for example,
                    // computes `s_and_b32 vcc_lo, loop_index, 3` and immediately compares VCC_LO as
                    // a uint. True VOPC/mask inputs have no scalar representation and continue into
                    // the wave-mask path below.
                    const uint32_t result = logical_u32(val(in.src[0]), val(in.src[1]));
                    if (!ok) return true;
                    rs.sreg[in.dst.value] = result;
                    rs.sreg_srt.erase(in.dst.value);
                    rs.scc = b.ucmp(Op_INotEqual, result, b.uconst(0));
                    uint32_t lane = b.guest_lane_id();
                    lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(b.wave_size - 1));
                    const bool writes_hi = in.dst.value == 107;
                    const uint32_t in_written_half = writes_hi
                        ? b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32))
                        : b.ucmp(Op_ULessThan, lane, b.uconst(32));
                    const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
                    const uint32_t result_bit = b.ucmp(
                        Op_INotEqual,
                        b.ibin(Op_BitwiseAnd,
                               b.ibin(Op_ShiftRightLogical, result, bit), b.uconst(1)),
                        b.uconst(0));
                    rs.vcc = b.bsel(in_written_half, result_bit, rs.vcc);
                    rs.sreg_bool[in.dst.value] = rs.vcc;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    return true;
                }

                uint32_t lane = b.guest_lane_id();
                lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(b.wave_size - 1));
                const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
                const bool writes_hi = in.dst.value == 107;
                const uint32_t in_written_half = writes_hi
                    ? b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32))
                    : b.ucmp(Op_ULessThan, lane, b.uconst(32));
                auto mask_bit = [&](const Operand& source) -> uint32_t {
                    if (source.kind == OperandKind::Special &&
                        (source.value == 106 || source.value == 107)) {
                        // Reading the same VCC half maps directly to this invocation's predicate.
                        // Cross-half bit movement needs another guest lane and remains unsupported.
                        if (source.value != in.dst.value) { ok = false; return b.bfalse(); }
                        return rs.vcc;
                    }
                    const uint32_t raw = val(source);
                    return b.ucmp(Op_INotEqual,
                                  b.ibin(Op_BitwiseAnd,
                                         b.ibin(Op_ShiftRightLogical, raw, bit), b.uconst(1)),
                                  b.uconst(0));
                };
                const uint32_t m0 = mask_bit(in.src[0]);
                const uint32_t m1 = mask_bit(in.src[1]);
                if (!ok) return true;
                const uint32_t n0 = b.logical_not(m0), n1 = b.logical_not(m1);
                const uint32_t x = b.bsel(m0, n1, m1);
                const uint32_t result = in.opcode == 0x0e ? b.land(m0, m1)
                                      : in.opcode == 0x10 ? b.lor(m0, m1)
                                      : in.opcode == 0x12 ? x
                                      : in.opcode == 0x14 ? b.land(m0, n1)
                                      : in.opcode == 0x16 ? b.lor(m0, n1)
                                      : in.opcode == 0x18 ? b.lor(n0, n1)
                                      : in.opcode == 0x1a ? b.land(n0, n1)
                                      : b.logical_not(x);
                rs.vcc = b.bsel(in_written_half, result, rs.vcc);
                rs.sreg_bool[in.dst.value] = rs.vcc;
                rs.sreg_bool_narrowed[in.dst.value] = true;
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                rs.scc = 0; // SCC=(complete written dword != 0) is a guest-wave reduction
                return true;
            }
            // 32-bit scalar DATA writes into EXEC halves are rejected (see the SOP1 uint path).
            if (in.dst.value == 126 || in.dst.value == 127) { ok = false; return true; }
            uint32_t a = val(in.src[0]), c = val(in.src[1]); uint32_t& d = rs.sreg[in.dst.value];
            auto scc_nz = [&](uint32_t v){ rs.scc = b.ucmp(Op_INotEqual, v, b.uconst(0)); };  // SCC = (result != 0)
            switch (in.opcode) {
                case 0x00: d = b.ibin(Op_IAdd, a, c); rs.scc = b.ucmp(Op_ULessThan, d, a); break;  // s_add_u32 (SCC=carry)
                case 0x01: d = b.ibin(Op_ISub, a, c); rs.scc = b.ucmp(Op_ULessThan, a, c); break;  // s_sub_u32 (SCC=borrow)
                // Signed add/sub: two's-complement result is bit-identical to the unsigned op; SCC is
                // signed OVERFLOW. add ovf = operands same sign AND result sign differs: (~(a^c))&(a^d);
                // sub ovf = operands differ in sign AND result sign differs from a: (a^c)&(a^d). Bit 31 = ovf.
                case 0x02: { d = b.ibin(Op_IAdd, a, c);                                              // s_add_i32
                             uint32_t o = b.ibin(Op_BitwiseAnd, b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c)),
                                                               b.ibin(Op_BitwiseXor, a, d));
                             rs.scc = b.ucmp(Op_INotEqual, b.ibin(Op_ShiftRightLogical, o, b.uconst(31)), b.uconst(0)); break; }
                case 0x03: { d = b.ibin(Op_ISub, a, c);                                              // s_sub_i32
                             uint32_t o = b.ibin(Op_BitwiseAnd, b.ibin(Op_BitwiseXor, a, c),
                                                               b.ibin(Op_BitwiseXor, a, d));
                             rs.scc = b.ucmp(Op_INotEqual, b.ibin(Op_ShiftRightLogical, o, b.uconst(31)), b.uconst(0)); break; }
                case 0x04: {   // s_addc_u32: dst = src0 + src1 + SCC(carry-in); SCC = carry-out. The high
                               // half of a 64-bit add (pairs with s_add_u32's SCC). Round-trip llvm-mc
                               // gfx1010: 0x82252580 → s_addc_u32 s37, 0, s37. CONFIDENCE: HIGH.
                    if (!rs.scc) { ok = false; break; }   // SCC poisoned by a mask op: carry-in unknown
                    uint32_t cin = b.sel(rs.scc, b.uconst(1), b.uconst(0));
                    uint32_t s1 = b.ibin(Op_IAdd, a, c);
                    uint32_t k1 = b.ucmp(Op_ULessThan, s1, a);        // wrap in a+c
                    d = b.ibin(Op_IAdd, s1, cin);
                    uint32_t k2 = b.ucmp(Op_ULessThan, d, s1);        // wrap in +cin
                    rs.scc = b.bsel(k1, b.btrue(), k2); break;
                }
                // s_min/max (0x06 min_i32, 0x07 min_u32, 0x08 max_i32, 0x09 max_u32): SCC = "src0 was
                // selected", STRICT in both directions per the ISA pseudocode (S_MIN: SCC = S0 < S1;
                // S_MAX_I32: "D.i = (S0.i > S1.i) ? S0.i : S1.i; SCC = (S0.i > S1.i)", doc 70648
                // sec 12.1 ops 6-9) — on a tie SCC = 0 for both min and max. An earlier change (#397)
                // flipped max to non-strict `>=` quoting a pseudocode line that is not in the actual
                // document; the 2026-07 ISA audit (#879) re-derived the strict form from the PDF.
                // D is unaffected either way (the tie value is identical). Round-trip llvm-mc
                // gfx1010: 0x83000201/0x83800201/0x84000201/0x84800201. CONFIDENCE: HIGH.
                case 0x06: rs.scc = b.scmp(Op_SLessThan, a, c);    d = b.sext2(Glsl_SMin, a, c); break;
                case 0x07: rs.scc = b.ucmp(Op_ULessThan, a, c);    d = b.uext2(Glsl_UMin, a, c); break;
                case 0x08: rs.scc = b.scmp(Op_SGreaterThan, a, c); d = b.sext2(Glsl_SMax, a, c); break;
                case 0x09: rs.scc = b.ucmp(Op_UGreaterThan, a, c); d = b.uext2(Glsl_UMax, a, c); break;
                case 0x0A:   // s_cselect_b32: SCC ? src0 : src1 (reads SCC, writes none)
                    if (!rs.scc) { ok = false; break; }   // SCC poisoned by a mask op
                    d = b.sel(rs.scc, a, c); break;
                case 0x0E: d = b.ibin(Op_BitwiseAnd, a, c); scc_nz(d); break;   // s_and_b32
                case 0x10: d = b.ibin(Op_BitwiseOr,  a, c); scc_nz(d); break;   // s_or_b32
                case 0x12: d = b.ibin(Op_BitwiseXor, a, c); scc_nz(d); break;   // s_xor_b32
                case 0x1E: { uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(31));   // s_lshl_b32
                             d = b.ibin(Op_ShiftLeftLogical, a, sh); scc_nz(d); break; }  // dst = src0 << (src1 & 31)
                case 0x24: { uint32_t w  = b.ibin(Op_BitwiseAnd, a, b.uconst(0x1f));   // s_bfm_b32: bitfield mask
                             uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(0x1f));   // dst = ((1<<(src0&31))-1)<<(src1&31)
                             uint32_t mask = b.ibin(Op_ISub, b.ibin(Op_ShiftLeftLogical, b.uconst(1), w), b.uconst(1));
                             d = b.ibin(Op_ShiftLeftLogical, mask, sh); break; }       // no SCC write
                case 0x20: { uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(31));   // s_lshr_b32
                             d = b.ibin(Op_ShiftRightLogical, a, sh); scc_nz(d); break; }  // dst = src0 >> (src1 & 31)
                case 0x22: { uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(31));   // s_ashr_i32
                             d = b.sbin(Op_ShiftRightArithmetic, a, sh); scc_nz(d); break; }  // dst = src0 >>a (src1 & 31)
                case 0x26: d = b.ibin(Op_IMul, a, c); break;         // s_mul_i32 (low 32 bits; no SCC)
                case 0x2E: case 0x2F: case 0x30: case 0x31: {
                    // s_lshl{1,2,3,4}_add_u32 = (src0<<N)+src1; SCC is unsigned carry-out of the
                    // FULL 64-bit sum (RDNA2 ISA ops 46-49: ((u64)S0<<N)+S1 >= 2^32). If any of
                    // S0's top N bits are set, the shift alone overflowed; otherwise overflow is
                    // exactly the ordinary low-dword addition wrap (D < S1).
                    const uint32_t shift = in.opcode - 0x2Du;
                    uint32_t shifted = b.ibin(Op_ShiftLeftLogical, a, b.uconst(shift));
                    d = b.ibin(Op_IAdd, shifted, c);
                    uint32_t shifted_out = b.ucmp(Op_INotEqual,
                        b.ibin(Op_ShiftRightLogical, a, b.uconst(32u - shift)), b.uconst(0));
                    uint32_t wrapped = b.ucmp(Op_ULessThan, d, c);
                    rs.scc = b.bsel(shifted_out, b.btrue(), wrapped); break;
                }
                case 0x32: { // s_pack_ll_b32_b16: D={S1[15:0],S0[15:0]}
                    const uint32_t lo = b.ibin(Op_BitwiseAnd, a, b.uconst(0xFFFFu));
                    const uint32_t hi = b.ibin(Op_ShiftLeftLogical,
                        b.ibin(Op_BitwiseAnd, c, b.uconst(0xFFFFu)), b.uconst(16));
                    d = b.ibin(Op_BitwiseOr, lo, hi);
                    break;
                }
                case 0x35: d = b.umul_hi(a, c); break;               // s_mul_hi_u32 (high 32 bits; no SCC)
                case 0x36: d = b.smul_hi(a, c); break;               // s_mul_hi_i32 (high 32 bits; no SCC).
                                                                     // gfx10 SOP2 opcode is 0x36 (llvm-mc:
                                                                     // 0x9b000201>>23&0x7f); 0x37 was WRONG
                                                                     // (invalid encoding), so the handler was
                                                                     // dead and s_mul_hi_i32 got rejected. #462
                case 0x27: {                                         // s_bfe_u32: offset=src1[4:0], width=src1[22:16]
                    uint32_t off = b.ibin(Op_BitwiseAnd, c, b.uconst(0x1f));
                    uint32_t width = b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, c, b.uconst(16)), b.uconst(0x7f));
                    d = b.bfe_u(a, off, width); scc_nz(d); break;
                }
                default: ok = false;
            }
            // Every modeled SOP2 operation changes the scalar bits (including cselect unless both
            // inputs happen to be identical). A previous descriptor tag on the destination is stale;
            // retaining it can bind an unrelated later SRSRC to the old buffer.
            if (ok) {
                rs.sreg_srt.erase(in.dst.value);
                if (in.opcode == 0x29) rs.sreg_srt.erase(in.dst.value + 1);
            }
            return true;
        }
        case Rdna2Format::VOP3P: {
            if (in.opcode == 0x0F || in.opcode == 0x10) {  // v_pk_add_f16 / v_pk_mul_f16
                const uint32_t old_d = vreg_old(b, rs, in.dst.value);
                auto half = [&](int source, bool high_result) -> uint32_t {
                    uint32_t v = val(in.src[source]);
                    const uint8_t selectors = high_result ? in.vop3p_opsel_hi : in.vop3p_opsel;
                    if (in.src[source].kind == OperandKind::InlineFloat) {
                        // A 16-bit operand receives the f16 encoding of the constant (replicated to
                        // both halves, so the half select is a no-op) — exact even for 1/(2*pi).
                        v = b.unpack_half(b.uconst(inline_float_f16_bits(in.src[source].value)), 0);
                    } else if (in.src[source].kind == OperandKind::InlineInt) {
                        // Integer inline constants are raw two's-complement bits at operand width
                        // (ISA sec 6.2): the packed operand's halves are the low/high 16 bits of the
                        // sign-extended value — inline 1 reads as the f16 denormal 0x0001, NOT 1.0.
                        v = b.unpack_half(b.uconst(static_cast<uint32_t>(in.src[source].value)),
                                          (selectors >> source) & 1u);
                    } else {
                        v = b.unpack_half(v, (selectors >> source) & 1u);
                    }
                    const bool negate = high_result
                        ? ((in.vop3p_neg_hi >> source) & 1u) != 0
                        : in.src_neg[source];
                    return negate ? b.fneg(v) : v;
                };
                auto operation = [&](bool high) {
                    uint32_t r = in.opcode == 0x0F
                        ? b.fbin(Op_FAdd, half(0, high), half(1, high))
                        : b.fbin(Op_FMul, half(0, high), half(1, high));
                    if (in.clamp) r = b.clamp01(r);
                    return r;
                };
                const uint32_t lo = b.pack_half_lo(operation(false));
                const uint32_t hi = b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(operation(true)), b.uconst(16));
                rs.vreg[in.dst.value] = b.ibin(Op_BitwiseOr, lo, hi);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // Mixed-precision FMA family, trivial form only (all sources full f32 — the decoder set
            // has_modifier for any opsel/neg/clamp bits, rejected above). v_fma_mix_f32 (0x20):
            // d = s0*s1+s2. v_fma_mixlo/hi_f16 (0x21/0x22): the f32 result converts to f16 into the
            // LOW/HIGH half of d, PRESERVING the other half (DOLL's box-blur PS packs its result
            // this way, #273). VERIFIED(round-trip llvm-mc gfx1010: 0xcc210000 0x041600f2 ->
            // v_fma_mixlo_f16 v0, 1.0, v0, v5 — the live blur bytes).
            if (in.opcode != 0x20 && in.opcode != 0x21 && in.opcode != 0x22) { ok = false; return true; }
            uint32_t old_d = vreg_old(b, rs, in.dst.value);
            // Per-source mix resolve (#273): OPSEL_HI[k] -> f16 half (OPSEL[k]: 0=lo,1=hi) converted
            // to f32; else full f32. An inline constant is already a full-width value, so the half
            // select is a no-op for it (hardware reads the f16-converted constant — same value).
            // NEG_HI = abs, NEG = negate (abs first, hardware order).
            auto mixv = [&](int k) -> uint32_t {
                uint32_t v = val(in.src[k]);
                const bool half = (in.vop3p_opsel_hi >> k) & 1u;
                if (half) {
                    if (in.src[k].kind == OperandKind::InlineFloat)
                        // f16-half read of an inline float: the exact documented f16 encoding.
                        v = b.unpack_half(b.uconst(inline_float_f16_bits(in.src[k].value)), 0);
                    else if (in.src[k].kind == OperandKind::InlineInt)
                        // Raw two's-complement bits at operand width: the selected half of the
                        // sign-extended value (inline 1 -> f16 denormal, not 1.0).
                        v = b.unpack_half(b.uconst(static_cast<uint32_t>(in.src[k].value)),
                                          (in.vop3p_opsel >> k) & 1u);
                    else
                        v = b.unpack_half(v, (in.vop3p_opsel >> k) & 1u);
                }
                if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                if (in.src_neg[k]) v = b.fneg(v);
                return v;
            };
            uint32_t r = b.fbin(Op_FAdd, b.fbin(Op_FMul, mixv(0), mixv(1)), mixv(2));
            if (in.clamp) r = b.clamp01(r);
            uint32_t& d = rs.vreg[in.dst.value];
            if (in.opcode == 0x20) d = r;
            else if (in.opcode == 0x21)
                d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), b.pack_half_lo(r));
            else
                d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                           b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(r), b.uconst(16)));
            if (ok) predicate_write(b, rs, in.dst.value, old_d);
            return true;
        }
        case Rdna2Format::SOPC: {
            // s_bitcmp0/1_b32 (0x0c/0x0d): SCC = bit (src0 >> (src1 & 31)) & 1, negated for bitcmp0.
            // DOLL's scene PS tests feature-flag bits 0..3 of an s_buffer_load'd word with s_cselect
            // chains (#273). VERIFIED(round-trip llvm-mc gfx1010: 0xbf0d8014 -> s_bitcmp1_b32 s20, 0;
            // 0xbf0c8114 -> s_bitcmp0_b32 s20, 1 — NOT the u64 compares an opcode-table guess said).
            if (in.opcode == 0x0c || in.opcode == 0x0d) {
                uint32_t a = val(in.src[0]), c = val(in.src[1]);
                uint32_t sh  = b.ibin(Op_BitwiseAnd, c, b.uconst(31));
                uint32_t bit = b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, a, sh), b.uconst(1));
                uint32_t nz  = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                rs.scc = (in.opcode == 0x0d) ? nz : b.bsel(nz, b.bfalse(), b.btrue());
                return true;
            }
            // Scalar compare -> SCC (read by s_cselect / s_cbranch_scc). eq/lg are bitwise (sign-agnostic);
            // the ordered compares are signed for i32 (0x02-0x05), unsigned for u32 (0x08-0x0b).
            // Wave32 compilers also compare the physical VCC_LO dword with zero. The mask has no
            // per-invocation integer representation, but EQ/LG and the bounded unsigned forms below
            // reduce exactly to none/any across the guest wave.
            if (allow_wave && b.is_compute && b.wave_size == 32) {
                auto zero = [](const Operand& operand) {
                    return operand.kind == OperandKind::InlineInt && operand.value == 0;
                };
                auto mask = [&](const Operand& operand) -> uint32_t {
                    if (operand.kind != OperandKind::SGPR &&
                        operand.kind != OperandKind::Special) return 0;
                    auto found = rs.sreg_bool.find(operand.value);
                    return found == rs.sreg_bool.end() ? 0 : found->second;
                };
                const uint32_t first_mask = mask(in.src[0]);
                const uint32_t second_mask = mask(in.src[1]);
                const bool mask_first = first_mask && zero(in.src[1]) &&
                    (in.opcode == 0x06 || in.opcode == 0x07 || in.opcode == 0x08 ||
                     in.opcode == 0x0b);
                const bool mask_second = second_mask && zero(in.src[0]) &&
                    (in.opcode == 0x06 || in.opcode == 0x07 || in.opcode == 0x09 ||
                     in.opcode == 0x0a);
                if (mask_first || mask_second) {
                    const uint32_t any = b.native_subgroup_size
                        ? b.native_wave_any(mask_first ? first_mask : second_mask)
                        : b.guest_wave_any(mask_first ? first_mask : second_mask);
                    const bool invert = in.opcode == 0x06 ||
                        (mask_first ? in.opcode == 0x0b : in.opcode == 0x09);
                    rs.scc = invert ? b.logical_not(any) : any;
                    return true;
                }
            }
            //
            // A B64 compare may consume EXEC, VCC, or a saved wave mask. Those values intentionally
            // have no scalar-data representation: one SPIR-V bool represents this invocation's bit.
            // At a wave-uniform fragment site, reduce the per-lane mismatch across the enforced
            // 64-lane subgroup. This is the exact SCC result of s_cmp_eq/lg_u64 and, unlike reading
            // VCC_LO as uint data, also preserves masks built by vector comparisons and saveexec.
            if ((in.opcode == 0x12 || in.opcode == 0x13) && allow_wave && b.is_fragment) {
                auto mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 106 || o.value == 107) return rs.vcc;
                    if (o.value == 126 || o.value == 127) return rs.exec;
                    if (o.kind == OperandKind::SGPR) {
                        auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                    }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    return 0;
                };
                const uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (m0 && m1) {
                    const uint32_t mismatch = b.bsel(m0, b.logical_not(m1), m1);
                    const uint32_t different = b.fragment_wave_any(mismatch);
                    rs.scc = in.opcode == 0x12 ? b.logical_not(different) : different;
                    return true;
                }
            }
            uint32_t a = val(in.src[0]), c = val(in.src[1]);
            switch (in.opcode) {
                case 0x00: case 0x06: rs.scc = b.ucmp(Op_IEqual, a, c); break;        // s_cmp_eq_i32/u32
                case 0x01: case 0x07: rs.scc = b.ucmp(Op_INotEqual, a, c); break;     // s_cmp_lg_i32/u32
                case 0x02: rs.scc = b.scmp(Op_SGreaterThan, a, c); break;             // s_cmp_gt_i32
                case 0x03: rs.scc = b.scmp(Op_SGreaterThanEqual, a, c); break;        // s_cmp_ge_i32
                case 0x04: rs.scc = b.scmp(Op_SLessThan, a, c); break;                // s_cmp_lt_i32
                case 0x05: rs.scc = b.scmp(Op_SLessThanEqual, a, c); break;           // s_cmp_le_i32
                case 0x08: rs.scc = b.ucmp(Op_UGreaterThan, a, c); break;             // s_cmp_gt_u32
                case 0x09: rs.scc = b.ucmp(Op_UGreaterThanEqual, a, c); break;        // s_cmp_ge_u32
                case 0x0A: rs.scc = b.ucmp(Op_ULessThan, a, c); break;                // s_cmp_lt_u32
                case 0x0B: rs.scc = b.ucmp(Op_ULessThanEqual, a, c); break;           // s_cmp_le_u32
                case 0x12: case 0x13: {                                                // s_cmp_eq/lg_u64
                    // AMD RDNA2 ISA 12.4: each encoded source names the low half of an SGPR pair.
                    // Inline integer/literal sources are extended to 64 bits; Astro compares
                    // s[2:3] against inline zero while rebuilding a wave mask.
                    auto high = [&](const Operand& o) -> uint32_t {
                        if (o.kind == OperandKind::SGPR ||
                            (o.kind == OperandKind::Special && o.value >= 106 && o.value < 124)) {
                            auto it = rs.sreg.find(o.value + 1);
                            if (it != rs.sreg.end()) return it->second;
                            ok = false;
                            return b.uconst(0);
                        }
                        if (o.kind == OperandKind::InlineInt)
                            return b.uconst(o.value < 0 ? UINT32_MAX : 0);
                        // An inline FLOAT in a 64-bit operand supplies the DOUBLE bit pattern
                        // (significant bits in the HIGH dword — e.g. 1.0 -> 0x3FF00000_00000000,
                        // 1/(2*pi) -> 0x3fc45f30_6dc9c882), which val() cannot model (it returned
                        // the f32 pattern as the LOW dword). Never observed live — reject.
                        if (o.kind == OperandKind::InlineFloat) { ok = false; return b.uconst(0); }
                        if (o.kind == OperandKind::Literal ||
                            (o.kind == OperandKind::Special && o.value == 125))
                            return b.uconst(0);
                        ok = false;
                        return b.uconst(0);
                    };
                    const uint32_t equal = b.land(
                        b.ucmp(Op_IEqual, a, c),
                        b.ucmp(Op_IEqual, high(in.src[0]), high(in.src[1])));
                    rs.scc = in.opcode == 0x12 ? equal : b.logical_not(equal);
                    break;
                }
                default: ok = false;
            }
            return true;
        }
        case Rdna2Format::SOPK: {
            // 16-bit-immediate scalar ops. The decoder sign-extends simm16 for signed operations;
            // unsigned comparisons use the original 16-bit bit pattern. SOPK comparisons name their
            // scalar source in the encoded SDST field and write only SCC.
            switch (in.opcode) {
                case 0x00:                                  // s_movk_i32
                    rs.sreg[in.dst.value] = b.uconst((uint32_t)in.simm16);
                    rs.sreg_srt.erase(in.dst.value);
                    break;
                case 0x10: {                                // s_mulk_i32 (read-modify-write)
                    // D = low32(D * sign_extend(SIMM16)). Unlike s_addk_i32, MULK does not write
                    // SCC. Astro Bot uses the exact `s_mulk_i32 vcc_lo, 276` word to turn a
                    // scalar table index into a byte offset before s_buffer_load_dwordx4.
                    const uint32_t a = val(in.dst);
                    rs.sreg[in.dst.value] = b.ibin(
                        Op_IMul, a, b.uconst(static_cast<uint32_t>(in.simm16)));
                    rs.sreg_srt.erase(in.dst.value);
                    break;
                }
                case 0x03: case 0x04: case 0x05: case 0x06:
                case 0x07: case 0x08: {                      // s_cmpk_{eq,lg,gt,ge,lt,le}_i32
                    const uint32_t a = val(in.dst);
                    const uint32_t c = b.uconst((uint32_t)in.simm16);
                    switch (in.opcode) {
                        case 0x03: rs.scc = b.ucmp(Op_IEqual, a, c); break;
                        case 0x04: rs.scc = b.ucmp(Op_INotEqual, a, c); break;
                        case 0x05: rs.scc = b.scmp(Op_SGreaterThan, a, c); break;
                        case 0x06: rs.scc = b.scmp(Op_SGreaterThanEqual, a, c); break;
                        case 0x07: rs.scc = b.scmp(Op_SLessThan, a, c); break;
                        case 0x08: rs.scc = b.scmp(Op_SLessThanEqual, a, c); break;
                    }
                    break;
                }
                case 0x09: case 0x0A: case 0x0B:
                case 0x0C: case 0x0D: case 0x0E: {           // s_cmpk_{eq,lg,gt,ge,lt,le}_u32
                    const uint32_t a = val(in.dst);
                    const uint32_t c = b.uconst((uint32_t)(uint16_t)in.simm16);
                    switch (in.opcode) {
                        case 0x09: rs.scc = b.ucmp(Op_IEqual, a, c); break;
                        case 0x0A: rs.scc = b.ucmp(Op_INotEqual, a, c); break;
                        case 0x0B: rs.scc = b.ucmp(Op_UGreaterThan, a, c); break;
                        case 0x0C: rs.scc = b.ucmp(Op_UGreaterThanEqual, a, c); break;
                        case 0x0D: rs.scc = b.ucmp(Op_ULessThan, a, c); break;
                        case 0x0E: rs.scc = b.ucmp(Op_ULessThanEqual, a, c); break;
                    }
                    break;
                }
                // s_waitcnt_vscnt/vmcnt/expcnt/lgkmcnt: wait-for-counter — benign no-ops in our
                // synchronous model (like SOPP s_waitcnt). These are SOPK on gfx10, NOT SOPP 0x7d
                // as previously claimed (that case was unreachable dead code, and a real
                // s_waitcnt_vscnt — routinely emitted after buffer/image stores — failed the whole
                // shader recompile via this default). VERIFIED(round-trip llvm-mc gfx1010):
                // vscnt=0xBBFD0000 (op 0x17), vmcnt=0xBC7D0000 (0x18), expcnt=0xBCFD0000 (0x19),
                // lgkmcnt=0xBD7D0000 (0x1A).
                case 0x17: case 0x18: case 0x19: case 0x1A: break;
                default: ok = false;
            }
            return true;
        }
        case Rdna2Format::VOP1: {
            uint32_t a = val(in.src[0]); uint32_t old_d = vreg_old(b, rs, in.dst.value);
            uint32_t dpp_active = 0;
            // DPP16 quad_perm on src0 (#273): fragment shaders reconstruct the selected quad lane
            // from derivatives. Compute shaders use the exact subgroup quad-swap operation for the
            // three XOR permutations emitted by Astro Bot's blur kernels: horizontal (0xb1), vertical
            // (0x4e), and diagonal (0x1b). Quad boundaries are architectural and remain exact even
            // when the host subgroup width differs from the guest wave width.
            if (in.has_dpp) {
                const bool row_shr = in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11Fu;
                if (row_shr) {
                    // The portable NGG vertex shell represents the one live guest lane as lane 0.
                    // Every non-zero row-right shift therefore addresses a lane before the start of
                    // its row; BOUND_CTRL=1 supplies the architectural zero.  An unbounded access
                    // retains the prior destination value and cannot be represented here, so reject.
                    if (b.is_fragment || (!b.is_compute && !in.dpp_bound_ctrl) ||
                        (b.is_compute && in.opcode != 0x01)) {
                        ok = false; return true;
                    }
                    if (b.is_compute) {
                        b.mark_subgroup_min16();
                        const uint32_t shift = in.dpp_ctrl - 0x110u;
                        const uint32_t lane = b.subgroup_local_id();
                        const uint32_t row_lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(15));
                        dpp_active = b.ucmp(Op_UGreaterThanEqual, row_lane, b.uconst(shift));
                        const uint32_t source_lane = b.sel(
                            dpp_active, b.ibin(Op_ISub, lane, b.uconst(shift)), lane);
                        const uint32_t shuffled = b.subgroup_shuffle(a, source_lane);
                        a = in.dpp_bound_ctrl ? b.sel(dpp_active, shuffled, b.uconst(0))
                                              : shuffled;
                    } else {
                        a = b.uconst(0);
                    }
                } else {
                if (b.is_fragment) {
                    if (in.opcode != 0x01) { ok = false; return true; }
                    a = b.dpp_quad(a, in.dpp_ctrl);
                } else if (b.is_compute) {
                    // DPP transforms SRC0 before the VOP1 operation. v_mov and the numeric
                    // v_cvt_u32_f32 used by UE light-grid kernels are lane-pure afterward.
                    if (in.opcode != 0x01 && in.opcode != 0x07) { ok = false; return true; }
                    a = b.subgroup_quad_permute(a, in.dpp_ctrl);
                } else {
                    ok = false; return true;
                }
                }
            }
            // SDWA float source modifiers on src0 (abs then neg). v_cvt_f32_f16 applies them after
            // selecting and unpacking the requested half below; applying them to the packed u32 as
            // though it were an f32 changes both the value and the selected sign bit.
            if (in.opcode != 0x0B) {
                if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
                if (in.src_neg[0]) a = b.fbin(Op_FSub, b.uconst(0), a);
            }
            if ((in.opcode == 0x05 || in.opcode == 0x06) && in.sdwa_src0_sel <= 5) {
                const uint32_t bits = in.sdwa_src0_sel <= 3 ? 8u : 16u;
                const uint32_t offset = in.sdwa_src0_sel <= 3
                    ? 8u * in.sdwa_src0_sel : 16u * (in.sdwa_src0_sel - 4u);
                a = ((in.words[1] >> 19) & 1u)
                    ? b.bfe_s(a, b.uconst(offset), b.uconst(bits))
                    : b.bfe_u(a, b.uconst(offset), b.uconst(bits));
            }
            if (in.opcode == 0x02) {   // v_readfirstlane_b32: SGPR dst = value of the lowest active lane
                // Cross-lane broadcast. Our per-lane scalar model has no cross-lane reduction, so we use
                // THIS lane's value. SPECULATIVE(confidence: med): exact only when src0 is wave-uniform —
                // which is the standard use (reading a uniformly-computed VGPR into an SGPR, e.g. the
                // integer-divide reciprocal in the game's shaders). Writes an SGPR, not a VGPR.
                rs.sreg[in.dst.value] = a;
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if (in.opcode == 0x42) {   // v_movreld_b32: VGPR[VDST + M0] = SRC0
                // RDNA2 ISA sec. 6.6. M0 is a runtime unsigned VGPR offset. Represent the indexed
                // write as selects over every statically referenced destination at or above VDST;
                // an out-of-range destination is unobservable because no later instruction names it.
                // This preserves exact loop-carried SSA while avoiding an architectural 256-word
                // register array for shaders that never use relative source addressing.
                auto m0 = rs.sreg.find(124);
                if (m0 == rs.sreg.end()) { ok = false; return true; }
                for (int reg = in.dst.value; reg <= rs.max_vgpr; ++reg) {
                    const uint32_t old = vreg_old(b, rs, reg);
                    rs.vreg[reg] = b.sel(
                        b.ucmp(Op_IEqual, m0->second,
                               b.uconst(static_cast<uint32_t>(reg - in.dst.value))),
                        a, old);
                    predicate_write(b, rs, reg, old);
                }
                return true;
            }
            uint32_t& d = vreg[in.dst.value];
            // WORD-select v_mov_b32_sdwa (#273): extract the selected 16-bit source half and insert it
            // into the selected dest half, preserving the other (the f16 half-move; decode accepted
            // only dst WORD_0/1 + PRESERVE with src DWORD/WORD_0/WORD_1).
            if (in.opcode == 0x01 && in.sdwa_dst_sel != 6) {
                uint32_t v = a;
                if (in.sdwa_src0_sel == 5)      v = b.ibin(Op_ShiftRightLogical, a, b.uconst(16));
                uint32_t v16 = b.ibin(Op_BitwiseAnd, v, b.uconst(0xFFFFu));
                d = (in.sdwa_dst_sel == 4)
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), v16)
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, v16, b.uconst(16)));
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // BYTE-select v_mov_b32_sdwa (#273 — DOLL's title post PSes unpack a packed dword:
            // `v_mov_b32_sdwa v6, v15 src0_sel:BYTE_0`): dst is the whole dword (UNUSED_PAD), so the
            // result is the selected byte zero-extended.
            if (in.opcode == 0x01 && in.sdwa_dst_sel == 6 && in.sdwa_src0_sel <= 3) {
                d = b.bfe_u(a, b.uconst(8u * in.sdwa_src0_sel), b.uconst(8));
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // Packed unary f16 SDWA: select one source half, compute in f32, round back to f16, and
            // insert while preserving the opposite destination half. Trig input is in revolutions.
            if ((in.opcode == 0x54 || in.opcode == 0x55 || in.opcode == 0x61) &&
                in.sdwa_dst_sel != 6) {
                uint32_t raw = a;   // inline constants: f16-width encoding, not the f32 pattern
                if (in.src[0].kind == OperandKind::InlineFloat)
                    raw = b.uconst(inline_float_f16_bits(in.src[0].value)
                                   << (in.sdwa_src0_sel == 5 ? 16 : 0));
                else if (in.src[0].kind == OperandKind::InlineInt)
                    raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
                uint32_t x = b.unpack_half(raw, in.sdwa_src0_sel == 5 ? 1 : 0);
                uint32_t result = in.opcode == 0x54 ? b.frcp(x)
                                : in.opcode == 0x55 ? b.fext1(Glsl_Sqrt, x)
                                : b.fext1(Glsl_Cos,
                                          b.fbin(Op_FMul, x, b.uconst(fbits(6.28318530717958647692f))));
                uint32_t r16 = b.pack_half_lo(result);
                d = in.sdwa_dst_sel == 5
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // v_cvt_f16_f32_sdwa inserts the converted half into the selected destination word and
            // preserves the other word (unlike the plain form, whose result occupies the low half).
            if (in.opcode == 0x0A && in.sdwa_dst_sel != 6) {
                uint32_t r16 = b.pack_half_lo(a);
                d = in.sdwa_dst_sel == 5
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // v_cvt_i32_f32_sdwa converts the full source, then inserts the selected low/high
            // result word while preserving the other destination word. The decoder accepts only
            // this exact WORD + UNUSED_PRESERVE + DWORD-source subset.
            if (in.opcode == 0x08 && in.sdwa_dst_sel != 6) {
                const uint32_t result = b.cvt_f2i(a);
                const uint32_t word = b.ibin(Op_BitwiseAnd, result, b.uconst(0xFFFFu));
                d = in.sdwa_dst_sel == 5
                    ? b.ibin(Op_BitwiseOr,
                             b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, word, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr,
                             b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), word);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            switch (in.opcode) {
                case 0x00: return true;                              // v_nop — no-op (writes nothing; common
                                                                     // scheduling/hazard filler in real shaders)
                case 0x01: d = a; break;                              // v_mov_b32
                case 0x05: d = b.cvt_i2f(a); break;                   // v_cvt_f32_i32
                case 0x06: d = b.cvt_u2f(a); break;                   // v_cvt_f32_u32
                case 0x07: d = b.cvt_f2u(a); break;                   // v_cvt_u32_f32
                case 0x08: d = b.cvt_f2i(a); break;                   // v_cvt_i32_f32
                // f16<->f32 converts (#273 — DOLL's title post PSes carry f16 intermediates):
                // v_cvt_f16_f32 packs into the LOW half (high bits zero); v_cvt_f32_f16 unpacks the
                // low half. VERIFIED(round-trip llvm-mc gfx1030: 0x0a/0x0b).
                case 0x0A:
                    // Plain-form 16-bit results write D.f16_lo and PRESERVE bits [31:16] (the gfx10
                    // 16-bit-VALU contract; zero-fill is only the SDWA DWORD+UNUSED_PAD behavior,
                    // which the decoder marks with has_sdwa — the accepted cvt SDWA subsets all use
                    // WORD dsts and take the preserve branch above, so has_sdwa here means PAD).
                    d = in.has_sdwa
                        ? b.pack_half_lo(a)
                        : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)),
                                 b.pack_half_lo(a));
                    break;                                          // v_cvt_f16_f32
                case 0x0B: {                                        // v_cvt_f32_f16 (SDWA may select high half)
                    // An inline constant in a 16-bit operand position supplies its f16-width
                    // encoding (float) or raw two's-complement bits (int) — NOT the f32 pattern
                    // val() materializes (unpacking THAT low half turned inline 1.0 into 0.0).
                    uint32_t raw = a;
                    if (in.src[0].kind == OperandKind::InlineFloat)
                        raw = b.uconst(inline_float_f16_bits(in.src[0].value)
                                       << (in.sdwa_src0_sel == 5 ? 16 : 0));
                    else if (in.src[0].kind == OperandKind::InlineInt)
                        raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
                    d = b.unpack_half(raw, in.sdwa_src0_sel == 5 ? 1 : 0);
                    if (in.src_abs[0]) d = b.fext1(Glsl_FAbs, d);
                    if (in.src_neg[0]) d = b.fneg(d);
                    break;
                }
                case 0x0D:                                          // v_cvt_flr_i32_f32
                    // AMD RDNA2 ISA: floor the f32 value before converting to signed i32. This is
                    // observably different from v_cvt_i32_f32's truncation for negative fractions.
                    d = b.cvt_f2i(b.fext1(Glsl_Floor, a));
                    break;
                // v_cvt_off_f32_i4: sign-extend the low 4-bit integer and scale by 1/16.
                // AMD RDNA2 ISA: "4-bit signed int to 32-bit float"; LLVM's intrinsic
                // contract specifies result = 0.0625f * src_i4. This is the only opcode
                // that blocked The Messenger's 1024x32 grading-LUT producer (#527).
                case 0x0E:
                    d = b.fbin(Op_FMul,
                               b.cvt_i2f(b.bfe_s(a, b.uconst(0), b.uconst(4))),
                               b.uconst(fbits(0.0625f)));
                    break;
                // AMD RDNA2 ISA: select one unsigned byte from the source dword and convert it
                // directly to f32. The opcode number selects BYTE_0 through BYTE_3.
                case 0x11: case 0x12: case 0x13: case 0x14:
                    d = b.cvt_u2f(b.bfe_u(a,
                                          b.uconst(8u * (in.opcode - 0x11u)),
                                          b.uconst(8)));
                    break;
                case 0x20: d = b.fext1(Glsl_Fract, a); break;         // v_fract_f32
                case 0x21: d = b.fext1(Glsl_Trunc, a); break;         // v_trunc_f32
                case 0x22: d = b.fext1(Glsl_Ceil, a); break;          // v_ceil_f32
                case 0x23: d = b.fext1(Glsl_RoundEven, a); break;     // v_rndne_f32 (round to nearest even)
                case 0x24: d = b.fext1(Glsl_Floor, a); break;         // v_floor_f32
                case 0x25: d = b.fext1(Glsl_Exp2, a); break;          // v_exp_f32 (2^x)
                case 0x27: d = b.fext1(Glsl_Log2, a); break;          // v_log_f32 (log2)
                case 0x2A: d = b.frcp(a); break;                      // v_rcp_f32
                case 0x2B: d = b.frcp(a); break;                      // v_rcp_iflag_f32 (~= v_rcp_f32)
                case 0x2E: d = b.fext1(Glsl_InverseSqrt, a); break;   // v_rsq_f32
                case 0x33: d = b.fext1(Glsl_Sqrt, a); break;          // v_sqrt_f32
                // v_sin_f32 (0x35) / v_cos_f32 (0x36): the RDNA trig input is in REVOLUTIONS (units of
                // 2π radians) — the compiler pre-multiplies by 1/2π (0.15915494) before these, so
                // d = sin/cos(2π·src). VERIFIED(round-trip llvm-mc gfx1010: 0x7e026b02/0x7e026d02 →
                // v_sin/cos_f32; DOLL's dither PS does exactly `v_mul 0.15915494, x` → `v_cos`).
                // CONFIDENCE: HIGH (RDNA2 ISA V_SIN_F32: D = sin(S0 * 2π)).
                case 0x35: d = b.fext1(Glsl_Sin, b.fbin(Op_FMul, a, b.uconst(fbits(6.28318530717958647692f)))); break;
                case 0x36: d = b.fext1(Glsl_Cos, b.fbin(Op_FMul, a, b.uconst(fbits(6.28318530717958647692f)))); break;
                case 0x37: d = b.iun(Op_Not, a); break;               // v_not_b32
                case 0x38: d = b.iun(Op_BitReverse, a); break;        // v_bfrev_b32
                case 0x43: {   // v_movrels_b32: dst = VGPR[src0# + M0] (relative-indexed VGPR read)
                    // M0 is written by plain scalar ALU (s_mov/s_or m0, … decode dst as SGPR 124), so
                    // its per-invocation value lives in rs.sreg[124]. The source register NUMBER is
                    // src0# + M0 — a runtime value — so lower to a bounded select chain over every
                    // tracked VGPR at src0#+k (k = M0 candidate): dst = Σ sel(m0==k, vreg[src0+k]).
                    // Matches the hardware contract for all in-range M0 (reading an unwritten VGPR is
                    // undefined on HW too — those candidates read our 0 placeholder). An UNTRACKED M0
                    // rejects, never silently indexes 0. VERIFIED(round-trip llvm-mc gfx1010:
                    // 0x7e408706 → v_movrels_b32_e32 v32, v6; DOLL UI/skinned VS #273).
                    auto m0it = rs.sreg.find(124);
                    if (m0it == rs.sreg.end()) { ok = false; break; }
                    uint32_t acc = b.uconst(0);
                    const int base = in.src[0].value;
                    for (const auto& kv : vreg) {
                        if (kv.first < base || !kv.second) continue;   // (!kv.second: the dst slot the
                                                                       // enclosing `vreg[dst]` may have
                                                                       // default-inserted — no value yet)
                        acc = b.sel(b.ucmp(Op_IEqual, m0it->second, b.uconst((uint32_t)(kv.first - base))),
                                    kv.second, acc);
                    }
                    d = acc; break;
                }
                default: ok = false;
            }
            // SDWA output modifiers: OMOD scale (×2/×4/×0.5) then CLAMP saturate, on FLOAT-result
            // opcodes only (mirrors the VOP2/VOP3 fresult path; DOLL VS: `v_exp_f32_sdwa … clamp`).
            // A modifier on a non-float-result op would silently drop — reject loudly instead.
            if (ok && (in.omod || in.clamp)) switch (in.opcode) {
                case 0x05: case 0x06: case 0x0B: case 0x0E: case 0x11: case 0x12: case 0x13: case 0x14:
                case 0x20: case 0x21: case 0x22: case 0x23: case 0x24:
                case 0x25: case 0x27: case 0x2A: case 0x2B: case 0x2E: case 0x33: case 0x35: case 0x36:
                    if      (in.omod == 1) d = b.fbin(Op_FMul, d, b.uconst(fbits(2.0f)));
                    else if (in.omod == 2) d = b.fbin(Op_FMul, d, b.uconst(fbits(4.0f)));
                    else if (in.omod == 3) d = b.fbin(Op_FMul, d, b.uconst(fbits(0.5f)));
                    if (in.clamp) d = b.clamp01(d);
                    break;
                default: ok = false; break;
            }
            if (ok) {
                if (dpp_active && !in.dpp_bound_ctrl) d = b.sel(dpp_active, d, old_d);
                predicate_write(b, rs, in.dst.value, old_d);
            }
            return true;
        }
        case Rdna2Format::VOP2: {
            uint32_t a = val(in.src[0]), c = val(in.src[1]); uint32_t old_d = vreg_old(b, rs, in.dst.value);
            uint32_t dpp_active = 0;
            // DPP16 quad_perm on src0 (#273): fragment FLOAT ops and compute quad swaps.  Bounded
            // ROW_SHR in the proven one-live-lane NGG projection supplies zero for lane 0.
            if (in.has_dpp) {
                const bool row_shr = in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11Fu;
                const bool fop = in.opcode == 0x03 || in.opcode == 0x04 || in.opcode == 0x05 ||
                                 in.opcode == 0x08 || in.opcode == 0x0F || in.opcode == 0x10 ||
                                 in.opcode == 0x1F || in.opcode == 0x2B;
                if (row_shr) {
                    if (b.is_vertex) {
                        if (!b.ngg_one_lane || in.opcode != 0x25 || !in.dpp_bound_ctrl) {
                            ok = false; return true;
                        }
                        a = b.uconst(0);
                    } else if (b.is_fragment) {
                        // Astro's material mask uses unbounded v_or_b32 ROW_SHR:{1,2,4,8}. An
                        // out-of-row or EXEC-inactive source disables the instruction, preserving
                        // the old destination even when VDST and the two sources are distinct.
                        if (in.opcode != 0x1c || in.dpp_bound_ctrl) {
                            ok = false; return true;
                        }
                        a = b.subgroup_row_shr(
                            a, rs.exec, in.dpp_ctrl - 0x110u, &dpp_active);
                    } else {
                        if (!b.is_compute || !fop) { ok = false; return true; }
                        // ROW_SHR:N reads SRC0 from lane-N inside each architectural 16-lane row.
                        // BOUND_CTRL=1 substitutes zero before the row; BOUND_CTRL=0 disables the
                        // instruction for those lanes, preserving the old destination exactly.
                        b.mark_subgroup_min16();
                        const uint32_t shift = in.dpp_ctrl - 0x110u;
                        const uint32_t lane = b.subgroup_local_id();
                        const uint32_t row_lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(15));
                        dpp_active = b.ucmp(Op_UGreaterThanEqual, row_lane, b.uconst(shift));
                        // Never issue a shuffle with an out-of-range unsigned lane after subtraction;
                        // inactive lanes address themselves and are then zeroed or masked off.
                        const uint32_t source_lane = b.sel(
                            dpp_active, b.ibin(Op_ISub, lane, b.uconst(shift)), lane);
                        const uint32_t shuffled = b.subgroup_shuffle(a, source_lane);
                        a = in.dpp_bound_ctrl ? b.sel(dpp_active, shuffled, b.uconst(0))
                                              : shuffled;
                    }
                } else {
                    if (!fop) { ok = false; return true; }
                    if (b.is_fragment) {
                        a = b.dpp_quad(a, in.dpp_ctrl);
                    } else if (b.is_compute) {
                        // DPP transforms src0 only; src1 remains in this lane. A general shuffle covers
                        // all 256 architectural quad_perm tables, including broadcasts and duplicates.
                        a = b.subgroup_quad_permute(a, in.dpp_ctrl);
                    } else {
                        ok = false; return true;
                    }
                }
            }
            // SDWA float source modifiers (only ever set on float ops by the assembler): abs then neg.
            // Packed-f16 ops apply these after selecting/unpacking the half below.
            const bool packed_f16 = in.opcode == 0x32 || in.opcode == 0x33 || in.opcode == 0x35 ||
                                    in.opcode == 0x39 || in.opcode == 0x3A;
            const bool integer_sdwa = in.has_sdwa &&
                (in.opcode == 0x0B || (in.opcode >= 0x11 && in.opcode <= 0x14) ||
                 in.opcode == 0x16 || in.opcode == 0x18 ||
                 (in.opcode >= 0x1A && in.opcode <= 0x1E) ||
                 (in.opcode >= 0x25 && in.opcode <= 0x2A));
            if (integer_sdwa) {
                auto select = [&](uint32_t raw, uint8_t sel) {
                    if (sel <= 3u) return b.bfe_u(raw, b.uconst(8u * sel), b.uconst(8));
                    if (sel <= 5u) return b.bfe_u(raw, b.uconst(16u * (sel - 4u)), b.uconst(16));
                    return raw;
                };
                a = select(a, in.sdwa_src0_sel);
                c = select(c, in.sdwa_src1_sel);
            } else if (!packed_f16) {
                if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
                if (in.src_neg[0]) a = b.fneg(a);
                if (in.src_abs[1]) c = b.fext1(Glsl_FAbs, c);
                if (in.src_neg[1]) c = b.fneg(c);
            }
            uint32_t& d = vreg[in.dst.value];
            switch (in.opcode) {
                case 0x01: {                                         // v_cndmask_b32: dst = vcc ? src1 : src0
                    if (in.sdwa_dst_sel == 6) { d = b.sel(vcc, c, a); break; }
                    auto word = [&](uint32_t raw, uint8_t sel) {
                        if (sel == 5) raw = b.ibin(Op_ShiftRightLogical, raw, b.uconst(16));
                        return b.ibin(Op_BitwiseAnd, raw, b.uconst(0xFFFFu));
                    };
                    uint32_t selected = b.sel(vcc, word(c, in.sdwa_src1_sel), word(a, in.sdwa_src0_sel));
                    d = in.sdwa_dst_sel == 5
                        ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                                 b.ibin(Op_ShiftLeftLogical, selected, b.uconst(16)))
                        : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), selected);
                    break;
                }
                case 0x03: d = b.fbin(Op_FAdd, a, c); break;          // v_add_f32
                case 0x04: d = b.fbin(Op_FSub, a, c); break;          // v_sub_f32
                case 0x05: d = b.fbin(Op_FSub, c, a); break;          // v_subrev_f32 (src1 - src0; e32 form of
                                                                      // VOP3 0x105 — round-trip llvm-mc gfx1010 0x0a020702)
                case 0x08: d = b.fbin(Op_FMul, a, c); break;          // v_mul_f32
                case 0x0B: {                                        // v_mul_u32_u24
                    // Only the low 24 bits of each source participate; the result is the low
                    // 32 bits of the unsigned product (AMD RDNA2 ISA 11.6).
                    const uint32_t mask = b.uconst(0x00FFFFFFu);
                    d = b.ibin(Op_IMul, b.ibin(Op_BitwiseAnd, a, mask),
                                          b.ibin(Op_BitwiseAnd, c, mask));
                    break;
                }
                // v_min/v_max: hardware returns the OTHER operand when exactly one input is NaN
                // (ISA 12.7 ops 15/16: "if (S0 == NaN) D = S1; ..."), which is GLSL NMin/NMax.
                // Plain FMin/FMax are NaN-UNDEFINED — llvmpipe propagates a second-operand NaN, so
                // the ubiquitous max(min(x,K),lo) clamp idiom could yield NaN pixels where hardware
                // yields the bound.
                case 0x0F: d = b.fext2(Glsl_NMin, a, c); break;       // v_min_f32
                case 0x10: d = b.fext2(Glsl_NMax, a, c); break;       // v_max_f32
                case 0x11: d = b.sext2(Glsl_SMin, a, c); break;       // v_min_i32
                case 0x12: d = b.sext2(Glsl_SMax, a, c); break;       // v_max_i32
                case 0x13: d = b.uext2(Glsl_UMin, a, c); break;       // v_min_u32
                case 0x14: d = b.uext2(Glsl_UMax, a, c); break;       // v_max_u32
                case 0x16: { uint32_t sh = b.ibin(Op_BitwiseAnd, a, b.uconst(31));   // v_lshrrev_b32
                             d = b.ibin(Op_ShiftRightLogical, c, sh); break; }       // dst = src1 >> (src0 & 31)
                case 0x18: { uint32_t sh = b.ibin(Op_BitwiseAnd, a, b.uconst(31));   // v_ashrrev_i32
                             d = b.sbin(Op_ShiftRightArithmetic, c, sh); break; }    // dst = src1 >>a (src0 & 31)
                case 0x1A: { uint32_t sh = b.ibin(Op_BitwiseAnd, a, b.uconst(31));   // v_lshlrev_b32
                             d = b.ibin(Op_ShiftLeftLogical, c, sh); break; }        // dst = src1 << (src0 & 31)
                case 0x1B: d = b.ibin(Op_BitwiseAnd, a, c); break;    // v_and_b32
                case 0x1C: d = b.ibin(Op_BitwiseOr,  a, c); break;    // v_or_b32
                case 0x1D: d = b.ibin(Op_BitwiseXor, a, c); break;    // v_xor_b32
                case 0x1E: d = b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c)); break; // v_xnor_b32
                case 0x25: d = b.ibin(Op_IAdd, a, c); break;          // v_add_nc_u32
                case 0x26: d = b.ibin(Op_ISub, a, c); break;          // v_sub_nc_u32
                case 0x27: d = b.ibin(Op_ISub, c, a); break;          // v_subrev_nc_u32 (reverse: src1 - src0)
                // Carry ops (VOP2 e32 form): carry-in + carry-out are VCC. v_add_co_ci(0x28)/
                // v_sub_co_ci(0x29)/v_subrev_co_ci(0x2a). Mirrors the VOP3B 0x128/129/12A logic with VCC.
                case 0x28: case 0x29: case 0x2A: {
                    uint32_t cin = b.sel(vcc, b.uconst(1), b.uconst(0));
                    uint32_t carry;
                    if (in.opcode == 0x28) {                          // (a + c) + cin
                        uint32_t s1 = b.ibin(Op_IAdd, a, c); uint32_t k1 = b.ucmp(Op_ULessThan, s1, a);
                        d = b.ibin(Op_IAdd, s1, cin); uint32_t k2 = b.ucmp(Op_ULessThan, d, s1);
                        carry = b.bsel(k1, b.btrue(), k2);
                    } else {                                          // (x - y) - cin  (subrev swaps)
                        uint32_t x = in.opcode == 0x29 ? a : c, y = in.opcode == 0x29 ? c : a;
                        uint32_t s1 = b.ibin(Op_ISub, x, y); uint32_t k1 = b.ucmp(Op_ULessThan, x, y);
                        d = b.ibin(Op_ISub, s1, cin); uint32_t k2 = b.ucmp(Op_ULessThan, s1, cin);
                        carry = b.bsel(k1, b.btrue(), k2);
                    }
                    // Carry-out masks follow ISA 3.9 like compare results: an EXEC-inactive lane's
                    // VCC bit is written 0, not its raw carry (wave votes must not see phantom bits).
                    vcc = rs.exec_narrowed ? b.land(rs.exec, carry) : carry;
                    break;
                }
                // v_mac_f32 (0x1f) / v_fmac_f32 (0x2b): dst = src0*src1 + dst (accumulate into the dest).
                // mac vs fmac differ only in fused rounding — immaterial here. old_d = the dst accumulator.
                // NOTE(opcode ID): op 0x1f is v_mac_f32 on the PS5's ISA. v_mac_f32 was REMOVED on desktop
                // RDNA2/gfx1030 (where 0x1f is invalid and llvm-mc reads canonical v_dot2c at 0x02), but the
                // PS5 GPU retains the RDNA1/gfx1010 encoding — VERIFIED by round-tripping the scene VS's
                // actual op-0x1f word 0x3e261221 through `llvm-mc -mcpu=gfx1010` → `v_mac_f32_e32`. The VOP3
                // (e64) form of this same op is 0x11f, handled in the VOP3 switch below.
                case 0x1F: case 0x2B: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, c), old_d); break;
                // The four mul-add-with-literal-K ops (K = in.literal). madmk/fmamk = src0*K + src1;
                // madak/fmaak = src0*src1 + K. (mad vs fma differ only in fused rounding — immaterial here.)
                case 0x20: case 0x2C: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, b.uconst(in.literal)), c); break;  // v_madmk / v_fmamk
                case 0x21: case 0x2D: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, c), b.uconst(in.literal)); break;  // v_madak / v_fmaak
                case 0x2F: d = b.pack_half2x16_rtz(a, c); break;      // v_cvt_pkrtz_f16_f32 (e32 form): RTZ clamp (#452)
                case 0x3C: {                                         // v_pk_fmac_f16: packed dst += src0*src1
                    auto fmac_half = [&](uint32_t half) {
                        return b.fbin(Op_FAdd,
                                      b.fbin(Op_FMul, b.unpack_half(a, half), b.unpack_half(c, half)),
                                      b.unpack_half(old_d, half));
                    };
                    uint32_t lo = b.pack_half_lo(fmac_half(0));
                    uint32_t hi = b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(fmac_half(1)), b.uconst(16));
                    d = b.ibin(Op_BitwiseOr, lo, hi);
                    break;
                }
                case 0x32: case 0x33: case 0x35: case 0x39: case 0x3A: { // v_add/sub/mul/max/min_f16
                    // (SDWA WORD_1 = high 16; DWORD/WORD_0 = low 16 — an f16 op reads bits[15:0]);
                    // f16xf16 products are exact in f32, so multiply in f32 and round once to f16.
                    // The 16-bit result inserts into the selected dest half PRESERVING the other
                    // (dst_sel WORD_1 for the SDWA pack idiom; DWORD/WORD_0 = the plain e32 form's
                    // "write [15:0], preserve [31:16]" gfx10 f16-VOP2 contract). #273 (DOLL box-blur).
                    auto source = [&](uint32_t raw, const Operand& operand, uint8_t sel, int k) {
                        // An inline constant in a 16-bit operand carries its f16-width encoding
                        // (float — exact even for 1/(2*pi)) or raw two's-complement bits (int:
                        // inline 1 is the f16 denormal 0x0001, never 1.0). Register/scalar operands
                        // carry packed halves and must be selected before abs/neg.
                        uint32_t v = operand.kind == OperandKind::InlineFloat
                                       ? b.unpack_half(b.uconst(inline_float_f16_bits(operand.value)), 0)
                                   : operand.kind == OperandKind::InlineInt
                                       ? b.unpack_half(b.uconst(static_cast<uint32_t>(operand.value)),
                                                       sel == 5 ? 1 : 0)
                                       : b.unpack_half(raw, sel == 5 ? 1 : 0);
                        if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                        if (in.src_neg[k]) v = b.fneg(v);
                        return v;
                    };
                    uint32_t x = source(a, in.src[0], in.sdwa_src0_sel, 0);
                    uint32_t y = source(c, in.src[1], in.sdwa_src1_sel, 1);
                    uint32_t p = in.opcode == 0x32 ? b.fbin(Op_FAdd, x, y)
                               : in.opcode == 0x33 ? b.fbin(Op_FSub, x, y)
                               : in.opcode == 0x35 ? b.fbin(Op_FMul, x, y)
                               : in.opcode == 0x39 ? b.fext2(Glsl_NMax, x, y)   // NaN -> other operand
                                                   : b.fext2(Glsl_NMin, x, y);
                    if (in.clamp) p = b.clamp01(p);
                    uint32_t r16 = b.pack_half_lo(p);
                    // dst_sel==6 covers TWO encodings the decoder now distinguishes: the SDWA
                    // DWORD+UNUSED_PAD form (zero-fill, has_sdwa) and the plain e32 form, whose
                    // gfx10 contract writes [15:0] and PRESERVES [31:16] (the comment above always
                    // said so; the code zero-filled both, corrupting live packed high halves).
                    d = in.sdwa_dst_sel == 6
                        ? (in.has_sdwa
                               ? r16
                               : b.ibin(Op_BitwiseOr,
                                        b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16))
                      : (in.sdwa_dst_sel == 5)
                        ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                                 b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                        : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
                    break;
                }
                default: ok = false;
            }
            // SDWA output modifier: OMOD scale (×2/×4/×0.5) then CLAMP saturate, on FLOAT-result opcodes
            // only (int ops never carry omod). Mirrors the VOP3 fresult path; a no-op when omod/clamp unset.
            if (ok && (in.omod || in.clamp) && !packed_f16) switch (in.opcode) {
                case 0x03: case 0x04: case 0x05: case 0x08: case 0x0F: case 0x10:
                case 0x1F: case 0x2B: case 0x20: case 0x2C: case 0x21: case 0x2D:
                    if      (in.omod == 1) d = b.fbin(Op_FMul, d, b.uconst(fbits(2.0f)));
                    else if (in.omod == 2) d = b.fbin(Op_FMul, d, b.uconst(fbits(4.0f)));
                    else if (in.omod == 3) d = b.fbin(Op_FMul, d, b.uconst(fbits(0.5f)));
                    if (in.clamp) d = b.clamp01(d);
                    break;
                // A non-float-result opcode carrying a modifier (e.g. an INTEGER SDWA op with CLAMP =
                // integer saturation) is not modeled by the float-domain omod/clamp above, so applying
                // nothing would SILENTLY drop the saturation and emit a valid-but-wrong shader. Reject
                // loudly instead — the same fail-visibly-over-miscompile discipline as the forward-if
                // clamp (#129/#174). The guard means this only fires for a modifier-carrying op.
                default: ok = false; break;
            }
            if (ok) {
                if (dpp_active && !in.dpp_bound_ctrl) d = b.sel(dpp_active, d, old_d);
                predicate_write(b, rs, in.dst.value, old_d);
            }
            return true;
        }
        case Rdna2Format::VOPC: {                                     // v_cmp_* -> VCC; v_cmpx_* also -> EXEC
            const uint32_t ra = val(in.src[0]), rc = val(in.src[1]);  // raw bits (f16 compares re-derive)
            uint32_t a = ra, c = rc;
            // Float source modifiers (abs then neg — hardware order), set only on FLOAT compares by the
            // assembler (VOP3-encoded e64 or SDWA forms; e.g. DOLL's `v_cmp_gt_f32_sdwa vcc, |v5|, s4`).
            if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
            if (in.src_neg[0]) a = b.fneg(a);
            if (in.src_abs[1]) c = b.fext1(Glsl_FAbs, c);
            if (in.src_neg[1]) c = b.fneg(c);
            // v_cmpx_* shares each type's compare set at base+0x10 (f32 0x10-0x1f, i32 0x90-0x9f,
            // u32 0xd0-0xdf); it writes EXEC in addition to VCC. Map to the base compare, then narrow.
            uint32_t op = in.opcode;
            bool is_cmpx = vopc_is_cmpx(op);
            if (is_cmpx && !allow_exec_update) { ok = false; return true; }
            uint32_t eff = is_cmpx ? op - 0x10 : op;
            const bool integer_compare =
                (eff >= 0x81u && eff <= 0x86u) ||
                (eff >= 0xC1u && eff <= 0xC6u);
            if (integer_compare) {
                auto sdwa_integer = [&](uint32_t raw, uint8_t sel) {
                    if (sel <= 3u) return b.bfe_u(raw, b.uconst(8u * sel), b.uconst(8));
                    if (sel <= 5u) return b.bfe_u(raw, b.uconst(16u * (sel - 4u)), b.uconst(16));
                    return raw;
                };
                a = sdwa_integer(ra, in.sdwa_src0_sel);
                c = sdwa_integer(rc, in.sdwa_src1_sel);
            }
            uint32_t cmp = 0;
            switch (eff) {
                case 0x00: cmp = b.bfalse(); break;                              // v_cmp_f_f32
                case 0x01: cmp = b.fcmp(Op_FOrdLessThan, a, c); break;         // v_cmp_lt_f32
                case 0x02: cmp = b.fcmp(Op_FOrdEqual, a, c); break;            // v_cmp_eq_f32
                case 0x03: cmp = b.fcmp(Op_FOrdLessThanEqual, a, c); break;    // v_cmp_le_f32
                case 0x04: cmp = b.fcmp(Op_FOrdGreaterThan, a, c); break;      // v_cmp_gt_f32
                case 0x05: cmp = b.fcmp(Op_FOrdNotEqual, a, c); break;         // v_cmp_lg_f32
                case 0x06: cmp = b.fcmp(Op_FOrdGreaterThanEqual, a, c); break; // v_cmp_ge_f32
                case 0x07: {                                                    // v_cmp_o_f32
                    const uint32_t a_ordered = b.fcmp(Op_FOrdEqual, a, a);
                    const uint32_t c_ordered = b.fcmp(Op_FOrdEqual, c, c);
                    cmp = b.land(a_ordered, c_ordered);
                    break;
                }
                case 0x08: {                                                    // v_cmp_u_f32
                    const uint32_t a_nan = b.fcmp(Op_FUnordNotEqual, a, a);
                    const uint32_t c_nan = b.fcmp(Op_FUnordNotEqual, c, c);
                    cmp = b.lor(a_nan, c_nan);
                    break;
                }
                // NaN-inclusive f32 compares (the "n"-prefix set is the unordered negation of 0x1-0x6):
                case 0x09: cmp = b.fcmp(Op_FUnordLessThan, a, c); break;       // v_cmp_nge_f32 = !(a>=b)
                case 0x0A: cmp = b.fcmp(Op_FUnordEqual, a, c); break;          // v_cmp_nlg_f32 = !(a!=b)
                case 0x0B: cmp = b.fcmp(Op_FUnordLessThanEqual, a, c); break;  // v_cmp_ngt_f32 = !(a>b)
                case 0x0C: cmp = b.fcmp(Op_FUnordGreaterThan, a, c); break;    // v_cmp_nle_f32 = !(a<=b)
                case 0x0D: cmp = b.fcmp(Op_FUnordNotEqual, a, c); break;       // v_cmp_neq_f32 = !(a==b)
                case 0x0E: cmp = b.fcmp(Op_FUnordGreaterThanEqual, a, c); break;// v_cmp_nlt_f32 = !(a<b)
                case 0x0F: cmp = b.btrue(); break;                              // v_cmp_tru_f32
                case 0x81: cmp = b.scmp(Op_SLessThan, a, c); break;            // v_cmp_lt_i32
                case 0x82: cmp = b.ucmp(Op_IEqual, a, c); break;               // v_cmp_eq_i32
                case 0x83: cmp = b.scmp(Op_SLessThanEqual, a, c); break;       // v_cmp_le_i32
                case 0x84: cmp = b.scmp(Op_SGreaterThan, a, c); break;         // v_cmp_gt_i32
                case 0x85: cmp = b.ucmp(Op_INotEqual, a, c); break;            // v_cmp_ne_i32 (sign-agnostic)
                case 0x86: cmp = b.scmp(Op_SGreaterThanEqual, a, c); break;    // v_cmp_ge_i32
                case 0x88: {                                                   // v_cmp_class_f32
                    // CLASS tests the raw IEEE-754 category rather than doing a floating-point
                    // comparison. Keep this entirely in the integer domain so signalling/quiet
                    // NaNs and the sign of zero/NaN survive on every SPIR-V target.
                    uint32_t class_raw = ra;
                    if (in.src_abs[0])
                        class_raw = b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x7fffffffu));
                    if (in.src_neg[0])
                        class_raw = b.ibin(Op_BitwiseXor, class_raw, b.uconst(0x80000000u));
                    const uint32_t sign = b.ucmp(
                        Op_INotEqual,
                        b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x80000000u)), b.uconst(0));
                    const uint32_t exponent =
                        b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x7f800000u));
                    const uint32_t mantissa =
                        b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x007fffffu));
                    const uint32_t exponent_zero = b.ucmp(Op_IEqual, exponent, b.uconst(0));
                    const uint32_t exponent_all = b.ucmp(Op_IEqual, exponent, b.uconst(0x7f800000u));
                    const uint32_t mantissa_zero = b.ucmp(Op_IEqual, mantissa, b.uconst(0));
                    const uint32_t quiet_nan = b.ucmp(
                        Op_INotEqual,
                        b.ibin(Op_BitwiseAnd, mantissa, b.uconst(0x00400000u)), b.uconst(0));

                    // AMD's mask order is sNaN, qNaN, -Inf, -normal, -subnormal, -zero,
                    // +zero, +subnormal, +normal, +Inf. Select the input's one-hot class bit,
                    // then test it against SRC1 (Astro's live packet uses 3 = either NaN).
                    const uint32_t nan_class = b.sel(quiet_nan, b.uconst(2), b.uconst(1));
                    const uint32_t inf_class = b.sel(sign, b.uconst(4), b.uconst(512));
                    const uint32_t exp_all_class = b.sel(mantissa_zero, inf_class, nan_class);
                    const uint32_t zero_class = b.sel(sign, b.uconst(32), b.uconst(64));
                    const uint32_t subnormal_class = b.sel(sign, b.uconst(16), b.uconst(128));
                    const uint32_t exp_zero_class = b.sel(mantissa_zero, zero_class, subnormal_class);
                    const uint32_t normal_class = b.sel(sign, b.uconst(8), b.uconst(256));
                    const uint32_t class_bit = b.sel(
                        exponent_all, exp_all_class,
                        b.sel(exponent_zero, exp_zero_class, normal_class));
                    cmp = b.ucmp(Op_INotEqual,
                                 b.ibin(Op_BitwiseAnd, rc, class_bit), b.uconst(0));
                    break;
                }
                case 0xC1: cmp = b.ucmp(Op_ULessThan, a, c); break;            // v_cmp_lt_u32
                case 0xC2: cmp = b.ucmp(Op_IEqual, a, c); break;               // v_cmp_eq_u32
                case 0xC3: cmp = b.ucmp(Op_ULessThanEqual, a, c); break;       // v_cmp_le_u32
                case 0xC4: cmp = b.ucmp(Op_UGreaterThan, a, c); break;         // v_cmp_gt_u32
                case 0xC5: cmp = b.ucmp(Op_INotEqual, a, c); break;            // v_cmp_ne_u32
                case 0xC6: cmp = b.ucmp(Op_UGreaterThanEqual, a, c); break;    // v_cmp_ge_u32
                // f16 compares (0xC8-0xCF; cmpx at +0x10 = 0xD8-0xDF folds here too — DOLL's title
                // post PSes: `v_cmp_lt_f16_sdwa s6, 0, v7`). VERIFIED(round-trip llvm-mc gfx1030:
                // v_cmp_lt/eq/le/gt/lg/ge_f16 = VOPC 0xC9-0xCE). The f16 value lives in the source's
                // LOW half — unpack to f32 and compare there (exact: f16 order-embeds into f32); an
                // inline FLOAT constant is already an f32 value, so it is used directly. abs/neg
                // modifiers are applied after conversion (equivalent, conversion is monotone/exact).
                case 0xC9: case 0xCA: case 0xCB: case 0xCC: case 0xCD: case 0xCE: {
                    auto f16v = [&](int k, uint32_t raw) -> uint32_t {
                        // Inline constants supply their 16-bit operand encoding: floats the f16
                        // pattern (0x3C00 for 1.0), ints raw two's-complement low bits (1 -> the
                        // f16 denormal 0x0001). Reading the raw f32-pattern (float) or the value as
                        // full-width f32 BITS (int) modeled a different number than hardware.
                        uint32_t v = in.src[k].kind == OperandKind::InlineFloat
                                         ? b.unpack_half(b.uconst(inline_float_f16_bits(in.src[k].value)), 0)
                                   : in.src[k].kind == OperandKind::InlineInt
                                         ? b.unpack_half(b.uconst(static_cast<uint32_t>(in.src[k].value)), 0)
                                         : b.unpack_half(raw, in.sdwa_src0_sel == 5u && k == 0 ? 1u :
                                                              in.sdwa_src1_sel == 5u && k == 1 ? 1u : 0u);
                        if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                        if (in.src_neg[k]) v = b.fneg(v);
                        return v;
                    };
                    uint32_t ha = f16v(0, ra), hc = f16v(1, rc);
                    switch (eff) {
                        case 0xC9: cmp = b.fcmp(Op_FOrdLessThan, ha, hc); break;          // v_cmp_lt_f16
                        case 0xCA: cmp = b.fcmp(Op_FOrdEqual, ha, hc); break;             // v_cmp_eq_f16
                        case 0xCB: cmp = b.fcmp(Op_FOrdLessThanEqual, ha, hc); break;     // v_cmp_le_f16
                        case 0xCC: cmp = b.fcmp(Op_FOrdGreaterThan, ha, hc); break;       // v_cmp_gt_f16
                        case 0xCD: cmp = b.fcmp(Op_FOrdNotEqual, ha, hc); break;          // v_cmp_lg_f16
                        default:   cmp = b.fcmp(Op_FOrdGreaterThanEqual, ha, hc); break;  // v_cmp_ge_f16
                    }
                    break;
                }
                default: ok = false;
            }
            // A compare result is a per-lane mask. The e64/SDWAB form can target an SGPR pair (SDST)
            // instead of VCC — track it in sreg_bool so a later v_cndmask_b32_e64 / s_cselect can read it.
            // Otherwise it writes VCC; keep VCC's narrowed-state in sync (106/107 = VCC_LO/HI).
            if (ok) {
                if (is_cmpx) {
                    // v_cmpx writes EXEC ONLY on gfx10 (EXEC &= cmp) — it has NO VCC/SGPR destination.
                    // The old shared handler fell into the `else` and set vcc = cmp for cmpx too,
                    // clobbering a VCC value kept live ACROSS the cmpx, so a later v_cndmask/s_cbranch_vccz/
                    // v_add_co_ci reading VCC got the compare mask instead of the real predicate (#464).
                    rs.exec = b.land(rs.exec, cmp); rs.exec_narrowed = true;
                } else {
                    // ISA 3.9: "VCC[n] = EXEC[n] & (test passed for thread n)" — a lane inactive in
                    // EXEC gets its mask bit forced to 0, never the raw test result. Masking at the
                    // write keeps wave reductions (vccz votes, mbcnt publications, v_writelane mask
                    // spills) from seeing phantom bits from inactive lanes. No-op when EXEC is full.
                    const uint32_t masked = rs.exec_narrowed ? b.land(rs.exec, cmp) : cmp;
                    if (in.dst.kind == OperandKind::SGPR && in.dst.value <= 105) {
                        rs.sreg_bool[in.dst.value] = masked; rs.sreg_bool_narrowed[in.dst.value] = true;
                        if (b.allow_b32_masks &&
                            (b.is_fragment || (b.is_compute && b.wave_size == 32)))
                            rs.sreg_bool_b32.insert(in.dst.value);
                    } else if (b.allow_b32_masks && in.dst.kind == OperandKind::SGPR &&
                               (in.dst.value == 106 || in.dst.value == 107)) {
                        // Wave32 SDWA/e64 compares can explicitly target either one-word VCC half as
                        // an independent saved mask (Astro world-map PC1060 targets VCC_HI, then
                        // consumes it with s_andn2_b32). Keep that physical destination distinct;
                        // only VCC_LO is also the implicit condition used by vccz/cndmask forms.
                        rs.sreg_bool[in.dst.value] = masked;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        rs.sreg_bool_b32.insert(in.dst.value);
                        rs.sreg.erase(in.dst.value);
                        rs.sreg_srt.erase(in.dst.value);
                        if (in.dst.value == 106) vcc = masked;
                    } else {
                        // VOPC architecturally overwrites VCC, including any earlier scalar-data use
                        // of its physical words. In Wave32 the low word is itself the complete mask;
                        // retain that domain marker so a following s_mov_b32 save sees the fresh VCC
                        // rather than stale dispatcher-loaded scalar data.
                        vcc = masked;
                        mask_write_clobbers_pair(rs, 106);
                        rs.sreg_bool_narrowed[106] = true;
                        rs.sreg_bool_narrowed[107] = true;
                        if (b.allow_b32_masks &&
                            (b.is_fragment || (b.is_compute && b.wave_size == 32))) {
                            rs.sreg_bool[106] = masked;
                            rs.sreg_bool_b32.insert(106);
                        }
                    }
                }
            }
            return true;
        }
        case Rdna2Format::VOP3: {
            // SCALAR-SPILL lane slots (#273): v_writelane_b32 (0x361) / v_readlane_b32 (0x360) with a
            // COMPILE-TIME lane index — the pack-scalars-into-a-VGPR's-lanes idiom (DOLL's big post PS
            // spills 19 s_buffer_load results into v36 and reads them back). Per-invocation each
            // (vgpr, lane) is a named wave-uniform scalar. The portable NGG vertex shell additionally
            // projects an ordinary VGPR read from any physical lane onto its one represented live
            // lane; this is the same collapsed-wave contract used for MBCNT and Function LDS. Other
            // compute/fragment stages use a native subgroup shuffle for ordinary VGPR reads and
            // therefore support both scalar and inline lane selectors. Neither op is EXEC-predicated
            // on hardware, so no predicate_write.
            // VERIFIED(round-trip llvm-mc gfx1030: 0xd761 v_writelane_b32 / 0xd760 v_readlane_b32).
            if (in.opcode == 0x361) {                                 // v_writelane_b32 vDST, sSRC, lane
                if (in.src[1].kind != OperandKind::InlineInt || in.src[1].value < 0 || in.src[1].value > 63) {
                    ok = false; return true;
                }
                const int lane = in.src[1].value;
                rs.invalidated_vgpr_lane_slots.erase(in.dst.value);
                uint32_t mask_value = 0;
                bool mask_source = false;
                if (in.src[0].value == 126 || in.src[0].value == 127) {
                    mask_value = rs.exec; mask_source = true;
                } else if ((in.src[0].value == 106 || in.src[0].value == 107) &&
                           !rs.sreg.count(in.src[0].value)) {
                    mask_value = rs.vcc; mask_source = true;
                } else if (in.src[0].kind == OperandKind::SGPR) {
                    auto saved = rs.sreg_bool.find(in.src[0].value);
                    if (saved != rs.sreg_bool.end()) { mask_value = saved->second; mask_source = true; }
                }
                rs.vreg.erase(in.dst.value);                            // spill-array lifetime
                if (mask_source) {
                    rs.vgpr_lane_mask_slots[in.dst.value][lane] = mask_value;
                    auto data = rs.vgpr_lane_slots.find(in.dst.value);
                    if (data != rs.vgpr_lane_slots.end()) {
                        data->second.erase(lane);
                        if (data->second.empty()) rs.vgpr_lane_slots.erase(data);
                    }
                } else {
                    rs.vgpr_lane_slots[in.dst.value][lane] = val(in.src[0]);
                    auto masks = rs.vgpr_lane_mask_slots.find(in.dst.value);
                    if (masks != rs.vgpr_lane_mask_slots.end()) {
                        masks->second.erase(lane);
                        if (masks->second.empty()) rs.vgpr_lane_mask_slots.erase(masks);
                    }
                }
                return true;
            }
            if (in.opcode == 0x360) {                                 // v_readlane_b32 sDST, vSRC, lane
                if (b.is_fragment && b.wave_size == 32 &&
                    in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 32) {
                    ok = false; return true;
                }
                auto vit = rs.vgpr_lane_slots.find(in.src[0].value);
                auto mit = rs.vgpr_lane_mask_slots.find(in.src[0].value);
                const bool vertex_shell = !b.is_compute && !b.is_fragment;
                if (!vertex_shell && vit == rs.vgpr_lane_slots.end() &&
                    mit == rs.vgpr_lane_mask_slots.end()) {
                    auto source = rs.vreg.find(in.src[0].value);
                    if (source == rs.vreg.end()) { ok = false; return true; }
                    const uint32_t selector = val(in.src[1]);
                    if (!ok) return true;
                    if (b.wave_size == 64) b.mark_subgroup_min64();
                    else b.mark_subgroup_min32();
                    const uint32_t native_lane = b.subgroup_local_id();
                    const uint32_t wave_mask = b.uconst(b.wave_size - 1u);
                    const uint32_t wave_base = b.ibin(
                        Op_BitwiseAnd, native_lane, b.uconst(~(b.wave_size - 1u)));
                    const uint32_t source_lane = b.ibin(
                        Op_BitwiseOr, wave_base, b.ibin(Op_BitwiseAnd, selector, wave_mask));
                    const uint32_t result = b.subgroup_shuffle(source->second, source_lane);
                    rs.sreg[in.dst.value] = result;
                    rs.sreg_bool.erase(in.dst.value);
                    rs.sreg_bool_narrowed.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    if (in.dst.value == 106) {
                        const uint32_t bit = b.ibin(Op_BitwiseAnd, result, b.uconst(1));
                        rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                        rs.sreg_bool_narrowed[106] = true;
                    }
                    return true;
                }
                const bool static_lane = in.src[1].kind == OperandKind::InlineInt &&
                                         in.src[1].value >= 0 && in.src[1].value <= 63;
                const bool dynamic_absent_ngg_peer = b.ngg_one_lane &&
                    in.src[1].kind == OperandKind::Special &&
                    in.src[1].value >= 106 && in.src[1].value <= 124;
                if (!static_lane && !dynamic_absent_ngg_peer) {
                    ok = false; return true;
                }
                const int lane = static_lane ? in.src[1].value : -1;
                if (mit != rs.vgpr_lane_mask_slots.end()) {
                    auto sit = mit->second.find(lane);
                    if (sit != mit->second.end()) {
                        rs.sreg_bool[in.dst.value] = sit->second;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        // The compute CFG dispatcher can persist a physical spill lane that is
                        // recycled between scalar-data and wave-mask lifetimes. It exposes both
                        // views after a block join; keep both destination domains when present so
                        // the statically typed consumer selects the representation it needs.
                        if (vit != rs.vgpr_lane_slots.end()) {
                            auto data = vit->second.find(lane);
                            if (data != vit->second.end()) rs.sreg[in.dst.value] = data->second;
                            else rs.sreg.erase(in.dst.value);
                        } else {
                            rs.sreg.erase(in.dst.value);
                        }
                        rs.sreg_srt.erase(in.dst.value);
                        return true;
                    }
                }
                if (vit == rs.vgpr_lane_slots.end()) {
                    if (rs.invalidated_vgpr_lane_slots.count(in.src[0].value)) {
                        ok = false; return true;
                    }
                    // Fragment shaders execute in an enforced 64-lane Vulkan subgroup, so a static
                    // hardware v_readlane maps exactly to a subgroup shuffle. Unlike ordinary VALU,
                    // v_readlane ignores EXEC; do not mask the source or predicate the scalar write.
                    if (b.is_fragment && static_lane) {
                        rs.sreg[in.dst.value] = b.subgroup_shuffle(
                            val(in.src[0]), b.uconst(static_cast<uint32_t>(lane)));
                        rs.sreg_bool.erase(in.dst.value);
                        rs.sreg_bool_narrowed.erase(in.dst.value);
                        rs.sreg_srt.erase(in.dst.value);
                        return true;
                    }
                    // NGG's final wave-packing tail reads peer lanes (Astro Bot uses lanes 63 and 3)
                    // from an ordinary VGPR. A Vulkan vertex invocation models guest lane zero: a
                    // lane-zero read returns this invocation's source, while every absent peer reads
                    // zero. The dynamic form remains exact by selecting on its scalar lane index.
                    if (!b.ngg_one_lane) { ok = false; return true; }
                    const uint32_t source = val(in.src[0]);
                    uint32_t reads_self = 0;
                    if (static_lane) {
                        reads_self = lane == 0 ? b.btrue() : b.bfalse();
                    } else {
                        auto lane_value = rs.sreg.find(in.src[1].value);
                        if (lane_value == rs.sreg.end()) { ok = false; return true; }
                        reads_self = b.ucmp(Op_IEqual, lane_value->second, b.uconst(0));
                    }
                    rs.sreg[in.dst.value] = b.sel(reads_self, source, b.uconst(0));
                    rs.sreg_bool.erase(in.dst.value);
                    rs.sreg_bool_narrowed.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    return true;
                }
                auto sit = vit->second.find(lane);
                if (sit == vit->second.end()) { ok = false; return true; }          // slot never written
                rs.sreg[in.dst.value] = sit->second;   // dst field is the SGPR number (like readfirstlane)
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if (in.opcode == 0x377 || in.opcode == 0x378) {            // v_permlane{x}16_b32
                // AMD RDNA ISA 12.12: SRC1:SRC2 is a packed table of sixteen four-bit lane
                // selectors. PERMLANE16 gathers within the current 16-lane row; PERMLANEX16 gathers
                // from the paired row (0<->1, 2<->3). These are untyped operations; every ordinary
                // VOP3 float modifier is reserved. The two overloaded OPSEL bits were retained by
                // the decoder as FI and BOUND_CTRL.
                const uint32_t reserved_opsel = (in.words[0] >> 13) & 3u;
                if ((!b.is_compute && !b.is_fragment) || reserved_opsel || in.clamp || in.omod ||
                    in.src_abs[0] || in.src_abs[1] || in.src_abs[2] ||
                    in.src_neg[0] || in.src_neg[1] || in.src_neg[2] ||
                    in.src[0].kind != OperandKind::VGPR ||
                    in.src[1].kind == OperandKind::VGPR || in.src[2].kind == OperandKind::VGPR) {
                    ok = false; return true;
                }
                if (in.opcode == 0x378) b.mark_subgroup_min32();
                else b.mark_subgroup_min16();
                uint32_t source_lane = 0;
                const uint32_t shuffled = b.subgroup_permlane16(
                    val(in.src[0]), val(in.src[1]), val(in.src[2]),
                    in.opcode == 0x378, &source_lane);
                uint32_t result = shuffled;
                if (!in.permlane_fetch_inactive) {
                    const uint32_t active_word = b.sel(rs.exec, b.uconst(1), b.uconst(0));
                    const uint32_t fetched_active = b.subgroup_shuffle(active_word, source_lane);
                    const uint32_t source_active = b.ucmp(
                        Op_INotEqual, fetched_active, b.uconst(0));
                    result = b.sel(source_active, shuffled,
                                   in.permlane_bound_ctrl ? b.uconst(0)
                                                          : vreg_old(b, rs, in.dst.value));
                }
                const uint32_t old_d = vreg_old(b, rs, in.dst.value);
                vreg[in.dst.value] = result;
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            uint32_t old_d = vreg_old(b, rs, in.dst.value);
            // Float source with its VOP3 modifiers applied: abs then a sign-bit-exact negate
            // (hardware order neg(abs(x))). Returns raw bits (fbin/fext re-bitcast), so it's a
            // drop-in for val() in the FLOAT ops only — integer VOP3 ops keep val() (modifiers are
            // float-domain; assemblers don't set them on int ops).
            auto fv = [&](int k) -> uint32_t {
                uint32_t bits = val(in.src[k]);
                if (in.src_abs[k]) bits = b.fext1(Glsl_FAbs, bits);
                if (in.src_neg[k]) bits = b.fneg(bits);
                return bits;
            };
            // Output modifiers on a FLOAT result: OMOD scale (×2/×4/×0.5) then CLAMP saturate to [0,1]
            // (hardware order: clamp(omod(x))). Wrap each float op's result through this. Opcodes
            // that route through it mark the CLAMP bit consumed; a set CLAMP on any opcode that
            // does NOT (integer ops, cndmask, the pack family) means unmodeled INTEGER saturation
            // (ISA 6.5) or an unhandled combination — the dispatch tail rejects it fail-visibly
            // instead of silently dropping the saturation.
            bool clamp_routed = false;
            auto fresult = [&](uint32_t bits) -> uint32_t {
                clamp_routed = true;
                if (in.omod == 1)      bits = b.fbin(Op_FMul, bits, b.uconst(fbits(2.0f)));
                else if (in.omod == 2) bits = b.fbin(Op_FMul, bits, b.uconst(fbits(4.0f)));
                else if (in.omod == 3) bits = b.fbin(Op_FMul, bits, b.uconst(fbits(0.5f)));
                if (in.clamp) bits = b.clamp01(bits);
                return bits;
            };
            if (in.opcode == 0x34B) {                                // v_fma_f16: one selected packed half
                auto f16src = [&](int k) {
                    const Operand& operand = in.src[k];
                    uint32_t raw = val(operand);
                    // Inline constants supply their 16-bit operand encoding (see the VOP2 f16 path).
                    uint32_t value = operand.kind == OperandKind::InlineFloat
                                       ? b.unpack_half(b.uconst(inline_float_f16_bits(operand.value)), 0)
                                   : operand.kind == OperandKind::InlineInt
                                       ? b.unpack_half(b.uconst(static_cast<uint32_t>(operand.value)),
                                                       (in.vop3p_opsel >> k) & 1u)
                                       : b.unpack_half(raw, (in.vop3p_opsel >> k) & 1u);
                    if (in.src_abs[k]) value = b.fext1(Glsl_FAbs, value);
                    if (in.src_neg[k]) value = b.fneg(value);
                    return value;
                };
                uint32_t result = fresult(
                    b.fbin(Op_FAdd, b.fbin(Op_FMul, f16src(0), f16src(1)), f16src(2)));
                uint32_t r16 = b.pack_half_lo(result);
                vreg[in.dst.value] = (in.vop3p_opsel & 8u)
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
            } else if (in.opcode == 0x14B || in.opcode == 0x141) {    // v_fma_f32 / v_mad_f32 = src0*src1 + src2
                // v_mad_f32 (op 0x141) is a gfx10.1 (Navi) instruction REMOVED in gfx10.3, so llvm-mc
                // -mcpu=gfx1030 rejects it as invalid — but the PS5 shader compiler targets gfx10.1 and
                // emits it (real game shaders 5,26-29: manual attribute interpolation p0+i*p1). Its result
                // (unfused mul-then-add) maps exactly to OpFMul+OpFAdd; v_fma's fused rounding is
                // immaterial here. VERIFIED(round-trip llvm-mc gfx1010, both directions): VOP3 op 0x141.
                uint32_t m = b.fbin(Op_FMul, fv(0), fv(1));
                vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, m, fv(2)));
            } else if (in.opcode == 0x176) {                          // v_mad_u64_u32
                // AMD RDNA2: {carry,D.u64} = S0.u32*S1.u32 + S2.u64. A literal
                // or positive inline constant used as the 64-bit addend is zero-extended;
                // a negative integer inline constant is sign-extended. Register addends
                // consume the consecutive high register.
                if (in.dst.value >= 255) { ok = false; }
                else {
                    auto high_half = [&](const Operand& operand) -> uint32_t {
                        if (operand.kind == OperandKind::InlineInt)
                            return b.uconst(operand.value < 0 ? 0xFFFFFFFFu : 0u);
                        // An inline FLOAT 64-bit addend supplies the DOUBLE bit pattern (high dword
                        // carries the exponent/mantissa; 1/(2*pi) has a nonzero LOW dword too) —
                        // the previous {f32-bits, 0} model was wrong in both halves. No compiler
                        // emits a float inline as an integer-mad addend: reject, stay fail-visible.
                        if (operand.kind == OperandKind::InlineFloat) { ok = false; return b.uconst(0); }
                        if (operand.kind == OperandKind::Literal)
                            return b.uconst(0);
                        if (operand.kind == OperandKind::Special && operand.value == 125)
                            return b.uconst(0);                       // null pair
                        if (operand.kind == OperandKind::VGPR ||
                            operand.kind == OperandKind::SGPR ||
                            (operand.kind == OperandKind::Special &&
                             operand.value >= 106 && operand.value < 124)) {
                            Operand next = operand;
                            ++next.value;
                            return val(next);
                        }
                        ok = false;
                        return b.uconst(0);
                    };

                    const uint32_t a = val(in.src[0]), c = val(in.src[1]);
                    const uint32_t add_lo = val(in.src[2]), add_hi = high_half(in.src[2]);
                    const uint32_t mul_lo = b.ibin(Op_IMul, a, c);
                    const uint32_t mul_hi = b.umul_hi(a, c);
                    const uint32_t result_lo = b.ibin(Op_IAdd, mul_lo, add_lo);
                    const uint32_t carry_lo = b.ucmp(Op_ULessThan, result_lo, mul_lo);
                    const uint32_t high_sum = b.ibin(Op_IAdd, mul_hi, add_hi);
                    const uint32_t carry_hi0 = b.ucmp(Op_ULessThan, high_sum, mul_hi);
                    const uint32_t carry_word = b.sel(carry_lo, b.uconst(1), b.uconst(0));
                    const uint32_t result_hi = b.ibin(Op_IAdd, high_sum, carry_word);
                    const uint32_t carry_hi1 = b.ucmp(Op_ULessThan, result_hi, high_sum);
                    const uint32_t carry_out = b.lor(carry_hi0, carry_hi1);

                    const int hi_dst = in.dst.value + 1;
                    const uint32_t old_hi = vreg_old(b, rs, hi_dst);
                    vreg[in.dst.value] = result_lo;
                    vreg[hi_dst] = result_hi;
                    predicate_write(b, rs, hi_dst, old_hi);
                    // ISA 3.9 carry-mask rule: an EXEC-inactive lane's bit is written 0, never the
                    // raw carry (wave votes/spills must not see phantom bits from inactive lanes).
                    const uint32_t carry_masked =
                        rs.exec_narrowed ? b.land(rs.exec, carry_out) : carry_out;
                    if (in.sdst.value == 106 || in.sdst.value == 107) rs.vcc = carry_masked;
                    else if (in.sdst.kind == OperandKind::SGPR) rs.sreg_bool[in.sdst.value] = carry_masked;
                    else ok = false;
                }
            } else if (in.opcode == 0x30F || in.opcode == 0x310 || in.opcode == 0x319) {
                // v_add_co_u32 (0x30F) / v_sub_co_u32 (0x310) / v_subrev_co_u32 (0x319): 32-bit add/
                // subtract that writes a carry/borrow-out to the VOP3B sdst mask (VCC or an SGPR pair).
                // No carry-IN — that is the _co_ci_ family (VOP2 0x28-0x2A / VOP3B 0x128-0x12A). Mirrors
                // the VOP2 e32 carry emit (0x28-0x2A above) plus the v_mad_u64_u32 carry-out sdst write.
                // GTA V (PPSA04263) UI/content compute shaders use these; the missing op dropped the whole
                // shader, so its output texture rendered as bare untextured triangles (#1163/#1165).
                // CONFIDENCE: HIGH (RDNA2 ISA 6.5/3.9; coverage test in test_recompile_coverage).
                const uint32_t a = val(in.src[0]), c = val(in.src[1]);
                uint32_t d, carry;
                if (in.opcode == 0x30F) {                     // a + c ; carry = unsigned overflow
                    d = b.ibin(Op_IAdd, a, c);
                    carry = b.ucmp(Op_ULessThan, d, a);
                } else {                                      // sub: a-c ; subrev: c-a ; borrow = x < y
                    const uint32_t x = in.opcode == 0x310 ? a : c;
                    const uint32_t y = in.opcode == 0x310 ? c : a;
                    d = b.ibin(Op_ISub, x, y);
                    carry = b.ucmp(Op_ULessThan, x, y);
                }
                vreg[in.dst.value] = d;
                // ISA 3.9 carry-mask rule: an EXEC-inactive lane's bit is written 0, not its raw carry.
                const uint32_t carry_masked = rs.exec_narrowed ? b.land(rs.exec, carry) : carry;
                if (in.sdst.value == 106 || in.sdst.value == 107) rs.vcc = carry_masked;
                else if (in.sdst.kind == OperandKind::SGPR) rs.sreg_bool[in.sdst.value] = carry_masked;
                else ok = false;
            } else if (in.opcode == 0x169) {                          // v_mul_lo_u32
                vreg[in.dst.value] = b.ibin(Op_IMul, val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x16a) {                          // v_mul_hi_u32 (high 32 bits)
                vreg[in.dst.value] = b.umul_hi(val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x16c) {                          // v_mul_hi_i32 (high 32 bits, signed)
                vreg[in.dst.value] = b.smul_hi(val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x157) {                          // v_med3_f32 = median(s0,s1,s2)
                // ISA op 343: "if (isNan(S0) || isNan(S1) || isNan(S2)) D = V_MIN3_F32(S0,S1,S2)".
                // The min/max legs use the NaN-aware NMin/NMax (return-the-other-operand), and the
                // any-NaN case selects min3 explicitly — the plain max(min(...)) formula does not
                // reproduce the documented NaN fallback on its own.
                uint32_t s0 = fv(0), s1 = fv(1), s2 = fv(2);
                uint32_t mn = b.fext2(Glsl_NMin, s0, s1), mx = b.fext2(Glsl_NMax, s0, s1);
                uint32_t med = b.fext2(Glsl_NMax, mn, b.fext2(Glsl_NMin, mx, s2));
                uint32_t min3 = b.fext2(Glsl_NMin, mn, s2);
                uint32_t nan_any = b.lor(b.fcmp(Op_FUnordNotEqual, s0, s0),
                                         b.lor(b.fcmp(Op_FUnordNotEqual, s1, s1),
                                               b.fcmp(Op_FUnordNotEqual, s2, s2)));
                vreg[in.dst.value] = fresult(b.sel(nan_any, min3, med));
            } else if (in.opcode == 0x159) {                          // v_med3_u32
                // Unsigned median of three values: max(min(a,b), min(max(a,b),c)).
                // Astro Bot uses this to clamp a material index into [0,31] before its world-map
                // depth prepass. VERIFIED(llvm-mc gfx1030: VOP3 0x159 = v_med3_u32).
                const uint32_t s0 = val(in.src[0]), s1 = val(in.src[1]), s2 = val(in.src[2]);
                const uint32_t mn = b.uext2(Glsl_UMin, s0, s1);
                const uint32_t mx = b.uext2(Glsl_UMax, s0, s1);
                vreg[in.dst.value] = b.uext2(Glsl_UMax, mn, b.uext2(Glsl_UMin, mx, s2));
            } else if (in.opcode == 0x151 || in.opcode == 0x154) {    // v_min3_f32 / v_max3_f32
                // min/max of three floats (DOLL's AA-clamp PS). VERIFIED(round-trip llvm-mc gfx1010:
                // VOP3 0x151 = v_min3_f32, 0x154 = v_max3_f32 — 0xd551…/0xd554…). NaN-aware NMin/
                // NMax per the ISA one-NaN rule. CONFIDENCE: HIGH.
                uint32_t op = in.opcode == 0x151 ? (uint32_t)Glsl_NMin : (uint32_t)Glsl_NMax;
                vreg[in.dst.value] = fresult(b.fext2(op, b.fext2(op, fv(0), fv(1)), fv(2)));
            } else if (in.opcode == 0x368 || in.opcode == 0x369) {    // v_cvt_pknorm_{i16,u16}_f32
                // Clamp and normalize two f32 values, then pack src0 in bits[15:0] and src1 in
                // bits[31:16]. Astro's title ship VS uses the unsigned form (exact first word
                // d7690002). AMD specifies round-to-nearest-even for the normalized conversion.
                const bool is_signed = in.opcode == 0x368;
                const float scale = is_signed ? 32767.0f : 65535.0f;
                const uint32_t lo = b.pack_norm(fv(0), 16, is_signed, scale);
                const uint32_t hi = b.pack_norm(fv(1), 16, is_signed, scale);
                vreg[in.dst.value] = b.ibin(
                    Op_BitwiseOr, lo, b.ibin(Op_ShiftLeftLogical, hi, b.uconst(16)));
            } else if (in.opcode == 0x36A) {                          // v_cvt_pk_u16_u32
                // Pack two u32 into u16 halves with UNSIGNED SATURATION: lo = min(s0,0xFFFF),
                // hi = min(s1,0xFFFF); dst = lo | hi<<16. VERIFIED(round-trip llvm-mc gfx1010:
                // VOP3 0x36a — 0xd76a…). CONFIDENCE: HIGH.
                uint32_t lo = b.uext2(Glsl_UMin, val(in.src[0]), b.uconst(0xFFFFu));
                uint32_t hi = b.uext2(Glsl_UMin, val(in.src[1]), b.uconst(0xFFFFu));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, lo, b.ibin(Op_ShiftLeftLogical, hi, b.uconst(16)));
            } else if (in.opcode >= 0x144 && in.opcode <= 0x147) {
                // Cubemap coordinate ops (#273 — DOLL's title post PSes' reflection-probe math):
                // v_cubeid_f32 (0x144) face id, v_cubesc_f32 (0x145) S numerator, v_cubetc_f32
                // (0x146) T numerator, v_cubema_f32 (0x147) 2*major-axis. Src order (x, y, z);
                // face-major selection per the AMD ISA / GL cubemap convention:
                //   |z|>=|x| && |z|>=|y| : id = z<0?5:4  sc = z<0?-x:x  tc = -y      ma = 2z
                //   else |y|>=|x|        : id = y<0?3:2  sc = x         tc = z<0.. = y<0?-z...
                //   (see per-op emission below)
                // VERIFIED(round-trip llvm-mc gfx1030: 0xd544-0xd547). Execution-tested (kernel in
                // test_rdna2_to_spirv) against the GL major-axis table. CONFIDENCE: MED.
                uint32_t x = fv(0), y = fv(1), z = fv(2);
                uint32_t ax = b.fext1(Glsl_FAbs, x), ay = b.fext1(Glsl_FAbs, y), az = b.fext1(Glsl_FAbs, z);
                uint32_t zmaj = b.land(b.fcmp(Op_FOrdGreaterThanEqual, az, ax),
                                       b.fcmp(Op_FOrdGreaterThanEqual, az, ay));
                uint32_t ymaj = b.fcmp(Op_FOrdGreaterThanEqual, ay, ax);      // (only used when !zmaj)
                uint32_t zneg = b.fcmp(Op_FOrdLessThan, z, b.uconst(0));
                uint32_t yneg = b.fcmp(Op_FOrdLessThan, y, b.uconst(0));
                uint32_t xneg = b.fcmp(Op_FOrdLessThan, x, b.uconst(0));
                auto fneg = [&](uint32_t v) { return b.fbin(Op_FSub, b.uconst(0), v); };
                uint32_t r2;
                switch (in.opcode) {
                    case 0x144: {   // face id: z-major 4/5, y-major 2/3, x-major 0/1 (negative = +1)
                        uint32_t idz = b.sel(zneg, b.uconst(fbits(5.0f)), b.uconst(fbits(4.0f)));
                        uint32_t idy = b.sel(yneg, b.uconst(fbits(3.0f)), b.uconst(fbits(2.0f)));
                        uint32_t idx2 = b.sel(xneg, b.uconst(fbits(1.0f)), b.uconst(fbits(0.0f)));
                        r2 = b.sel(zmaj, idz, b.sel(ymaj, idy, idx2)); break;
                    }
                    case 0x145: {   // sc: z-major: z<0 ? -x : x; y-major: x; x-major: x<0 ? z : -z
                        uint32_t scz = b.sel(zneg, fneg(x), x);
                        uint32_t scx = b.sel(xneg, z, fneg(z));
                        r2 = b.sel(zmaj, scz, b.sel(ymaj, x, scx)); break;
                    }
                    case 0x146: {   // tc: z-major: -y; y-major: y<0 ? -z : z; x-major: -y
                        uint32_t tcy = b.sel(yneg, fneg(z), z);
                        r2 = b.sel(zmaj, fneg(y), b.sel(ymaj, tcy, fneg(y))); break;
                    }
                    default: {      // 0x147 ma: 2 * major axis (signed)
                        uint32_t maj = b.sel(zmaj, z, b.sel(ymaj, y, x));
                        r2 = b.fbin(Op_FMul, b.uconst(fbits(2.0f)), maj); break;
                    }
                }
                vreg[in.dst.value] = fresult(r2);
            } else if (in.opcode == 0x143) {                          // v_mad_u32_u24 = (s0&0xFFFFFF)*(s1&0xFFFFFF)+s2
                uint32_t m24 = b.uconst(0xFFFFFF);
                uint32_t p = b.ibin(Op_IMul, b.ibin(Op_BitwiseAnd, val(in.src[0]), m24),
                                              b.ibin(Op_BitwiseAnd, val(in.src[1]), m24));
                vreg[in.dst.value] = b.ibin(Op_IAdd, p, val(in.src[2]));
            } else if (in.opcode == 0x148 || in.opcode == 0x149) {    // v_bfe_u32 / v_bfe_i32
                uint32_t off = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
                uint32_t cnt = b.ibin(Op_BitwiseAnd, val(in.src[2]), b.uconst(31));
                vreg[in.dst.value] = (in.opcode == 0x148) ? b.bfe_u(val(in.src[0]), off, cnt)
                                                          : b.bfe_s(val(in.src[0]), off, cnt);
            } else if (in.opcode == 0x14A) {                          // v_bfi_b32 = (s0&s1)|(~s0&s2)
                uint32_t s0 = val(in.src[0]);
                uint32_t t1 = b.ibin(Op_BitwiseAnd, s0, val(in.src[1]));
                uint32_t t2 = b.ibin(Op_BitwiseAnd, b.iun(Op_Not, s0), val(in.src[2]));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, t1, t2);
            } else if (in.opcode == 0x364) {                          // v_bcnt_u32_b32
                // AMD RDNA2: D = popcount(S0) + S1. The third VOP3 source field is unused.
                vreg[in.dst.value] = b.ibin(Op_IAdd,
                                             b.iun(Op_BitCount, val(in.src[0])),
                                             val(in.src[1]));
            } else if (in.opcode == 0x36D) {                          // v_add3_u32 = s0+s1+s2
                vreg[in.dst.value] = b.ibin(Op_IAdd, b.ibin(Op_IAdd, val(in.src[0]), val(in.src[1])), val(in.src[2]));
            } else if (in.opcode == 0x346) {                          // v_lshl_add_u32 = (s0<<(s1&31))+s2
                uint32_t sh = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
                vreg[in.dst.value] = b.ibin(Op_IAdd, b.ibin(Op_ShiftLeftLogical, val(in.src[0]), sh), val(in.src[2]));
            } else if (in.opcode == 0x371) {                          // v_and_or_b32 = (s0&s1)|s2
                uint32_t t = b.ibin(Op_BitwiseAnd, val(in.src[0]), val(in.src[1]));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, t, val(in.src[2]));
            } else if (in.opcode == 0x372) {                          // v_or3_b32 = s0|s1|s2
                uint32_t t = b.ibin(Op_BitwiseOr, val(in.src[0]), val(in.src[1]));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, t, val(in.src[2]));
            } else if (in.opcode == 0x178) {                          // v_xor3_b32 = s0^s1^s2
                uint32_t t = b.ibin(Op_BitwiseXor, val(in.src[0]), val(in.src[1]));
                vreg[in.dst.value] = b.ibin(Op_BitwiseXor, t, val(in.src[2]));
            } else if (in.opcode == 0x36F) {                          // v_lshl_or_b32 = (s0<<(s1&31))|s2
                uint32_t sh = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, val(in.src[0]), sh), val(in.src[2]));
            } else if (in.opcode == 0x345) {                          // v_xad_u32 = (s0^s1)+s2
                uint32_t t = b.ibin(Op_BitwiseXor, val(in.src[0]), val(in.src[1]));
                vreg[in.dst.value] = b.ibin(Op_IAdd, t, val(in.src[2]));
            } else if (in.opcode == 0x347) {                          // v_add_lshl_u32 = (s0+s1)<<(s2&31)
                uint32_t sh = b.ibin(Op_BitwiseAnd, val(in.src[2]), b.uconst(31));
                vreg[in.dst.value] = b.ibin(Op_ShiftLeftLogical, b.ibin(Op_IAdd, val(in.src[0]), val(in.src[1])), sh);
            } else if (in.opcode == 0x128 || in.opcode == 0x129 || in.opcode == 0x12A) {
                // Add/sub with carry-in + carry-out (VOP3B): v_add_co_ci_u32 (0x128), v_sub_co_ci (0x129),
                // v_subrev_co_ci (0x12A). carry-in = src2 mask (VCC or an SGPR bool); carry-out -> sdst mask.
                // dst = s0 (+/-) s1 (+/-) cin; carryout = unsigned overflow (add) / borrow (sub).
                // Carry-in from an UNTRACKED mask must reject (like cndmask above), not silently
                // default to 0 — a wrong carry produces 64-bit address math off by one in the low
                // word with no diagnostic.
                const Operand& s2 = in.src[2]; uint32_t cin_mask = 0;
                if (s2.value == 106 || s2.value == 107) {
                    auto it = rs.sreg_bool.find(s2.value);
                    if (rs.sreg_bool_b32.contains(s2.value) && it != rs.sreg_bool.end())
                        cin_mask = it->second;
                    else
                        cin_mask = rs.vcc;
                }
                else if (s2.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(s2.value); if (it != rs.sreg_bool.end()) cin_mask = it->second; }
                if (!cin_mask) ok = false;
                else {
                    uint32_t a = val(in.src[0]), c = val(in.src[1]);
                    uint32_t cin = b.sel(cin_mask, b.uconst(1), b.uconst(0));
                    uint32_t res, cout;
                    if (in.opcode == 0x128) {                             // add: (a + b) + cin
                        uint32_t s1 = b.ibin(Op_IAdd, a, c);
                        uint32_t c1 = b.ucmp(Op_ULessThan, s1, a);        // wrap in a+b
                        res = b.ibin(Op_IAdd, s1, cin);
                        uint32_t c2 = b.ucmp(Op_ULessThan, res, s1);      // wrap in +cin
                        cout = b.bsel(c1, b.btrue(), c2);                 // c1 || c2
                    } else {                                              // sub / subrev: (a - b) - cin (borrow)
                        uint32_t x = in.opcode == 0x129 ? a : c, y = in.opcode == 0x129 ? c : a;  // subrev swaps
                        uint32_t s1 = b.ibin(Op_ISub, x, y);
                        uint32_t b1 = b.ucmp(Op_ULessThan, x, y);         // borrow in x-y
                        res = b.ibin(Op_ISub, s1, cin);
                        uint32_t b2 = b.ucmp(Op_ULessThan, s1, cin);      // borrow in -cin
                        cout = b.bsel(b1, b.btrue(), b2);
                    }
                    vreg[in.dst.value] = res;
                    // carry-out -> sdst mask (VCC or a saved SGPR-pair bool). ISA 3.9: an
                    // EXEC-inactive lane's mask bit is written 0, never its raw carry.
                    const uint32_t cout_masked = rs.exec_narrowed ? b.land(rs.exec, cout) : cout;
                    if (in.sdst.kind == OperandKind::SGPR && b.allow_b32_masks &&
                        (b.is_fragment || (b.is_compute && b.wave_size == 32))) {
                        // In Wave32 every VOP3B carry destination is one physical word. In
                        // particular, a VCC_LO carry write must not clobber an independently-live
                        // VCC_HI compare mask (Astro world-map PC1879 -> PC1881).
                        rs.sreg_bool[in.sdst.value] = cout_masked;
                        rs.sreg_bool_narrowed[in.sdst.value] = true;
                        rs.sreg_bool_b32.insert(in.sdst.value);
                        rs.sreg.erase(in.sdst.value);
                        rs.sreg_srt.erase(in.sdst.value);
                        if (in.sdst.value == 106) rs.vcc = cout_masked;
                    } else if (in.sdst.value == 106 || in.sdst.value == 107) {
                        rs.vcc = cout_masked;
                    } else if (in.sdst.kind == OperandKind::SGPR) {
                        rs.sreg_bool[in.sdst.value] = cout_masked;
                    }
                }
            } else if ((in.opcode == 0x365 || in.opcode == 0x366) && b.ngg_logical_lane &&
                       in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                // A proven no-GS passthrough executes only the logical vertex producer; Vulkan's
                // flattened vertex/instance invocation is the corresponding guest ES lane.  For
                // the canonical all-ones MBCNT pair, LOW contributes min(lane, 32) and HIGH
                // contributes max(lane - 32, 0). General masks remain fail-closed because a vertex
                // invocation has no peer-lane mask state from which to reconstruct them.
                const uint32_t lane = b.guest_lane_id();
                const uint32_t high = b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
                const uint32_t count = in.opcode == 0x365
                    ? b.sel(high, b.uconst(32), lane)
                    : b.sel(high, b.ibin(Op_ISub, lane, b.uconst(32)), b.uconst(0));
                vreg[in.dst.value] = b.ibin(Op_IAdd, val(in.src[1]), count);
            } else if ((in.opcode == 0x365 || in.opcode == 0x366) && b.ngg_one_lane) {
                // NGG is deliberately lowered as one guest lane per Vulkan vertex invocation. No
                // lane precedes that invocation, so either half of MBCNT contributes zero and leaves
                // its accumulator unchanged. This is the scalar counterpart of recompile_vertex's
                // existing s3=1 (one ES vertex, no GS primitive) ABI model.
                vreg[in.dst.value] = val(in.src[1]);
            } else if ((in.opcode == 0x365 || in.opcode == 0x366) && allow_wave &&
                       (b.is_compute || b.is_fragment)) {
                // v_mbcnt_lo/hi_u32_b32 (cross-lane): dst = src1 + count of lanes below this one whose mask
                // bit (src0) is set, in the low/high 32. The per-lane "mask bit" comes from src0: EXEC
                // (126/127) -> this lane's exec bool; inline -1 (all-ones) -> always set (mbcnt = lane
                // index, the common "get my lane id" idiom, e.g. shader 037); inline 0 -> never; an SGPR
                // pair -> that saved mask's bool. A general computed 32-bit mask VALUE isn't representable
                // per-lane, so reject that. Portable compute uses LDS+barriers at barrier-uniform sites;
                // exact-size compute subgroups and fragments use a native exclusive subgroup sum.
                const uint32_t active = mbcnt_source_bit(
                    b, rs, in.src[0], in.opcode == 0x366);
                if (active) {
                    const uint32_t acc = val(in.src[1]);
                    const uint32_t lo = in.opcode == 0x365 ? b.btrue() : b.bfalse();
                    vreg[in.dst.value] = b.is_fragment
                        ? b.fragment_mbcnt(active, acc, in.opcode == 0x365)
                        : (b.native_subgroup_size
                            ? b.native_compute_mbcnt(active, acc, lo)
                            : b.mbcnt(active, acc, in.opcode == 0x365));
                }
                else ok = false;
            } else if (in.opcode == 0x12F) {                          // v_cvt_pkrtz_f16_f32 = pack(s0->lo, s1->hi)
                vreg[in.dst.value] = b.pack_half2x16_rtz(fv(0), fv(1)); // v_cvt_pkrtz VOP3: RTZ clamp (#452)
            } else if (in.opcode == 0x103) {                          // v_add_f32 (VOP3 form)
                vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, fv(0), fv(1)));
            } else if (in.opcode == 0x104) {                          // v_sub_f32 (VOP3 form) = s0 - s1
                vreg[in.dst.value] = fresult(b.fbin(Op_FSub, fv(0), fv(1)));
            } else if (in.opcode == 0x105) {                          // v_subrev_f32 (VOP3 form) = s1 - s0
                vreg[in.dst.value] = fresult(b.fbin(Op_FSub, fv(1), fv(0)));
            } else if (in.opcode == 0x108) {                          // v_mul_f32 (VOP3 form)
                vreg[in.dst.value] = fresult(b.fbin(Op_FMul, fv(0), fv(1)));
            } else if (in.opcode == 0x10F) {                          // v_min_f32 (VOP3 form; NaN -> other operand)
                vreg[in.dst.value] = fresult(b.fext2(Glsl_NMin, fv(0), fv(1)));
            } else if (in.opcode == 0x110) {                          // v_max_f32 (VOP3 form; NaN -> other operand)
                vreg[in.dst.value] = fresult(b.fext2(Glsl_NMax, fv(0), fv(1)));
            } else if (in.opcode == 0x107) {                          // v_mul_legacy_f32: DX9 multiply —
                // 0 * x == 0 for ALL x including ±Inf/NaN (that guarantee is WHY compilers emit it,
                // e.g. attenuation=0 times 1/dist). Plain IEEE FMul gives NaN for 0*Inf, so emit
                // select(s0==0 || s1==0, 0, s0*s1). ±0.0 both compare equal to 0.0 under FOrdEqual.
                uint32_t s0b = fv(0), s1b = fv(1), zb = b.uconst(0);
                uint32_t anyz = b.lor(b.fcmp(Op_FOrdEqual, s0b, zb), b.fcmp(Op_FOrdEqual, s1b, zb));
                vreg[in.dst.value] = fresult(b.sel(anyz, zb, b.fbin(Op_FMul, s0b, s1b)));
            } else if (in.opcode == 0x101) {                          // v_cndmask_b32_e64: src2_mask ? src1 : src0
                const Operand& s2 = in.src[2]; uint32_t m = 0;        // src2 is an SGPR-pair (or VCC) wave mask
                if (s2.value == 106 || s2.value == 107) m = rs.vcc;
                else if (s2.kind == OperandKind::SGPR) {
                    auto it = rs.sreg_bool.find(s2.value);
                    if (it != rs.sreg_bool.end()) {
                        m = it->second;
                    } else if (b.ngg_one_lane) {
                        // The exact Astro NGG projection represents guest lane zero. Some wrappers
                        // construct a literal/dynamic B64 lane mask in ordinary scalar DATA registers
                        // rather than through a mask-domain instruction (7f5f: s4:s5=0xaaaaaaaa...
                        // before pc3870). For lane zero, consuming that pair as a wave mask is exactly
                        // its low dword's bit zero. Keep arbitrary vertex shaders fail-closed.
                        auto data = rs.sreg.find(s2.value);
                        if (data != rs.sreg.end())
                            m = b.ucmp(Op_INotEqual,
                                b.ibin(Op_BitwiseAnd, data->second, b.uconst(1)),
                                b.uconst(0));
                    }
                }
                // fv (not val): cndmask is float-modifier-capable and compilers emit it with -v/|v|
                // sources (sign-select idioms). Raw val() silently dropped neg/abs — the shader
                // recompiled "successfully" and computed the un-negated value. fv == val when no
                // modifier bits are set.
                if (!m) ok = false; else vreg[in.dst.value] = b.sel(m, fv(1), fv(0));
            } else if (in.opcode == 0x11F) {                          // v_mac_f32_e64 (VOP3 form of VOP2 0x1f)
                // dst = src0*src1 + dst, with the VOP3 float source modifiers (neg/abs via fv) and output
                // modifiers (omod/clamp via fresult). The scene VS emits this e64 form with a `-|v10|`
                // modifier (round-trip: `llvm-mc -mcpu=gfx1010` of 0xd51f020a → v_mac_f32_e64 v10,v28,-|v10|).
                vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, b.fbin(Op_FMul, fv(0), fv(1)), old_d));
            } else ok = false;
            // A set CLAMP bit on an opcode that does not route through fresult means unmodeled
            // INTEGER saturation (ISA 6.5: "for integer operations, it clamps the result to the
            // largest and smallest representable value") or an unhandled pack/select combination
            // (v_cvt_pkrtz, v_cndmask_e64, the VOP3B carry ops) — reject fail-visibly rather than
            // silently emitting the wrapping/unclamped result. OMOD on integer results is
            // architecturally ignored, so only CLAMP gates here.
            if (ok && in.clamp && !clamp_routed) ok = false;
            if (ok) predicate_write(b, rs, in.dst.value, old_d);
            return true;
        }
        case Rdna2Format::SOPP: {
            // Control flow. The only branch we can safely linearize today is the common forward
            // s_cbranch_execz "skip the EXEC-predicated block if no lane is active" idiom. Branches
            // on SCC/VCC (or EXECNZ) can skip code for reasons not represented by EXEC predication,
            // so accepting them as no-ops would silently execute the wrong path.
            switch (in.opcode) {
                // Hints / sync with no effect in our synchronous SSA model — safe no-ops.
                case 0x00:   // s_nop
                case 0x0c:   // s_waitcnt        (no async memory latency to wait on)
                case 0x10:   // s_sendmsg        (NGG GS_ALLOC_REQ etc. — no wave/primitive allocation in
                             //                   our per-invocation model; only meaningful for NGG/GS,
                             //                   which we lower per-invocation, so it's a safe no-op)
                case 0x17:   // s_cbranch_cdbgsys (Prosper exposes no attached GPU system debugger, so
                             //                    COND_DBG_SYS is permanently clear and the branch falls through)
                case 0x20:   // s_inst_prefetch  (I-cache hint)
                case 0x21:   // s_clause         (memory-clause scheduling hint)
                case 0x22:   // s_wait_idle
                    break;   // (s_waitcnt_vscnt is SOPK on gfx10, not SOPP 0x7d — see the SOPK case)
                case 0x08:                                          // s_cbranch_execz
                    if (in.simm16 < 0) ok = false;                 // backward = loop -> unsupported
                    else if (rs.exec_narrowed && (!safe_execz || !safe_execz->count(in.pc))) ok = false;
                    break;                                          // forward = no-op (predication covers it)
                case 0x0a:                                          // s_barrier
                    if (b.is_compute || b.ngg_private_lds) b.barrier();
                    else ok = false;                                // barrier only meaningful in compute
                    break;
                case 0x04: case 0x05:                              // s_cbranch_scc0 / scc1
                    // An alpha-test / clip() kill-mask early-out (SCC = "any lane survives", set by a 64-bit
                    // wave-mask op — see mask_test_branches) is a pure wave optimization: per-invocation the
                    // survivor mask narrows EXEC in the block below and the export OpKills the failed lanes,
                    // so the branch LINEARIZES away exactly like a forward s_cbranch_execz. Only these
                    // recognized branches are dropped; any other SCC branch is still a real uniform-if we
                    // can't model straight-line, so it rejects.
                    if (safe_execz && safe_execz->count(in.pc)) break;   // recognized kill-mask branch -> no-op
                    ok = false;
                    break;
                case 0x06: case 0x07:                              // s_cbranch_vccz / vccnz
                    // A byte-exact Astro NGG terminal suffix may skip only trailing PARAM exports
                    // after POS has already been exported. In the one-lane projection emitting those
                    // otherwise-unused values is harmless; the gate proof records only that bounded
                    // branch in safe_execz. General VCC branches remain unsupported below.
                    if (safe_execz && safe_execz->count(in.pc)) break;
                    ok = false;
                    break;
                case 0x09:                                         // s_cbranch_execnz
                    ok = false;
                    break;
                default: ok = false;   // s_branch / s_sendmsg / s_setreg / etc. -> reject
            }
            return true;
        }
        case Rdna2Format::SMEM: {
            // Scalar memory load. Modeled as a load from a single bound constant buffer indexed by the
            // immediate byte offset (>>2 -> dword index); SBASE/descriptor base is folded into the
            // binding. N consecutive dwords -> SDATA..SDATA+N-1. Only the compute path binds the cbuf,
            // so reject in graphics stages (allow_smem=false). Register-offset SMEM not yet handled.
            uint32_t n = 0;
            switch (in.opcode) {
                case 0x0: case 0x8: n = 1;  break;   // s_load_dword     / s_buffer_load_dword
                case 0x1: case 0x9: n = 2;  break;   // s_load_dwordx2   / s_buffer_load_dwordx2
                case 0x2: case 0xA: n = 4;  break;   // s_load_dwordx4   / s_buffer_load_dwordx4
                case 0x3: case 0xB: n = 8;  break;   // s_load_dwordx8   / s_buffer_load_dwordx8
                case 0x4: case 0xC: n = 16; break;   // s_load_dwordx16  / s_buffer_load_dwordx16
                default: ok = false; return true;    // stores / others not yet
            }
            // PC-relative scalar embedded table (#1054): this s_buffer_load consumes a descriptor
            // built from s_getpc_b64 and a bounded table carried by the shader blob. Resolve it before
            // the external-resource gate, exactly like the established MUBUF form. The byte offset is
            // still the real tracked scalar value; an untracked SOFFSET remains fail-visible.
            auto pcrel = rs.smem_pcrel_tables.find(in.pc);
            if (pcrel != rs.smem_pcrel_tables.end()) {
                uint32_t soff = 0;
                const bool soff_null = in.src[1].kind == OperandKind::Special &&
                                       in.src[1].value == 125;
                if (soff_null) {
                    soff = b.uconst(0);
                } else if (in.src[1].kind == OperandKind::SGPR ||
                           (in.src[1].kind == OperandKind::Special &&
                            in.src[1].value >= 106 && in.src[1].value <= 123)) {
                    auto tracked = rs.sreg.find(in.src[1].value);
                    if (tracked == rs.sreg.end()) { ok = false; return true; }
                    soff = tracked->second;
                } else if (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0) {
                    soff = b.uconst(static_cast<uint32_t>(in.src[1].value));
                } else {
                    ok = false; return true;
                }
                uint32_t idx = b.ibin(Op_ShiftRightLogical,
                    b.ibin(Op_IAdd, soff, b.uconst(in.literal)), b.uconst(2));
                for (uint32_t k = 0; k < n; ++k) {
                    uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                    uint32_t value = b.uconst(0); // bounded V# OOB contract
                    for (uint32_t t = 0; t < pcrel->second.size(); ++t)
                        value = b.sel(b.ucmp(Op_IEqual, kidx, b.uconst(t)),
                                      b.uconst(pcrel->second[t]), value);
                    rs.sreg[in.dst.value + static_cast<int>(k)] = value;
                    rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
                }
                return true;
            }
            if (!allow_smem) {
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr, "[smem-reject] pc=%u reason=graphics-disabled op=0x%x\n",
                            in.pc, in.opcode);
                ok = false; return true;
            }
            // SOFFSET handling. Immediate-only loads encode SOFFSET = SGPR_NULL (125). A register
            // SOFFSET adds an SGPR-computed byte offset:
            //  * a DESCRIPTOR s_load (x4/x8 = V#/T#) with a computed offset is the bindless fetch's
            //    V#-table read (`s_load_dwordx4 s[8:11],s[24:25],vcc_hi`) — the const-fold
            //    (resolve_dynamic_fetch -> by_fetch_pc) resolves the fetch through that V#, so the
            //    SGPR result is unused per-invocation — no-op it (placeholder 0) so the VS still
            //    recompiles (the "mask=0xF draws never render" gap on PPSA01885). CONFIDENCE: HIGH.
            //  * an s_buffer_load (0x8..0xC) with a TRACKED scalar offset is a computed constant-
            //    buffer read (DOLL's bloom-combine PS: per-tap weights at `vcc_lo = 16*(tap/2)`
            //    inside its counted loop, #273) — model it as a DYNAMIC dword index into the
            //    resolved cbuf binding: idx = (soffset + imm) >> 2. An UNTRACKED offset register
            //    (a runtime user-SGPR we have no value for) still rejects — never fold as 0.
            const bool soff_null = (in.src[1].kind == OperandKind::Special && in.src[1].value == 125);
            uint32_t soff_bits = 0; bool soff_dyn = false;
            if (!soff_null) {
                if (rt && (in.opcode == 0x2 || in.opcode == 0x3)) {
                    for (uint32_t k = 0; k < n; k++) rs.sreg[in.dst.value + (int)k] = b.uconst(0);
                    return true;
                }
                bool tracked = false;
                if (in.opcode >= 0x8 && in.opcode <= 0xC) {
                    if (in.src[1].kind == OperandKind::SGPR ||
                        (in.src[1].kind == OperandKind::Special && in.src[1].value >= 106 && in.src[1].value <= 123)) {
                        auto it = rs.sreg.find(in.src[1].value);
                        if (it != rs.sreg.end()) { soff_bits = it->second; tracked = true; }
                    } else if (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0) {
                        soff_bits = b.uconst((uint32_t)in.src[1].value); tracked = true;
                    }
                }
                if (!tracked) {
                    if (getenv("PROSPER_DBG"))
                        fprintf(stderr,
                                "[smem-reject] pc=%u reason=untracked-soffset op=0x%x "
                                "src1-kind=%d src1=%d\n",
                                in.pc, in.opcode, static_cast<int>(in.src[1].kind),
                                in.src[1].value);
                    ok = false; return true;
                }
                soff_dyn = true;
            } else if ((int32_t)in.literal < 0) { ok = false; return true; }   // negative imm-only would wrap
            uint32_t base_idx = soff_dyn ? 0 : in.literal >> 2;    // immediate byte offset -> dword index
            // Descriptor provenance: pick which bound constant buffer via the resource table, routing this
            // load to that buffer's OWN binding (N-buffer model) — so Unity's several constant buffers
            // (per-draw transform, per-frame, …) don't collapse onto one. For s_buffer_load, SBASE
            // (src[0]) is the V#: resolve it by an earlier s_load's SRT tag (indirect) or directly by its
            // user-data SGPR index (the V# was placed in SGPRs by the driver). Default binding 2.
            uint32_t binding = 2; bool cbuf_resolved = false;
            if (rt) { const ShaderResource* res = nullptr;
                // Descriptor-table folding emits pc-only entries when a V# has no stable SRT key
                // (or that key collides). Exact per-use provenance must win, as it does for MUBUF/MIMG.
                res = rt->by_fetch_pc(in.pc);
                if (res && res->cls != ResourceClass::ConstantBuffer) res = nullptr;
                uint32_t srt_tag = 0;
                if (!res && sreg_srt_range_tag(rs, in.src[0].value, 4, srt_tag))
                    res = rt->by_srt_offset(srt_tag);
                // A scalar buffer load reads a CONSTANT buffer — resolve the SBASE SGPR to a constant
                // buffer specifically (the same SGPR may also hold a vertex-buffer V# elsewhere).
                if (!res && !sreg_range_written(rs, in.src[0].value, 4))
                    res = rt->by_sgpr_base_cls(in.src[0].value, ResourceClass::ConstantBuffer);
                if (res) { binding = res->binding; cbuf_resolved = true; } }
            if (getenv("PROSPER_CBUFLOG"))
                fprintf(stderr, "[cbuf] pc=%u s_buffer_load x%u src0=s%d off=0x%x(dw%u) dyn=%d -> binding=%u %s\n",
                        in.pc, n, in.src[0].value, in.literal, base_idx, (int)soff_dyn, binding,
                        cbuf_resolved ? "resolved" : "DEFAULT-2");
            // Immediate s_load_dwordx4/x8 is the same descriptor-table fetch as the dynamic form
            // handled above. When the front-half table already decoded that V#/T#/S#, its raw words
            // are provenance only: emitting loads from fallback binding 2 creates a statically-used
            // descriptor the runtime table does not contain (#515). Preserve the SRT tag and use
            // placeholders, exactly like the dynamic descriptor-fetch path. A genuinely resolved
            // constant-buffer resource still executes the load below.
            if (rt && !cbuf_resolved && (in.opcode == 0x2 || in.opcode == 0x3)) {
                for (uint32_t k = 0; k < n; ++k) {
                    rs.sreg[in.dst.value + (int)k] = b.uconst(0);
                    rs.sreg_srt[in.dst.value + (int)k] = in.literal;
                }
                return true;
            }
            // A data load with a runtime table may retain the legacy binding-2 convention only when
            // that table actually binds a buffer there. VS/compute tables start at binding 2 and use
            // this path legitimately; PS tables start at 32, where keeping the fallback would emit an
            // interface the renderer cannot satisfy (#719).
            if (rt && !cbuf_resolved) {
                const bool fallback_bound = std::any_of(
                    rt->resources.begin(), rt->resources.end(), [](const ShaderResource& resource) {
                        return resource.binding == 2 &&
                               (resource.cls == ResourceClass::ConstantBuffer ||
                                resource.cls == ResourceClass::VertexBuffer);
                    });
                if (!fallback_bound) {
                    if (getenv("PROSPER_DBG"))
                        fprintf(stderr,
                                "[smem-reject] pc=%u reason=unresolved-cbuf op=0x%x "
                                "src0=s%d dyn=%d\n",
                                in.pc, in.opcode, in.src[0].value, (int)soff_dyn);
                    ok = false; return true;
                }
            }
            if (soff_dyn) {
                // Dynamic dword index: (soffset + signed imm) >> 2 (uint add == two's-complement add).
                uint32_t idx0 = b.ibin(Op_ShiftRightLogical,
                                       b.ibin(Op_IAdd, soff_bits, b.uconst(in.literal)), b.uconst(2));
                for (uint32_t k = 0; k < n; k++) {
                    uint32_t kidx = k ? b.ibin(Op_IAdd, idx0, b.uconst(k)) : idx0;
                    rs.sreg[in.dst.value + (int)k] = b.cbuf_load(kidx, binding);
                    rs.sreg_srt.erase(in.dst.value + (int)k);   // data load: drop any stale descriptor tag
                }
                return true;
            }
            for (uint32_t k = 0; k < n; k++)
                rs.sreg[in.dst.value + (int)k] = b.cbuf_load(b.uconst(base_idx + k), binding);
            // A wide scalar load is a descriptor fetch — tag its dest SGPRs with the SRT offset so a
            // later buffer/image op using them resolves to the right resource (provenance). x4 = V#/S#
            // (buffers, samplers), x8 = T# (textures, 8 dwords).
            if (rt && (in.opcode == 0x2 || in.opcode == 0x3)) {
                for (uint32_t k = 0; k < n; k++) rs.sreg_srt[in.dst.value + (int)k] = in.literal;
            } else {
                // Scalar data loads overwrite the destination; they do not carry descriptor identity.
                for (uint32_t k = 0; k < n; k++) rs.sreg_srt.erase(in.dst.value + (int)k);
            }
            return true;
        }
        case Rdna2Format::FLAT: {
            const FlatAccessInfo access = flat_access_info(in.opcode);
            // General (non-scratch) FLAT LOAD from a raw 64-bit guest address (#1171). If the executor
            // resolved this load's base pointer to user SGPRs s[flat_base_sgpr : +1] and bound the
            // containing guest allocation as an SSBO (a ConstantBuffer-class resource keyed by this
            // load's pc), lower it to an indexed read of that window at byte offset (address_lo -
            // base_lo). base_lo is the ORIGINAL push-constant value (the address the executor bound the
            // window at), so the module stays dispatch-independent and correct even if the shader later
            // reuses the base SGPR. Unresolved forms fall through to the scratch/reject path below.
            const ShaderResource* fw =
                (rt && access.valid && !access.store && !in.flat_lds &&
                 in.src[0].kind == OperandKind::VGPR &&
                 in.src[1].kind == OperandKind::Special && in.src[1].value == 125)
                    ? rt->by_fetch_pc(in.pc) : nullptr;
            if (fw && fw->flat_base_sgpr != 0xFFFFFFFFu) {
                const uint32_t addr_lo = val(in.src[0]);                     // low dword of the address
                const uint32_t base_lo = b.load_push_constant(fw->flat_base_sgpr);
                // Byte offset in the window: (address - base) mod 2^32, which equals the true offset for
                // any 0 <= offset < window <= 256 MiB (the low-dword subtraction wraps complementarily to
                // the address's own IAdd). A negative offset (address < base) becomes a huge unsigned index
                // -> out-of-window -> robustBufferAccess returns 0 (defined, but a loose-bounds divergence
                // from HW; decode kernels use non-negative offsets).
                const uint32_t byte0 = b.ibin(Op_ISub, addr_lo, base_lo);
                for (uint32_t c = 0; c < access.components; ++c) {
                    const uint32_t addr = access.bits == 32
                        ? b.ibin(Op_IAdd, byte0, b.uconst(c * 4)) : byte0;
                    const uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
                    uint32_t value;
                    if (access.bits == 32) {
                        value = b.cbuf_load(idx, fw->binding);
                    } else {
                        // Sub-dword (ubyte/ushort) extract, mirroring the MUBUF raw path: a 16-bit
                        // access may begin at byte 3 and straddle two dwords, so join the adjacent words.
                        const uint32_t dw0 = b.cbuf_load(idx, fw->binding);
                        const uint32_t byte_in_dw = b.ibin(Op_BitwiseAnd, addr, b.uconst(3));
                        const uint32_t shift = b.ibin(Op_ShiftLeftLogical, byte_in_dw, b.uconst(3));
                        uint32_t joined = b.ibin(Op_ShiftRightLogical, dw0, shift);
                        if (access.bits == 16) {
                            const uint32_t dw1 =
                                b.cbuf_load(b.ibin(Op_IAdd, idx, b.uconst(1)), fw->binding);
                            const uint32_t inv_shift = b.ibin(Op_BitwiseAnd,
                                b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
                            const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
                            joined = b.ibin(Op_BitwiseOr, joined,
                                b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
                        }
                        value = access.sign_extend
                            ? b.bfe_s(joined, b.uconst(0), b.uconst(access.bits))
                            : b.bfe_u(joined, b.uconst(0), b.uconst(access.bits));
                    }
                    const int reg = in.dst.value + static_cast<int>(c);
                    const uint32_t old_value = vreg_old(b, rs, reg);
                    rs.vreg[reg] = value;
                    predicate_write(b, rs, reg, old_value);
                }
                return true;
            }
            // Model the compiler's private spill area only. Each generated shader invocation owns
            // one Function-storage array, so the entry-provided hardware base has no host-visible
            // address to preserve. Other FLAT/GLOBAL forms remain deliberately unsupported.
            if (!access.valid || in.flat_segment != 1u || in.flat_lds || !b.guest_scratch ||
                in.src[0].kind != OperandKind::None || in.src[1].kind != OperandKind::SGPR ||
                in.src[1].value != b.guest_scratch_saddr ||
                rs.sreg_written.count(in.src[1].value)) {
                ok = false;
                return true;
            }
            const int32_t base = static_cast<int32_t>(in.literal);
            const int64_t storage_begin = b.guest_scratch_min_byte;
            const int64_t storage_end = storage_begin + static_cast<int64_t>(b.guest_scratch_dwords) * 4;
            for (uint32_t component = 0; component < access.components; ++component) {
                const int32_t byte_offset = base +
                    static_cast<int32_t>(component * (access.bits / 8u));
                const int64_t access_end = static_cast<int64_t>(byte_offset) + access.bits / 8u;
                if (byte_offset < storage_begin || access_end > storage_end) {
                    ok = false;
                    return true;
                }
                const int reg = in.dst.value + static_cast<int>(component);
                if (access.store) {
                    b.guest_scratch_store_bits(byte_offset, access.bits, vreg_old(b, rs, reg),
                                               rs.exec_narrowed, rs.exec);
                } else {
                    const uint32_t old_value = vreg_old(b, rs, reg);
                    rs.vreg[reg] = b.guest_scratch_load_bits(byte_offset, access.bits,
                                                            access.sign_extend);
                    predicate_write(b, rs, reg, old_value);
                }
            }
            return true;
        }
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF: {
            // Untyped buffer LOAD — the per-lane fetch mechanism (vertex fetch et al.). Modeled as a
            // per-lane load from the bound constant buffer: byte addr = (offen ? VADDR : 0) + SOFFSET
            // + inst-offset; index = addr>>2; N dwords -> VDATA..+N-1. Descriptor (SRSRC), idxen*stride,
            // and the format-converting buffer_load_format_* variants are deferred. Compute-only (cbuf).
            // LDS bit set = transfer between LDS and memory instead of VGPRs (ISA Table 98). The
            // VDATA VGPRs stay untouched on hardware and the data lands in LDS — translating it as
            // a VGPR load would clobber a live register AND drop the LDS write. Reject until a
            // buffer->LDS model exists.
            if (in.fmt == Rdna2Format::MUBUF && in.mubuf_lds) { ok = false; return true; }
            // MTBUF opcodes 8..15 pack D16 results/inputs two per VGPR. Reusing the ordinary or raw
            // MUBUF cases below would silently apply the wrong register layout, so fail closed until
            // that packing is modeled.
            if (in.fmt == Rdna2Format::MTBUF && in.opcode >= 8u) { ok = false; return true; }
            // TFE appends a fault/status result after the data VGPRs. Dropping that write can corrupt
            // later register consumers, so reject until status-return semantics are implemented.
            if (in.fmt == Rdna2Format::MTBUF && in.mtbuf_tfe) { ok = false; return true; }
            uint32_t n = 0; bool is_format = false, is_store = false, is_atomic = false;
            uint32_t atomic_op = 0;   // SPIR-V atomic RMW opcode for is_atomic (set by the switch)
            bool raw_subword = false, raw_signed = false;
            uint32_t raw_bits = 32;
            switch (in.opcode) {
                case 0x8: n = 1; raw_subword = true; raw_bits = 8; break;   // buffer_load_ubyte
                case 0x9: n = 1; raw_subword = true; raw_signed = true; raw_bits = 8; break; // sbyte
                case 0xA: n = 1; raw_subword = true; raw_bits = 16; break;  // buffer_load_ushort
                case 0xB: n = 1; raw_subword = true; raw_signed = true; raw_bits = 16; break; // sshort
                case 0xC: n = 1; break;                     // buffer_load_dword
                case 0xD: n = 2; break;                     // buffer_load_dwordx2
                case 0xE: n = 4; break;                     // buffer_load_dwordx4
                case 0xF: n = 3; break;                     // buffer_load_dwordx3 (round-trip llvm-mc gfx1010:
                                                            // 0xe03c… op 0x0F — x3 sorts AFTER x4 in this ISA)
                case 0x0: n = 1; is_format = true; break;   // buffer_load_format_x  (vertex fetch)
                case 0x1: n = 2; is_format = true; break;   // buffer_load_format_xy
                case 0x2: n = 3; is_format = true; break;   // buffer_load_format_xyz
                case 0x3: n = 4; is_format = true; break;   // buffer_load_format_xyzw
                case 0x1C: n = 1; is_store = true; break;   // buffer_store_dword
                case 0x1D: n = 2; is_store = true; break;   // buffer_store_dwordx2
                case 0x1E: n = 4; is_store = true; break;   // buffer_store_dwordx4
                case 0x1F: n = 3; is_store = true; break;   // buffer_store_dwordx3 (like loads,
                                                            // x3 sorts after x4 in this ISA)
                case 0x4: n = 1; is_format = true; is_store = true; break;   // buffer_store_format_x
                case 0x5: n = 2; is_format = true; is_store = true; break;   // buffer_store_format_xy
                case 0x6: n = 3; is_format = true; is_store = true; break;   // buffer_store_format_xyz
                case 0x7: n = 4; is_format = true; is_store = true; break;   // buffer_store_format_xyzw
                // 32-bit atomic RMW family (RDNA2 MUBUF opcodes 0x30..0x3b). Each does
                // mem = mem OP VDATA and returns the PRE-op value in VDATA. They all share the generic
                // OpAtomic<Op>(ptr, Device, AcqRel, value) shape emitted by cbuf_atomic_rtn. VDATA is one
                // dword. CMPSWAP (0x31, two-operand), CSUB (0x34, conditional-subtract), INC/DEC
                // (0x3c/0x3d, wrap semantics), and the x2 64-bit variants stay deferred (fail-visible)
                // via the default case below — they need distinct lowering, not a plain RMW.
                case 0x30: n = 1; is_atomic = true; atomic_op = Op_AtomicExchange; break; // swap
                case 0x32: n = 1; is_atomic = true; atomic_op = Op_AtomicIAdd;     break; // add
                case 0x33: n = 1; is_atomic = true; atomic_op = Op_AtomicISub;     break; // sub
                case 0x35: n = 1; is_atomic = true; atomic_op = Op_AtomicSMin;     break; // smin (signed)
                case 0x36: n = 1; is_atomic = true; atomic_op = Op_AtomicUMin;     break; // umin (unsigned)
                case 0x37: n = 1; is_atomic = true; atomic_op = Op_AtomicSMax;     break; // smax (signed)
                case 0x38: n = 1; is_atomic = true; atomic_op = Op_AtomicUMax;     break; // umax (unsigned)
                case 0x39: n = 1; is_atomic = true; atomic_op = Op_AtomicAnd;      break; // and
                case 0x3a: n = 1; is_atomic = true; atomic_op = Op_AtomicOr;       break; // or
                case 0x3b: n = 1; is_atomic = true; atomic_op = Op_AtomicXor;      break; // xor
                default: ok = false; return true;           // remaining typed/atomic opcodes deferred
            }
            uint32_t offset = in.literal & 0xFFFu;
            bool offen = (in.literal >> 12) & 1u, idxen = (in.literal >> 13) & 1u;
            // PC-relative EMBEDDED TABLE (#273): this load's V# was built from s_getpc_b64 and the
            // table bytes live inside the shader blob — detect_pcrel_tables already copied them out.
            // Fold to a compile-time constant lookup: dword index = (inst offset + offen VADDR) >> 2;
            // out-of-range indexes read 0 (the hardware's OOB contract for a bounded V#).
            if (!is_format && !is_store && !is_atomic && !raw_subword) {
                auto pt = rs.mubuf_pcrel_tables.find(in.pc);
                if (pt != rs.mubuf_pcrel_tables.end()) {
                    const std::vector<uint32_t>& tab = pt->second;
                    uint32_t addr = b.uconst(offset);
                    if (offen) { Operand ov{OperandKind::VGPR, in.src[0].value};
                                 addr = b.ibin(Op_IAdd, addr, val(ov)); }
                    uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
                    for (uint32_t k = 0; k < n; k++) {
                        int d = in.dst.value + (int)k;
                        uint32_t old = vreg_old(b, rs, d);
                        uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                        uint32_t acc = b.uconst(0);   // OOB -> 0
                        for (uint32_t t = 0; t < (uint32_t)tab.size(); t++)
                            acc = b.sel(b.ucmp(Op_IEqual, kidx, b.uconst(t)), b.uconst(tab[t]), acc);
                        rs.vreg[d] = acc;
                        predicate_write(b, rs, d, old);
                    }
                    return true;
                }
            }
            // A proven PC-relative table is self-contained shader data and needs no runtime resource
            // bindings. Only external buffer accesses require the stage's SMEM/MUBUF resource gate.
            // Keep this check after the fold so resource-free graphics shaders (Astro Bot's loading VS)
            // can consume their embedded lookup table without making unresolved V# accesses permissive.
            if (!allow_smem) { ok = false; return true; }
            uint32_t binding = 2, stride = 0;   // overwritten by SRSRC resolution below whenever a resource
                                                // table is present (format AND raw ops); the binding-2 default
                                                // survives only on the table-less offline path (see below)
            // Format of the fetched components. Untyped buffer_load_dword* is raw 32-bit (comp_bytes=4);
            // buffer_load_format_* takes the format from the resolved V# descriptor.
            DataFormat fmt = DataFormat::Uint32;   // untyped default: raw dwords
            uint32_t fmt_ncomp = 0;    // the V#'s real component count (format loads only); 0 = don't default-fill
            bool dyn_vfetch = false;   // set when the V# came from by_fetch_pc — a const-folded per-vertex
                                       // attribute fetch, whose element address is exactly gl_VertexIndex*stride.
            bool instance_vfetch = false;
            bool folded_vfetch = false; // by-fetch V# base already includes OFFSET/SOFFSET
            if (is_format) {
                // A format load reads a vertex/buffer attribute — it needs the V# descriptor for the
                // binding, stride, and data format. Resolve SRSRC (src[1]) via provenance: an s_load
                // tag (indirect) else the SGPR index (direct/user-data).
                const ShaderResource* res = nullptr;
                // A format load (vertex fetch) reads a VERTEX buffer — resolve the SRSRC SGPR to a vertex
                // buffer specifically (that SGPR may hold a constant-buffer V# at other points; the const-
                // fold-resolved vertex buffer is keyed by this SRSRC SGPR). Fall back to an s_load SRT tag.
                if (rt) {
                    // Any write to the four-dword SRSRC range invalidates its entry-time direct V#.
                    // Exact per-fetch and s_load provenance below may still identify the live descriptor,
                    // but a missing/rejected dynamic result must never fall back to stale user data.
                    const bool srsrc_rewritten = sreg_range_written(rs, in.src[1].value, 4);
                    // PER-FETCH first: a reloaded SRSRC holds a different V# per attribute, so match this
                    // exact fetch instruction's pc; fall back to untouched SGPR user data or an s_load
                    // SRT tag. A rewritten direct descriptor without either provenance stays unresolved.
                    // Only a VERTEX-buffer pc entry implies the vertex-index address model — a pc-keyed
                    // CONSTANT/structured buffer (a PS's per-lane table fetch, #273) keeps the faithful
                    // VADDR*stride+offset address below.
                    res = rt->by_fetch_pc(in.pc);
                    // A pc-keyed entry is not itself proof that OFFSET/SOFFSET was folded into the
                    // bound base. Shader mode deliberately binds DynFetch::unshifted_desc and must
                    // retain both VADDR terms (idxen+offen), OFFSET and SOFFSET exactly as encoded.
                    folded_vfetch = res && res->cls == ResourceClass::VertexBuffer &&
                                      res->fetch_index_mode != VertexFetchIndexMode::Shader;
                    // The NGG fetch-prologue shortcut applies only to an untouched ABI element index. It
                    // must NOT apply after the shader has selected or computed VADDR. DOLL lays out packed
                    // attributes as two/three descriptor records per vertex and computes
                    // 2*vertex+channel / 3*vertex+channel in v0/v4/v5/v6. DQ also uses a modeled
                    // v_cndmask merged-wave selector to choose instance_id for a per-instance transform
                    // lookup and vertex_id for positions. Replacing either shader value with
                    // gl_VertexIndex reads allocator metadata as transform indices and emits NaN/giant
                    // triangles.
                    // A pc-keyed VertexBuffer can also be a directly supplied structured V# whose SRSRC
                    // was never rewritten; retain the established non-v0 faithful-address behavior there.
                    if (res) {
                        const int vaddr = in.src[0].value;
                        switch (res->fetch_index_mode) {
                            case VertexFetchIndexMode::Vertex:
                                dyn_vfetch = true;
                                break;
                            case VertexFetchIndexMode::Instance:
                                instance_vfetch = true;
                                break;
                            case VertexFetchIndexMode::Shader:
                                break;
                            case VertexFetchIndexMode::Automatic:
                                // Backward-compatible fallback for metadata resources, hand-built
                                // tests, and captures predating explicit dynamic-fold provenance.
                                dyn_vfetch = res->cls == ResourceClass::VertexBuffer &&
                                             (vaddr == 0 || srsrc_rewritten);
                                break;
                        }
                    }
                    // MTBUF must resolve through the exact dynamic-use entry. The fold validates the
                    // live V# FORMAT != INVALID before publishing that pc. Falling back to an older
                    // metadata resource here can resurrect an unbound V# that happens to share its
                    // SGPR/SRT identity. MUBUF retains its established metadata fallbacks.
                    if (in.fmt != Rdna2Format::MTBUF) {
                        if (!res && !srsrc_rewritten)
                            res = rt->by_sgpr_base_cls(in.src[1].value, ResourceClass::VertexBuffer);
                        uint32_t srt_tag = 0;
                        if (!res && sreg_srt_range_tag(rs, in.src[1].value, 4, srt_tag))
                            res = rt->by_srt_offset(srt_tag);
                        // DIRECT user-data V# of any class (#273 — DOLL's title post PSes format-fetch
                        // through a V# the metadata labels a CONSTANT buffer sharp at s[24:27]): the class
                        // label doesn't change the descriptor's fields. Only when the SGPR was never
                        // REWRITTEN in-shader (no rs.sreg entry in its four-dword range) — a reloaded
                        // register no longer holds the seed-time sharp, and trusting it would fetch through
                        // a stale descriptor.
                        if (!res && !srsrc_rewritten)
                            res = rt->by_sgpr_base(in.src[1].value);
                    }
                }
                if (!res) {
                    if (getenv("PROSPER_DBG")) {   // which provenance step failed for this format load
                        uint32_t srt_tag = 0;
                        const bool has_srt_tag = sreg_srt_range_tag(
                            rs, in.src[1].value, 4, srt_tag);
                        const ShaderResource* tagged_res = has_srt_tag && rt
                                                               ? rt->by_srt_offset(srt_tag)
                                                               : nullptr;
                        fprintf(stderr, "[mubuf-unresolved] pc=%u srsrc=s%d srt_tag=%s0x%x key_res=%s (%zu res)\n",
                                in.pc, in.src[1].value, has_srt_tag ? "" : "NONE ",
                                has_srt_tag ? srt_tag : 0u,
                                !rt ? "no-table" : (tagged_res ? "yes" : "null"),
                                rt ? rt->resources.size() : 0u);
                    }
                    ok = false; return true;
                }
                binding = res->binding;
                stride = res->stride;
                if (in.fmt == Rdna2Format::MTBUF) {
                    // Unlike MUBUF format ops, MTBUF owns the type in the instruction. gfx1030 uses
                    // the same combined 7-bit BUF_FMT table as Gen5 V# descriptors.
                    rdna2_buffer_format(in.mtbuf_format, &fmt, &fmt_ncomp);
                    if (fmt == DataFormat::Unknown || fmt_ncomp == 0) {
                        ok = false; return true;
                    }
                } else {
                    fmt = res->format;
                    fmt_ncomp = res->num_components;   // format default-fill below (#368)
                }
            } else if (rt) {
                // RAW (untyped) MUBUF with a resource table: resolve SRSRC (src[1]) exactly like the
                // format path — a raw buffer op targets whatever buffer its V# describes, NOT a fixed
                // binding (#91: the old hardcoded binding-2 silently read/wrote the wrong buffer for
                // any other target). Raw ops don't imply a resource class the way a format load implies
                // VertexBuffer, so the direct-SGPR lookup is class-unrestricted (by_sgpr_base).
                // Provenance order mirrors the format path: exact fetch pc, then s_load SRT tag
                // (indirect), then user-data SGPR (direct).
                const ShaderResource* res = rt->by_fetch_pc(in.pc);
                uint32_t srt_tag = 0;
                const bool has_srt_tag = sreg_srt_range_tag(rs, in.src[1].value, 4, srt_tag);
                if (!res && has_srt_tag)
                    res = rt->by_srt_offset(srt_tag);
                if (!res && !sreg_range_written(rs, in.src[1].value, 4))
                    res = rt->by_sgpr_base(in.src[1].value);
                if (!res) {
                    if (getenv("PROSPER_DBG")) {
                        const ShaderResource* pp = rt->by_fetch_pc(in.pc);
                        fprintf(stderr,
                                "[mubuf-raw-unresolved] pc=%u srsrc=s%d srt_tag=%s0x%x "
                                "key_res=%s pc_res=%s rewritten=%d (%zu res)\n",
                                in.pc, in.src[1].value, has_srt_tag ? "" : "NONE ", srt_tag,
                                has_srt_tag && rt->by_srt_offset(srt_tag) ? "yes" : "null",
                                pp ? "yes" : "null",
                                sreg_range_written(rs, in.src[1].value, 4), rt->resources.size());
                    }
                    ok = false; return true;   // unresolvable V# -> reject; NEVER default to binding 2
                }
                binding = res->binding;
                stride  = res->stride;
                // fmt stays raw Uint32: untyped ops move raw dwords regardless of the V#'s declared format.
            }
            if (raw_subword) {
                if (raw_bits == 8) fmt = raw_signed ? DataFormat::Sint8 : DataFormat::Uint8;
                else               fmt = raw_signed ? DataFormat::Sint16 : DataFormat::Uint16;
            }
            // else (rt == nullptr): table-less offline compute shell (recompile_valu without a resource
            // table — the unit-test harness). Keep the legacy single-cbuf convention: binding 2, stride 0.
            // The live graphics path can never reach here table-less: recompile_vertex/recompile_fragment
            // set allow_smem = (rt != nullptr), so MUBUF already rejected above when rt is null there.
            const bool packed_10_11_11 = fmt == DataFormat::Float10_11_11;
            const bool packed_2_10_10_10 =
                fmt == DataFormat::Unorm2_10_10_10 || fmt == DataFormat::Snorm2_10_10_10 ||
                fmt == DataFormat::Uint2_10_10_10  || fmt == DataFormat::Sint2_10_10_10;
            const bool packed_word = packed_10_11_11 || packed_2_10_10_10;
            const uint32_t comp_bytes = data_format_bytes(fmt);
            if (!packed_word && comp_bytes == 0) { ok = false; return true; } // unknown / unsupported
            // Per-component decode. 4-byte formats (Float32/Uint32/Sint32) are a raw dword load — no
            // conversion in our bit model. Sub-dword formats are unpacked: UNORM/SNORM normalize an
            // integer field, Float16 unpacks a packed half. num_components components pack tightly.
            const bool packed = packed_word || comp_bytes < 4;
            bool is_snorm = (fmt == DataFormat::Snorm8 || fmt == DataFormat::Snorm16 ||
                             fmt == DataFormat::Snorm2_10_10_10);
            bool is_half  = (fmt == DataFormat::Float16);
            // Integer sub-dword formats deliver the raw (un-normalized) INTEGER in the VGPR — the
            // hardware's UINT/SINT format-load contract. DOLL's skinned scene VS fetches its bone
            // indices as Uint8 x4 (stride 8, paired with Unorm8 weights); rejecting them dropped
            // every scene-geometry draw (#273). Zero-/sign-extend the field; no normalization.
            bool is_uint = (fmt == DataFormat::Uint8 || fmt == DataFormat::Uint16 ||
                            fmt == DataFormat::Uint2_10_10_10);
            bool is_sint = (fmt == DataFormat::Sint8 || fmt == DataFormat::Sint16 ||
                            fmt == DataFormat::Sint2_10_10_10);
            // A buffer_load_FORMAT whose V# type is a sub-dword integer (Uint8/Sint8/Uint16/Sint16)
            // reads tightly-packed integer components at a RUNTIME byte address — exactly like the raw
            // buffer_load_ubyte/ushort path, NOT the descriptor-defined statically-aligned packing the
            // norm/half formats use. DOLL's post-process LUT compute kernels index a stride-1 Uint8
            // table this way (`buffer_load_format_x v, vIDX, V#`, idxen), which the aligned packed path
            // below wrongly rejected as unaligned. Handle it with a runtime byte/halfword extract.
            const bool int_subword = is_format && !raw_subword && (is_uint || is_sint) &&
                                     (comp_bytes == 1 || comp_bytes == 2);
            float norm = 0.0f;
            switch (fmt) {
                case DataFormat::Unorm8:  norm = 255.0f;   break;
                case DataFormat::Snorm8:  norm = 127.0f;   break;
                case DataFormat::Unorm16: norm = 65535.0f; break;
                case DataFormat::Snorm16: norm = 32767.0f; break;
                default: break;
            }
            if (packed && !packed_word && !is_half && !is_uint && !is_sint && norm == 0.0f) {
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr, "[mubuf-badfmt] pc=%u fmt=%u comp_bytes=%u stride=%u n=%u\n",
                            in.pc, (unsigned)fmt, comp_bytes, stride, n);
                ok = false; return true;
            }
            // Most packed (sub-dword) components below use static fields relative to a DWORD-ALIGNED
            // element base. Float16 LOADS have a general runtime-address path: each component is read
            // from addr+k*2, joining adjacent dwords if the 16-bit field starts at byte 3. Other packed
            // formats still require a proven aligned base; reject them instead of silently dropping the
            // low address bits. Aligned iff: inst offset %4==0; stride %4==0 when idxen; no offen; and
            // SOFFSET is NULL/0. Stores retain their existing stricter paths.
            bool dyn_half = false;   // Float16 components at an arbitrary runtime byte address
            bool dyn_int = false;    // integer sub-dword FORMAT component at a runtime (unaligned) byte addr
            bool dyn_norm = false;   // normalized sub-dword FORMAT component at an arbitrary byte addr
            bool dyn_int_store = false;  // integer sub-dword FORMAT store via race-free atomic clear+set
            if (packed && !raw_subword) {
                bool base_aligned;
                if (folded_vfetch) {
                    // The dyn_vfetch address path (below) is exactly gl_VertexIndex*stride and DROPS the
                    // shader's inst-offset / offen / SOFFSET — the resolved V# base already folds in this
                    // attribute's in-record byte offset. So packed-component alignment depends ONLY on the
                    // per-element stride being dword-aligned (vertex records are, and the V# base with
                    // them). Requiring soff_zero here (as the general path does) wrongly rejected Unorm8×4
                    // packed-color attributes whose fetch carries a register SOFFSET that dyn_vfetch drops
                    // anyway — the "mask=0xF draws never render" gap on PPSA01885. CONFIDENCE: HIGH.
                    base_aligned = !idxen || (stride & 3u) == 0;
                } else {
                    bool soff_zero = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                                     (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                    base_aligned = ((offset & 3u) == 0) && !offen &&
                                   (!idxen || (stride & 3u) == 0) && soff_zero;
                }
                if (!base_aligned) {
                    // Float16 format loads use the complete runtime byte address for every requested
                    // component. This covers both the stride-2 single-half path (#273) and Astro's
                    // Float16x4 vertex record with a register SOFFSET. Loads only: packed half stores
                    // remain fail-visible until their race-free sub-dword write contract is modeled.
                    bool soff_zero = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                                     (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                    dyn_half = !packed_word && !is_store && is_half;
                    // An integer sub-dword FORMAT load extracts each component at its runtime byte
                    // address (join the straddled dwords, then bfe), so it needs no static alignment.
                    // Only LOADS: the packed store path still rejects sub-dword ints (they don't pack).
                    if (!dyn_half) dyn_int = int_subword && !is_store;
                    // UNORM/SNORM 8/16 loads use the same runtime extraction, followed by the format's
                    // normalization. Astro's world-map VS fetches SNORM16x3 with a shader-computed
                    // SOFFSET; treating it as statically packed rejected the complete map draw.
                    if (!dyn_half && !dyn_int)
                        dyn_norm = !packed_word && !is_store && norm != 0.0f &&
                                   (comp_bytes == 1 || comp_bytes == 2);
                    // An integer sub-dword FORMAT STORE writes ONE lane's disjoint bit field of the
                    // containing dword. A plain masked read-modify-write would race (adjacent lanes' fields
                    // share a dword), but atomicAnd(clear field)+atomicOr(set field) COMMUTE across lanes
                    // writing DISJOINT fields, so the store is race-free. Requires the field to lie within
                    // a single dword: address comp_bytes-aligned, no runtime per-lane byte offset. (A
                    // straddling field would span two dwords -> deferred fail-visibly.)
                    if (!dyn_half && !dyn_int && is_store && int_subword && !offen && !dyn_vfetch &&
                        (offset % comp_bytes) == 0 && (!idxen || (stride % comp_bytes) == 0) && soff_zero)
                        dyn_int_store = true;
                    if (!dyn_half && !dyn_int && !dyn_norm && !dyn_int_store) {
                        if (getenv("PROSPER_DBG"))
                            fprintf(stderr, "[mubuf-unaligned] pc=%u fmt=%u off=%u offen=%d idxen=%d stride=%u\n",
                                    in.pc, (unsigned)fmt, offset, (int)offen, (int)idxen, stride);
                        ok = false; return true;
                    }
                }
            }
            // Byte address of the element (#148): idxen -> a VADDR VGPR is an element index (×stride);
            // offen -> a per-lane byte offset; both terms ADD; when idxen AND offen are set VADDR is TWO
            // consecutive VGPRs ([0]=index, [1]=byte offset). Plus the inst offset and SOFFSET.
            uint32_t addr;
            // Const-folded per-vertex attribute fetch (#206): the element address is exactly
            // gl_VertexIndex*stride. (1) The NGG fetch-shader prologue that computes the element index
            // isn't fully modeled and folds to a constant, so every vertex would read record 0 -> a
            // degenerate single point that rasterizes nothing (the whole scene stayed the blue clear).
            // (2) The resolved V# base ALREADY includes this attribute's byte offset within the
            // interleaved record, so the shader's inst-offset + SOFFSET must NOT be added again
            // (double-counting pushes the read OOB -> robustBufferAccess 0, the same collapse). So use
            // gl_VertexIndex*stride and drop the shader's VADDR/offset/SOFFSET; everything else keeps the
            // faithful address (incl. #148's idxen+offen both-terms fix). CONFIDENCE: HIGH — makes
            // PPSA01885 (Unity/IL2CPP) render real geometry instead of a degenerate collapse.
            if (folded_vfetch && idxen && stride) {
                const uint32_t element = dyn_vfetch ? b.load_vertex_index()
                                       : instance_vfetch ? b.load_instance_index()
                                                         : val(in.src[0]);
                addr = b.ibin(Op_IMul, element, b.uconst(stride));
            } else {
                addr = b.uconst(offset);
                if (idxen && stride) addr = b.ibin(Op_IAdd, addr, b.ibin(Op_IMul, val(in.src[0]), b.uconst(stride)));
                if (offen) {
                    Operand off_vgpr{ OperandKind::VGPR, idxen ? in.src[0].value + 1 : in.src[0].value };
                    addr = b.ibin(Op_IAdd, addr, val(off_vgpr));
                }
                addr = b.ibin(Op_IAdd, addr, val(in.src[2]));          // SOFFSET
            }
            uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
            if (is_atomic) {
                const int d = in.dst.value;
                const uint32_t old = vreg_old(b, rs, d);
                const auto it = rs.vreg.find(d);
                const uint32_t value = it == rs.vreg.end() ? b.uconst(0) : it->second;
                const uint32_t pre = b.cbuf_atomic_rtn(atomic_op, idx, value, binding,
                                                       rs.exec_narrowed, rs.exec, old);
                // ISA 8.1 / Table 98: for atomics GLC means "return pre-op value to VGPR". With
                // GLC=0 hardware leaves VDATA untouched (it still holds the DATA operand) — the
                // unconditional write clobbered it (the exercised Astro packet 0xe0e00004 is GLC=0).
                if (in.mubuf_glc) rs.vreg[d] = pre;
                return true;
            }
            if (is_store) {
                auto vread = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
                if (dyn_int_store) {
                    // Integer sub-dword store: clear then set THIS lane's disjoint field of the containing
                    // dword with two atomics. Disjoint fields commute (And clears only this field's bits,
                    // Or sets only this field's bits), so no read-modify-write lock is needed; EXEC
                    // predication keeps inactive lanes from writing (like cbuf_store). CONFIDENCE: HIGH —
                    // this diverges from a hardware byte-enable store ONLY in already-UB situations: two
                    // lanes storing to the SAME element OR-merge instead of one-winner, and a racing reader
                    // could observe the transient post-And zero. Both require a data race a well-formed
                    // shader never has. A straddling field (excluded by the alignment guard above) is the
                    // one shape this can't express and stays deferred.
                    const uint32_t field_mask = comp_bytes == 2 ? 0xffffu : 0xffu;
                    for (uint32_t k = 0; k < n; k++) {
                        const uint32_t caddr = k ? b.ibin(Op_IAdd, addr, b.uconst(k * comp_bytes)) : addr;
                        const uint32_t cidx  = b.ibin(Op_ShiftRightLogical, caddr, b.uconst(2));
                        const uint32_t bitpos = b.ibin(Op_ShiftLeftLogical,
                                                       b.ibin(Op_BitwiseAnd, caddr, b.uconst(3)), b.uconst(3));
                        const uint32_t mask = b.ibin(Op_ShiftLeftLogical, b.uconst(field_mask), bitpos);
                        const uint32_t v = b.ibin(Op_ShiftLeftLogical,
                                                  b.ibin(Op_BitwiseAnd, vread(in.dst.value + (int)k),
                                                         b.uconst(field_mask)), bitpos);
                        b.cbuf_atomic_rtn(Op_AtomicAnd, cidx, b.iun(Op_Not, mask), binding,
                                          rs.exec_narrowed, rs.exec, b.uconst(0));
                        b.cbuf_atomic_rtn(Op_AtomicOr, cidx, v, binding,
                                          rs.exec_narrowed, rs.exec, b.uconst(0));
                    }
                    return true;
                }
                // Store the VDATA VGPRs (in.dst..+n-1). Integer sub-dword formats reach the atomic path
                // above when in-dword-provable; anything else that can't pack (packed_word, or an
                // integer field that could straddle) rejects rather than mis-store.
                if (packed_word || (packed && (is_uint || is_sint))) { ok = false; return true; }
                // MTBUF's instruction format owns the physical component count. A wider opcode still uses
                // identity selection (for example XY00), so Z/W must not spill into adjacent memory.
                const uint32_t store_n = in.fmt == Rdna2Format::MTBUF && fmt_ncomp < n
                                           ? fmt_ncomp : n;
                if (!packed) {
                    // Raw/Float32/Uint32: one dword per component.
                    for (uint32_t k = 0; k < store_n; k++) {
                        uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                        b.cbuf_store(kidx, vread(in.dst.value + (int)k), binding, rs.exec_narrowed, rs.exec);
                    }
                } else {
                    // Packed UNORM/SNORM/Float16: pack the components tightly into ceil(n*bytes/4) dwords
                    // (inverse of the packed load). Each dword ORs together the fields that land in it.
                    const uint32_t dwords = (store_n * comp_bytes + 3) / 4;
                    for (uint32_t d = 0; d < dwords; d++) {
                        uint32_t acc = b.uconst(0);
                        for (uint32_t k = 0; k < store_n; k++) {
                            uint32_t byte_off = k * comp_bytes;
                            if (byte_off / 4 != d) continue;
                            uint32_t field = is_half ? b.pack_half_lo(vread(in.dst.value + (int)k))
                                                     : b.pack_norm(vread(in.dst.value + (int)k), comp_bytes * 8, is_snorm, norm);
                            uint32_t sh = (byte_off % 4) * 8;
                            if (sh) field = b.ibin(Op_ShiftLeftLogical, field, b.uconst(sh));
                            acc = b.ibin(Op_BitwiseOr, acc, field);
                        }
                        uint32_t did = d ? b.ibin(Op_IAdd, idx, b.uconst(d)) : idx;
                        b.cbuf_store(did, acc, binding, rs.exec_narrowed, rs.exec);
                    }
                }
                return true;
            }
            // Integer-typed format? Its absent-component default for W/A is integer 1, not float 1.0.
            const bool fmt_is_int = (fmt == DataFormat::Uint8  || fmt == DataFormat::Uint16 || fmt == DataFormat::Uint32 ||
                                     fmt == DataFormat::Sint8  || fmt == DataFormat::Sint16 || fmt == DataFormat::Sint32 ||
                                     fmt == DataFormat::Uint2_10_10_10 || fmt == DataFormat::Sint2_10_10_10);
            for (uint32_t k = 0; k < n; k++) {
                int d = in.dst.value + (int)k;
                uint32_t old = vreg_old(b, rs, d);
                uint32_t value;
                // Format default-fill (#368): a requested component beyond the format's component
                // count is not read from adjacent memory. MUBUF takes DST_SEL from the V# contract
                // (0 for G/B/Z, 1 for A/W). MTBUF forces identity selection from its instruction
                // format (X000/XY00/XYZ0/XYZW), so every absent component is zero.
                if (is_format && fmt_ncomp && k >= fmt_ncomp) {
                    uint32_t one = fmt_is_int ? 1u : 0x3f800000u;   // integer 1 vs float 1.0 (raw bits)
                    value = b.uconst(in.fmt != Rdna2Format::MTBUF && k == 3 ? one : 0u);
                } else if (raw_subword) {
                    // Raw byte/short loads use their full byte address, unlike typed packed-format
                    // loads whose component packing is descriptor-defined. A 16-bit access may begin
                    // at byte 3 and straddle two dwords, so join the adjacent words before extracting.
                    const uint32_t dw0 = b.cbuf_load(idx, binding);
                    const uint32_t byte_in_dw = b.ibin(Op_BitwiseAnd, addr, b.uconst(3));
                    const uint32_t shift = b.ibin(Op_ShiftLeftLogical, byte_in_dw, b.uconst(3));
                    uint32_t joined = b.ibin(Op_ShiftRightLogical, dw0, shift);
                    if (raw_bits == 16) {
                        const uint32_t dw1 = b.cbuf_load(b.ibin(Op_IAdd, idx, b.uconst(1)), binding);
                        const uint32_t inv_shift = b.ibin(
                            Op_BitwiseAnd,
                            b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
                        const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
                        joined = b.ibin(Op_BitwiseOr, joined,
                                        b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)),
                                              b.uconst(0), upper));
                    }
                    value = raw_signed
                        ? b.bfe_s(joined, b.uconst(0), b.uconst(raw_bits))
                        : b.bfe_u(joined, b.uconst(0), b.uconst(raw_bits));
                } else if (!packed) {
                    uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                    value = b.cbuf_load(kidx, binding);                  // raw 32-bit component
                } else if (packed_word) {
                    // All requested components share one packed dword. GFX10 names layouts from high
                    // field to low field, so 2_10_10_10 is logical R/G/B in bits 0/10/20 and A in 30;
                    // 10_11_11 is R/G/B in bits 0/11/22 with widths 11/11/10.
                    uint32_t dw = b.cbuf_load(idx, binding);
                    uint32_t boff = packed_10_11_11 ? (k == 0 ? 0u : k == 1 ? 11u : 22u)
                                                    : (k == 0 ? 0u : k == 1 ? 10u : k == 2 ? 20u : 30u);
                    uint32_t bits = packed_10_11_11 ? (k < 2 ? 11u : 10u) : (k < 3 ? 10u : 2u);
                    if (packed_10_11_11) {
                        value = b.unpack_ufloat(dw, boff, bits);
                    } else if (is_uint) {
                        value = b.bfe_u(dw, b.uconst(boff), b.uconst(bits));
                    } else if (is_sint) {
                        value = b.bfe_s(dw, b.uconst(boff), b.uconst(bits));
                    } else {
                        float field_norm = is_snorm ? (bits == 2 ? 1.0f : 511.0f)
                                                    : (bits == 2 ? 3.0f : 1023.0f);
                        value = b.unpack_norm(dw, boff, bits, is_snorm, field_norm);
                    }
                } else if (dyn_half) {
                    // Float16 component k at byte address addr+k*2. Join the following dword when
                    // the field begins at byte 3; masking the inverse shift avoids SPIR-V's undefined
                    // shift-by-32 case, and the select discards that word when shift==0.
                    const uint32_t caddr = k ? b.ibin(Op_IAdd, addr, b.uconst(k * 2)) : addr;
                    const uint32_t cidx  = b.ibin(Op_ShiftRightLogical, caddr, b.uconst(2));
                    const uint32_t shift = b.ibin(Op_ShiftLeftLogical,
                                                  b.ibin(Op_BitwiseAnd, caddr, b.uconst(3)), b.uconst(3));
                    uint32_t joined = b.ibin(Op_ShiftRightLogical, b.cbuf_load(cidx, binding), shift);
                    const uint32_t dw1 = b.cbuf_load(b.ibin(Op_IAdd, cidx, b.uconst(1)), binding);
                    const uint32_t inv_shift = b.ibin(Op_BitwiseAnd,
                        b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
                    const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
                    joined = b.ibin(Op_BitwiseOr, joined,
                                    b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
                    value = b.unpack_half(joined, 0);
                } else if (dyn_int || dyn_norm) {
                    // Integer/normalized sub-dword FORMAT component k at a runtime byte address
                    // (addr + k*comp_bytes):
                    // shift the loaded dword right by (byteaddr&3)*8, join the next dword when a 16-bit
                    // field straddles the boundary, then extend or normalize the low field. Mirrors the
                    // raw_subword path but per packed component (stride-1 Uint8, unaligned u16/SNORM16).
                    const uint32_t caddr = k ? b.ibin(Op_IAdd, addr, b.uconst(k * comp_bytes)) : addr;
                    const uint32_t cidx  = b.ibin(Op_ShiftRightLogical, caddr, b.uconst(2));
                    const uint32_t shift = b.ibin(Op_ShiftLeftLogical,
                                                  b.ibin(Op_BitwiseAnd, caddr, b.uconst(3)), b.uconst(3));
                    uint32_t joined = b.ibin(Op_ShiftRightLogical, b.cbuf_load(cidx, binding), shift);
                    if (comp_bytes == 2) {
                        const uint32_t dw1 = b.cbuf_load(b.ibin(Op_IAdd, cidx, b.uconst(1)), binding);
                        const uint32_t inv_shift = b.ibin(Op_BitwiseAnd,
                            b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
                        const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
                        joined = b.ibin(Op_BitwiseOr, joined,
                                        b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
                    }
                    value = dyn_norm
                        ? b.unpack_norm(joined, 0, comp_bytes * 8, is_snorm, norm)
                        : is_sint ? b.bfe_s(joined, b.uconst(0), b.uconst(comp_bytes * 8))
                                  : b.bfe_u(joined, b.uconst(0), b.uconst(comp_bytes * 8));
                } else {
                    // Component k lives at byte k*comp_bytes within the element: pick its dword + field.
                    uint32_t byte_off = k * comp_bytes;
                    uint32_t drel = byte_off / 4, boff = (byte_off % 4) * 8;
                    uint32_t did = drel ? b.ibin(Op_IAdd, idx, b.uconst(drel)) : idx;
                    uint32_t dw  = b.cbuf_load(did, binding);
                    value = is_half ? b.unpack_half(dw, boff ? 1u : 0u)
                          : is_uint ? b.bfe_u(dw, b.uconst(boff), b.uconst(comp_bytes * 8))
                          : is_sint ? b.bfe_s(dw, b.uconst(boff), b.uconst(comp_bytes * 8))
                                    : b.unpack_norm(dw, boff, comp_bytes * 8, is_snorm, norm);
                }
                rs.vreg[d] = value;
                predicate_write(b, rs, d, old);
            }
            return true;
        }
        case Rdna2Format::MIMG: {
            // Image op. Needs the resource table for the binding, so gated on allow_smem + rt. Two paths,
            // selected by the resolved resource's class: a STORAGE image (image_load/image_store, no
            // sampler — compute copy/blit) or a sampled TEXTURE (image_sample* / image_load via a combined
            // image+sampler). Other opcodes / NSA / gradient / compare variants are rejected (deferred).
            if (!allow_smem || !rt) { ok = false; return true; }
            const uint32_t SQ_DIM_2D = 1u;
            auto vread = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
            // Resolve the T#/U# via SRSRC (src[1]) provenance: s_load tag (indirect) else user-data SGPR.
            const ShaderResource* res = rt->by_fetch_pc(in.pc);
            if (!res || (res->cls != ResourceClass::Texture && res->cls != ResourceClass::StorageImage)) {
                res = nullptr;
                uint32_t srt_tag = 0;
                if (sreg_srt_range_tag(rs, in.src[1].value, 8, srt_tag))
                    res = rt->by_srt_offset(srt_tag);
                if (!res && !sreg_range_written(rs, in.src[1].value, 8))
                    res = rt->by_sgpr_base(in.src[1].value);
            }
            // Exact per-use provenance wins over table keys. A sample and store may consume the same
            // T# through a colliding offset but require different Vulkan descriptor classes.
            const bool storage_only_op = in.opcode == 0x08 || in.opcode == 0x0f ||
                                         in.opcode == 0x11;
            if ((storage_only_op && res && res->cls != ResourceClass::StorageImage) ||
                (in.opcode != 0x00 && !storage_only_op && in.opcode != 0x0e &&
                 res && res->cls != ResourceClass::Texture))
                res = nullptr;
            if ((!res || (res->cls != ResourceClass::Texture && res->cls != ResourceClass::StorageImage))
                && getenv("PROSPER_DBG")) {
                // Resolution-failure diagnostic: which provenance step failed for this image op.
                uint32_t srt_tag = 0;
                const bool has_srt_tag = sreg_srt_range_tag(rs, in.src[1].value, 8, srt_tag);
                const ShaderResource* pk = has_srt_tag ? rt->by_srt_offset(srt_tag) : nullptr;
                const ShaderResource* pp = rt->by_fetch_pc(in.pc);
                fprintf(stderr, "[mimg-unresolved] pc=%u srsrc=s%d srt_tag=%s0x%x key_res=%s pc_res=%s (%zu res)\n",
                        in.pc, in.src[1].value, has_srt_tag ? "" : "NONE ",
                        has_srt_tag ? srt_tag : 0u,
                        pk ? (pk->cls == ResourceClass::Texture ? "tex" : "other-cls") : "null",
                        pp ? (pp->cls == ResourceClass::Texture ? "tex" : "other-cls") : "null",
                        rt->resources.size());
            }
            if (!res) { ok = false; return true; }
            const bool uint_texture = res->format == DataFormat::Uint8 ||
                                      res->format == DataFormat::Uint16 ||
                                      res->format == DataFormat::Uint32;

            // --- Storage-image path: image_load (0x00), image_store (0x08), and R32_UINT
            // image_atomic_swap/add (0x0f/0x11), without a sampler. ---
            if (res->cls == ResourceClass::StorageImage) {
                const bool is_ld = in.opcode == 0x00;
                const bool is_st = in.opcode == 0x08;
                const bool is_atomic_swap = in.opcode == 0x0f;
                const bool is_atomic_add = in.opcode == 0x11;
                const bool is_atomic = is_atomic_swap || is_atomic_add;
                if (!is_ld && !is_st && !is_atomic) { ok = false; return true; }
                uint32_t dim, ncoord; bool arrayed = false, ms = false;
                switch (in.mimg_dim) {   // SQ_RSRC dim -> SPIR-V Dim + coord count (+ array layer / MSAA sample)
                    case 0: dim = Dim_1D; ncoord = 1; break;                       // 1D
                    case 1: dim = Dim_2D; ncoord = 2; break;                       // 2D
                    case 2: dim = Dim_3D; ncoord = 3; break;                       // 3D
                    case 4: dim = Dim_1D; ncoord = 2; arrayed = true; break;       // 1D_ARRAY (x, layer)
                    case 5: dim = Dim_2D; ncoord = 3; arrayed = true; break;       // 2D_ARRAY (x, y, layer)
                    case 6: dim = Dim_2D; ncoord = 2; ms = true; break;            // 2D_MSAA (x, y) + sample index
                    case 7: dim = Dim_2D; ncoord = 3; arrayed = true; ms = true; break;  // 2D_MSAA_ARRAY (x,y,layer)+sample
                    default: ok = false; return true;   // cube storage images deferred
                }
                if (ms && (is_st || is_atomic)) { ok = false; return true; }
                const uint32_t components = res->num_components ? res->num_components : 1;
                // The live Astro Bot visibility image is an ordinary 2D R32_UINT surface. Keep the
                // first atomic implementation exact and fail-visible for every other image shape;
                // atomics require a typed integer image in Vulkan/SPIR-V rather than Format=Unknown.
                if (is_atomic &&
                    (in.mimg_dim != SQ_DIM_2D || arrayed || ms || in.len_dwords != 2 ||
                     in.mimg_unorm || in.mimg_dmask != 1u ||
                     res->img_dim != 1u || res->depth != 1u || res->depth_compare ||
                     res->format != DataFormat::Uint32 || components != 1u ||
                     !res->width || !res->height || res->in_mip_tail ||
                     res->compression_enabled)) {
                    ok = false;
                    return true;
                }
                const bool ordinary_2d = res->img_dim == 1 && res->depth == 1 &&
                                         !res->depth_compare;
                const bool native_float = ordinary_2d && native_float_storage_image_supported(
                    res->format, components, res->srgb,
                    (b.native_storage_format_support &
                     native_storage_format_support_bit(res->format, components)) != 0);
                const bool packed_r11 = !native_float && b.packed_r11_storage && ordinary_2d &&
                    !arrayed && !ms && !is_atomic &&
                    res->format == DataFormat::Float10_11_11 && components == 3;
                // Existing load/store-only uint images retain the raw uvec4/Format=Unknown contract
                // used by the compute backend. The atomic's exact R32_UINT gate above is the only path
                // that opts into a typed R32ui image.
                const bool compute_atomic_buffer = is_atomic && b.is_compute;
                if (compute_atomic_buffer) {
                    if (!b.declare_compute_atomic_image_buffer(res->binding)) {
                        ok = false;
                        return true;
                    }
                } else {
                    const bool native_r32ui = is_atomic;
                    b.declare_storage_image(res->binding, dim, arrayed, ms, native_float,
                                            (native_r32ui || packed_r11)
                                                ? ImgFmt_R32ui : ImgFmt_Unknown,
                                            packed_r11);
                }
                // Coordinate VGPR per axis. Non-NSA (len==2): consecutive from VADDR (src[0]). NSA (len>2):
                // split across the extra address dwords — coord0 = VADDR, coord k>=1 = byte (k-1) of
                // words[2..3] (dword2 = addr1..4, dword3 = addr5..8). Layout verified via llvm-mc gfx1010.
                const bool nsa = in.len_dwords > 2;
                int va = in.src[0].value;
                auto coord_vgpr = [&](uint32_t k) -> int {
                    if (!nsa || k == 0) return va + (nsa ? 0 : (int)k);
                    uint32_t j = k - 1;
                    return (int)((in.words[2 + j / 4] >> (8 * (j % 4))) & 0xFFu);
                };
                uint32_t coords[3] = {0, 0, 0};
                for (uint32_t k = 0; k < ncoord; k++) coords[k] = vread(coord_vgpr(k));
                uint32_t sample = ms ? vread(coord_vgpr(ncoord)) : 0;   // MSAA sample index = coord after the spatial ones
                if (is_ld) {
                    uint32_t out[4]; b.image_read(res->binding, dim, arrayed, ncoord, coords, out, ms, sample);
                    int vd = in.dst.value, w = 0;   // dmask -> consecutive VDATA VGPRs (LSB=R first)
                    for (uint32_t c = 0; c < 4; c++) if (in.mimg_dmask & (1u << c)) {
                        uint32_t old = vreg_old(b, rs, vd + w); rs.vreg[vd + w] = out[c];
                        predicate_write(b, rs, vd + w, old); w++;
                    }
                } else if (is_st) {
                    // image_store: gather the VDATA VGPRs selected by dmask into an RGBA texel (channels
                    // absent from dmask store as 0). Under a narrowed EXEC (e.g. a grid-tail bounds check),
                    // the write is EXEC-predicated so inactive lanes don't write out-of-range texels.
                    uint32_t vals[4] = { b.uconst(0), b.uconst(0), b.uconst(0), b.uconst(0) };
                    int vd = in.dst.value, w = 0;
                    for (uint32_t c = 0; c < 4; c++) if (in.mimg_dmask & (1u << c)) { vals[c] = vread(vd + w); w++; }
                    b.image_write(res->binding, dim, arrayed, ncoord, coords, vals, rs.exec_narrowed, rs.exec);
                } else {
                    // IMAGE_ATOMIC_SWAP/ADD reads its operand from VDATA. GLC=1 overwrites that VGPR
                    // with the pre-operation texel; GLC=0 leaves VDATA unchanged. The helper's phi
                    // preserves the old register value for EXEC-inactive lanes.
                    const int vd = in.dst.value;
                    const uint32_t old = vreg_old(b, rs, vd);
                    const uint16_t atomic_op = is_atomic_add ? Op_AtomicIAdd : Op_AtomicExchange;
                    uint32_t result;
                    if (compute_atomic_buffer) {
                        const uint32_t index = b.ibin(
                            Op_IAdd, coords[0],
                            b.ibin(Op_IMul, coords[1], b.uconst(res->width)));
                        uint32_t active = b.land(
                            b.ucmp(Op_ULessThan, coords[0], b.uconst(res->width)),
                            b.ucmp(Op_ULessThan, coords[1], b.uconst(res->height)));
                        if (rs.exec_narrowed) active = b.land(rs.exec, active);
                        result = b.cbuf_atomic_rtn(
                            atomic_op, index, vread(vd), res->binding,
                            true, active, old);
                    } else {
                        result = b.image_atomic_u32(
                            atomic_op, res->binding, ncoord, coords, vread(vd),
                            rs.exec_narrowed, rs.exec, old);
                    }
                    if (in.mimg_glc) rs.vreg[vd] = result;
                }
                return true;
            }

            // image_get_resinfo (0x0e): sampled-image dimensions at the integer LOD in VADDR. The
            // DOLL UE4 volume initializer uses dim:3D, dmask:xyz to bounds-check its 8x8x8 dispatch
            // before loading and writing the volume. Array/cube queries remain deferred with their
            // corresponding sampled-image representations.
            if (in.opcode == 0x0e) {
                uint32_t dim;
                if (in.mimg_dim == 0u) dim = Dim_1D;
                else if (in.mimg_dim == 1u) dim = Dim_2D;
                else if (in.mimg_dim == 2u) dim = Dim_3D;
                else { ok = false; return true; }
                if (res->cls != ResourceClass::Texture) { ok = false; return true; }
                if (!b.declare_texture(res->binding, dim, uint_texture)) {
                    ok = false; return true;
                }
                uint32_t out[4]; b.image_get_resinfo(res->binding, dim, vread(in.src[0].value), out);
                int vd = in.dst.value, w = 0;
                for (uint32_t c = 0; c < 4; c++) if (in.mimg_dmask & (1u << c)) {
                    uint32_t old = vreg_old(b, rs, vd + w); rs.vreg[vd + w] = out[c];
                    predicate_write(b, rs, vd + w, old); w++;
                }
                return true;
            }

            // --- Sampled-texture path: image_sample* (0x20/0x24/0x25/0x27) / image_gather4_lz (0x47) /
            // image_load (0x00). 2D (any LOD variant) or 3D (implicit-LOD or LOD-0 sample); NSA allowed
            // (coords gathered below). image_sample = 0x20 (implicit-LOD), image_sample_l = 0x24
            // (explicit LOD in last coord), image_sample_b = 0x25 (implicit-LOD + BIAS in FIRST vaddr),
            // image_sample_lz = 0x27 (LOD 0), image_gather4_lz = 0x47 (2x2 single-channel gather,
            // base level), image_load = 0x00 (integer texel fetch). Opcodes round-trip-verified
            // via llvm-mc gfx1010 (#273).
            // op 0xa0 = the high-bit sibling of IMAGE_SAMPLE (0x20): the decoder builds the 8-bit
            // MIMG opcode as ((word0&1)<<7)|bits[24:18], so 0xa0 = 0x80|0x20. GTA V's (PPSA04263,
            // RAGE) intro/composite pipeline (es=0x2042d6a200 / ps=0x2042d83c00) issues it as a plain
            // 2D texture sample; rejecting it dropped that pipeline's draws and blacked the whole
            // frame (#1140). Lowered as an ordinary implicit-LOD sample it renders the animated
            // Rockstar Games intro logo correctly (live PPSA04263 capture). CONFIDENCE: MED — the base
            // op and the "samples a 2D texture" behavior are live-title evidence; the exact high-bit
            // family (an RDNA2 gfx10.3 sample variant) is not yet llvm-mc round-trip-verified, so a
            // future title exercising its distinguishing modifier may need a dedicated lowering.
            const bool is_sample = (in.opcode == 0x20) || (in.opcode == 0xa0), is_load = (in.opcode == 0x00);
            const bool is_sample_l = (in.opcode == 0x24), is_sample_lz = (in.opcode == 0x27);
            const bool is_sample_b = (in.opcode == 0x25), is_gather_lz = (in.opcode == 0x47);
            const bool is_sample_c_lz = (in.opcode == 0x2f);
            // image_gather4_lz_o = 0x57 (gather at base level with the _o packed-offset operand in the
            // FIRST vaddr — llvm-mc gfx1030 round-trip on live DOLL bytes: 0xf15c0808 "image_gather4_lz_o
            // v[4:7], [v0, v18, v19], ..." — coords follow the offset, matching image_sample_b's
            // modifier-first vaddr convention). DOLL's FXAA/upsample pass PS (#294).
            const bool is_gather_lz_o = (in.opcode == 0x57);
            // image_sample_lz_o = 0x37 (LOD-0 sample with the _o packed-offset FIRST vaddr — llvm-mc
            // gfx1010 round-trip on live DOLL FXAA bytes: 0xf0dc0808/0xf0dc080a "image_sample_lz_o";
            // the offset-adjust folds into the normalized coords, see image_sample_lz_offset_2d).
            const bool is_sample_lz_o = (in.opcode == 0x37);
            // image_sample_d = 0x22: sample with EXPLICIT gradients. For 2D the vaddr packs four
            // derivative dwords FIRST — [Ds/Dx, Dt/Dx, Ds/Dy, Dt/Dy] — then the (u,v) coords, per ISA
            // 8.2.4's "{derivative}{body}" order. Blue Prince (PPSA25009) issues it as a plain 2D texture
            // sample in a fragment shader (ps=0x2011d60100); rejecting it dropped that whole pipeline's
            // draws. Preserve those explicit derivatives with OpImageSampleExplicitLod + Grad: projected,
            // wrapped, atlas, and uniform coordinates do not share the screen quad's implicit derivative.
            // CONFIDENCE: HIGH — operand order is ISA-defined and an execution regression distinguishes the
            // requested gradient-selected mip from the implicit-derivative result.
            const bool is_sample_d = (in.opcode == 0x22);
            // Most 2D_ARRAY forms retain the historical base-slice fallback (#325). Explicit-LOD
            // SAMPLE_L/LZ is different: Astro Bot's world-map kernel selects both layers of a wide
            // texture atlas, so dropping cvg(2) destroys the lookup. Those forms use a real array below.
            const bool dim2d = (in.mimg_dim == 1u || in.mimg_dim == 5u), dim3d = (in.mimg_dim == 2u);
            const bool dimcube = (in.mimg_dim == 3u);   // CUBE: stacked-face 2D lowering (#273, below)
            if (in.mimg_dim == 5u && getenv("PROSPER_GFXLOG"))
                fprintf(stderr, "[recompile] 2D_ARRAY image_sample -> sampled as base slice 0 (array index dropped; #325)\n");
            if ((!is_sample && !is_load && !is_sample_l && !is_sample_lz && !is_sample_b &&
                 !is_sample_c_lz && !is_gather_lz &&
                 !is_gather_lz_o && !is_sample_lz_o && !is_sample_d) || (!dim2d && !dim3d && !dimcube)) { ok = false; return true; }
            if (res->cls != ResourceClass::Texture) { ok = false; return true; }
            // UNRM=1 supplies unnormalized (texel-space) coordinates (Table 100). Compilers set it
            // only on loads/stores/atomics — and our fetch paths are already texel-space — but a
            // SAMPLER op with UNRM set would treat texel coords as normalized and sample a wildly
            // wrong location. Reject the sampler forms until a live title exercises one.
            if (in.mimg_unorm && !is_load) { ok = false; return true; }
            // coords (normalized float for sample, integer texel for load). Non-NSA: consecutive from VADDR.
            // NSA (len>2): coord0 = VADDR, coord k>=1 = byte (k-1) of the extra address dwords words[2..3].
            const bool nsa = in.len_dwords > 2;
            int va = in.src[0].value;
            auto cvg = [&](uint32_t k) -> int { if (!nsa || k == 0) return va + (nsa ? 0 : (int)k);
                uint32_t j = k - 1; return (int)((in.words[2 + j / 4] >> (8 * (j % 4))) & 0xFFu); };
            uint32_t out[4];
            if (dimcube) {
                // CUBE sample (#273 — DOLL's title post PSes sample their reflection probes /
                // skybox with `image_sample_l ..., dim:CUBE`). The compiled coords are the standard
                // AMD cube-processed form (Mesa ac_prepare_cube_coords): vaddr = { sc*rcp(|ma|)+1.5,
                // tc*rcp(|ma|)+1.5, face_id [, lod] } — face coords centered at 1.5 (span [1,2]),
                // face id an integral float from v_cubeid. Our texture backend uploads the cube as
                // SIX FACES STACKED VERTICALLY in one 2D image (w x 6h — see the live_renderer cube
                // path), so the sample lowers to a plain 2D sample at u = x-1, v = (face + clamp
                // (y-1)) / 6 at base LOD (mips aren't uploaded; a >0 LOD clamps to the one level).
                // The in-face clamp stops bilinear bleed across face seams. CONFIDENCE: MED — the
                // coordinate convention is Mesa-verified; face memory layout is validated visually.
                if (!is_sample && !is_sample_l && !is_sample_lz && !is_sample_b &&
                    !is_sample_c_lz) { ok = false; return true; }
                // Modifier-first order: _b carries bias in vaddr0; _c_lz carries DREF there.
                const uint32_t ci = (is_sample_b || is_sample_c_lz) ? 1u : 0u;
                uint32_t x = vread(cvg(ci)), y = vread(cvg(ci + 1)), fid = vread(cvg(ci + 2));
                const uint32_t one = b.uconst(fbits(1.0f)), zero = b.uconst(fbits(0.0f));
                uint32_t uf = b.fbin(Op_FSub, x, one);
                uint32_t vf = b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, b.fbin(Op_FSub, y, one), one), zero);
                uint32_t layer = b.fext1(Glsl_RoundEven,
                                     b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, fid, b.uconst(fbits(5.0f))), zero));
                uint32_t v6 = b.fbin(Op_FMul, b.fbin(Op_FAdd, layer, vf), b.uconst(fbits(1.0f / 6.0f)));
                if (is_sample_c_lz) {
                    // Point-light shadow maps use the same cube-processed coordinates as ordinary
                    // samples, with DREF prepended: [dref, x, y, face]. The renderer uploads cube
                    // faces as one vertical 2D stack, so compare the transformed face coordinate
                    // manually through its ordinary non-compare sampler (#1167/#1169).
                    if (uint_texture || !res->depth_compare) { ok = false; return true; }
                    if (!b.declare_texture(res->binding, Dim_2D, false)) {
                        ok = false; return true;
                    }
                    b.image_sample_dref_manual_2d(res->binding, uf, v6, vread(cvg(0)),
                                                  res->depth_compare_func,
                                                  res->mag_filter != 0u, res->addr_uvw[0],
                                                  res->addr_uvw[1], res->border_color_type, out);
                } else {
                    if (!b.declare_texture(res->binding, Dim_2D, uint_texture)) {
                        ok = false; return true;
                    }
                    b.image_sample_lod_2d(res->binding, uf, v6, b.uconst(0), out);
                }
            } else if (dim3d) {
                // 3D: implicit-LOD / LOD-0 sample, or an integer texel FETCH (image_load — DOLL's
                // color-grade 3D LUT, #273).
                if (!is_sample && !is_sample_lz && !is_load) { ok = false; return true; }
                if (!b.declare_texture(res->binding, Dim_3D, uint_texture)) {
                    ok = false; return true;
                }
                if (is_sample)      b.image_sample_3d(res->binding, vread(cvg(0)), vread(cvg(1)), vread(cvg(2)), out);
                else if (is_load)   b.image_fetch_3d(res->binding, vread(cvg(0)), vread(cvg(1)), vread(cvg(2)), out);
                else                b.image_sample_lod_3d(res->binding, vread(cvg(0)), vread(cvg(1)), vread(cvg(2)),
                                                          b.uconst(0), out);   // _lz: base level
            } else if (is_sample_c_lz) {
                // Astro Bot shadow/visibility packet (opcode 0x2f, dim 2D_ARRAY, NSA). ISA 8.2.5
                // vaddr order is "{offset}{bias}{z-compare}{derivative}{body}" — the z-compare
                // reference PRECEDES the coordinates, so SAMPLE_C_LZ 2D_ARRAY reads
                // [dref, u, v, slice] (the same modifier-first rule the _b/_o paths already use).
                // The earlier [u,v,layer,dref] mapping rotated every operand (#883).
                // EVIDENCE: no rendered-frame validation exists for either order — Astro crashes
                // in engine init before it reaches shadow rendering (#825), and the prior
                // "[u,v,layer,dref]" comment was an un-round-tripped interpretation of the packet,
                // not a captured/disassembled fact. This order is grounded in the ISA plus an
                // llvm-mc gfx1030 round-trip of a canonical c_lz 2D_ARRAY NSA packet
                // (image_sample_c_lz v5, [v10,v11,v12,v13], ... = 0xf0bc012a 0x0040050a 0x000d0c0b:
                // four consecutive address VGPRs, dref at slot 0 per 8.2.5) and LLVM's own
                // llvm.amdgcn.image.sample.c.lz lowering (dref before coords). CONFIDENCE: MED —
                // the #825 lane must re-validate against a real rendered shadow once it lights up.
                // Integer views are not legal for this lowering, and a non-shadow S# means the
                // provenance drifted; reject rather than silently turning a comparison sample into
                // an ordinary color read.
                if (uint_texture || !res->depth_compare) { ok = false; return true; }
                if (in.mimg_dim == 5u) {
                    if (b.is_compute) {
                        // Compute has a backend-reflected 2D-array path. Preserve [dref,u,v,slice]
                        // and perform the comparison manually over an ordinary color sampler, so no
                        // compare-sampler/Dref Vulkan contract is required. Astro Bot's world-map
                        // visibility shader selects among sixteen depth layers here.
                        if (!b.declare_texture(res->binding, Dim_2D, false, true)) {
                            ok = false; return true;
                        }
                        b.image_sample_dref_manual_2d(
                            res->binding, vread(cvg(1)), vread(cvg(2)), vread(cvg(0)),
                            res->depth_compare_func, res->mag_filter != 0u,
                            res->addr_uvw[0], res->addr_uvw[1], res->border_color_type,
                            out, true, vread(cvg(3)));
                    } else {
                        // The shared graphics backend currently exposes 2D_ARRAY textures as its
                        // documented base-slice 2D view (#325). Keep that established fallback for
                        // graphics until its resource uploader can create matching array views.
                        if (!b.declare_texture(res->binding, Dim_2D, false)) {
                            ok = false; return true;
                        }
                        b.image_sample_dref_manual_2d(
                            res->binding, vread(cvg(1)), vread(cvg(2)), vread(cvg(0)),
                            res->depth_compare_func, res->mag_filter != 0u,
                            res->addr_uvw[0], res->addr_uvw[1], res->border_color_type, out);
                    }
                } else if (in.mimg_dim == 1u) {
                    // Plain 2D form (Blue Prince's lit-material PSes, #1271: 436 rejects/run, all
                    // op 0x2f dim 1 dmask 0x1 — the entire shadowed lighting pass dropped and the
                    // scene rendered unattenuated/blown-out). Same 8.2.5 vaddr order with no array
                    // slice: [dref, u, v]. Lowered as a manual compare against the color-sampled
                    // shadow map (see image_sample_dref_manual_2d for why not a compare sampler).
                    if (!b.declare_texture(res->binding, Dim_2D, false)) {
                        ok = false; return true;
                    }
                    b.image_sample_dref_manual_2d(res->binding, vread(cvg(1)), vread(cvg(2)),
                                                  vread(cvg(0)), res->depth_compare_func,
                                                  res->mag_filter != 0u, res->addr_uvw[0],
                                                  res->addr_uvw[1], res->border_color_type, out);
                } else { ok = false; return true; }
            } else if (is_gather_lz || is_gather_lz_o) {
                // gather4 dmask selects ONE channel (must be a single bit); the result is always the
                // four texels of that channel -> 4 consecutive VDATA VGPRs, gather order preserved.
                uint32_t dm = in.mimg_dmask;
                if (dm != 1u && dm != 2u && dm != 4u && dm != 8u) { ok = false; return true; }
                uint32_t comp = dm == 1u ? 0u : dm == 2u ? 1u : dm == 4u ? 2u : 3u;
                if (!b.declare_texture(res->binding, Dim_2D, uint_texture)) {
                    ok = false; return true;
                }
                if (is_gather_lz_o)   // vaddr order for _o: [packed offset, u, v]
                    b.image_gather_offset_2d(res->binding, vread(cvg(1)), vread(cvg(2)), comp, vread(cvg(0)), out);
                else
                    b.image_gather_2d(res->binding, vread(cvg(0)), vread(cvg(1)), comp, out);
                int vd = in.dst.value;
                for (uint32_t c = 0; c < 4; c++) {
                    uint32_t old = vreg_old(b, rs, vd + (int)c);
                    rs.vreg[vd + (int)c] = out[c];
                    predicate_write(b, rs, vd + (int)c, old);
                }
                return true;
            } else {
                const bool array_sample = b.is_compute && in.mimg_dim == 5u &&
                    res->img_dim == 5u && (is_sample || is_sample_l || is_sample_lz);
                if (!b.declare_texture(res->binding, Dim_2D, uint_texture, array_sample)) {
                    ok = false; return true;
                }
                if (array_sample) {
                    // Compute SAMPLE and SAMPLE_LZ both resolve level zero; SAMPLE_L supplies its
                    // explicit LOD. All retain the 2D-array slice in the SPIR-V coordinate.
                    b.image_sample_lod_2d_array(
                        res->binding, vread(cvg(0)), vread(cvg(1)),
                        vread(cvg(2)),
                        is_sample_l ? vread(cvg(3)) : b.uconst(fbits(0.0f)), out);
                } else if (is_sample_b) {      // vaddr order for _b: [bias, u, v]
                    b.image_sample_bias_2d(res->binding, vread(cvg(1)), vread(cvg(2)), vread(cvg(0)), out);
                } else if (is_sample_lz_o) {   // vaddr order for _o: [packed offset, u, v]
                    b.image_sample_lz_offset_2d(res->binding, vread(cvg(1)), vread(cvg(2)), vread(cvg(0)), out);
                } else if (is_sample_d) {   // vaddr order for _d: [Ds/Dx, Dt/Dx, Ds/Dy, Dt/Dy, u, v]
                    b.image_sample_grad_2d(
                        res->binding, vread(cvg(4)), vread(cvg(5)),
                        vread(cvg(0)), vread(cvg(1)), vread(cvg(2)), vread(cvg(3)), out);
                } else {
                    uint32_t cu = vread(cvg(0)), cv = vread(cvg(1));
                    if (is_sample)         b.image_sample_2d(res->binding, cu, cv, out);
                    else if (is_sample_lz) b.image_sample_lod_2d(res->binding, cu, cv, b.uconst(0), out);      // LOD 0
                    // Explicit-LOD plain 2D uses [u,v,lod]. Graphics keeps the established
                    // base-slice view for DIM=5, whose address is [u,v,slice,lod].
                    else if (is_sample_l)  b.image_sample_lod_2d(res->binding, cu, cv,
                        vread(cvg(in.mimg_dim == 5u && res->img_dim == 5u ? 3u : 2u)), out);
                    else                   b.image_fetch_2d (res->binding, cu, cv, out);
                }
            }
            // dmask selects which result components are written to consecutive VDATA VGPRs (LSB=R first).
            int vd = in.dst.value, w = 0;
            for (uint32_t c = 0; c < 4; c++) if (in.mimg_dmask & (1u << c)) {
                uint32_t old = vreg_old(b, rs, vd + w);
                rs.vreg[vd + w] = out[c];
                predicate_write(b, rs, vd + w, old);
                w++;
            }
            return true;
        }
        case Rdna2Format::DS: {
            if (in.opcode == 0x35) {                    // ds_swizzle_b32: VDST = lane-gather(ADDR)
                if (!b.is_compute && !b.is_fragment) { ok = false; return true; }
                uint32_t source_lane = 0;
                if (!b.ds_swizzle_source_lane(in.literal, &source_lane)) {
                    ok = false; return true;
                }
                auto source = rs.vreg.find(in.src[0].value);
                const uint32_t source_value =
                    source == rs.vreg.end() ? b.uconst(0) : source->second;
                const uint32_t shuffled = b.subgroup_shuffle(source_value, source_lane);
                const uint32_t active_word = b.sel(rs.exec, b.uconst(1), b.uconst(0));
                const uint32_t fetched_active = b.subgroup_shuffle(active_word, source_lane);
                const uint32_t result = b.sel(
                    b.ucmp(Op_INotEqual, fetched_active, b.uconst(0)),
                    shuffled, b.uconst(0));
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = result;
                predicate_write(b, rs, in.dst.value, old);
                return true;
            }
            // GDS=1 targets the device-global data share, not workgroup LDS — running it against
            // our LDS model would silently give per-workgroup semantics. Reject — EXCEPT
            // ds_append/ds_consume (0x3e/0x3d): Astro Bot's live counter words carry GDS=1
            // (0xd8fa/0xd8f6 — llvm-mc gfx1030 round-trip: plain append is 0xd8f8), and the
            // existing wave_append model was landed and live-validated against exactly those
            // packets (#554/#580). Kept as a documented per-workgroup approximation of the
            // device-global counter (exact for the exercised dispatch shapes). CONFIDENCE: MED.
            if (in.ds_gds && in.opcode != 0x3d && in.opcode != 0x3e &&
                !(b.is_compute && in.opcode == 0x0d)) { ok = false; return true; }
            // ds_write_addtid_b32 (0xb0) / ds_read_addtid_b32 (0xb1). AMD RDNA2 ISA 10.4:
            //   LDS address = LDS_BASE + immediate + M0[15:0] + TID(0..63)*4.
            // TID is the lane within the hardware wave, not the workgroup-linear invocation id.
            // Compute therefore uses real Workgroup LDS and wraps only the lane id at wave_size;
            // M0 supplies a distinct base when a workgroup contains more than one wave. The final
            // address itself does not wrap (RDNA2 LDS allocations explicitly do not wrap).
            // Astro Bot's world-map material kernel uses several such lane arrays before reducing
            // them through explicit DS reads and barriers.
            if (b.is_compute && (in.opcode == 0xb0 || in.opcode == 0xb1)) {
                auto m0 = rs.sreg.find(124);
                if (m0 == rs.sreg.end()) { ok = false; return true; }
                b.declare_lds();
                if (!b.lds_var) { ok = false; return true; }
                const uint32_t base = b.ibin(Op_BitwiseAnd, m0->second, b.uconst(0xffffu));
                const uint32_t tid = b.ibin(
                    Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1u));
                const uint32_t byte_addr = b.ibin(
                    Op_IAdd,
                    b.ibin(Op_IAdd, base, b.uconst(in.literal)),
                    b.ibin(Op_ShiftLeftLogical, tid, b.uconst(2)));
                const uint32_t idx = b.ibin(Op_ShiftRightLogical, byte_addr, b.uconst(2));
                if (in.opcode == 0xb0) {
                    auto value = rs.vreg.find(in.src[1].value);
                    b.lds_store(idx, value == rs.vreg.end() ? b.uconst(0) : value->second,
                                rs.exec_narrowed, rs.exec);
                } else {
                    const uint32_t old = vreg_old(b, rs, in.dst.value);
                    rs.vreg[in.dst.value] = b.lds_load(idx);
                    predicate_write(b, rs, in.dst.value, old);
                }
                return true;
            }
            // In a GRAPHICS stage (#273), ADDTID is a compiler spill/reload through per-wave LDS:
            // a per-lane VGPR spill through LDS (addr = M0 + offset + tid*4) — DOLL's title post
            // PSes spill v15 before their accumulation loop and reload it after. Per-invocation the
            // slot is ONE value: track it in rs.lds_addtid keyed by (M0 SSA id, inst offset); the
            // matching read returns the spilled SSA value. A write with UNTRACKED M0 still no-ops
            // (nothing in this model can observe it) but poisons nothing; a read with untracked M0
            // or a never-written slot rejects loudly. VERIFIED(round-trip llvm-mc gfx1010:
            // 0xdac00000/0x00000f00 -> ds_write_addtid_b32 v15; llvm-mc gfx1030:
            // 0xdac40000/0x0f000000 -> ds_read_addtid_b32 v15).
            if (in.opcode == 0xb0 || in.opcode == 0xb1) {
                auto m0 = rs.sreg.find(124);
                if (in.opcode == 0xb0) {
                    if (m0 != rs.sreg.end()) {
                        auto vr = rs.vreg.find(in.src[1].value);
                        rs.lds_addtid[((uint64_t)m0->second << 32) | in.literal] =
                            vr != rs.vreg.end() ? vr->second : b.uconst(0);
                    }
                    return true;   // spill: unobservable beyond the tracked slot
                }
                if (m0 == rs.sreg.end()) { ok = false; return true; }
                auto slot = rs.lds_addtid.find(((uint64_t)m0->second << 32) | in.literal);
                if (slot == rs.lds_addtid.end()) { ok = false; return true; }
                uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = slot->second;
                predicate_write(b, rs, in.dst.value, old);
                return true;
            }
            if (in.opcode == 0x3d || in.opcode == 0x3e) {
                auto m0 = rs.sreg.find(124);
                if (!allow_wave || m0 == rs.sreg.end()) { ok = false; return true; }
                const uint32_t base = b.ibin(Op_BitwiseAnd, m0->second, b.uconst(0xFFFFu));
                const uint32_t byte_addr = b.ibin(
                    Op_BitwiseAnd, b.ibin(Op_IAdd, base, b.uconst(in.literal)),
                    b.uconst(0xFFFFu));
                const uint32_t idx = b.ibin(Op_ShiftRightLogical, byte_addr, b.uconst(2));
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                if (b.is_fragment && in.ds_gds) {
                    rs.vreg[in.dst.value] = b.native_gds_append(
                        idx, rs.exec, in.opcode == 0x3d);
                } else if (b.is_compute) {
                    if (!in.ds_gds) b.declare_lds();
                    rs.vreg[in.dst.value] = b.wave_append(
                        idx, rs.exec, in.opcode == 0x3d, in.ds_gds);
                } else {
                    ok = false; return true;
                }
                predicate_write(b, rs, in.dst.value, old);
                return true;
            }
            // Compute uses Workgroup LDS. Only a mechanically proven NGG projection may use the
            // private graphics backing store; every other graphics DS shape remains fail-closed.
            if ((!b.is_compute && !b.ngg_private_lds) ||
                (in.opcode != 0x00 && in.opcode != 0x05 && in.opcode != 0x06 &&
                                  in.opcode != 0x07 && in.opcode != 0x08 &&
                                  in.opcode != 0x09 && in.opcode != 0x0a &&
                                  in.opcode != 0x0b && in.opcode != 0x20 &&
                                  in.opcode != 0x2d &&
                                  in.opcode != 0x0d && in.opcode != 0x0e &&
                                  in.opcode != 0x36 && in.opcode != 0x37 &&
                                  in.opcode != 0x3d && in.opcode != 0x3e && in.opcode != 0x4d && in.opcode != 0x4e &&
                                  in.opcode != 0x76 && in.opcode != 0x77 &&
                                  in.opcode != 0xde && in.opcode != 0xdf &&
                                  in.opcode != 0xfe && in.opcode != 0xff)) {
                ok = false; return true;
            }
            b.declare_lds();
            if (!b.lds_var) { ok = false; return true; }
            auto vread = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
            if (in.opcode == 0x0e) {                    // ds_write2_b32: two dwords at offset0/offset1
                // AMD RDNA2 ISA 12.13: MEM[ADDR + OFFSET0/1 * 4] = DATA0/1. The packed offsets
                // mirror ds_read2_b32 below; Astro Bot's world-map reduction uses both operations.
                const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
                const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst(in.literal & 0xFFu));
                const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst((in.literal >> 8) & 0xFFu));
                b.lds_store(idx0, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
                // Equal offsets encode one memory access and use DATA0; DATA1 is ignored.
                if ((in.literal & 0xFFu) != ((in.literal >> 8) & 0xFFu))
                    b.lds_store(idx1, vread(in.src[2].value), rs.exec_narrowed, rs.exec);
                return true;
            }
            if (in.opcode == 0x4e) {                    // ds_write2_b64: two VGPR pairs at offset0/offset1
                const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
                const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst((in.literal & 0xFFu) * 2u));
                const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst(((in.literal >> 8) & 0xFFu) * 2u));
                b.lds_store(idx0, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
                b.lds_store(b.ibin(Op_IAdd, idx0, b.uconst(1)), vread(in.src[1].value + 1),
                            rs.exec_narrowed, rs.exec);
                b.lds_store(idx1, vread(in.src[2].value), rs.exec_narrowed, rs.exec);
                b.lds_store(b.ibin(Op_IAdd, idx1, b.uconst(1)), vread(in.src[2].value + 1),
                            rs.exec_narrowed, rs.exec);
                return true;
            }
            if (in.opcode == 0x0e) {                    // ds_write2_b32: two dwords at offset0/offset1
                // AMD RDNA2 ISA 12.13: DATA0/1 go to ADDR + OFFSET0/1 * 4. The packed
                // 8-bit offsets have the same layout as DS_READ2_B32 below.
                const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
                const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst(in.literal & 0xFFu));
                const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst((in.literal >> 8) & 0xFFu));
                b.lds_store(idx0, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
                b.lds_store(idx1, vread(in.src[2].value), rs.exec_narrowed, rs.exec);
                return true;
            }
            if (in.opcode == 0x77) {                    // ds_read2_b64: two pairs at scaled offsets
                const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
                const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst((in.literal & 0xFFu) * 2u));
                const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst(((in.literal >> 8) & 0xFFu) * 2u));
                const uint32_t indices[4] = {
                    idx0, b.ibin(Op_IAdd, idx0, b.uconst(1)),
                    idx1, b.ibin(Op_IAdd, idx1, b.uconst(1)),
                };
                for (int k = 0; k < 4; ++k) {
                    const uint32_t old = vreg_old(b, rs, in.dst.value + k);
                    rs.vreg[in.dst.value + k] = b.lds_load(indices[k]);
                    predicate_write(b, rs, in.dst.value + k, old);
                }
                return true;
            }
            if (in.opcode == 0x37) {                    // ds_read2_b32: two dwords at offset0/offset1
                // AMD RDNA2 ISA 12.13: RETURN_DATA[0/1] = MEM[ADDR + OFFSET0/1 * 4].
                // The two 8-bit offsets share the instruction's 16-bit offset field. Astro's
                // loading compositor uses both adjacent (0,1) and non-zero (16,17) pairs.
                const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
                const uint32_t indices[2] = {
                    b.ibin(Op_IAdd, base, b.uconst(in.literal & 0xFFu)),
                    b.ibin(Op_IAdd, base, b.uconst((in.literal >> 8) & 0xFFu)),
                };
                for (int k = 0; k < 2; ++k) {
                    const uint32_t old = vreg_old(b, rs, in.dst.value + k);
                    rs.vreg[in.dst.value + k] = b.lds_load(indices[k]);
                    predicate_write(b, rs, in.dst.value + k, old);
                }
                return true;
            }
            if (b.ngg_private_lds && in.opcode == 0x36 &&
                in.pc == b.ngg_vertex_index_read_pc && b.ngg_vertex_index_value) {
                // The merged NGG prologue distributes hardware vertex ids between wave lanes through
                // LDS, then the ES consumes that value as its first MUBUF index. In the one-invocation
                // Vulkan shell there is no peer-lane LDS routing: the equivalent value is precisely
                // BuiltIn VertexIndex. This handoff is identified in recompile_vertex from the first
                // MUBUF's vaddr reaching back to this DS read; all other vertex LDS reads remain real.
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = b.ngg_vertex_index_value;
                predicate_write(b, rs, in.dst.value, old);
                return true;
            }
            uint32_t addr = b.ibin(Op_IAdd, vread(in.src[0].value), b.uconst(in.literal));
            if (in.ds_gds)
                addr = b.ibin(Op_BitwiseAnd, addr, b.uconst(0xFFFFu));
            uint32_t idx  = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
            if (in.ds_gds && in.opcode == 0x0d) {
                b.compute_gds_store(idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
                return true;
            }
            if (b.ngg_private_lds && (in.opcode == 0x00 || in.opcode == 0x07 || in.opcode == 0x08 ||
                                     in.opcode == 0x20 || in.opcode == 0x2d)) {
                // Cross-lane/atomic NGG LDS effects do not have a proven single-lane reduction yet.
                ok = false; return true;
            }
            if (in.opcode == 0x00) {                     // ds_add_u32: LDS += DATA0, no VGPR return
                b.lds_atomic(Op_AtomicIAdd, idx, vread(in.src[1].value),
                             rs.exec_narrowed, rs.exec);
            } else if (in.opcode >= 0x05 && in.opcode <= 0x0b) {
                // Non-returning 32-bit LDS atomics map directly to SPIR-V atomics. DS_OR_B32
                // (0x0a) occurs in Plucky's first-gameplay compute path; supporting the adjacent
                // signed/unsigned min/max and bitwise family keeps this architectural rather than
                // making the acceptance rule title-specific.
                const uint32_t atomic_op =
                    in.opcode == 0x05 ? Op_AtomicSMin :
                    in.opcode == 0x06 ? Op_AtomicSMax :
                    in.opcode == 0x07 ? Op_AtomicUMin :
                    in.opcode == 0x08 ? Op_AtomicUMax :
                    in.opcode == 0x09 ? Op_AtomicAnd  :
                    in.opcode == 0x0a ? Op_AtomicOr   : Op_AtomicXor;
                b.lds_atomic(atomic_op, idx, vread(in.src[1].value),
                             rs.exec_narrowed, rs.exec);
            } else if (in.opcode == 0x20) {             // ds_add_rtn_u32: VDST = old LDS; LDS += DATA0
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = b.lds_atomic_rtn(
                    Op_AtomicIAdd, idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec, old);
                predicate_write(b, rs, in.dst.value, old);
            } else if (in.opcode == 0x2d) {             // ds_wrxchg_rtn_b32: swap DATA0, return old LDS
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = b.lds_atomic_rtn(
                    Op_AtomicExchange, idx, vread(in.src[1].value),
                    rs.exec_narrowed, rs.exec, old);
                predicate_write(b, rs, in.dst.value, old);
            } else if (in.opcode == 0x0d) {             // ds_write_b32: LDS[idx] = DATA0
                b.lds_store(idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
            } else if (in.opcode == 0x4d || in.opcode == 0xde || in.opcode == 0xdf) {
                // ds_write_b64 / ds_write_b96 / ds_write_b128. The wide forms consume consecutive
                // DATA0 VGPRs and write consecutive LDS dwords from the ordinary byte address.
                const int dwords = in.opcode == 0x4d ? 2 : in.opcode == 0xde ? 3 : 4;
                for (int k = 0; k < dwords; ++k) {
                    const uint32_t at = k ? b.ibin(Op_IAdd, idx, b.uconst((uint32_t)k)) : idx;
                    b.lds_store(at, vread(in.src[1].value + k), rs.exec_narrowed, rs.exec);
                }
            } else if (in.opcode == 0x76 || in.opcode == 0xfe || in.opcode == 0xff) {
                // ds_read_b64 / ds_read_b96 / ds_read_b128. RDNA2 ISA 12.13 opcodes 118/254/255;
                // all return consecutive dwords beginning at the ordinary LDS byte address.
                const int dwords = in.opcode == 0x76 ? 2 : in.opcode == 0xfe ? 3 : 4;
                for (int k = 0; k < dwords; ++k) {
                    const uint32_t old = vreg_old(b, rs, in.dst.value + k);
                    const uint32_t at = k ? b.ibin(Op_IAdd, idx, b.uconst((uint32_t)k)) : idx;
                    rs.vreg[in.dst.value + k] = b.lds_load(at);
                    predicate_write(b, rs, in.dst.value + k, old);
                }
            } else {                                    // ds_read_b32: VDST = LDS[idx]
                uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = b.lds_load(idx);
                predicate_write(b, rs, in.dst.value, old);
            }
            return true;
        }
        case Rdna2Format::VINTRP: {
            // Pixel-shader attribute interpolation. The rasterizer performs the interpolation; v_interp_p1
            // (opcode 0) initializes and is a no-op here, while v_interp_p2 (1) and v_interp_mov (2)
            // deliver the interpolated attribute component from the matching Input varying. Fragment-only.
            if (!b.is_fragment) { ok = false; return true; }
            if (in.opcode == 0) return true;   // p1: no-op (p2/mov produce the value)
            uint32_t old = vreg_old(b, rs, in.dst.value);
            if (in.opcode == 2) {
                const uint32_t parameter = b.interp_parameter(
                    in.vintrp_attr, in.vintrp_chan, in.src[0].value);
                if (!parameter) { ok = false; return true; }
                rs.vreg[in.dst.value] = parameter;
            } else {
                rs.vreg[in.dst.value] = b.interp_read(in.vintrp_attr, in.vintrp_chan);
            }
            predicate_write(b, rs, in.dst.value, old);
            return true;
        }
        default: return false;   // not a VALU format
    }
}

// PC-relative EMBEDDED-TABLE detection (#273). Compilers put small constant lookup tables (a 4x4
// ordered-dither matrix, etc.) directly after the shader code and address them with an inline-built
// V#: `s_getpc_b64 s[a:a+1]; s_add_u32 sa, <off>, sa; s_addc_u32 sa+1, 0, sa+1; s_mov_b32 sa+2,
// <num_records>; s_mov_b32 sa+3, <cfg>; buffer_load_dwordx4 v, v, s[a:a+3], 0 offen`. The table
// BYTES are inside the code blob we are recompiling, so the load can be folded to a compile-time
// constant lookup. This pass does forward constant propagation over the LINEAR stream (facts are
// killed on any other write; a fact is only used when no branch target enters between its def and
// the load) and returns: mubuf pc -> the table's dwords copied out of the blob.
// CONFIDENCE: MED-HIGH — exact for the compiler idiom (verified against DOLL's dither PS); every
// guard failure falls back to the old reject.
struct PcrelTables {
    std::unordered_map<uint32_t, std::vector<uint32_t>> mubuf;
    std::unordered_map<uint32_t, std::vector<uint32_t>> smem;
};

PcrelTables detect_pcrel_tables(
        const std::vector<Rdna2Inst>& ins, const uint32_t* code, size_t dwords,
        size_t* required_dwords = nullptr) {
    PcrelTables out;
    std::unordered_map<int, uint64_t> pcoff;   // reg -> byte offset into the code blob (pair LO half)
    std::unordered_set<int> pchi;              // regs holding the matching HI half
    std::unordered_map<int, uint32_t> kconst;  // reg -> compile-time constant (s_mov_b32 literal/inline)
    std::unordered_map<int, uint32_t> fact_pc; // reg -> pc where its current fact was established
    std::vector<uint32_t> br_targets;          // all branch targets (for the entry-soundness check)
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
            in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03)
            br_targets.push_back(in.pc + in.len_dwords + (uint32_t)(int32_t)in.simm16);
    }
    auto kill = [&](int r) { pcoff.erase(r); pchi.erase(r); kconst.erase(r); fact_pc.erase(r); };
    auto imm_of = [&](const Operand& o, const Rdna2Inst& in, uint32_t* v) -> bool {
        if (o.kind == OperandKind::Literal)   { *v = in.literal; return true; }
        if (o.kind == OperandKind::InlineInt) { *v = (uint32_t)o.value; return true; }
        if (o.kind == OperandKind::SGPR)      { auto it = kconst.find(o.value);
                                                if (it != kconst.end()) { *v = it->second; return true; } }
        return false;
    };
    for (const auto& in : ins) {
        if (in.is_end) break;
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                kill(in.dst.value);
                if (in.opcode == 0x1f) {                     // s_getpc_b64: pair = address of NEXT inst
                    kill(in.dst.value + 1);
                    pcoff[in.dst.value] = (uint64_t)(in.pc + in.len_dwords) * 4u;
                    pchi.insert(in.dst.value + 1);
                    fact_pc[in.dst.value] = in.pc; fact_pc[in.dst.value + 1] = in.pc;
                } else if (in.opcode == 0x03) {              // s_mov_b32 <const>
                    uint32_t v;
                    if (imm_of(in.src[0], in, &v)) { kconst[in.dst.value] = v; fact_pc[in.dst.value] = in.pc; }
                } else if (in.opcode == 0x04) {              // s_mov_b64 (pair write)
                    kill(in.dst.value + 1);
                }
                break;
            case Rdna2Format::SOP2: {
                int d = in.dst.value;
                if (in.opcode == 0x00) {                     // s_add_u32: pcrel-lo + immediate
                    uint32_t imm; uint64_t base; bool got = false;
                    if (in.src[0].kind == OperandKind::SGPR && pcoff.count(in.src[0].value) &&
                        imm_of(in.src[1], in, &imm)) { base = pcoff[in.src[0].value]; got = true; }
                    else if (in.src[1].kind == OperandKind::SGPR && pcoff.count(in.src[1].value) &&
                             imm_of(in.src[0], in, &imm)) { base = pcoff[in.src[1].value]; got = true; }
                    kill(d);
                    if (got) { pcoff[d] = base + imm; fact_pc[d] = in.pc; }
                } else if (in.opcode == 0x04) {              // s_addc_u32: pcrel-hi + 0 (+carry; a real
                    // carry needs the lo word to wrap over a <4 KB offset — not a shader-blob shape)
                    uint32_t imm;
                    bool hi_ok = (in.src[0].kind == OperandKind::SGPR && pchi.count(in.src[0].value) &&
                                  imm_of(in.src[1], in, &imm) && imm == 0) ||
                                 (in.src[1].kind == OperandKind::SGPR && pchi.count(in.src[1].value) &&
                                  imm_of(in.src[0], in, &imm) && imm == 0);
                    kill(d);
                    if (hi_ok) { pchi.insert(d); fact_pc[d] = in.pc; }
                } else {
                    kill(d);
                    if (in.opcode == 0x29) kill(d + 1);      // s_bfe_u64 writes a pair
                }
                break;
            }
            case Rdna2Format::SOPK:
                // s_movk_i32 is also a common way to fill V# word2 (num_records). Astro Bot's
                // fullscreen-table VS uses it immediately before s_getpc_b64; treating every SOPK
                // as an unknown write prevented the otherwise-proven embedded table from folding.
                // The immediate is sign-extended by the ISA, matching the scalar register value.
                if (sopk_writes_scalar_data(in.opcode)) kill(in.dst.value);
                if (in.opcode == 0x00) {
                    kconst[in.dst.value] = static_cast<uint32_t>(static_cast<int32_t>(in.simm16));
                    fact_pc[in.dst.value] = in.pc;
                }
                break;
            case Rdna2Format::SMEM: {
                uint32_t n = 1;
                switch (in.opcode) { case 0x1: case 0x9: n=2; break; case 0x2: case 0xA: n=4; break;
                    case 0x3: case 0xB: n=8; break; case 0x4: case 0xC: n=16; break; default: break; }
                // Scalar-buffer form of the same compiler idiom: SBASE is the getpc-built V#, while
                // SOFFSET selects a row from the embedded table. Inspect it before killing SDATA facts.
                const int sb = in.src[0].value;
                if (in.opcode >= 0x8 && in.opcode <= 0xC && pcoff.count(sb) &&
                    pchi.count(sb + 1) && kconst.count(sb + 2) && kconst.count(sb + 3)) {
                    const uint32_t nrec = kconst[sb + 2];
                    const uint64_t off = pcoff[sb];
                    uint32_t newest = 0;
                    for (int r : {sb, sb + 1, sb + 2, sb + 3}) {
                        auto fact = fact_pc.find(r);
                        if (fact != fact_pc.end() && fact->second > newest) newest = fact->second;
                    }
                    bool entered = false;
                    for (uint32_t target : br_targets)
                        if (target > newest && target <= in.pc) { entered = true; break; }
                    const bool supported_soffset =
                        (in.src[1].kind == OperandKind::Special && in.src[1].value == 125) ||
                        in.src[1].kind == OperandKind::SGPR ||
                        (in.src[1].kind == OperandKind::Special && in.src[1].value >= 106 &&
                         in.src[1].value <= 123) ||
                        (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0);
                    if (!entered && supported_soffset && static_cast<int32_t>(in.literal) >= 0 &&
                        !(off & 3u) && !(nrec & 3u) && nrec && nrec <= 1024 &&
                        off / 4 + nrec / 4 <= dwords) {
                        if (required_dwords)
                            *required_dwords = std::max(*required_dwords,
                                static_cast<size_t>(off / 4 + nrec / 4));
                        out.smem[in.pc] = std::vector<uint32_t>(
                            code + off / 4, code + off / 4 + nrec / 4);
                    }
                }
                for (uint32_t k = 0; k < n; k++) kill(in.dst.value + (int)k);
                break;
            }
            case Rdna2Format::VOP1:
                if (in.opcode == 0x02) kill(in.dst.value);   // v_readfirstlane -> SGPR
                break;
            case Rdna2Format::VOPC:
                if (in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)
                    { kill(in.dst.value); kill(in.dst.value + 1); }   // e64 compare -> SGPR pair
                break;
            case Rdna2Format::VOP3:
                if (in.sdst.kind == OperandKind::SGPR) { kill(in.sdst.value); kill(in.sdst.value + 1); }
                break;
            case Rdna2Format::MUBUF: {
                if (in.opcode < 0xCu || in.opcode > 0xFu) break;   // raw loads only
                const int sb = in.src[1].value;                    // SRSRC base SGPR of the V# quad
                if (!pcoff.count(sb) || !pchi.count(sb + 1) || !kconst.count(sb + 2)) break;
                const uint32_t nrec = kconst[sb + 2];              // V# word2 = num_records (bytes here)
                const bool offen = (in.literal >> 12) & 1u, idxen = (in.literal >> 13) & 1u;
                const bool soff0 = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                                   (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                const uint64_t off = pcoff[sb];
                // Entry soundness: no branch may enter between the newest contributing fact and here.
                uint32_t newest = 0;
                for (int r : {sb, sb + 1, sb + 2}) { auto it = fact_pc.find(r); if (it != fact_pc.end() && it->second > newest) newest = it->second; }
                bool entered = false;
                for (uint32_t t : br_targets) if (t > newest && t <= in.pc) { entered = true; break; }
                if (entered) break;
                if (idxen || !soff0 || (off & 3u) || (nrec & 3u) || nrec == 0 || nrec > 1024 ||
                    off / 4 + nrec / 4 > dwords) break;            // unprovable / out of window -> reject
                if (required_dwords)
                    *required_dwords = std::max(*required_dwords,
                                                static_cast<size_t>(off / 4 + nrec / 4));
                out.mubuf[in.pc] = std::vector<uint32_t>(code + off / 4, code + off / 4 + nrec / 4);
                (void)offen;                                       // offen just adds the runtime index
                break;
            }
            default: break;   // formats that don't write SGPRs
        }
    }
    return out;
}

namespace {

const Rdna2Inst* last_scalar_writer(const std::vector<Rdna2Inst>& ins, uint32_t before_pc,
                                    int reg) {
    const Rdna2Inst* result = nullptr;
    for (const auto& in : ins) {
        if (in.is_end || in.pc >= before_pc) break;
        bool writes_reg = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            writes_reg |= reg >= base && reg < base + static_cast<int>(width);
        });
        if (writes_reg) result = &in;
    }
    return result;
}

bool reg_operand(const Operand& operand, int reg) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value == reg;
}

bool immediate_operand(const Operand& operand, const Rdna2Inst& in, uint32_t& value) {
    if (operand.kind == OperandKind::Literal) { value = in.literal; return true; }
    if (operand.kind == OperandKind::InlineInt) {
        value = static_cast<uint32_t>(operand.value);
        return true;
    }
    return false;
}

bool binary_reg_immediate(const Rdna2Inst& in, int reg, uint32_t& immediate) {
    return (reg_operand(in.src[0], reg) && immediate_operand(in.src[1], in, immediate)) ||
           (reg_operand(in.src[1], reg) && immediate_operand(in.src[0], in, immediate));
}

bool ordered_reg_immediate(const Rdna2Inst& in, int reg, uint32_t& immediate) {
    return reg_operand(in.src[0], reg) && immediate_operand(in.src[1], in, immediate);
}

bool binary_regs(const Rdna2Inst& in, int a, int b) {
    return (reg_operand(in.src[0], a) && reg_operand(in.src[1], b)) ||
           (reg_operand(in.src[0], b) && reg_operand(in.src[1], a));
}

uint32_t scalar_branch_target(const Rdna2Inst& in) {
    return in.pc + in.len_dwords + static_cast<uint32_t>(in.simm16);
}

PcrelDispatchInfo detect_pcrel_dispatch(const std::vector<Rdna2Inst>& ins,
                                        const uint32_t* code, size_t dwords,
                                        size_t program_dwords) {
    PcrelDispatchInfo out;
    if (!code || ins.empty()) return out;

    std::unordered_set<uint32_t> instruction_pcs;
    std::vector<uint32_t> branch_targets;
    for (const auto& in : ins) {
        instruction_pcs.insert(in.pc);
        if (in.is_end) continue;
        if (in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
            in.opcode != 0x03)
            branch_targets.push_back(scalar_branch_target(in));
    }

    for (const auto& setpc : ins) {
        if (setpc.is_end || setpc.fmt != Rdna2Format::SOP1 || setpc.opcode != 0x20) continue;
        const int jump_lo = setpc.src[0].value;
        const Rdna2Inst* jump_add = last_scalar_writer(ins, setpc.pc, jump_lo);
        const Rdna2Inst* jump_addc = last_scalar_writer(ins, setpc.pc, jump_lo + 1);
        if (!jump_add || !jump_addc || jump_add->fmt != Rdna2Format::SOP2 ||
            jump_add->opcode != 0x00 || jump_addc->fmt != Rdna2Format::SOP2 ||
            jump_addc->opcode != 0x04 || jump_add->pc >= jump_addc->pc) continue;

        int table_lo = -1;
        for (const Operand& source : jump_add->src) {
            if ((source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) &&
                source.value != jump_lo) table_lo = source.value;
        }
        if (table_lo < 0 || !binary_regs(*jump_add, jump_lo, table_lo) ||
            !binary_regs(*jump_addc, jump_lo + 1, table_lo + 1)) continue;
        const Rdna2Inst* target_getpc = last_scalar_writer(ins, jump_add->pc, jump_lo);
        if (!target_getpc || target_getpc->fmt != Rdna2Format::SOP1 ||
            target_getpc->opcode != 0x1f) continue;

        const Rdna2Inst* table_load = last_scalar_writer(ins, jump_add->pc, table_lo);
        if (!table_load || table_load != last_scalar_writer(ins, jump_add->pc, table_lo + 1) ||
            table_load->fmt != Rdna2Format::SMEM || table_load->opcode != 0x01 ||
            table_load->literal != 0 || table_load->src[0].kind != OperandKind::SGPR) continue;
        const int table_base_lo = table_load->src[0].value;
        const int selector = table_load->src[1].value;

        const Rdna2Inst* table_add = last_scalar_writer(ins, table_load->pc, table_base_lo);
        const Rdna2Inst* table_addc = last_scalar_writer(ins, table_load->pc, table_base_lo + 1);
        if (!table_add || !table_addc || table_add->fmt != Rdna2Format::SOP2 ||
            table_add->opcode != 0x00 || table_addc->fmt != Rdna2Format::SOP2 ||
            table_addc->opcode != 0x04 || table_add->pc >= table_addc->pc) continue;
        uint32_t table_delta = 0, high_zero = 1;
        if (!binary_reg_immediate(*table_add, table_base_lo, table_delta) ||
            !binary_reg_immediate(*table_addc, table_base_lo + 1, high_zero) || high_zero != 0)
            continue;
        const Rdna2Inst* table_getpc = last_scalar_writer(ins, table_add->pc, table_base_lo);
        if (!table_getpc || table_getpc->fmt != Rdna2Format::SOP1 ||
            table_getpc->opcode != 0x1f) continue;

        const Rdna2Inst* shift = last_scalar_writer(ins, table_load->pc, selector);
        if (!shift || shift->fmt != Rdna2Format::SOP2 || shift->opcode != 0x1e) continue;
        uint32_t shift_amount = 0;
        // s_lshl_b32 is ordered: selector << 3 scales the qword index, while 3 << selector
        // is a different program and must not be admitted by the commutative matcher.
        if (!ordered_reg_immediate(*shift, selector, shift_amount) || shift_amount != 3) continue;
        const Rdna2Inst* clamp = last_scalar_writer(ins, shift->pc, selector);
        if (!clamp || clamp->fmt != Rdna2Format::SOP2 || clamp->opcode != 0x07) continue;
        uint32_t selector_max = 0;
        if (!binary_reg_immediate(*clamp, selector, selector_max) || selector_max > 63) continue;

        const Rdna2Inst* adjust = last_scalar_writer(ins, clamp->pc, selector);
        int32_t selector_addend = 0;
        if (adjust && adjust->fmt == Rdna2Format::SOP2 &&
            (adjust->opcode == 0x00 || adjust->opcode == 0x02)) {
            uint32_t addend = 0;
            if (!binary_reg_immediate(*adjust, selector, addend)) continue;
            selector_addend = static_cast<int32_t>(addend);
        } else {
            adjust = nullptr;
        }
        const Rdna2Inst* selector_load = last_scalar_writer(
            ins, adjust ? adjust->pc : clamp->pc, selector);
        if (!selector_load || selector_load->fmt != Rdna2Format::SMEM ||
            selector_load->opcode != 0x08 || selector_load->src[0].kind != OperandKind::SGPR ||
            selector_load->src[1].kind != OperandKind::Special ||
            selector_load->src[1].value != 125) continue;

        const uint64_t table_byte =
            static_cast<uint64_t>(table_getpc->pc + table_getpc->len_dwords) * 4u + table_delta;
        const size_t entry_count = static_cast<size_t>(selector_max) + 1;
        if ((table_byte & 7u) || table_byte / 4 < program_dwords ||
            table_byte / 4 + entry_count * 2 > dwords) continue;

        std::vector<uint32_t> targets;
        targets.reserve(entry_count);
        const int64_t target_pc_byte =
            static_cast<int64_t>(target_getpc->pc + target_getpc->len_dwords) * 4;
        bool table_ok = true;
        for (size_t index = 0; index < entry_count; ++index) {
            const size_t word = static_cast<size_t>(table_byte / 4) + index * 2;
            const int64_t relative = static_cast<int64_t>(
                (static_cast<uint64_t>(code[word + 1]) << 32) | code[word]);
            const int64_t target_byte = target_pc_byte + relative;
            if (target_byte < 0 || (target_byte & 3) ||
                target_byte / 4 > static_cast<int64_t>(UINT32_MAX) ||
                !instruction_pcs.contains(static_cast<uint32_t>(target_byte / 4))) {
                table_ok = false;
                break;
            }
            targets.push_back(static_cast<uint32_t>(target_byte / 4));
        }
        if (!table_ok || targets.empty()) continue;
        const uint32_t merge_pc = *std::max_element(targets.begin(), targets.end());
        if (merge_pc <= setpc.pc) continue;
        for (uint32_t target : targets) if (target < setpc.pc + setpc.len_dwords || target > merge_pc)
            table_ok = false;
        if (!table_ok) continue;

        const std::vector<uint32_t> setup = {
            selector_load->pc,
            adjust ? adjust->pc : UINT32_MAX,
            clamp->pc, shift->pc, table_getpc->pc, table_add->pc, table_addc->pc,
            table_load->pc, target_getpc->pc, jump_add->pc, jump_addc->pc, setpc.pc,
        };
        const uint32_t setup_first = selector_load->pc;
        for (uint32_t target : branch_targets) {
            if (target > setup_first && target <= setpc.pc) { table_ok = false; break; }
        }
        if (!table_ok) continue;

        out.valid = true;
        out.selector_sgpr_base = static_cast<uint32_t>(selector_load->src[0].value);
        out.selector_byte_offset = selector_load->literal;
        out.selector_addend = selector_addend;
        out.selector_max = selector_max;
        out.setpc_pc = setpc.pc;
        out.merge_pc = merge_pc;
        out.required_dwords = static_cast<size_t>(table_byte / 4) + entry_count * 2;
        out.target_pcs = std::move(targets);
        for (uint32_t pc : setup) if (pc != UINT32_MAX) out.setup_pcs.push_back(pc);
        return out;
    }
    return out;
}

bool specialize_pcrel_dispatch(std::vector<Rdna2Inst>& ins, const PcrelDispatchInfo& info,
                               uint32_t selected_target) {
    if (!info.valid || std::find(info.target_pcs.begin(), info.target_pcs.end(), selected_target) ==
                           info.target_pcs.end()) return false;
    std::unordered_set<uint32_t> remove(info.setup_pcs.begin(), info.setup_pcs.end());

    // A compiler may jump over an alternate entry prologue before it starts the dispatch setup. Fold
    // only forward unconditional branches wholly contained in that prelude; any external entry into a
    // skipped range makes the specialization unprovable.
    for (const auto& branch : ins) {
        if (branch.pc >= info.setpc_pc || branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x02)
            continue;
        const uint32_t target = scalar_branch_target(branch);
        if (target <= branch.pc || target > info.setpc_pc) return false;
        for (const auto& other : ins) {
            if (other.fmt != Rdna2Format::SOPP || other.pc == branch.pc || other.opcode < 0x02 ||
                other.opcode > 0x09 || other.opcode == 0x03) continue;
            const uint32_t entered = scalar_branch_target(other);
            if (entered > branch.pc + branch.len_dwords && entered < target) return false;
        }
        remove.insert(branch.pc);
        for (const auto& skipped : ins)
            if (skipped.pc >= branch.pc + branch.len_dwords && skipped.pc < target)
                remove.insert(skipped.pc);
    }

    uint32_t selected_end = info.merge_pc;
    for (const auto& in : ins) {
        if (in.pc < selected_target || in.pc >= info.merge_pc) continue;
        if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x02) continue;
        // Internal forward branches and loop back-edges remain in the selected routine and are
        // validated/structured by emit_body. Only the compiler's route terminator jumps to the
        // common merge and can be removed as a now-redundant branch.
        if (scalar_branch_target(in) != info.merge_pc) continue;
        selected_end = in.pc;
        remove.insert(in.pc);
        break;
    }

    std::vector<Rdna2Inst> specialized;
    specialized.reserve(ins.size());
    for (const auto& in : ins) {
        const bool prelude = in.pc < info.setpc_pc;
        const bool selected = in.pc >= selected_target && in.pc < selected_end;
        const bool merge = in.pc >= info.merge_pc;
        if ((prelude || selected || merge) && !remove.contains(in.pc)) specialized.push_back(in);
    }
    if (specialized.empty()) return false;

    // No surviving branch may enter an omitted alternative. The ordinary structurizer performs the
    // remaining detailed CFG checks after this coarse specialization boundary check.
    std::unordered_set<uint32_t> retained;
    uint32_t end_pc = 0;
    for (const auto& in : specialized) retained.insert(in.pc);
    for (const auto& in : specialized) if (in.is_end) { end_pc = in.pc; break; }
    for (const auto& in : specialized) {
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03 || in.simm16 < 0) continue;
        const uint32_t target = scalar_branch_target(in);
        // Existing forward-if validation accepts a compiler early-out just beyond the primary
        // S_ENDPGM only after proving that raw target terminates immediately. Preserve that case for
        // the detailed validator; targets into an omitted alternative still fail here.
        if (!retained.contains(target) && target <= end_pc) return false;
    }
    ins = std::move(specialized);
    return true;
}

} // namespace

bool rdna2_specialize_pcrel_dispatch(std::vector<Rdna2Inst>& instructions,
                                     const PcrelDispatchInfo& info,
                                     uint32_t selected_target) {
    return specialize_pcrel_dispatch(instructions, info, selected_target);
}

PcrelDispatchInfo rdna2_pcrel_dispatch_info(const uint32_t* code, size_t dwords) {
    PcrelDispatchInfo out;
    if (!code || !dwords) return out;
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    return detect_pcrel_dispatch(ins, code, dwords, program_dwords);
}

size_t rdna2_recompile_code_span(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return 0;
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    size_t required = program_dwords;
    std::vector<Rdna2Inst> terminating_cfg = ins;
    size_t terminating_span = program_dwords;
    if (extend_terminating_if_else(code, dwords, terminating_cfg, &terminating_span))
        required = std::max(required, terminating_span);
    // Detection both proves the compiler idiom and bounds every referenced table. Do not retain an
    // arbitrary post-ENDPGM trailer: only bytes that can affect the generated SPIR-V belong in the key.
    (void)detect_pcrel_tables(terminating_cfg, code, dwords, &required);
    const PcrelDispatchInfo dispatch = detect_pcrel_dispatch(ins, code, dwords, program_dwords);
    if (dispatch.valid) required = std::max(required, dispatch.required_dwords);
    return std::min(required, dwords);
}

// Arbitrary CFG fallback. Some UE4 volume-lighting kernels and Astro Bot material shaders contain
// nested EXEC loops plus scalar/VCC branches. They are valid reducible machine CFGs, but deliberately
// exceed the narrow pattern structurizer below. Lower them as a structured dispatcher loop: one
// switch case per basic block, with the emulated register file persisted in Function variables
// between iterations. Graphics stages can branch directly on this invocation's SCC/VCC/EXEC bit;
// compute needs the guest-wave reduction described below.
// VCCZ/EXECZ reduce over the dispatch's 32/64-lane hardware wave, not the host Vulkan subgroup.
// Native subgroup widths are implementation-defined (llvmpipe is commonly 8), so every invocation
// remains in the dispatcher until the whole workgroup is done and exchanges its vote at the common
// switch merge. Dedicated Workgroup scratch keeps multiple hardware waves independent. The fallback
// is gated by emit_body to complex compute CFGs without guest barriers. Ordinary LDS reads, writes,
// and atomics remain valid while waves visit different cases. V_MBCNT is split into a dedicated
// common phase so every workgroup invocation reaches its synthesized barriers in uniform control flow.
bool emit_cfg_state_machine(
    SpirvCompute& b, RegState& initial, const std::vector<Rdna2Inst>& ins,
    const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
    bool allow_exec_update, bool allow_smem,
    const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn,
    const uint32_t* code, size_t dwords) {
    const bool graphics = b.is_fragment || b.is_vertex;
    auto reject_cfg = [&](uint32_t pc, const char* reason) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[graphics-cfg-reject] pc=%u reason=%s\n", pc, reason);
        return false;
    };
    if ((!b.is_compute && !graphics) || ins.empty()) return false;
    // `safe` branches have already been proven equivalent to straight-line predication by the
    // stage-specific analysis (fragment alpha-test wave early-outs, safe EXECZ regions, and the
    // bounded NGG terminal export gate).  The compact SSA emitter feeds them to emit_alu, which
    // deliberately no-ops the scalar branch while retaining the per-invocation EXEC effect.  Do the
    // same in the CFG fallback: treating one as a basic-block terminator makes a kill-mask SCC look
    // like an ordinary scalar boolean, even though the mask lowering intentionally poisons that
    // cross-lane SCC.  Astro Bot's complex material PS combines both shapes and was rejected there.
    auto linearized_branch = [&](const Rdna2Inst& in) {
        return graphics && in.fmt == Rdna2Format::SOPP && safe.contains(in.pc);
    };
    auto cfg_terminator = [&](const Rdna2Inst& in) {
        if (in.is_end) return true;
        if (linearized_branch(in) || in.fmt != Rdna2Format::SOPP) return false;
        const bool branch = in.opcode >= 0x02 && in.opcode <= 0x09 &&
            in.opcode != 0x03;
        return branch || in.opcode == 0x12; // s_trap terminates the guest wave
    };

    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    if (end_pc == UINT32_MAX) return false;
    auto proven_exit_target = [&](uint32_t target) {
        if (target <= end_pc || !code || target >= dwords) return false;
        std::vector<Rdna2Inst> tail;
        rdna2_walk(code + target, dwords - target, tail);
        for (const auto& in : tail) {
            if (in.is_end) return true;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x00) continue;
            break;
        }
        return false;
    };
    if (b.is_compute &&
        ((b.wave_size != 32 && b.wave_size != 64) || !b.local_count || b.local_count > 1024))
        return false;
    const bool direct_dispatch = graphics || b.native_subgroup_size;
    const bool proven_wave32_masks = b.allow_b32_masks &&
        (b.is_fragment || (b.is_compute && b.wave_size == 32));
    const uint32_t wave_count = b.is_compute
        ? (b.local_count + b.wave_size - 1) / b.wave_size : 0;
    const uint32_t padded_lanes = wave_count * b.wave_size;
    const uint32_t wave_result_base = padded_lanes;
    const uint32_t group_active_slot = wave_result_base + wave_count;
    if (b.is_compute && !b.native_subgroup_size)
        b.declare_cfg_scratch(group_active_slot + 1);

    // Discover every scalar pair that lives in the per-lane mask domain.  Besides defining the
    // dispatcher variables below, this identifies s_cmp_{eq,lg}_u64 mask,0: its SCC result is a
    // whole-wave reduction and therefore must be lowered in the common synchronized phase.
    std::set<int> static_mask_keys;
    for (const auto& kv : initial.sreg_bool) static_mask_keys.insert(kv.first);
    for (const auto& in : ins) {
        if (in.is_end) break;
        for_each_scalar_write(in, [&](int base, uint32_t) {
            const bool wave32_one_word_mask = proven_wave32_masks &&
                ((in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) ||
                 (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 &&
                  in.opcode <= 0x12a && base == in.sdst.value));
            if (wave32_one_word_mask)
                static_mask_keys.insert(base);
            else if (base <= 105 && scalar_write_is_b64_mask(in, base))
                static_mask_keys.insert(base);
        }, proven_wave32_masks);
    }
    // Keep block discovery independent of the mask-domain dataflow below.  In particular, a
    // physical Wave32 SGPR can hold a mask in one lifetime and ordinary scalar data in another.
    // Conservatively split every syntactically eligible zero comparison here, then consult the
    // block-entry RegState while emitting it to decide which lifetime is actually live.
    auto mask_zero_compare_candidate_source = [&](const Rdna2Inst& in) -> int {
        if (in.fmt != Rdna2Format::SOPC) return -1;
        auto zero = [](const Operand& o) {
            return o.kind == OperandKind::InlineInt && o.value == 0;
        };
        auto possible_mask = [](const Operand& o) {
            return o.kind == OperandKind::SGPR || o.kind == OperandKind::Special;
        };
        // B64 EQ/LG and these B32 unsigned zero comparisons depend only on whether any mask bit is
        // set. In particular, Wave32 code commonly uses `s_cmp_gt_u32 vcc_lo, 0` after VOPC.
        const bool b64_compare = in.opcode == 0x12 || in.opcode == 0x13;
        const bool b32_mask_first = proven_wave32_masks &&
            (in.opcode == 0x06 || in.opcode == 0x07 ||
             in.opcode == 0x08 || in.opcode == 0x0b);
        const bool b32_mask_second = proven_wave32_masks &&
            (in.opcode == 0x06 || in.opcode == 0x07 ||
             in.opcode == 0x09 || in.opcode == 0x0a);
        if (possible_mask(in.src[0]) && zero(in.src[1]) &&
            (b64_compare || b32_mask_first))
            return in.src[0].value;
        if (possible_mask(in.src[1]) && zero(in.src[0]) &&
            (b64_compare || b32_mask_second))
            return in.src[1].value;
        return -1;
    };
    auto mask_zero_compare_inverts = [&](const Rdna2Inst& in) {
        if (in.opcode == 0x12 || in.opcode == 0x06) return true; // EQ mask,0 / 0,mask
        const bool mask_first =
            (in.src[0].kind == OperandKind::SGPR ||
             in.src[0].kind == OperandKind::Special) &&
            mask_zero_compare_candidate_source(in) == in.src[0].value;
        return mask_first ? in.opcode == 0x0b                 // mask <= 0
                          : in.opcode == 0x09;                // 0 >= mask
    };

    // Split at every branch target/fallthrough and around every cross-lane operation. Case values are
    // dense block indices, not guest PCs. A cross-lane op must end its block so the common synchronized
    // phase can publish its result before any invocation advances to the following guest instruction.
    std::set<uint32_t> start_set{ins.front().pc};
    std::unordered_map<uint32_t, uint32_t> mbcnt_event_for_pc;
    std::unordered_map<uint32_t, uint32_t> append_event_for_pc;
    bool has_gds_append = false;
    bool has_lds_append = false;
    std::unordered_set<uint32_t> swizzle_pcs;
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3 &&
            (in.opcode == 0x365 || in.opcode == 0x366)) {
            mbcnt_event_for_pc.emplace(in.pc,
                static_cast<uint32_t>(mbcnt_event_for_pc.size()));
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::DS && (in.opcode == 0x3d || in.opcode == 0x3e)) {
            append_event_for_pc.emplace(in.pc,
                static_cast<uint32_t>(append_event_for_pc.size()));
            if (in.ds_gds) has_gds_append = true;
            else has_lds_append = true;
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::DS && in.opcode == 0x35) {
            swizzle_pcs.insert(in.pc);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (mask_zero_compare_candidate_source(in) >= 0) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
            continue;
        }
        if (linearized_branch(in) || in.fmt != Rdna2Format::SOPP ||
            in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03) continue;
        const uint32_t target = branch_target(in);
        if (target <= end_pc) start_set.insert(target);
        if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc) start_set.insert(ins[i + 1].pc);
    }
    start_set.insert(end_pc);
    std::vector<uint32_t> starts(start_set.begin(), start_set.end());
    if (starts.empty() || starts.front() != ins.front().pc) return false;
    std::unordered_map<uint32_t, uint32_t> block_for_pc;
    for (uint32_t i = 0; i < starts.size(); ++i) block_for_pc[starts[i]] = i;

    // Persist only registers that the stream reads or writes, plus the caller's initialized inputs.
    std::set<int> vregs, sregs;
    loop_written_regs(ins, 0, end_pc, vregs, sregs);
    for (const auto& in : ins) {
        if (in.is_end) break;
        for (uint32_t k = 0; k < in.n_src; ++k) {
            const Operand& src = in.src[k];
            if (src.kind == OperandKind::VGPR) vregs.insert(src.value);
            else if (src.kind == OperandKind::SGPR ||
                     (src.kind == OperandKind::Special && src.value >= 106 && src.value <= 124))
                sregs.insert(src.value);
        }
        // 64-bit scalar compares encode only the low SGPR of each pair in their two source fields.
        if (in.fmt == Rdna2Format::SOPC && (in.opcode == 0x12 || in.opcode == 0x13))
            for (uint32_t k = 0; k < 2; ++k)
                if (in.src[k].kind == OperandKind::SGPR) sregs.insert(in.src[k].value + 1);
        if (in.fmt == Rdna2Format::SOP2 &&
            (in.opcode == 0x1f || in.opcode == 0x21) &&
            (in.src[0].kind == OperandKind::SGPR ||
             (in.src[0].kind == OperandKind::Special &&
              in.src[0].value >= 106 && in.src[0].value < 124)))
            sregs.insert(in.src[0].value + 1);
    }
    for (const auto& kv : initial.vreg) vregs.insert(kv.first);
    for (const auto& kv : initial.sreg) sregs.insert(kv.first);

    // Direct user-data descriptors are deliberately absent from the initial scalar SSA map: their
    // identity lives in the resource table, and presence in sreg means shader code overwrote them.
    // The dispatcher persists every referenced SGPR in a Function variable, so recover that
    // distinction per block until the first static write; otherwise a direct V# such as UE4's s[8:11]
    // looks rewritten merely because the dispatcher loaded its zero-initialized backing variable.
    std::set<int> direct_descriptor_sregs;
    if (rt) {
        for (const auto& resource : rt->resources) {
            if (resource.srt_offset != 0xFFFFFFFFu || resource.sgpr_base == 0xFFFFFFFFu) continue;
            const uint32_t words = (resource.cls == ResourceClass::Texture ||
                                    resource.cls == ResourceClass::StorageImage) ? 8u : 4u;
            for (uint32_t word = 0; word < words; ++word)
                direct_descriptor_sregs.insert(static_cast<int>(resource.sgpr_base + word));
        }
    }

    // The dispatcher reloads scalar values from Function variables, so map membership cannot say
    // whether shader code has overwritten an entry-time descriptor. Compute a forward MAY-write set
    // for every reachable basic-block entry instead. MAY is intentional: if one reachable predecessor
    // overwrote the descriptor, falling back to entry metadata after the join would be wrong on that
    // path. Entry-rooted reachability excludes writes in dead blocks, while backedges participate in
    // the fixed point and prevent stale fallback on later loop iterations.
    std::vector<std::unordered_set<int>> scalar_writes(starts.size());
    std::vector<std::set<int>> vector_writes(starts.size());
    std::vector<std::set<int>> vector_reads(starts.size());
    std::vector<std::vector<uint32_t>> successors(starts.size());
    for (uint32_t block = 0; block < starts.size(); ++block) {
        const uint32_t lo = starts[block];
        const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
        std::set<int> ignored_scalar_writes;
        loop_written_regs(ins, lo, hi, vector_writes[block], ignored_scalar_writes);
        vector_reads[block] = vector_writes[block];
        bool reads_dynamic_vector_range = false;
        for (const auto& in : ins) {
            if (in.pc < lo || in.pc >= hi) continue;
            for (uint32_t source = 0; source < in.n_src; ++source) {
                if (in.src[source].kind == OperandKind::VGPR) {
                    // DS read/write2 and 64/96/128-bit transfers consume consecutive VGPRs that the
                    // packet represents by their base register. Four is a safe architectural upper
                    // bound for the forms accepted by emit_alu.
                    const uint32_t words = in.fmt == Rdna2Format::DS ? 4u : 1u;
                    for (uint32_t word = 0; word < words; ++word)
                        vector_reads[block].insert(
                            in.src[source].value + static_cast<int>(word));
                }
            }
            // Buffer/image packets have format- and dmask-dependent implicit register ranges (and
            // stores encode VDATA in the decoded destination field). Relative VGPR reads similarly
            // select from a runtime range. Keep those uncommon blocks fully conservative; ordinary
            // ALU/branch blocks can still avoid loading the rest of the vector register file.
            reads_dynamic_vector_range |=
                in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF ||
                in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::FLAT ||
                (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x43);
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                const bool wave32_vop3b = proven_wave32_masks &&
                    in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 &&
                    in.opcode <= 0x12a && base == in.sdst.value;
                const uint32_t effective_width = wave32_vop3b ? 1u : width;
                for (uint32_t word = 0; word < effective_width; ++word)
                    scalar_writes[block].insert(base + static_cast<int>(word));
            }, proven_wave32_masks);
            if (b.allow_b32_masks && in.fmt == Rdna2Format::VOPC &&
                !vopc_is_cmpx(in.opcode) && in.dst.kind == OperandKind::SGPR &&
                (in.dst.value == 106 || in.dst.value == 107))
                scalar_writes[block].insert(in.dst.value);
        }

        const Rdna2Inst* terminator = nullptr;
        for (const auto& in : ins) {
            if (in.pc < lo || in.pc >= hi) continue;
            if (cfg_terminator(in)) {
                terminator = &in;
                break;
            }
        }
        auto add_successor = [&](uint32_t pc) {
            auto next = block_for_pc.find(pc);
            if (pc <= end_pc && next != block_for_pc.end() &&
                std::find(successors[block].begin(), successors[block].end(), next->second) ==
                    successors[block].end())
                successors[block].push_back(next->second);
        };
        if (!terminator) {
            add_successor(hi);
        } else if (!terminator->is_end && terminator->opcode != 0x12) {
            add_successor(branch_target(*terminator));
            if (terminator->opcode != 0x02)
                add_successor(terminator->pc + terminator->len_dwords);
        }
        if (reads_dynamic_vector_range) vector_reads[block] = vregs;
    }

    // Wave32 saved masks occupy one physical SGPR and can therefore be replaced by an ordinary
    // scalar-data lifetime. The dispatcher reloads a statically-shaped register file at every case,
    // so carry the B32 domain as a compile-time property of each basic-block entry. This is exact
    // whenever all reachable predecessors agree. A disagreement means that the same word is a mask
    // on one edge and scalar data on another; keep that genuinely dynamic type join fail-visible.
    //
    // This is deliberately a MUST/equality analysis rather than a union: loading a stale Boolean on
    // the scalar-data edge would be a silent miscompile. Compiler-generated save/restore regions,
    // including Astro Bot's large Wave32 material loop, have identical domains at their joins.
    const std::set<int> static_b64_mask_keys = static_mask_keys;
    std::vector<std::set<int>> b32_mask_in(starts.size());
    std::vector<std::set<int>> b32_mask_ambiguous_in(starts.size());
    std::vector<bool> b32_mask_reachable(starts.size(), false);
    if (b.allow_b32_masks && !starts.empty()) {
        b32_mask_in.front().insert(
            initial.sreg_bool_b32.begin(), initial.sreg_bool_b32.end());
        b32_mask_reachable.front() = true;
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> masks = b32_mask_in[block];
            std::set<int> ambiguous = b32_mask_ambiguous_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;

                bool reads_ambiguous = false;
                for (uint32_t source = 0; source < in.n_src; ++source) {
                    if ((in.src[source].kind == OperandKind::SGPR ||
                         in.src[source].kind == OperandKind::Special) &&
                        ambiguous.contains(in.src[source].value))
                        reads_ambiguous = true;
                }
                if (in.fmt == Rdna2Format::SOPP &&
                    (in.opcode == 0x06 || in.opcode == 0x07) &&
                    ambiguous.contains(106))
                    reads_ambiguous = true;
                if (reads_ambiguous) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[graphics-cfg-reject] pc=%u reason=wave32-ambiguous-mask-read\n",
                                     in.pc);
                    return false;
                }

                bool writes_b32_mask = false;
                if (in.fmt == Rdna2Format::SOP1 &&
                    (in.opcode == 0x03 || in.opcode == 0x09 ||
                     in.opcode == 0x3c || in.opcode == 0x40 || in.opcode == 0x44)) {
                    const bool source_mask = in.src[0].value == 126 ||
                        ((in.src[0].kind == OperandKind::SGPR ||
                          in.src[0].kind == OperandKind::Special) &&
                         (masks.contains(in.src[0].value) ||
                          static_b64_mask_keys.contains(in.src[0].value))) ||
                        (in.dst.value == 126 && in.src[0].kind == OperandKind::InlineInt);
                    writes_b32_mask = source_mask && in.dst.value != 127 &&
                        ((in.opcode != 0x3c && in.opcode != 0x40 && in.opcode != 0x44) ||
                         in.dst.value != 126);
                }
                if (in.fmt == Rdna2Format::SOP2 &&
                    (in.opcode == 0x0a || (in.opcode >= 0x0e &&
                                           in.opcode <= 0x1c && (in.opcode & 1u) == 0))) {
                    auto source_mask = [&](const Operand& source) {
                        return source.value == 126 || masks.contains(source.value) ||
                            (source.kind == OperandKind::SGPR &&
                             static_b64_mask_keys.contains(source.value)) ||
                            source.kind == OperandKind::InlineInt;
                    };
                    const bool mask_domain = in.dst.value == 126 || in.src[0].value == 126 ||
                        in.src[1].value == 126 ||
                        (((in.src[0].kind == OperandKind::SGPR ||
                           in.src[0].kind == OperandKind::Special) &&
                          masks.contains(in.src[0].value))) ||
                        (((in.src[1].kind == OperandKind::SGPR ||
                           in.src[1].kind == OperandKind::Special) &&
                          masks.contains(in.src[1].value)));
                    writes_b32_mask = mask_domain && source_mask(in.src[0]) &&
                        source_mask(in.src[1]) && in.dst.value != 127;
                }

                int b32_write_reg = writes_b32_mask ? in.dst.value : -1;
                if (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 &&
                    in.opcode <= 0x12a && in.sdst.kind == OperandKind::SGPR) {
                    const Operand& carry_in = in.src[2];
                    const bool source_mask =
                        ((carry_in.kind == OperandKind::SGPR ||
                          carry_in.kind == OperandKind::Special) &&
                         (masks.contains(carry_in.value) ||
                          static_b64_mask_keys.contains(carry_in.value)));
                    if (source_mask) b32_write_reg = in.sdst.value;
                }

                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    const bool one_word_write = base == b32_write_reg;
                    const uint32_t effective_width = one_word_write ? 1u : width;
                    for (uint32_t word = 0; word < effective_width; ++word) {
                        const int reg = base + static_cast<int>(word);
                        if (!one_word_write || reg != b32_write_reg) {
                            masks.erase(reg);
                            ambiguous.erase(reg);
                        }
                    }
                    if (one_word_write && b32_write_reg != 126) {
                        masks.insert(b32_write_reg);
                        ambiguous.erase(b32_write_reg);
                    }
                }, proven_wave32_masks);

                // Every non-CMPX VOPC destination is a one-word mask in proven Wave32, including
                // explicit ordinary SGPRs. CMPX writes EXEC only and establishes no VCC lifetime.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
                    const int destination = in.dst.kind == OperandKind::SGPR ? in.dst.value : 106;
                    masks.insert(destination);
                    ambiguous.erase(destination);
                }
            }

            for (uint32_t successor : successors[block]) {
                if (!b32_mask_reachable[successor]) {
                    b32_mask_reachable[successor] = true;
                    b32_mask_in[successor] = masks;
                    b32_mask_ambiguous_in[successor] = ambiguous;
                    pending.push_back(successor);
                } else {
                    std::set<int> joined_masks;
                    std::set<int> joined_ambiguous = b32_mask_ambiguous_in[successor];
                    std::set_intersection(
                        b32_mask_in[successor].begin(), b32_mask_in[successor].end(),
                        masks.begin(), masks.end(),
                        std::inserter(joined_masks, joined_masks.end()));
                    for (int reg : b32_mask_in[successor])
                        if (!masks.contains(reg)) joined_ambiguous.insert(reg);
                    for (int reg : masks)
                        if (!b32_mask_in[successor].contains(reg)) joined_ambiguous.insert(reg);
                    joined_ambiguous.insert(ambiguous.begin(), ambiguous.end());
                    for (int reg : joined_ambiguous) joined_masks.erase(reg);
                    if (joined_masks != b32_mask_in[successor] ||
                        joined_ambiguous != b32_mask_ambiguous_in[successor]) {
                        b32_mask_in[successor] = std::move(joined_masks);
                        b32_mask_ambiguous_in[successor] = std::move(joined_ambiguous);
                        pending.push_back(successor);
                    }
                }
            }
        }
        for (const auto& masks : b32_mask_in)
            for (int reg : masks)
                if (reg <= 107) static_mask_keys.insert(reg);
    }

    std::vector<std::unordered_set<int>> scalar_may_write_in(starts.size());
    std::vector<bool> scalar_reachable(starts.size(), false);
    if (!scalar_may_write_in.empty()) {
        scalar_may_write_in.front() = initial.sreg_written;
        scalar_reachable.front() = true;
    }
    bool provenance_changed = true;
    while (provenance_changed) {
        provenance_changed = false;
        for (uint32_t block = 0; block < starts.size(); ++block) {
            if (!scalar_reachable[block]) continue;
            std::unordered_set<int> out = scalar_may_write_in[block];
            out.insert(scalar_writes[block].begin(), scalar_writes[block].end());
            for (uint32_t successor : successors[block]) {
                if (!scalar_reachable[successor]) {
                    scalar_reachable[successor] = true;
                    provenance_changed = true;
                }
                const size_t before = scalar_may_write_in[successor].size();
                scalar_may_write_in[successor].insert(out.begin(), out.end());
                provenance_changed |= scalar_may_write_in[successor].size() != before;
            }
        }
    }

    // The native exact-wave path does not need to return to a synchronized common phase after a
    // plain unconditional edge.  Fuse maximal forward chains whose successor has no other
    // predecessor; the successor cannot be entered from outside the chain, so keeping its register
    // state in SSA is equivalent to a dispatcher save/reload without duplicating guest code.  Keep
    // backward edges as dispatcher iterations (they form loops), and end a chain at every wave op
    // that must execute as one uniform switch case.
    std::vector<std::vector<uint32_t>> dispatch_blocks;
    std::vector<uint32_t> dispatch_for_block(starts.size(), UINT32_MAX);
    std::vector<uint32_t> predecessor_count(starts.size(), 0);
    for (const auto& edges : successors)
        for (uint32_t successor : edges)
            if (successor < predecessor_count.size()) ++predecessor_count[successor];
    std::vector<bool> synchronized_block(starts.size(), false);
    std::vector<bool> conditional_block(starts.size(), false);
    for (uint32_t block = 0; block < starts.size(); ++block) {
        const uint32_t lo = starts[block];
        const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
        for (const auto& in : ins) {
            if (in.pc < lo || in.pc >= hi) continue;
            synchronized_block[block] = synchronized_block[block] ||
                mbcnt_event_for_pc.contains(in.pc) || append_event_for_pc.contains(in.pc) ||
                swizzle_pcs.contains(in.pc) ||
                mask_zero_compare_candidate_source(in) >= 0;
            conditional_block[block] = conditional_block[block] ||
                (!linearized_branch(in) && in.fmt == Rdna2Format::SOPP &&
                 in.opcode >= 0x04 && in.opcode <= 0x09 && in.opcode != 0x03);
        }
    }
    for (uint32_t first = 0; first < starts.size(); ++first) {
        if (dispatch_for_block[first] != UINT32_MAX) continue;
        const uint32_t dispatch = static_cast<uint32_t>(dispatch_blocks.size());
        dispatch_blocks.push_back({});
        uint32_t block = first;
        while (true) {
            dispatch_for_block[block] = dispatch;
            dispatch_blocks.back().push_back(block);
            if (!direct_dispatch || synchronized_block[block] || conditional_block[block] ||
                successors[block].size() != 1) break;
            const uint32_t successor = successors[block].front();
            if (successor <= block || predecessor_count[successor] != 1 ||
                dispatch_for_block[successor] != UINT32_MAX) break;
            block = successor;
        }
    }

    std::vector<std::set<int>> dispatch_vector_reads(dispatch_blocks.size());
    std::vector<std::set<int>> dispatch_vector_writes(dispatch_blocks.size());
    std::vector<std::unordered_set<int>> dispatch_scalar_writes(dispatch_blocks.size());
    for (uint32_t dispatch = 0; dispatch < dispatch_blocks.size(); ++dispatch) {
        for (uint32_t block : dispatch_blocks[dispatch]) {
            dispatch_vector_reads[dispatch].insert(
                vector_reads[block].begin(), vector_reads[block].end());
            dispatch_vector_writes[dispatch].insert(
                vector_writes[block].begin(), vector_writes[block].end());
            dispatch_scalar_writes[dispatch].insert(
                scalar_writes[block].begin(), scalar_writes[block].end());
        }
    }

    // Saved mask pairs and scalar-spill lane slots have their own value domains.
    std::set<int> mask_keys = static_mask_keys;
    std::set<std::pair<int, int>> lane_slots, mask_lane_slots;
    // A lane spill can precede another static definition of its mask in a back-edge block, hence the
    // complete discovery pass above occurs before classifying any spill slots here.
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361 &&
            in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 && in.src[1].value <= 63) {
            const std::pair<int, int> slot{in.dst.value, in.src[1].value};
            const bool is_mask = in.src[0].value == 106 || in.src[0].value == 107 ||
                                 in.src[0].value == 126 || in.src[0].value == 127 ||
                                 (in.src[0].kind == OperandKind::SGPR && mask_keys.count(in.src[0].value));
            (is_mask ? mask_lane_slots : lane_slots).insert(slot);
        }
    }
    for (const auto& vg : initial.vgpr_lane_slots)
        for (const auto& slot : vg.second) lane_slots.emplace(vg.first, slot.first);
    for (const auto& vg : initial.vgpr_lane_mask_slots)
        for (const auto& slot : vg.second) mask_lane_slots.emplace(vg.first, slot.first);
    // A physical spill lane may be recycled between scalar-data and wave-mask lifetimes. Persist
    // both domains across dispatcher blocks; v_readlane retains both destination views when both
    // are present, and the statically typed consumer selects the representation it needs.

    // Map presence is the scalar-spill validity bit in RegState. Function variables cannot encode
    // that compile-time type state: saving an erased slot as zero and reconstructing it in the next
    // dispatcher case would turn an invalid lifetime into a valid scalar zero. Conservatively reject
    // any CFG path on which an ordinary VGPR write can invalidate a spill array before v_readlane;
    // a later v_writelane starts a fresh lifetime and clears the tombstone, matching emit_alu.
    std::set<int> spill_vgprs;
    for (const auto& slot : lane_slots) spill_vgprs.insert(slot.first);
    for (const auto& slot : mask_lane_slots) spill_vgprs.insert(slot.first);
    if (!spill_vgprs.empty()) {
        std::vector<std::set<int>> invalidated_in(starts.size());
        std::vector<bool> invalidated_reachable(starts.size(), false);
        invalidated_in.front().insert(initial.invalidated_vgpr_lane_slots.begin(),
                                      initial.invalidated_vgpr_lane_slots.end());
        invalidated_reachable.front() = true;
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> invalidated = invalidated_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;
                if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
                    invalidated.erase(in.dst.value);
                    continue;
                }
                if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 &&
                    invalidated.contains(in.src[0].value))
                    return reject_cfg(in.pc, "invalidated-vgpr-lane-slot");
                const uint32_t writes = vgpr_write_count(in);
                for (uint32_t word = 0; word < writes; ++word) {
                    const int reg = in.dst.value + static_cast<int>(word);
                    if (spill_vgprs.contains(reg)) invalidated.insert(reg);
                }
            }
            for (uint32_t successor : successors[block]) {
                if (!invalidated_reachable[successor]) {
                    invalidated_reachable[successor] = true;
                    invalidated_in[successor] = invalidated;
                    pending.push_back(successor);
                    continue;
                }
                const size_t before = invalidated_in[successor].size();
                invalidated_in[successor].insert(invalidated.begin(), invalidated.end());
                if (invalidated_in[successor].size() != before) pending.push_back(successor);
            }
        }
    }

    uint32_t ptr_u32 = 0, ptr_bool = 0;
    std::map<int, uint32_t> vv, sv, mv;
    std::map<std::pair<int, int>, uint32_t> lv, lmv;
    for (int r : vregs) vv[r] = b.function_var(b.t_u32, ptr_u32);
    for (int r : sregs) sv[r] = b.function_var(b.t_u32, ptr_u32);
    for (int r : mask_keys) mv[r] = b.function_var(b.t_bool, ptr_bool);
    for (const auto& slot : lane_slots) lv[slot] = b.function_var(b.t_u32, ptr_u32);
    for (const auto& slot : mask_lane_slots) lmv[slot] = b.function_var(b.t_bool, ptr_bool);
    const uint32_t scc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vcc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t exec_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t pc_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t active_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_value_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_invert_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_to_scc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_taken_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t vote_next_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_mask_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_low_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_write_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_event_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_acc_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_dst_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_sum_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_active_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_event_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_consume_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_gds_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_idx_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_dst_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_count_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t swizzle_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t swizzle_active_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t swizzle_source_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t swizzle_source_lane_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t swizzle_dst_var = b.function_var(b.t_u32, ptr_u32);

    const uint32_t zero = b.uconst(0), no = b.bfalse(), yes = b.btrue();
    for (const auto& kv : vv) {
        auto it = initial.vreg.find(kv.first);
        b.store_function(kv.second, it == initial.vreg.end() ? zero : it->second);
    }
    for (const auto& kv : sv) {
        auto it = initial.sreg.find(kv.first);
        b.store_function(kv.second, it == initial.sreg.end() ? zero : it->second);
    }
    for (const auto& kv : mv) {
        auto it = initial.sreg_bool.find(kv.first);
        b.store_function(kv.second, it == initial.sreg_bool.end() ? no : it->second);
    }
    for (const auto& kv : lv) {
        uint32_t value = zero;
        auto vg = initial.vgpr_lane_slots.find(kv.first.first);
        if (vg != initial.vgpr_lane_slots.end()) {
            auto slot = vg->second.find(kv.first.second);
            if (slot != vg->second.end()) value = slot->second;
        }
        b.store_function(kv.second, value);
    }
    for (const auto& kv : lmv) {
        uint32_t value = no;
        auto vg = initial.vgpr_lane_mask_slots.find(kv.first.first);
        if (vg != initial.vgpr_lane_mask_slots.end()) {
            auto slot = vg->second.find(kv.first.second);
            if (slot != vg->second.end()) value = slot->second;
        }
        b.store_function(kv.second, value);
    }
    b.store_function(scc_var, initial.scc);
    b.store_function(vcc_var, initial.vcc);
    b.store_function(exec_var, initial.exec);
    b.store_function(pc_var, b.uconst(0));
    b.store_function(active_var, yes);
    b.store_function(swizzle_source_var, zero);
    b.store_function(swizzle_source_lane_var, zero);
    b.store_function(swizzle_dst_var, zero);

    auto load_state = [&](uint32_t dispatch = UINT32_MAX) {
        RegState state;
        state.sreg_input = initial.sreg_input;
        for (const auto& kv : vv) {
            if (dispatch != UINT32_MAX &&
                !dispatch_vector_reads[dispatch].contains(kv.first)) continue;
            state.vreg[kv.first] = b.load_function(b.t_u32, kv.second);
        }
        for (const auto& kv : sv) state.sreg[kv.first] = b.load_function(b.t_u32, kv.second);
        const std::set<int>* entry_b32 = nullptr;
        if (dispatch != UINT32_MAX && b.allow_b32_masks) {
            const uint32_t entry_block = dispatch_blocks[dispatch].front();
            if (b32_mask_reachable[entry_block]) entry_b32 = &b32_mask_in[entry_block];
        }
        for (const auto& kv : mv) {
            if (!static_b64_mask_keys.contains(kv.first) &&
                (!entry_b32 || !entry_b32->contains(kv.first)))
                continue;
            state.sreg_bool[kv.first] = b.load_function(b.t_bool, kv.second);
            state.sreg_bool_narrowed[kv.first] = true;
        }
        for (const auto& kv : lv)
            state.vgpr_lane_slots[kv.first.first][kv.first.second] =
                b.load_function(b.t_u32, kv.second);
        for (const auto& kv : lmv)
            state.vgpr_lane_mask_slots[kv.first.first][kv.first.second] =
                b.load_function(b.t_bool, kv.second);
        state.scc = b.load_function(b.t_bool, scc_var);
        state.vcc = b.load_function(b.t_bool, vcc_var);
        state.exec = b.load_function(b.t_bool, exec_var);
        if (entry_b32) {
            for (int reg : *entry_b32) {
                state.sreg.erase(reg);
                state.sreg_input.erase(reg);
                state.sreg_bool_b32.insert(reg);
                if (reg == 106) {
                    state.sreg_bool[106] = state.vcc;
                    state.sreg_bool_narrowed[106] = true;
                }
            }
        }
        state.exec_narrowed = true; // state-machine joins may carry a narrowed EXEC edge
        state.mubuf_pcrel_tables = initial.mubuf_pcrel_tables;
        state.smem_pcrel_tables = initial.smem_pcrel_tables;
        return state;
    };
    auto save_state = [&](const RegState& state, uint32_t dispatch) {
        // A dispatcher case starts from the persistent register file, executes exactly one guest
        // basic block, then returns to the common loop.  Only registers WRITTEN by that block need
        // to be copied back.  Saving every tracked VGPR/SGPR at every edge generated thousands of
        // redundant Function-memory stores for large kernels (Astro Bot's title compute shader has
        // around one hundred tracked registers and dozens of cases), creating severe register
        // pressure and scratch traffic in the host driver.  The conservative static writer sets
        // include multi-register loads and secondary scalar destinations; unchanged variables keep
        // the value loaded at the preceding dispatcher edge.
        for (const auto& kv : vv) {
            if (!dispatch_vector_writes[dispatch].contains(kv.first)) continue;
            auto it = state.vreg.find(kv.first);
            b.store_function(kv.second, it == state.vreg.end() ? zero : it->second);
        }
        for (const auto& kv : sv) {
            if (!dispatch_scalar_writes[dispatch].contains(kv.first)) continue;
            auto it = state.sreg.find(kv.first);
            b.store_function(kv.second, it == state.sreg.end() ? zero : it->second);
        }
        for (const auto& kv : mv) {
            if (!dispatch_scalar_writes[dispatch].contains(kv.first)) continue;
            auto it = state.sreg_bool.find(kv.first);
            b.store_function(kv.second, it == state.sreg_bool.end() ? no : it->second);
        }
        for (const auto& kv : lv) {
            uint32_t value = zero;
            auto vg = state.vgpr_lane_slots.find(kv.first.first);
            if (vg != state.vgpr_lane_slots.end()) {
                auto slot = vg->second.find(kv.first.second);
                if (slot != vg->second.end()) value = slot->second;
            }
            b.store_function(kv.second, value);
        }
        for (const auto& kv : lmv) {
            uint32_t value = no;
            auto vg = state.vgpr_lane_mask_slots.find(kv.first.first);
            if (vg != state.vgpr_lane_mask_slots.end()) {
                auto slot = vg->second.find(kv.first.second);
                if (slot != vg->second.end()) value = slot->second;
            }
            b.store_function(kv.second, value);
        }
        // A poisoned (0) SCC degrades to bfalse at the block boundary: the same-block consumers
        // reject on the sentinel; cross-block staleness matches the pre-poison model.
        b.store_function(scc_var, state.scc ? state.scc : b.bfalse());
        b.store_function(vcc_var, state.vcc);
        b.store_function(exec_var, state.exec);
    };

    const uint32_t loop_header = b.id(), switch_header = b.id(), switch_merge = b.id();
    const uint32_t loop_continue = b.id(), loop_merge = b.id(), fallback = b.id();
    std::vector<uint32_t> labels(dispatch_blocks.size());
    std::vector<std::pair<uint32_t, uint32_t>> switch_cases;
    for (uint32_t i = 0; i < dispatch_blocks.size(); ++i) {
        labels[i] = b.id();
        switch_cases.emplace_back(i, labels[i]);
    }
    b.emit_branch(loop_header);
    b.emit_label(loop_header);
    // Every live hardware wave executes one guest basic block per dispatcher iteration. Inactive
    // invocations remain in the loop as synchronization participants until all waves finish.
    // With an exact native subgroup, one host subgroup is one guest wave and its scalar PC is
    // uniform.  Cross-lane operations can therefore execute directly in their switch case.  The
    // old publish/merge phase remains necessary only for the portable workgroup-scratch fallback;
    // resetting all of its mailboxes on every native dispatcher iteration was a surprisingly large
    // SALU/function-memory tax for branch-heavy kernels.
    if (!direct_dispatch) {
        b.store_function(vote_pending_var, no);
        b.store_function(vote_value_var, no);
        b.store_function(vote_invert_var, no);
        b.store_function(vote_to_scc_var, no);
        b.store_function(vote_taken_var, zero);
        b.store_function(vote_next_var, zero);
        b.store_function(mbcnt_pending_var, no);
        b.store_function(mbcnt_mask_var, no);
        b.store_function(mbcnt_low_var, no);
        b.store_function(mbcnt_write_var, no);
        b.store_function(mbcnt_event_var, zero);
        b.store_function(mbcnt_acc_var, zero);
        b.store_function(mbcnt_dst_var, zero);
        b.store_function(append_pending_var, no);
        b.store_function(append_active_var, no);
        b.store_function(append_event_var, zero);
        b.store_function(append_consume_var, no);
        b.store_function(append_gds_var, no);
        b.store_function(append_idx_var, zero);
        b.store_function(append_dst_var, zero);
    }
    b.store_function(swizzle_pending_var, no);
    b.store_function(swizzle_active_var, no);
    b.emit_loopmerge(loop_merge, loop_continue);
    b.emit_branch(switch_header);
    b.emit_label(switch_header);
    const uint32_t selector = b.sel(b.load_function(b.t_bool, active_var),
                                    b.load_function(b.t_u32, pc_var), b.uconst(UINT32_MAX));
    b.emit_selmerge(switch_merge);
    b.emit_switch(selector, fallback, switch_cases);

    auto set_next = [&](uint32_t pc) {
        auto found = block_for_pc.find(pc);
        if (pc > end_pc) {
            if (!proven_exit_target(pc)) return false;
            b.store_function(active_var, no);
        }
        else if (found == block_for_pc.end()) {
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[compute-cfg-successor-reject] pc=%u end=%u blocks=%zu\n",
                             pc, end_pc, block_for_pc.size());
            if (getenv("PROSPER_DBG")) {
                std::fprintf(stderr, "[compute-cfg-successors]");
                for (const auto& entry : block_for_pc)
                    std::fprintf(stderr, " %u", entry.first);
                std::fprintf(stderr, "\n");
            }
            return false;
        }
        else b.store_function(pc_var, b.uconst(dispatch_for_block[found->second]));
        return true;
    };
    for (uint32_t dispatch = 0; dispatch < dispatch_blocks.size(); ++dispatch) {
        const uint32_t entry_block = dispatch_blocks[dispatch].front();
        b.emit_label(labels[dispatch]);
        RegState state = load_state(dispatch);
        state.sreg_written = scalar_may_write_in[entry_block];
        for (int reg : state.sreg_written) state.sreg_input.erase(reg);
        if (!direct_descriptor_sregs.empty()) {
            for (int reg : direct_descriptor_sregs)
                if (!state.sreg_written.count(reg)) state.sreg.erase(reg);
        }
        const Rdna2Inst* terminator = nullptr;
        const Rdna2Inst* mbcnt = nullptr;
        const Rdna2Inst* append = nullptr;
        const Rdna2Inst* swizzle = nullptr;
        const Rdna2Inst* mask_compare = nullptr;
        const uint32_t final_block = dispatch_blocks[dispatch].back();
        const uint32_t final_hi = final_block + 1 < starts.size()
            ? starts[final_block + 1] : UINT32_MAX;
        for (size_t member = 0; member < dispatch_blocks[dispatch].size(); ++member) {
            const uint32_t block = dispatch_blocks[dispatch][member];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            const Rdna2Inst* block_terminator = nullptr;
            const Rdna2Inst* block_mbcnt = nullptr;
            const Rdna2Inst* block_append = nullptr;
            const Rdna2Inst* block_swizzle = nullptr;
            const Rdna2Inst* block_mask_compare = nullptr;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi) continue;
                if (cfg_terminator(in)) {
                    block_terminator = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::VOP3 &&
                    (in.opcode == 0x365 || in.opcode == 0x366)) {
                    block_mbcnt = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::DS && (in.opcode == 0x3d || in.opcode == 0x3e)) {
                    block_append = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::DS && in.opcode == 0x35) {
                    block_swizzle = &in;
                    break;
                }
                const int mask_compare_source =
                    mask_zero_compare_candidate_source(in);
                if (mask_compare_source >= 0 &&
                    state.sreg_bool.contains(mask_compare_source)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-mask-compare] pc=%u source=s%d op=0x%x\n",
                                     in.pc, mask_compare_source, in.opcode);
                    block_mask_compare = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::EXP) {
                    if (!exp_fn(state, in)) return reject_cfg(in.pc, "export");
                    continue;
                }
                bool ok = true;
                const bool handled = emit_alu(b, state, in, ok, allow_exec_update, &safe,
                                              allow_smem, rt, /*allow_wave*/false);
                if (handled && ok) record_scalar_write(state, in);
                if (!handled || !ok) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr, "[cfg-recompile-reject] pc=%u fmt=%d op=0x%x\n",
                                     in.pc, static_cast<int>(in.fmt), in.opcode);
                    return false;
                }
            }
            const bool last = member + 1 == dispatch_blocks[dispatch].size();
            if (!last) {
                // Group construction admits only one-successor plain blocks before the tail.
                if (block_mbcnt || block_append || block_swizzle || block_mask_compare ||
                    (block_terminator && (block_terminator->is_end ||
                                          block_terminator->opcode != 0x02)))
                    return reject_cfg(starts[block], "invalid-fused-block");
                continue; // consume an unconditional guest branch without a dispatcher round-trip
            }
            terminator = block_terminator;
            mbcnt = block_mbcnt;
            append = block_append;
            swizzle = block_swizzle;
            mask_compare = block_mask_compare;
        }
        if (mbcnt) {
            if (graphics) {
                if (getenv("PROSPER_DBG"))
                    std::fprintf(stderr,
                                 "[graphics-cfg-reject] pc=%u reason=mbcnt-cross-lane\n",
                                 mbcnt->pc);
                return false;
            }
            bool operand_ok = true;
            const uint32_t mask = mbcnt_source_bit(
                b, state, mbcnt->src[0], mbcnt->opcode == 0x366);
            const uint32_t acc = operand_bits(b, state, *mbcnt, mbcnt->src[1], &operand_ok);
            const auto event = mbcnt_event_for_pc.find(mbcnt->pc);
            if (!mask || !operand_ok || event == mbcnt_event_for_pc.end()) return false;
            if (b.native_subgroup_size) {
                // The switch selector is scalar within this exact-size subgroup, so every guest
                // lane reaches the same case.  Execute the wave prefix count here and retain masked-
                // off lanes exactly as an RDNA VGPR write does, instead of dynamically selecting a
                // destination across the entire persistent register file in the common phase.
                const uint32_t result = b.native_compute_mbcnt(
                    mask, acc, mbcnt->opcode == 0x365 ? yes : no);
                const int dst = mbcnt->dst.value;
                const auto old = state.vreg.find(dst);
                state.vreg[dst] = b.sel(
                    state.exec, result, old == state.vreg.end() ? zero : old->second);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(mbcnt_pending_var, yes);
                b.store_function(mbcnt_mask_var, mask);
                b.store_function(mbcnt_low_var, mbcnt->opcode == 0x365 ? yes : no);
                // load_state conservatively marks EXEC narrowed. Writing under the current per-lane EXEC
                // is equivalent for a known-full mask and preserves inactive VGPR lanes for divergent code.
                b.store_function(mbcnt_write_var, state.exec);
                b.store_function(mbcnt_event_var, b.uconst(event->second));
                b.store_function(mbcnt_acc_var, acc);
                b.store_function(mbcnt_dst_var, b.uconst(static_cast<uint32_t>(mbcnt->dst.value)));
            }
        }
        if (append) {
            if (graphics) {
                if (getenv("PROSPER_DBG"))
                    std::fprintf(stderr,
                                 "[graphics-cfg-reject] pc=%u reason=gds-cross-lane\n",
                                 append->pc);
                return false;
            }
            const auto m0 = state.sreg.find(124);
            const auto event = append_event_for_pc.find(append->pc);
            if (m0 == state.sreg.end() || event == append_event_for_pc.end()) return false;
            if (!append->ds_gds) b.declare_lds();
            const uint32_t base = b.ibin(Op_BitwiseAnd, m0->second, b.uconst(0xffffu));
            const uint32_t byte_addr = b.ibin(Op_IAdd, base, b.uconst(append->literal));
            const uint32_t idx = b.ibin(Op_ShiftRightLogical, byte_addr, b.uconst(2));
            if (b.native_subgroup_size) {
                const uint32_t result = append->ds_gds
                    ? b.native_gds_append(idx, state.exec, append->opcode == 0x3d)
                    : b.native_wave_append(
                          idx, state.exec, append->opcode == 0x3d ? yes : no);
                const int dst = append->dst.value;
                const auto old = state.vreg.find(dst);
                state.vreg[dst] = b.sel(
                    state.exec, result, old == state.vreg.end() ? zero : old->second);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(append_pending_var, yes);
                b.store_function(append_active_var, state.exec);
                b.store_function(append_event_var, b.uconst(event->second));
                b.store_function(append_consume_var, append->opcode == 0x3d ? yes : no);
                b.store_function(append_gds_var, append->ds_gds ? yes : no);
                b.store_function(append_idx_var, idx);
                b.store_function(append_dst_var,
                    b.uconst(static_cast<uint32_t>(append->dst.value)));
            }
        }
        if (swizzle) {
            if (!swizzle_pcs.contains(swizzle->pc)) return false;
            uint32_t source_lane = 0;
            if (!b.ds_swizzle_source_lane(swizzle->literal, &source_lane)) return false;
            const auto source = state.vreg.find(swizzle->src[0].value);
            b.store_function(swizzle_pending_var, yes);
            b.store_function(swizzle_active_var, state.exec);
            b.store_function(swizzle_source_var,
                source == state.vreg.end() ? zero : source->second);
            b.store_function(swizzle_source_lane_var, source_lane);
            b.store_function(swizzle_dst_var,
                b.uconst(static_cast<uint32_t>(swizzle->dst.value)));
        }
        if (mask_compare) {
            if (b.is_vertex) {
                if (getenv("PROSPER_DBG"))
                    std::fprintf(stderr,
                                 "[graphics-cfg-reject] pc=%u reason=wave-mask-compare\n",
                                 mask_compare->pc);
                return false;
            }
            const int source = mask_zero_compare_candidate_source(*mask_compare);
            const auto value = state.sreg_bool.find(source);
            if (value == state.sreg_bool.end())
                return reject_cfg(mask_compare->pc, "missing-mask-compare-source");
            if (b.native_subgroup_size || b.is_fragment) {
                const uint32_t wave_any = b.is_fragment
                    ? b.fragment_wave_any(value->second)
                    : b.native_wave_any(value->second);
                if (!wave_any) return reject_cfg(mask_compare->pc, "mask-vote");
                state.scc = mask_zero_compare_inverts(*mask_compare)
                    ? b.logical_not(wave_any) : wave_any;
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var, value->second);
                b.store_function(vote_invert_var,
                    mask_zero_compare_inverts(*mask_compare) ? yes : no);
                b.store_function(vote_to_scc_var, yes);
            }
        }
        save_state(state, dispatch);
        if (mbcnt) {
            if (!set_next(mbcnt->pc + mbcnt->len_dwords))
                return reject_cfg(mbcnt->pc, "mbcnt-successor");
        } else if (append) {
            if (!set_next(append->pc + append->len_dwords))
                return reject_cfg(append->pc, "append-successor");
        } else if (swizzle) {
            if (!set_next(swizzle->pc + swizzle->len_dwords))
                return reject_cfg(swizzle->pc, "swizzle-successor");
        } else if (mask_compare) {
            if (!set_next(mask_compare->pc + mask_compare->len_dwords))
                return reject_cfg(mask_compare->pc, "mask-compare-successor");
        } else if (!terminator) {
            if (!set_next(final_hi)) return reject_cfg(starts[final_block], "fallthrough-successor");
        } else if (terminator->is_end || terminator->opcode == 0x12) {
            // Vulkan has no compute-stage trap instruction. Ending this dispatcher invocation is
            // the closest fail-closed model of s_trap when no guest trap handler is emulated; the
            // overwhelmingly common release-shader shape branches around the trap on valid data.
            b.store_function(active_var, no);
        } else if (terminator->opcode == 0x02) {
            if (!set_next(branch_target(*terminator)))
                return reject_cfg(terminator->pc, "branch-successor");
        } else {
            uint32_t condition = 0;
            switch (terminator->opcode) {
                // state.scc == 0 marks SCC poisoned by a 64-bit mask op inside this block (its
                // hardware SCC is a cross-lane reduction) — reject rather than branch on a stale
                // value an older s_cmp produced.
                case 0x04: if (!state.scc) return reject_cfg(terminator->pc, "poisoned-scc");
                           condition = b.logical_not(state.scc); break; // s_cbranch_scc0
                case 0x05: if (!state.scc) return reject_cfg(terminator->pc, "poisoned-scc");
                           condition = state.scc; break;
                case 0x06: case 0x07: case 0x08: case 0x09: break;
                default: return reject_cfg(terminator->pc, "branch-opcode");
            }
            const uint32_t target = branch_target(*terminator);
            const uint32_t fallthrough = terminator->pc + terminator->len_dwords;
            auto taken = block_for_pc.find(target), next = block_for_pc.find(fallthrough);
            const bool taken_exit = target > end_pc && proven_exit_target(target);
            const bool next_exit = fallthrough > end_pc && proven_exit_target(fallthrough);
            if ((!taken_exit && taken == block_for_pc.end()) ||
                (!next_exit && next == block_for_pc.end()))
                return reject_cfg(terminator->pc, "branch-target");
            if ((taken_exit || next_exit) && b.is_compute && !b.native_subgroup_size)
                return reject_cfg(terminator->pc, "portable-compute-exit");
            const uint32_t taken_dispatch = taken_exit ? 0 : dispatch_for_block[taken->second];
            const uint32_t next_dispatch = next_exit ? 0 : dispatch_for_block[next->second];
            auto route = [&](uint32_t branch_condition) {
                if (taken_exit || next_exit) {
                    const uint32_t remains_active = taken_exit
                        ? b.logical_not(branch_condition) : branch_condition;
                    b.store_function(active_var, remains_active);
                    b.store_function(pc_var, b.uconst(
                        taken_exit ? next_dispatch : taken_dispatch));
                } else {
                    b.store_function(pc_var,
                        b.sel(branch_condition, b.uconst(taken_dispatch),
                              b.uconst(next_dispatch)));
                }
            };
            if (terminator->opcode == 0x04 || terminator->opcode == 0x05) {
                route(condition);
            } else if (graphics) {
                const uint32_t lane_condition =
                    terminator->opcode <= 0x07 ? state.vcc : state.exec;
                const uint32_t branch_condition =
                    terminator->opcode == 0x06 || terminator->opcode == 0x08
                        ? b.logical_not(lane_condition) : lane_condition;
                route(branch_condition);
            } else if (b.native_subgroup_size) {
                const uint32_t wave_any = b.native_wave_any(
                    terminator->opcode <= 0x07 ? state.vcc : state.exec);
                const uint32_t wave_condition =
                    terminator->opcode == 0x06 || terminator->opcode == 0x08
                        ? b.logical_not(wave_any) : wave_any;
                route(wave_condition);
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var,
                    terminator->opcode <= 0x07 ? state.vcc : state.exec);
                b.store_function(vote_invert_var,
                    terminator->opcode == 0x06 || terminator->opcode == 0x08 ? yes : no);
                b.store_function(vote_taken_var, b.uconst(taken_dispatch));
                b.store_function(vote_next_var, b.uconst(next_dispatch));
            }
        }
        b.emit_branch(switch_merge);
    }
    b.emit_label(fallback);
    b.store_function(active_var, no);
    b.emit_branch(switch_merge);
    b.emit_label(switch_merge);
    b.emit_branch(loop_continue);
    b.emit_label(loop_continue);

    // DS_SWIZZLE common phase. The dispatcher cases only publish source data and a lane selector;
    // every invocation executes the actual subgroup gathers here in uniform control flow. This is
    // required even though the instruction does not touch LDS: subgroup operations in a divergent
    // switch arm would have undefined participation on Vulkan.
    if (!swizzle_pcs.empty()) {
        const uint32_t swizzle_pending = b.load_function(b.t_bool, swizzle_pending_var);
        const uint32_t swizzle_active = b.load_function(b.t_bool, swizzle_active_var);
        const uint32_t swizzle_lane = b.load_function(b.t_u32, swizzle_source_lane_var);
        const uint32_t swizzle_value = b.subgroup_shuffle(
            b.load_function(b.t_u32, swizzle_source_var), swizzle_lane);
        const uint32_t source_active_word = b.sel(
            b.land(swizzle_pending, swizzle_active), b.uconst(1), zero);
        const uint32_t source_active = b.ucmp(
            Op_INotEqual, b.subgroup_shuffle(source_active_word, swizzle_lane), zero);
        const uint32_t swizzle_result = b.sel(source_active, swizzle_value, zero);
        const uint32_t swizzle_dst = b.load_function(b.t_u32, swizzle_dst_var);
        const uint32_t swizzle_write = b.land(swizzle_pending, swizzle_active);
        for (const auto& kv : vv) {
            const uint32_t selected = b.land(
                swizzle_write,
                b.ucmp(Op_IEqual, swizzle_dst,
                       b.uconst(static_cast<uint32_t>(kv.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, swizzle_result, old));
        }
        // An ordinary VGPR definition ends a scalar lane-spill lifetime at that physical register.
        for (const auto& kv : lv) {
            const uint32_t selected = b.land(
                swizzle_pending,
                b.ucmp(Op_IEqual, swizzle_dst,
                       b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, zero, old));
        }
        for (const auto& kv : lmv) {
            const uint32_t selected = b.land(
                swizzle_pending,
                b.ucmp(Op_IEqual, swizzle_dst,
                       b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_bool, kv.second);
            b.store_function(kv.second, b.bsel(selected, no, old));
        }
    }

    if (direct_dispatch) {
        // Native wave operations and votes were already resolved in the selected case.  PC and
        // ACTIVE are scalar guest-wave state, so every invocation in this exact-size subgroup has
        // the same value; another subgroup in the workgroup may still leave independently because
        // this path contains no workgroup barriers.
        b.emit_condbranch(b.load_function(b.t_bool, active_var),
                          loop_header, loop_merge);
    } else {
    // MBCNT common phase. Cases publish an event-tagged mask bit and accumulator, but never emit a
    // barrier themselves. Every invocation—including ended waves and lanes currently at a different
    // guest block—therefore executes these two barriers in identical structured control flow. Event
    // tags keep contributions from different static MBCNT sites isolated if malformed/nonuniform
    // scalar state ever lets waves reach different sites during the same dispatcher iteration.
    const uint32_t mbcnt_wave_index = b.ibin(
        Op_ShiftRightLogical, b.linear_localid,
        b.uconst(b.wave_size == 32 ? 5u : 6u));
    const uint32_t mbcnt_lane = b.ibin(
        Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1));

    if (!mbcnt_event_for_pc.empty()) {
    const uint32_t mbcnt_pending = b.load_function(b.t_bool, mbcnt_pending_var);
    const uint32_t mbcnt_mask = b.load_function(b.t_bool, mbcnt_mask_var);
    const uint32_t mbcnt_tag = b.ibin(
        Op_IAdd, b.load_function(b.t_u32, mbcnt_event_var), b.uconst(1));
    const uint32_t mbcnt_encoded = b.sel(
        mbcnt_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, mbcnt_tag, b.uconst(1)),
               b.sel(mbcnt_mask, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(b.linear_localid, mbcnt_encoded);
    b.barrier();

    const uint32_t mbcnt_low = b.load_function(b.t_bool, mbcnt_low_var);
    b.store_function(mbcnt_sum_var, zero);
    const uint32_t mbcnt_scan = b.id(), mbcnt_scanned = b.id();
    b.emit_selmerge(mbcnt_scanned);
    b.emit_condbranch(mbcnt_pending, mbcnt_scan, mbcnt_scanned);
    b.emit_label(mbcnt_scan);
    uint32_t mbcnt_sum = zero;
    const uint32_t mbcnt_wave_base = b.ibin(
        Op_ShiftLeftLogical, mbcnt_wave_index,
        b.uconst(b.wave_size == 32 ? 5u : 6u));
    const uint32_t half_size = std::min(32u, b.wave_size);
    for (uint32_t i = 0; i < half_size; ++i) {
        const uint32_t candidate_lane = b.sel(
            mbcnt_low, b.uconst(i), b.uconst(i + (b.wave_size == 64 ? 32u : 0u)));
        const uint32_t candidate = b.cfg_scratch_load(
            b.ibin(Op_IAdd, mbcnt_wave_base, candidate_lane));
        const uint32_t candidate_tag = b.ibin(
            Op_ShiftRightLogical, candidate, b.uconst(1));
        const uint32_t candidate_bit = b.ibin(Op_BitwiseAnd, candidate, b.uconst(1));
        const uint32_t below = b.ucmp(
            Op_ULessThan, candidate_lane, mbcnt_lane);
        uint32_t include = below;
        if (b.wave_size == 32) include = b.land(include, mbcnt_low);
        include = b.land(include, b.ucmp(Op_IEqual, candidate_tag, mbcnt_tag));
        mbcnt_sum = b.b_iadd(mbcnt_sum, b.sel(include, candidate_bit, zero));
    }
    b.store_function(mbcnt_sum_var, mbcnt_sum);
    b.emit_branch(mbcnt_scanned);
    b.emit_label(mbcnt_scanned);
    const uint32_t mbcnt_result = b.b_iadd(
        b.load_function(b.t_u32, mbcnt_acc_var),
        b.load_function(b.t_u32, mbcnt_sum_var));
    const uint32_t mbcnt_dst = b.load_function(b.t_u32, mbcnt_dst_var);
    const uint32_t mbcnt_write = b.land(
        mbcnt_pending, b.load_function(b.t_bool, mbcnt_write_var));
    for (const auto& kv : vv) {
        const uint32_t selected = b.land(
            mbcnt_write, b.ucmp(Op_IEqual, mbcnt_dst, b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, mbcnt_result, old));
    }
    // An ordinary VGPR write ends any scalar-spill lifetime even for lanes masked off by EXEC.
    for (const auto& kv : lv) {
        const uint32_t selected = b.land(
            mbcnt_pending,
            b.ucmp(Op_IEqual, mbcnt_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        const uint32_t selected = b.land(
            mbcnt_pending,
            b.ucmp(Op_IEqual, mbcnt_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    if (!append_event_for_pc.empty()) {
    // DS_APPEND/DS_CONSUME common phase. Each wave reduces popcount(EXEC), its lane zero performs
    // exactly one LDS atomic add/subtract, and the pre-op value is broadcast through the result slot.
    // The event tag prevents a wave at one static append from counting another event's lanes.
    const uint32_t append_pending = b.load_function(b.t_bool, append_pending_var);
    const uint32_t append_active = b.load_function(b.t_bool, append_active_var);
    const uint32_t append_tag = b.ibin(
        Op_IAdd, b.load_function(b.t_u32, append_event_var), b.uconst(1));
    const uint32_t append_encoded = b.sel(
        append_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, append_tag, b.uconst(1)),
               b.sel(append_active, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(b.linear_localid, append_encoded);
    b.barrier();

    b.store_function(append_count_var, zero);
    const uint32_t append_scan = b.id(), append_scanned = b.id();
    b.emit_selmerge(append_scanned);
    b.emit_condbranch(append_pending, append_scan, append_scanned);
    b.emit_label(append_scan);
    uint32_t append_count = zero;
    const uint32_t append_wave_base = b.ibin(
        Op_ShiftLeftLogical, mbcnt_wave_index,
        b.uconst(b.wave_size == 32 ? 5u : 6u));
    for (uint32_t i = 0; i < b.wave_size; ++i) {
        const uint32_t candidate_index = b.ibin(
            Op_IAdd, append_wave_base, b.uconst(i));
        const uint32_t candidate = b.cfg_scratch_load(candidate_index);
        const uint32_t candidate_tag = b.ibin(
            Op_ShiftRightLogical, candidate, b.uconst(1));
        uint32_t include = b.ucmp(
            Op_ULessThan, candidate_index, b.uconst(b.local_count));
        include = b.land(include, b.ucmp(Op_IEqual, candidate_tag, append_tag));
        append_count = b.b_iadd(
            append_count,
            b.sel(include, b.ibin(Op_BitwiseAnd, candidate, b.uconst(1)), zero));
    }
    b.store_function(append_count_var, append_count);
    b.emit_branch(append_scanned);
    b.emit_label(append_scanned);
    const uint32_t append_is_leader = b.land(
        append_pending, b.ucmp(Op_IEqual, mbcnt_lane, zero));
    const uint32_t append_leader = b.id(), append_reduced = b.id();
    b.emit_selmerge(append_reduced);
    b.emit_condbranch(append_is_leader, append_leader, append_reduced);
    b.emit_label(append_leader);
    const uint32_t append_count_value = b.load_function(b.t_u32, append_count_var);
    const uint32_t append_delta = b.sel(
        b.load_function(b.t_bool, append_consume_var),
        b.ibin(Op_ISub, zero, append_count_value), append_count_value);
    const uint32_t append_index = b.load_function(b.t_u32, append_idx_var);
    uint32_t append_old = 0;
    if (has_gds_append && has_lds_append) {
        const uint32_t gds_block = b.id(), lds_block = b.id(), memory_merge = b.id();
        b.emit_selmerge(memory_merge);
        b.emit_condbranch(b.load_function(b.t_bool, append_gds_var),
                          gds_block, lds_block);
        b.emit_label(gds_block);
        const uint32_t gds_old = b.compute_gds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta);
        const uint32_t gds_end = b.cur_block;
        b.emit_branch(memory_merge);
        b.emit_label(lds_block);
        const uint32_t lds_old = b.lds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta, false, yes, zero);
        const uint32_t lds_end = b.cur_block;
        b.emit_branch(memory_merge);
        b.emit_label(memory_merge);
        append_old = b.emit_phi_2way(
            b.t_u32, gds_old, gds_end, lds_old, lds_end);
    } else if (has_gds_append) {
        append_old = b.compute_gds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta);
    } else {
        append_old = b.lds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta, false, yes, zero);
    }
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index), append_old);
    b.emit_branch(append_reduced);
    b.emit_label(append_reduced);
    b.barrier();

    const uint32_t append_result = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index));
    const uint32_t append_dst = b.load_function(b.t_u32, append_dst_var);
    const uint32_t append_write = b.land(append_pending, append_active);
    for (const auto& kv : vv) {
        const uint32_t selected = b.land(
            append_write,
            b.ucmp(Op_IEqual, append_dst, b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, append_result, old));
    }
    for (const auto& kv : lv) {
        const uint32_t selected = b.land(
            append_pending,
            b.ucmp(Op_IEqual, append_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        const uint32_t selected = b.land(
            append_pending,
            b.ucmp(Op_IEqual, append_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    // Publish this lane's pending vote bit and liveness. The switch merge is reached by every
    // invocation on every iteration, including lanes whose emulated wave has already ended.
    const uint32_t pending = b.load_function(b.t_bool, vote_pending_var);
    const uint32_t vote_value = b.load_function(b.t_bool, vote_value_var);
    const uint32_t vote_bit = b.sel(b.land(pending, vote_value), b.uconst(1), zero);
    const uint32_t active_bit = b.sel(b.load_function(b.t_bool, active_var), b.uconst(2), zero);
    b.cfg_scratch_store(b.linear_localid, b.ibin(Op_BitwiseOr, vote_bit, active_bit));
    b.barrier();

    const uint32_t wave_shift = b.wave_size == 32 ? 5u : 6u;
    const uint32_t wave_index = b.ibin(Op_ShiftRightLogical, b.linear_localid,
                                       b.uconst(wave_shift));
    const uint32_t wave_base = b.ibin(Op_ShiftLeftLogical, wave_index,
                                      b.uconst(wave_shift));
    const uint32_t lane_in_wave = b.ibin(Op_BitwiseAnd, b.linear_localid,
                                         b.uconst(b.wave_size - 1));

    // One lane per hardware wave reduces that wave's vote. Padding keeps every dynamic access in
    // bounds for a partial final wave; padded values are explicitly masked out.
    uint32_t wave_leader = b.id(), wave_reduced = b.id();
    const uint32_t is_wave_leader = b.ucmp(Op_IEqual, lane_in_wave, zero);
    b.emit_selmerge(wave_reduced);
    b.emit_condbranch(is_wave_leader, wave_leader, wave_reduced);
    b.emit_label(wave_leader);
    uint32_t wave_flags = zero;
    for (uint32_t lane = 0; lane < b.wave_size; ++lane) {
        const uint32_t idx = b.ibin(Op_IAdd, wave_base, b.uconst(lane));
        uint32_t flags = b.cfg_scratch_load(idx);
        if (padded_lanes != b.local_count)
            flags = b.sel(b.ucmp(Op_ULessThan, idx, b.uconst(b.local_count)), flags, zero);
        wave_flags = b.ibin(Op_BitwiseOr, wave_flags,
                            b.ibin(Op_BitwiseAnd, flags, b.uconst(1)));
    }
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(wave_result_base), wave_index), wave_flags);
    b.emit_branch(wave_reduced);
    b.emit_label(wave_reduced);

    // Lane zero also reduces workgroup liveness. This uniform dispatcher loop is what makes the
    // internal barriers legal even after one hardware wave reaches S_ENDPGM before another.
    uint32_t group_leader = b.id(), group_reduced = b.id();
    const uint32_t is_group_leader = b.ucmp(Op_IEqual, b.linear_localid, zero);
    b.emit_selmerge(group_reduced);
    b.emit_condbranch(is_group_leader, group_leader, group_reduced);
    b.emit_label(group_leader);
    uint32_t group_flags = zero;
    for (uint32_t lane = 0; lane < b.local_count; ++lane) {
        const uint32_t flags = b.cfg_scratch_load(b.uconst(lane));
        group_flags = b.ibin(Op_BitwiseOr, group_flags,
                             b.ibin(Op_BitwiseAnd, flags, b.uconst(2)));
    }
    b.cfg_scratch_store(b.uconst(group_active_slot), group_flags);
    b.emit_branch(group_reduced);
    b.emit_label(group_reduced);
    b.barrier();

    const uint32_t wave_flags_result = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), wave_index));
    const uint32_t wave_any = b.ucmp(
        Op_INotEqual, b.ibin(Op_BitwiseAnd, wave_flags_result, b.uconst(1)), zero);
    const uint32_t vote_condition = b.bsel(
        b.load_function(b.t_bool, vote_invert_var), b.logical_not(wave_any), wave_any);
    const uint32_t selected_pc = b.sel(vote_condition,
        b.load_function(b.t_u32, vote_taken_var), b.load_function(b.t_u32, vote_next_var));
    const uint32_t vote_to_scc = b.load_function(b.t_bool, vote_to_scc_var);
    const uint32_t write_scc = b.land(pending, vote_to_scc);
    b.store_function(scc_var,
        b.bsel(write_scc, vote_condition, b.load_function(b.t_bool, scc_var)));
    const uint32_t write_pc = b.land(pending, b.logical_not(vote_to_scc));
    b.store_function(pc_var, b.sel(write_pc, selected_pc, b.load_function(b.t_u32, pc_var)));
    const uint32_t group_active = b.ucmp(
        Op_INotEqual, b.cfg_scratch_load(b.uconst(group_active_slot)), zero);
    b.emit_condbranch(group_active, loop_header, loop_merge);
    }
    b.emit_label(loop_merge);

    // Expose the final emulated state to the caller. Graphics exports are emitted in their exact
    // cases, while any post-body bookkeeping still sees this invocation's final register values.
    initial = load_state();
    return true;
}

struct BarrierPhasedCompute {
    size_t guard_index = 0;
    size_t end_index = 0;
    std::vector<size_t> barriers;
    bool found = false;
};

// Recognize a workgroup-uniform scalar terminal guard around barrier-separated compute phases.
// Keeping this proof shared is important: emit_body uses it to split the shader, while the native
// subgroup policy uses the same result to make nested wave votes legal inside an individual phase.
BarrierPhasedCompute analyze_barrier_phased_compute(const std::vector<Rdna2Inst>& ins) {
    BarrierPhasedCompute result;
    result.end_index = ins.size();
    uint32_t end_pc = UINT32_MAX;
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!ins[i].is_end) continue;
        result.end_index = i;
        end_pc = ins[i].pc;
        break;
    }

    result.guard_index = ins.size();
    size_t first_barrier = ins.size();
    size_t branch_count = 0;
    for (size_t i = 0; i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a &&
            first_barrier == ins.size())
            first_barrier = i;
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03) continue;
        ++branch_count;
        if (result.guard_index == ins.size() &&
            (in.opcode == 0x04 || in.opcode == 0x05) && branch_target(in) == end_pc)
            result.guard_index = i;
    }

    bool valid = result.guard_index < first_barrier && first_barrier < result.end_index &&
        branch_count > 2;
    for (size_t i = 0; valid && i < result.guard_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // Scalar ALU/loads retain workgroup-uniform values when entered from compute user data and
        // TGIDs. EXEC is lane state, so any explicit access to either half invalidates this proof.
        // VCC physical words may be scalar scratch provided no vector instruction enters the prefix.
        const bool scalar = in.fmt == Rdna2Format::SOP1 ||
            in.fmt == Rdna2Format::SOP2 || in.fmt == Rdna2Format::SOPK ||
            in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::SMEM ||
            (in.fmt == Rdna2Format::SOPP && sopp_is_noop(in));
        if (!scalar) {
            valid = false;
            break;
        }
        auto is_exec = [](const Operand& operand) {
            return (operand.kind == OperandKind::SGPR ||
                    operand.kind == OperandKind::Special) &&
                   (operand.value == 126 || operand.value == 127);
        };
        if (is_exec(in.dst)) valid = false;
        for (uint32_t source = 0; source < in.n_src; ++source)
            if (is_exec(in.src[source])) valid = false;
    }

    for (size_t i = result.guard_index + 1; valid && i < result.end_index; ++i)
        if (ins[i].fmt == Rdna2Format::SOPP && ins[i].opcode == 0x0a)
            result.barriers.push_back(i);
    // A phase boundary is valid only when no guest branch jumps across it in either direction.
    for (size_t barrier_index : result.barriers) {
        const uint32_t barrier_pc = ins[barrier_index].pc;
        for (size_t i = result.guard_index + 1; valid && i < result.end_index; ++i) {
            const Rdna2Inst& in = ins[i];
            if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
                in.opcode == 0x03 || in.opcode == 0x0a) continue;
            const uint32_t target = branch_target(in);
            if ((in.pc < barrier_pc && target >= barrier_pc) ||
                (in.pc > barrier_pc && target <= barrier_pc))
                valid = false;
        }
    }
    result.found = valid && !result.barriers.empty();
    return result;
}

// Emit the instruction body (shared by every stage). Handles a single recognized COUNTED loop as a real
// structured SPIR-V loop (OpLoopMerge + OpPhi for loop-carried registers); loop-FREE streams walk straight
// through, byte-identical to the pre-loop-feature behavior. `exp_fn` handles an EXP instruction per stage
// (compute: reject; fragment: MRT color; vertex: POS/PARAM). Returns false if any instruction is
// unsupported. allow_exec_update / allow_smem match the stage's emit_alu flags.
bool emit_body(SpirvCompute& b, RegState& rs, const std::vector<Rdna2Inst>& ins,
               const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
               bool allow_exec_update, bool allow_smem,
               const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn,
               const uint32_t* code = nullptr, size_t dwords = 0,
               const std::unordered_set<uint32_t>* inherited_dead_masks = nullptr) {
               // code/dwords: raw stream for forward-if target checks; inherited_dead_masks keeps
               // whole-shader liveness valid when a barrier-separated body is compiled in phases.
    rs.max_vgpr = std::max(rs.max_vgpr, shader_max_vgpr(ins));
    // Fold PC-relative embedded-table loads (s_getpc_b64-built V#s) before the walk — emit_alu's
    // SMEM/MUBUF/SOP1 handlers consult the proven table maps (#273/#1054).
    if (code) {
        PcrelTables tables = detect_pcrel_tables(ins, code, dwords);
        rs.mubuf_pcrel_tables = std::move(tables.mubuf);
        rs.smem_pcrel_tables = std::move(tables.smem);
    }
    std::unordered_set<uint32_t> local_dead_masks;
    if (!inherited_dead_masks) local_dead_masks = dead_wave_mask_writes(ins);
    const std::unordered_set<uint32_t>& dead_masks = inherited_dead_masks
        ? *inherited_dead_masks : local_dead_masks;

    // A large generated compute kernel may put a workgroup-uniform scalar early-out around several
    // barrier-separated phases, then use arbitrary (but barrier-free) control flow in its final
    // phase. The whole-stream CFG dispatcher cannot legally place an OpControlBarrier in one switch
    // case, while the narrow structurizer rejects the final phase's branch graph. Peel the terminal
    // scalar guard and compile each barrier-free phase independently. Every invocation in one
    // workgroup takes the guard together, and no guest edge may cross a split, so each explicit
    // barrier remains uniform while the final phase can use the ordinary CFG dispatcher.
    if (b.is_compute) {
        const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
        if (phased.found) {
            const std::vector<Rdna2Inst> prefix(
                ins.begin(), ins.begin() + phased.guard_index);
            if (!prefix.empty() &&
                !emit_body(b, rs, prefix, safe, rt, allow_exec_update, allow_smem,
                           exp_fn, code, dwords, &dead_masks))
                return false;
            if (!rs.scc) return false;
            const uint32_t execute_body = ins[phased.guard_index].opcode == 0x04
                ? rs.scc : b.logical_not(rs.scc);
            const uint32_t body_label = b.id(), merge_label = b.id();
            b.emit_selmerge(merge_label);
            b.emit_condbranch(execute_body, body_label, merge_label);
            b.emit_label(body_label);

            size_t phase_begin = phased.guard_index + 1;
            for (size_t barrier_index : phased.barriers) {
                const std::vector<Rdna2Inst> phase(
                    ins.begin() + phase_begin, ins.begin() + barrier_index);
                if (!phase.empty()) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr, "[compute-phase] begin=%u end=%u barrier=%u\n",
                                     phase.front().pc, phase.back().pc, ins[barrier_index].pc);
                    if (!emit_body(b, rs, phase, safe, rt, allow_exec_update, allow_smem,
                                   exp_fn, code, dwords, &dead_masks)) {
                        if (getenv("PROSPER_DBG"))
                            std::fprintf(stderr, "[compute-phase-reject] begin=%u end=%u\n",
                                         phase.front().pc, phase.back().pc);
                        return false;
                    }
                }
                b.barrier();
                phase_begin = barrier_index + 1;
            }
            const std::vector<Rdna2Inst> tail(ins.begin() + phase_begin, ins.end());
            if (!tail.empty()) {
                if (getenv("PROSPER_DBG"))
                    std::fprintf(stderr, "[compute-phase] begin=%u end=%u tail=1\n",
                                 tail.front().pc, tail.back().pc);
                if (!emit_body(b, rs, tail, safe, rt, allow_exec_update, allow_smem,
                               exp_fn, code, dwords, &dead_masks)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr, "[compute-phase-reject] begin=%u end=%u tail=1\n",
                                     tail.front().pc, tail.back().pc);
                    return false;
                }
            }
            b.emit_branch(merge_label);
            b.emit_label(merge_label);
            return true;
        }
    }
    std::unordered_set<uint32_t> effective_safe = safe;
    const CountedLoop L = detect_counted_loop(ins);
    size_t idx = 0;
    // Cross-lane wave ops (mbcnt) emit LDS + barriers, which are only valid at wave-uniform points — so
    // they're allowed ONLY in the straight-line path (no divergent loop/if around them). Set true below.
    bool wave_ok = false;
    auto emit_range = [&](uint32_t pc_lo, uint32_t pc_hi) -> bool {   // emit ins whose pc ∈ [pc_lo, pc_hi)
        for (; idx < ins.size(); ++idx) {
            const Rdna2Inst& in = ins[idx];
            if (in.is_end || in.pc >= pc_hi) break;
            if (in.pc < pc_lo) continue;
            if (dead_masks.count(in.pc)) continue;
            if (in.fmt == Rdna2Format::EXP) { if (!exp_fn(rs, in)) return false; continue; }
            bool ok = true;
            const bool handled = emit_alu(
                b, rs, in, ok, allow_exec_update, &effective_safe, allow_smem, rt, wave_ok);
            if (handled && ok) record_scalar_write(rs, in);
            // Shader I/O tap: snapshot this instruction's destination VGPR (+3) if it is the tapped PC.
            if (handled && ok && in.pc == b.tap_pc && in.dst.kind == OperandKind::VGPR) {
                auto tv = [&](int r) { auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
                b.set_tap(tv(in.dst.value), tv(in.dst.value + 1), tv(in.dst.value + 2), tv(in.dst.value + 3));
            }
            if (!handled || !ok) {
                // PROSPER_DBG (gated, off by default): report the instruction that fails recompilation —
                // the first unsupported op / unresolved resource that makes a shader return empty.
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr, "[recompile-reject] pc=%u words=%08x,%08x fmt=%d op=0x%x "
                                    "dst=%d(kind%d) src=%d(k%d),%d(k%d),%d(k%d) dmask=0x%x "
                                    "dim=%u glc=%d len=%u modifier=%d dpp=%d sdwa=%u/%u/%u/%u\n",
                        in.pc, in.words[0], in.words[1], (int)in.fmt, in.opcode,
                        in.dst.value, (int)in.dst.kind,
                        in.src[0].value, (int)in.src[0].kind,
                        in.src[1].value, (int)in.src[1].kind,
                        in.src[2].value, (int)in.src[2].kind,
                        in.mimg_dmask, in.mimg_dim, (int)in.mimg_glc, in.len_dwords, (int)in.has_modifier,
                        (int)in.has_dpp, in.sdwa_dst_sel, in.sdwa_dst_unused,
                        in.sdwa_src0_sel, in.sdwa_src1_sel);
                return false;
            }
        }
        return true;
    };
    auto& safe_branches = effective_safe;
    if (L.found) {
        auto vget = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
        auto sget = [&](int r){ auto it = rs.sreg.find(r); return it == rs.sreg.end() ? b.uconst(0) : it->second; };
        bool guarded_narrow_entry = false;
        // saveexec -> execz -> matching EXEC restore around a side-effect-free counted region is a
        // whole-wave empty-work optimization. In the per-invocation shell we may run the uniform
        // scalar loop for every invocation while narrowed EXEC predicates vector writes; inactive
        // lanes retain their old VGPRs until the exact restore. Reject stores/exports/barriers and
        // unclassified memory so this never becomes a general branch-linearization escape hatch.
        // Scan inside-out so an already-proven nested guard may contribute its balanced save/restore
        // pair without making an otherwise-safe outer guarded loop look like it leaks narrowed EXEC.
        struct GuardedExecRegion { uint32_t save_pc, restore_pc; };
        std::vector<GuardedExecRegion> guarded_exec_regions;
        for (size_t branch_index = ins.size(); branch_index-- > 0;) {
            const Rdna2Inst& branch = ins[branch_index];
            if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 || branch.simm16 <= 0)
                continue;
            size_t previous = branch_index;
            while (previous > 0) {
                --previous;
                if (!sopp_is_noop(ins[previous])) break;
            }
            if (previous >= branch_index) continue;
            const Rdna2Inst& saveexec = ins[previous];
            if (saveexec.fmt != Rdna2Format::SOP1 ||
                (saveexec.opcode != 0x24 && saveexec.opcode != 0x25) ||
                saveexec.dst.kind != OperandKind::SGPR || saveexec.dst.value > 104) continue;
            const uint32_t target = branch_target(branch);
            const Rdna2Inst* restore = nullptr;
            for (const auto& candidate : ins) if (candidate.pc == target) { restore = &candidate; break; }
            if (!restore || restore->fmt != Rdna2Format::SOP1 || restore->opcode != 0x04 ||
                restore->dst.value < 126 || !reg_operand(restore->src[0], saveexec.dst.value)) continue;
            // A lexical save/restore pair is not necessarily balanced along the counted-loop CFG.
            // In particular, a save in the body with its restore after the backedge leaves EXEC
            // narrowed between iterations (EXEC has no loop phi), and a zero-trip path reaches an
            // undominated restore. Accept only a pair contained in one straight-line loop segment,
            // or a true preheader-to-postloop wrapper around the complete loop.
            const bool same_preloop = saveexec.pc < L.header_pc && target < L.header_pc;
            const bool same_condition = saveexec.pc >= L.header_pc && target < L.exit_branch_pc;
            const bool same_body = saveexec.pc > L.exit_branch_pc && target < L.backedge_pc;
            const bool same_postloop = saveexec.pc >= L.exit_pc;
            const bool wraps_loop = saveexec.pc < L.header_pc && target >= L.exit_pc;
            if (!same_preloop && !same_condition && !same_body && !same_postloop && !wraps_loop)
                continue;
            bool side_effect_free = true;
            for (const auto& candidate : ins) {
                if (candidate.pc <= branch.pc || candidate.pc >= target) continue;
                bool clobbers_guard_mask = false;
                for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                    clobbers_guard_mask |= base < saveexec.dst.value + 2 &&
                        saveexec.dst.value < base + static_cast<int>(width);
                });
                bool balanced_nested_exec = false;
                for (const auto& nested : guarded_exec_regions) {
                    if (nested.save_pc > branch.pc && nested.restore_pc < target &&
                        (candidate.pc == nested.save_pc || candidate.pc == nested.restore_pc)) {
                        balanced_nested_exec = true;
                        break;
                    }
                }
                if (candidate.fmt == Rdna2Format::EXP || candidate.fmt == Rdna2Format::DS ||
                    candidate.fmt == Rdna2Format::MUBUF || candidate.fmt == Rdna2Format::MTBUF ||
                    candidate.fmt == Rdna2Format::MIMG || candidate.fmt == Rdna2Format::FLAT ||
                    (instruction_may_change_exec(candidate) && !balanced_nested_exec) ||
                    clobbers_guard_mask ||
                    (candidate.fmt == Rdna2Format::SOPP && candidate.opcode == 0x0a)) {
                    side_effect_free = false;
                    break;
                }
            }
            if (!side_effect_free) continue;
            effective_safe.insert(branch.pc);
            guarded_exec_regions.push_back({saveexec.pc, target});
            if (branch.pc < L.header_pc && target >= L.exit_pc) guarded_narrow_entry = true;
        }
        // 1. Pre-loop body. A compiler may place one ordinary uniform if/else before the canonical
        // counted loop (Evergate selects one of two constant blocks this way; Astro's NGG culling
        // prelude also has a one-arm conditional). Structure that choice with the same two-arm PHIs
        // as the general forward-if path, then enter the existing counted-loop lowering. Anything
        // nested/more complex stays unsupported and rejects visibly.
        std::vector<Rdna2Inst> preloop;
        for (const auto& in : ins) {
            if (in.pc >= L.header_pc) break;
            // A forward execz spanning the counted loop is a redundant wave-empty guard when the
            // prelude reaches it with full EXEC. Leave it in the emitted stream (emit_alu proves
            // full EXEC and otherwise rejects), but do not ask the pre-loop uniform-if detector to
            // model a region whose body deliberately extends beyond its artificial end marker.
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                branch_target(in) >= L.header_pc) continue;
            preloop.push_back(in);
        }
        Rdna2Inst preloop_end;
        preloop_end.pc = L.header_pc;
        preloop_end.is_end = true;
        preloop.push_back(preloop_end);
        bool preloop_rejected = false;
        const std::vector<ForwardIf> preloop_ifs = detect_forward_ifs(
            preloop, /*allow_vcc*/!b.is_compute, code, dwords, &effective_safe, nullptr,
            &preloop_rejected, /*compute_wave_branches*/b.is_compute);
        // detect_forward_ifs clamps a branch to an immediate s_endpgm at its artificial end marker
        // and records it as early_out. In this truncated prelude that can be a real branch over the
        // entire counted loop, so it cannot be structured as an ordinary one-arm conditional.
        const bool preloop_if_unsupported = !preloop_ifs.empty() &&
            (preloop_ifs[0].early_out ||
             (preloop_ifs[0].has_else ? preloop_ifs[0].merge_pc : preloop_ifs[0].target_pc) >
                 L.header_pc);
        if (preloop_rejected || preloop_ifs.size() > 1 || preloop_if_unsupported) {
            if (getenv("PROSPER_DBG"))
                fprintf(stderr,
                        "[recompile-reject] counted-loop prelude cfg rejected=%u ifs=%zu header=%u\n",
                        preloop_rejected, preloop_ifs.size(), L.header_pc);
            return false;
        }
        if (preloop_ifs.empty()) {
            if (!emit_range(0, L.header_pc)) return false;
        } else {
            const ForwardIf F = preloop_ifs[0];
            if (!emit_range(0, F.branch_pc)) return false;
            if (idx >= ins.size() || ins[idx].pc != F.branch_pc) return false;
            ++idx; // consume the conditional branch
            // An scc-conditioned forward-if with a POISONED SCC (rs.scc == 0: a 64-bit mask op was
            // the last architectural SCC writer, unrepresentable per-lane) must reject — this is
            // exactly the non-adjacent stale-SCC consumer from the ISA audit (#879).
            if (!F.on_exec && !F.on_vcc && !rs.scc) return false;
            uint32_t condition = F.on_exec ? rs.exec : (F.on_vcc ? rs.vcc : rs.scc);
            if (b.is_fragment && (F.on_exec || F.on_vcc))
                condition = b.fragment_wave_any(condition);
            condition = F.on_scc0 ? condition : b.logical_not(condition);
            const RegState before = rs;
            std::set<int> written_v, written_s;
            const uint32_t then_end = F.has_else ? F.sb_pc : F.target_pc;
            const uint32_t merge_pc = F.has_else ? F.merge_pc : F.target_pc;
            loop_written_regs(ins, F.branch_pc + 1, then_end, written_v, written_s);
            if (F.has_else)
                loop_written_regs(ins, F.target_pc, merge_pc, written_v, written_s);
            const uint32_t then_label = b.id(), else_label = b.id(), merge_label = b.id();
            b.emit_selmerge(merge_label);
            b.emit_condbranch(condition, then_label, else_label);

            b.emit_label(then_label);
            if (!emit_range(F.branch_pc + 1, then_end)) return false;
            if (F.has_else) {
                if (idx >= ins.size() || ins[idx].pc != F.sb_pc) return false;
                ++idx; // consume the then arm's jump to the merge
            }
            const uint32_t then_block = b.cur_block;
            std::unordered_map<int, uint32_t> then_v, then_s;
            for (int reg : written_v) then_v[reg] = vget(reg);
            for (int reg : written_s) then_s[reg] = sget(reg);
            const uint32_t then_scc = rs.scc, then_vcc = rs.vcc, then_exec = rs.exec;
            const bool then_narrowed = rs.exec_narrowed;
            const auto then_bool = rs.sreg_bool;
            const auto then_bool_b32 = rs.sreg_bool_b32;
            const auto then_written = rs.sreg_written;
            b.emit_branch(merge_label);

            rs = before;
            b.emit_label(else_label);
            if (F.has_else && !emit_range(F.target_pc, merge_pc)) return false;
            const uint32_t else_block = b.cur_block;
            b.emit_branch(merge_label);
            b.emit_label(merge_label);
            for (int reg : written_v) {
                const uint32_t else_value = vget(reg);
                if (then_v[reg] != else_value)
                    rs.vreg[reg] = b.emit_phi_2way(
                        b.t_u32, then_v[reg], then_block, else_value, else_block);
            }
            for (int reg : written_s) {
                const uint32_t else_value = sget(reg);
                if (then_s[reg] != else_value)
                    rs.sreg[reg] = b.emit_phi_2way(
                        b.t_u32, then_s[reg], then_block, else_value, else_block);
            }
            if (then_scc != rs.scc)
                // A poisoned (0) input degrades to bfalse across the merge — no invalid SPIR-V and
                // no stricter rejection than the pre-poison behavior; straight-line consumers of a
                // poisoned SCC are still rejected at their own sites.
                rs.scc = b.emit_phi_2way(b.t_bool, then_scc ? then_scc : b.bfalse(), then_block,
                                         rs.scc ? rs.scc : b.bfalse(), else_block);
            if (then_vcc != rs.vcc)
                rs.vcc = b.emit_phi_2way(b.t_bool, then_vcc, then_block, rs.vcc, else_block);
            if (then_exec != rs.exec)
                rs.exec = b.emit_phi_2way(b.t_bool, then_exec, then_block, rs.exec, else_block);
            rs.exec_narrowed = then_narrowed || rs.exec_narrowed;
            rs.sreg_written.insert(then_written.begin(), then_written.end());
            for (int reg : then_written) rs.sreg_input.erase(reg);
            if (then_bool != rs.sreg_bool || then_bool_b32 != rs.sreg_bool_b32) {
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr, "[recompile-reject] counted-loop prelude changes mask domain\n");
                return false; // no mask-domain PHIs in this narrow composition
            }
            if (!emit_range(merge_pc, L.header_pc)) return false;
        }
        // Ordinary counted loops require full EXEC at entry. An NGG vertex is already represented by
        // one independent Vulkan invocation, however, so its EXEC bit is an ordinary per-invocation
        // Boolean. Carry that bit through the loop just like SCC/VCC instead of rejecting Astro Bot's
        // fixed-trip culling loop merely because some guest lanes were masked before its header.
        const bool carry_vertex_exec = b.ngg_one_lane && rs.exec_narrowed;
        if (rs.exec_narrowed && !guarded_narrow_entry && !carry_vertex_exec) {
            if (getenv("PROSPER_DBG"))
                fprintf(stderr, "[recompile-reject] counted-loop enters with narrowed EXEC\n");
            return false;
        }
        // 2. Loop-carried registers -> a header OpPhi each. `cond_written` = regs written in the CONDITION
        // region [header, exit_branch): those execute on the exiting iteration too, so their post-loop
        // value is the condition-block value (which dominates the merge), NOT the phi (defect A). SCC/VCC
        // always get a phi so any cross-iteration carry is valid SSA (defect C); they're recomputed each
        // iteration in practice, so the phi is usually dead — harmless.
        std::set<int> cv, cs, condv, conds, scalar_may_writes;
        loop_written_regs(ins, L.header_pc, L.backedge_pc, cv, cs);
        loop_written_regs(ins, L.header_pc, L.exit_branch_pc, condv, conds);
        loop_scalar_may_writes(ins, L.header_pc, L.backedge_pc, scalar_may_writes);
        for (int reg : rs.sreg_bool_b32)
            if (scalar_may_writes.contains(reg)) return false;
        if (b.allow_b32_masks &&
            has_unpersisted_b32_mask_lifetime(
                ins, L.header_pc, L.backedge_pc, rs))
            return false;
        const uint32_t preheader = b.cur_block;
        const uint32_t hdr = b.id(), check = b.id(), body = b.id(), cont = b.id(), merge = b.id();
        b.emit_branch(hdr); b.emit_label(hdr);
        struct PhiRec { int reg; int dom; uint32_t phi; size_t patch; };   // dom: 0=vreg,1=sreg,2=scc,3=vcc,4=exec
        std::vector<PhiRec> phis;
        for (int r : cv) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, vget(r), preheader, p); rs.vreg[r] = ph; phis.push_back({r, 0, ph, p}); }
        for (int r : cs) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, sget(r), preheader, p); rs.sreg[r] = ph; phis.push_back({r, 1, ph, p}); }
        // A poisoned (0) SCC live-in degrades to bfalse — the loop shapes re-produce SCC via their
        // in-loop s_cmp before any read, so the phi seed is dead in practice; 0 would be invalid SSA.
        { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.scc ? rs.scc : b.bfalse(), preheader, p); rs.scc = ph; phis.push_back({0, 2, ph, p}); }
        { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.vcc, preheader, p); rs.vcc = ph; phis.push_back({0, 3, ph, p}); }
        if (carry_vertex_exec) {
            size_t p;
            uint32_t ph = b.emit_phi2(b.t_bool, rs.exec, preheader, p);
            rs.exec = ph;
            phis.push_back({0, 4, ph, p});
        }
        // The header executes again after the back-edge. A direct/SRT descriptor overwritten
        // anywhere in the loop is therefore not an invariant entry descriptor at header compile
        // time. An exact descriptor load in the header may establish fresh provenance afterward.
        invalidate_loop_descriptor_provenance(rs, scalar_may_writes);
        b.emit_loopmerge(merge, cont); b.emit_branch(check); b.emit_label(check);
        // 3. Condition block: emit [header, exit_branch); the SCC exit becomes OpBranchConditional.
        if (!emit_range(L.header_pc, L.exit_branch_pc)) return false;
        if (!rs.scc) return false;   // condition region left SCC poisoned: the exit test is unknowable
        // Snapshot condition-region register values (these dominate the merge — see defect A above).
        std::unordered_map<int,uint32_t> condv_val, conds_val;
        for (int r : condv) condv_val[r] = vget(r);
        for (int r : conds) conds_val[r] = sget(r);
        const uint32_t cond_exec = rs.exec;
        const bool cond_exec_narrowed = rs.exec_narrowed;
        // s_cbranch_scc0 exits when SCC==0 (so the loop CONTINUES when SCC!=0); scc1 is the inverse.
        uint32_t loop_cond = L.exit_on_scc0 ? rs.scc : b.bsel(rs.scc, b.bfalse(), b.btrue());
        b.emit_condbranch(loop_cond, body, merge);
        while (idx < ins.size() && ins[idx].pc < L.exit_branch_pc) ++idx;   // (already past it)
        if (idx < ins.size() && ins[idx].pc == L.exit_branch_pc) ++idx;     // skip the exit branch itself
        b.emit_label(body);
        // 4. Body [after exit_branch, back-edge). Must restore EXEC before looping (bail if left narrowed).
        if (!emit_range(L.exit_branch_pc + 1, L.backedge_pc)) return false;
        if (rs.exec_narrowed && !guarded_narrow_entry && !carry_vertex_exec) {
            if (getenv("PROSPER_DBG"))
                fprintf(stderr, "[recompile-reject] counted-loop body leaves EXEC narrowed\n");
            return false;
        }
        if (idx < ins.size() && ins[idx].pc == L.backedge_pc) ++idx;        // skip the back-edge branch
        // 5. Continue block branches back to the header; patch each phi's back-edge (value = current, cont).
        b.emit_branch(cont); b.emit_label(cont);
        for (auto& pr : phis) {
            uint32_t nv = pr.dom == 0 ? vget(pr.reg)
                        : pr.dom == 1 ? sget(pr.reg)
                        : pr.dom == 2 ? rs.scc
                        : pr.dom == 3 ? rs.vcc : rs.exec;
            if (!nv && pr.dom == 2) nv = b.bfalse();   // poisoned SCC back-edge value: bfalse (dead in practice)
            b.patch_phi(pr.patch, nv, cont);
        }
        b.emit_branch(hdr);
        // 6. Merge (loop exit): a condition-region reg keeps its exit-iteration (%check) value; a body-only
        //    reg (and scc/vcc) takes the header phi (its value when the loop exited).
        b.emit_label(merge);
        for (auto& pr : phis) {
            if (pr.dom == 0)      rs.vreg[pr.reg] = condv.count(pr.reg) ? condv_val[pr.reg] : pr.phi;
            else if (pr.dom == 1) rs.sreg[pr.reg] = conds.count(pr.reg) ? conds_val[pr.reg] : pr.phi;
            // SCC/VCC take the phi (not a condition-region snapshot): reading a wave flag AFTER a loop is
            // not a real codegen pattern (flags are transient, consumed by their branch), so the A-class
            // exit-iteration refinement is intentionally omitted for them.
            else if (pr.dom == 2) rs.scc = pr.phi;
            else if (pr.dom == 3) rs.vcc = pr.phi;
            else                  rs.exec = cond_exec;
        }
        if (carry_vertex_exec) rs.exec_narrowed = cond_exec_narrowed;
        // 7. Post-loop body. Feed the suffix back through the ordinary body selector: with this
        // counted back-edge removed it can use the established nested forward-if/divergent-loop
        // structurizer. This composes a counted loop with a non-trivial shared postlude without
        // duplicating that CFG machinery or admitting arbitrary branches inside the counted loop.
        std::vector<Rdna2Inst> postloop;
        for (const auto& in : ins) if (in.pc >= L.exit_pc) postloop.push_back(in);
        if (!postloop.empty() &&
            !emit_body(b, rs, postloop, effective_safe, rt, allow_exec_update, allow_smem,
                       exp_fn, code, dwords)) return false;
    } else if (std::vector<DivLoop> Ls; true) {
        // EXEC/VCC-exit loops (#273/#615) + structured scalar IFs. Each is a real structured SPIR-V
        // loop with header phis for carried register/mask state. Fragment conditions are exact wave64
        // votes; vertex and the guarded compute cases retain their per-invocation form. The IF
        // machinery recurses into loop bodies and handles their nested forward-execz regions.
        Ls = detect_divergent_loops(ins, safe, b.is_fragment);
        if (b.is_compute) {
            // Compute VCC-exit loops (#590, extending #615): the fragment-stage uniformity proof is
            // data-provenance-based, not stage-based — vcc_exit_is_wave_uniform accepts a compare only
            // when every input is scalar/inline/literal or a VGPR whose nearest definition is an
            // unmodified uniform VOP1 move from a scalar. With that proven, every lane's compare bool
            // is identical, so the wave-empty vccz exit lowers to THIS invocation's !cond exactly as
            // in the fragment shell (tid-derived/varying inputs fail the proof and keep rejecting).
            // One compute-specific guard (see the per-condition detail below for why both Vcc- and
            // Exec-condition loops are safe here — the Exec case is GTA V's exec_cs_2042d47600 grid-stride
            // decode loop, #1183):
            //   * the body must be barrier/LDS/cross-lane-free — the proof is per-WAVE, and a barrier
            //     inside a loop whose trip count could differ across the workgroup's waves would be
            //     workgroup-divergent control flow (UB). DOLL's blocked light/fill kernels are
            //     straight-line bodies, so nothing observed is lost. CONFIDENCE: MED-HIGH (shared
            //     emit machinery; spirv-val + coverage tests + Messenger guard gate it).
            // Condition::Vcc loops are accepted under the detector's uniformity proof (#615/#590).
            // Condition::Exec loops (#590 — DOLL's nested post-process kernel is the observed compute
            // case) lower with the per-invocation model: this invocation
            // iterates while ITS EXEC bit holds after the header's v_cmpx recompute. Both flavors
            // require the barrier/LDS/cross-lane-free body below — the per-invocation trip count can
            // differ across a workgroup, so a barrier inside the loop would be workgroup-divergent
            // control flow (UB); barriers AFTER the loop are fine (the loop merge reconverges).
            bool compute_ok = !Ls.empty();
            for (const auto& L : Ls) {
                if (!compute_ok) break;
                for (const auto& in : ins) {
                    if (in.is_end || in.pc >= L.exit_pc) break;
                    if (in.pc < L.header_pc) continue;
                    if (in.fmt == Rdna2Format::DS ||
                        (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a) ||
                        (in.fmt == Rdna2Format::VOP3 &&
                         (in.opcode == 0x365 || in.opcode == 0x366))) { compute_ok = false; break; }
                }
                // Fail-visible marker for the per-invocation approximation's one known divergence: a
                // POST-loop read of an SGPR advanced inside an Exec-condition loop. With a lane-varying
                // bound, hardware advances in-loop scalars to the wave's MAX trip count while the
                // per-invocation lowering yields each invocation its own exit value. Uniform bounds
                // (all observed shapes) are exact. Diagnose loudly instead of silently diverging; if a
                // real kernel trips this AND has a varying bound, that is the evidence to revisit.
                if (compute_ok && L.condition == DivLoop::Condition::Exec && getenv("PROSPER_DBG")) {
                    std::set<int> lv, lsr;
                    loop_written_regs(ins, L.header_pc, L.backedge_pc, lv, lsr);
                    for (const auto& in : ins) {
                        if (in.is_end) break;
                        if (in.pc < L.exit_pc) continue;
                        for (uint32_t oi = 0; oi < in.n_src; ++oi)
                            if (in.src[oi].kind == OperandKind::SGPR && lsr.count((int)in.src[oi].value))
                                fprintf(stderr,
                                        "[compute-exec-loop] post-loop read of loop-advanced s%u at pc=%u "
                                        "(per-invocation value; wave max-trip on hardware)\n",
                                        (unsigned)in.src[oi].value, in.pc);
                    }
                }
            }
            if (!compute_ok) Ls.clear();   // unchanged behavior: the branch reaches emit_alu -> loud reject
        }
        bool cf_rejected = false;
        const std::vector<ForwardIf> Fs = detect_forward_ifs(ins, /*allow_vcc*/!b.is_compute, code, dwords, &safe,
                                                             Ls.empty() ? nullptr : &Ls, &cf_rejected,
                                                             /*compute_wave_branches*/b.is_compute);

        // UE4's volume-lighting kernels combine nested EXEC loops with a backward VCC vote loop.
        // That shape is intentionally outside the narrow pattern structurizer above. Use the
        // dispatcher fallback only when there is unmistakably complex compute control flow and the
        // body is free of operations whose workgroup-uniform placement the dispatcher cannot prove.
        // Raw DS operations are wave-local memory effects and are valid in a case. Guest barriers
        // remain unsupported; MBCNT is hoisted into the dispatcher's uniform synchronized phase.
        // Existing straight-line, single-if, and recognized single-loop shaders keep their old path.
        size_t cfg_branches = 0;
        const bool graphics_cfg = b.is_fragment || b.is_vertex;
        bool cfg_has_backedge = false, cfg_dispatch_safe = b.is_compute || graphics_cfg;
        for (const auto& in : ins) {
            if (in.is_end) break;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a)
                cfg_dispatch_safe = false;
            if (in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
                in.opcode != 0x03) {
                ++cfg_branches;
                if (branch_target(in) <= in.pc) cfg_has_backedge = true;
            }
        }
        // A lone exit + back-edge is the ordinary single-loop shape. Keep rejecting it when its
        // wave-uniformity proof fails; the dispatcher is reserved for genuinely nested/multi-branch CFGs.
        // The fallback is equally valid for an acyclic branch tree: Astro Bot's screen-space kernels
        // contain several nested forward EXEC/SCC early-outs but no back-edge. Requiring a back-edge
        // left those valid CFGs in the straight-line path, where their first branch rejected.
        const bool complex_compute_cfg = b.is_compute && cfg_dispatch_safe && cfg_branches > 2;
        const bool complex_graphics_cfg = graphics_cfg && cfg_dispatch_safe && cfg_branches > 2;
        const bool exact_compute_wave_cfg = b.is_compute &&
            std::any_of(Fs.begin(), Fs.end(), [](const ForwardIf& branch) {
                return branch.on_exec || branch.on_vcc;
            });
        // A structured shader may contain top-level guest-wave branches and workgroup-uniform barriers
        // between those regions (DOLL's title grading kernel and UE4's barrier-separated reductions).
        // The generic CFG dispatcher cannot contain guest barriers, but routing the entire shader there
        // is unnecessary: reduce each top-level branch with exact 32/64-lane scratch votes and retain the
        // structured emitter. Loops are not required; a sequence of forward top-level reductions has the
        // same safety argument. Neither a vote nor a guest barrier may be nested in a varying region.
        auto top_level_pc = [&](uint32_t pc) {
            for (const auto& parent : Fs) {
                const uint32_t parent_end = parent.has_else ? parent.merge_pc : parent.target_pc;
                if (parent.branch_pc < pc && pc < parent_end)
                    return false;
            }
            for (const auto& loop : Ls)
                if (pc >= loop.header_pc && pc <= loop.backedge_pc)
                    return false;
            return true;
        };
        const bool barriers_are_top_level =
            std::all_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
                return in.fmt != Rdna2Format::SOPP || in.opcode != 0x0a ||
                       top_level_pc(in.pc);
            });
        const bool structured_compute_wave_cfg = exact_compute_wave_cfg && !cf_rejected &&
            barriers_are_top_level &&
            std::all_of(Fs.begin(), Fs.end(), [&](const ForwardIf& branch) {
                // An exact native guest-size subgroup performs the vote without synthesized
                // workgroup barriers, so nested wave branches are safe. Portable scratch votes
                // must remain top-level so every workgroup invocation reaches their barriers.
                return (!branch.on_exec && !branch.on_vcc) || b.native_subgroup_size ||
                       top_level_pc(branch.branch_pc);
            });
        if (b.is_compute && cfg_branches && getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[compute-cfg] branches=%zu backedge=%d dispatch_safe=%d complex=%d "
                         "structured_ifs=%zu loops=%zu cf_rejected=%d exact_wave=%d "
                         "structured_wave=%d local=%u wave=%u\n",
                         cfg_branches, cfg_has_backedge, cfg_dispatch_safe, complex_compute_cfg,
                         Fs.size(), Ls.size(), cf_rejected, exact_compute_wave_cfg,
                         structured_compute_wave_cfg,
                         b.local_count, b.wave_size);
        if (graphics_cfg && cfg_branches && getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[graphics-cfg] stage=%s branches=%zu backedge=%d complex=%d "
                         "structured_ifs=%zu loops=%zu cf_rejected=%d\n",
                         b.is_fragment ? "fragment" : "vertex", cfg_branches,
                         cfg_has_backedge, complex_graphics_cfg, Fs.size(), Ls.size(), cf_rejected);
        if (exact_compute_wave_cfg && !structured_compute_wave_cfg) {
            // Native Vulkan subgroup widths may be 8/16/32 while the guest wave is 32/64. A native
            // subgroupAny would let different pieces of one guest wave take different scalar edges.
            // The dispatcher performs the reduction through Workgroup scratch and synchronized common
            // phases. Top-level wave branches use the compact structured reduction above even without
            // a guest barrier; nested wave branches still need this dispatcher. If a guest barrier
            // makes that transformation unsafe, reject rather than silently changing the branch domain.
            if (!cfg_dispatch_safe ||
                !emit_cfg_state_machine(b, rs, ins, safe, rt,
                                        allow_exec_update, allow_smem, exp_fn, code, dwords))
                return false;
            return true;
        }
        if (complex_compute_cfg && (cf_rejected || Ls.empty()) &&
            emit_cfg_state_machine(b, rs, ins, safe, rt,
                                   allow_exec_update, allow_smem, exp_fn, code, dwords))
            return true;
        // In graphics, the SPIR-V invocation already represents one guest lane. Complex reducible
        // control flow therefore needs no workgroup vote: the dispatcher selects the next block from
        // this pixel/vertex's SCC, VCC, or EXEC bit. Keep ordinary structured shaders on their compact
        // SSA path and use the Function-variable fallback only after the narrow structurizer rejects.
        if (complex_graphics_cfg && cf_rejected &&
            emit_cfg_state_machine(b, rs, ins, safe, rt,
                                   allow_exec_update, allow_smem, exp_fn, code, dwords))
            return true;
        if (cf_rejected) Ls.clear();   // unmodeled CF somewhere: fall through to straight-line (loud reject)
        if (Fs.empty() && Ls.empty()) {
            wave_ok = true;   // straight-line: barriers are wave-uniform, so cross-lane mbcnt is safe here
            if (!emit_range(0, UINT32_MAX)) return false;   // loop-free: straight-line, unchanged behavior
            (void)safe_branches;
            return true;
        }
        // Structured uniform IFs (forward s_cbranch_scc*/vcc*), possibly SEQUENTIAL and/or NESTED
        // (detect_forward_ifs verified the region tree). Each if emits as OpSelectionMerge +
        // OpBranchConditional on the SCC/VCC bool, with an OpPhi per register written in the
        // conditional block that is live after the merge — the same phi machinery as the original
        // single-if path (which this generalizes 1:1: a single if takes exactly the old shape).
        // Recursion handles nesting: the then-block emitter re-enters for inner branches.
        // CONFIDENCE: MED-HIGH — guarded by the test suite + exec-diff; DOLL's two-vccz color-grade
        // PS and nested-vccz lighting PS are the motivating real shaders (#273).
        auto vget = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
        auto sget = [&](int r){ auto it = rs.sreg.find(r); return it == rs.sreg.end() ? b.uconst(0) : it->second; };
        size_t bi = 0;   // next unconsumed branch in Fs (pc order; recursion consumes nested ones)
        size_t li = 0;   // next unconsumed loop in Ls (pc order)
        const DivLoop* active_direct_wave_loop = nullptr;
        uint32_t* active_direct_wave_continue = nullptr;
        // `cont` = the pc control flows to after `hi` — the enclosing construct's merge chain. An
        // if/else whose merge ESCAPES the current region (the shared-outer-merge cascade) is legal
        // only when it targets exactly this continuation (no skipped instructions); anything else
        // rejects, fail-visible. Fragment EXEC/VCC branches vote over the enforced wave64; compute
        // uses its exact guest-wave reduction paths. Either may enter/leave with EXEC narrowed, and
        // EXEC is phi'd across the merge like any other value.
        std::function<bool(uint32_t, uint32_t, uint32_t)> emit_structured;
        // Emit one EXEC/VCC-exit loop (#273/#615) as structured SPIR-V. Same block shape as
        // the counted-loop path (hdr -> chk -> body -> cont -> hdr, exit chk->merge) with three
        // differences: (1) fragment votes the complete wave's EXEC/VCC after the header recompute
        // (other guarded stages consume this lane's bool); (2) the body is
        // emitted RECURSIVELY (nested forward-execz if regions live inside it); (3) per-lane MASKS
        // (VCC/EXEC/saved sreg_bool pairs) are loop-carried too, so each gets a header phi. At the
        // merge, a register written in the condition region keeps its exit-iteration (chk) value —
        // it dominates the merge — while body-written state takes the header phi (its value when the
        // exiting check ran); masks CREATED inside the loop are dropped at the merge (an SSA id from
        // inside the body does not dominate it — a later read then rejects loudly instead of
        // emitting invalid SPIR-V). Execution tests cover both wave-vote outcomes and direct breaks.
        std::function<bool(const DivLoop&)> emit_divloop = [&](const DivLoop& L) -> bool {
            const bool entry_exec_narrowed = rs.exec_narrowed;
            std::set<int> cv, cs, condv, conds, scalar_may_writes;
            loop_written_regs(ins, L.header_pc, L.backedge_pc, cv, cs);
            loop_written_regs(ins, L.header_pc, L.exit_branch_pc, condv, conds);
            loop_scalar_may_writes(ins, L.header_pc, L.backedge_pc, scalar_may_writes);
            for (int reg : rs.sreg_bool_b32)
                if (scalar_may_writes.contains(reg)) return false;
            if (b.allow_b32_masks &&
                has_unpersisted_b32_mask_lifetime(
                    ins, L.header_pc, L.backedge_pc, rs))
                return false;
            const uint32_t preheader = b.cur_block;
            const uint32_t hdr = b.id(), chk = b.id(), body = b.id(), cont = b.id(), merge = b.id();
            b.emit_branch(hdr); b.emit_label(hdr);
            struct PhiRec { int reg; int dom; uint32_t phi; size_t patch; };  // dom: 0=vreg,1=sreg,2=scc,3=vcc,4=exec,5=mask
            std::vector<PhiRec> phis;
            for (int r : cv) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, vget(r), preheader, p); rs.vreg[r] = ph; phis.push_back({r, 0, ph, p}); }
            for (int r : cs) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, sget(r), preheader, p); rs.sreg[r] = ph; phis.push_back({r, 1, ph, p}); }
            // A poisoned (0) SCC live-in degrades to bfalse (invalid as an SSA phi input; dead in
            // practice — the loop shapes re-produce SCC before any read).
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.scc ? rs.scc : b.bfalse(), preheader, p); rs.scc = ph; phis.push_back({0, 2, ph, p}); }
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.vcc, preheader, p); rs.vcc = ph; phis.push_back({0, 3, ph, p}); }
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.exec, preheader, p); rs.exec = ph; phis.push_back({0, 4, ph, p}); }
            std::vector<int> mask_keys;                        // saved masks live at entry: loop-carried bools
            for (auto& kv : rs.sreg_bool) mask_keys.push_back(kv.first);
            std::sort(mask_keys.begin(), mask_keys.end());     // deterministic emission order
            for (int k : mask_keys) { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.sreg_bool[k], preheader, p); rs.sreg_bool[k] = ph; phis.push_back({k, 5, ph, p}); }
            invalidate_loop_descriptor_provenance(rs, scalar_may_writes);
            b.emit_loopmerge(merge, cont); b.emit_branch(chk); b.emit_label(chk);
            // An EXEC-governed loop predicates vector writes. A VCC-governed loop branches on this
            // invocation's VCC bit but does not itself change EXEC, matching the hardware loop body.
            if (L.condition == DivLoop::Condition::Exec) rs.exec_narrowed = true;
            // Condition region [header, exit_branch): branch-free (validated), straight-line.
            if (!emit_range(L.header_pc, L.exit_branch_pc)) return false;
            // chk-end snapshots: the exit path flows THROUGH this block, so these dominate the merge.
            std::unordered_map<int, uint32_t> condv_val, conds_val;
            for (int r : condv) condv_val[r] = vget(r);
            for (int r : conds) conds_val[r] = sget(r);
            const uint32_t exec_chk = rs.exec, vcc_chk = rs.vcc, scc_chk = rs.scc;
            const std::unordered_map<int, uint32_t> bool_chk = rs.sreg_bool;
            uint32_t loop_cond = L.condition == DivLoop::Condition::Exec ? rs.exec : rs.vcc;
            // EXECZ/VCCZ are scalar wave decisions. Keeping every fragment invocation in the loop
            // until the complete guest wave becomes empty makes scalar state and nested wave votes
            // exact; vector writes remain predicated by the per-lane EXEC bool.
            if (b.is_fragment) loop_cond = b.fragment_wave_any(loop_cond);
            const uint32_t chk_end = b.cur_block;
            b.emit_condbranch(loop_cond, body, merge);         // execz/vccz exit: continue while bit set
            while (idx < ins.size() && ins[idx].pc < L.exit_branch_pc) ++idx;
            if (idx < ins.size() && ins[idx].pc == L.exit_branch_pc) ++idx;   // consume the exit branch
            b.emit_label(body);
            // Body (recursive: nested if regions + breaks); ends just before the back-edge.
            uint32_t direct_wave_continue = b.btrue();
            const DivLoop* prior_direct_wave_loop = active_direct_wave_loop;
            uint32_t* prior_direct_wave_continue = active_direct_wave_continue;
            active_direct_wave_loop = &L;
            active_direct_wave_continue = &direct_wave_continue;
            const bool body_ok = emit_structured(
                L.exit_branch_pc + 1, L.backedge_pc, L.backedge_pc);
            active_direct_wave_loop = prior_direct_wave_loop;
            active_direct_wave_continue = prior_direct_wave_continue;
            if (!body_ok) return false;
            const bool body_exec_narrowed = rs.exec_narrowed;
            if (idx < ins.size() && ins[idx].pc == L.backedge_pc) ++idx;      // consume the back-edge
            const uint32_t body_end = b.cur_block;
            // An unconditional back-edge can re-enable EXEC at the header. When an interior EXECZ
            // targets the loop exit, send the inactive invocation straight to the loop merge; the
            // ordinary path still reaches the continue block and patches the loop-carried phis.
            uint32_t continue_condition = direct_wave_continue;
            if (L.direct_exec_breaks) continue_condition = b.land(continue_condition, rs.exec);
            if (L.direct_exec_breaks || L.direct_wave_breaks)
                b.emit_condbranch(continue_condition, cont, merge);
            else
                b.emit_branch(cont);
            b.emit_label(cont);
            for (auto& pr : phis) {
                uint32_t nv = pr.dom == 0 ? vget(pr.reg)
                            : pr.dom == 1 ? sget(pr.reg)
                            : pr.dom == 2 ? rs.scc
                            : pr.dom == 3 ? rs.vcc
                            : pr.dom == 4 ? rs.exec
                            : (rs.sreg_bool.count(pr.reg) ? rs.sreg_bool[pr.reg] : pr.phi);
                if (!nv && pr.dom == 2) nv = b.bfalse();   // poisoned SCC back-edge value
                b.patch_phi(pr.patch, nv, cont);
            }
            b.emit_branch(hdr);
            b.emit_label(merge);
            for (auto& pr : phis) {
                uint32_t chk_value = pr.dom == 0 ? (condv.count(pr.reg) ? condv_val[pr.reg] : pr.phi)
                                   : pr.dom == 1 ? (conds.count(pr.reg) ? conds_val[pr.reg] : pr.phi)
                                   : pr.dom == 2 ? (scc_chk ? scc_chk : b.bfalse())
                                   : pr.dom == 3 ? vcc_chk
                                   : pr.dom == 4 ? exec_chk
                                   : (bool_chk.count(pr.reg) ? bool_chk.at(pr.reg) : pr.phi);
                uint32_t body_value = pr.dom == 0 ? vget(pr.reg)
                                    : pr.dom == 1 ? sget(pr.reg)
                                    : pr.dom == 2 ? (rs.scc ? rs.scc : b.bfalse())
                                    : pr.dom == 3 ? rs.vcc
                                    : pr.dom == 4 ? rs.exec
                                    : (rs.sreg_bool.count(pr.reg) ? rs.sreg_bool[pr.reg] : pr.phi);
                const uint32_t merged = (L.direct_exec_breaks || L.direct_wave_breaks) &&
                                                chk_value != body_value
                    ? b.emit_phi_2way(pr.dom <= 1 ? b.t_u32 : b.t_bool,
                                      chk_value, chk_end, body_value, body_end)
                    : chk_value;
                if (pr.dom == 0)      rs.vreg[pr.reg] = merged;
                else if (pr.dom == 1) rs.sreg[pr.reg] = merged;
                else if (pr.dom == 2) rs.scc = merged;
                else if (pr.dom == 3) rs.vcc = merged;
                else if (pr.dom == 4) rs.exec = merged;
                else                  rs.sreg_bool[pr.reg] = merged;
            }
            // Masks CREATED inside the loop: their ids do not dominate the merge — drop them.
            for (auto it = rs.sreg_bool.begin(); it != rs.sreg_bool.end();) {
                if (!std::binary_search(mask_keys.begin(), mask_keys.end(), it->first)) {
                    rs.sreg_bool_narrowed.erase(it->first);
                    rs.sreg_bool_b32.erase(it->first);
                    it = rs.sreg_bool.erase(it);
                } else ++it;
            }
            // An execz exit leaves this lane inactive until the compiled restore. A vccz exit leaves
            // EXEC unchanged; preserve any narrowing that existed on entry or occurred in the body.
            rs.exec_narrowed = L.condition == DivLoop::Condition::Exec
                ? true : (entry_exec_narrowed || body_exec_narrowed);
            return true;
        };
        emit_structured =
            [&](uint32_t lo, uint32_t hi, uint32_t cont) -> bool {
            for (;;) {
                const uint32_t next_br = (bi < Fs.size() && Fs[bi].branch_pc < hi) ? Fs[bi].branch_pc : hi;
                const uint32_t next_lp = (li < Ls.size() && Ls[li].header_pc < hi) ? Ls[li].header_pc : hi;
                if (next_lp < next_br) {                     // a loop begins before the next if
                    if (!emit_range(lo, next_lp)) return false;
                    const DivLoop& L = Ls[li++];
                    if (!emit_divloop(L)) return false;
                    lo = L.exit_pc;
                    continue;
                }
                if (!emit_range(lo, next_br)) return false;
                if (next_br == hi) return true;
                const ForwardIf F = Fs[bi++];
                if (idx < ins.size() && ins[idx].pc == F.branch_pc) ++idx;   // skip the branch itself
                // scc0/vccz/execz: branch (skip block) taken when the flag==0 → the block runs when
                // flag!=0; scc1/vccnz are the inverse. Compute and fragment reduce per-invocation
                // mask bits to the architecture's wave-wide "any lane active" predicate.
                // A poisoned SCC (0: last written by a 64-bit mask op) cannot condition a real
                // structured if — reject (the ISA-audit #879 stale-SCC consumer).
                if (!F.on_exec && !F.on_vcc && !rs.scc) return false;
                // Compute VCC/EXEC branches normally return through the exact guest-wave dispatcher.
                // The narrow structured-wave path above accepts only top-level vote sites, where all
                // workgroup invocations may participate in the scratch barriers uniformly.
                if (b.is_compute && (F.on_exec || F.on_vcc) && !structured_compute_wave_cfg)
                    return false;
                uint32_t cond_reg = F.on_exec ? rs.exec : (F.on_vcc ? rs.vcc : rs.scc);
                if (b.is_compute && (F.on_exec || F.on_vcc))
                    cond_reg = b.native_subgroup_size
                        ? b.native_wave_any(cond_reg)
                        : b.guest_wave_any(cond_reg);
                else if (b.is_fragment && (F.on_exec || F.on_vcc))
                    cond_reg = b.fragment_wave_any(cond_reg);
                uint32_t exec_cond = F.on_scc0 ? cond_reg : b.bsel(cond_reg, b.bfalse(), b.btrue());
                if (active_direct_wave_loop && active_direct_wave_continue &&
                    active_direct_wave_loop->direct_wave_breaks && F.on_vcc &&
                    std::find(active_direct_wave_loop->break_pcs.begin(),
                              active_direct_wave_loop->break_pcs.end(), F.branch_pc) !=
                        active_direct_wave_loop->break_pcs.end())
                    *active_direct_wave_continue =
                        b.land(*active_direct_wave_continue, exec_cond);
                const uint32_t preblock = b.cur_block;      // block holding the OpBranchConditional
                if (!F.has_else) {
                    std::set<int> ifv, ifs;
                    loop_written_regs(ins, F.branch_pc + 1, F.target_pc, ifv, ifs);
                    std::unordered_map<int,uint32_t> pre_v, pre_s;
                    for (int r : ifv) pre_v[r] = vget(r);
                    for (int r : ifs) pre_s[r] = sget(r);
                    uint32_t pre_scc = rs.scc, pre_vcc = rs.vcc, pre_exec = rs.exec;
                    const bool pre_narrowed = rs.exec_narrowed;
                    const std::unordered_map<int,uint32_t> pre_bool = rs.sreg_bool;   // mask-domain snapshot
                    const auto pre_bool_b32 = rs.sreg_bool_b32;
                    uint32_t thenL = b.id(), mergeL = b.id();
                    b.emit_selmerge(mergeL); b.emit_condbranch(exec_cond, thenL, mergeL);
                    b.emit_label(thenL);
                    if (!emit_structured(F.branch_pc + 1, F.target_pc, F.target_pc)) return false;
                    const uint32_t thenEnd = b.cur_block;   // last block of the then-body (nested ifs move it)
                    std::unordered_map<int,uint32_t> then_v, then_s;
                    for (int r : ifv) then_v[r] = vget(r);
                    for (int r : ifs) then_s[r] = sget(r);
                    uint32_t then_scc = rs.scc, then_vcc = rs.vcc, then_exec = rs.exec;
                    const bool then_narrowed = rs.exec_narrowed;
                    // The skipped edge retains the entry-time physical-word lifetime. A B32 mask
                    // created or invalidated only in the taken arm therefore needs a validity phi,
                    // which this narrow merge does not represent. Reject rather than attach the
                    // taken arm's marker to the synthesized bool value on both paths.
                    std::set<int> dead_b32_at_merge;
                    std::set_symmetric_difference(
                        rs.sreg_bool_b32.begin(), rs.sreg_bool_b32.end(),
                        pre_bool_b32.begin(), pre_bool_b32.end(),
                        std::inserter(dead_b32_at_merge, dead_b32_at_merge.end()));
                    const bool dead_domain_difference =
                        std::all_of(dead_b32_at_merge.begin(), dead_b32_at_merge.end(),
                                    [&](int reg) {
                                        return sgpr_dead_at_merge(ins, F.target_pc, reg);
                                    });
                    if (!dead_domain_difference) {
                        if (getenv("PROSPER_DBG"))
                            std::fprintf(stderr,
                                         "[compute-struct-reject] live b32 mask domain differs "
                                         "across branch pc=%u merge=%u\n",
                                         F.branch_pc, F.target_pc);
                        return false;
                    }
                    b.emit_branch(mergeL); b.emit_label(mergeL);
                    for (int r : ifv) rs.vreg[r] = b.emit_phi_2way(b.t_u32,  pre_v[r], preblock, then_v[r], thenEnd);
                    for (int r : ifs) rs.sreg[r] = b.emit_phi_2way(b.t_u32,  pre_s[r], preblock, then_s[r], thenEnd);
                    if (then_scc != pre_scc)   // poisoned (0) inputs degrade to bfalse across the merge
                        rs.scc = b.emit_phi_2way(b.t_bool, pre_scc ? pre_scc : b.bfalse(), preblock,
                                                 then_scc ? then_scc : b.bfalse(), thenEnd);
                    if (then_vcc != pre_vcc) rs.vcc = b.emit_phi_2way(b.t_bool, pre_vcc, preblock, then_vcc, thenEnd);
                    // EXEC changed inside the arm (saveexec / v_cmpx / restore): merge it like any value.
                    // Narrowed-ness is sticky (either edge narrowed → post-merge writes stay predicated).
                    if (then_exec != pre_exec) rs.exec = b.emit_phi_2way(b.t_bool, pre_exec, preblock, then_exec, thenEnd);
                    rs.exec_narrowed = pre_narrowed || then_narrowed;
                    // A saved MASK (sreg_bool) created or changed inside the block must dominate both
                    // merge predecessors. A newly-created per-lane mask is false on the skipped edge.
                    for (auto& kv : rs.sreg_bool) {
                        auto p = pre_bool.find(kv.first);
                        const uint32_t before = p != pre_bool.end() ? p->second : b.bfalse();
                        if (before != kv.second) {
                            kv.second = b.emit_phi_2way(b.t_bool, before, preblock, kv.second, thenEnd);
                            rs.sreg_bool_narrowed[kv.first] = true;   // conservative: provenance now mixed
                        }
                    }
                    // A differing physical-word domain needs no validity phi when that word is
                    // provably overwritten before every post-merge read. Drop its stale typed view
                    // on both synthesized paths; the later defining instruction recreates the
                    // appropriate scalar or mask lifetime.
                    for (int reg : dead_b32_at_merge) {
                        rs.sreg_bool_b32.erase(reg);
                        rs.sreg_bool.erase(reg);
                        rs.sreg_bool_narrowed.erase(reg);
                        if (reg == 106) rs.vcc = 0;
                    }
                    lo = F.target_pc;   // continue after the merge (further sequential ifs handled here)
                } else {
                    // IF/ELSE: then = [branch_pc+1, sb_pc) (its s_branch terminator is consumed);
                    // else = [target_pc, merge). A merge escaping this region must be exactly the
                    // enclosing continuation `cont` (the cascade shape) — the else-arm then runs to
                    // `hi` and the merge coincides with the region end.
                    uint32_t else_hi = F.merge_pc;
                    if (F.merge_pc >= hi) {
                        if (F.merge_pc != cont && F.merge_pc != hi) return false;
                        else_hi = hi;
                    }
                    const RegState pre = rs;                // FULL snapshot: the else-arm re-runs from it
                    std::set<int> wv, ws;                   // regs written in EITHER arm
                    loop_written_regs(ins, F.branch_pc + 1, F.sb_pc, wv, ws);
                    loop_written_regs(ins, F.target_pc, else_hi, wv, ws);
                    uint32_t thenL = b.id(), elseL = b.id(), mergeL = b.id();
                    b.emit_selmerge(mergeL); b.emit_condbranch(exec_cond, thenL, elseL);
                    b.emit_label(thenL);
                    if (!emit_structured(F.branch_pc + 1, F.sb_pc, F.merge_pc)) return false;
                    if (idx < ins.size() && ins[idx].pc == F.sb_pc) ++idx;   // consume the arm's s_branch
                    const uint32_t thenEnd = b.cur_block;
                    std::unordered_map<int,uint32_t> then_v, then_s;
                    for (int r : wv) then_v[r] = vget(r);
                    for (int r : ws) then_s[r] = sget(r);
                    uint32_t then_scc = rs.scc, then_vcc = rs.vcc, then_exec = rs.exec;
                    const bool then_narrowed = rs.exec_narrowed;
                    const std::unordered_map<int,uint32_t> then_bool = rs.sreg_bool;
                    const auto then_bool_b32 = rs.sreg_bool_b32;
                    const auto then_written = rs.sreg_written;
                    b.emit_branch(mergeL);
                    rs = pre;                               // else-arm starts from the pre-branch state
                    b.emit_label(elseL);
                    if (!emit_structured(F.target_pc, else_hi, F.merge_pc)) return false;
                    const uint32_t elseEnd = b.cur_block;
                    b.emit_branch(mergeL); b.emit_label(mergeL);
                    for (int r : wv) { uint32_t ev = vget(r);
                        if (then_v[r] != ev) rs.vreg[r] = b.emit_phi_2way(b.t_u32, then_v[r], thenEnd, ev, elseEnd); }
                    for (int r : ws) { uint32_t es = sget(r);
                        if (then_s[r] != es) rs.sreg[r] = b.emit_phi_2way(b.t_u32, then_s[r], thenEnd, es, elseEnd); }
                    if (then_scc != rs.scc)   // poisoned (0) inputs degrade to bfalse across the merge
                        rs.scc = b.emit_phi_2way(b.t_bool, then_scc ? then_scc : b.bfalse(), thenEnd,
                                                 rs.scc ? rs.scc : b.bfalse(), elseEnd);
                    if (then_vcc != rs.vcc) rs.vcc = b.emit_phi_2way(b.t_bool, then_vcc, thenEnd, rs.vcc, elseEnd);
                    if (then_exec != rs.exec) rs.exec = b.emit_phi_2way(b.t_bool, then_exec, thenEnd, rs.exec, elseEnd);
                    rs.exec_narrowed = then_narrowed || rs.exec_narrowed;
                    rs.sreg_written.insert(then_written.begin(), then_written.end());
                    for (int reg : then_written) rs.sreg_input.erase(reg);
                    if (then_bool_b32 != rs.sreg_bool_b32) {
                        if (getenv("PROSPER_DBG"))
                            std::fprintf(stderr,
                                         "[compute-struct-reject] b32 mask domain differs across "
                                         "if/else pc=%u merge=%u\n",
                                         F.branch_pc, F.merge_pc);
                        return false;
                    }
                    // Merge the UNION of mask keys. A mask created in only one arm is false in the
                    // other arm; leaving that arm-local SSA id live after the merge is invalid SPIR-V.
                    std::set<int> bool_keys;
                    for (const auto& kv : then_bool) bool_keys.insert(kv.first);
                    for (const auto& kv : rs.sreg_bool) bool_keys.insert(kv.first);
                    for (int key : bool_keys) {
                        auto t = then_bool.find(key);
                        auto e = rs.sreg_bool.find(key);
                        const uint32_t tv = t != then_bool.end() ? t->second : b.bfalse();
                        const uint32_t ev = e != rs.sreg_bool.end() ? e->second : b.bfalse();
                        rs.sreg_bool[key] = tv == ev ? tv
                            : b.emit_phi_2way(b.t_bool, tv, thenEnd, ev, elseEnd);
                        if (tv != ev) rs.sreg_bool_narrowed[key] = true;
                    }
                    lo = else_hi;   // continue after the merge (== hi for the escaping-cascade shape)
                }
            }
        };
        // Every scalar branch and loop condition is subgroup-uniform in the fragment shell (SCC is
        // scalar already; EXECZ/VCCZ use fragment_wave_any), so native wave operations in any
        // structured arm observe the complete guest wave.
        wave_ok = b.is_fragment;
        if (!emit_structured(0, UINT32_MAX, UINT32_MAX)) return false;
    }
    (void)safe_branches;
    return true;
}

std::vector<uint32_t> recompile_valu(const uint32_t* code, size_t dwords,
                                     uint32_t num_inputs, uint32_t out_vgpr,
                                     const ShaderResourceTable* rt, uint32_t lds_bytes) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    // This shell publishes register state after the structured region, so both terminal arms can
    // converge in the host SPIR-V without inventing further guest execution. Graphics exports keep
    // separate side-effect bookkeeping and remain on the conservative reject path for this shape.
    (void)extend_terminating_if_else(code, dwords, ins);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);
    SpirvCompute b;
    // Size the LDS array from the shader's real allocation when known (#130): bytes -> dwords, at
    // least the ds ops need, clamped to the RDNA2 64 KB (16384-dword) max. 0 keeps the 16 KB default.
    if (lds_bytes) {
        uint32_t dw = (lds_bytes + 3) / 4;
        b.lds_dwords = dw > 16384u ? 16384u : (dw ? dw : 1u);
    }
    b.begin(num_inputs ? num_inputs : 1, rt);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    for (uint32_t k = 0; k < num_inputs; k++) rs.vreg[(int)k] = b.load_input(k);
    // Compute kernels have no EXP output; reject if one appears.
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true, /*allow_smem*/true,
                   [](RegState&, const Rdna2Inst&){ return false; }, code, dwords)) return {};
    auto it = rs.vreg.find((int)out_vgpr);
    uint32_t outbits = it == rs.vreg.end() ? b.uconst(0) : it->second;
    // If EXEC is still narrowed (a v_cmpx with no restore), masked-off lanes keep the output slot's prior
    // value; if it was restored to all-lanes-on, every lane stores.
    if (!rs.exec_narrowed) b.store_output(outbits);
    else                   b.store_output_pred(outbits, rs.exec);
    return b.finish();
}

std::vector<uint32_t> recompile_compute(const uint32_t* code, size_t dwords,
                                        const ShaderResourceTable* rt,
                                        const ComputeShaderConfig& config) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    // See recompile_valu: compute has no branch-external EXP state, so the common host-shell merge is
    // only a place to finish the invocation after either guest arm has terminated.
    (void)extend_terminating_if_else(code, dwords, ins);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);
    SpirvCompute b;
    if (config.lds_bytes) {
        uint32_t dw = (config.lds_bytes + 3) / 4;
        b.lds_dwords = std::min(16384u, std::max(1u, dw));
    }
    const uint32_t local_x = std::max(1u, config.local_x);
    const uint32_t local_y = std::max(1u, config.local_y);
    const uint32_t local_z = std::max(1u, config.local_z);
    const uint32_t wave_size = config.wave_size == 32 ? 32u : 64u;
    const uint64_t local_count = static_cast<uint64_t>(local_x) * local_y * local_z;
    b.native_subgroup_size = config.native_subgroup_size == wave_size &&
        local_count <= UINT32_MAX && local_count % wave_size == 0 ? wave_size : 0u;
    b.native_storage_format_support = config.native_storage_format_support;
    b.packed_r11_storage = config.packed_r11_storage;
    b.begin(1, rt, local_x, local_y, local_z, wave_size,
            static_cast<uint32_t>(config.user_sgprs.size()));
    b.allow_b32_masks = wave_size == 32;
    b.declare_guest_scratch(scratch);
    const bool has_partial_workgroup = config.threads_x % local_x != 0 ||
                                       config.threads_y % local_y != 0 ||
                                       config.threads_z % local_z != 0;
    if (config.exact_thread_extent && has_partial_workgroup)
        b.guard_invocation_extent(config.threads_x, config.threads_y, config.threads_z);

    RegState rs;
    rs.vcc = b.bfalse();
    rs.scc = b.bfalse();
    rs.exec = b.btrue();
    // Inline descriptors are represented by the resource table, not scalar SSA values. Leaving
    // their SGPR range absent also preserves the existing direct-provenance rule: a format MUBUF may
    // fall back to by_sgpr_base only while its SRSRC has not been overwritten by shader code.
    std::set<uint32_t> descriptor_sgprs;
    if (rt) {
        for (const auto& resource : rt->resources) {
            if (resource.srt_offset != 0xFFFFFFFFu || resource.sgpr_base == 0xFFFFFFFFu) continue;
            uint32_t words = (resource.cls == ResourceClass::Texture ||
                              resource.cls == ResourceClass::StorageImage) ? 8u : 4u;
            for (uint32_t word = 0; word < words; word++)
                descriptor_sgprs.insert(resource.sgpr_base + word);
        }
    }
    for (size_t i = 0; i < config.user_sgprs.size(); i++) {
        const uint32_t value = b.load_push_constant(static_cast<uint32_t>(i));
        if (descriptor_sgprs.count(static_cast<uint32_t>(i)))
            rs.sreg_input[static_cast<int>(i)] = value;
        else
            rs.sreg[static_cast<int>(i)] = value;
    }

    rs.vreg[0] = b.localid_comp[0];
    if (config.tidig_comp_cnt >= 1) rs.vreg[1] = b.localid_comp[1];
    if (config.tidig_comp_cnt >= 2) rs.vreg[2] = b.localid_comp[2];

    int system_sgpr = static_cast<int>(config.user_sgprs.size());
    if (config.tgid_x_en) rs.sreg[system_sgpr++] = b.groupid[0];
    if (config.tgid_y_en) rs.sreg[system_sgpr++] = b.groupid[1];
    if (config.tgid_z_en) rs.sreg[system_sgpr++] = b.groupid[2];
    if (config.tg_size_en)
        rs.sreg[system_sgpr] = b.uconst(local_x * local_y * local_z);

    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                   /*allow_smem*/true, [](RegState&, const Rdna2Inst&) { return false; }, code, dwords))
        return {};
    // The entry guard is intentionally divergent only in the final partial workgroup. Vulkan requires
    // every workgroup invocation to participate uniformly in OpControlBarrier, including barriers the
    // recompiler synthesizes for wave operations. Reject this uncommon combination instead of emitting
    // a module that could deadlock or observe undefined workgroup-memory behavior.
    if (has_partial_workgroup && b.uses_barrier) return {};
    return b.finish();
}

bool compute_shader_prefers_native_multiwave(const std::vector<Rdna2Inst>& ins,
                                             const uint32_t* code, size_t dwords) {
    bool low = false;
    bool high = false;
    bool guest_barrier = false;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3) {
            low |= in.opcode == 0x365;
            high |= in.opcode == 0x366;
        }
        guest_barrier |= in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a;
        if (low && high) return true;
    }
    if (!guest_barrier || !code) return false;

    // The phase splitter's shared proof is stronger than the whole-stream CFG shape: branches and
    // loops inside one barrier-free phase do not make the outer uniform guard or its barriers
    // divergent. Such kernels need an exact native subgroup for nested guest-wave votes.
    if (analyze_barrier_phased_compute(ins).found) return true;

    // Mirror the conservative, acyclic subset of emit_body's structured-compute admission. Counting
    // raw VCC/EXEC opcodes is insufficient: kill-mask branches may be safely linearized, loop exits
    // are owned by another emitter, and rejected CFGs never reach guest_wave_any. Requiring the same
    // accepted ForwardIf regions proves that portable lowering really emits two scratch barriers per
    // counted vote and that exact-subgroup lowering removes them. Loops deliberately stay behind the
    // explicit experiment until their additional compute guards are shared with this analysis.
    auto safe = safe_execz_branches(ins);
    for (uint32_t pc : waterfall_branches(ins)) safe.insert(pc);
    const std::vector<DivLoop> loops = detect_divergent_loops(ins, safe, /*fragment*/false);
    if (!loops.empty()) return false;

    bool rejected = false;
    const std::vector<ForwardIf> branches = detect_forward_ifs(
        ins, /*allow_vcc*/false, code, dwords, &safe, nullptr, &rejected,
        /*compute_wave_branches*/true);
    if (rejected) return false;

    auto top_level_pc = [&](uint32_t pc) {
        for (const ForwardIf& parent : branches) {
            const uint32_t parent_end = parent.has_else ? parent.merge_pc : parent.target_pc;
            if (parent.branch_pc < pc && pc < parent_end) return false;
        }
        return true;
    };
    const bool barriers_are_top_level = std::all_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
        return in.fmt != Rdna2Format::SOPP || in.opcode != 0x0a || top_level_pc(in.pc);
    });
    if (!barriers_are_top_level) return false;

    size_t structured_wave_votes = 0;
    for (const ForwardIf& branch : branches) {
        if (!branch.on_exec && !branch.on_vcc) continue;
        if (!top_level_pc(branch.branch_pc)) return false;
        ++structured_wave_votes;
    }
    // Four proven scratch-emulated votes keep the default narrower than the all-multi-wave experiment.
    return structured_wave_votes >= 4;
}

bool compute_shader_prefers_native_multiwave(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    return compute_shader_prefers_native_multiwave(ins, code, dwords);
}

// See FlatLoadInfo / analyze_flat_loads in the header (#1171). A `base + offset` flat address VGPR pair
// v[N:N+1] is computed as vN = add(base_lo_sgpr, offset_lo) [carry-out] and v(N+1) =
// add(base_hi_sgpr, offset_hi, carry) [carry-in], where s[base_lo:base_hi] are consecutive user SGPRs
// holding a 64-bit guest pointer (the low kernel-arg pointer dword feeds the low address dword). We
// identify the base by the NEAREST prior definition of each address dword being an integer add that
// reads a user-range SGPR, and require the low dword to add the SGPR one below the high dword's.
// Anything else (a non-add producer, a non-user SGPR, a store/atomic/LDS/global-with-saddr form) leaves
// the load unresolved so the caller keeps failing visibly. flat_access_info lives in this TU's
// anonymous namespace but is visible here (internal linkage is still TU-wide).
// Resolution is over LINEAR program order (not CFG-aware). That is exact for the target decode kernels
// (the address adds sit in the same block immediately before the load) and the base is a loop-invariant
// kernel-arg pointer, so the resolved base is stable across loop iterations; the executor's
// guest_readable_mapping_containing validation is the runtime backstop against a bogus base.
FlatLoadAnalysis analyze_flat_loads(const uint32_t* code, size_t dwords, uint32_t user_sgpr_count) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    FlatLoadAnalysis out;

    auto writes_vgpr = [](const Rdna2Inst& d, uint32_t reg) {
        return d.dst.kind == OperandKind::VGPR && static_cast<uint32_t>(d.dst.value) == reg;
    };
    // If `d` is an integer add that could produce a 64-bit-pointer dword, return the user-range SGPR it
    // adds (else -1). Recognizes the add_co / add_co_ci / add_nc forms a compiler emits for pointer
    // arithmetic: VOP2 0x25 (add_nc) / 0x28 (add_co_ci), VOP3 0x30F (add_co) / 0x125 (add_nc) / 0x128
    // (add_co_ci).
    auto add_user_sgpr = [&](const Rdna2Inst& d) -> int32_t {
        const bool is_add =
            (d.fmt == Rdna2Format::VOP2 && (d.opcode == 0x25 || d.opcode == 0x28)) ||
            (d.fmt == Rdna2Format::VOP3 &&
             (d.opcode == 0x30F || d.opcode == 0x125 || d.opcode == 0x128));
        if (!is_add) return -1;
        for (int s = 0; s < 2; ++s)
            if (d.src[s].kind == OperandKind::SGPR && d.src[s].value >= 0 &&
                static_cast<uint32_t>(d.src[s].value) < user_sgpr_count)
                return d.src[s].value;
        return -1;
    };
    // The user-range SGPR added by the nearest prior writer of `reg` (or -1 if that writer is not a
    // user-SGPR add). We trust only the immediate definition — a later non-add redefinition breaks the
    // pattern and must not be skipped over.
    auto prior_add_sgpr = [&](uint32_t reg, size_t before) -> int32_t {
        for (size_t j = before; j-- > 0;)
            if (!ins[j].is_end && writes_vgpr(ins[j], reg))
                return add_user_sgpr(ins[j]);
        return -1;
    };

    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::FLAT) continue;
        if (in.flat_segment == 1u) continue;   // scratch spill: analyze_static_scratch owns it
        out.any = true;
        const FlatAccessInfo access = flat_access_info(in.opcode);
        FlatLoadInfo info;
        info.load_pc = in.pc;
        info.vaddr_lo_reg = in.src[0].kind == OperandKind::VGPR
                                ? static_cast<uint32_t>(in.src[0].value) : 0u;
        info.dst_reg = static_cast<uint32_t>(in.dst.value);
        info.bits = access.bits;
        info.components = access.components;
        info.sign_extend = access.sign_extend;
        // Resolvable only as a plain LOAD with a VGPR address pair and a null SADDR (true flat segment).
        const bool shape = access.valid && !access.store && !in.flat_lds &&
                           in.src[0].kind == OperandKind::VGPR &&
                           in.src[1].kind == OperandKind::Special && in.src[1].value == 125;
        if (shape) {
            const uint32_t lo = info.vaddr_lo_reg, hi = lo + 1;
            const int32_t hi_base = prior_add_sgpr(hi, i);
            const int32_t lo_base = prior_add_sgpr(lo, i);
            if (hi_base > 0 && lo_base == hi_base - 1)
                info.base_sgpr = lo_base;
        }
        if (info.base_sgpr < 0) out.all_resolved = false;
        out.loads.push_back(info);
    }
    return out;
}

std::vector<uint32_t> safe_execz_branches_for_test(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const std::unordered_set<uint32_t> s = safe_execz_branches(ins);
    return std::vector<uint32_t>(s.begin(), s.end());
}

std::vector<uint32_t> mask_test_branches_for_test(const uint32_t* code, size_t dwords,
                                                  bool wave32) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const auto branches = mask_test_branches(ins, wave32);
    return std::vector<uint32_t>(branches.begin(), branches.end());
}

RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    uint32_t synthetic_branch_pc = UINT32_MAX;
    (void)extend_terminating_if_else(code, dwords, ins, nullptr, &synthetic_branch_pc);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    // A scratch builder/state so emit_alu can run; its emitted code is discarded — we only want `ok`.
    SpirvCompute b; b.begin(1);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // emit_alu is a per-instruction check and rejects control-flow branches, but the whole-stream emit_body
    // RECONSTRUCTS a counted loop and a forward uniform-if. Credit the branches emit_body consumes (loop
    // back-edge + exit, and the forward-if branch) as handled, so coverage matches what actually recompiles
    // (previously the MSAA-resolve loop shaders 031-034 were mis-flagged "blocked" at their s_cbranch_scc0).
    const CountedLoop cL = detect_counted_loop(ins);
    const std::vector<ForwardIf> cFs = detect_forward_ifs(ins, /*allow_vcc*/false, code, dwords,
                                                          nullptr, nullptr, nullptr,
                                                          /*compute_wave_branches*/true);   // matches the compute shell (#590)
    auto cf_reconstructed = [&](const Rdna2Inst& i) {
        if (cL.found && (i.pc == cL.backedge_pc || i.pc == cL.exit_branch_pc)) return true;
        for (const auto& F : cFs)
            if (i.pc == F.branch_pc || (F.has_else && i.pc == F.sb_pc)) return true;
        return false;
    };

    RecompileCoverage cov;
    for (const auto& in : ins) {
        if (in.is_end) break;
        // The rewritten first s_endpgm is compiler-only control flow. It lets the structured emitter
        // reuse its ordinary if/else path, but coverage describes decoded guest instructions and has
        // always excluded s_endpgm, so do not inflate total/ALU with this synthetic arm skip.
        if (in.pc == synthetic_branch_pc) continue;
        cov.total++;
        if (in.fmt == Rdna2Format::EXP) { cov.exports++; continue; }   // handled by the stage recompilers
        bool ok = true;
        const bool emitted = emit_alu(
            b, rs, in, ok, /*allow_exec_update*/true, &safe_branches,
            /*allow_smem*/true, /*rt*/nullptr, /*allow_wave*/true);
        if (emitted && ok) record_scalar_write(rs, in);
        bool handled = cf_reconstructed(in) || (emitted && ok);
        // Shapes the recompiler handles only in context (a resource table for MIMG sample/load/LOD/store
        // and buffer_load/store_format; a fragment stage for VINTRP). This table-less compute-shell pass
        // rejects them, so count them apart from truly-unsupported (cross-lane, etc.). Instruction-aware
        // for MIMG so deferred variants (NSA multi-dword addr; arrayed/cube/MSAA dims) are NOT overcounted
        // as recompilable — they still land in `unsupported`, matching what the recompiler actually accepts.
        auto table_dependent = [](const Rdna2Inst& i) {
            switch (i.fmt) {
                case Rdna2Format::MIMG: {
                    // Storage load/store handle 1D/2D/3D + 1D/2D_ARRAY (dims 0,1,2,4,5) and NSA; image_load
                    // also handles 2D_MSAA (dim 6). sample* go through the sampled-texture path (2D, non-NSA).
                    const bool st_dim = i.mimg_dim <= 2u || i.mimg_dim == 4u || i.mimg_dim == 5u;
                    if (i.opcode == 0x00u) return st_dim || i.mimg_dim == 6u || i.mimg_dim == 7u;   // image_load (+ 2D_MSAA[_ARRAY])
                    if (i.opcode == 0x08u) return st_dim;                       // image_store (no per-sample MSAA store)
                    if (i.opcode == 0x0fu || i.opcode == 0x11u)                // image_atomic_swap/add R32_UINT 2D
                        return i.mimg_dim == 1u && i.mimg_dmask == 1u &&
                               !i.mimg_unorm && i.len_dwords == 2u;
                    if (i.opcode == 0x0eu) return i.mimg_dim <= 2u;             // image_get_resinfo 1D/2D/3D
                    // sample*: 2D (NSA ok); plus implicit-LOD image_sample (0x20) / LOD-0 image_sample_lz
                    // (0x27) from a 3D texture; sample_b (0x25) and gather4_lz (0x47) are 2D. 2D_ARRAY (dim 5)
                    // is accepted for all sample paths and handled as its base 2D slice (array index dropped,
                    // #325) — so array-sampling draws recompile+render instead of being skipped.
                    // 0xa0 is the high-bit sibling of image_sample (0x20), lowered identically (GTA V, #1145).
                    if (i.opcode == 0x20u || i.opcode == 0x27u || i.opcode == 0xa0u)
                        return i.mimg_dim == 1u || i.mimg_dim == 2u || i.mimg_dim == 5u;
                    if (i.opcode == 0x22u) return i.mimg_dim == 1u || i.mimg_dim == 5u;
                    if (i.opcode == 0x2fu)
                        return i.mimg_dim == 1u || i.mimg_dim == 3u || i.mimg_dim == 5u;
                    if (i.opcode == 0x24u || i.opcode == 0x25u || i.opcode == 0x47u) return i.mimg_dim == 1u || i.mimg_dim == 5u;
                    return false;
                }
                case Rdna2Format::MUBUF:  return i.opcode <= 0x07u ||                    // load/store_format_*
                                                 (i.opcode >= 0x0Cu && i.opcode <= 0x0Fu);  // load_dword/x2/x4/x3 (need the V#)
                case Rdna2Format::MTBUF:  return i.opcode <= 0x07u && !i.mtbuf_tfe;
                // Wide scalar loads are descriptor-table fetches. A real resource table lets the
                // emitter preserve their provenance without reading a fallback buffer, including the
                // register-offset bindless form; the table-less coverage shell must not call that a
                // newly unsupported instruction.
                case Rdna2Format::SMEM:   return i.opcode == 0x02u || i.opcode == 0x03u;
                case Rdna2Format::VINTRP: return true;               // handled in the fragment shell
                default: return false;
            }
        };
        if (handled) { cov.alu++; }
        else if (table_dependent(in)) { cov.table_dependent++; }
        else {
            cov.unsupported++;
            if (cov.first_bad_fmt < 0) {
                cov.first_bad_fmt = (int)in.fmt;
                cov.first_bad_op = in.opcode;
                cov.first_bad_pc = in.pc;
            }
        }
    }
    return cov;
}

static uint64_t shader_program_hash(const uint32_t* code, size_t dwords) {
    uint64_t hash = 1469598103934665603ull;
    const auto* bytes = reinterpret_cast<const uint8_t*>(code);
    for (size_t i = 0; i < dwords * sizeof(uint32_t); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint32_t effective_fragment_wave_size(uint32_t requested_wave_size,
                                             size_t program_dwords,
                                             uint64_t program_hash) {
    if (requested_wave_size != 32 && requested_wave_size != 64) return 0;
    // Compatibility for the one captured Astro fragment whose older producer omitted
    // SPI_PS_IN_CONTROL.PS_W32_EN. Its complete byte identity proves the same Wave32 contract; that
    // contract must select both one-word mask semantics and a 32-lane native subgroup.
    const bool legacy_wave32 = requested_wave_size == 64 && program_dwords == 3142 &&
        program_hash == 0x616dd4c0b241fbb1ull;
    return requested_wave_size == 32 || legacy_wave32 ? 32u : 64u;
}

uint32_t fragment_effective_wave_size_for_test(uint32_t requested_wave_size,
                                               size_t program_dwords,
                                               uint64_t program_hash) {
    return effective_fragment_wave_size(requested_wave_size, program_dwords, program_hash);
}

uint32_t fragment_color_export_mask(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    uint32_t packed = 0;
    std::array<bool, 2> realized{};
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::EXP || in.exp_target >= realized.size() ||
            realized[in.exp_target] || in.exp_en == 0)
            continue;
        packed |= (in.exp_en & 0xFu) << (in.exp_target * 4u);
        realized[in.exp_target] = true;
    }
    return packed;
}

static std::vector<uint32_t> recompile_fragment_impl(
        const uint32_t* code, size_t dwords,
        const ShaderResourceTable* rt,
        const PixelSystemInputMapping* system_inputs,
        uint32_t pcrel_dispatch_target,
        const FragmentInterpolationLayout* interpolation,
        uint32_t wave_size) {
    if (wave_size != 32 && wave_size != 64) return {};
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    if (pcrel_dispatch_target != UINT32_MAX) {
        const PcrelDispatchInfo dispatch = rdna2_pcrel_dispatch_info(code, dwords);
        if (!specialize_pcrel_dispatch(ins, dispatch, pcrel_dispatch_target)) {
            if (getenv("PROSPER_DBG"))
                fprintf(stderr, "[recompile-reject] pcrel dispatch specialization target=%u\n",
                        pcrel_dispatch_target);
            return {};
        }
    }
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    // Preserve hardware target locations. MRT0 and MRT1 are backed by real Vulkan attachments; later
    // targets remain unsupported and must never be silently remapped to location 0 (#635).
    uint32_t color_mask = 0;
    bool has_null_export = false;
    bool has_depth_export = false;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::EXP) continue;
        if (in.exp_target < 2) color_mask |= 1u << in.exp_target;
        else if (in.exp_target == 8 && !in.exp_compr && (in.exp_en & 1u))
            has_depth_export = true;
        else if (in.exp_target == 9) has_null_export = true;
    }
    // A NULL export is a real fragment-shader terminator. Depth/stencil-only draws use it after
    // narrowing EXEC to the surviving samples, so the module intentionally has no color outputs.
    // Keep other unsupported MRT-only programs fail-visible instead of accepting every no-color PS.
    if (!color_mask && !has_null_export && !has_depth_export) {
        if (getenv("PROSPER_DBG")) {
            fprintf(stderr,
                    "[recompile-reject] fragment has no supported export dwords=%zu ins=%zu "
                    "first=%08x last=%08x",
                    dwords, ins.size(), dwords ? code[0] : 0u,
                    dwords ? code[dwords - 1] : 0u);
            if (pcrel_dispatch_target != UINT32_MAX)
                fprintf(stderr, " pcrel-target=%u", pcrel_dispatch_target);
            fputc('\n', stderr);
        }
        return {};
    }

    const FragmentInterpolationLayout derived_interpolation = interpolation
        ? *interpolation : fragment_interpolation_layout(code, dwords, system_inputs);
    if (!derived_interpolation.valid) {
        if (getenv("PROSPER_DBG"))
            fprintf(stderr, "[recompile-reject] invalid fragment interpolation layout\n");
        return {};
    }
    const uint32_t effective_wave_size = effective_fragment_wave_size(
        wave_size, program_dwords, shader_program_hash(code, program_dwords));
    SpirvCompute b;
    b.wave_size = effective_wave_size;
    b.begin_fragment(rt, color_mask);
    // SPI_PS_IN_CONTROL.PS_W32_EN proves that EXEC_HI/VCC_HI are unused and the low-half mask
    // operations below represent the complete wave. Keep the older byte-exact captured exception
    // until every replay/capture producer carries the stage register into this entry point.
    b.allow_b32_masks = effective_wave_size == 32;
    // Fragment I/O value tap (PROSPER_FS_TAP=draw:pc): redirect the MRT0 colour export to the intermediate
    // VGPR produced at that PC so the rendered frame visualises the value. The `draw:` prefix is consumed by
    // gpu_replay (which re-recompiles only that draw's FS). Parse the same complete selector here so an
    // invalid or overflowing PC cannot silently become PC zero or truncate to 32 bits.
    if (const char* tap = getenv("PROSPER_FS_TAP")) {
        uint64_t draw = 0;
        uint32_t pc = 0;
        if (parse_fragment_tap_selector(tap, draw, pc)) b.tap_pc = pc;
    }
    b.declare_guest_scratch(scratch);
    b.fragment_interpolation = &derived_interpolation;
    // P0-only attributes retain the cheap Flat varying path. Mixed smooth/explicit-parameter reads
    // are legal with the portable geometry stage and use separate packed locations there.
    if (!derived_interpolation.requires_geometry) {
        for (const auto& in : ins) {
            if (in.is_end) break;
            if (in.fmt == Rdna2Format::VINTRP && in.opcode == 2)
                b.flat_attrs.insert(in.vintrp_attr);
        }
    }
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    if (system_inputs) {
        // RDNA2 PS system values are packed in field order. ADDR reserves each field's documented
        // width even when ENA is clear, allowing a driver to keep later VGPR numbers stable. Vulkan
        // exposes the four floating-point position terms directly as FragCoord.xyzw.
        static constexpr uint8_t widths[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        uint32_t vgpr = 0;
        for (uint32_t field = 0; field < 16; ++field) {
            const uint32_t bit = 1u << field;
            if (!(system_inputs->addr & bit)) continue;
            if (system_inputs->ena & bit) {
                if (field <= 6 && derived_interpolation.requires_geometry) {
                    for (uint32_t component = 0; component < widths[field]; ++component) {
                        const uint32_t value = b.system_interpolation_component(field, component);
                        if (!value) {
                            if (getenv("PROSPER_DBG"))
                                fprintf(stderr,
                                        "[recompile-reject] missing fragment system interpolation "
                                        "field=%u component=%u\n",
                                        field, component);
                            return {};
                        }
                        rs.vreg[(int)(vgpr + component)] = value;
                    }
                } else if (field >= 8 && field <= 11) {
                    rs.vreg[(int)vgpr] = b.fragcoord_component(field - 8);
                }
            }
            vgpr += widths[field];
        }
    }
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // FRAGMENT-only: also linearize alpha-test / clip() kill-mask s_cbranch_scc0/scc1 early-outs (Unity
    // cutout + text draws). Merged into the same set so emit_alu drops the branch and detect_forward_if
    // skips it; the block's EXEC narrow + the export's OpKill do the per-invocation discard (#102). This is
    // NOT added for the vertex/compute shells (their scc branches are real uniform-ifs / NGG culling).
    for (uint32_t pc : mask_test_branches(ins, b.allow_b32_masks)) safe_branches.insert(pc);
    std::array<bool, 2> exported{};
    auto exp_fn = [&](RegState& state, const Rdna2Inst& in) -> bool { // EXP MRT0/MRT1 -> matching output
        // An export while EXEC is narrowed (lanes killed by an alpha test / v_cmpx and not restored to
        // all-on) must not write the inactive lanes. Lower it to a real fragment discard: OpKill the lanes
        // whose EXEC bit is false, then export from the survivors under full EXEC. This is exactly the
        // alpha-tested-sprite shape (image_sample -> v_cmp alpha<ref -> s_andn2 saved,saved,vcc -> s_wqm
        // exec,saved -> shade -> export): the surviving lanes are the ones that passed the test. (When EXEC
        // was never narrowed this is a no-op — the common sRGB/tonemap restore-then-export path.)
        if (state.exec_narrowed) {
            b.discard_unless(state.exec);
            state.exec = b.btrue();
            state.exec_narrowed = false;
        }
        // MRTZ (target 8): the shader-exported depth. EN bit 0 enables the Z VGPR (compilers emit
        // EN=0x1); COMPR depth is unmodeled. Previously this export was silently DROPPED, leaving
        // fixed-function interpolated Z where hardware uses the shader's value (#883).
        if (in.exp_target == 8) {
            if (in.exp_compr || !(in.exp_en & 1u)) return false;
            bool eok = true;
            const uint32_t z = operand_bits(b, state, in, in.src[0], &eok);
            if (!eok) return false;
            b.export_depth(z);
            return true;
        }
        if (in.exp_target < exported.size()) {
            // EN (Table 56) selects which VSRC channels the export sends; hardware does not update
            // disabled components. The executor maps this mask to Vulkan colorWriteMask, so the SPIR-V
            // value in a disabled channel is irrelevant and must not force a read of a stale VGPR.
            if (in.exp_en == 0) return true;
            bool eok = true;   // a Special (wave-mask) source has no data value — reject, don't export 0 (#134)
            if (in.exp_compr) {
                // COMPR: the 4 channels are two f16x2 pairs — src[0] holds (r,g), src[1] holds (b,a).
                // Unpack each half to a float and reassemble the vec4 (the pkrtz'd tonemap/sRGB output).
                const uint32_t p0 = (in.exp_en & 0x3u)
                    ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0);
                const uint32_t p1 = (in.exp_en & 0xCu)
                    ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0);
                b.export_color(in.exp_target,
                               (in.exp_en & 0x1u) ? b.unpack_half(p0, 0) : b.uconst(0),
                               (in.exp_en & 0x2u) ? b.unpack_half(p0, 1) : b.uconst(0),
                               (in.exp_en & 0x4u) ? b.unpack_half(p1, 0) : b.uconst(0),
                               (in.exp_en & 0x8u) ? b.unpack_half(p1, 1) : b.uconst(0));
            } else {
                b.export_color(in.exp_target,
                               (in.exp_en & 0x1u) ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0),
                               (in.exp_en & 0x2u) ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0),
                               (in.exp_en & 0x4u) ? operand_bits(b, state, in, in.src[2], &eok) : b.uconst(0),
                               (in.exp_en & 0x8u) ? operand_bits(b, state, in, in.src[3], &eok) : b.uconst(0));
            }
            if (!eok) return false;
            exported[in.exp_target] = true;
        }
        return true;   // NULL carries the EXEC/discard effect above; ignore additional exports for now
    };
    // cmpx is now ALLOWED (allow_exec_update=true): a fragment divergent-if (v_cmpx ... s_mov exec,saved)
    // is handled by EXEC predication like compute, and the export is guarded above. Memory ops need a
    // resource table. Loops (if any) are reconstructed by emit_body.
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true, /*allow_smem*/rt != nullptr, exp_fn, code, dwords)) {
        if (getenv("PROSPER_DBG") && pcrel_dispatch_target != UINT32_MAX)
            fprintf(stderr, "[recompile-reject] pcrel target=%u body failed\n", pcrel_dispatch_target);
        return {};
    }
    if (!exported[0] && !exported[1] && !has_null_export && !has_depth_export) {
        if (getenv("PROSPER_DBG"))
            fprintf(stderr, "[recompile-reject] emitted no fragment color%s%u\n",
                    pcrel_dispatch_target != UINT32_MAX ? " pcrel-target=" : "",
                    pcrel_dispatch_target != UINT32_MAX ? pcrel_dispatch_target : 0u);
        return {};
    }
    return b.finish();
}

std::vector<uint32_t> recompile_fragment(const uint32_t* code, size_t dwords,
                                         const ShaderResourceTable* rt,
                                         const PixelSystemInputMapping* system_inputs,
                                         uint32_t pcrel_dispatch_target,
                                         const FragmentInterpolationLayout* interpolation,
                                         bool wave32) {
    return recompile_fragment_impl(code, dwords, rt, system_inputs,
                                   pcrel_dispatch_target, interpolation,
                                   wave32 ? 32u : 64u);
}

std::vector<uint32_t> recompile_fragment_wave32_for_test(
        const uint32_t* code, size_t dwords) {
    return recompile_fragment_impl(code, dwords, nullptr, nullptr,
                                   UINT32_MAX, nullptr, 32);
}

uint32_t fragment_spirv_required_subgroup_size(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    constexpr char prefix[] = "Prosper.FragmentSubgroupSize=";
    bool legacy_arithmetic_marker = false;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_Capability && words == 2 &&
            spirv[offset + 1] == Cap_GroupNonUniformArithmetic)
            legacy_arithmetic_marker = true;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value = std::strtoul(
                        text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length && (value == 32 || value == 64))
                        return static_cast<uint32_t>(value);
                }
            }
        }
        offset += words;
    }
    // Compatibility for cached/captured modules produced before explicit module metadata.
    return legacy_arithmetic_marker ? 64u : 0u;
}

uint32_t compute_spirv_min_subgroup_size(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    constexpr char prefix[] = "Prosper.ComputeSubgroupMin=";
    uint32_t required = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value = std::strtoul(text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length &&
                        (value == 4 || value == 16 || value == 32 || value == 64))
                        required = std::max(required, static_cast<uint32_t>(value));
                }
            }
        }
        offset += words;
    }
    return required;
}

uint32_t fragment_spirv_required_subgroup_features(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    uint32_t features = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_Capability && words == 2) {
            if (spirv[offset + 1] == Cap_GroupNonUniformVote)
                features |= kFragmentSubgroupVote;
            else if (spirv[offset + 1] == Cap_GroupNonUniformArithmetic)
                features |= kFragmentSubgroupArithmetic;
            else if (spirv[offset + 1] == Cap_GroupNonUniformShuffle)
                features |= kFragmentSubgroupShuffle;
        }
        offset += words;
    }
    return features;
}

bool fragment_spirv_uses_internal_gds(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return false;
    std::unordered_map<uint32_t, uint32_t> sets, bindings;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return false;
        if (opcode == Op_Decorate && words == 4) {
            if (spirv[offset + 2] == Dec_DescriptorSet)
                sets[spirv[offset + 1]] = spirv[offset + 3];
            else if (spirv[offset + 2] == Dec_Binding)
                bindings[spirv[offset + 1]] = spirv[offset + 3];
        }
        offset += words;
    }
    for (const auto& [variable, set] : sets) {
        auto binding = bindings.find(variable);
        if (set == 1 && binding != bindings.end() && binding->second == 0) return true;
    }
    return false;
}

// Astro Bot's observed NGG wrappers use wave-shared plumbing for vertex allocation/compaction. The
// current Vulkan vertex shell intentionally projects those complete wrappers to one private guest
// lane; applying the projection to arbitrary NGG programs would silently miscompile peer-lane state.
// Keep each exception byte-exact and fail closed for every other wrapper. The hashes are FNV-1a over
// the little-endian instruction bytes through S_ENDPGM, matching the raw hashes in capture
// diagnostics. A proven PC-relative constant-table tail is deliberately excluded from the wrapper
// identity while remaining available to the recompiler.
static bool is_astro_bot_ngg_one_lane_wrapper(const uint32_t* code, size_t dwords) {
    if (!code) return false;
    std::vector<Rdna2Inst> instructions;
    const size_t program_dwords = rdna2_walk(code, dwords, instructions);
    if (program_dwords != 54 && program_dwords != 734 && program_dwords != 749 &&
        program_dwords != 3124 && program_dwords != 3435 && program_dwords != 3455 &&
        program_dwords != 3917)
        return false;
    const uint64_t hash = shader_program_hash(code, program_dwords);
    return (program_dwords == 54 && hash == 0x9e9d8e37bcc70607ull) ||
           (program_dwords == 734 && hash == 0x79eb2b954b07dc8eull) ||
           // The same 734-word culling wrapper is live-linked after its exact 15-word fetch prolog.
           (program_dwords == 749 && hash == 0xb440349937df751eull) ||
           (program_dwords == 3124 && hash == 0x41e6ac616c18d295ull) ||
           (program_dwords == 3435 && hash == 0xfad7a9f486523cfcull) ||
           // The same 3435-word wrapper is live-linked after its exact 20-word fetch prolog.
           // Hashing the complete concatenated program keeps the one-lane projection byte-exact.
           (program_dwords == 3455 && hash == 0x562ce5ad01c4c6e3ull) ||
           (program_dwords == 3917 && hash == 0x7f5f2349e2816f5eull);
}

VertexPrologInfo rdna2_vertex_prolog_info(const uint32_t* code, size_t dwords) {
    VertexPrologInfo result;
    if (!code || !dwords) return result;

    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    for (const Rdna2Inst& instruction : instructions) {
        // A fetch prolog has no architectural output or program termination of its own. Encountering
        // either before the transfer means this is a complete/different shader, not the split ABI.
        if (instruction.is_end || instruction.fmt == Rdna2Format::EXP ||
            instruction.fmt == Rdna2Format::Unknown)
            return {};
        if (instruction.fmt != Rdna2Format::SOP1 || instruction.opcode != 0x20)
            continue;

        // GFX9+ merged-stage fetch prologs receive the continuation PC in reserved s[6:7]. Keeping
        // this exact pair in the recognizer prevents an arbitrary indirect jump from becoming host
        // fallthrough merely because a second program happened to be bound.
        if (instruction.n_src != 1 || instruction.src[0].kind != OperandKind::SGPR ||
            instruction.src[0].value != 6 || instruction.len_dwords != 1)
            return {};
        result.valid = instruction.pc != 0;
        result.setpc_pc = instruction.pc;
        result.prefix_dwords = instruction.pc;
        break;
    }
    if (!result.valid) return {};

    // Replacing the transfer with fallthrough is valid only when every direct branch in the prolog
    // remains inside the retained prefix (or lands exactly on the transfer, which becomes main pc0).
    // A branch into discarded padding/data would otherwise be silently redirected into unrelated
    // main code. The linked body recompiler performs the remaining structured-CFG validation.
    for (const Rdna2Inst& instruction : instructions) {
        if (instruction.pc >= result.setpc_pc) break;
        if (instruction.fmt != Rdna2Format::SOPP ||
            (instruction.opcode != 0x02 &&
             (instruction.opcode < 0x04 || instruction.opcode > 0x09)))
            continue;
        const int64_t target = static_cast<int64_t>(instruction.pc) + instruction.len_dwords +
                               static_cast<int64_t>(instruction.simm16);
        if (target < 0 || target > result.setpc_pc) return {};
    }
    return result;
}

namespace {

// A no-GS NGG program is split into two machine-code allocations by the guest compiler: the
// logical vertex producer writes one compact per-vertex LDS record, then a compiler-generated NGG
// wrapper culls/compacts primitives and exports fields from that record.  Vulkan's vertex stage
// already launches exactly the logical draw vertices and performs primitive assembly itself.  When
// both sides of this ABI can be proven from the machine code, execute only the producer and export
// the same LDS fields directly.  This avoids pretending that Function-private LDS can communicate
// between independent Vulkan vertex invocations.
struct NggPassthroughLayout {
    bool valid = false;
    uint32_t producer_base_vgpr = 0;
    uint32_t producer_base_byte = 0;
    uint32_t record_stride_bytes = 0;
    std::array<int32_t, 4> position = {-1, -1, -1, -1};
    std::array<std::array<int32_t, 4>, 32> params{};
    uint32_t param_mask = 0;

    NggPassthroughLayout() {
        for (auto& param : params) param.fill(-1);
    }
};

struct NggLdsSource {
    bool valid = false;
    uint32_t byte_offset = 0;
    uint32_t stride_bytes = 0;
    uint32_t index_vgpr = 0;
};

// Resolve the terminal wrapper's canonical `stride * exporter + constant` LDS address.  Requiring
// the nearest writer to be this exact u24 MAD shape keeps the optimization fail-closed when a user
// GS or a different compiler layout performs real output computation.
NggLdsSource ngg_terminal_lds_source(const std::vector<Rdna2Inst>& ins, size_t load_index,
                                     uint32_t output_vgpr) {
    const Rdna2Inst& load = ins[load_index];
    uint32_t address_vgpr = 0, component_byte = 0;
    if (load.fmt != Rdna2Format::DS || load.ds_gds) return {};
    if (load.opcode == 0x36u) {                         // ds_read_b32
        if (output_vgpr != static_cast<uint32_t>(load.dst.value)) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        component_byte = load.literal;
    } else if (load.opcode == 0x37u) {                  // ds_read2_b32
        const uint32_t first = static_cast<uint32_t>(load.dst.value);
        if (output_vgpr < first || output_vgpr > first + 1u) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        const uint32_t component = output_vgpr - first;
        component_byte = ((load.literal >> (component * 8u)) & 0xffu) * 4u;
    } else if (load.opcode == 0x76u || load.opcode == 0xfeu || load.opcode == 0xffu) {
        const uint32_t count = load.opcode == 0x76u ? 2u : load.opcode == 0xfeu ? 3u : 4u;
        const uint32_t first = static_cast<uint32_t>(load.dst.value);
        if (output_vgpr < first || output_vgpr >= first + count) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        component_byte = load.literal + (output_vgpr - first) * 4u;
    } else {
        return {};
    }

    for (size_t j = load_index; j-- > 0;) {
        const Rdna2Inst& writer = ins[j];
        if (writer.dst.kind != OperandKind::VGPR ||
            static_cast<uint32_t>(writer.dst.value) != address_vgpr)
            continue;
        // The compacted record address has either of the two canonical compiler forms below:
        //
        //   v_mad_u32_u24 addr, stride, exporter, constant
        //   v_mul_u32_u24 addr, stride, exporter; ds_read ... offset:constant
        //
        // The latter avoids a MAD when the entire constant fits in the DS instruction's immediate.
        // Both prove the same `stride * exporter + constant` identity; accepting only these exact
        // integer-u24 forms keeps arbitrary wrapper address arithmetic fail-closed.
        const bool mad = writer.fmt == Rdna2Format::VOP3 && writer.opcode == 0x143u &&
                         !writer.has_modifier && writer.has_literal;
        const bool mul = writer.fmt == Rdna2Format::VOP2 && writer.opcode == 0x0bu &&
                         !writer.has_modifier && !writer.has_literal;
        if (!mad && !mul) return {};
        // The first two operands may be swapped.
        int stride_src = -1, index_src = -1;
        for (int k = 0; k < 2; ++k) {
            if (writer.src[k].kind == OperandKind::InlineInt && writer.src[k].value > 0)
                stride_src = k;
            else if (writer.src[k].kind == OperandKind::VGPR)
                index_src = k;
        }
        if (stride_src < 0 || index_src < 0 ||
            (mad && writer.src[2].kind != OperandKind::Literal))
            return {};
        const uint32_t stride = static_cast<uint32_t>(writer.src[stride_src].value);
        if ((stride & 3u) || stride < 16u || stride > 4096u ||
            (mad && writer.literal > UINT32_MAX - component_byte))
            return {};
        return {true, (mad ? writer.literal : 0u) + component_byte, stride,
                static_cast<uint32_t>(writer.src[index_src].value)};
    }
    return {};
}

NggLdsSource ngg_find_terminal_output(const std::vector<Rdna2Inst>& ins, size_t export_index,
                                      uint32_t output_vgpr) {
    // The direct output loads sit in the export block.  Stop at the nearest writer; skipping a
    // transform or phi would turn a user GS into a false passthrough.
    for (size_t j = export_index; j-- > 0;) {
        const Rdna2Inst& writer = ins[j];
        if (writer.fmt == Rdna2Format::DS) {
            const NggLdsSource source = ngg_terminal_lds_source(ins, j, output_vgpr);
            if (source.valid) return source;
            const uint32_t first = static_cast<uint32_t>(writer.dst.value);
            const uint32_t count = writer.opcode == 0x37u || writer.opcode == 0x76u ? 2u
                                 : writer.opcode == 0xfeu ? 3u
                                 : writer.opcode == 0xffu ? 4u : 1u;
            if (output_vgpr >= first && output_vgpr < first + count) return {};
        }
        if (writer.dst.kind == OperandKind::VGPR &&
            static_cast<uint32_t>(writer.dst.value) == output_vgpr)
            return {};
    }
    return {};
}

NggPassthroughLayout analyze_ngg_passthrough(const uint32_t* prolog, size_t prefix_dwords,
                                             const uint32_t* main, size_t main_dwords) {
    NggPassthroughLayout out;
    auto reject = [&](const char* reason) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-ngg-passthrough-reject] %s\n", reason);
        return NggPassthroughLayout{};
    };
    std::vector<Rdna2Inst> producer, wrapper;
    rdna2_walk(prolog, prefix_dwords, producer);
    rdna2_walk(main, main_dwords, wrapper);
    if (producer.empty() || wrapper.empty()) return reject("empty producer/wrapper");

    // Prove that every LDS operation in the producer is a non-atomic store into one record base.
    // The private-LDS execution is valid only before the cross-lane wrapper starts reading it.
    bool saw_store = false;
    uint32_t base_vgpr = UINT32_MAX;
    std::set<uint32_t> stored_bytes;
    for (const Rdna2Inst& in : producer) {
        if (in.fmt == Rdna2Format::EXP || in.is_end) return reject("producer terminates or exports");
        if (in.fmt != Rdna2Format::DS) continue;
        if (in.ds_gds || in.src[0].kind != OperandKind::VGPR)
            return reject("producer uses non-LDS DS address");
        const uint32_t base = static_cast<uint32_t>(in.src[0].value);
        if (base_vgpr == UINT32_MAX) base_vgpr = base;
        if (base != base_vgpr) return reject("producer has multiple LDS record bases");
        auto store_byte = [&](uint32_t byte) {
            if ((byte & 3u) == 0u) stored_bytes.insert(byte);
        };
        if (in.opcode == 0x0du) {
            store_byte(in.literal);
        } else if (in.opcode == 0x0eu) {
            store_byte((in.literal & 0xffu) * 4u);
            store_byte(((in.literal >> 8) & 0xffu) * 4u);
        } else if (in.opcode == 0x4du || in.opcode == 0xdeu || in.opcode == 0xdfu) {
            const uint32_t count = in.opcode == 0x4du ? 2u : in.opcode == 0xdeu ? 3u : 4u;
            for (uint32_t k = 0; k < count; ++k) store_byte(in.literal + k * 4u);
        } else {
            return reject("producer has a non-store/cross-lane DS operation");
        }
        saw_store = true;
    }
    if (!saw_store || stored_bytes.empty()) return reject("producer writes no aligned LDS record");

    bool saw_alloc = false, saw_primitive_export = false, saw_position = false;
    uint32_t common_stride = 0, common_index = UINT32_MAX;
    std::vector<uint32_t> absolute_offsets;
    auto accept_source = [&](const NggLdsSource& source, int32_t& destination) -> bool {
        if (!source.valid || (common_stride && source.stride_bytes != common_stride) ||
            (common_index != UINT32_MAX && source.index_vgpr != common_index))
            return false;
        common_stride = source.stride_bytes;
        common_index = source.index_vgpr;
        destination = static_cast<int32_t>(source.byte_offset);
        absolute_offsets.push_back(source.byte_offset);
        return true;
    };
    for (size_t i = 0; i < wrapper.size(); ++i) {
        const Rdna2Inst& in = wrapper[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10u) saw_alloc = true;
        if (in.fmt != Rdna2Format::EXP) continue;
        if (in.exp_target == 20u) { saw_primitive_export = true; continue; }
        if (in.exp_target != 12u && in.exp_target < 32u) continue;
        if (in.exp_compr) return reject("terminal output uses a compressed export");
        if (in.exp_target == 12u) {
            if (saw_position || in.exp_en != 0xfu)
                return reject("POS0 is duplicated or incomplete");
            saw_position = true;
            for (uint32_t component = 0; component < 4; ++component) {
                if (!accept_source(ngg_find_terminal_output(
                                       wrapper, i, static_cast<uint32_t>(in.src[component].value)),
                                   out.position[component]))
                    return reject("POS0 does not directly load the terminal LDS record");
            }
        } else {
            const uint32_t param = in.exp_target - 32u;
            if (param >= out.params.size() || (out.param_mask & (1u << param)))
                return reject("PARAM export is out of range or duplicated");
            out.param_mask |= 1u << param;
            for (uint32_t component = 0; component < 4; ++component) {
                if (!(in.exp_en & (1u << component))) continue;
                if (!accept_source(ngg_find_terminal_output(
                                       wrapper, i, static_cast<uint32_t>(in.src[component].value)),
                                   out.params[param][component]))
                    return reject("PARAM does not directly load the terminal LDS record");
            }
        }
    }
    if (!saw_alloc || !saw_primitive_export || !saw_position || absolute_offsets.empty() ||
        !common_stride || base_vgpr == UINT32_MAX)
        return reject("wrapper lacks the no-GS allocation/export shape");

    const uint32_t wrapper_base = *std::min_element(absolute_offsets.begin(), absolute_offsets.end());
    const uint32_t producer_base = *stored_bytes.begin();
    auto normalize = [&](int32_t& byte) -> bool {
        if (byte < 0) return true;
        const uint32_t absolute = static_cast<uint32_t>(byte);
        if (absolute < wrapper_base) return false;
        const uint32_t relative = absolute - wrapper_base;
        if ((relative & 3u) || relative >= common_stride ||
            producer_base > UINT32_MAX - relative ||
            !stored_bytes.count(producer_base + relative))
            return false;
        byte = static_cast<int32_t>((producer_base + relative) / 4u);
        return true;
    };
    for (int32_t& component : out.position)
        if (!normalize(component)) return reject("POS0 offset does not match the producer record");
    for (auto& param : out.params)
        for (int32_t& component : param)
            if (!normalize(component)) return reject("PARAM offset does not match the producer record");

    out.valid = true;
    out.producer_base_vgpr = base_vgpr;
    out.producer_base_byte = producer_base;
    out.record_stride_bytes = common_stride;
    return out;
}

} // namespace

static std::vector<uint32_t> recompile_vertex_impl(const uint32_t* code, size_t dwords,
                                                   const ShaderResourceTable* rt,
                                                   const PixelInputMapping* pixel_inputs,
                                                   bool capture_position,
                                                   uint32_t virtual_lds_dwords,
                                                   const NggPassthroughLayout* passthrough,
                                                   bool allow_test_ngg_output_gate,
                                                   bool allow_test_ngg_one_lane) {
    const uint32_t passthrough_mask =
        pixel_inputs ? pixel_inputs->effective_passthrough_mask() : 0u;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    SpirvCompute b;
    b.capture_position = capture_position;   // geometry-probe: mark gl_Position for xfb capture (gated)
    b.vertex_lds_dwords = std::min(virtual_lds_dwords, 16384u);
    b.vertices_per_instance = rt ? rt->vertices_per_instance : 0u;
    // Shader I/O value tap (PROSPER_SHADER_TAP=pc): redirect the position export to the intermediate VGPR
    // produced at that PC. Applies to the vertex stage only; captured via the geometry probe.
    if (const char* tap = getenv("PROSPER_SHADER_TAP")) b.tap_pc = static_cast<uint32_t>(strtoul(tap, nullptr, 0));
    b.begin_vertex(rt);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // NGG vertex shaders (the exact GS_ALLOC_REQ message present) carry the vertex index in v5, not v0, and wrap
    // the body in wave-packing plumbing (s_sendmsg / exp prim / s_lshr_b64 exec) that lowers to no-ops in
    // our per-invocation model. Detect NGG and bind the index to v5 as well.
    bool ngg = passthrough && passthrough->valid;
    for (const auto& in : ins) { if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10 &&
            in.words[0] == 0xBF900009u) { ngg = true; break; } }
    const bool exact_ngg_projection = ngg && is_astro_bot_ngg_one_lane_wrapper(code, dwords);
    // The generic split-stage path uses private LDS only after the producer/wrapper analyzer has
    // proven that all visible outputs come from one per-vertex record. Wave-sensitive instruction
    // approximations remain restricted to the byte-exact projection above.
    b.ngg_private_lds = exact_ngg_projection || (passthrough && passthrough->valid) ||
                        (ngg && allow_test_ngg_output_gate);
    // The legacy byte-exact NGG projection and its explicit unit-test hook predate the linked-stage
    // ABI's exact LDS allocation. Preserve their proven 16 KiB private scratch contract when no
    // allocation was supplied; generic linked producers still require their real plumbed size.
    if (!b.vertex_lds_dwords && (exact_ngg_projection || allow_test_ngg_output_gate))
        b.vertex_lds_dwords = 4096;
    // Every wave/peer approximation is an exception for the one captured Astro wrapper, not a
    // property of the GS_ALLOC_REQ opcode. Other NGG programs retain only the ordinary merged-stage
    // ABI setup below and fail closed if they reach a lane-sensitive operation.
    b.ngg_one_lane = exact_ngg_projection || (ngg && allow_test_ngg_one_lane);
    // A LO+HI all-ones pair is the compiler's explicit wave64 lane-index construction. Infer the
    // width from that machine-code proof instead of assuming every NGG program is wave64. A low-only
    // producer may be wave32 or may use only half of a wave64 mask, so it remains fail-closed until
    // the graphics wave-size contract is plumbed independently.
    bool logical_mbcnt_lo = false, logical_mbcnt_hi = false, logical_mbcnt_invalid = false;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::VOP3 || (in.opcode != 0x365 && in.opcode != 0x366))
            continue;
        const bool all_ones = in.src[0].kind == OperandKind::InlineInt &&
                              in.src[0].value == -1;
        logical_mbcnt_invalid |= !all_ones;
        logical_mbcnt_lo |= all_ones && in.opcode == 0x365;
        logical_mbcnt_hi |= all_ones && in.opcode == 0x366;
    }
    b.ngg_logical_lane = passthrough && passthrough->valid && !logical_mbcnt_invalid &&
                         logical_mbcnt_lo && logical_mbcnt_hi;
    if (b.ngg_logical_lane) b.wave_size = 64;
    b.allow_b32_masks = b.ngg_one_lane;
    uint32_t ngg_output_gate_begin = UINT32_MAX;
    uint32_t ngg_output_gate_end = 0;
    if (ngg) {
        // A terminal compacted-output suffix may reconstruct its values from a shader-embedded
        // constant table. Detect those loads with the same bounded PC-relative proof used by the
        // emitter; an arbitrary external load or any buffer write must not broaden this gate.
        PcrelTables output_tables;
        if (b.ngg_private_lds)
            output_tables = detect_pcrel_tables(ins, code, dwords);
        auto scalar_output_setup = [](const Rdna2Inst& candidate) {
            if (candidate.fmt != Rdna2Format::SOP1 &&
                candidate.fmt != Rdna2Format::SOP2 &&
                candidate.fmt != Rdna2Format::SOPK)
                return false;
            bool wrote_data = false;
            bool safe = !instruction_may_change_exec(candidate);
            for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                wrote_data = true;
                safe &= base >= 0 && base + static_cast<int>(width) <= 106;
                safe &= !scalar_write_is_b64_mask(candidate, base);
            });
            return wrote_data && safe;
        };
        auto embedded_output_load = [&](const Rdna2Inst& candidate) {
            if (candidate.fmt != Rdna2Format::MUBUF || candidate.mubuf_lds)
                return false;
            const bool read_only = candidate.opcode <= 0x03u ||
                (candidate.opcode >= 0x0cu && candidate.opcode <= 0x0fu);
            return read_only && output_tables.mubuf.contains(candidate.pc);
        };
        uint32_t end_pc = UINT32_MAX;
        for (const auto& in : ins)
            if (in.is_end) { end_pc = in.pc; break; }
        // An NGG primitive shader finishes by compacting surviving vertices, CMPX-testing whether
        // this lane owns one of those vertices, then exporting POS/PARAM values before S_ENDPGM.
        // Vulkan's vertex shell already represents one surviving guest vertex per invocation, but
        // retaining the condition is still useful when the compacted count is zero. Permit exports
        // under narrowed EXEC only for this mechanically bounded terminal output gate; ordinary
        // vertex CMPX/export programs remain rejected below.
        for (size_t i = 1; i < ins.size(); ++i) {
            const Rdna2Inst& branch = ins[i];
            if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 ||
                branch.simm16 <= 0 || branch_target(branch) != end_pc)
                continue;
            size_t previous = i;
            do { --previous; } while (previous > 0 && sopp_is_noop(ins[previous]));
            if (ins[previous].fmt != Rdna2Format::VOPC ||
                !vopc_is_cmpx(ins[previous].opcode))
                continue;
            // Production accepts data-dependent vertex suppression only for the byte-exact Astro
            // wrapper. The explicit test hook below exercises active/inactive export selection with
            // a tiny shader without turning that shader shape into a runtime allow-list exception.
            if (!b.ngg_private_lds && !allow_test_ngg_output_gate)
                continue;
            bool has_position = false;
            bool output_only = true;
            std::vector<uint32_t> trailing_vcc_branches;
            for (size_t j = i + 1; j < ins.size() && ins[j].pc < end_pc; ++j) {
                const Rdna2Inst& candidate = ins[j];
                if (candidate.fmt == Rdna2Format::EXP) {
                    has_position |= candidate.exp_target == 12;
                    continue;
                }
                // Astro's compacted-output suffix reconstructs the surviving vertex from private
                // LDS or a bounded shader-embedded table immediately before exporting it. Vector,
                // DS, and table-load destination writes are EXEC-predicated by emit_alu; scalar ALU
                // may only build ordinary data/descriptor registers. Admit them only for the
                // byte-exact wrapper (or the explicit test hook); arbitrary NGG shaders never reach
                // this exception, and buffer stores/external reads remain rejected.
                const bool output_rebuild = b.ngg_private_lds || allow_test_ngg_output_gate;
                if (output_rebuild &&
                    (candidate.fmt == Rdna2Format::VOP1 ||
                     candidate.fmt == Rdna2Format::VOP2 ||
                     candidate.fmt == Rdna2Format::VOP3 ||
                     candidate.fmt == Rdna2Format::VOP3P ||
                     (candidate.fmt == Rdna2Format::VOPC &&
                      !vopc_is_cmpx(candidate.opcode)) ||
                     candidate.fmt == Rdna2Format::DS ||
                     scalar_output_setup(candidate) ||
                     embedded_output_load(candidate)))
                    continue;
                if (candidate.fmt == Rdna2Format::SOPC) continue;
                if (sopp_is_noop(candidate)) continue;
                if (candidate.fmt == Rdna2Format::SOPP &&
                    (candidate.opcode == 0x04 || candidate.opcode == 0x05) &&
                    candidate.simm16 > 0 && branch_target(candidate) == end_pc)
                    continue;
                // The 7f5f wrapper exports POS, compares a per-vertex flag into VCC, then conditionally
                // skips only its trailing PARAM exports. Those exports cannot affect position/topology;
                // linearizing the branch merely supplies otherwise-undefined varyings for that path.
                if (has_position && candidate.fmt == Rdna2Format::SOPP &&
                    (candidate.opcode == 0x06 || candidate.opcode == 0x07) &&
                    candidate.simm16 > 0 && branch_target(candidate) == end_pc) {
                    trailing_vcc_branches.push_back(candidate.pc);
                    continue;
                }
                output_only = false;
                break;
            }
            if (has_position && output_only) {
                ngg_output_gate_begin = branch.pc;
                ngg_output_gate_end = end_pc;
                safe_branches.insert(trailing_vcc_branches.begin(), trailing_vcc_branches.end());
                break;
            }
        }
    }
    uint32_t vidx = b.load_vertex_index();
    uint32_t iidx = b.load_instance_index();
    rs.vreg[0] = vidx;                       // Legacy VS ABI: v0 = vertex index
    rs.vreg[3] = iidx;                       // Legacy VS ABI: v3 = instance index
    if (ngg) {
        // Locate the NGG prologue's LDS-to-ES vertex-index handoff without hard-coding a register or
        // program counter: the first MUBUF vaddr must be most recently defined by a scalar DS read.
        // The one-lane backend substitutes BuiltIn VertexIndex at that exact read (see DS lowering).
        for (size_t use = 0; use < ins.size(); ++use) {
            if (ins[use].is_end) break;
            if ((ins[use].fmt != Rdna2Format::MUBUF && ins[use].fmt != Rdna2Format::MTBUF) ||
                ins[use].src[0].kind != OperandKind::VGPR)
                continue;
            const int index_reg = ins[use].src[0].value;
            for (size_t def = use; def-- > 0;) {
                const Rdna2Inst& candidate = ins[def];
                if (candidate.dst.kind != OperandKind::VGPR || candidate.dst.value != index_reg)
                    continue;
                if (candidate.fmt == Rdna2Format::DS && candidate.opcode == 0x36) {
                    b.ngg_vertex_index_read_pc = candidate.pc;
                    b.ngg_vertex_index_value = vidx;
                }
                break;
            }
            break;
        }
        // GFX10's merged GS/ES ABI enters the ES prolog with vertex/instance indices in v5/v8 and
        // merged-wave info in s3: per-wave ES/GS counts [7:0]/[15:8], wave-in-TG [27:24], and
        // TG wave count [31:28]. The host draw has already omitted padding invocations, so expose a
        // full logical wave while deriving the architectural wave ID from the flattened invocation.
        rs.vreg[5] = vidx;
        rs.vreg[8] = iidx;
        if (exact_ngg_projection) {
            rs.sreg[3] = b.uconst(1);

        // The merged NGG wrapper guards its whole counted ES loop with EXECZ. In the single-lane
        // model above that one ES lane is active by construction, so the wave-empty shortcut cannot
        // be taken. Let the ordinary counted-loop lowering consume the loop instead of rejecting the
        // redundant outer guard merely because the original hardware mask was vector-shaped.
        const CountedLoop loop = detect_counted_loop(ins);
        if (loop.found) {
            // NGG culling unrolls several EXEC-predicated LDS blocks inside its counted loop. A
            // larger block may contain a smaller already-safe EXECZ block plus CMPX comparisons,
            // and ends by restoring a VCC-saved mask to EXEC. The generic safe-execz pass deliberately
            // rejects CMPX writes. Here they are exact: the branch immediately follows a CMPX that
            // narrowed EXEC, all effects in the skipped block are EXEC-predicated (further CMPX can
            // only narrow it again), and the common target performs the same EXEC=VCC restore on both
            // paths. Scan inside-out so nested blocks are proven before their parents.
            for (size_t branch_index = ins.size(); branch_index-- > 0;) {
                const Rdna2Inst& branch = ins[branch_index];
                if (branch.pc < loop.header_pc || branch.pc > loop.backedge_pc ||
                    branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 ||
                    branch.simm16 <= 0)
                    continue;
                size_t previous = branch_index;
                while (previous > 0) {
                    --previous;
                    if (!sopp_is_noop(ins[previous])) break;
                }
                if (previous >= branch_index || ins[previous].fmt != Rdna2Format::VOPC ||
                    !vopc_is_cmpx(ins[previous].opcode))
                    continue;
                const uint32_t target_pc = branch_target(branch);
                size_t target_index = ins.size();
                for (size_t i = 0; i < ins.size(); ++i)
                    if (ins[i].pc == target_pc) { target_index = i; break; }
                bool restores_saved_exec = false;
                for (size_t i = target_index; i < ins.size() && i < target_index + 3; ++i) {
                    const Rdna2Inst& candidate = ins[i];
                    if (candidate.fmt == Rdna2Format::SOP1 && candidate.opcode == 0x04 &&
                        candidate.dst.value >= 126 &&
                        (candidate.src[0].value == 106 || candidate.src[0].value == 107)) {
                        restores_saved_exec = true;
                        break;
                    }
                    bool clobbers_saved_mask = instruction_may_change_exec(candidate) ||
                        candidate.fmt == Rdna2Format::VOPC;
                    for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                        clobbers_saved_mask |= base < 108 && 106 < base + static_cast<int>(width);
                    });
                    if (clobbers_saved_mask ||
                        (candidate.fmt == Rdna2Format::SOPP && candidate.opcode >= 0x02 &&
                         candidate.opcode <= 0x09 && candidate.opcode != 0x03))
                        break;
                }
                if (!restores_saved_exec)
                    continue;
                bool safe_block = true;
                for (const auto& candidate : ins) {
                    if (candidate.pc <= branch.pc || candidate.pc >= target_pc) continue;
                    const bool predicated_or_masked =
                        candidate.fmt == Rdna2Format::VOP1 ||
                        candidate.fmt == Rdna2Format::VOP2 ||
                        candidate.fmt == Rdna2Format::VOP3 ||
                        candidate.fmt == Rdna2Format::VOP3P ||
                        (candidate.fmt == Rdna2Format::VOPC &&
                         vopc_is_cmpx(candidate.opcode)) ||
                        candidate.fmt == Rdna2Format::MIMG ||
                        candidate.fmt == Rdna2Format::MUBUF ||
                        candidate.fmt == Rdna2Format::MTBUF ||
                        candidate.fmt == Rdna2Format::DS ||
                        candidate.fmt == Rdna2Format::FLAT;
                    const bool nested_safe = candidate.fmt == Rdna2Format::SOPP &&
                        candidate.opcode == 0x08 && safe_branches.count(candidate.pc);
                    if (!predicated_or_masked && !nested_safe && !sopp_is_noop(candidate)) {
                        safe_block = false;
                        break;
                    }
                }
                if (safe_block) safe_branches.insert(branch.pc);
            }
            for (const auto& in : ins) {
                if (in.pc >= loop.header_pc) break;
                if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                    branch_target(in) == loop.exit_pc)
                    safe_branches.insert(in.pc);
            }
        }
        } else {
            const uint32_t wave = b.ibin(
                Op_BitwiseAnd,
                b.ibin(Op_ShiftRightLogical, b.vertex_invocation_id(), b.uconst(6)),
                b.uconst(0xFu));
            rs.sreg[3] = b.ibin(Op_BitwiseOr, b.uconst(0x40004040u),
                                b.ibin(Op_ShiftLeftLogical, wave, b.uconst(24)));
        }
    }
    bool exported = false;
    auto exp_fn = [&](RegState& state, const Rdna2Inst& in) -> bool { // EXP POS0..3 -> gl_Position; PARAM -> varyings
        bool eok = true;   // a Special (wave-mask) source has no data value — reject, don't export 0 (#134)
        // COMPR pos/param exports carry two packed f16x2 pairs, not four f32 fields — reading the
        // VSRCs as full floats would pass packed-half bit patterns as x/y and stale registers as
        // z/w. Never observed in a vertex stage (compilers export positions/params at 32 bits);
        // reject fail-visibly until a live title exercises one.
        if (in.exp_compr) {
            if (getenv("PROSPER_DBG"))
                fprintf(stderr, "[recompile-reject] vertex compressed export pc=%u target=%u\n",
                        in.pc, in.exp_target);
            return false;
        }
        // v_cmpx is now allowed in the vertex shell (allow_exec_update=true below): a divergent block
        // (v_cmpx … s_mov_b64 exec, saved — DOLL's per-vertex lighting/fog attenuation) predicates its
        // VGPR writes like compute. A vertex MUST still export from full EXEC — the compiled shape
        // always restores EXEC before its pos/param exports; if one ever arrives narrowed, reject
        // (fail-visibly) rather than export possibly-inactive-lane values.
        const bool terminal_ngg_output = ngg_output_gate_begin != UINT32_MAX &&
            in.pc > ngg_output_gate_begin && in.pc < ngg_output_gate_end;
        if (state.exec_narrowed && !terminal_ngg_output && (in.exp_target >= 32 ||
            (in.exp_target >= 12 && in.exp_target <= 16))) {
            if (getenv("PROSPER_DBG"))
                fprintf(stderr,
                        "[recompile-reject] vertex export under narrowed exec pc=%u target=%u\n",
                        in.pc, in.exp_target);
            return false;
        }
        if (in.exp_target == 12) {
            // POS0 is the mandatory x/y/z/w position vector. POS1..POS4 carry ancillary position
            // data (clip/cull distances, point size, viewport/layer selection according to the
            // programmed position format) and must never be mistaken for gl_Position merely because
            // an NGG shader emits one before POS0. Until those built-ins are modeled, retain the
            // existing deliberate behavior of ignoring them.
            // A position export must supply all four components (EN=0xF); a partial POS0 is not
            // meaningfully completable in the current model, so reject rather than invent components.
            if (in.exp_en != 0xFu) {
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr,
                            "[recompile-reject] partial vertex position export pc=%u en=0x%x\n",
                            in.pc, in.exp_en);
                return false;
            }
            uint32_t x = operand_bits(b, state, in, in.src[0], &eok);
            uint32_t y = operand_bits(b, state, in, in.src[1], &eok);
            uint32_t z = operand_bits(b, state, in, in.src[2], &eok);
            uint32_t w = operand_bits(b, state, in, in.src[3], &eok);
            if (terminal_ngg_output && state.exec_narrowed) {
                // The exact compacted-output wrapper emits a contiguous prefix of complete
                // primitives. Map every inactive suffix invocation to one identical clip point;
                // primitives assembled solely from that suffix are therefore degenerate instead of
                // accidentally reusing the last active vertex. Keep the real values on the true path.
                const uint32_t zero = b.uconst(0);
                x = b.sel(state.exec, x, zero);
                y = b.sel(state.exec, y, zero);
                z = b.sel(state.exec, z, zero);
                w = b.sel(state.exec, w, b.uconst(fbits(1.0f)));
            }
            b.export_position(x, y, z, w);
            exported = true;
        } else if (in.exp_target >= 32) {                    // PARAM0.. -> remapped PS input varying
            const uint32_t source = in.exp_target - 32;
            // EN gates which channels the export sends (vec2/vec3 varyings use EN=0x3/0x7):
            // hardware leaves disabled channels unwritten (undefined for the PS). Substitute a
            // deterministic 0.0 for them instead of exporting stale VGPR data.
            uint32_t x = (in.exp_en & 1u) ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0);
            uint32_t y = (in.exp_en & 2u) ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0);
            uint32_t z = (in.exp_en & 4u) ? operand_bits(b, state, in, in.src[2], &eok) : b.uconst(0);
            uint32_t w = (in.exp_en & 8u) ? operand_bits(b, state, in, in.src[3], &eok) : b.uconst(0);
            if (terminal_ngg_output && state.exec_narrowed) {
                const uint32_t zero = b.uconst(0);
                x = b.sel(state.exec, x, zero); y = b.sel(state.exec, y, zero);
                z = b.sel(state.exec, z, zero); w = b.sel(state.exec, w, zero);
            }
            const uint32_t gx = x, gy = y, gz = z, gw = w;
            if (!pixel_inputs || source >= 32 || !(pixel_inputs->valid_mask & (1u << source)))
                b.export_param(source, gx, gy, gz, gw);       // absent control retains identity wiring
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    const uint32_t raw_offset = pixel_inputs->controls[ps_input] & 0x3Fu;
                    const uint32_t offset = (passthrough_mask & (1u << ps_input))
                        ? (raw_offset & 0x1fu) : raw_offset;
                    if (offset == source) b.export_param(ps_input, gx, gy, gz, gw);
                }
            }
        }
        if (!eok && getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[vertex-export-reject] pc=%u target=%u unresolved export operand\n",
                         in.pc, in.exp_target);
        return eok;
    };
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                   /*allow_smem*/rt != nullptr, exp_fn, code, dwords)) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-recompile-reject] body or export lowering failed\n");
        return {};
    }
    if (passthrough && passthrough->valid) {
        const auto base_it = rs.vreg.find(static_cast<int>(passthrough->producer_base_vgpr));
        if (base_it == rs.vreg.end() || !b.vertex_lds_dwords) {
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[vertex-ngg-passthrough-reject] missing producer LDS base/allocation\n");
            return {};
        }
        b.declare_lds();
        const uint32_t base_dword = b.ibin(
            Op_ShiftRightLogical, base_it->second, b.uconst(2u));
        auto record_load = [&](int32_t dword) -> uint32_t {
            return b.lds_load(dword == 0
                ? base_dword
                : b.ibin(Op_IAdd, base_dword, b.uconst(static_cast<uint32_t>(dword))));
        };
        b.export_position(record_load(passthrough->position[0]),
                          record_load(passthrough->position[1]),
                          record_load(passthrough->position[2]),
                          record_load(passthrough->position[3]));
        exported = true;

        for (uint32_t source = 0; source < passthrough->params.size(); ++source) {
            if (!(passthrough->param_mask & (1u << source))) continue;
            std::array<uint32_t, 4> value{};
            for (uint32_t component = 0; component < 4; ++component) {
                const int32_t dword = passthrough->params[source][component];
                value[component] = dword >= 0 ? record_load(dword) : b.uconst(0u);
            }
            if (!pixel_inputs || !(pixel_inputs->valid_mask & (1u << source)))
                b.export_param(source, value[0], value[1], value[2], value[3]);
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    if ((pixel_inputs->controls[ps_input] & 0x3fu) == source)
                        b.export_param(ps_input, value[0], value[1], value[2], value[3]);
                }
            }
        }
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[vertex-ngg-passthrough] base=v%u stride=%u params=%08x\n",
                         passthrough->producer_base_vgpr,
                         passthrough->record_stride_bytes, passthrough->param_mask);
    }
    if (!exported) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-recompile-reject] shader emitted no POS0\n");
        return {};
    }
    // OFFSET=0x20 asks the interpolator to synthesize a constant instead of consuming a PARAM
    // export. GFX10 DEFAULT_VAL encodes 0000, 0001, 1110, and 1111. Materialize those outputs in
    // the Vulkan vertex stage, whose fixed-function interface has no equivalent default source.
    if (pixel_inputs) {
        for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
            if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
            const uint32_t control = pixel_inputs->controls[ps_input];
            if ((passthrough_mask & (1u << ps_input)) ||
                (control & 0x3Fu) != 0x20u) continue;
            const uint32_t one = b.uconst(0x3F800000u), zero = b.uconst(0u);
            const uint32_t default_val = (control >> 8) & 0x3u;
            const bool xyz_one = (default_val & 0x2u) != 0;
            const bool w_one = (default_val & 0x1u) != 0;
            b.export_param(ps_input, xyz_one ? one : zero, xyz_one ? one : zero,
                           xyz_one ? one : zero, w_one ? one : zero);
        }
    }
    return b.finish();
}

std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords,
                                       const ShaderResourceTable* rt,
                                       const PixelInputMapping* pixel_inputs,
                                       bool capture_position,
                                       uint32_t virtual_lds_dwords) {
    return recompile_vertex_impl(code, dwords, rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false);
}

std::vector<uint32_t> recompile_vertex_terminal_ngg_gate_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, true, false);
}

std::vector<uint32_t> recompile_vertex_ngg_one_lane_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, false, true);
}

std::vector<uint32_t> recompile_vertex_chain(const uint32_t* prolog, size_t prolog_dwords,
                                             const uint32_t* main, size_t main_dwords,
                                             const ShaderResourceTable* rt,
                                             const PixelInputMapping* pixel_inputs,
                                             bool capture_position,
                                             uint32_t virtual_lds_dwords) {
    const VertexPrologInfo info = rdna2_vertex_prolog_info(prolog, prolog_dwords);
    if (!info.valid || !main || !main_dwords) return {};

    const size_t main_span = rdna2_recompile_code_span(main, main_dwords);
    if (!main_span || info.prefix_dwords > SIZE_MAX - main_span) return {};
    const NggPassthroughLayout passthrough =
        analyze_ngg_passthrough(prolog, info.prefix_dwords, main, main_span);
    if (passthrough.valid) {
        return recompile_vertex_impl(prolog, info.prefix_dwords, rt, pixel_inputs,
                                     capture_position, virtual_lds_dwords, &passthrough, false, false);
    }
    std::vector<uint32_t> linked;
    linked.reserve(info.prefix_dwords + main_span);
    linked.insert(linked.end(), prolog, prolog + info.prefix_dwords);
    linked.insert(linked.end(), main, main + main_span);
    return recompile_vertex_impl(linked.data(), linked.size(), rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false);
}

} // namespace prosper::gpu
