// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
#include "rdna2_to_spirv.hpp"
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
namespace {

enum : uint32_t {
    Op_ExtInstImport=11, Op_ExtInst=12, Op_MemoryModel=14, Op_EntryPoint=15, Op_ExecutionMode=16,
    Op_Capability=17, Op_TypeVoid=19, Op_TypeBool=20, Op_TypeInt=21, Op_TypeFloat=22, Op_TypeVector=23,
    Op_TypeRuntimeArray=29, Op_TypeStruct=30, Op_TypePointer=32, Op_TypeFunction=33,
    Op_ConstantTrue=41, Op_ConstantFalse=42, Op_Constant=43, Op_Function=54, Op_FunctionEnd=56, Op_Variable=59,
    Op_LogicalOr=166, Op_LogicalAnd=167, Op_Select=169, Op_FOrdEqual=180, Op_FOrdNotEqual=182, Op_FOrdLessThan=184, Op_FOrdGreaterThan=186,
    Op_FOrdLessThanEqual=188, Op_FOrdGreaterThanEqual=190,
    Op_FUnordEqual=181, Op_FUnordNotEqual=183, Op_FUnordLessThan=185, Op_FUnordGreaterThan=187,   // NaN-inclusive ("n"-prefix) compares
    Op_FUnordLessThanEqual=189, Op_FUnordGreaterThanEqual=191,
    Op_Load=61, Op_Store=62, Op_AccessChain=65, Op_Decorate=71, Op_MemberDecorate=72,
    Op_ConvertFToU=109, Op_ConvertFToS=110, Op_ConvertSToF=111, Op_ConvertUToF=112, Op_Bitcast=124,
    Op_CompositeConstruct=80, Op_CompositeExtract=81, Op_IAdd=128, Op_FAdd=129, Op_ISub=130, Op_FSub=131, Op_IMul=132, Op_FMul=133,
    Op_UMulExtended=151, Op_SMulExtended=152,   // {lo,hi} struct results (for mul_hi)
    Op_FDiv=136, Op_IEqual=170, Op_INotEqual=171, Op_UGreaterThan=172, Op_SGreaterThan=173,
    Op_UGreaterThanEqual=174, Op_SGreaterThanEqual=175, Op_ULessThan=176, Op_SLessThan=177,
    Op_ULessThanEqual=178, Op_SLessThanEqual=179,
    Op_ShiftRightLogical=194, Op_ShiftRightArithmetic=195, Op_ShiftLeftLogical=196, Op_BitwiseOr=197,
    Op_BitwiseXor=198, Op_BitwiseAnd=199, Op_Not=200, Op_BitFieldSExtract=202, Op_BitFieldUExtract=203,
    Op_BitReverse=204, Op_UConvert=113,
    Op_TypeImage=25, Op_TypeSampledImage=27, Op_SampledImage=86,
    Op_ImageSampleImplicitLod=87, Op_ImageSampleExplicitLod=88, Op_ImageFetch=95, Op_ImageGather=96, Op_Image=100,
    Op_ImageRead=98, Op_ImageWrite=99, Op_ImageQuerySizeLod=103,
    Op_TypeArray=28, Op_ControlBarrier=224,
    Op_DPdx=207, Op_DPdy=208,   // screen-space derivatives (Fragment; plain Shader capability)
    Op_Phi=245, Op_LoopMerge=246,
    Op_SelectionMerge=247, Op_Label=248, Op_Branch=249, Op_BranchConditional=250, Op_Kill=252, Op_Return=253,
};
// GLSL.std.450 extended-instruction numbers.
enum : uint32_t { Glsl_FAbs=4, Glsl_RoundEven=2, Glsl_Trunc=3, Glsl_Floor=8, Glsl_Ceil=9, Glsl_Fract=10, Glsl_Sin=13, Glsl_Cos=14,
                  Glsl_Exp2=29, Glsl_Log2=30,
                  Glsl_Sqrt=31, Glsl_InverseSqrt=32, Glsl_FMin=37, Glsl_UMin=38, Glsl_SMin=39, Glsl_FMax=40,
                  Glsl_UMax=41, Glsl_SMax=42, Glsl_PackHalf2x16=58, Glsl_UnpackHalf2x16=62 };
enum : uint32_t {
    Cap_Shader=1, Cap_Int64=11, Addr_Logical=0, Mem_GLSL450=1, Exec_Vertex=0, Exec_Fragment=4, Exec_GLCompute=5,
    EM_OriginUpperLeft=7, EM_LocalSize=17,
    SC_Input=1, SC_UniformConstant=0, SC_Output=3, SC_StorageBuffer=12, FC_None=0,
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
    ImgOp_Bias=1, ImgOp_Lod=2, ImgOp_Sample=0x40,   // ImageOperands bits: LOD bias / explicit LOD / MSAA Sample index.
    SC_Workgroup=4, Scope_Workgroup=2, MemSem_WGAcqRel=0x108,   // LDS: Workgroup storage + barrier scope/semantics
    Dec_Block=2, Dec_ArrayStride=6, Dec_BuiltIn=11, Dec_Flat=14, Dec_Location=30, Dec_Binding=33,
    Dec_DescriptorSet=34, Dec_Offset=35,
    BI_Position=0, BI_FragCoord=15, BI_WorkgroupId=26, BI_LocalInvocationId=27,
    BI_GlobalInvocationId=28, BI_VertexIndex=42,
};

uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// A compute-shader SPIR-V builder specialized for "load N floats -> compute over SSA floats ->
// store 1 float", with helpers the VALU translator drives.
struct SpirvCompute {
    std::vector<uint32_t> caps, extimp, mem, entry, exec, deco, types, code;
    std::unordered_map<uint32_t, uint32_t> fconst_cache, uconst_cache;
    uint32_t next_id = 1;
    uint32_t stride = 1;
    // fixed ids (set in begin()):
    uint32_t t_void=0, t_fn=0, t_f32=0, t_u32=0, t_i32=0, t_v3u=0, t_bool=0, t_ptr_sb_f32=0;
    uint32_t v_gid=0, v_groupid=0, v_in=0, v_out=0, gidx=0, f_main=0, glsl=0, bconst_false=0;
    uint32_t groupid[3] = {0, 0, 0}, localid_comp[3] = {0, 0, 0};
    uint32_t v_cbuf=0, v_cbuf1=0, t_ptr_sb_u32=0;   // scalar-memory constant buffers (bindings 2 and 3)
    std::map<uint32_t, uint32_t> cbuf_var;          // binding -> storage-buffer var (N-buffer model; 2/3 map to v_cbuf/v_cbuf1)
    bool     is_fragment=0;                          // true in the fragment shell (gates VINTRP interp)
    bool     is_compute=0;                            // true in the compute shell (gates LDS / s_barrier)
    // Descriptor set for this stage's resources. VS and PS share ONE Vulkan pipeline, so they must NOT
    // reuse binding numbers within one set (both stages number their cbuf/texture from binding 2 -> a
    // set-0 collision made the descriptor layout invalid, corrupting the VS's reads -> degenerate
    // geometry). Each stage owns its own set: VS=set 0, PS=set 1 (mirrors the PS5's per-stage resource
    // tables). Set in begin_fragment(); the host binds one descriptor set per stage.
    uint32_t desc_set=0;
    uint32_t exec_model=0;                           // deferred EntryPoint (emitted in finish() so lazily-
    std::vector<uint32_t> iface;                     // declared I/O varyings can join the interface list)

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
    // v_cvt_i32_f32 SATURATES: NaN -> 0, clamp to [INT_MIN, INT_MAX] (#135). 2147483520.0f
    // (0x4EFFFFFF) is the largest float < 2^31, so the high clamp is exact in range and inputs
    // >= 2^31 select INT_MAX; -2^31 is exactly representable, so the low clamp needs no select.
    uint32_t cvt_f2i(uint32_t bits) {
        uint32_t f = bcf(bits);
        uint32_t nan = id();  put(code, Op_FUnordNotEqual, {t_bool, nan, f, f});   // true iff NaN
        uint32_t safe = id(); put(code, Op_Select, {t_f32, safe, nan, fconstf(0.0f), f});
        uint32_t hi = id();   putv(code, Op_ExtInst, {t_f32, hi, glsl, Glsl_FMin, safe, fconstf(2147483520.0f)});
        uint32_t cl = id();   putv(code, Op_ExtInst, {t_f32, cl, glsl, Glsl_FMax, hi, fconstf(-2147483648.0f)});
        uint32_t ri = id();   put(code, Op_ConvertFToS, {t_i32, ri, cl});
        uint32_t big = id();  put(code, Op_FOrdGreaterThanEqual, {t_bool, big, f, fconstf(2147483648.0f)});
        return sel(big, uconst(0x7FFFFFFFu), i2u(ri));
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
    uint32_t uconst64(uint64_t v) { uint32_t c = id(); put(types, Op_Constant, {t_u64(), c, (uint32_t)v, (uint32_t)(v >> 32)}); return c; }
    uint32_t u64_from_lohi(uint32_t lo, uint32_t hi) {   // (u64)hi<<32 | (u64)lo  — combine an SGPR pair
        uint32_t l = id(); put(code, Op_UConvert, {t_u64(), l, lo});
        uint32_t h = id(); put(code, Op_UConvert, {t_u64(), h, hi});
        uint32_t hs = id(); put(code, Op_ShiftLeftLogical, {t_u64(), hs, h, uconst64(32)});
        uint32_t r = id(); put(code, Op_BitwiseOr, {t_u64(), r, l, hs}); return r;
    }
    uint32_t bfe_u64(uint32_t base64, uint32_t off, uint32_t cnt) {   // 64-bit unsigned bitfield extract
        // res = (base << (64-off-cnt)) >> (64-cnt), all logical u64 (portable — OpBitFieldUExtract on a
        // 64-bit base isn't reliably supported, e.g. llvmpipe returns 0). Valid for off+cnt<=64, cnt in [1,64].
        uint32_t total = ibin(Op_IAdd, off, cnt);
        uint32_t lsh32 = ibin(Op_ISub, uconst(64), total);
        uint32_t rsh32 = ibin(Op_ISub, uconst(64), cnt);
        uint32_t lsh = id(); put(code, Op_UConvert, {t_u64(), lsh, lsh32});
        uint32_t rsh = id(); put(code, Op_UConvert, {t_u64(), rsh, rsh32});
        uint32_t sl  = id(); put(code, Op_ShiftLeftLogical,  {t_u64(), sl, base64, lsh});
        uint32_t r   = id(); put(code, Op_ShiftRightLogical, {t_u64(), r, sl, rsh}); return r; }
    uint32_t u64_lo(uint32_t v64) { uint32_t r = id(); put(code, Op_UConvert, {t_u32, r, v64}); return r; }  // truncate low 32
    uint32_t u64_hi(uint32_t v64) { uint32_t s = id(); put(code, Op_ShiftRightLogical, {t_u64(), s, v64, uconst64(32)});
        uint32_t r = id(); put(code, Op_UConvert, {t_u32, r, s}); return r; }
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

    // Combined image+sampler support (MIMG image_sample). One float OpTypeImage/OpTypeSampledImage PER DIM
    // (2D/3D) is shared; each texture is a COMBINED_IMAGE_SAMPLER UniformConstant at its binding.
    uint32_t t_image = 0, t_sampled_image = 0;            // 2D (the common case; kept for existing callers)
    std::unordered_map<uint32_t, uint32_t> tex_var;       // binding -> combined-sampler OpVariable id
    std::unordered_map<uint32_t, uint32_t> tex_simg_dim;  // SPIR-V Dim -> OpTypeSampledImage id
    std::unordered_map<uint32_t, uint32_t> tex_img_dim;   // SPIR-V Dim -> its OpTypeImage id (OpImage results)
    uint32_t t_v3f_cache = 0;
    uint32_t t_v3f() { if (!t_v3f_cache) { t_v3f_cache = id(); put(types, Op_TypeVector, {t_v3f_cache, t_f32, 3}); } return t_v3f_cache; }
    uint32_t sampled_image_type(uint32_t dim) {
        auto it = tex_simg_dim.find(dim); if (it != tex_simg_dim.end()) return it->second;
        uint32_t ti = id(); put(types, Op_TypeImage, {ti, t_f32, dim, 0, 0, 0, 1, 0});  // sampled f32, Sampled=1
        uint32_t si = id(); put(types, Op_TypeSampledImage, {si, ti});
        if (dim == Dim_2D) { t_image = ti; t_sampled_image = si; }
        tex_simg_dim[dim] = si; tex_img_dim[dim] = ti; return si;
    }
    // Declare (idempotently) a combined image+sampler of SPIR-V `dim` at descriptor-set 0, `binding`.
    void declare_texture(uint32_t binding, uint32_t dim = Dim_2D) {
        uint32_t simg = sampled_image_type(dim);
        if (tex_var.count(binding)) return;
        uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_UniformConstant, simg});
        uint32_t v = id();     put(types, Op_Variable,    {t_ptr, v, SC_UniformConstant});
        put(deco, Op_Decorate, {v, Dec_DescriptorSet, desc_set});
        put(deco, Op_Decorate, {v, Dec_Binding, binding});
        tex_var[binding] = v;
    }
    // image_sample 2D: sample the combined sampler at `binding` with (u,v) float-BITS coords; fills
    // out[0..3] with the RGBA result components as raw VGPR bits. Implicit LOD is only legal in the
    // Fragment execution model (#151) — the compute/vertex shells have no derivatives, so there we
    // sample at explicit LOD 0 (what a non-pixel-shader image_sample resolves to without gradients).
    void image_sample_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {t_sampled_image, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id();
        if (is_fragment) put(code, Op_ImageSampleImplicitLod, {t_v4f, res, si, coord});
        else             put(code, Op_ImageSampleExplicitLod, {t_v4f, res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, res, c}); out[c] = bcu(e);
        }
    }
    // image_sample 3D: (u,v,w) float-BITS coords -> RGBA. Uses the Dim_3D sampled image; same
    // implicit-LOD-only-in-Fragment rule as image_sample_2d.
    void image_sample_3d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t w_bits, uint32_t out[4]) {
        uint32_t simg  = sampled_image_type(Dim_3D);
        uint32_t si    = id(); put(code, Op_Load, {simg, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v3f(), coord, bcf(u_bits), bcf(v_bits), bcf(w_bits)});
        uint32_t res   = id();
        if (is_fragment) put(code, Op_ImageSampleImplicitLod, {t_v4f, res, si, coord});
        else             put(code, Op_ImageSampleExplicitLod, {t_v4f, res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, res, c}); out[c] = bcu(e);
        }
    }
    // image_sample_l / _lz: sample with an EXPLICIT LOD (lod_bits float). Stage-agnostic (no derivatives).
    void image_sample_lod_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t lod_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {t_sampled_image, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id(); put(code, Op_ImageSampleExplicitLod, {t_v4f, res, si, coord, ImgOp_Lod, bcf(lod_bits)});
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, res, c}); out[c] = bcu(e);
        }
    }
    // image_sample_lz from a 3D texture: explicit LOD (usually 0) on a (u,v,w) coord. Stage-agnostic.
    void image_sample_lod_3d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t w_bits,
                             uint32_t lod_bits, uint32_t out[4]) {
        uint32_t simg  = sampled_image_type(Dim_3D);
        uint32_t si    = id(); put(code, Op_Load, {simg, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v3f(), coord, bcf(u_bits), bcf(v_bits), bcf(w_bits)});
        uint32_t res   = id(); put(code, Op_ImageSampleExplicitLod, {t_v4f, res, si, coord, ImgOp_Lod, bcf(lod_bits)});
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, res, c}); out[c] = bcu(e);
        }
    }
    // image_sample_b 2D: implicit-LOD sample with an LOD BIAS (bias_bits float). Bias only means
    // anything with implicit LOD (fragment derivatives); outside the fragment stage the op resolves
    // like the other samples there — explicit LOD 0 (bias dropped, matching image_sample_2d's rule).
    void image_sample_bias_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t bias_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {t_sampled_image, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id();
        if (is_fragment) put(code, Op_ImageSampleImplicitLod, {t_v4f, res, si, coord, ImgOp_Bias, bcf(bias_bits)});
        else             put(code, Op_ImageSampleExplicitLod, {t_v4f, res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, res, c}); out[c] = bcu(e);
        }
    }
    // image_gather4_lz 2D: OpImageGather of component `comp` (0..3) — the 2x2 footprint's four texels
    // of one channel, in the DX/GL gather order ((0,1),(1,1),(1,0),(0,0)), which the AMD gather4
    // result order matches. Gather always samples the base level (== the _lz behavior). out[0..3] =
    // the four gathered values as raw bits.
    void image_gather_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t comp, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {t_sampled_image, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id(); put(code, Op_ImageGather, {t_v4f, res, si, coord, uconst(comp)});
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, res, c}); out[c] = bcu(e);
        }
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
        uint32_t si    = id(); put(code, Op_Load, {t_sampled_image, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        // signed 6-bit texel offsets. NOTE: bfe_s takes SPIR-V IDs — raw integers here (the #296
        // original) emitted OpBitFieldSExtract with invalid operand IDs; never caught live because
        // the only gather4_lz_o user (DOLL's FXAA PS) still rejected upstream on its execz region.
        uint32_t ox    = bfe_s(off_bits, uconst(0), uconst(6)), oy = bfe_s(off_bits, uconst(8), uconst(6));
        uint32_t offv  = id(); put(code, Op_CompositeConstruct, {t_v2i(), offv, bcs(ox), bcs(oy)});
        uint32_t res   = id(); put(code, Op_ImageGather, {t_v4f, res, si, coord, uconst(comp), ImgOp_Offset, offv});
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, res, c}); out[c] = bcu(e);
        }
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
        uint32_t si   = id(); put(code, Op_Load,  {t_sampled_image, si, tex_var[binding]});
        uint32_t img  = id(); put(code, Op_Image, {t_image, img, si});
        uint32_t size = id(); put(code, Op_ImageQuerySizeLod, {t_v2i(), size, img, uconst(0)});
        uint32_t w_i  = id(); put(code, Op_CompositeExtract, {t_i32, w_i, size, 0});
        uint32_t h_i  = id(); put(code, Op_CompositeExtract, {t_i32, h_i, size, 1});
        uint32_t ox = bfe_s(off_bits, uconst(0), uconst(6)), oy = bfe_s(off_bits, uconst(8), uconst(6));   // signed 6-bit texel offsets
        uint32_t du = fbin(Op_FDiv, cvt_i2f(ox), cvt_i2f(i2u(w_i)));
        uint32_t dv = fbin(Op_FDiv, cvt_i2f(oy), cvt_i2f(i2u(h_i)));
        image_sample_lod_2d(binding, fbin(Op_FAdd, u_bits, du), fbin(Op_FAdd, v_bits, dv), uconst(0), out);
    }
    // 2-component uint vector (integer texel coordinates for OpImageFetch).
    uint32_t t_v2u_cache = 0;
    uint32_t t_v2u() { if (!t_v2u_cache) { t_v2u_cache = id(); put(types, Op_TypeVector, {t_v2u_cache, t_u32, 2}); } return t_v2u_cache; }
    // image_load 2D (image_load): texelFetch the image at the combined sampler's `binding` with INTEGER
    // (x,y) coords (raw VGPR bits). OpImage strips the sampler; OpImageFetch at explicit LOD 0.
    void image_fetch_2d(uint32_t binding, uint32_t x_bits, uint32_t y_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load,  {t_sampled_image, si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {t_image, img, si});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2u(), coord, x_bits, y_bits});
        uint32_t res   = id(); put(code, Op_ImageFetch, {t_v4f, res, img, coord, ImgOp_Lod, uconst(0)});
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, res, c}); out[c] = bcu(e);
        }
    }

    // image_load from a 3D texture (integer texel fetch through the combined sampler — DOLL's
    // color-grade LUT, #273): OpImage strips the sampler; OpImageFetch with (x,y,z) integer coords.
    uint32_t t_v3u_cache2 = 0;
    uint32_t t_v3u_fetch() { if (!t_v3u_cache2) { t_v3u_cache2 = id(); put(types, Op_TypeVector, {t_v3u_cache2, t_u32, 3}); } return t_v3u_cache2; }
    void image_fetch_3d(uint32_t binding, uint32_t x_bits, uint32_t y_bits, uint32_t z_bits, uint32_t out[4]) {
        uint32_t simg  = sampled_image_type(Dim_3D);
        uint32_t t_img3 = tex_img_dim[Dim_3D];   // OpImage's result type must be the pair's Image type
        uint32_t si    = id(); put(code, Op_Load,  {simg, si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {t_img3, img, si});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v3u_fetch(), coord, x_bits, y_bits, z_bits});
        uint32_t res   = id(); put(code, Op_ImageFetch, {t_v4f, res, img, coord, ImgOp_Lod, uconst(0)});
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, res, c}); out[c] = bcu(e);
        }
    }

    // --- STORAGE images (MIMG image_load / image_store WITHOUT a sampler; compute copy/blit) ---
    // Modeled with a UINT-sampled OpTypeImage, Format=Unknown, Sampled=2 (storage), so OpImageRead/Write
    // move raw 32-bit texels — an exact fit for our untyped-VGPR model (any real format reinterpretation
    // lives in the bound image view / T# descriptor). Requires the read/write-without-format caps.
    uint32_t t_v4u_cache = 0;
    uint32_t t_v4u() { if (!t_v4u_cache) { t_v4u_cache = id(); put(types, Op_TypeVector, {t_v4u_cache, t_u32, 4}); } return t_v4u_cache; }
    uint32_t t_v3u_c = 0;   // integer coordinate vector type (uvec3); 2D reuses the shared t_v2u()
    std::unordered_map<uint32_t, uint32_t> stg_img_type;   // (dim | arrayed<<8 | ms<<9) -> OpTypeImage id
    std::unordered_map<uint32_t, uint32_t> stg_img_var;    // binding -> storage-image OpVariable id
    bool declared_read_wo_fmt = false, declared_write_wo_fmt = false, declared_sampled1d = false, declared_ms = false, declared_msarray = false;
    static uint32_t stg_key(uint32_t dim, bool arrayed, bool ms) { return dim | (arrayed ? 0x100u : 0u) | (ms ? 0x200u : 0u); }
    // Declare (idempotently) a uint storage image of SPIR-V `dim` (arrayed = layer in the coord; ms =
    // multisampled) at set 0, `binding`. Each (dim,arrayed,ms) is a distinct OpTypeImage, keyed separately.
    void declare_storage_image(uint32_t binding, uint32_t dim, bool arrayed = false, bool ms = false) {
        if (dim == Dim_1D && !declared_sampled1d) {   // SPIR-V: Dim=1D needs Sampled1D; storage 1D also needs Image1D
            put(caps, Op_Capability, {Cap_Sampled1D});
            put(caps, Op_Capability, {Cap_Image1D});
            declared_sampled1d = true;
        }
        if (ms && !declared_ms) { put(caps, Op_Capability, {Cap_StorageImageMultisample}); declared_ms = true; }
        if (ms && arrayed && !declared_msarray) { put(caps, Op_Capability, {Cap_ImageMSArray}); declared_msarray = true; }
        uint32_t key = stg_key(dim, arrayed, ms);
        if (!stg_img_type.count(key)) {
            uint32_t ti = id();
            put(types, Op_TypeImage, {ti, t_u32, dim, 0, arrayed ? 1u : 0u, ms ? 1u : 0u, Img_Sampled_Storage, ImgFmt_Unknown});
            stg_img_type[key] = ti;
        }
        if (stg_img_var.count(binding)) return;
        uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_UniformConstant, stg_img_type[key]});
        uint32_t v = id();     put(types, Op_Variable,    {t_ptr, v, SC_UniformConstant});
        put(deco, Op_Decorate, {v, Dec_DescriptorSet, desc_set});
        put(deco, Op_Decorate, {v, Dec_Binding, binding});
        stg_img_var[binding] = v;
    }
    uint32_t stg_img_id(uint32_t dim, bool arrayed, bool ms = false) { return stg_img_type[stg_key(dim, arrayed, ms)]; }
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
        if (!declared_read_wo_fmt) { put(caps, Op_Capability, {Cap_StorageImageReadWithoutFormat}); declared_read_wo_fmt = true; }
        uint32_t img   = id(); put(code, Op_Load,      {stg_img_id(dim, arrayed, ms), img, stg_img_var[binding]});
        uint32_t coord = stg_coord(ncoord, coords);
        uint32_t res   = id();
        if (ms) put(code, Op_ImageRead, {t_v4u(), res, img, coord, ImgOp_Sample, sample});  // MSAA: read sample `sample`
        else    put(code, Op_ImageRead, {t_v4u(), res, img, coord});
        for (uint32_t c = 0; c < 4; c++) { uint32_t e = id(); put(code, Op_CompositeExtract, {t_u32, e, res, c}); out[c] = e; }
    }
    // image_store: OpImageWrite raw-bit VGPR components vals[0..3] as a uvec4 texel to the storage image.
    // When `predicated` (narrowed EXEC), the write is wrapped in a selection merge on `pred` (the per-lane
    // EXEC bool) so inactive lanes do not write — a real conditional store, like cbuf_store. (Image OOB is
    // not covered by robustBufferAccess, so guarding matters: it also skips a lane's write when EXEC is off
    // e.g. a grid-tail bounds check.)
    void image_write(uint32_t binding, uint32_t dim, bool arrayed, uint32_t ncoord, const uint32_t* coords,
                     const uint32_t vals[4], bool predicated = false, uint32_t pred = 0) {
        if (!declared_write_wo_fmt) { put(caps, Op_Capability, {Cap_StorageImageWriteWithoutFormat}); declared_write_wo_fmt = true; }
        uint32_t img   = id(); put(code, Op_Load, {stg_img_id(dim, arrayed), img, stg_img_var[binding]});
        uint32_t coord = stg_coord(ncoord, coords);
        uint32_t texel = id(); put(code, Op_CompositeConstruct, {t_v4u(), texel, vals[0], vals[1], vals[2], vals[3]});
        if (!predicated) { put(code, Op_ImageWrite, {img, coord, texel}); return; }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        put(code, Op_ImageWrite, {img, coord, texel});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
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
    // LDS (Local Data Share) — a workgroup-shared u32 array for compute ds_read/ds_write. Declared on
    // first use, sized to `lds_dwords`. RDNA2's per-workgroup LDS MAX is 64 KB (16384 dwords); the real
    // per-shader allocation is COMPUTE_PGM_RSRC2.LDS_SIZE. `lds_dwords` defaults to 4096 (16 KB) — a
    // shader whose real allocation exceeds that would ds_read/write past the array (OOB Workgroup
    // access, UB — NOT covered by robustBufferAccess), so recompile_valu raises it from the plumbed
    // size when known (#130). Kept at 16 KB by default because it must also stay within the target
    // device's VkPhysicalDeviceLimits::maxComputeSharedMemorySize (e.g. llvmpipe = 32 KB), so we can't
    // just declare the full 64 KB unconditionally.
    uint32_t lds_dwords = 4096;
    uint32_t lds_var = 0, t_ptr_wg_u32 = 0;
    void declare_lds() {
        if (lds_var) return;
        uint32_t len = uconst(lds_dwords);
        uint32_t t_arr = id();        put(types, Op_TypeArray, {t_arr, t_u32, len});
        uint32_t t_ptr_wg_arr = id(); put(types, Op_TypePointer, {t_ptr_wg_arr, SC_Workgroup, t_arr});
        lds_var = id();               put(types, Op_Variable, {t_ptr_wg_arr, lds_var, SC_Workgroup});
        t_ptr_wg_u32 = id();          put(types, Op_TypePointer, {t_ptr_wg_u32, SC_Workgroup, t_u32});
    }
    uint32_t lds_load(uint32_t idx) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32, p, lds_var, idx});
        uint32_t r = id(); put(code, Op_Load, {t_u32, r, p}); return r;
    }
    // Store to LDS[idx]; EXEC-predicated (conditional store) under a narrowed mask, like cbuf_store.
    void lds_store(uint32_t idx, uint32_t value, bool predicated, uint32_t pred) {
        if (!predicated) {
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32, p, lds_var, idx});
            put(code, Op_Store, {p, value}); return;
        }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32, p, lds_var, idx});
        put(code, Op_Store, {p, value});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }
    // s_barrier: workgroup execution + memory barrier (OpControlBarrier).
    void barrier() {
        put(code, Op_ControlBarrier, {uconst(Scope_Workgroup), uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel)});
    }

    // --- Wave model (cross-lane ops via a workgroup-as-wave + LDS). The compute shell dispatches one
    // 64-invocation workgroup per wave; gl_LocalInvocationID.x is the lane. A dedicated LDS scratch array
    // (separate from ds_read/write's lds_var) holds each lane's contribution so cross-lane reductions work.
    // Inactive lanes still EXECUTE (exec is a per-lane predication bool), so all 64 reach the barriers —
    // valid only at wave-uniform points (the caller must not emit these inside divergent control flow). ---
    uint32_t v_localid = 0, localid = 0, lds_wave = 0, t_ptr_wg_u32b = 0;
    void declare_wave_lds() {
        if (lds_wave) return;
        uint32_t t_arr = id();  put(types, Op_TypeArray, {t_arr, t_u32, uconst(64)});
        uint32_t t_ptr = id();  put(types, Op_TypePointer, {t_ptr, SC_Workgroup, t_arr});
        lds_wave = id();        put(types, Op_Variable, {t_ptr, lds_wave, SC_Workgroup});
        t_ptr_wg_u32b = id();   put(types, Op_TypePointer, {t_ptr_wg_u32b, SC_Workgroup, t_u32});
    }
    // v_mbcnt_lo/hi: count active lanes below this one. active_bool = this lane's mask bit (EXEC); acc =
    // src1 accumulator (bits); `lo` selects the [0,32) half (lo) or [32,64) half (hi). Combined lo→hi over
    // a wave = the lane's compaction index among active lanes. Populate LDS[lane]=active, barrier, then an
    // unrolled prefix-count over the half; trailing barrier so a following mbcnt can safely re-populate.
    uint32_t mbcnt(uint32_t active_bool, uint32_t acc_bits, bool lo) {
        declare_wave_lds();
        uint32_t bit = sel(active_bool, uconst(1), uconst(0));
        { uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32b, p, lds_wave, localid}); put(code, Op_Store, {p, bit}); }
        barrier();
        uint32_t sum = uconst(0);
        for (uint32_t i = (lo ? 0u : 32u); i < (lo ? 32u : 64u); i++) {
            uint32_t cond = ucmp(Op_ULessThan, uconst(i), localid);        // i < lane
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32b, p, lds_wave, uconst(i)});
            uint32_t v = id(); put(code, Op_Load, {t_u32, v, p});
            sum = b_iadd(sum, sel(cond, v, uconst(0)));
        }
        barrier();
        return b_iadd(acc_bits, sum);
    }
    uint32_t b_iadd(uint32_t a, uint32_t b_) { uint32_t r = id(); put(code, Op_IAdd, {t_u32, r, a, b_}); return r; }
    // Declare the two scalar-memory constant/vertex buffers (bindings 2 & 3) that SMEM / buffer_load_
    // format_* read. Called by every shell (compute/vertex/fragment) so cbuf_load works in each.
    // Requires t_u32 to already be declared. Unused by shaders without memory ops.
    void declare_cbufs(const ShaderResourceTable* rt = nullptr) {
        uint32_t t_rta_u = id(), t_struct_u = id(), t_ptr_sb_struct_u = id();
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
    // Logical AND / OR of two bools (EXEC narrowing / saveexec).
    uint32_t land(uint32_t a, uint32_t b_) { uint32_t r = id(); put(code, Op_LogicalAnd, {t_bool, r, a, b_}); return r; }
    uint32_t lor(uint32_t a, uint32_t b_)  { uint32_t r = id(); put(code, Op_LogicalOr,  {t_bool, r, a, b_}); return r; }

    void begin(uint32_t input_stride, const ShaderResourceTable* rt = nullptr,
               uint32_t local_x = 64, uint32_t local_y = 1, uint32_t local_z = 1) {
        stride = input_stride;
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
        put(exec, Op_ExecutionMode, {f_main, EM_LocalSize, local_x, local_y, local_z});
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
        uint32_t ld = id(); put(code, Op_Load, {t_v3u, ld, v_gid});
        gidx = id(); put(code, Op_CompositeExtract, {t_u32, gidx, ld, 0});
        uint32_t ldl = id(); put(code, Op_Load, {t_v3u, ldl, v_localid});
        for (uint32_t c = 0; c < 3; c++) {
            localid_comp[c] = id();
            put(code, Op_CompositeExtract, {t_u32, localid_comp[c], ldl, c});
        }
        localid = localid_comp[0];   // wave lane index
        uint32_t ldg = id(); put(code, Op_Load, {t_v3u, ldg, v_groupid});
        for (uint32_t c = 0; c < 3; c++) {
            groupid[c] = id();
            put(code, Op_CompositeExtract, {t_u32, groupid[c], ldg, c});
        }
    }
    // --- Fragment-shader shell: a location-0 vec4 color output; EXP MRT0 stores to it. ---
    uint32_t t_v4f = 0, v_color = 0;
    void begin_fragment(const ShaderResourceTable* rt = nullptr) { bool with_cbufs = rt != nullptr;
        t_void = id(); t_fn = id(); t_f32 = id(); t_u32 = id(); t_i32 = id(); t_bool = id();
        t_v4f = id(); uint32_t t_ptr_out = id(); v_color = id(); f_main = id(); uint32_t lbl = id(); glsl = id();
        put(caps, Op_Capability, {Cap_Shader});
        { std::vector<uint32_t> o{glsl}; pstr(o, "GLSL.std.450"); putv(extimp, Op_ExtInstImport, o); }
        put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
        is_fragment = true;
        desc_set = 1;   // PS resources live in descriptor set 1 (VS owns set 0) — no cross-stage binding collision
        exec_model = Exec_Fragment; iface = {v_color};   // EntryPoint deferred to finish()
        put(exec, Op_ExecutionMode, {f_main, EM_OriginUpperLeft});
        put(deco, Op_Decorate, {v_color, Dec_Location, 0});
        put(types, Op_TypeVoid, {t_void});
        put(types, Op_TypeFunction, {t_fn, t_void});
        put(types, Op_TypeFloat, {t_f32, 32});
        put(types, Op_TypeInt, {t_u32, 32, 0});
        put(types, Op_TypeInt, {t_i32, 32, 1});
        put(types, Op_TypeBool, {t_bool});
        put(types, Op_TypeVector, {t_v4f, t_f32, 4});
        put(types, Op_TypePointer, {t_ptr_out, SC_Output, t_v4f});
        put(types, Op_Variable, {t_ptr_out, v_color, SC_Output});
        if (with_cbufs) declare_cbufs(rt);   // only when the shader has memory ops (keeps no-op renders binding-free)
        put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
        put(code, Op_Label, {lbl}); cur_block = lbl;
    }
    // Write a vec4(r,g,b,a) (bit-operands) to the fragment color output (EXP MRT0).
    void export_color(uint32_t r, uint32_t g, uint32_t bl, uint32_t a) {
        uint32_t v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(r), bcf(g), bcf(bl), bcf(a)});
        put(code, Op_Store, {v_color, v});
    }

    // --- Vertex-shader shell: gl_VertexIndex input + gl_Position (member 0 of a gl_PerVertex Block). ---
    uint32_t v_vid = 0, v_pos = 0, t_ptr_out_v4f = 0;
    void begin_vertex(const ShaderResourceTable* rt = nullptr) { bool with_cbufs = rt != nullptr;
        t_void = id(); t_fn = id(); t_f32 = id(); t_u32 = id(); t_i32 = id(); t_bool = id();
        t_v4f = id();
        uint32_t t_ptr_in_i32 = id(); v_vid = id();
        uint32_t t_pv = id(), t_ptr_out_pv = id(); v_pos = id(); t_ptr_out_v4f = id();
        f_main = id(); uint32_t lbl = id(); glsl = id();
        put(caps, Op_Capability, {Cap_Shader});
        { std::vector<uint32_t> o{glsl}; pstr(o, "GLSL.std.450"); putv(extimp, Op_ExtInstImport, o); }
        put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
        exec_model = Exec_Vertex; iface = {v_vid, v_pos};   // EntryPoint deferred to finish()
        put(deco, Op_Decorate, {v_vid, Dec_BuiltIn, BI_VertexIndex});
        put(deco, Op_MemberDecorate, {t_pv, 0, Dec_BuiltIn, BI_Position});   // gl_PerVertex.gl_Position
        put(deco, Op_Decorate, {t_pv, Dec_Block});
        put(types, Op_TypeVoid, {t_void});
        put(types, Op_TypeFunction, {t_fn, t_void});
        put(types, Op_TypeFloat, {t_f32, 32});
        put(types, Op_TypeInt, {t_u32, 32, 0});
        put(types, Op_TypeInt, {t_i32, 32, 1});
        put(types, Op_TypeBool, {t_bool});
        put(types, Op_TypeVector, {t_v4f, t_f32, 4});
        put(types, Op_TypePointer, {t_ptr_in_i32, SC_Input, t_i32});
        put(types, Op_Variable, {t_ptr_in_i32, v_vid, SC_Input});
        put(types, Op_TypeStruct, {t_pv, t_v4f});
        put(types, Op_TypePointer, {t_ptr_out_pv, SC_Output, t_pv});
        put(types, Op_Variable, {t_ptr_out_pv, v_pos, SC_Output});
        put(types, Op_TypePointer, {t_ptr_out_v4f, SC_Output, t_v4f});
        if (with_cbufs) declare_cbufs(rt);   // vertex fetch (buffer_load_format_*) reads these
        put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
        put(code, Op_Label, {lbl}); cur_block = lbl;
    }
    // Load gl_VertexIndex as raw bits (VGPR v0 for a vertex shader).
    uint32_t load_vertex_index() { uint32_t r = id(); put(code, Op_Load, {t_i32, r, v_vid}); return i2u(r); }
    // Write vec4(x,y,z,w) (bit-operands) to gl_Position (EXP POS0).
    void export_position(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
        // PROSPER_FORCE_W: diagnostic — force the clip-space w to 1.0. Some shaders' factored MVP
        // multiply leaves w at 0 under our (still-incomplete) descriptor decode, collapsing the
        // perspective divide; forcing w=1 reveals whether the x/y are otherwise on-screen.
        if (getenv("PROSPER_FORCE_W")) w = uconst(0x3f800000u);   // raw bits of 1.0f (bcf bitcasts to float)
        uint32_t v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(x), bcf(y), bcf(z), bcf(w)});
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_out_v4f, p, v_pos, uconst(0)});
        put(code, Op_Store, {p, v});
    }

    // --- Interpolated I/O varyings: VS EXP PARAM_n (output) <-> FS v_interp attribute (input) ---
    std::unordered_map<uint32_t, uint32_t> in_varying, out_varying;
    uint32_t t_ptr_in_v4f = 0;
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
    // FS: gl_FragCoord (BuiltIn 15), lazily declared — used by the DPP quad_perm lowering.
    uint32_t v_fragcoord = 0;
    uint32_t fragcoord_var() {
        if (v_fragcoord) return v_fragcoord;
        if (!t_ptr_in_v4f) { t_ptr_in_v4f = id(); put(types, Op_TypePointer, {t_ptr_in_v4f, SC_Input, t_v4f}); }
        v_fragcoord = id(); put(types, Op_Variable, {t_ptr_in_v4f, v_fragcoord, SC_Input});
        put(deco, Op_Decorate, {v_fragcoord, Dec_BuiltIn, BI_FragCoord});
        iface.push_back(v_fragcoord); return v_fragcoord;
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

    std::vector<uint32_t> finish() {
        put(code, Op_Return, {}); put(code, Op_FunctionEnd, {});
        // EntryPoint is emitted here (not in begin_*) so lazily-declared Input/Output varyings — added
        // to `iface` as v_interp / EXP PARAM are encountered — appear in the interface list (SPIR-V 1.3).
        { std::vector<uint32_t> o{exec_model, f_main}; pstr(o, "main");
          for (uint32_t v : iface) o.push_back(v); putv(entry, Op_EntryPoint, o); }
        std::vector<uint32_t> m{0x07230203u, 0x00010300u, 0u, next_id, 0u};
        for (auto* s : {&caps, &extimp, &mem, &entry, &exec, &deco, &types, &code}) m.insert(m.end(), s->begin(), s->end());
        return m;
    }
};

}  // namespace

// Machine state during recompilation: the VGPR and SGPR files (VGPR/SGPR number -> current SSA bits
// id) and VCC (current bool condition). VGPRs and SGPRs are separate register files; VALU/EXP source
// operands may reference either (SGPR is a valid ALU operand), so both are resolved by operand_bits.
struct RegState {
    std::unordered_map<int, uint32_t> vreg, sreg;
    std::unordered_map<int, uint32_t> sreg_bool;   // SGPR (pairs) holding a saved per-lane mask (bool id)
    std::unordered_map<int, bool> sreg_bool_narrowed;  // was EXEC narrowed when this mask was saved? (restores it)
    std::unordered_map<int, uint32_t> sreg_srt;    // SGPR holding a descriptor -> its user_data/SRT byte offset
                                                   // (descriptor provenance: s_load_dwordx4 tags, s_buffer_load resolves)
    uint32_t vcc = 0;
    uint32_t scc = 0;          // scalar condition code (bool); set by s_cmp_*/SCC-writing SOP2, read by s_cselect
    uint32_t exec = 0;         // per-lane execution mask (bool); v_cmpx narrows it, output store honors it
    bool exec_narrowed = false; // true once EXEC is narrowed below all-lanes-on (so VGPR writes predicate)
    // PC-relative EMBEDDED TABLES (#273): mubuf pc -> the table's dwords, resolved by
    // detect_pcrel_tables (an s_getpc_b64-derived V# whose num_records is a known constant). The
    // shader BLOB carries the table; the recompiler folds it into a compile-time constant lookup.
    std::unordered_map<uint32_t, std::vector<uint32_t>> mubuf_pcrel_tables;
    // SCALAR-SPILL VGPR (#273 — DOLL's big post PS): the compiler packs excess wave-uniform scalars
    // into one VGPR's lanes via `v_writelane_b32 vN, sX, <const lane>` and reads them back with
    // `v_readlane_b32 sY, vN, <const lane>`. Per-invocation each (vgpr, lane) slot is just a named
    // scalar: vgpr -> lane -> SSA id. A vgpr used as a spill array must never be read as ordinary
    // per-lane data — operand_bits rejects that (fail-visible), keeping the model honest.
    std::unordered_map<int, std::unordered_map<int, uint32_t>> vgpr_lane_slots;
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
}
inline uint32_t vreg_old(SpirvCompute& b, RegState& rs, int idx) {
    auto it = rs.vreg.find(idx); return it == rs.vreg.end() ? b.uconst(0) : it->second;
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

// Is SGPR `R` provably DEAD at pc `target` — i.e. redefined before any read on the fall-through, so a
// write to it inside a divergent (execz) block linearized before `target` cannot be observed by later
// code? Sound/conservative: we only reason across the simple wave-uniform ALU formats whose SGPR source
// operands we can fully enumerate (at most a single reg or a 64-bit pair). At the first memory / control-
// flow / interp / export / unknown instruction (whose reads of R we can't bound — e.g. a wide T#/V#
// descriptor source) we give up and report NOT-dead. Checking value ∈ {R, R-1} covers both a 32-bit read
// of R and a 64-bit pair whose low half is R-1. (RE-TAG: divergent-block scalar liveness.)
inline bool sgpr_dead_at_merge(const std::vector<Rdna2Inst>& ins, uint32_t target, int R) {
    // A "dead by redefinition at pc P" claim is only sound if execution cannot ENTER the region
    // (target, P] from a branch elsewhere (which would skip the redef and reach a later read).
    auto entered_between = [&](uint32_t upto) -> bool {
        for (const auto& br : ins) {
            if (br.fmt != Rdna2Format::SOPP || sopp_is_noop(br)) continue;
            if (br.opcode < 0x02 || br.opcode > 0x09 || br.opcode == 0x03) continue;   // branches only
            uint32_t t = br.pc + br.len_dwords + (uint32_t)(int32_t)br.simm16;
            if (t > target && t <= upto) return true;
        }
        return false;
    };
    for (const auto& in : ins) {
        if (in.pc < target) continue;
        if (in.is_end) return true;                        // never read past here -> dead
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
                // after the merge). Branches (and anything else) invalidate the LINEAR scan — a skipped
                // redefinition would make the dead claim unsound — so bail conservatively.
                if (sopp_is_noop(in)) break;
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
                if (R >= in.dst.value && R < in.dst.value + (int)n) return !entered_between(in.pc);  // redefined
                break;
            }
            case Rdna2Format::SOP1: case Rdna2Format::SOP2: case Rdna2Format::SOPC:
            case Rdna2Format::VOP1: case Rdna2Format::VOP2: case Rdna2Format::VOP3: case Rdna2Format::VOPC:
            case Rdna2Format::EXP:   // EXP data sources are all VGPRs — it can never read an SGPR
                for (int k = 0; k < in.n_src; k++)
                    if ((in.src[k].kind == OperandKind::SGPR || in.src[k].kind == OperandKind::Special) &&
                        (in.src[k].value == R || in.src[k].value == R - 1)) return false;   // read before redef -> live
                // A VOP3B carry-out (sdst) is a 64-bit mask write — reads none of R beyond its sources.
                if (in.sdst.kind == OperandKind::SGPR &&
                    (in.sdst.value == R || in.sdst.value + 1 == R)) return !entered_between(in.pc);   // redefined (pair)
                // A VOPC e32 compare writes the whole VCC pair — kills both halves.
                if (in.fmt == Rdna2Format::VOPC &&
                    (in.dst.value == 106 || in.dst.value == 107) && (R == 106 || R == 107))
                    return !entered_between(in.pc);
                // A redefinition kills R for a plain numbered SGPR dst (s0..s105) — and now also for
                // VCC_LO/HI (106/107), whose implicit readers are enumerated above. EXEC/M0
                // (124/126/127) still can't be proven dead (implicit reads everywhere).
                if (in.dst.kind == OperandKind::SGPR && in.dst.value == R && R <= 107)
                    return !entered_between(in.pc);
                break;
            default:
                return false;   // memory/branch/interp/unknown: can't bound reads of R -> assume live
        }
    }
    return true;   // fell off the end without a read
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
            //   VOP3B 0x128-0x12A (write the carry-out SGPR pair/VCC).
            const bool scalar_side_effect =
                (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02) ||
                (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2A) ||
                (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 && in.opcode <= 0x12A);
            if (!scalar_side_effect &&
                (in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 || in.fmt == Rdna2Format::VOP3 ||
                 in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::DS ||
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

// FRAGMENT alpha-test / clip() discard via a SCALAR BRANCH. A per-lane condition (v_cmp -> VCC) is folded
// into a saved-EXEC survivor mask by a 64-bit wave-mask op (s_and/s_andn2_b64 sDST,sDST,vcc — which on HW
// sets SCC = "any lane survives"), then `s_cbranch_scc0 <fwd>` skips the shading when NO lane survives; the
// block then narrows EXEC (s_wqm exec, sDST) and shades, and the export lowers to an OpKill of the failed
// lanes. Per-invocation the wave early-out is a pure optimization — running the block for a lane that will
// be OpKill'd at export is harmless — so the branch is safe to LINEARIZE (drop it, run the block straight-
// line) exactly like a forward s_cbranch_execz. Recognize it by the mask op IMMEDIATELY preceding the
// branch (hints ignored). Returns the pc of each such branch. This is the shape of every Unity cutout /
// text draw; rejecting the branch dropped all of them (The Messenger's missing cutscene text, #102).
std::unordered_set<uint32_t> mask_test_branches(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> out;
    const Rdna2Inst* prev = nullptr;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (sopp_is_noop(in)) continue;                         // hints don't break the mask->branch pairing
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04 || in.opcode == 0x05) && in.simm16 > 0) {
            // scc0/scc1 FORWARD branch whose SCC was set by a 64-bit wave-mask op: SOP2 s_and_b64(0x0f) /
            // s_or_b64(0x11) / s_xor_b64(0x13) / s_andn2_b64(0x15), or SOP1 s_and/or_saveexec_b64 (0x24/0x25),
            // writing a plain SGPR-pair kill mask. (A branch on a v_cmp/s_cmp SCC is a REAL uniform-if and is
            // NOT matched — prev would be a SOPC/ALU, not a mask op.)
            if (prev) {
                bool mask_sop2 = prev->fmt == Rdna2Format::SOP2 &&
                    (prev->opcode == 0x0f || prev->opcode == 0x11 || prev->opcode == 0x13 || prev->opcode == 0x15);
                bool mask_saveexec = prev->fmt == Rdna2Format::SOP1 && (prev->opcode == 0x24 || prev->opcode == 0x25);
                // The kill mask may live in a plain SGPR pair (s0..s105) OR in VCC itself — DOLL's
                // alpha-cull PS does `s_andn2_b64 vcc, exec, vcc; s_cbranch_scc0 <null-export>` then
                // `s_mov_b64 exec, vcc; export`. The SOP2 dst field decodes VCC_LO as SGPR 106, and
                // emit_alu's mask ops route a 106/107 dst to rs.vcc, so the same linearization holds
                // (the branch is a whole-wave early-out; per-invocation the export's OpKill covers it).
                if ((mask_sop2 || mask_saveexec) && prev->dst.kind == OperandKind::SGPR && prev->dst.value <= 106)
                    out.insert(in.pc);
            }
        }
        prev = &in;
    }
    return out;
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
                p.fmt == Rdna2Format::VOPC || p.fmt == Rdna2Format::MIMG || p.fmt == Rdna2Format::MUBUF ||
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

// A recognized COUNTED loop (the game's MSAA-resolve / accumulation shape): a single backward
// unconditional branch (the back-edge) to a header, with exactly one forward SCC exit branch inside.
// Anything more complex (nested loops, VCC/EXECNZ exits, multiple back-edges, mid-loop s_branch) is
// rejected — the recompiler then falls back to the straight-line path (which fails on the loop, as before).
struct CountedLoop {
    bool found = false;
    uint32_t header_pc = 0;       // loop header (target of the back-edge; condition eval starts here)
    uint32_t exit_branch_pc = 0;  // the forward s_cbranch_scc0/scc1 that leaves the loop
    uint32_t backedge_pc = 0;     // the backward s_branch
    uint32_t exit_pc = 0;         // first pc after the loop (== backedge_pc + its length)
    bool exit_on_scc0 = true;     // s_cbranch_scc0 (exit when SCC==0) vs s_cbranch_scc1 (exit when SCC==1)
};
inline uint32_t branch_target(const Rdna2Inst& in) { return in.pc + in.len_dwords + (uint32_t)(int32_t)in.simm16; }

CountedLoop detect_counted_loop(const std::vector<Rdna2Inst>& ins) {
    CountedLoop L;
    const Rdna2Inst* back = nullptr; int nback = 0;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x02 && in.simm16 < 0) { back = &in; nback++; }
    }
    if (nback != 1) return L;                       // 0 -> no loop; >1 -> nested/multiple (unhandled)
    const uint32_t header = branch_target(*back);
    bool header_ok = false;
    for (const auto& in : ins) if (in.pc == header && in.pc < back->pc) { header_ok = true; break; }
    if (!header_ok) return L;
    const uint32_t exit_pc = back->pc + back->len_dwords;
    const Rdna2Inst* exitbr = nullptr; int nexit = 0;
    for (const auto& in : ins) {                    // scan the loop body [header, back-edge]
        if (in.pc < header || in.pc > back->pc || in.fmt != Rdna2Format::SOPP) continue;
        switch (in.opcode) {
            case 0x04: case 0x05:                   // s_cbranch_scc0 / scc1 — must be the single exit
                if (branch_target(in) != exit_pc) return L;
                exitbr = &in; nexit++; break;
            case 0x02:                              // any s_branch other than the back-edge -> reject
                if (&in != back) return L; break;
            case 0x06: case 0x07: case 0x09:        // vcc / execnz branches -> reject
                return L;
            case 0x08: default: break;              // execz (forward, predication-handled) / hints ok
        }
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
// Per-invocation semantics: THIS lane iterates while its EXEC bit (rs.exec, a bool) holds after the
// header recompute — the canonical `while (cond) body` divergent loop. Hardware keeps the whole wave
// looping until EVERY lane's bit clears, but a cleared lane's vector writes are masked from then on,
// so exiting the lane immediately at its own EXECZ is value-identical for everything it can observe
// EXCEPT scalar state advanced by the extra wave iterations (a loop counter read after the loop
// would show the wave's MAX trip count, not this lane's) — compiled code consumes such counters
// inside the loop, so this is the standard per-invocation approximation. CONFIDENCE: MED, gated by
// spirv-val + execution kernels + the live-boot A/B.
//
// A second flavor (DOLL's scalar-indexed unroll): the back-edge is a backward s_cbranch_EXECNZ and
// the header IS the execz exit (empty condition region); the body may carry an extra forward
// vccz/execz BREAK to the exit. A break lowers as a plain forward IF over the remainder of the body
// (skip-to-backedge): the broken lane's EXEC bit is already clear (the compiled break condition
// mirrors the exec recompute), so the next header check exits it — we REQUIRE the execnz back-edge
// (exec-governed continue) for any loop carrying breaks, which makes that reasoning structural.
struct DivLoop {
    enum class Condition : uint8_t { Exec, Vcc };
    uint32_t header_pc = 0;        // back-edge target; condition region = [header_pc, exit_branch_pc)
    uint32_t exit_branch_pc = 0;   // canonical forward execz/vccz branch whose target is exit_pc
    uint32_t backedge_pc = 0;      // backward s_branch (unconditional) or s_cbranch_execnz
    uint32_t exit_pc = 0;          // backedge_pc + its length (first pc after the loop)
    std::vector<uint32_t> break_pcs;   // extra forward vccz/execz -> exit_pc (lowered as body ifs)
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
        case Rdna2Format::VOP2: case Rdna2Format::VOP3: case Rdna2Format::VOP3P:
            return 1;
        case Rdna2Format::DS:
            return in.opcode == 0x36 || in.opcode == 0xb1 ? 1 : 0;
        case Rdna2Format::MUBUF:
            switch (in.opcode) {
                case 0x1: case 0x5: case 0xD: return 2;
                case 0x2: case 0x6: case 0xF: return 3;
                case 0x3: case 0x7: case 0xE: return 4;
                default: return 1;
            }
        case Rdna2Format::MIMG: {
            if (in.opcode == 0x47 || in.opcode == 0x57) return 4;  // gather4 always returns four texels
            uint32_t n = 0;
            for (uint32_t c = 0; c < 4; ++c) n += (in.mimg_dmask >> c) & 1u;
            return n ? n : 4;                                     // unknown/empty mask: reject conservatively
        }
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
    auto is_exec = [](const Operand& operand) {
        return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
               (operand.value == 126 || operand.value == 127);
    };
    return is_exec(in.dst) || is_exec(in.sdst);
}

// A scalar VCCZ branch tests whether ANY active lane set VCC, not this lane's bit. Lowering that
// branch as per-invocation structured control flow is exact only when the VCC producer is provably
// uniform across the wave. Dead Cells' light loops use the narrow compiler shape
//   v_cvt_i32_f32 vBOUND, sBOUND; ...; v_cmp_* vcc, sCOUNTER, vBOUND; s_cbranch_vccz EXIT
// so every active lane makes the same comparison. Keep the proof deliberately local and reject a
// varying/unresolved VGPR bound rather than silently changing wave semantics.
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
        case Rdna2Format::VOP3:  return in.opcode >= 0x140 && in.opcode < 0x300 &&
                                        sgpr_dst_misses_vcc(1);    // plain-VALU window only
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

    auto uniform_operand = [&](const Operand& operand) -> bool {
        if (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::InlineInt ||
            operand.kind == OperandKind::InlineFloat || operand.kind == OperandKind::Literal)
            return true;
        if (operand.kind != OperandKind::VGPR) return false;
        for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
            if (it->pc >= compare->pc) continue;
            const uint32_t writes = vgpr_write_count(*it);
            if (!writes || operand.value < it->dst.value ||
                operand.value >= it->dst.value + (int32_t)writes) continue;
            const bool uniform_unary = it->fmt == Rdna2Format::VOP1 &&
                (it->opcode == 0x01 || (it->opcode >= 0x05 && it->opcode <= 0x08));
            const bool uniform_definition = it->dst.value == operand.value && uniform_unary &&
                !it->has_modifier && !it->has_dpp && it->sdwa_dst_sel == 6 &&
                it->sdwa_src0_sel == 6 && it->n_src == 1 &&
                (it->src[0].kind == OperandKind::SGPR ||
                 it->src[0].kind == OperandKind::InlineInt ||
                 it->src[0].kind == OperandKind::InlineFloat ||
                 it->src[0].kind == OperandKind::Literal);
            if (!uniform_definition) return false;
            // The write makes every currently active lane uniform. It remains so only while EXEC is
            // unchanged; a later widen could reactivate lanes that retained an older varying value.
            for (const auto& between : ins)
                if (between.pc > it->pc && between.pc < compare->pc &&
                    instruction_may_change_exec(between)) return false;
            return true;
        }
        return false;
    };
    for (uint32_t i = 0; i < compare->n_src; ++i)
        if (!uniform_operand(compare->src[i])) return false;
    return compare->n_src != 0;
}

// Returns the loops in header-pc order, or {} when any backward branch doesn't fit the shape (the
// caller then rejects the stream loudly, exactly as before this feature). `safe` carries the
// linearized branches (waterfalls etc.) which are not loop back-edges.
std::vector<DivLoop> detect_divergent_loops(const std::vector<Rdna2Inst>& ins,
                                            const std::unordered_set<uint32_t>& safe) {
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
    // Loops must be strictly DISJOINT and in order (nested loops are not modeled). Backward branches
    // appear in pc order, so headers must too — and each loop must end before the next begins.
    for (size_t i = 1; i < out.size(); i++)
        if (out[i].header_pc < out[i - 1].exit_pc) return {};
    // Pass 2: validate each loop's interior branches and find the canonical exit + breaks.
    for (size_t li = 0; li < out.size(); li++) {
        DivLoop& L = out[li];
        const bool execnz = backedge_execnz[li];
        for (const auto& in : ins) {
            if (in.is_end || in.pc >= L.backedge_pc) break;
            if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
            switch (in.opcode) {
                case 0x02: case 0x04: case 0x05: case 0x06: case 0x07: case 0x08: case 0x09: break;
                default: continue;   // hints
            }
            if (in.simm16 < 0) return {};                       // second back-edge inside -> nested loop
            if (safe.count(in.pc)) continue;                    // linearized (kill-mask / safe-execz)
            uint32_t tgt = branch_target(in);
            if (in.opcode == 0x02) {                            // forward s_branch: an else-arm terminator
                if (tgt > L.backedge_pc) return {};             // may not leave the body
                continue;                                       // (validated by detect_forward_ifs)
            }
            if (tgt > L.exit_pc) return {};                     // conditional jumping past the loop
            if (tgt == L.exit_pc) {                             // an exit test
                if (!L.exit_branch_pc) {                        // first one = the canonical exit
                    if (in.opcode == 0x08) {                    // execz -> EXIT
                        L.condition = DivLoop::Condition::Exec;
                    } else if (in.opcode == 0x06 && !execnz &&
                               vcc_exit_is_wave_uniform(ins, in.pc)) {
                        // The compare is uniform, so VCC is either set for every active lane or empty.
                        // Compute still never calls this detector; its general VCC contract remains a
                        // wave mask requiring a reduction (#590).
                        L.condition = DivLoop::Condition::Vcc;
                    } else {
                        return {};
                    }
                    L.exit_branch_pc = in.pc;
                } else {                                        // later ones = breaks
                    if (in.opcode != 0x06 && in.opcode != 0x08) return {};  // vccz/execz only
                    if (!execnz) return {};                     // breaks need the exec-governed back-edge
                    L.break_pcs.push_back(in.pc);
                }
            }
            // (tgt <= backedge_pc: an interior forward if — validated by detect_forward_ifs.)
        }
        if (!L.exit_branch_pc) return {};                       // no exit test: not this shape
        // The canonical exit must be the FIRST branch in the loop, so the condition region
        // [header, exit_branch) is branch-free (it is emitted straight-line).
        for (const auto& in : ins) {
            if (in.pc >= L.exit_branch_pc) break;
            if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
            if (in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03 && !safe.count(in.pc)) return {};
        }
        // The execnz flavor's unconditional-continue lowering requires the header check to
        // immediately re-test EXEC (empty condition region) — see the shape comment.
        if (execnz && L.exit_branch_pc != L.header_pc) return {};
    }
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
    // (v_cmpx … s_cbranch_execz — DOLL's FXAA PS). Per-invocation the condition is THIS lane's EXEC
    // bool; the block may itself narrow/restore EXEC (saveexec idioms), so EXEC is phi'd at the merge.
    bool on_exec = false;
    // IF/ELSE (#273): the then-arm [branch_pc+1, sb_pc) is terminated by `s_branch merge_pc` at sb_pc
    // (the instruction immediately before target_pc); the else-arm starts at target_pc and runs to
    // merge_pc (or to the enclosing region's end when merge_pc is the shared OUTER merge — DOLL's
    // color-grade if/else-if cascade, where every arm's s_branch jumps to the outermost merge).
    bool has_else = false;
    uint32_t sb_pc = 0, merge_pc = 0;
};
// allow_vcc: also accept a forward s_cbranch_vccz/vccnz. Only the per-invocation VS/FS shells set this —
// there each SPIR-V invocation IS one lane, so branching on this lane's VCC bit (rs.vcc) is exactly the
// per-vertex / per-pixel divergent-if the hardware runs. The 64-lane COMPUTE shell must NOT (its VCC is a
// wave mask needing a wave-uniform reduce — test_recompile_coverage guards this), so it leaves allow_vcc
// false and vcc branches fall through to straight-line (which rejects them).
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
// branch). s_branch / execz / execnz still reject wholesale, exactly like detect_forward_if.
// CONFIDENCE: HIGH on the structure (guarded by the phi machinery shared with the single-if path).
std::vector<ForwardIf> detect_forward_ifs(const std::vector<Rdna2Inst>& ins, bool allow_vcc,
                                          const uint32_t* code, size_t dwords,
                                          const std::unordered_set<uint32_t>* skip = nullptr,
                                          const std::vector<DivLoop>* loops = nullptr,
                                          bool* rejected = nullptr,
                                          bool uniform_vcc_compute = false) {
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
                return reject();
            case 0x08:                                               // execz: safe-linearized predication
                if (skip && skip->count(in.pc)) continue;            // branch -> emit_alu no-ops it; else a
                if (loop_exit(in.pc)) continue;                      // loop exit test: the loop emitter's
                if (!allow_vcc) return reject();                     // OpBranchConditional consumes it
                break;                                               // DIVERGENT-REGION if (per-invocation
                                                                     // stages only — compute has wave VCC/EXEC)
            case 0x06: case 0x07:                                    // vccz / vccnz
                if (loop_exit(in.pc)) continue;                      // canonical loop condition: loop emitter owns it
                if (!allow_vcc) {
                    // Compute (#590, extending #615): accept the forward VCC if ONLY under the same
                    // uniformity proof as the VCC-exit loops — every compare input scalar/inline or a
                    // uniform-VOP1-from-scalar VGPR, and no possible VCC rewrite between compare and
                    // branch (cannot_write_vcc walk). Every lane's bool is then identical, so the
                    // wave-empty vccz test lowers to this invocation's bool. Varying compares keep
                    // rejecting loudly (per-invocation lowering of a varying wave test is wrong).
                    if (!uniform_vcc_compute || !vcc_exit_is_wave_uniform(ins, in.pc)) return reject();
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
        if (compute_uniform_vcc && loop_break(in.pc)) return reject();
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
            if (!code || tgt >= dwords) return reject();             // verify the target terminates (see
            std::vector<Rdna2Inst> tail;                             // detect_forward_if for the rationale)
            rdna2_walk(code + tgt, dwords - tgt, tail);
            bool ends_immediately = false;
            for (const auto& ti : tail) {
                if (ti.is_end) { ends_immediately = true; break; }
                if (ti.fmt == Rdna2Format::SOPP && ti.opcode == 0x00) continue;
                break;
            }
            if (!ends_immediately) return reject();
            tgt = end_pc; early = true;
        }
        if (tgt <= in.pc || tgt > end_pc) return reject();           // must be forward, within the stream
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
                if (sb.simm16 <= 0) return reject();                 // backward else-jump: a loop, not an if
                uint32_t lm = branch_target(sb);
                if (lm <= tgt || lm > end_pc) return reject();       // merge must be forward, in-stream
                F.has_else = true; F.sb_pc = sb.pc; F.merge_pc = lm;
                break;
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
                     (r.opcode == 0x365 || r.opcode == 0x366))) return reject();
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
        if (!claimed) return reject();
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
                if (!clampable) return reject();         // plain if / loop escaping its region: not a tree
                span_end = open.back();                  // cascade if/else: clamped to the enclosing
            }                                            // region for nesting purposes (see above)
            open.push_back(span_end);
            if (take_loop) li++; else fi++;
        }
    }
    return out;
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
                else vregs.insert(in.dst.value); break;
            case Rdna2Format::VOP2: case Rdna2Format::VOP3P:
                vregs.insert(in.dst.value); break;
            case Rdna2Format::VOP3:
                if (in.opcode == 0x360) sregs.insert(in.dst.value);       // v_readlane -> SGPR
                else vregs.insert(in.dst.value); break;                   // (writelane: slots, not SSA)
            case Rdna2Format::DS:                                          // ds_read writes one VGPR
                if (in.opcode == 0x36 || in.opcode == 0xb1) vregs.insert(in.dst.value); break;
            case Rdna2Format::MUBUF: case Rdna2Format::MIMG:
                for (uint32_t k = 0; k < vgpr_write_count(in); ++k)
                    vregs.insert(in.dst.value + (int)k);
                break;
            case Rdna2Format::SOP1: case Rdna2Format::SOP2: case Rdna2Format::SOPK:
                sregs.insert(in.dst.value); break;
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
// in the per-invocation model, and M0/SCC/ttmp aren't modeled at all). Only SGPR_NULL (field 125)
// has a defined value, 0. Previously every Special silently read as 0 and the shader computed
// garbage (#134); now it rejects, matching the SDWA/DPP reject-rather-than-miscompute discipline.
uint32_t operand_bits(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, const Operand& o, bool* ok = nullptr) {
    switch (o.kind) {
        case OperandKind::VGPR: {
            // A scalar-spill vgpr (v_writelane slots) has no per-lane data value — reject, don't read 0.
            if (rs.vgpr_lane_slots.count(o.value)) { if (ok) *ok = false; return b.uconst(0); }
            auto it = rs.vreg.find(o.value); return it == rs.vreg.end() ? b.uconst(0) : it->second; }
        case OperandKind::SGPR: { auto it = rs.sreg.find(o.value); return it == rs.sreg.end() ? b.uconst(0) : it->second; }
        case OperandKind::InlineInt:   return b.uconst((uint32_t)o.value);
        case OperandKind::InlineFloat: return b.uconst(fbits(inline_float_value((uint32_t)o.value)));
        case OperandKind::Literal:     return b.uconst(in.literal);
        case OperandKind::Special:
            if (o.value == 125) return b.uconst(0);   // SGPR_NULL: the one Special whose data value IS 0
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
            if (ok) *ok = false;
            return b.uconst(0);
        default:
            if (ok) *ok = false;
            return b.uconst(0);
    }
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
            // 64-bit per-lane MASK ops (EXEC / VCC / saved masks). In our per-invocation model a wave
            // mask is a single bool for this lane. EXEC=SGPR 126/127, VCC=106/107; a saved mask lives
            // in sreg_bool. These implement divergent control flow (if/endif via saveexec + restore).
            if (in.opcode == 0x04 || in.opcode == 0x0a || in.opcode == 0x24 || in.opcode == 0x25) {
                auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
                auto src_mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 106 || o.value == 107) return rs.vcc;    // VCC
                    if (o.value == 126 || o.value == 127) return rs.exec;   // EXEC
                    if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second; }
                    if (o.kind == OperandKind::InlineInt) { if (o.value == -1) return b.btrue();
                        if (o.value == 0) return b.bfalse(); }
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
                    // SCC = (mask != 0) is a CROSS-lane reduction our per-lane model can't form, so — like
                    // every mask op here (s_mov_b64/saveexec) — we intentionally do not write rs.scc.
                    uint32_t m = src_mask(in.src[0]);
                    if (!m) { ok = false; return true; }
                    if (is_exec(in.dst)) {
                        if (is_exec(in.src[0])) { /* exec <- wqm(exec): identity; exec & narrowed unchanged */ }
                        else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                            rs.exec = b.btrue(); rs.exec_narrowed = false;      // exec = all lanes on
                        } else { rs.exec = m; rs.exec_narrowed = true; }        // replaced by a (maybe narrowed) mask
                    } else { rs.sreg_bool[in.dst.value] = m; rs.sreg_bool_narrowed[in.dst.value] = true; }  // conservative: WQM widens
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
                        if (!m && !data_copied) ok = false;
                    }
                } else {                                    // s_and/or_saveexec_b64 sDST, src
                    rs.sreg_bool[in.dst.value] = rs.exec;   // save current EXEC to the dest SGPR pair
                    rs.sreg_bool_narrowed[in.dst.value] = rs.exec_narrowed;   // ...and its narrowed-state
                    uint32_t m = src_mask(in.src[0]);
                    if (!m) ok = false;
                    else { rs.exec = (in.opcode == 0x24) ? b.land(rs.exec, m) : b.lor(rs.exec, m);
                           rs.exec_narrowed = true; }
                }
                return true;
            }
            if (in.opcode == 0x1f) {   // s_getpc_b64
                // Accepted ONLY when the pcrel pre-pass FOLDED an embedded-table load from this
                // shader (rs.mubuf_pcrel_tables non-empty) — the pair then only feeds that folded
                // chain. Otherwise the PC would flow into unmodeled address math: keep rejecting.
                if (rs.mubuf_pcrel_tables.empty()) { ok = false; return true; }
                for (int k = 0; k < 2; k++) {
                    rs.sreg.erase(in.dst.value + k); rs.sreg_srt.erase(in.dst.value + k);
                }
                return true;
            }
            uint32_t a = val(in.src[0]); uint32_t& d = rs.sreg[in.dst.value];
            switch (in.opcode) {
                case 0x03: d = a; break;                    // s_mov_b32
                case 0x07: d = b.iun(Op_Not, a); break;     // s_not_b32
                default: ok = false;
            }
            return true;
        }
        case Rdna2Format::SOP2: {
            if (in.opcode == 0x0b) {   // s_cselect_b64: 64-bit MASK dst = SCC ? src0 : src1 (mask domain)
                // Operands/dest are wave masks (EXEC/VCC/saved/inline), NOT uint bits — resolve like the
                // SOP1 mask ops and select in the bool domain. (s_cselect_b32 0x0a stays in the uint path.)
                auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
                auto mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 106 || o.value == 107) return rs.vcc;
                    if (o.value == 126 || o.value == 127) return rs.exec;
                    if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second; }
                    if (o.kind == OperandKind::InlineInt) { if (o.value == -1) return b.btrue();
                        if (o.value == 0) return b.bfalse(); }
                    return 0;   // not a recognizable mask
                };
                uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (!m0 || !m1) { ok = false; return true; }
                uint32_t r = b.bsel(rs.scc, m0, m1);
                if (is_exec(in.dst)) { rs.exec = r; rs.exec_narrowed = true; }
                else if (in.dst.value == 106 || in.dst.value == 107) { rs.vcc = r; rs.sreg_bool_narrowed[in.dst.value] = true; }
                else { rs.sreg_bool[in.dst.value] = r; rs.sreg_bool_narrowed[in.dst.value] = true; }  // conservative flag
                return true;
            }
            if (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 || in.opcode == 0x15) {
                // 64-bit wave-mask LOGICAL ops on per-lane bools: s_and_b64(0x0f)/s_or_b64(0x11)/
                // s_xor_b64(0x13)/s_andn2_b64(0x15, = m0 AND NOT m1). Used for lane-mask arithmetic around
                // divergent control flow / ballot. SCC=(result!=0) is a cross-lane reduction we can't form
                // per-lane, so (like every mask op) we don't write SCC. Same mask-resolution as s_cselect_b64.
                auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
                auto mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 106 || o.value == 107) return rs.vcc;
                    if (o.value == 126 || o.value == 127) return rs.exec;
                    if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second; }
                    if (o.kind == OperandKind::InlineInt) { if (o.value == -1) return b.btrue();
                        if (o.value == 0) return b.bfalse(); }
                    return 0;
                };
                uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (!m0 || !m1) { ok = false; return true; }
                uint32_t r = in.opcode == 0x0f ? b.land(m0, m1)
                           : in.opcode == 0x11 ? b.lor(m0, m1)
                           : in.opcode == 0x13 ? b.bsel(m0, b.bsel(m1, b.bfalse(), b.btrue()), m1)   // xor = m0?!m1:m1
                           : b.land(m0, b.bsel(m1, b.bfalse(), b.btrue()));                            // andn2 = m0 & !m1
                if (is_exec(in.dst)) { rs.exec = r; rs.exec_narrowed = true; }
                else if (in.dst.value == 106 || in.dst.value == 107) { rs.vcc = r; rs.sreg_bool_narrowed[in.dst.value] = true; }
                else { rs.sreg_bool[in.dst.value] = r; rs.sreg_bool_narrowed[in.dst.value] = true; }
                return true;
            }
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
                    uint32_t cin = b.sel(rs.scc, b.uconst(1), b.uconst(0));
                    uint32_t s1 = b.ibin(Op_IAdd, a, c);
                    uint32_t k1 = b.ucmp(Op_ULessThan, s1, a);        // wrap in a+c
                    d = b.ibin(Op_IAdd, s1, cin);
                    uint32_t k2 = b.ucmp(Op_ULessThan, d, s1);        // wrap in +cin
                    rs.scc = b.bsel(k1, b.btrue(), k2); break;
                }
                // s_min/max (0x06 min_i32, 0x07 min_u32, 0x08 max_i32, 0x09 max_u32): SCC = "src0 was
                // selected". The RDNA2 ISA min/max SCC split is intentionally ASYMMETRIC so a min/max
                // pair partitions ties consistently: S_MIN uses strict `S0 < S1`, S_MAX uses non-strict
                // `S0 >= S1` (S_MAX_I32: D=(S0>=S1)?S0:S1, SCC=(S0>=S1)). Using strict `>` for max set
                // SCC=0 on exact-equality operands where hardware sets 1 (#397) — D is unaffected (SMax
                // returns the equal value either way), so it was purely an SCC-flag defect that could
                // mis-step a downstream s_cbranch_scc/s_cselect on a tie. Round-trip llvm-mc gfx1010:
                // 0x83000201/0x83800201/0x84000201/0x84800201. CONFIDENCE: HIGH.
                case 0x06: rs.scc = b.scmp(Op_SLessThan, a, c);         d = b.sext2(Glsl_SMin, a, c); break;
                case 0x07: rs.scc = b.ucmp(Op_ULessThan, a, c);         d = b.uext2(Glsl_UMin, a, c); break;
                case 0x08: rs.scc = b.scmp(Op_SGreaterThanEqual, a, c); d = b.sext2(Glsl_SMax, a, c); break;
                case 0x09: rs.scc = b.ucmp(Op_UGreaterThanEqual, a, c); d = b.uext2(Glsl_UMax, a, c); break;
                case 0x0A: d = b.sel(rs.scc, a, c); break;           // s_cselect_b32: SCC ? src0 : src1
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
                case 0x21:   // s_lshr_b64 — only the NGG wave-packing form (dst = EXEC) is modeled: it sets
                             // EXEC to the count of active vertices/primitives in the wave. A per-invocation
                             // SPIR-V shader has no wave to pack (each invocation is one vertex), so leave
                             // EXEC full — no narrowing. Non-EXEC 64-bit shifts stay unsupported.
                             // (RE-TAG: NGG exec packing.)
                    if (in.dst.value != 126 && in.dst.value != 127) ok = false;
                    break;
                case 0x26: d = b.ibin(Op_IMul, a, c); break;         // s_mul_i32 (low 32 bits; no SCC)
                case 0x31: d = b.ibin(Op_IAdd, b.ibin(Op_ShiftLeftLogical, a, b.uconst(4)), c); break;  // s_lshl4_add_u32 = (src0<<4)+src1
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
                case 0x29: {   // s_bfe_u64: 64-bit unsigned bitfield extract of the SGPR pair src0=[lo,hi];
                    // offset=src1[5:0], width=src1[22:16]; dst is a pair. (a = low reg; read the high reg too.)
                    auto sread = [&](int r) -> uint32_t { auto it = rs.sreg.find(r); return it == rs.sreg.end() ? b.uconst(0) : it->second; };
                    uint32_t off = b.ibin(Op_BitwiseAnd, c, b.uconst(0x3f));
                    uint32_t wid = b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, c, b.uconst(16)), b.uconst(0x7f));
                    uint32_t res = b.bfe_u64(b.u64_from_lohi(a, sread(in.src[0].value + 1)), off, wid);
                    uint32_t lo = b.u64_lo(res), hi = b.u64_hi(res);
                    rs.sreg[in.dst.value] = lo; rs.sreg[in.dst.value + 1] = hi;   // (map insert may rehash — don't use `d` after)
                    rs.scc = b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseOr, lo, hi), b.uconst(0));
                    break;
                }
                default: ok = false;
            }
            return true;
        }
        case Rdna2Format::VOP3P: {
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
                if (half && in.src[k].kind != OperandKind::InlineFloat && in.src[k].kind != OperandKind::InlineInt)
                    v = b.unpack_half(v, (in.vop3p_opsel >> k) & 1u);
                if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                if (in.src_neg[k]) v = b.fbin(Op_FSub, b.uconst(0), v);
                return v;
            };
            uint32_t r = b.fbin(Op_FAdd, b.fbin(Op_FMul, mixv(0), mixv(1)), mixv(2));
            if (in.clamp) r = b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, r, b.uconst(fbits(1.0f))), b.uconst(fbits(0.0f)));
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
                default: ok = false;
            }
            return true;
        }
        case Rdna2Format::SOPK: {
            // 16-bit-immediate scalar ops. s_movk_i32 (0x00): dst = sign-extend(simm16). The decoder
            // already sign-extended simm16 to 32 bits. (s_cmovk/s_cmpk/s_addk/s_mulk deferred.)
            switch (in.opcode) {
                case 0x00: rs.sreg[in.dst.value] = b.uconst((uint32_t)in.simm16); break;   // s_movk_i32
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
            // DPP16 quad_perm on src0 (#273): reconstruct the selected quad lane's value from
            // screen-space derivatives (fragment-only; v_mov is the observed carrier — the manual
            // ddx/ddy idiom). Other VOP1 ops with DPP stay rejected (derivatives of non-float moves
            // have no meaning in this lowering).
            if (in.has_dpp) {
                if (!b.is_fragment || in.opcode != 0x01) { ok = false; return true; }
                a = b.dpp_quad(a, in.dpp_ctrl);
            }
            // SDWA float source modifiers on src0 (abs then neg) — only set on float ops by the assembler.
            if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
            if (in.src_neg[0]) a = b.fbin(Op_FSub, b.uconst(0), a);
            if (in.opcode == 0x02) {   // v_readfirstlane_b32: SGPR dst = value of the lowest active lane
                // Cross-lane broadcast. Our per-lane scalar model has no cross-lane reduction, so we use
                // THIS lane's value. SPECULATIVE(confidence: med): exact only when src0 is wave-uniform —
                // which is the standard use (reading a uniformly-computed VGPR into an SGPR, e.g. the
                // integer-divide reciprocal in the game's shaders). Writes an SGPR, not a VGPR.
                rs.sreg[in.dst.value] = a; return true;
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
                case 0x0A: d = b.pack_half_lo(a); break;              // v_cvt_f16_f32
                case 0x0B: d = b.unpack_half(a, 0); break;            // v_cvt_f32_f16
                // v_cvt_off_f32_i4: sign-extend the low 4-bit integer and scale by 1/16.
                // AMD RDNA2 ISA: "4-bit signed int to 32-bit float"; LLVM's intrinsic
                // contract specifies result = 0.0625f * src_i4. This is the only opcode
                // that blocked The Messenger's 1024x32 grading-LUT producer (#527).
                case 0x0E:
                    d = b.fbin(Op_FMul,
                               b.cvt_i2f(b.bfe_s(a, b.uconst(0), b.uconst(4))),
                               b.uconst(fbits(0.0625f)));
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
                case 0x05: case 0x06: case 0x0B: case 0x0E: case 0x20: case 0x21: case 0x22: case 0x23: case 0x24:
                case 0x25: case 0x27: case 0x2A: case 0x2B: case 0x2E: case 0x33: case 0x35: case 0x36:
                    if      (in.omod == 1) d = b.fbin(Op_FMul, d, b.uconst(fbits(2.0f)));
                    else if (in.omod == 2) d = b.fbin(Op_FMul, d, b.uconst(fbits(4.0f)));
                    else if (in.omod == 3) d = b.fbin(Op_FMul, d, b.uconst(fbits(0.5f)));
                    if (in.clamp) d = b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, d, b.uconst(fbits(1.0f))), b.uconst(fbits(0.0f)));
                    break;
                default: ok = false; break;
            }
            if (ok) predicate_write(b, rs, in.dst.value, old_d);
            return true;
        }
        case Rdna2Format::VOP2: {
            uint32_t a = val(in.src[0]), c = val(in.src[1]); uint32_t old_d = vreg_old(b, rs, in.dst.value);
            // DPP16 quad_perm on src0 (#273): fragment-only, FLOAT ops only (add/sub/subrev/mul/min/
            // max/mac — the manual-derivative idiom's carriers); anything else stays rejected.
            if (in.has_dpp) {
                const bool fop = in.opcode == 0x03 || in.opcode == 0x04 || in.opcode == 0x05 ||
                                 in.opcode == 0x08 || in.opcode == 0x0F || in.opcode == 0x10 ||
                                 in.opcode == 0x1F || in.opcode == 0x2B;
                if (!b.is_fragment || !fop) { ok = false; return true; }
                a = b.dpp_quad(a, in.dpp_ctrl);
            }
            // SDWA float source modifiers (only ever set on float ops by the assembler): abs then neg.
            if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
            if (in.src_neg[0]) a = b.fbin(Op_FSub, b.uconst(0), a);
            if (in.src_abs[1]) c = b.fext1(Glsl_FAbs, c);
            if (in.src_neg[1]) c = b.fbin(Op_FSub, b.uconst(0), c);
            uint32_t& d = vreg[in.dst.value];
            switch (in.opcode) {
                case 0x01: d = b.sel(vcc, c, a); break;               // v_cndmask_b32: dst = vcc ? src1 : src0
                case 0x03: d = b.fbin(Op_FAdd, a, c); break;          // v_add_f32
                case 0x04: d = b.fbin(Op_FSub, a, c); break;          // v_sub_f32
                case 0x05: d = b.fbin(Op_FSub, c, a); break;          // v_subrev_f32 (src1 - src0; e32 form of
                                                                      // VOP3 0x105 — round-trip llvm-mc gfx1010 0x0a020702)
                case 0x08: d = b.fbin(Op_FMul, a, c); break;          // v_mul_f32
                case 0x0F: d = b.fext2(Glsl_FMin, a, c); break;       // v_min_f32
                case 0x10: d = b.fext2(Glsl_FMax, a, c); break;       // v_max_f32
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
                case 0x25: d = b.ibin(Op_IAdd, a, c); break;          // v_add_nc_u32
                case 0x26: d = b.ibin(Op_ISub, a, c); break;          // v_sub_nc_u32
                case 0x27: d = b.ibin(Op_ISub, c, a); break;          // v_subrev_nc_u32 (reverse: src1 - src0)
                // Carry ops (VOP2 e32 form): carry-in + carry-out are VCC. v_add_co_ci(0x28)/
                // v_sub_co_ci(0x29)/v_subrev_co_ci(0x2a). Mirrors the VOP3B 0x128/129/12A logic with VCC.
                case 0x28: case 0x29: case 0x2A: {
                    uint32_t cin = b.sel(vcc, b.uconst(1), b.uconst(0));
                    if (in.opcode == 0x28) {                          // (a + c) + cin
                        uint32_t s1 = b.ibin(Op_IAdd, a, c); uint32_t k1 = b.ucmp(Op_ULessThan, s1, a);
                        d = b.ibin(Op_IAdd, s1, cin); uint32_t k2 = b.ucmp(Op_ULessThan, d, s1);
                        vcc = b.bsel(k1, b.btrue(), k2);
                    } else {                                          // (x - y) - cin  (subrev swaps)
                        uint32_t x = in.opcode == 0x29 ? a : c, y = in.opcode == 0x29 ? c : a;
                        uint32_t s1 = b.ibin(Op_ISub, x, y); uint32_t k1 = b.ucmp(Op_ULessThan, x, y);
                        d = b.ibin(Op_ISub, s1, cin); uint32_t k2 = b.ucmp(Op_ULessThan, s1, cin);
                        vcc = b.bsel(k1, b.btrue(), k2);
                    }
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
                case 0x35: {   // v_mul_f16: 16-bit float multiply. Sources read their selected halves
                    // (SDWA WORD_1 = high 16; DWORD/WORD_0 = low 16 — an f16 op reads bits[15:0]);
                    // f16xf16 products are exact in f32, so multiply in f32 and round once to f16.
                    // The 16-bit result inserts into the selected dest half PRESERVING the other
                    // (dst_sel WORD_1 for the SDWA pack idiom; DWORD/WORD_0 = the plain e32 form's
                    // "write [15:0], preserve [31:16]" gfx10 f16-VOP2 contract). #273 (DOLL box-blur).
                    auto half_of = [&](uint32_t x, uint8_t s) {
                        return s == 5 ? b.ibin(Op_ShiftRightLogical, x, b.uconst(16)) : x;
                    };
                    uint32_t p = b.fbin(Op_FMul, b.unpack_half(half_of(a, in.sdwa_src0_sel), 0),
                                                 b.unpack_half(half_of(c, in.sdwa_src1_sel), 0));
                    uint32_t r16 = b.pack_half_lo(p);
                    d = (in.sdwa_dst_sel == 5)
                        ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                                 b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                        : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
                    break;
                }
                default: ok = false;
            }
            // SDWA output modifier: OMOD scale (×2/×4/×0.5) then CLAMP saturate, on FLOAT-result opcodes
            // only (int ops never carry omod). Mirrors the VOP3 fresult path; a no-op when omod/clamp unset.
            if (ok && (in.omod || in.clamp)) switch (in.opcode) {
                case 0x03: case 0x04: case 0x05: case 0x08: case 0x0F: case 0x10:
                case 0x1F: case 0x2B: case 0x20: case 0x2C: case 0x21: case 0x2D:
                    if      (in.omod == 1) d = b.fbin(Op_FMul, d, b.uconst(fbits(2.0f)));
                    else if (in.omod == 2) d = b.fbin(Op_FMul, d, b.uconst(fbits(4.0f)));
                    else if (in.omod == 3) d = b.fbin(Op_FMul, d, b.uconst(fbits(0.5f)));
                    if (in.clamp) d = b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, d, b.uconst(fbits(1.0f))), b.uconst(fbits(0.0f)));
                    break;
                // A non-float-result opcode carrying a modifier (e.g. an INTEGER SDWA op with CLAMP =
                // integer saturation) is not modeled by the float-domain omod/clamp above, so applying
                // nothing would SILENTLY drop the saturation and emit a valid-but-wrong shader. Reject
                // loudly instead — the same fail-visibly-over-miscompile discipline as the forward-if
                // clamp (#129/#174). The guard means this only fires for a modifier-carrying op.
                default: ok = false; break;
            }
            if (ok) predicate_write(b, rs, in.dst.value, old_d);
            return true;
        }
        case Rdna2Format::VOPC: {                                     // v_cmp_* -> VCC; v_cmpx_* also -> EXEC
            const uint32_t ra = val(in.src[0]), rc = val(in.src[1]);  // raw bits (f16 compares re-derive)
            uint32_t a = ra, c = rc;
            // Float source modifiers (abs then neg — hardware order), set only on FLOAT compares by the
            // assembler (VOP3-encoded e64 or SDWA forms; e.g. DOLL's `v_cmp_gt_f32_sdwa vcc, |v5|, s4`).
            if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
            if (in.src_neg[0]) a = b.fbin(Op_FSub, b.uconst(0), a);
            if (in.src_abs[1]) c = b.fext1(Glsl_FAbs, c);
            if (in.src_neg[1]) c = b.fbin(Op_FSub, b.uconst(0), c);
            // v_cmpx_* shares each type's compare set at base+0x10 (f32 0x10-0x1f, i32 0x90-0x9f,
            // u32 0xd0-0xdf); it writes EXEC in addition to VCC. Map to the base compare, then narrow.
            uint32_t op = in.opcode;
            bool is_cmpx = vopc_is_cmpx(op);
            if (is_cmpx && !allow_exec_update) { ok = false; return true; }
            uint32_t eff = is_cmpx ? op - 0x10 : op;
            uint32_t cmp = 0;
            switch (eff) {
                case 0x01: cmp = b.fcmp(Op_FOrdLessThan, a, c); break;         // v_cmp_lt_f32
                case 0x02: cmp = b.fcmp(Op_FOrdEqual, a, c); break;            // v_cmp_eq_f32
                case 0x03: cmp = b.fcmp(Op_FOrdLessThanEqual, a, c); break;    // v_cmp_le_f32
                case 0x04: cmp = b.fcmp(Op_FOrdGreaterThan, a, c); break;      // v_cmp_gt_f32
                case 0x05: cmp = b.fcmp(Op_FOrdNotEqual, a, c); break;         // v_cmp_lg_f32
                case 0x06: cmp = b.fcmp(Op_FOrdGreaterThanEqual, a, c); break; // v_cmp_ge_f32
                // NaN-inclusive f32 compares (the "n"-prefix set is the unordered negation of 0x1-0x6):
                case 0x09: cmp = b.fcmp(Op_FUnordLessThan, a, c); break;       // v_cmp_nge_f32 = !(a>=b)
                case 0x0A: cmp = b.fcmp(Op_FUnordEqual, a, c); break;          // v_cmp_nlg_f32 = !(a!=b)
                case 0x0B: cmp = b.fcmp(Op_FUnordLessThanEqual, a, c); break;  // v_cmp_ngt_f32 = !(a>b)
                case 0x0C: cmp = b.fcmp(Op_FUnordGreaterThan, a, c); break;    // v_cmp_nle_f32 = !(a<=b)
                case 0x0D: cmp = b.fcmp(Op_FUnordNotEqual, a, c); break;       // v_cmp_neq_f32 = !(a==b)
                case 0x0E: cmp = b.fcmp(Op_FUnordGreaterThanEqual, a, c); break;// v_cmp_nlt_f32 = !(a<b)
                case 0x81: cmp = b.scmp(Op_SLessThan, a, c); break;            // v_cmp_lt_i32
                case 0x82: cmp = b.ucmp(Op_IEqual, a, c); break;               // v_cmp_eq_i32
                case 0x83: cmp = b.scmp(Op_SLessThanEqual, a, c); break;       // v_cmp_le_i32
                case 0x84: cmp = b.scmp(Op_SGreaterThan, a, c); break;         // v_cmp_gt_i32
                case 0x85: cmp = b.ucmp(Op_INotEqual, a, c); break;            // v_cmp_ne_i32 (sign-agnostic)
                case 0x86: cmp = b.scmp(Op_SGreaterThanEqual, a, c); break;    // v_cmp_ge_i32
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
                        uint32_t v = (in.src[k].kind == OperandKind::InlineFloat ||
                                      in.src[k].kind == OperandKind::InlineInt)
                                         ? raw                       // inline: already a full-width value
                                         : b.unpack_half(raw, 0);    // register/literal: f16 in the low half
                        if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                        if (in.src_neg[k]) v = b.fbin(Op_FSub, b.uconst(0), v);
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
                } else if (in.dst.kind == OperandKind::SGPR && in.dst.value <= 105) {
                    rs.sreg_bool[in.dst.value] = cmp; rs.sreg_bool_narrowed[in.dst.value] = true;
                } else {
                    vcc = cmp; rs.sreg_bool_narrowed[106] = true; rs.sreg_bool_narrowed[107] = true;
                }
            }
            return true;
        }
        case Rdna2Format::VOP3: {
            // SCALAR-SPILL lane slots (#273): v_writelane_b32 (0x361) / v_readlane_b32 (0x360) with a
            // COMPILE-TIME lane index — the pack-scalars-into-a-VGPR's-lanes idiom (DOLL's big post PS
            // spills 19 s_buffer_load results into v36 and reads them back). Per-invocation each
            // (vgpr, lane) is a named wave-uniform scalar; a dynamic lane index (a real cross-lane
            // read) still rejects. Neither op is EXEC-predicated on hardware, so no predicate_write.
            // VERIFIED(round-trip llvm-mc gfx1030: 0xd761 v_writelane_b32 / 0xd760 v_readlane_b32).
            if (in.opcode == 0x361) {                                 // v_writelane_b32 vDST, sSRC, lane
                if (in.src[1].kind != OperandKind::InlineInt || in.src[1].value < 0 || in.src[1].value > 63) {
                    ok = false; return true;
                }
                rs.vgpr_lane_slots[in.dst.value][in.src[1].value] = val(in.src[0]);
                return true;
            }
            if (in.opcode == 0x360) {                                 // v_readlane_b32 sDST, vSRC, lane
                if (in.src[1].kind != OperandKind::InlineInt || in.src[1].value < 0 || in.src[1].value > 63) {
                    ok = false; return true;
                }
                auto vit = rs.vgpr_lane_slots.find(in.src[0].value);
                if (vit == rs.vgpr_lane_slots.end()) { ok = false; return true; }   // not a spill array
                auto sit = vit->second.find(in.src[1].value);
                if (sit == vit->second.end()) { ok = false; return true; }          // slot never written
                rs.sreg[in.dst.value] = sit->second;   // dst field is the SGPR number (like readfirstlane)
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            uint32_t old_d = vreg_old(b, rs, in.dst.value);
            // Float source with its VOP3 modifiers applied: OpFAbs then OpFNegate (hardware order neg(abs(x))).
            // Returns raw bits (fbin/fext re-bitcast), so it's a drop-in for val() in the FLOAT ops only —
            // integer VOP3 ops keep val() (modifiers are float-domain; assemblers don't set them on int ops).
            auto fv = [&](int k) -> uint32_t {
                uint32_t bits = val(in.src[k]);
                if (in.src_abs[k]) bits = b.fext1(Glsl_FAbs, bits);
                if (in.src_neg[k]) bits = b.fbin(Op_FSub, b.uconst(0), bits);   // 0.0 - x = -x
                return bits;
            };
            // Output modifiers on a FLOAT result: OMOD scale (×2/×4/×0.5) then CLAMP saturate to [0,1]
            // (hardware order: clamp(omod(x))). Wrap each float op's result through this.
            auto fresult = [&](uint32_t bits) -> uint32_t {
                if (in.omod == 1)      bits = b.fbin(Op_FMul, bits, b.uconst(fbits(2.0f)));
                else if (in.omod == 2) bits = b.fbin(Op_FMul, bits, b.uconst(fbits(4.0f)));
                else if (in.omod == 3) bits = b.fbin(Op_FMul, bits, b.uconst(fbits(0.5f)));
                if (in.clamp) bits = b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, bits, b.uconst(fbits(1.0f))), b.uconst(fbits(0.0f)));
                return bits;
            };
            if (in.opcode == 0x14B || in.opcode == 0x141) {           // v_fma_f32 / v_mad_f32 = src0*src1 + src2
                // v_mad_f32 (op 0x141) is a gfx10.1 (Navi) instruction REMOVED in gfx10.3, so llvm-mc
                // -mcpu=gfx1030 rejects it as invalid — but the PS5 shader compiler targets gfx10.1 and
                // emits it (real game shaders 5,26-29: manual attribute interpolation p0+i*p1). Its result
                // (unfused mul-then-add) maps exactly to OpFMul+OpFAdd; v_fma's fused rounding is
                // immaterial here. VERIFIED(round-trip llvm-mc gfx1010, both directions): VOP3 op 0x141.
                uint32_t m = b.fbin(Op_FMul, fv(0), fv(1));
                vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, m, fv(2)));
            } else if (in.opcode == 0x169) {                          // v_mul_lo_u32
                vreg[in.dst.value] = b.ibin(Op_IMul, val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x16a) {                          // v_mul_hi_u32 (high 32 bits)
                vreg[in.dst.value] = b.umul_hi(val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x16c) {                          // v_mul_hi_i32 (high 32 bits, signed)
                vreg[in.dst.value] = b.smul_hi(val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x157) {                          // v_med3_f32 = median(s0,s1,s2)
                uint32_t s0 = fv(0), s1 = fv(1), s2 = fv(2);
                uint32_t mn = b.fext2(Glsl_FMin, s0, s1), mx = b.fext2(Glsl_FMax, s0, s1);
                vreg[in.dst.value] = fresult(b.fext2(Glsl_FMax, mn, b.fext2(Glsl_FMin, mx, s2)));
            } else if (in.opcode == 0x151 || in.opcode == 0x154) {    // v_min3_f32 / v_max3_f32
                // min/max of three floats (DOLL's AA-clamp PS). VERIFIED(round-trip llvm-mc gfx1010:
                // VOP3 0x151 = v_min3_f32, 0x154 = v_max3_f32 — 0xd551…/0xd554…). CONFIDENCE: HIGH.
                uint32_t op = in.opcode == 0x151 ? (uint32_t)Glsl_FMin : (uint32_t)Glsl_FMax;
                vreg[in.dst.value] = fresult(b.fext2(op, b.fext2(op, fv(0), fv(1)), fv(2)));
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
                if (s2.value == 106 || s2.value == 107) cin_mask = rs.vcc;
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
                    // carry-out -> sdst mask (VCC or a saved SGPR-pair bool).
                    if (in.sdst.value == 106 || in.sdst.value == 107) rs.vcc = cout;
                    else if (in.sdst.kind == OperandKind::SGPR) rs.sreg_bool[in.sdst.value] = cout;
                }
            } else if ((in.opcode == 0x365 || in.opcode == 0x366) && allow_wave && b.is_compute) {
                // v_mbcnt_lo/hi_u32_b32 (cross-lane): dst = src1 + count of lanes below this one whose mask
                // bit (src0) is set, in the low/high 32. The per-lane "mask bit" comes from src0: EXEC
                // (126/127) -> this lane's exec bool; inline -1 (all-ones) -> always set (mbcnt = lane
                // index, the common "get my lane id" idiom, e.g. shader 037); inline 0 -> never; an SGPR
                // pair -> that saved mask's bool. A general computed 32-bit mask VALUE isn't representable
                // per-lane, so reject that. LDS+barriers (allow_wave => straight-line, barrier-uniform).
                const Operand& s0 = in.src[0]; uint32_t active = 0;
                if (s0.value == 126 || s0.value == 127) active = rs.exec;
                else if (s0.kind == OperandKind::InlineInt && s0.value == -1) active = b.btrue();
                else if (s0.kind == OperandKind::InlineInt && s0.value == 0)  active = b.bfalse();
                else if (s0.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(s0.value); if (it != rs.sreg_bool.end()) active = it->second; }
                if (active) vreg[in.dst.value] = b.mbcnt(active, val(in.src[1]), in.opcode == 0x365);
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
            } else if (in.opcode == 0x10F) {                          // v_min_f32 (VOP3 form)
                vreg[in.dst.value] = fresult(b.fext2(Glsl_FMin, fv(0), fv(1)));
            } else if (in.opcode == 0x110) {                          // v_max_f32 (VOP3 form)
                vreg[in.dst.value] = fresult(b.fext2(Glsl_FMax, fv(0), fv(1)));
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
                else if (s2.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(s2.value); if (it != rs.sreg_bool.end()) m = it->second; }
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
                case 0x20:   // s_inst_prefetch  (I-cache hint)
                case 0x21:   // s_clause         (memory-clause scheduling hint)
                case 0x22:   // s_wait_idle
                    break;   // (s_waitcnt_vscnt is SOPK on gfx10, not SOPP 0x7d — see the SOPK case)
                case 0x08:                                          // s_cbranch_execz
                    if (in.simm16 < 0) ok = false;                 // backward = loop -> unsupported
                    else if (rs.exec_narrowed && (!safe_execz || !safe_execz->count(in.pc))) ok = false;
                    break;                                          // forward = no-op (predication covers it)
                case 0x0a:                                          // s_barrier
                    if (b.is_compute) b.barrier();                  // workgroup exec+memory barrier (LDS sync)
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
            if (!allow_smem) { ok = false; return true; }
            uint32_t n = 0;
            switch (in.opcode) {
                case 0x0: case 0x8: n = 1;  break;   // s_load_dword     / s_buffer_load_dword
                case 0x1: case 0x9: n = 2;  break;   // s_load_dwordx2   / s_buffer_load_dwordx2
                case 0x2: case 0xA: n = 4;  break;   // s_load_dwordx4   / s_buffer_load_dwordx4
                case 0x3: case 0xB: n = 8;  break;   // s_load_dwordx8   / s_buffer_load_dwordx8
                case 0x4: case 0xC: n = 16; break;   // s_load_dwordx16  / s_buffer_load_dwordx16
                default: ok = false; return true;    // stores / others not yet
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
                if (!tracked) { ok = false; return true; }
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
                auto it = rs.sreg_srt.find(in.src[0].value);
                if (it != rs.sreg_srt.end()) res = rt->by_srt_offset(it->second);
                // A scalar buffer load reads a CONSTANT buffer — resolve the SBASE SGPR to a constant
                // buffer specifically (the same SGPR may also hold a vertex-buffer V# elsewhere).
                if (!res) res = rt->by_sgpr_base_cls(in.src[0].value, ResourceClass::ConstantBuffer);
                if (res) { binding = res->binding; cbuf_resolved = true; } }
            if (getenv("PROSPER_CBUFLOG"))
                fprintf(stderr, "[cbuf] s_buffer_load x%u src0=s%d off=0x%x(dw%u) dyn=%d -> binding=%u %s\n",
                        n, in.src[0].value, in.literal, base_idx, (int)soff_dyn, binding,
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
            if (rt && (in.opcode == 0x2 || in.opcode == 0x3))
                for (uint32_t k = 0; k < n; k++) rs.sreg_srt[in.dst.value + (int)k] = in.literal;
            return true;
        }
        case Rdna2Format::MUBUF: {
            // Untyped buffer LOAD — the per-lane fetch mechanism (vertex fetch et al.). Modeled as a
            // per-lane load from the bound constant buffer: byte addr = (offen ? VADDR : 0) + SOFFSET
            // + inst-offset; index = addr>>2; N dwords -> VDATA..+N-1. Descriptor (SRSRC), idxen*stride,
            // and the format-converting buffer_load_format_* variants are deferred. Compute-only (cbuf).
            if (!allow_smem) { ok = false; return true; }
            uint32_t n = 0; bool is_format = false, is_store = false;
            switch (in.opcode) {
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
                case 0x4: n = 1; is_format = true; is_store = true; break;   // buffer_store_format_x
                case 0x5: n = 2; is_format = true; is_store = true; break;   // buffer_store_format_xy
                case 0x6: n = 3; is_format = true; is_store = true; break;   // buffer_store_format_xyz
                case 0x7: n = 4; is_format = true; is_store = true; break;   // buffer_store_format_xyzw
                default: ok = false; return true;           // typed / atomics not yet
            }
            uint32_t offset = in.literal & 0xFFFu;
            bool offen = (in.literal >> 12) & 1u, idxen = (in.literal >> 13) & 1u;
            // PC-relative EMBEDDED TABLE (#273): this load's V# was built from s_getpc_b64 and the
            // table bytes live inside the shader blob — detect_pcrel_tables already copied them out.
            // Fold to a compile-time constant lookup: dword index = (inst offset + offen VADDR) >> 2;
            // out-of-range indexes read 0 (the hardware's OOB contract for a bounded V#).
            if (!is_format && !is_store) {
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
            uint32_t binding = 2, stride = 0;   // overwritten by SRSRC resolution below whenever a resource
                                                // table is present (format AND raw ops); the binding-2 default
                                                // survives only on the table-less offline path (see below)
            // Format of the fetched components. Untyped buffer_load_dword* is raw 32-bit (comp_bytes=4);
            // buffer_load_format_* takes the format from the resolved V# descriptor.
            DataFormat fmt = DataFormat::Uint32;   // untyped default: raw dwords
            uint32_t fmt_ncomp = 0;    // the V#'s real component count (format loads only); 0 = don't default-fill
            bool dyn_vfetch = false;   // set when the V# came from by_fetch_pc — a const-folded per-vertex
                                       // attribute fetch, whose element address is exactly gl_VertexIndex*stride.
            if (is_format) {
                // A format load reads a vertex/buffer attribute — it needs the V# descriptor for the
                // binding, stride, and data format. Resolve SRSRC (src[1]) via provenance: an s_load
                // tag (indirect) else the SGPR index (direct/user-data).
                const ShaderResource* res = nullptr;
                // A format load (vertex fetch) reads a VERTEX buffer — resolve the SRSRC SGPR to a vertex
                // buffer specifically (that SGPR may hold a constant-buffer V# at other points; the const-
                // fold-resolved vertex buffer is keyed by this SRSRC SGPR). Fall back to an s_load SRT tag.
                if (rt) {
                    // PER-FETCH first: a reloaded SRSRC holds a different V# per attribute, so match this
                    // exact fetch instruction's pc; fall back to the SGPR (direct) then s_load SRT tag.
                    // Only a VERTEX-buffer pc entry implies the vertex-index address model — a pc-keyed
                    // CONSTANT/structured buffer (a PS's per-lane table fetch, #273) keeps the faithful
                    // VADDR*stride+offset address below.
                    res = rt->by_fetch_pc(in.pc);
                    if (res) dyn_vfetch = (res->cls == ResourceClass::VertexBuffer);
                    if (!res) res = rt->by_sgpr_base_cls(in.src[1].value, ResourceClass::VertexBuffer);
                    if (!res) { auto it = rs.sreg_srt.find(in.src[1].value);
                        if (it != rs.sreg_srt.end()) res = rt->by_srt_offset(it->second); }
                    // DIRECT user-data V# of any class (#273 — DOLL's title post PSes format-fetch
                    // through a V# the metadata labels a CONSTANT buffer sharp at s[24:27]): the class
                    // label doesn't change the descriptor's fields. Only when the SGPR was never
                    // REWRITTEN in-shader (no rs.sreg entry) — a reloaded register no longer holds the
                    // seed-time sharp, and trusting it would fetch through a stale descriptor.
                    if (!res && rs.sreg.find(in.src[1].value) == rs.sreg.end())
                        res = rt->by_sgpr_base(in.src[1].value);
                }
                if (!res) {
                    if (getenv("PROSPER_DBG")) {   // which provenance step failed for this format load
                        auto it = rs.sreg_srt.find(in.src[1].value);
                        fprintf(stderr, "[mubuf-unresolved] pc=%u srsrc=s%d srt_tag=%s0x%x key_res=%s (%zu res)\n",
                                in.pc, in.src[1].value, it != rs.sreg_srt.end() ? "" : "NONE ",
                                it != rs.sreg_srt.end() ? it->second : 0u,
                                it != rs.sreg_srt.end() && rt->by_srt_offset(it->second) ? "yes" : "null",
                                rt->resources.size());
                    }
                    ok = false; return true;
                }
                binding = res->binding;
                stride = res->stride;
                fmt = res->format;
                fmt_ncomp = res->num_components;   // for the format default-fill below (#368)
            } else if (rt) {
                // RAW (untyped) MUBUF with a resource table: resolve SRSRC (src[1]) exactly like the
                // format path — a raw buffer op targets whatever buffer its V# describes, NOT a fixed
                // binding (#91: the old hardcoded binding-2 silently read/wrote the wrong buffer for
                // any other target). Raw ops don't imply a resource class the way a format load implies
                // VertexBuffer, so the direct-SGPR lookup is class-unrestricted (by_sgpr_base).
                // Provenance order mirrors the format path: exact fetch pc, then s_load SRT tag
                // (indirect), then user-data SGPR (direct).
                const ShaderResource* res = rt->by_fetch_pc(in.pc);
                if (!res) { auto it = rs.sreg_srt.find(in.src[1].value);
                    if (it != rs.sreg_srt.end()) res = rt->by_srt_offset(it->second); }
                if (!res) res = rt->by_sgpr_base(in.src[1].value);
                if (!res) { ok = false; return true; }   // unresolvable V# -> reject; NEVER default to binding 2
                binding = res->binding;
                stride  = res->stride;
                // fmt stays raw Uint32: untyped ops move raw dwords regardless of the V#'s declared format.
            }
            // else (rt == nullptr): table-less offline compute shell (recompile_valu without a resource
            // table — the unit-test harness). Keep the legacy single-cbuf convention: binding 2, stride 0.
            // The live graphics path can never reach here table-less: recompile_vertex/recompile_fragment
            // set allow_smem = (rt != nullptr), so MUBUF already rejected above when rt is null there.
            const uint32_t comp_bytes = data_format_bytes(fmt);
            if (comp_bytes == 0) { ok = false; return true; }   // unknown / unsupported format
            // Per-component decode. 4-byte formats (Float32/Uint32/Sint32) are a raw dword load — no
            // conversion in our bit model. Sub-dword formats are unpacked: UNORM/SNORM normalize an
            // integer field, Float16 unpacks a packed half. num_components components pack tightly.
            const bool packed = comp_bytes < 4;
            bool is_snorm = (fmt == DataFormat::Snorm8 || fmt == DataFormat::Snorm16);
            bool is_half  = (fmt == DataFormat::Float16);
            // Integer sub-dword formats deliver the raw (un-normalized) INTEGER in the VGPR — the
            // hardware's UINT/SINT format-load contract. DOLL's skinned scene VS fetches its bone
            // indices as Uint8 x4 (stride 8, paired with Unorm8 weights); rejecting them dropped
            // every scene-geometry draw (#273). Zero-/sign-extend the field; no normalization.
            bool is_uint = (fmt == DataFormat::Uint8 || fmt == DataFormat::Uint16);
            bool is_sint = (fmt == DataFormat::Sint8 || fmt == DataFormat::Sint16);
            float norm = 0.0f;
            switch (fmt) {
                case DataFormat::Unorm8:  norm = 255.0f;   break;
                case DataFormat::Snorm8:  norm = 127.0f;   break;
                case DataFormat::Unorm16: norm = 65535.0f; break;
                case DataFormat::Snorm16: norm = 32767.0f; break;
                default: break;
            }
            if (packed && !is_half && !is_uint && !is_sint && norm == 0.0f) {
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr, "[mubuf-badfmt] pc=%u fmt=%u comp_bytes=%u stride=%u n=%u\n",
                            in.pc, (unsigned)fmt, comp_bytes, stride, n);
                ok = false; return true;
            }
            // Packed (sub-dword) components are extracted at STATIC byte offsets relative to a
            // DWORD-ALIGNED element base — the low 2 bits of the address (addr&3) are dropped by the
            // addr>>2 dword index and never folded into the bit extraction (#150). That is only correct
            // when the element base is provably 4-byte aligned; otherwise a component (a half2 UV after
            // a snorm16x3 normal, an unorm8 at a byte offset) decodes the wrong bits. A fully general
            // fix needs a runtime bit position that can straddle dwords; until then, prove alignment
            // from the address terms and REJECT the packed load/store when it can't be proven (surfacing
            // the gap) rather than silently mis-decode. Aligned iff: inst offset %4==0; stride %4==0
            // when idxen; no offen (a runtime per-lane byte offset is unprovable); SOFFSET is NULL/0.
            bool dyn_half = false;   // 16-bit element at a runtime dword half (stride-2 buffers, #273)
            if (packed) {
                bool base_aligned;
                if (dyn_vfetch) {
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
                    // HALFWORD-ALIGNED single-component 16-bit load (#273 — DOLL's title post PSes
                    // fetch from a STRIDE-2 uint16 table: `buffer_load_format_x v, vIDX, V#` with
                    // stride 2). The element sits at a RUNTIME dword half — bit offset (addr&2)*8 —
                    // which never straddles a dword, so extract it dynamically instead of rejecting.
                    // Provable iff every address term is 2-aligned and there is no per-lane byte
                    // offset (offen). Loads only (the packed store path still rejects sub-dword ints).
                    bool soff_zero = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                                     (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                    dyn_half = !is_store && n == 1 && comp_bytes == 2 && !offen && !dyn_vfetch &&
                               (offset & 1u) == 0 && (!idxen || (stride & 1u) == 0) && soff_zero;
                    if (!dyn_half) {
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
            if (dyn_vfetch && idxen && stride) {
                addr = b.ibin(Op_IMul, b.load_vertex_index(), b.uconst(stride));
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
            if (is_store) {
                // Store the VDATA VGPRs (in.dst..+n-1). Integer sub-dword formats (Uint8/Sint8/...) have
                // no packing path here -> reject rather than mis-store (loads extend them; stores don't pack).
                if (packed && (is_uint || is_sint)) { ok = false; return true; }
                auto vread = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
                if (!packed) {
                    // Raw/Float32/Uint32: one dword per component.
                    for (uint32_t k = 0; k < n; k++) {
                        uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                        b.cbuf_store(kidx, vread(in.dst.value + (int)k), binding, rs.exec_narrowed, rs.exec);
                    }
                } else {
                    // Packed UNORM/SNORM/Float16: pack the components tightly into ceil(n*bytes/4) dwords
                    // (inverse of the packed load). Each dword ORs together the fields that land in it.
                    const uint32_t dwords = (n * comp_bytes + 3) / 4;
                    for (uint32_t d = 0; d < dwords; d++) {
                        uint32_t acc = b.uconst(0);
                        for (uint32_t k = 0; k < n; k++) {
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
                                     fmt == DataFormat::Sint8  || fmt == DataFormat::Sint16 || fmt == DataFormat::Sint32);
            for (uint32_t k = 0; k < n; k++) {
                int d = in.dst.value + (int)k;
                uint32_t old = vreg_old(b, rs, d);
                uint32_t value;
                // Format default-fill (#368): BUFFER_LOAD_FORMAT_* returns only the components the V#'s
                // format defines; a component the OPCODE requests beyond that (e.g. _xyzw against a
                // 32_32_32 position, or _xyz against a 32_32 UV) is filled with the standard vector
                // default — 0 for G/B/Z, 1 for A/W — NOT read from adjacent memory (which yielded the
                // next vertex's bytes, or robust-0 at the buffer tail, so a vec4 read of a vec3 attribute
                // got W = garbage/0 instead of 1.0 -> broken transforms). Only for format loads with a
                // known component count; untyped raw MUBUF loads keep every opcode component.
                if (is_format && fmt_ncomp && k >= fmt_ncomp) {
                    uint32_t one = fmt_is_int ? 1u : 0x3f800000u;   // integer 1 vs float 1.0 (raw bits)
                    value = b.uconst(k == 3 ? one : 0u);            // W/A -> 1, G/B/Z -> 0
                } else if (!packed) {
                    uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                    value = b.cbuf_load(kidx, binding);                  // raw 32-bit component
                } else if (dyn_half) {
                    // Runtime dword half (n==1, 16-bit element, 2-aligned address): shift the loaded
                    // dword right by (addr&2)*8 and decode the 16-bit field at bit 0.
                    uint32_t dw   = b.cbuf_load(idx, binding);
                    uint32_t boff = b.ibin(Op_ShiftLeftLogical, b.ibin(Op_BitwiseAnd, addr, b.uconst(2)), b.uconst(3));
                    uint32_t dws  = b.ibin(Op_ShiftRightLogical, dw, boff);
                    value = is_half ? b.unpack_half(dws, 0)
                          : is_uint ? b.bfe_u(dws, b.uconst(0), b.uconst(16))
                          : is_sint ? b.bfe_s(dws, b.uconst(0), b.uconst(16))
                                    : b.unpack_norm(dws, 0, 16, is_snorm, norm);
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
            const ShaderResource* res = nullptr;
            { auto it = rs.sreg_srt.find(in.src[1].value);
              if (it != rs.sreg_srt.end()) res = rt->by_srt_offset(it->second);
              if (!res) res = rt->by_sgpr_base(in.src[1].value); }
            // PER-USE pc provenance fallback (#273 — DOLL's title-composite image_sample_b): the
            // executor's const-fold walk snapshots the T# each image op consumes and keys it by the
            // INSTRUCTION pc (ShaderResource::fetch_pc). Used when the key/SGPR chains found nothing,
            // or found a NON-image resource — the load-immediate key model collides when two different
            // tables reuse one immediate (a key-0 EUD sharp vs the key-0 table T# here).
            if (!res || (res->cls != ResourceClass::Texture && res->cls != ResourceClass::StorageImage)) {
                if (const ShaderResource* pr = rt->by_fetch_pc(in.pc);
                    pr && (pr->cls == ResourceClass::Texture || pr->cls == ResourceClass::StorageImage))
                    res = pr;
            }
            if ((!res || (res->cls != ResourceClass::Texture && res->cls != ResourceClass::StorageImage))
                && getenv("PROSPER_DBG")) {
                // Resolution-failure diagnostic: which provenance step failed for this image op.
                auto it = rs.sreg_srt.find(in.src[1].value);
                const ShaderResource* pk = it != rs.sreg_srt.end() ? rt->by_srt_offset(it->second) : nullptr;
                const ShaderResource* pp = rt->by_fetch_pc(in.pc);
                fprintf(stderr, "[mimg-unresolved] pc=%u srsrc=s%d srt_tag=%s0x%x key_res=%s pc_res=%s (%zu res)\n",
                        in.pc, in.src[1].value, it != rs.sreg_srt.end() ? "" : "NONE ",
                        it != rs.sreg_srt.end() ? it->second : 0u,
                        pk ? (pk->cls == ResourceClass::Texture ? "tex" : "other-cls") : "null",
                        pp ? (pp->cls == ResourceClass::Texture ? "tex" : "other-cls") : "null",
                        rt->resources.size());
            }
            if (!res) { ok = false; return true; }

            // --- Storage-image path: image_load (0x00) / image_store (0x08), no sampler, any dim ---
            if (res->cls == ResourceClass::StorageImage) {
                const bool is_ld = (in.opcode == 0x00), is_st = (in.opcode == 0x08);
                if (!is_ld && !is_st) { ok = false; return true; }
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
                if (ms && is_st) { ok = false; return true; }   // per-sample MSAA store not modeled (resolve shaders read)
                b.declare_storage_image(res->binding, dim, arrayed, ms);
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
                } else {
                    // image_store: gather the VDATA VGPRs selected by dmask into an RGBA texel (channels
                    // absent from dmask store as 0). Under a narrowed EXEC (e.g. a grid-tail bounds check),
                    // the write is EXEC-predicated so inactive lanes don't write out-of-range texels.
                    uint32_t vals[4] = { b.uconst(0), b.uconst(0), b.uconst(0), b.uconst(0) };
                    int vd = in.dst.value, w = 0;
                    for (uint32_t c = 0; c < 4; c++) if (in.mimg_dmask & (1u << c)) { vals[c] = vread(vd + w); w++; }
                    b.image_write(res->binding, dim, arrayed, ncoord, coords, vals, rs.exec_narrowed, rs.exec);
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
            const bool is_sample = (in.opcode == 0x20), is_load = (in.opcode == 0x00);
            const bool is_sample_l = (in.opcode == 0x24), is_sample_lz = (in.opcode == 0x27);
            const bool is_sample_b = (in.opcode == 0x25), is_gather_lz = (in.opcode == 0x47);
            // image_gather4_lz_o = 0x57 (gather at base level with the _o packed-offset operand in the
            // FIRST vaddr — llvm-mc gfx1030 round-trip on live DOLL bytes: 0xf15c0808 "image_gather4_lz_o
            // v[4:7], [v0, v18, v19], ..." — coords follow the offset, matching image_sample_b's
            // modifier-first vaddr convention). DOLL's FXAA/upsample pass PS (#294).
            const bool is_gather_lz_o = (in.opcode == 0x57);
            // image_sample_lz_o = 0x37 (LOD-0 sample with the _o packed-offset FIRST vaddr — llvm-mc
            // gfx1010 round-trip on live DOLL FXAA bytes: 0xf0dc0808/0xf0dc080a "image_sample_lz_o";
            // the offset-adjust folds into the normalized coords, see image_sample_lz_offset_2d).
            const bool is_sample_lz_o = (in.opcode == 0x37);
            // 2D_ARRAY (dim=5) is sampled here as its base 2D slice: the (u,v) coords are read as usual and
            // the array-index coord is dropped, so the shader RECOMPILES instead of being rejected (previously
            // dim!=1&&dim!=2 -> ok=false, silently SKIPPING the whole draw — real content loss, #325). Correct
            // for single-layer arrays (e.g. Unity's default textures, which are 4x4x1); a multi-layer array is
            // sampled at slice 0, a documented limitation pending full VK_IMAGE_VIEW_TYPE_2D_ARRAY support.
            const bool dim2d = (in.mimg_dim == 1u || in.mimg_dim == 5u), dim3d = (in.mimg_dim == 2u);
            const bool dimcube = (in.mimg_dim == 3u);   // CUBE: stacked-face 2D lowering (#273, below)
            if (in.mimg_dim == 5u && getenv("PROSPER_GFXLOG"))
                fprintf(stderr, "[recompile] 2D_ARRAY image_sample -> sampled as base slice 0 (array index dropped; #325)\n");
            if ((!is_sample && !is_load && !is_sample_l && !is_sample_lz && !is_sample_b && !is_gather_lz &&
                 !is_gather_lz_o && !is_sample_lz_o) || (!dim2d && !dim3d && !dimcube)) { ok = false; return true; }
            if (res->cls != ResourceClass::Texture) { ok = false; return true; }
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
                if (!is_sample && !is_sample_l && !is_sample_lz && !is_sample_b) { ok = false; return true; }
                const uint32_t ci = is_sample_b ? 1u : 0u;              // _b: bias occupies vaddr0
                uint32_t x = vread(cvg(ci)), y = vread(cvg(ci + 1)), fid = vread(cvg(ci + 2));
                const uint32_t one = b.uconst(fbits(1.0f)), zero = b.uconst(fbits(0.0f));
                uint32_t uf = b.fbin(Op_FSub, x, one);
                uint32_t vf = b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, b.fbin(Op_FSub, y, one), one), zero);
                uint32_t layer = b.fext1(Glsl_RoundEven,
                                     b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, fid, b.uconst(fbits(5.0f))), zero));
                uint32_t v6 = b.fbin(Op_FMul, b.fbin(Op_FAdd, layer, vf), b.uconst(fbits(1.0f / 6.0f)));
                b.declare_texture(res->binding, Dim_2D);
                b.image_sample_lod_2d(res->binding, uf, v6, b.uconst(0), out);
            } else if (dim3d) {
                // 3D: implicit-LOD / LOD-0 sample, or an integer texel FETCH (image_load — DOLL's
                // color-grade 3D LUT, #273).
                if (!is_sample && !is_sample_lz && !is_load) { ok = false; return true; }
                b.declare_texture(res->binding, Dim_3D);
                if (is_sample)      b.image_sample_3d(res->binding, vread(cvg(0)), vread(cvg(1)), vread(cvg(2)), out);
                else if (is_load)   b.image_fetch_3d(res->binding, vread(cvg(0)), vread(cvg(1)), vread(cvg(2)), out);
                else                b.image_sample_lod_3d(res->binding, vread(cvg(0)), vread(cvg(1)), vread(cvg(2)),
                                                          b.uconst(0), out);   // _lz: base level
            } else if (is_gather_lz || is_gather_lz_o) {
                // gather4 dmask selects ONE channel (must be a single bit); the result is always the
                // four texels of that channel -> 4 consecutive VDATA VGPRs, gather order preserved.
                uint32_t dm = in.mimg_dmask;
                if (dm != 1u && dm != 2u && dm != 4u && dm != 8u) { ok = false; return true; }
                uint32_t comp = dm == 1u ? 0u : dm == 2u ? 1u : dm == 4u ? 2u : 3u;
                b.declare_texture(res->binding, Dim_2D);
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
                b.declare_texture(res->binding, Dim_2D);
                if (is_sample_b) {      // vaddr order for _b: [bias, u, v]
                    b.image_sample_bias_2d(res->binding, vread(cvg(1)), vread(cvg(2)), vread(cvg(0)), out);
                } else if (is_sample_lz_o) {   // vaddr order for _o: [packed offset, u, v]
                    b.image_sample_lz_offset_2d(res->binding, vread(cvg(1)), vread(cvg(2)), vread(cvg(0)), out);
                } else {
                    uint32_t cu = vread(cvg(0)), cv = vread(cvg(1));
                    if (is_sample)         b.image_sample_2d(res->binding, cu, cv, out);
                    else if (is_sample_lz) b.image_sample_lod_2d(res->binding, cu, cv, b.uconst(0), out);      // LOD 0
                    // Explicit-LOD sample: LOD is the coord AFTER the spatial (+ array) coords. Plain 2D
                    // vaddr = [u, v, lod] -> cvg(2); 2D_ARRAY (dim 5) vaddr = [u, v, slice, lod] -> cvg(3).
                    // The slice sits at cvg(2), so reading LOD from cvg(2) on a 2D_ARRAY took the array
                    // index as the LOD (#373). (Slice itself is still dropped — 2D lowering, #325.)
                    else if (is_sample_l)  b.image_sample_lod_2d(res->binding, cu, cv,
                                                                 vread(cvg(in.mimg_dim == 5u ? 3 : 2)), out);
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
            // ds_write_addtid_b32 (0xb0) / ds_read_addtid_b32 (0xb1) in a GRAPHICS stage (#273):
            // a per-lane VGPR spill through LDS (addr = M0 + offset + tid*4) — DOLL's title post
            // PSes spill v15 before their accumulation loop and reload it after. Per-invocation the
            // slot is ONE value: track it in rs.lds_addtid keyed by (M0 SSA id, inst offset); the
            // matching read returns the spilled SSA value. A write with UNTRACKED M0 still no-ops
            // (nothing in this model can observe it) but poisons nothing; a read with untracked M0
            // or a never-written slot rejects loudly. VERIFIED(round-trip llvm-mc gfx1010:
            // 0xdac00000/0x00000f00 -> ds_write_addtid_b32 v15; llvm-mc gfx1030:
            // 0xdac40000/0x0f000000 -> ds_read_addtid_b32 v15).
            if (!b.is_compute && (in.opcode == 0xb0 || in.opcode == 0xb1)) {
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
            // LDS (workgroup shared memory), compute-only. ds_write_b32 (0x0d) / ds_read_b32 (0x36);
            // GDS and wider widths deferred. Byte address = ADDR VGPR + inst offset; dword index = >>2.
            if (!b.is_compute || (in.opcode != 0x0d && in.opcode != 0x36)) { ok = false; return true; }
            b.declare_lds();
            auto vread = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
            uint32_t addr = b.ibin(Op_IAdd, vread(in.src[0].value), b.uconst(in.literal));
            uint32_t idx  = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
            if (in.opcode == 0x0d) {                    // ds_write_b32: LDS[idx] = DATA0
                b.lds_store(idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
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
            rs.vreg[in.dst.value] = b.interp_read(in.vintrp_attr, in.vintrp_chan);
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
std::unordered_map<uint32_t, std::vector<uint32_t>> detect_pcrel_tables(
        const std::vector<Rdna2Inst>& ins, const uint32_t* code, size_t dwords) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> out;
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
            case Rdna2Format::SOPK: kill(in.dst.value); break;
            case Rdna2Format::SMEM: {
                uint32_t n = 1;
                switch (in.opcode) { case 0x1: case 0x9: n=2; break; case 0x2: case 0xA: n=4; break;
                    case 0x3: case 0xB: n=8; break; case 0x4: case 0xC: n=16; break; default: break; }
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
                out[in.pc] = std::vector<uint32_t>(code + off / 4, code + off / 4 + nrec / 4);
                (void)offen;                                       // offen just adds the runtime index
                break;
            }
            default: break;   // formats that don't write SGPRs
        }
    }
    return out;
}

// Emit the instruction body (shared by every stage). Handles a single recognized COUNTED loop as a real
// structured SPIR-V loop (OpLoopMerge + OpPhi for loop-carried registers); loop-FREE streams walk straight
// through, byte-identical to the pre-loop-feature behavior. `exp_fn` handles an EXP instruction per stage
// (compute: reject; fragment: MRT color; vertex: POS/PARAM). Returns false if any instruction is
// unsupported. allow_exec_update / allow_smem match the stage's emit_alu flags.
bool emit_body(SpirvCompute& b, RegState& rs, const std::vector<Rdna2Inst>& ins,
               const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
               bool allow_exec_update, bool allow_smem,
               const std::function<bool(const Rdna2Inst&)>& exp_fn,
               const uint32_t* code = nullptr, size_t dwords = 0) {   // raw stream (forward-if target check)
    // Fold PC-relative embedded-table loads (s_getpc_b64-built V#s) before the walk — emit_alu's
    // MUBUF/SOP1 handlers consult rs.mubuf_pcrel_tables (#273).
    if (code) rs.mubuf_pcrel_tables = detect_pcrel_tables(ins, code, dwords);
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
            if (in.fmt == Rdna2Format::EXP) { if (!exp_fn(in)) return false; continue; }
            bool ok = true;
            if (!emit_alu(b, rs, in, ok, allow_exec_update, &safe, allow_smem, rt, wave_ok) || !ok) {
                // PROSPER_DBG (gated, off by default): report the instruction that fails recompilation —
                // the first unsupported op / unresolved resource that makes a shader return empty.
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr, "[recompile-reject] pc=%u fmt=%d op=0x%x dst=%d(kind%d) dmask=0x%x dim=%u len=%u\n",
                        in.pc, (int)in.fmt, in.opcode, in.dst.value, (int)in.dst.kind,
                        in.mimg_dmask, in.mimg_dim, in.len_dwords);
                return false;
            }
        }
        return true;
    };
    auto& safe_branches = safe;
    if (L.found) {
        auto vget = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
        auto sget = [&](int r){ auto it = rs.sreg.find(r); return it == rs.sreg.end() ? b.uconst(0) : it->second; };
        // 1. Pre-loop body. Loops require full EXEC at entry (per-lane loop semantics); bail if narrowed.
        if (!emit_range(0, L.header_pc)) return false;
        if (rs.exec_narrowed) return false;
        // 2. Loop-carried registers -> a header OpPhi each. `cond_written` = regs written in the CONDITION
        // region [header, exit_branch): those execute on the exiting iteration too, so their post-loop
        // value is the condition-block value (which dominates the merge), NOT the phi (defect A). SCC/VCC
        // always get a phi so any cross-iteration carry is valid SSA (defect C); they're recomputed each
        // iteration in practice, so the phi is usually dead — harmless.
        std::set<int> cv, cs, condv, conds;
        loop_written_regs(ins, L.header_pc, L.backedge_pc, cv, cs);
        loop_written_regs(ins, L.header_pc, L.exit_branch_pc, condv, conds);
        const uint32_t preheader = b.cur_block;
        const uint32_t hdr = b.id(), check = b.id(), body = b.id(), cont = b.id(), merge = b.id();
        b.emit_branch(hdr); b.emit_label(hdr);
        struct PhiRec { int reg; int dom; uint32_t phi; size_t patch; };   // dom: 0=vreg,1=sreg,2=scc,3=vcc
        std::vector<PhiRec> phis;
        for (int r : cv) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, vget(r), preheader, p); rs.vreg[r] = ph; phis.push_back({r, 0, ph, p}); }
        for (int r : cs) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, sget(r), preheader, p); rs.sreg[r] = ph; phis.push_back({r, 1, ph, p}); }
        { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.scc, preheader, p); rs.scc = ph; phis.push_back({0, 2, ph, p}); }
        { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.vcc, preheader, p); rs.vcc = ph; phis.push_back({0, 3, ph, p}); }
        b.emit_loopmerge(merge, cont); b.emit_branch(check); b.emit_label(check);
        // 3. Condition block: emit [header, exit_branch); the SCC exit becomes OpBranchConditional.
        if (!emit_range(L.header_pc, L.exit_branch_pc)) return false;
        // Snapshot condition-region register values (these dominate the merge — see defect A above).
        std::unordered_map<int,uint32_t> condv_val, conds_val;
        for (int r : condv) condv_val[r] = vget(r);
        for (int r : conds) conds_val[r] = sget(r);
        // s_cbranch_scc0 exits when SCC==0 (so the loop CONTINUES when SCC!=0); scc1 is the inverse.
        uint32_t loop_cond = L.exit_on_scc0 ? rs.scc : b.bsel(rs.scc, b.bfalse(), b.btrue());
        b.emit_condbranch(loop_cond, body, merge);
        while (idx < ins.size() && ins[idx].pc < L.exit_branch_pc) ++idx;   // (already past it)
        if (idx < ins.size() && ins[idx].pc == L.exit_branch_pc) ++idx;     // skip the exit branch itself
        b.emit_label(body);
        // 4. Body [after exit_branch, back-edge). Must restore EXEC before looping (bail if left narrowed).
        if (!emit_range(L.exit_branch_pc + 1, L.backedge_pc)) return false;
        if (rs.exec_narrowed) return false;
        if (idx < ins.size() && ins[idx].pc == L.backedge_pc) ++idx;        // skip the back-edge branch
        // 5. Continue block branches back to the header; patch each phi's back-edge (value = current, cont).
        b.emit_branch(cont); b.emit_label(cont);
        for (auto& pr : phis) {
            uint32_t nv = pr.dom == 0 ? vget(pr.reg) : pr.dom == 1 ? sget(pr.reg) : pr.dom == 2 ? rs.scc : rs.vcc;
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
            else                  rs.vcc = pr.phi;
        }
        // 7. Post-loop body.
        if (!emit_range(L.exit_pc, UINT32_MAX)) return false;
    } else if (std::vector<DivLoop> Ls; true) {
        // Per-invocation EXEC/VCC-exit loops (#273/#615) + structured uniform/divergent IFs. Loops are detected
        // for the per-invocation stages only (fragment/vertex — the compute shell keeps its
        // wave-level VCC/EXEC contract); each is emitted as a real structured SPIR-V loop
        // (OpLoopMerge + header OpPhi per carried register/mask) whose per-lane condition is THIS
        // lane's EXEC bool after the header's exec recompute. The IF machinery is unchanged and
        // recurses INTO loop bodies (their nested forward-execz regions).
        Ls = detect_divergent_loops(ins, safe);
        if (b.is_compute) {
            // Compute VCC-exit loops (#590, extending #615): the fragment-stage uniformity proof is
            // data-provenance-based, not stage-based — vcc_exit_is_wave_uniform accepts a compare only
            // when every input is scalar/inline/literal or a VGPR whose nearest definition is an
            // unmodified uniform VOP1 move from a scalar. With that proven, every lane's compare bool
            // is identical, so the wave-empty vccz exit lowers to THIS invocation's !cond exactly as
            // in the fragment shell (tid-derived/varying inputs fail the proof and keep rejecting).
            // Two compute-specific guards, both conservative:
            //   * accept ONLY Condition::Vcc loops (the detector sets it only under the uniform proof;
            //     per-lane Exec-condition loops stay graphics-only until a compute case is observed);
            //   * the body must be barrier/LDS/cross-lane-free — the proof is per-WAVE, and a barrier
            //     inside a loop whose trip count could differ across the workgroup's waves would be
            //     workgroup-divergent control flow (UB). DOLL's blocked light/fill kernels are
            //     straight-line bodies, so nothing observed is lost. CONFIDENCE: MED-HIGH (shared
            //     emit machinery; spirv-val + coverage tests + Messenger guard gate it).
            bool compute_ok = !Ls.empty();
            for (const auto& L : Ls) {
                if (!compute_ok) break;
                if (L.condition != DivLoop::Condition::Vcc) { compute_ok = false; break; }
                for (const auto& in : ins) {
                    if (in.is_end || in.pc >= L.exit_pc) break;
                    if (in.pc < L.header_pc) continue;
                    if (in.fmt == Rdna2Format::DS ||
                        (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a) ||
                        (in.fmt == Rdna2Format::VOP3 &&
                         (in.opcode == 0x365 || in.opcode == 0x366))) { compute_ok = false; break; }
                }
            }
            if (!compute_ok) Ls.clear();   // unchanged behavior: the branch reaches emit_alu -> loud reject
        }
        bool cf_rejected = false;
        const std::vector<ForwardIf> Fs = detect_forward_ifs(ins, /*allow_vcc*/!b.is_compute, code, dwords, &safe,
                                                             Ls.empty() ? nullptr : &Ls, &cf_rejected,
                                                             /*uniform_vcc_compute*/b.is_compute);
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
        // `cont` = the pc control flows to after `hi` — the enclosing construct's merge chain. An
        // if/else whose merge ESCAPES the current region (the shared-outer-merge cascade) is legal
        // only when it targets exactly this continuation (no skipped instructions); anything else
        // rejects, fail-visible. Divergent execz-ifs (#273) condition on THIS lane's EXEC bool and
        // may enter/leave with EXEC narrowed — EXEC is phi'd across the merge like any other value.
        // CONFIDENCE: MED — per-invocation lowering of wave-level branches; spirv-val + exec-diff
        // kernels + the live-boot A/B gate it.
        std::function<bool(uint32_t, uint32_t, uint32_t)> emit_structured;
        // Emit one per-invocation EXEC/VCC-exit loop (#273/#615) as structured SPIR-V. Same block shape as
        // the counted-loop path (hdr -> chk -> body -> cont -> hdr, exit chk->merge) with three
        // differences: (1) the loop condition is THIS lane's EXEC bool after the header's exec
        // recompute (v_cmpx / s_andn2 exec) — continue while set, exit at execz; (2) the body is
        // emitted RECURSIVELY (nested forward-execz if regions live inside it); (3) per-lane MASKS
        // (VCC/EXEC/saved sreg_bool pairs) are loop-carried too, so each gets a header phi. At the
        // merge, a register written in the condition region keeps its exit-iteration (chk) value —
        // it dominates the merge — while body-written state takes the header phi (its value when the
        // exiting check ran); masks CREATED inside the loop are dropped at the merge (an SSA id from
        // inside the body does not dominate it — a later read then rejects loudly instead of
        // emitting invalid SPIR-V). CONFIDENCE: MED (per-invocation lowering of a wave-level loop;
        // spirv-val + execution kernels + the live-boot A/B gate it).
        std::function<bool(const DivLoop&)> emit_divloop = [&](const DivLoop& L) -> bool {
            const bool entry_exec_narrowed = rs.exec_narrowed;
            std::set<int> cv, cs, condv, conds;
            loop_written_regs(ins, L.header_pc, L.backedge_pc, cv, cs);
            loop_written_regs(ins, L.header_pc, L.exit_branch_pc, condv, conds);
            const uint32_t preheader = b.cur_block;
            const uint32_t hdr = b.id(), chk = b.id(), body = b.id(), cont = b.id(), merge = b.id();
            b.emit_branch(hdr); b.emit_label(hdr);
            struct PhiRec { int reg; int dom; uint32_t phi; size_t patch; };  // dom: 0=vreg,1=sreg,2=scc,3=vcc,4=exec,5=mask
            std::vector<PhiRec> phis;
            for (int r : cv) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, vget(r), preheader, p); rs.vreg[r] = ph; phis.push_back({r, 0, ph, p}); }
            for (int r : cs) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, sget(r), preheader, p); rs.sreg[r] = ph; phis.push_back({r, 1, ph, p}); }
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.scc, preheader, p); rs.scc = ph; phis.push_back({0, 2, ph, p}); }
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.vcc, preheader, p); rs.vcc = ph; phis.push_back({0, 3, ph, p}); }
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.exec, preheader, p); rs.exec = ph; phis.push_back({0, 4, ph, p}); }
            std::vector<int> mask_keys;                        // saved masks live at entry: loop-carried bools
            for (auto& kv : rs.sreg_bool) mask_keys.push_back(kv.first);
            std::sort(mask_keys.begin(), mask_keys.end());     // deterministic emission order
            for (int k : mask_keys) { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.sreg_bool[k], preheader, p); rs.sreg_bool[k] = ph; phis.push_back({k, 5, ph, p}); }
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
            const uint32_t loop_cond = L.condition == DivLoop::Condition::Exec ? rs.exec : rs.vcc;
            b.emit_condbranch(loop_cond, body, merge);         // execz/vccz exit: continue while bit set
            while (idx < ins.size() && ins[idx].pc < L.exit_branch_pc) ++idx;
            if (idx < ins.size() && ins[idx].pc == L.exit_branch_pc) ++idx;   // consume the exit branch
            b.emit_label(body);
            // Body (recursive: nested if regions + breaks); ends just before the back-edge.
            if (!emit_structured(L.exit_branch_pc + 1, L.backedge_pc, L.backedge_pc)) return false;
            const bool body_exec_narrowed = rs.exec_narrowed;
            if (idx < ins.size() && ins[idx].pc == L.backedge_pc) ++idx;      // consume the back-edge
            b.emit_branch(cont); b.emit_label(cont);
            for (auto& pr : phis) {
                uint32_t nv = pr.dom == 0 ? vget(pr.reg)
                            : pr.dom == 1 ? sget(pr.reg)
                            : pr.dom == 2 ? rs.scc
                            : pr.dom == 3 ? rs.vcc
                            : pr.dom == 4 ? rs.exec
                            : (rs.sreg_bool.count(pr.reg) ? rs.sreg_bool[pr.reg] : pr.phi);
                b.patch_phi(pr.patch, nv, cont);
            }
            b.emit_branch(hdr);
            b.emit_label(merge);
            for (auto& pr : phis) {
                if (pr.dom == 0)      rs.vreg[pr.reg] = condv.count(pr.reg) ? condv_val[pr.reg] : pr.phi;
                else if (pr.dom == 1) rs.sreg[pr.reg] = conds.count(pr.reg) ? conds_val[pr.reg] : pr.phi;
                else if (pr.dom == 2) rs.scc = scc_chk;    // flags/masks: the chk (exit-iteration) value
                else if (pr.dom == 3) rs.vcc = vcc_chk;    // dominates the merge and is exact (the phi is
                else if (pr.dom == 4) rs.exec = exec_chk;  // the pre-recompute value)
                else { auto itc = bool_chk.find(pr.reg);
                       rs.sreg_bool[pr.reg] = itc != bool_chk.end() ? itc->second : pr.phi; }
            }
            // Masks CREATED inside the loop: their ids do not dominate the merge — drop them.
            for (auto it = rs.sreg_bool.begin(); it != rs.sreg_bool.end();) {
                if (!std::binary_search(mask_keys.begin(), mask_keys.end(), it->first)) {
                    rs.sreg_bool_narrowed.erase(it->first);
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
                // flag!=0; scc1/vccnz are the inverse. Condition = SCC/VCC/EXEC per-invocation bool.
                uint32_t cond_reg  = F.on_exec ? rs.exec : (F.on_vcc ? rs.vcc : rs.scc);
                uint32_t exec_cond = F.on_scc0 ? cond_reg : b.bsel(cond_reg, b.bfalse(), b.btrue());
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
                    b.emit_branch(mergeL); b.emit_label(mergeL);
                    for (int r : ifv) rs.vreg[r] = b.emit_phi_2way(b.t_u32,  pre_v[r], preblock, then_v[r], thenEnd);
                    for (int r : ifs) rs.sreg[r] = b.emit_phi_2way(b.t_u32,  pre_s[r], preblock, then_s[r], thenEnd);
                    if (then_scc != pre_scc) rs.scc = b.emit_phi_2way(b.t_bool, pre_scc, preblock, then_scc, thenEnd);
                    if (then_vcc != pre_vcc) rs.vcc = b.emit_phi_2way(b.t_bool, pre_vcc, preblock, then_vcc, thenEnd);
                    // EXEC changed inside the arm (saveexec / v_cmpx / restore): merge it like any value.
                    // Narrowed-ness is sticky (either edge narrowed → post-merge writes stay predicated).
                    if (then_exec != pre_exec) rs.exec = b.emit_phi_2way(b.t_bool, pre_exec, preblock, then_exec, thenEnd);
                    rs.exec_narrowed = pre_narrowed || then_narrowed;
                    // A saved MASK (sreg_bool) whose value CHANGED inside the block would not dominate the
                    // merge — phi it like SCC/VCC (a mask newly CREATED inside the block is left as-is,
                    // matching the previous single-if behavior; its post-merge use, if any, was already
                    // undominated before this path existed).
                    for (auto& kv : rs.sreg_bool) {
                        auto p = pre_bool.find(kv.first);
                        if (p != pre_bool.end() && p->second != kv.second) {
                            kv.second = b.emit_phi_2way(b.t_bool, p->second, preblock, kv.second, thenEnd);
                            rs.sreg_bool_narrowed[kv.first] = true;   // conservative: provenance now mixed
                        }
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
                    if (then_scc != rs.scc) rs.scc = b.emit_phi_2way(b.t_bool, then_scc, thenEnd, rs.scc, elseEnd);
                    if (then_vcc != rs.vcc) rs.vcc = b.emit_phi_2way(b.t_bool, then_vcc, thenEnd, rs.vcc, elseEnd);
                    if (then_exec != rs.exec) rs.exec = b.emit_phi_2way(b.t_bool, then_exec, thenEnd, rs.exec, elseEnd);
                    rs.exec_narrowed = then_narrowed || rs.exec_narrowed;
                    for (auto& kv : rs.sreg_bool) {         // masks changed differently per arm -> phi
                        auto t = then_bool.find(kv.first);
                        if (t != then_bool.end() && t->second != kv.second) {
                            kv.second = b.emit_phi_2way(b.t_bool, t->second, thenEnd, kv.second, elseEnd);
                            rs.sreg_bool_narrowed[kv.first] = true;
                        }
                    }
                    lo = else_hi;   // continue after the merge (== hi for the escaping-cascade shape)
                }
            }
        };
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
    SpirvCompute b;
    // Size the LDS array from the shader's real allocation when known (#130): bytes -> dwords, at
    // least the ds ops need, clamped to the RDNA2 64 KB (16384-dword) max. 0 keeps the 16 KB default.
    if (lds_bytes) {
        uint32_t dw = (lds_bytes + 3) / 4;
        b.lds_dwords = dw > 16384u ? 16384u : (dw ? dw : 1u);
    }
    b.begin(num_inputs ? num_inputs : 1, rt);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    for (uint32_t k = 0; k < num_inputs; k++) rs.vreg[(int)k] = b.load_input(k);
    // Compute kernels have no EXP output; reject if one appears.
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true, /*allow_smem*/true,
                   [](const Rdna2Inst&){ return false; }, code, dwords)) return {};
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
    SpirvCompute b;
    if (config.lds_bytes) {
        uint32_t dw = (config.lds_bytes + 3) / 4;
        b.lds_dwords = std::min(16384u, std::max(1u, dw));
    }
    const uint32_t local_x = std::max(1u, config.local_x);
    const uint32_t local_y = std::max(1u, config.local_y);
    const uint32_t local_z = std::max(1u, config.local_z);
    b.begin(1, rt, local_x, local_y, local_z);

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
    for (size_t i = 0; i < config.user_sgprs.size(); i++)
        if (!descriptor_sgprs.count(static_cast<uint32_t>(i)))
            rs.sreg[static_cast<int>(i)] = b.uconst(config.user_sgprs[i]);

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
                   /*allow_smem*/true, [](const Rdna2Inst&) { return false; }, code, dwords))
        return {};
    return b.finish();
}

RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);

    // A scratch builder/state so emit_alu can run; its emitted code is discarded — we only want `ok`.
    SpirvCompute b; b.begin(1);
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
                                                          /*uniform_vcc_compute*/true);   // matches the compute shell (#590)
    auto cf_reconstructed = [&](const Rdna2Inst& i) {
        if (cL.found && (i.pc == cL.backedge_pc || i.pc == cL.exit_branch_pc)) return true;
        for (const auto& F : cFs) if (i.pc == F.branch_pc) return true;
        return false;
    };

    RecompileCoverage cov;
    for (const auto& in : ins) {
        if (in.is_end) break;
        cov.total++;
        if (in.fmt == Rdna2Format::EXP) { cov.exports++; continue; }   // handled by the stage recompilers
        bool ok = true;
        bool handled = cf_reconstructed(in)
                     || (emit_alu(b, rs, in, ok, /*allow_exec_update*/true, &safe_branches, /*allow_smem*/true,
                                  /*rt*/nullptr, /*allow_wave*/true) && ok);
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
                    // sample*: 2D (NSA ok); plus implicit-LOD image_sample (0x20) / LOD-0 image_sample_lz
                    // (0x27) from a 3D texture; sample_b (0x25) and gather4_lz (0x47) are 2D. 2D_ARRAY (dim 5)
                    // is accepted for all sample paths and handled as its base 2D slice (array index dropped,
                    // #325) — so array-sampling draws recompile+render instead of being skipped.
                    if (i.opcode == 0x20u || i.opcode == 0x27u) return i.mimg_dim == 1u || i.mimg_dim == 2u || i.mimg_dim == 5u;
                    if (i.opcode == 0x24u || i.opcode == 0x25u || i.opcode == 0x47u) return i.mimg_dim == 1u || i.mimg_dim == 5u;
                    return false;
                }
                case Rdna2Format::MUBUF:  return i.opcode <= 0x07u ||                    // load/store_format_*
                                                 (i.opcode >= 0x0Cu && i.opcode <= 0x0Fu);  // load_dword/x2/x4/x3 (need the V#)
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

std::vector<uint32_t> recompile_fragment(const uint32_t* code, size_t dwords, const ShaderResourceTable* rt) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);

    SpirvCompute b;
    b.begin_fragment(rt);
    // Classify each interpolated attribute by HOW it's read (#152): v_interp_p2 (op 1) = smooth
    // rasterizer interpolation; v_interp_mov (op 2) = a raw per-vertex / provoking-vertex value (a
    // flat read). An attribute read only via v_interp_mov gets its Input varying decorated Flat so
    // the driver delivers the provoking vertex value, not a blend of the three. An attribute read
    // via BOTH is contradictory (a varying can't be smooth and flat at once) -> reject the shader.
    { std::unordered_set<uint32_t> smooth_attr;
      for (const auto& in : ins) {
          if (in.is_end) break;
          if (in.fmt != Rdna2Format::VINTRP) continue;
          if (in.opcode == 1) smooth_attr.insert(in.vintrp_attr);        // v_interp_p2
          else if (in.opcode == 2) b.flat_attrs.insert(in.vintrp_attr);  // v_interp_mov
      }
      for (uint32_t a : b.flat_attrs) if (smooth_attr.count(a)) return {};   // mixed smooth+flat: reject
    }
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // FRAGMENT-only: also linearize alpha-test / clip() kill-mask s_cbranch_scc0/scc1 early-outs (Unity
    // cutout + text draws). Merged into the same set so emit_alu drops the branch and detect_forward_if
    // skips it; the block's EXEC narrow + the export's OpKill do the per-invocation discard (#102). This is
    // NOT added for the vertex/compute shells (their scc branches are real uniform-ifs / NGG culling).
    for (uint32_t pc : mask_test_branches(ins)) safe_branches.insert(pc);
    bool exported = false;
    auto exp_fn = [&](const Rdna2Inst& in) -> bool {         // EXP MRT0..7 -> the color output
        // An export while EXEC is narrowed (lanes killed by an alpha test / v_cmpx and not restored to
        // all-on) must not write the inactive lanes. Lower it to a real fragment discard: OpKill the lanes
        // whose EXEC bit is false, then export from the survivors under full EXEC. This is exactly the
        // alpha-tested-sprite shape (image_sample -> v_cmp alpha<ref -> s_andn2 saved,saved,vcc -> s_wqm
        // exec,saved -> shade -> export): the surviving lanes are the ones that passed the test. (When EXEC
        // was never narrowed this is a no-op — the common sRGB/tonemap restore-then-export path.)
        if (rs.exec_narrowed) { b.discard_unless(rs.exec); rs.exec = b.btrue(); rs.exec_narrowed = false; }
        if (in.exp_target <= 7 && !exported) {
            bool eok = true;   // a Special (wave-mask) source has no data value — reject, don't export 0 (#134)
            if (in.exp_compr) {
                // COMPR: the 4 channels are two f16x2 pairs — src[0] holds (r,g), src[1] holds (b,a).
                // Unpack each half to a float and reassemble the vec4 (the pkrtz'd tonemap/sRGB output).
                uint32_t p0 = operand_bits(b, rs, in, in.src[0], &eok), p1 = operand_bits(b, rs, in, in.src[1], &eok);
                b.export_color(b.unpack_half(p0, 0), b.unpack_half(p0, 1),
                               b.unpack_half(p1, 0), b.unpack_half(p1, 1));
            } else {
                b.export_color(operand_bits(b, rs, in, in.src[0], &eok), operand_bits(b, rs, in, in.src[1], &eok),
                               operand_bits(b, rs, in, in.src[2], &eok), operand_bits(b, rs, in, in.src[3], &eok));
            }
            if (!eok) return false;
            exported = true;
        }
        return true;   // ignore NULL / additional exports for now
    };
    // cmpx is now ALLOWED (allow_exec_update=true): a fragment divergent-if (v_cmpx ... s_mov exec,saved)
    // is handled by EXEC predication like compute, and the export is guarded above. Memory ops need a
    // resource table. Loops (if any) are reconstructed by emit_body.
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true, /*allow_smem*/rt != nullptr, exp_fn, code, dwords)) return {};
    if (!exported) return {};
    return b.finish();
}

std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords, const ShaderResourceTable* rt) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);

    SpirvCompute b;
    b.begin_vertex(rt);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // NGG vertex shaders (s_sendmsg GS_ALLOC_REQ present) carry the vertex index in v5, not v0, and wrap
    // the body in wave-packing plumbing (s_sendmsg / exp prim / s_lshr_b64 exec) that lowers to no-ops in
    // our per-invocation model. Detect NGG and bind the index to v5 as well.
    bool ngg = false;
    for (const auto& in : ins) { if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10) { ngg = true; break; } }
    uint32_t vidx = b.load_vertex_index();
    rs.vreg[0] = vidx;                       // VS ABI: v0 = vertex index
    if (ngg) rs.vreg[5] = vidx;              // NGG VS ABI: v5 = vertex index
    bool exported = false;
    auto exp_fn = [&](const Rdna2Inst& in) -> bool {         // EXP POS0..3 -> gl_Position; PARAM -> varyings
        bool eok = true;   // a Special (wave-mask) source has no data value — reject, don't export 0 (#134)
        // v_cmpx is now allowed in the vertex shell (allow_exec_update=true below): a divergent block
        // (v_cmpx … s_mov_b64 exec, saved — DOLL's per-vertex lighting/fog attenuation) predicates its
        // VGPR writes like compute. A vertex MUST still export from full EXEC — the compiled shape
        // always restores EXEC before its pos/param exports; if one ever arrives narrowed, reject
        // (fail-visibly) rather than export possibly-inactive-lane values.
        if (rs.exec_narrowed && (in.exp_target >= 32 || (in.exp_target >= 12 && in.exp_target <= 15))) return false;
        if (in.exp_target >= 12 && in.exp_target <= 15 && !exported) {
            b.export_position(operand_bits(b, rs, in, in.src[0], &eok), operand_bits(b, rs, in, in.src[1], &eok),
                              operand_bits(b, rs, in, in.src[2], &eok), operand_bits(b, rs, in, in.src[3], &eok));
            exported = true;
        } else if (in.exp_target >= 32) {                    // PARAM0.. -> Output varying (location N)
            b.export_param(in.exp_target - 32, operand_bits(b, rs, in, in.src[0], &eok), operand_bits(b, rs, in, in.src[1], &eok),
                           operand_bits(b, rs, in, in.src[2], &eok), operand_bits(b, rs, in, in.src[3], &eok));
        }
        return eok;
    };
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true, /*allow_smem*/rt != nullptr, exp_fn, code, dwords)) return {};
    if (!exported) return {};
    return b.finish();
}

} // namespace prosper::gpu
