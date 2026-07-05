// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
#include "rdna2_to_spirv.hpp"
#include "rdna2_decode.hpp"
#include "shader_resources.hpp"
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace prosper::gpu {
namespace {

enum : uint32_t {
    Op_ExtInstImport=11, Op_ExtInst=12, Op_MemoryModel=14, Op_EntryPoint=15, Op_ExecutionMode=16,
    Op_Capability=17, Op_TypeVoid=19, Op_TypeBool=20, Op_TypeInt=21, Op_TypeFloat=22, Op_TypeVector=23,
    Op_TypeRuntimeArray=29, Op_TypeStruct=30, Op_TypePointer=32, Op_TypeFunction=33,
    Op_ConstantTrue=41, Op_ConstantFalse=42, Op_Constant=43, Op_Function=54, Op_FunctionEnd=56, Op_Variable=59,
    Op_LogicalOr=166, Op_LogicalAnd=167, Op_Select=169, Op_FOrdEqual=180, Op_FOrdNotEqual=182, Op_FOrdLessThan=184, Op_FOrdGreaterThan=186,
    Op_FOrdLessThanEqual=188, Op_FOrdGreaterThanEqual=190,
    Op_Load=61, Op_Store=62, Op_AccessChain=65, Op_Decorate=71, Op_MemberDecorate=72,
    Op_ConvertFToU=109, Op_ConvertFToS=110, Op_ConvertSToF=111, Op_ConvertUToF=112, Op_Bitcast=124,
    Op_CompositeConstruct=80, Op_CompositeExtract=81, Op_IAdd=128, Op_FAdd=129, Op_ISub=130, Op_FSub=131, Op_IMul=132, Op_FMul=133,
    Op_UMulExtended=151, Op_SMulExtended=152,   // {lo,hi} struct results (for mul_hi)
    Op_FDiv=136, Op_IEqual=170, Op_INotEqual=171, Op_UGreaterThan=172, Op_SGreaterThan=173,
    Op_UGreaterThanEqual=174, Op_SGreaterThanEqual=175, Op_ULessThan=176, Op_SLessThan=177,
    Op_ULessThanEqual=178, Op_SLessThanEqual=179,
    Op_ShiftRightLogical=194, Op_ShiftRightArithmetic=195, Op_ShiftLeftLogical=196, Op_BitwiseOr=197,
    Op_BitwiseXor=198, Op_BitwiseAnd=199, Op_Not=200, Op_BitFieldSExtract=202, Op_BitFieldUExtract=203,
    Op_BitReverse=204,
    Op_TypeImage=25, Op_TypeSampledImage=27, Op_SampledImage=86,
    Op_ImageSampleImplicitLod=87, Op_ImageSampleExplicitLod=88, Op_ImageFetch=95, Op_Image=100,
    Op_ImageRead=98, Op_ImageWrite=99,
    Op_TypeArray=28, Op_ControlBarrier=224,
    Op_SelectionMerge=247, Op_Label=248, Op_Branch=249, Op_BranchConditional=250, Op_Return=253,
};
// GLSL.std.450 extended-instruction numbers.
enum : uint32_t { Glsl_RoundEven=2, Glsl_Trunc=3, Glsl_Floor=8, Glsl_Ceil=9, Glsl_Fract=10, Glsl_Exp2=29, Glsl_Log2=30,
                  Glsl_Sqrt=31, Glsl_InverseSqrt=32, Glsl_FMin=37, Glsl_UMin=38, Glsl_SMin=39, Glsl_FMax=40,
                  Glsl_UMax=41, Glsl_SMax=42, Glsl_PackHalf2x16=58, Glsl_UnpackHalf2x16=62 };
enum : uint32_t {
    Cap_Shader=1, Addr_Logical=0, Mem_GLSL450=1, Exec_Vertex=0, Exec_Fragment=4, Exec_GLCompute=5,
    EM_OriginUpperLeft=7, EM_LocalSize=17,
    SC_Input=1, SC_UniformConstant=0, SC_Output=3, SC_StorageBuffer=12, FC_None=0,
    Dim_1D=0, Dim_2D=1, Dim_3D=2,   // SPIR-V Dim. (2D coincides with the SQ_RSRC 2D dim value, but distinct.)
    Cap_Sampled1D=43, Cap_Image1D=44,   // Dim=1D needs Sampled1D; a 1D STORAGE image (read/write) also needs Image1D
    Cap_StorageImageReadWithoutFormat=55, Cap_StorageImageWriteWithoutFormat=56,  // for Format=Unknown storage images
    Img_Sampled_Storage=2,   // OpTypeImage "Sampled" operand: 2 = used WITHOUT a sampler (read/write storage image)
    ImgFmt_Unknown=0,        // OpTypeImage "Image Format": Unknown (runtime view format; needs the caps above)
    ImgOp_Lod=2,   // ImageOperands bit: an explicit LOD follows (for OpImageFetch on a sampled image).
    SC_Workgroup=4, Scope_Workgroup=2, MemSem_WGAcqRel=0x108,   // LDS: Workgroup storage + barrier scope/semantics
    Dec_Block=2, Dec_ArrayStride=6, Dec_BuiltIn=11, Dec_Location=30, Dec_Binding=33,
    Dec_DescriptorSet=34, Dec_Offset=35,
    BI_Position=0, BI_GlobalInvocationId=28, BI_VertexIndex=42,
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
    uint32_t v_gid=0, v_in=0, v_out=0, gidx=0, f_main=0, glsl=0, bconst_false=0;
    uint32_t v_cbuf=0, v_cbuf1=0, t_ptr_sb_u32=0;   // scalar-memory constant buffers (bindings 2 and 3)
    bool     is_fragment=0;                          // true in the fragment shell (gates VINTRP interp)
    bool     is_compute=0;                            // true in the compute shell (gates LDS / s_barrier)
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
    uint32_t cvt_f2u(uint32_t bits) { uint32_t r = id(); put(code, Op_ConvertFToU, {t_u32, r, bcf(bits)}); return r; }
    // Float ordered compare on bit-operands -> bool (for VCC). select() picks bits by a bool condition.
    uint32_t fcmp(uint32_t cmpop, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, cmpop, {t_bool, r, bcf(a), bcf(b)}); return r; }
    uint32_t bfalse() { if (!bconst_false) { bconst_false = id(); put(types, Op_ConstantFalse, {t_bool, bconst_false}); } return bconst_false; }
    uint32_t sel(uint32_t cond, uint32_t tval, uint32_t fval) { uint32_t r = id(); put(code, Op_Select, {t_u32, r, cond, tval, fval}); return r; }
    uint32_t bsel(uint32_t cond, uint32_t tval, uint32_t fval) { uint32_t r = id(); put(code, Op_Select, {t_bool, r, cond, tval, fval}); return r; }  // bool-domain select (wave masks)
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
    uint32_t cvt_f2i(uint32_t bits) { uint32_t ri = id(); put(code, Op_ConvertFToS, {t_i32, ri, bcf(bits)}); return i2u(ri); }
    uint32_t cvt_i2f(uint32_t bits) { uint32_t rf = id(); put(code, Op_ConvertSToF, {t_f32, rf, bcs(bits)}); return bcu(rf); }
    // Integer compares -> bool. scmp treats operands as signed, ucmp as unsigned.
    uint32_t scmp(uint32_t op, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, op, {t_bool, r, bcs(a), bcs(b)}); return r; }
    uint32_t ucmp(uint32_t op, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, op, {t_bool, r, a, b}); return r; }
    // Bitfield extract (base, offset, count) -> bits. Unsigned and signed variants.
    uint32_t bfe_u(uint32_t base, uint32_t off, uint32_t cnt) { uint32_t r = id(); put(code, Op_BitFieldUExtract, {t_u32, r, base, off, cnt}); return r; }
    uint32_t bfe_s(uint32_t base, uint32_t off, uint32_t cnt) { uint32_t ri = id(); put(code, Op_BitFieldSExtract, {t_i32, ri, bcs(base), off, cnt}); return i2u(ri); }
    // Lazily declared 2-float vector type (types are emitted as a block before code, so on-demand is safe).
    uint32_t t_v2f_cache = 0;
    uint32_t t_v2f() { if (!t_v2f_cache) { t_v2f_cache = id(); put(types, Op_TypeVector, {t_v2f_cache, t_f32, 2}); } return t_v2f_cache; }
    // v_cvt_pkrtz_f16_f32: pack src0->low f16, src1->high f16 of a 32-bit result (raw VGPR bits).
    uint32_t pack_half2x16(uint32_t a, uint32_t b) {
        uint32_t vec = id(); put(code, Op_CompositeConstruct, {t_v2f(), vec, bcf(a), bcf(b)});
        uint32_t r = id(); putv(code, Op_ExtInst, {t_u32, r, glsl, Glsl_PackHalf2x16, vec}); return r;
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

    // Combined image+sampler support (MIMG image_sample). One float-2D OpTypeImage/OpTypeSampledImage
    // is shared across textures; each texture is a COMBINED_IMAGE_SAMPLER UniformConstant at its binding.
    uint32_t t_image = 0, t_sampled_image = 0;
    std::unordered_map<uint32_t, uint32_t> tex_var;   // binding -> combined-sampler OpVariable id
    // Declare (idempotently) a combined image+sampler at descriptor-set 0, `binding`. Requires t_f32.
    void declare_texture(uint32_t binding) {
        if (tex_var.count(binding)) return;
        if (!t_image) {
            t_image = id();   // sampled f32, Dim=2D, Depth=0, Arrayed=0, MS=0, Sampled=1, Format=Unknown
            put(types, Op_TypeImage, {t_image, t_f32, Dim_2D, 0, 0, 0, 1, 0});
            t_sampled_image = id();
            put(types, Op_TypeSampledImage, {t_sampled_image, t_image});
        }
        uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_UniformConstant, t_sampled_image});
        uint32_t v = id();     put(types, Op_Variable,    {t_ptr, v, SC_UniformConstant});
        put(deco, Op_Decorate, {v, Dec_DescriptorSet, 0});
        put(deco, Op_Decorate, {v, Dec_Binding, binding});
        tex_var[binding] = v;
    }
    // image_sample 2D: sample the combined sampler at `binding` with (u,v) float-BITS coords; fills
    // out[0..3] with the RGBA result components as raw VGPR bits. Fragment stage (implicit LOD).
    void image_sample_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {t_sampled_image, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
        uint32_t res   = id(); put(code, Op_ImageSampleImplicitLod, {t_v4f, res, si, coord});
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

    // --- STORAGE images (MIMG image_load / image_store WITHOUT a sampler; compute copy/blit) ---
    // Modeled with a UINT-sampled OpTypeImage, Format=Unknown, Sampled=2 (storage), so OpImageRead/Write
    // move raw 32-bit texels — an exact fit for our untyped-VGPR model (any real format reinterpretation
    // lives in the bound image view / T# descriptor). Requires the read/write-without-format caps.
    uint32_t t_v4u_cache = 0;
    uint32_t t_v4u() { if (!t_v4u_cache) { t_v4u_cache = id(); put(types, Op_TypeVector, {t_v4u_cache, t_u32, 4}); } return t_v4u_cache; }
    uint32_t t_v3u_c = 0;   // integer coordinate vector type (3D); 2D reuses the shared t_v2u()
    std::unordered_map<uint32_t, uint32_t> stg_img_type;   // spirv Dim -> OpTypeImage id
    std::unordered_map<uint32_t, uint32_t> stg_img_var;    // binding -> storage-image OpVariable id
    bool declared_read_wo_fmt = false, declared_write_wo_fmt = false, declared_sampled1d = false;
    // Declare (idempotently) a uint storage image of SPIR-V `dim` at set 0, `binding`.
    void declare_storage_image(uint32_t binding, uint32_t dim) {
        if (dim == Dim_1D && !declared_sampled1d) {   // SPIR-V: Dim=1D needs Sampled1D; storage 1D also needs Image1D
            put(caps, Op_Capability, {Cap_Sampled1D});
            put(caps, Op_Capability, {Cap_Image1D});
            declared_sampled1d = true;
        }
        if (!stg_img_type.count(dim)) {
            uint32_t ti = id();
            put(types, Op_TypeImage, {ti, t_u32, dim, 0, 0, 0, Img_Sampled_Storage, ImgFmt_Unknown});
            stg_img_type[dim] = ti;
        }
        if (stg_img_var.count(binding)) return;
        uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_UniformConstant, stg_img_type[dim]});
        uint32_t v = id();     put(types, Op_Variable,    {t_ptr, v, SC_UniformConstant});
        put(deco, Op_Decorate, {v, Dec_DescriptorSet, 0});
        put(deco, Op_Decorate, {v, Dec_Binding, binding});
        stg_img_var[binding] = v;
    }
    // Build the integer coordinate operand for `dim` from up to 3 raw-bit VGPR coords (used as u32 texel
    // indices). 1D = scalar u32; 2D = uvec2; 3D = uvec3.
    uint32_t stg_coord(uint32_t dim, const uint32_t* c) {
        if (dim == Dim_1D) return c[0];
        // 2D reuses the shared uvec2 helper (also used by texelFetch) so a shader mixing a texture
        // image_load and a 2D storage image doesn't emit a duplicate OpTypeVector %uint 2.
        if (dim == Dim_2D) { uint32_t v = id(); put(code, Op_CompositeConstruct, {t_v2u(), v, c[0], c[1]}); return v; }
        // 3D: reuse the compute shell's uvec3 (t_v3u, declared for gl_GlobalInvocationID) if present —
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
    void image_read(uint32_t binding, uint32_t dim, const uint32_t* coords, uint32_t out[4]) {
        if (!declared_read_wo_fmt) { put(caps, Op_Capability, {Cap_StorageImageReadWithoutFormat}); declared_read_wo_fmt = true; }
        uint32_t img   = id(); put(code, Op_Load,      {stg_img_type[dim], img, stg_img_var[binding]});
        uint32_t coord = stg_coord(dim, coords);
        uint32_t res   = id(); put(code, Op_ImageRead, {t_v4u(), res, img, coord});
        for (uint32_t c = 0; c < 4; c++) { uint32_t e = id(); put(code, Op_CompositeExtract, {t_u32, e, res, c}); out[c] = e; }
    }
    // image_store: OpImageWrite raw-bit VGPR components vals[0..3] as a uvec4 texel to the storage image.
    // When `predicated` (narrowed EXEC), the write is wrapped in a selection merge on `pred` (the per-lane
    // EXEC bool) so inactive lanes do not write — a real conditional store, like cbuf_store. (Image OOB is
    // not covered by robustBufferAccess, so guarding matters: it also skips a lane's write when EXEC is off
    // e.g. a grid-tail bounds check.)
    void image_write(uint32_t binding, uint32_t dim, const uint32_t* coords, const uint32_t vals[4],
                     bool predicated = false, uint32_t pred = 0) {
        if (!declared_write_wo_fmt) { put(caps, Op_Capability, {Cap_StorageImageWriteWithoutFormat}); declared_write_wo_fmt = true; }
        uint32_t img   = id(); put(code, Op_Load, {stg_img_type[dim], img, stg_img_var[binding]});
        uint32_t coord = stg_coord(dim, coords);
        uint32_t texel = id(); put(code, Op_CompositeConstruct, {t_v4u(), texel, vals[0], vals[1], vals[2], vals[3]});
        if (!predicated) { put(code, Op_ImageWrite, {img, coord, texel}); return; }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then});
        put(code, Op_ImageWrite, {img, coord, texel});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge});
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
    // Load one dword (raw bits) from constant-buffer `slot` (0 -> binding 2, 1 -> binding 3) at dword
    // index `idx` (SMEM). Multiple bindings let descriptor provenance route loads to distinct buffers.
    uint32_t cbuf_load(uint32_t idx, uint32_t slot = 0) {
        uint32_t buf = slot ? v_cbuf1 : v_cbuf;
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_u32, p, buf, uconst(0), idx});
        uint32_t r = id(); put(code, Op_Load, {t_u32, r, p}); return r;
    }
    // Store one dword `value` to constant/vertex buffer `slot` at dword index `idx` (MUBUF store). When
    // `predicated`, the store is wrapped in a selection merge on `pred` (the per-lane EXEC bool) so
    // inactive lanes do not write — a real conditional store, not a select of a loaded old value.
    void cbuf_store(uint32_t idx, uint32_t value, uint32_t slot, bool predicated, uint32_t pred) {
        uint32_t buf = slot ? v_cbuf1 : v_cbuf;
        if (!predicated) {
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_u32, p, buf, uconst(0), idx});
            put(code, Op_Store, {p, value}); return;
        }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then});
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_u32, p, buf, uconst(0), idx});
        put(code, Op_Store, {p, value});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge});
    }
    // LDS (Local Data Share) — a workgroup-shared u32 array for compute ds_read/ds_write. Declared on
    // first use. 4096 dwords (16 KB, the RDNA2 per-workgroup LDS size).
    uint32_t lds_var = 0, t_ptr_wg_u32 = 0;
    void declare_lds() {
        if (lds_var) return;
        uint32_t len = uconst(4096);
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
        put(code, Op_Label, {then});
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32, p, lds_var, idx});
        put(code, Op_Store, {p, value});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge});
    }
    // s_barrier: workgroup execution + memory barrier (OpControlBarrier).
    void barrier() {
        put(code, Op_ControlBarrier, {uconst(Scope_Workgroup), uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel)});
    }
    // Declare the two scalar-memory constant/vertex buffers (bindings 2 & 3) that SMEM / buffer_load_
    // format_* read. Called by every shell (compute/vertex/fragment) so cbuf_load works in each.
    // Requires t_u32 to already be declared. Unused by shaders without memory ops.
    void declare_cbufs() {
        uint32_t t_rta_u = id(), t_struct_u = id(), t_ptr_sb_struct_u = id();
        v_cbuf = id(); v_cbuf1 = id(); t_ptr_sb_u32 = id();
        put(deco, Op_Decorate, {t_rta_u, Dec_ArrayStride, 4});
        put(deco, Op_MemberDecorate, {t_struct_u, 0, Dec_Offset, 0});
        put(deco, Op_Decorate, {t_struct_u, Dec_Block});
        put(deco, Op_Decorate, {v_cbuf,  Dec_DescriptorSet, 0}); put(deco, Op_Decorate, {v_cbuf,  Dec_Binding, 2});
        put(deco, Op_Decorate, {v_cbuf1, Dec_DescriptorSet, 0}); put(deco, Op_Decorate, {v_cbuf1, Dec_Binding, 3});
        put(types, Op_TypeRuntimeArray, {t_rta_u, t_u32});
        put(types, Op_TypeStruct, {t_struct_u, t_rta_u});
        put(types, Op_TypePointer, {t_ptr_sb_struct_u, SC_StorageBuffer, t_struct_u});
        put(types, Op_Variable, {t_ptr_sb_struct_u, v_cbuf,  SC_StorageBuffer});
        put(types, Op_Variable, {t_ptr_sb_struct_u, v_cbuf1, SC_StorageBuffer});
        put(types, Op_TypePointer, {t_ptr_sb_u32, SC_StorageBuffer, t_u32});
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

    void begin(uint32_t input_stride) {
        stride = input_stride;
        t_void = id(); t_fn = id(); t_f32 = id(); t_u32 = id(); t_i32 = id(); t_v3u = id(); t_bool = id();
        uint32_t t_ptr_in_v3u = id(); v_gid = id();
        uint32_t t_rta = id(), t_struct = id(), t_ptr_sb_struct = id();
        v_in = id(); v_out = id(); t_ptr_sb_f32 = id();
        f_main = id(); uint32_t lbl = id(); glsl = id();

        put(caps, Op_Capability, {Cap_Shader});
        { std::vector<uint32_t> o{glsl}; pstr(o, "GLSL.std.450"); putv(extimp, Op_ExtInstImport, o); }
        put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
        is_compute = true;
        exec_model = Exec_GLCompute; iface = {v_gid};   // EntryPoint deferred to finish()
        put(exec, Op_ExecutionMode, {f_main, EM_LocalSize, 64, 1, 1});
        put(deco, Op_Decorate, {v_gid, Dec_BuiltIn, BI_GlobalInvocationId});
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
        put(types, Op_TypeBool, {t_bool});
        put(types, Op_TypePointer, {t_ptr_in_v3u, SC_Input, t_v3u});
        put(types, Op_Variable, {t_ptr_in_v3u, v_gid, SC_Input});
        put(types, Op_TypeRuntimeArray, {t_rta, t_f32});
        put(types, Op_TypeStruct, {t_struct, t_rta});
        put(types, Op_TypePointer, {t_ptr_sb_struct, SC_StorageBuffer, t_struct});
        put(types, Op_Variable, {t_ptr_sb_struct, v_in, SC_StorageBuffer});
        put(types, Op_Variable, {t_ptr_sb_struct, v_out, SC_StorageBuffer});
        put(types, Op_TypePointer, {t_ptr_sb_f32, SC_StorageBuffer, t_f32});
        declare_cbufs();   // scalar-memory constant/vertex buffers (bindings 2 & 3)
        put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
        put(code, Op_Label, {lbl});
        uint32_t ld = id(); put(code, Op_Load, {t_v3u, ld, v_gid});
        gidx = id(); put(code, Op_CompositeExtract, {t_u32, gidx, ld, 0});
    }
    // --- Fragment-shader shell: a location-0 vec4 color output; EXP MRT0 stores to it. ---
    uint32_t t_v4f = 0, v_color = 0;
    void begin_fragment(bool with_cbufs = false) {
        t_void = id(); t_fn = id(); t_f32 = id(); t_u32 = id(); t_i32 = id(); t_bool = id();
        t_v4f = id(); uint32_t t_ptr_out = id(); v_color = id(); f_main = id(); uint32_t lbl = id(); glsl = id();
        put(caps, Op_Capability, {Cap_Shader});
        { std::vector<uint32_t> o{glsl}; pstr(o, "GLSL.std.450"); putv(extimp, Op_ExtInstImport, o); }
        put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
        is_fragment = true;
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
        if (with_cbufs) declare_cbufs();   // only when the shader has memory ops (keeps no-op renders binding-free)
        put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
        put(code, Op_Label, {lbl});
    }
    // Write a vec4(r,g,b,a) (bit-operands) to the fragment color output (EXP MRT0).
    void export_color(uint32_t r, uint32_t g, uint32_t bl, uint32_t a) {
        uint32_t v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(r), bcf(g), bcf(bl), bcf(a)});
        put(code, Op_Store, {v_color, v});
    }

    // --- Vertex-shader shell: gl_VertexIndex input + gl_Position (member 0 of a gl_PerVertex Block). ---
    uint32_t v_vid = 0, v_pos = 0, t_ptr_out_v4f = 0;
    void begin_vertex(bool with_cbufs = false) {
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
        if (with_cbufs) declare_cbufs();   // vertex fetch (buffer_load_format_*) reads these
        put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
        put(code, Op_Label, {lbl});
    }
    // Load gl_VertexIndex as raw bits (VGPR v0 for a vertex shader).
    uint32_t load_vertex_index() { uint32_t r = id(); put(code, Op_Load, {t_i32, r, v_vid}); return i2u(r); }
    // Write vec4(x,y,z,w) (bit-operands) to gl_Position (EXP POS0).
    void export_position(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
        uint32_t v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(x), bcf(y), bcf(z), bcf(w)});
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_out_v4f, p, v_pos, uconst(0)});
        put(code, Op_Store, {p, v});
    }

    // --- Interpolated I/O varyings: VS EXP PARAM_n (output) <-> FS v_interp attribute (input) ---
    std::unordered_map<uint32_t, uint32_t> in_varying, out_varying;
    uint32_t t_ptr_in_v4f = 0;
    // FS: an Input vec4 at Location=attr, smooth-interpolated by the rasterizer (declared on first use).
    uint32_t frag_input(uint32_t attr) {
        auto it = in_varying.find(attr); if (it != in_varying.end()) return it->second;
        if (!t_ptr_in_v4f) { t_ptr_in_v4f = id(); put(types, Op_TypePointer, {t_ptr_in_v4f, SC_Input, t_v4f}); }
        uint32_t v = id(); put(types, Op_Variable, {t_ptr_in_v4f, v, SC_Input});
        put(deco, Op_Decorate, {v, Dec_Location, attr});
        in_varying[attr] = v; iface.push_back(v); return v;
    }
    // Read component `chan` (0..3) of interpolated attribute `attr` -> float bits (v_interp_p2 / mov).
    uint32_t interp_read(uint32_t attr, uint32_t chan) {
        uint32_t v = frag_input(attr);
        uint32_t vec = id(); put(code, Op_Load, {t_v4f, vec, v});
        uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, vec, chan}); return bcu(e);
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
    std::unordered_map<int, uint32_t> sreg_srt;    // SGPR holding a descriptor -> its user_data/SRT byte offset
                                                   // (descriptor provenance: s_load_dwordx4 tags, s_buffer_load resolves)
    uint32_t vcc = 0;
    uint32_t scc = 0;          // scalar condition code (bool); set by s_cmp_*/SCC-writing SOP2, read by s_cselect
    uint32_t exec = 0;         // per-lane execution mask (bool); v_cmpx narrows it, output store honors it
    bool exec_narrowed = false; // true once EXEC is narrowed below all-lanes-on (so VGPR writes predicate)
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
        case 0x7d:   // s_waitcnt_vscnt
            return true;
        default:
            return false;
    }
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
            if (in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 || in.fmt == Rdna2Format::VOP3 ||
                sopp_is_noop(in)) {
                continue;
            }
            ok = false;
            break;
        }
        if (ok) safe.insert(br.pc);
    }
    return safe;
}

// Resolve an operand to its raw 32-bit value (bits). Float ops bitcast these to float.
uint32_t operand_bits(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, const Operand& o) {
    switch (o.kind) {
        case OperandKind::VGPR: { auto it = rs.vreg.find(o.value); return it == rs.vreg.end() ? b.uconst(0) : it->second; }
        case OperandKind::SGPR: { auto it = rs.sreg.find(o.value); return it == rs.sreg.end() ? b.uconst(0) : it->second; }
        case OperandKind::InlineInt:   return b.uconst((uint32_t)o.value);
        case OperandKind::InlineFloat: return b.uconst(fbits(inline_float_value((uint32_t)o.value)));
        case OperandKind::Literal:     return b.uconst(in.literal);
        default: return b.uconst(0);
    }
}

// Emit one ALU instruction (VOP1/2/C/3 or SOP1/2) into `b`, updating `rs`. Returns true if `in` is an
// ALU format handled here; sets ok=false if it is an ALU op this stage doesn't support yet. Non-ALU
// formats (EXP/memory/...) return false so the stage-specific caller can handle them.
bool emit_alu(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_exec_update,
              const std::unordered_set<uint32_t>* safe_execz = nullptr, bool allow_smem = false,
              const ShaderResourceTable* rt = nullptr) {
    auto& vreg = rs.vreg; uint32_t& vcc = rs.vcc;
    auto val = [&](const Operand& o) { return operand_bits(b, rs, in, o); };
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
                    } else rs.sreg_bool[in.dst.value] = m;
                    return true;
                }
                if (in.opcode == 0x04) {                    // s_mov_b64
                    if (is_exec(in.dst)) {                  // set/restore EXEC
                        if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                            rs.exec = b.btrue(); rs.exec_narrowed = false;   // exec = all lanes on
                        } else { uint32_t m = src_mask(in.src[0]); if (!m) ok = false;
                                 else { rs.exec = m; rs.exec_narrowed = true; } }
                    } else {                                // s_mov_b64 sDST, <mask> : save a mask
                        uint32_t m = src_mask(in.src[0]); if (!m) ok = false;
                        else rs.sreg_bool[in.dst.value] = m;
                    }
                } else {                                    // s_and/or_saveexec_b64 sDST, src
                    rs.sreg_bool[in.dst.value] = rs.exec;   // save current EXEC to the dest SGPR pair
                    uint32_t m = src_mask(in.src[0]);
                    if (!m) ok = false;
                    else { rs.exec = (in.opcode == 0x24) ? b.land(rs.exec, m) : b.lor(rs.exec, m);
                           rs.exec_narrowed = true; }
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
                else if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = r;
                else rs.sreg_bool[in.dst.value] = r;
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
                case 0x26: d = b.ibin(Op_IMul, a, c); break;         // s_mul_i32 (low 32 bits; no SCC)
                case 0x35: d = b.umul_hi(a, c); break;               // s_mul_hi_u32 (high 32 bits; no SCC)
                case 0x37: d = b.smul_hi(a, c); break;               // s_mul_hi_i32 (high 32 bits; no SCC)
                case 0x27: {                                         // s_bfe_u32: offset=src1[4:0], width=src1[22:16]
                    uint32_t off = b.ibin(Op_BitwiseAnd, c, b.uconst(0x1f));
                    uint32_t width = b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, c, b.uconst(16)), b.uconst(0x7f));
                    d = b.bfe_u(a, off, width); scc_nz(d); break;
                }
                default: ok = false;
            }
            return true;
        }
        case Rdna2Format::SOPC: {
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
                default: ok = false;
            }
            return true;
        }
        case Rdna2Format::VOP1: {
            uint32_t a = val(in.src[0]); uint32_t old_d = vreg_old(b, rs, in.dst.value);
            if (in.opcode == 0x02) {   // v_readfirstlane_b32: SGPR dst = value of the lowest active lane
                // Cross-lane broadcast. Our per-lane scalar model has no cross-lane reduction, so we use
                // THIS lane's value. SPECULATIVE(confidence: med): exact only when src0 is wave-uniform —
                // which is the standard use (reading a uniformly-computed VGPR into an SGPR, e.g. the
                // integer-divide reciprocal in the game's shaders). Writes an SGPR, not a VGPR.
                rs.sreg[in.dst.value] = a; return true;
            }
            uint32_t& d = vreg[in.dst.value];
            switch (in.opcode) {
                case 0x01: d = a; break;                              // v_mov_b32
                case 0x05: d = b.cvt_i2f(a); break;                   // v_cvt_f32_i32
                case 0x06: d = b.cvt_u2f(a); break;                   // v_cvt_f32_u32
                case 0x07: d = b.cvt_f2u(a); break;                   // v_cvt_u32_f32
                case 0x08: d = b.cvt_f2i(a); break;                   // v_cvt_i32_f32
                case 0x20: d = b.fext1(Glsl_Fract, a); break;         // v_fract_f32
                case 0x21: d = b.fext1(Glsl_Trunc, a); break;         // v_trunc_f32
                case 0x22: d = b.fext1(Glsl_Ceil, a); break;          // v_ceil_f32
                case 0x24: d = b.fext1(Glsl_Floor, a); break;         // v_floor_f32
                case 0x25: d = b.fext1(Glsl_Exp2, a); break;          // v_exp_f32 (2^x)
                case 0x27: d = b.fext1(Glsl_Log2, a); break;          // v_log_f32 (log2)
                case 0x2A: d = b.frcp(a); break;                      // v_rcp_f32
                case 0x2B: d = b.frcp(a); break;                      // v_rcp_iflag_f32 (~= v_rcp_f32)
                case 0x2E: d = b.fext1(Glsl_InverseSqrt, a); break;   // v_rsq_f32
                case 0x33: d = b.fext1(Glsl_Sqrt, a); break;          // v_sqrt_f32
                case 0x37: d = b.iun(Op_Not, a); break;               // v_not_b32
                case 0x38: d = b.iun(Op_BitReverse, a); break;        // v_bfrev_b32
                default: ok = false;
            }
            if (ok) predicate_write(b, rs, in.dst.value, old_d);
            return true;
        }
        case Rdna2Format::VOP2: {
            uint32_t a = val(in.src[0]), c = val(in.src[1]); uint32_t old_d = vreg_old(b, rs, in.dst.value);
            uint32_t& d = vreg[in.dst.value];
            switch (in.opcode) {
                case 0x01: d = b.sel(vcc, c, a); break;               // v_cndmask_b32: dst = vcc ? src1 : src0
                case 0x03: d = b.fbin(Op_FAdd, a, c); break;          // v_add_f32
                case 0x04: d = b.fbin(Op_FSub, a, c); break;          // v_sub_f32
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
                case 0x2F: d = b.pack_half2x16(a, c); break;          // v_cvt_pkrtz_f16_f32 (e32 form)
                default: ok = false;
            }
            if (ok) predicate_write(b, rs, in.dst.value, old_d);
            return true;
        }
        case Rdna2Format::VOPC: {                                     // v_cmp_* -> VCC; v_cmpx_* also -> EXEC
            uint32_t a = val(in.src[0]), c = val(in.src[1]);
            // v_cmpx_* shares each type's compare set at base+0x10 (f32 0x10-0x1f, i32 0x90-0x9f,
            // u32 0xd0-0xdf); it writes EXEC in addition to VCC. Map to the base compare, then narrow.
            uint32_t op = in.opcode;
            bool is_cmpx = (op >= 0x10 && op <= 0x1f) || (op >= 0x90 && op <= 0x9f) ||
                           (op >= 0xd0 && op <= 0xdf);
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
                default: ok = false;
            }
            if (ok) { vcc = cmp; if (is_cmpx) { rs.exec = b.land(rs.exec, cmp); rs.exec_narrowed = true; } }
            return true;
        }
        case Rdna2Format::VOP3: {
            uint32_t old_d = vreg_old(b, rs, in.dst.value);
            if (in.opcode == 0x14B || in.opcode == 0x141) {           // v_fma_f32 / v_mad_f32 = src0*src1 + src2
                // v_mad_f32 (op 0x141) is a gfx10.1 (Navi) instruction REMOVED in gfx10.3, so llvm-mc
                // -mcpu=gfx1030 rejects it as invalid — but the PS5 shader compiler targets gfx10.1 and
                // emits it (real game shaders 5,26-29: manual attribute interpolation p0+i*p1). Its result
                // (unfused mul-then-add) maps exactly to OpFMul+OpFAdd; v_fma's fused rounding is
                // immaterial here. VERIFIED(round-trip llvm-mc gfx1010, both directions): VOP3 op 0x141.
                uint32_t m = b.fbin(Op_FMul, val(in.src[0]), val(in.src[1]));
                vreg[in.dst.value] = b.fbin(Op_FAdd, m, val(in.src[2]));
            } else if (in.opcode == 0x169) {                          // v_mul_lo_u32
                vreg[in.dst.value] = b.ibin(Op_IMul, val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x16a) {                          // v_mul_hi_u32 (high 32 bits)
                vreg[in.dst.value] = b.umul_hi(val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x16c) {                          // v_mul_hi_i32 (high 32 bits, signed)
                vreg[in.dst.value] = b.smul_hi(val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x157) {                          // v_med3_f32 = median(s0,s1,s2)
                uint32_t s0 = val(in.src[0]), s1 = val(in.src[1]), s2 = val(in.src[2]);
                uint32_t mn = b.fext2(Glsl_FMin, s0, s1), mx = b.fext2(Glsl_FMax, s0, s1);
                vreg[in.dst.value] = b.fext2(Glsl_FMax, mn, b.fext2(Glsl_FMin, mx, s2));
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
            } else if (in.opcode == 0x12F) {                          // v_cvt_pkrtz_f16_f32 = pack(s0->lo, s1->hi)
                vreg[in.dst.value] = b.pack_half2x16(val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x107) {                          // v_mul_legacy_f32 ~= s0*s1
                vreg[in.dst.value] = b.fbin(Op_FMul, val(in.src[0]), val(in.src[1]));
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
                case 0x20:   // s_inst_prefetch  (I-cache hint)
                case 0x21:   // s_clause         (memory-clause scheduling hint)
                case 0x22:   // s_wait_idle
                case 0x7d:   // s_waitcnt_vscnt
                    break;
                case 0x08:                                          // s_cbranch_execz
                    if (in.simm16 < 0) ok = false;                 // backward = loop -> unsupported
                    else if (rs.exec_narrowed && (!safe_execz || !safe_execz->count(in.pc))) ok = false;
                    break;                                          // forward = no-op (predication covers it)
                case 0x0a:                                          // s_barrier
                    if (b.is_compute) b.barrier();                  // workgroup exec+memory barrier (LDS sync)
                    else ok = false;                                // barrier only meaningful in compute
                    break;
                case 0x04: case 0x05:                              // s_cbranch_scc0 / scc1
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
                default: ok = false; return true;    // register-offset / stores / others not yet
            }
            uint32_t base_idx = in.literal >> 2;    // immediate byte offset -> dword index
            // Descriptor provenance: pick which bound constant buffer via the resource table. For
            // s_buffer_load, SBASE (src[0]) is the V# — if a prior s_load_dwordx4 tagged it with the
            // SRT offset it came from, resolve that to a ShaderResource and route to its binding.
            uint32_t slot = 0;
            if (rt) { auto it = rs.sreg_srt.find(in.src[0].value);
                if (it != rs.sreg_srt.end()) { const ShaderResource* res = rt->by_srt_offset(it->second);
                    if (res && res->binding >= 3) slot = 1; } }
            for (uint32_t k = 0; k < n; k++)
                rs.sreg[in.dst.value + (int)k] = b.cbuf_load(b.uconst(base_idx + k), slot);
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
            uint32_t slot = 0, stride = 0;
            // Format of the fetched components. Untyped buffer_load_dword* is raw 32-bit (comp_bytes=4);
            // buffer_load_format_* takes the format from the resolved V# descriptor.
            DataFormat fmt = DataFormat::Uint32;   // untyped default: raw dwords
            if (is_format) {
                // A format load reads a vertex/buffer attribute — it needs the V# descriptor for the
                // binding, stride, and data format. Resolve SRSRC (src[1]) via provenance: an s_load
                // tag (indirect) else the SGPR index (direct/user-data).
                const ShaderResource* res = nullptr;
                if (rt) { auto it = rs.sreg_srt.find(in.src[1].value);
                    if (it != rs.sreg_srt.end()) res = rt->by_srt_offset(it->second);
                    if (!res) res = rt->by_sgpr_base(in.src[1].value); }
                if (!res) { ok = false; return true; }
                slot = (res->binding >= 3) ? 1 : 0;
                stride = res->stride;
                fmt = res->format;
            }
            const uint32_t comp_bytes = data_format_bytes(fmt);
            if (comp_bytes == 0) { ok = false; return true; }   // unknown / unsupported format
            // Per-component decode. 4-byte formats (Float32/Uint32/Sint32) are a raw dword load — no
            // conversion in our bit model. Sub-dword formats are unpacked: UNORM/SNORM normalize an
            // integer field, Float16 unpacks a packed half. num_components components pack tightly.
            const bool packed = comp_bytes < 4;
            bool is_snorm = (fmt == DataFormat::Snorm8 || fmt == DataFormat::Snorm16);
            bool is_half  = (fmt == DataFormat::Float16);
            float norm = 0.0f;
            switch (fmt) {
                case DataFormat::Unorm8:  norm = 255.0f;   break;
                case DataFormat::Snorm8:  norm = 127.0f;   break;
                case DataFormat::Unorm16: norm = 65535.0f; break;
                case DataFormat::Snorm16: norm = 32767.0f; break;
                default: break;
            }
            // Integer sub-dword formats (Uint8/Sint8/Uint16/Sint16) aren't normalized floats and would
            // need an integer-attribute path; reject rather than mis-normalize.
            if (packed && !is_half && norm == 0.0f) { ok = false; return true; }
            // Byte address of the element: idxen -> VADDR is an element index (×stride); offen -> VADDR
            // is a byte offset; plus the inst offset and SOFFSET. dword index = addr>>2.
            uint32_t addr = b.uconst(offset);
            if (idxen && stride) addr = b.ibin(Op_IAdd, addr, b.ibin(Op_IMul, val(in.src[0]), b.uconst(stride)));
            else if (offen)      addr = b.ibin(Op_IAdd, addr, val(in.src[0]));
            addr = b.ibin(Op_IAdd, addr, val(in.src[2]));              // SOFFSET
            uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
            if (is_store) {
                // Store the VDATA VGPRs (in.dst..+n-1). Integer sub-dword formats (Uint8/Sint8/...) have
                // no packing path here -> reject rather than mis-store (packed && norm==0 was caught above).
                auto vread = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
                if (!packed) {
                    // Raw/Float32/Uint32: one dword per component.
                    for (uint32_t k = 0; k < n; k++) {
                        uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                        b.cbuf_store(kidx, vread(in.dst.value + (int)k), slot, rs.exec_narrowed, rs.exec);
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
                        b.cbuf_store(did, acc, slot, rs.exec_narrowed, rs.exec);
                    }
                }
                return true;
            }
            for (uint32_t k = 0; k < n; k++) {
                int d = in.dst.value + (int)k;
                uint32_t old = vreg_old(b, rs, d);
                uint32_t value;
                if (!packed) {
                    uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                    value = b.cbuf_load(kidx, slot);                  // raw 32-bit component
                } else {
                    // Component k lives at byte k*comp_bytes within the element: pick its dword + field.
                    uint32_t byte_off = k * comp_bytes;
                    uint32_t drel = byte_off / 4, boff = (byte_off % 4) * 8;
                    uint32_t did = drel ? b.ibin(Op_IAdd, idx, b.uconst(drel)) : idx;
                    uint32_t dw  = b.cbuf_load(did, slot);
                    value = is_half ? b.unpack_half(dw, boff ? 1u : 0u)
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
            if (!res) { ok = false; return true; }

            // --- Storage-image path: image_load (0x00) / image_store (0x08), no sampler, any dim ---
            if (res->cls == ResourceClass::StorageImage) {
                const bool is_ld = (in.opcode == 0x00), is_st = (in.opcode == 0x08);
                if (!is_ld && !is_st) { ok = false; return true; }
                uint32_t dim, ncoord;
                switch (in.mimg_dim) {   // SQ_RSRC dim -> SPIR-V Dim + coord count
                    case 0: dim = Dim_1D; ncoord = 1; break;
                    case 1: dim = Dim_2D; ncoord = 2; break;
                    case 2: dim = Dim_3D; ncoord = 3; break;
                    default: ok = false; return true;   // cube/array/msaa storage images deferred
                }
                b.declare_storage_image(res->binding, dim);
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
                if (is_ld) {
                    uint32_t out[4]; b.image_read(res->binding, dim, coords, out);
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
                    b.image_write(res->binding, dim, coords, vals, rs.exec_narrowed, rs.exec);
                }
                return true;
            }

            // --- Sampled-texture path: image_sample* (0x20/0x24/0x27) / image_load (0x00), 2D, non-NSA ---
            // image_sample = 0x20 (implicit-LOD), image_sample_l = 0x24 (explicit LOD in 3rd coord),
            // image_sample_lz = 0x27 (LOD 0), image_load = 0x00 (integer texel fetch).
            const bool is_sample = (in.opcode == 0x20), is_load = (in.opcode == 0x00);
            const bool is_sample_l = (in.opcode == 0x24), is_sample_lz = (in.opcode == 0x27);
            if ((!is_sample && !is_load && !is_sample_l && !is_sample_lz) ||
                in.mimg_dim != SQ_DIM_2D || in.len_dwords != 2) { ok = false; return true; }
            if (res->cls != ResourceClass::Texture) { ok = false; return true; }
            b.declare_texture(res->binding);
            // coords: VGPR[VADDR], VGPR[VADDR+1] — normalized float for sample, integer texel for load.
            int va = in.src[0].value;
            uint32_t out[4];
            if (is_sample)         b.image_sample_2d(res->binding, vread(va), vread(va + 1), out);
            else if (is_sample_lz) b.image_sample_lod_2d(res->binding, vread(va), vread(va + 1), b.uconst(0), out);   // LOD 0
            else if (is_sample_l)  b.image_sample_lod_2d(res->binding, vread(va), vread(va + 1), vread(va + 2), out); // LOD = 3rd coord
            else                   b.image_fetch_2d (res->binding, vread(va), vread(va + 1), out);
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

std::vector<uint32_t> recompile_valu(const uint32_t* code, size_t dwords,
                                     uint32_t num_inputs, uint32_t out_vgpr,
                                     const ShaderResourceTable* rt) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);

    SpirvCompute b;
    b.begin(num_inputs ? num_inputs : 1);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t k = 0; k < num_inputs; k++) rs.vreg[(int)k] = b.load_input(k);

    for (const auto& in : ins) {
        if (in.is_end) break;
        bool ok = true;
        if (!emit_alu(b, rs, in, ok, true, &safe_branches, /*allow_smem*/true, rt) || !ok) return {};   // compute kernels are pure ALU
    }

    auto it = rs.vreg.find((int)out_vgpr);
    uint32_t outbits = it == rs.vreg.end() ? b.uconst(0) : it->second;
    // If EXEC is still narrowed (a v_cmpx with no restore), masked-off lanes keep the output slot's
    // prior value; if it was restored to all-lanes-on, every lane stores.
    if (!rs.exec_narrowed) b.store_output(outbits);
    else                   b.store_output_pred(outbits, rs.exec);
    return b.finish();
}

RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);

    // A scratch builder/state so emit_alu can run; its emitted code is discarded — we only want `ok`.
    SpirvCompute b; b.begin(1);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);

    RecompileCoverage cov;
    for (const auto& in : ins) {
        if (in.is_end) break;
        cov.total++;
        if (in.fmt == Rdna2Format::EXP) { cov.exports++; continue; }   // handled by the stage recompilers
        bool ok = true;
        bool handled = emit_alu(b, rs, in, ok, /*allow_exec_update*/true, &safe_branches, /*allow_smem*/true) && ok;
        // Shapes the recompiler handles only in context (a resource table for MIMG sample/load/LOD/store
        // and buffer_load/store_format; a fragment stage for VINTRP). This table-less compute-shell pass
        // rejects them, so count them apart from truly-unsupported (cross-lane, etc.). Instruction-aware
        // for MIMG so deferred variants (NSA multi-dword addr; arrayed/cube/MSAA dims) are NOT overcounted
        // as recompilable — they still land in `unsupported`, matching what the recompiler actually accepts.
        auto table_dependent = [](const Rdna2Inst& i) {
            switch (i.fmt) {
                case Rdna2Format::MIMG: {
                    const bool dim_ok = i.mimg_dim <= 2u;      // 1D/2D/3D only (not 1D/2D_ARRAY, CUBE, MSAA)
                    // load/store go through the storage path, which handles NSA (split address dwords);
                    // sample* go through the sampled-texture path, still non-NSA only (len==2).
                    if (i.opcode == 0x00u || i.opcode == 0x08u) return dim_ok;
                    if (i.opcode == 0x20u || i.opcode == 0x24u || i.opcode == 0x27u) return dim_ok && i.len_dwords == 2;
                    return false;
                }
                case Rdna2Format::MUBUF:  return i.opcode <= 0x07u;   // buffer_load/store_format_* (need the V#)
                case Rdna2Format::VINTRP: return true;               // handled in the fragment shell
                default: return false;
            }
        };
        if (handled) { cov.alu++; }
        else if (table_dependent(in)) { cov.table_dependent++; }
        else {
            cov.unsupported++;
            if (cov.first_bad_fmt < 0) { cov.first_bad_fmt = (int)in.fmt; cov.first_bad_op = in.opcode; }
        }
    }
    return cov;
}

std::vector<uint32_t> recompile_fragment(const uint32_t* code, size_t dwords, const ShaderResourceTable* rt) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);

    SpirvCompute b;
    b.begin_fragment(rt != nullptr);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    bool exported = false;

    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::EXP) {                    // EXP MRT0..7 -> the color output
            if (in.exp_target <= 7 && !exported) {
                b.export_color(operand_bits(b, rs, in, in.src[0]), operand_bits(b, rs, in, in.src[1]),
                               operand_bits(b, rs, in, in.src[2]), operand_bits(b, rs, in, in.src[3]));
                exported = true;
            }
            continue;   // ignore NULL / additional exports for now
        }
        bool ok = true;
        // Graphics-stage exports do not yet model EXEC-masked export/discard, so reject cmpx for now
        // instead of accepting a shader that would export from inactive lanes. Memory ops (SMEM/MUBUF)
        // are allowed only when a resource table is supplied (so their bindings resolve).
        if (!emit_alu(b, rs, in, ok, false, &safe_branches, /*allow_smem*/rt != nullptr, rt) || !ok) return {};
    }
    if (!exported) return {};
    return b.finish();
}

std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords, const ShaderResourceTable* rt) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);

    SpirvCompute b;
    b.begin_vertex(rt != nullptr);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    rs.vreg[0] = b.load_vertex_index();     // VS ABI: v0 = vertex index
    bool exported = false;

    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::EXP) {                    // EXP POS0..3 -> gl_Position; PARAM -> varyings
            if (in.exp_target >= 12 && in.exp_target <= 15 && !exported) {
                b.export_position(operand_bits(b, rs, in, in.src[0]), operand_bits(b, rs, in, in.src[1]),
                                  operand_bits(b, rs, in, in.src[2]), operand_bits(b, rs, in, in.src[3]));
                exported = true;
            } else if (in.exp_target >= 32) {                // PARAM0.. -> Output varying (location N)
                b.export_param(in.exp_target - 32, operand_bits(b, rs, in, in.src[0]), operand_bits(b, rs, in, in.src[1]),
                               operand_bits(b, rs, in, in.src[2]), operand_bits(b, rs, in, in.src[3]));
            }
            continue;
        }
        bool ok = true;
        // cmpx rejected in graphics stages (no EXEC-masked export yet); SMEM/MUBUF allowed only with a
        // resource table so vertex-fetch / constant bindings resolve.
        if (!emit_alu(b, rs, in, ok, false, &safe_branches, /*allow_smem*/rt != nullptr, rt) || !ok) return {};
    }
    if (!exported) return {};
    return b.finish();
}

} // namespace prosper::gpu
