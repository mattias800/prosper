#pragma once

// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that
// operate on them can live in their own translation units. These are INTERNAL to the
// recompiler: nothing outside src/gpu/recompiler/ should include this header.

// rdna2_to_spirv_internal.hpp — SpirvCompute and the recompiler's shared internals. Lifted out
// of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions can live in their own
// translation units. INTERNAL to src/gpu/recompiler/; nothing outside it should include this.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

inline const char* recompile_diagnostic_stage_name(RecompileDiagnosticStage stage) {
    switch (stage) {
        case RecompileDiagnosticStage::Compute: return "compute";
        case RecompileDiagnosticStage::Vertex: return "vertex";
        case RecompileDiagnosticStage::Fragment: return "fragment";
        case RecompileDiagnosticStage::Standalone: return "standalone";
    }
    return "standalone";
}

// Recompiles can run concurrently. Build the complete diagnostic on the stack and submit it through
// one stdio call so one program's identity cannot be spliced onto another program's reject payload.
// The context is an explicit per-call value; no global or thread-local attribution state is involved.
// Last TERMINAL reject reason per program, recorded whether or not PROSPER_DBG is set.
//
// Every reject reason in this file sits behind PROSPER_DBG, and PROSPER_DBG is unusable on a routed
// run: it produces a log large enough to desync a timing-dependent pad script, so the route never
// reaches the phase whose rejects you wanted to read. The consequence is that
// `[compute] skip unsupported program 0x...` has printed a bare address for the whole life of the
// diagnostic, and the charter's rule that every skip is the next thing to implement cannot be
// followed from it.
//
// Recording only the terminal role keeps this to one short string per rejected program (thirteen on
// a 200 s GTA V route), so the map is bounded by the number of DISTINCT rejected programs rather
// than by the number of dispatches.
inline std::mutex& terminal_reject_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::map<uint64_t, std::string>& terminal_reject_reasons() {
    static std::map<uint64_t, std::string> reasons;
    return reasons;
}

inline void record_terminal_reject_reason(uint64_t program_address, const char* tag,
                                   const char* payload) {
    if (!program_address) return;
    // A cap, because a pathological guest could present unboundedly many distinct programs. Losing
    // late entries is strictly better than a diagnostic that grows without limit.
    constexpr size_t kMaximumRecordedPrograms = 4096;
    std::lock_guard lock(terminal_reject_mutex());
    auto& reasons = terminal_reject_reasons();
    if (reasons.size() >= kMaximumRecordedPrograms && !reasons.count(program_address)) return;
    reasons[program_address] = std::string(tag) + " " + payload;
}

// Is the VERBOSE recompiler diagnostic stream enabled for this guest program?
//
// `PROSPER_DBG` turns it on for every program, which is the right default for a focused offline
// repro and the wrong one for a routed title: the stream is ~1.5 GB on a Sonic Frontiers arm and
// slow enough to desync the pad script that reaches the phase being diagnosed. So the one question
// an investigation actually has -- "which route declined THIS program, on the run that reaches it"
// -- was the question the instrument could not be pointed at, and offline reproduction is not a
// substitute: `shader_inspect` supplies a default `ComputeShaderConfig` and no resource table, so
// the MUST dataflow starts from an empty scalar-word set and can decline at a site the live
// translation never reaches.
//
// `PROSPER_DBG_PROGRAM=0x2005717e00[,0x...]` enables the same lines for the listed program
// addresses only. Parsed once into a function-local static: this predicate sits on the reject path
// of every recompile, and re-reading the environment per call is the cost #2214 removed elsewhere.
// (The `PROSPER_DBG` latch is therefore also read once, where the fifteen remaining per-site gates
// in `rdna2_emit_cfg.cpp` still call `getenv` on every evaluation. Only a mid-process `setenv` could
// tell the difference, and nothing in prosper does that.)
//
// Each comma-separated element goes through `parse_diagnostic_uint64` -- the same strict parser
// every other PROSPER_* selector uses -- rather than a local `strtoull`. A hand-rolled loop accepts
// a malformed separator by stopping at it, so `PROSPER_DBG_PROGRAM=0xA;0xB` would silently select
// only the first address and the run would look like the second program simply never compiled.
// A rejected element is announced and the whole list refused, because a diagnostic that quietly
// narrows its own selector is the class of instrument this file exists to avoid (review of #2820).
inline bool recompile_diagnostic_verbose(uint64_t program_address) {
    static const bool all = std::getenv("PROSPER_DBG") != nullptr;
    if (all) return true;
    static const std::vector<uint64_t> selected = [] {
        std::vector<uint64_t> out;
        const char* list = std::getenv("PROSPER_DBG_PROGRAM");
        if (!list) return out;
        const std::string_view text(list);
        size_t start = 0;
        while (start <= text.size()) {
            size_t comma = text.find(',', start);
            if (comma == std::string_view::npos) comma = text.size();
            std::string_view element = text.substr(start, comma - start);
            while (!element.empty() && element.front() == ' ') element.remove_prefix(1);
            while (!element.empty() && element.back() == ' ') element.remove_suffix(1);
            uint64_t value = 0;
            if (!parse_diagnostic_uint64(element, value)) {
                std::fprintf(stderr,
                             "[recompile-diag] PROSPER_DBG_PROGRAM element '%.*s' is not a program "
                             "address -- the whole selector is ignored rather than silently "
                             "truncated\n",
                             static_cast<int>(element.size()), element.data());
                return std::vector<uint64_t>{};
            }
            out.push_back(value);
            start = comma + 1;
        }
        return out;
    }();
    if (selected.empty() || !program_address) return false;
    return std::find(selected.begin(), selected.end(), program_address) != selected.end();
}

inline void log_recompile_diagnostic(const RecompileDiagnosticContext& diagnostic,
                              const char* tag, const char* role, const char* format, ...) {
    // Formatted BEFORE the PROSPER_DBG gate: the terminal reason has to be recorded whether or not
    // anyone is reading the verbose stream.
    char payload[1536];
    va_list args;
    va_start(args, format);
    const int body = std::vsnprintf(payload, sizeof payload, format, args);
    va_end(args);
    if (body < 0) return;
    payload[sizeof(payload) - 1] = '\0';
    // Record every role EXCEPT "consequent". A consequent line only restates that an empty result
    // was returned -- it names the effect, never the cause -- so letting it overwrite a terminal or
    // route-decline reason would replace the answer with the question. Measured on a routed GTA V
    // run: recording "terminal" alone left twelve of thirteen skips reading `reason=unrecorded`,
    // because most reject sites use "route-decline".
    if (role && std::strcmp(role, "consequent") != 0)
        record_terminal_reject_reason(diagnostic.program_address, tag, payload);

    if (!recompile_diagnostic_verbose(diagnostic.program_address)) return;
    size_t payload_size = std::strlen(payload);
    while (payload_size && (payload[payload_size - 1] == '\n' ||
                            payload[payload_size - 1] == '\r'))
        payload[--payload_size] = '\0';

    // Preserve the established `[tag] payload` prefix consumed by issue-census scripts. Provenance
    // is an appended field group, not an insertion between the tag and its historical first field.
    char line[2048];
    const int written = diagnostic.program_address
        ? std::snprintf(line, sizeof line,
                        "[%s] %s stage=%s program=0x%llx role=%s\n",
                        tag, payload, recompile_diagnostic_stage_name(diagnostic.stage),
                        static_cast<unsigned long long>(diagnostic.program_address), role)
        : std::snprintf(line, sizeof line,
                        "[%s] %s stage=%s program=none role=%s\n",
                        tag, payload, recompile_diagnostic_stage_name(diagnostic.stage), role);
    if (written < 0) return;
    const size_t used = std::min(static_cast<size_t>(written), sizeof(line) - 1);
    if (used == sizeof(line) - 1) {
        line[sizeof(line) - 2] = '\n';
    }
    std::fwrite(line, 1, used, stderr);
}
enum : uint32_t {
    Op_Extension=10, Op_ExtInstImport=11, Op_ExtInst=12, Op_MemoryModel=14, Op_EntryPoint=15, Op_ExecutionMode=16,
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
    Op_ImageQueryLod=105, Op_ImageQueryLevels=106,
    Op_TypeArray=28, Op_ControlBarrier=224, Op_MemoryBarrier=225, Op_AtomicLoad=227,
    Op_AtomicExchange=229, Op_AtomicCompareExchange=230, Op_AtomicIAdd=234,
    Op_AtomicISub=235,
    Op_AtomicSMin=236, Op_AtomicUMin=237, Op_AtomicSMax=238, Op_AtomicUMax=239,
    Op_AtomicAnd=240, Op_AtomicOr=241, Op_AtomicXor=242,
    Op_DPdx=207, Op_DPdy=208,   // screen-space derivatives (Fragment; plain Shader capability)
    Op_Phi=245, Op_LoopMerge=246,
    Op_SelectionMerge=247, Op_Label=248, Op_Branch=249, Op_BranchConditional=250, Op_Switch=251,
    Op_EmitVertex=218, Op_EndPrimitive=219,
    Op_Kill=252, Op_Return=253, Op_ModuleProcessed=330, Op_GroupNonUniformAny=335,
    Op_GroupNonUniformBallot=339, Op_GroupNonUniformShuffle=345,
    Op_GroupNonUniformIAdd=349,
};
// GLSL.std.450 extended-instruction numbers.
enum : uint32_t { Glsl_FAbs=4, Glsl_RoundEven=2, Glsl_Trunc=3, Glsl_Floor=8, Glsl_Ceil=9, Glsl_Fract=10, Glsl_Sin=13, Glsl_Cos=14,
                  Glsl_Exp2=29, Glsl_Log2=30,
                  Glsl_Sqrt=31, Glsl_InverseSqrt=32, Glsl_FMin=37, Glsl_UMin=38, Glsl_SMin=39, Glsl_FMax=40,
                  Glsl_UMax=41, Glsl_SMax=42, Glsl_PackHalf2x16=58, Glsl_UnpackHalf2x16=62,
                  Glsl_FindUMsb=75,   // bit index of the highest set bit (undefined at zero)
                  Glsl_NMin=79, Glsl_NMax=80 };   // NaN-aware min/max: one-NaN operand -> the other operand
enum : uint32_t {
    Cap_Shader=1, Cap_Geometry=2, Cap_Int64=11, Cap_Int64Atomics=12,
    Cap_GroupNonUniform=61, Cap_GroupNonUniformVote=62,
    Cap_GroupNonUniformArithmetic=63, Cap_GroupNonUniformBallot=64, Cap_GroupNonUniformShuffle=65, Cap_GroupNonUniformQuad=68,
    Cap_TransformFeedback=53,   // VK_EXT_transform_feedback (geometry-probe capture of gl_Position, gated)
    // Descriptor indexing (#2412 stage 4b). These are CORE only from SPIR-V 1.5; this emitter writes
    // 1.3, so they additionally require OpExtension "SPV_EXT_descriptor_indexing" -- emitted together
    // by declare_descriptor_indexing() and only when an indexed binding actually exists.
    //
    // Cap_ShaderNonUniform is NOT one of the Cap_GroupNonUniform* values above. Those are wave/subgroup
    // operations that merely share the word; `grep -c NonUniform` on this file returns 45 hits and every
    // one of them is a GroupNonUniform op, which reads exactly like descriptor indexing is already
    // supported. It was not.
    Cap_ShaderNonUniform=5301,
    Cap_StorageBufferArrayNonUniformIndexing=5308,
    Addr_Logical=0, Mem_GLSL450=1, Exec_Vertex=0, Exec_Geometry=3, Exec_Fragment=4, Exec_GLCompute=5,
    EM_OriginUpperLeft=7, EM_DepthReplacing=12, EM_LocalSize=17, EM_Triangles=22,
    EM_OutputVertices=26, EM_OutputTriangleStrip=29, EM_Xfb=11,   // transform-feedback execution mode
    SC_Input=1, SC_UniformConstant=0, SC_Output=3, SC_Function=7, SC_PushConstant=9,
    SC_Image=11, SC_StorageBuffer=12, FC_None=0,
    Dim_1D=0, Dim_2D=1, Dim_3D=2,   // SPIR-V Dim. (2D coincides with the SQ_RSRC 2D dim value, but distinct.)
    Cap_Sampled1D=43, Cap_Image1D=44,   // Dim=1D needs Sampled1D; a 1D STORAGE image (read/write) also needs Image1D
    Cap_StorageImageMultisample=27,      // MS=1 storage image (read/write a multisampled image)
    Cap_ImageMSArray=48,                 // MS=1 AND Arrayed=1 image (2D_MSAA_ARRAY)
    Cap_StorageImageExtendedFormats=49,  // typed formats outside the core R32/RGBA32 set
    Cap_StorageImageReadWithoutFormat=55, Cap_StorageImageWriteWithoutFormat=56,  // for Format=Unknown storage images
    Cap_ImageGatherExtended=25,          // dynamic (non-const) Offset image operand on OpImageGather
    Cap_ImageQuery=50,                   // OpImageQuerySizeLod (the sample_*_o texel->UV offset fold)
    ImgOp_Offset=0x10,                   // ImageOperands bit: dynamic texel offset (needs ImageGatherExtended)
    Img_Sampled_Storage=2,   // OpTypeImage "Sampled" operand: 2 = used WITHOUT a sampler (read/write storage image)
    ImgFmt_Unknown=0,        // OpTypeImage "Image Format": Unknown (runtime view format; needs the caps above)
    ImgFmt_R32ui=kSpirvImageFormatR32ui, // exact uint32 storage image format required by image atomics
    ImgFmt_Rgba8ui=kSpirvImageFormatRgba8ui, // exact four-byte RGBA unsigned storage image
    ImgFmt_R16ui=kSpirvImageFormatR16ui, // exact halfword-width unsigned storage image
    ImgFmt_R8ui=kSpirvImageFormatR8ui,   // exact byte-width unsigned storage image
    ImgOp_Bias=1, ImgOp_Lod=2, ImgOp_Grad=4, ImgOp_Sample=0x40,   // ImageOperands bits.
    SC_Workgroup=4, Scope_Device=1, Scope_Workgroup=2, Scope_Subgroup=3,
    MemSem_UniformAcquire=0x42, MemSem_UniformRelease=0x44,
    MemSem_UniformAcqRel=0x48, MemSem_WGAcquire=0x102,   // memory class + ordering
    MemSem_WGAcqRel=0x108,                              // storage-buffer/LDS AcquireRelease semantics
    MemSem_ImageAcqRel=0x808,                            // AcquireRelease | ImageMemory
    MemAccess_Volatile=0x1,
    Dec_Block=2, Dec_ArrayStride=6, Dec_BuiltIn=11, Dec_NoPerspective=13, Dec_Flat=14,
    Dec_Centroid=16, Dec_Sample=17, Dec_Aliased=20, Dec_Coherent=23,
    Dec_Location=30, Dec_Binding=33,
    Dec_DescriptorSet=34, Dec_Offset=35, Dec_XfbBuffer=36, Dec_XfbStride=37,
    // Decoration 5300 -- descriptor indexing's NonUniform, unrelated to the GroupNonUniform ops.
    Dec_NonUniform=5300,
    BI_Position=0, BI_FragCoord=15, BI_SampleMask=20, BI_FragDepth=22, BI_HelperInvocation=23,
    BI_WorkgroupId=26, BI_LocalInvocationId=27,
    BI_GlobalInvocationId=28, BI_SubgroupId=40, BI_SubgroupLocalInvocationId=41,
    BI_VertexIndex=42, BI_InstanceIndex=43,
    GroupOp_Reduce=0, GroupOp_ExclusiveScan=2,
};

inline uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct FlatAccessInfo {
    bool valid = false;
    bool store = false;
    bool sign_extend = false;
    uint32_t bits = 0;
    uint32_t components = 0;
    uint32_t bytes() const { return (bits / 8u) * components; }
};

inline FlatAccessInfo flat_access_info(uint32_t opcode) {
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

inline StaticScratchLayout analyze_static_scratch(const std::vector<Rdna2Inst>& ins) {
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

// A compute-shader SPIR-V builder specialized for "load N floats -> compute over SSA floats ->
// store 1 float", with helpers the VALU translator drives.
struct SpirvCompute {
    // `exts` holds OpExtension. SPIR-V requires it AFTER every OpCapability and BEFORE
    // OpExtInstImport, which is why it is a separate section rather than appended to `caps` or
    // prepended to `extimp` -- see the module assembly order below. Getting that order wrong fails
    // spirv-val with a LAYOUT complaint that says nothing about descriptors.
    std::vector<uint32_t> caps, exts, extimp, mem, entry, exec, debug, deco, types, code;
    RecompileDiagnosticContext diagnostic{};
    // Ordinal of the next CFG dispatcher emitted for THIS module. A barrier-phased compute program
    // emits one dispatcher per barrier-free phase, and they are not interchangeable — each covers a
    // different guest pc range. Numbering them lets a diagnostic name which one it means.
    uint32_t cfg_phase_ordinal = 0;
    bool descriptor_indexing_declared = false;
    // Declare the capability set and extension for an indexed descriptor array, once per module and
    // only when one is actually declared, so every module that does not use one is byte-identical to
    // what this emitter produced before stage 4b.
    void declare_descriptor_indexing() {
        if (descriptor_indexing_declared) return;
        descriptor_indexing_declared = true;
        put(caps, Op_Capability, {Cap_ShaderNonUniform});
        put(caps, Op_Capability, {Cap_StorageBufferArrayNonUniformIndexing});
        std::vector<uint32_t> o; pstr(o, "SPV_EXT_descriptor_indexing");
        putv(exts, Op_Extension, o);
    }
    std::unordered_map<uint32_t, uint32_t> fconst_cache, uconst_cache;
    uint32_t next_id = 1;
    uint32_t stride = 1;
    // fixed ids (set in begin()):
    uint32_t t_void=0, t_fn=0, t_f32=0, t_u32=0, t_i32=0, t_v3u=0, t_bool=0, t_ptr_sb_f32=0;
    uint32_t v_gid=0, v_groupid=0, v_in=0, v_out=0, gidx=0, f_main=0, glsl=0, bconst_false=0;
    uint32_t globalid_comp[3] = {0, 0, 0}, groupid[3] = {0, 0, 0}, localid_comp[3] = {0, 0, 0};
    uint32_t invocation_guard_merge = 0;
    // Set only by the barrier-phase emitter when a partial workgroup remains in uniform control
    // flow and uses the dispatcher's separate ACTIVE bit to suppress padded guest invocations.
    // recompile_compute uses this proof bit to distinguish that route from the ordinary divergent
    // entry guard, which is never legal around a workgroup barrier.
    bool partial_barrier_phases_emitted = false;
    uint32_t v_push_constants = 0, t_ptr_push_u32 = 0;
    uint32_t push_constant_dword_count = 0;
    uint32_t linear_localid = 0, local_count = 64, wave_size = 64;
    uint32_t v_cbuf=0, v_cbuf1=0, t_ptr_sb_u32=0;   // scalar-memory constant buffers (bindings 2 and 3)
    uint32_t t_ptr_sb_struct_u=0;                    // shared runtime-u32 Block pointer type
    uint32_t t_ptr_sb_u64=0, t_ptr_sb_struct_u64=0; // lazy alias view for 64-bit buffer atomics
    uint32_t t_ptr_img_u32=0;                       // OpImageTexelPointer result for R32_UINT atomics
    uint32_t guest_scratch=0, t_ptr_guest_scratch_u32=0;
    int32_t guest_scratch_min_byte=0, guest_scratch_saddr=-1;
    uint32_t guest_scratch_dwords=0;
    std::map<uint32_t, uint32_t> cbuf_var;          // binding -> storage-buffer var (N-buffer model; 2/3 map to v_cbuf/v_cbuf1)
    std::map<uint32_t, uint32_t> cbuf_u64_var;      // binding -> runtime-u64 alias variable
    // binding -> declared array length for a TABLE-INDEXED binding (#2412). Absent means an ordinary
    // single descriptor, so an access chain for it takes no leading index. Kept beside `cbuf_var`
    // because every consumer of the variable also needs to know whether it is an array.
    std::map<uint32_t, uint32_t> cbuf_table_arity;
    // #2481: a DynamicSbufferByteOffset table's element index is not in user data — it is whatever
    // the producing `s_buffer_load_dwordx4` had in its SOFFSET at run time. The emitter stores that
    // already-divided index here when it lowers the producer, so `cbuf_element_ptr` can select with
    // a live SSA value instead of a push constant. Absent means no producer has executed on this
    // path yet, which must keep failing visibly rather than silently selecting element zero.
    std::map<uint32_t, uint32_t> cbuf_table_selector_value;
    // Label id of the block that emitted each selector above. A raw SSA id is only usable where the
    // producer DOMINATES the consumer, and nothing here guarantees that: the CFG dispatcher emits
    // one switch case per guest block, and sibling switch cases do not dominate one another (a
    // structured if/endif has the same problem). Consuming across that boundary would emit an
    // OpAccessChain against an id outside its dominance region -- malformed SPIR-V that `spirv-val`
    // does not catch here for want of a representative module, surfacing instead as a dropped
    // dispatch. Requiring producer and consumer to be the SAME block is the conservative subset
    // that is always sound; carrying the value through a Function variable would lift the
    // restriction and is the right long-term shape.
    std::map<uint32_t, uint32_t> cbuf_table_selector_block;
    // binding -> user SGPR holding the descriptor index, when one is available.
    std::map<uint32_t, uint32_t> cbuf_table_index_sgpr;
    // Set when an array access cannot be represented exactly. The backend already refuses writable
    // arrays; keeping the same boundary in the emitter prevents a caller that bypasses reflection
    // from receiving a module that redirects an invalid store/atomic to descriptor zero.
    bool invalid_cbuf_array_access = false;
    // A two-byte guest V# can only back our u32 SSBO ABI when every use of that binding is the exact
    // one-record Uint16/Float16 path. Ordinary load/store/atomic helpers blacklist their bindings;
    // finish() emits the explicit typed zero-pad contract only for candidates with no competing use.
    std::map<uint32_t, StorageBufferTailSemantic> cbuf_zero_pad_candidates;
    std::set<uint32_t> cbuf_ordinary_accesses;
    std::set<uint32_t> cbuf_coherent_vars;
    bool     is_fragment=0;                          // true in the fragment shell (gates VINTRP interp)
    bool     is_vertex=0;                            // true in every vertex shell
    // Why an instruction the emitter DECODED could not be lowered in THIS stage. `mode` on the
    // reject line separates "no lowering exists" (`unknown-encoding`) from "the lowering exists and
    // an operand/descriptor did not resolve" (`unresolved-operand`), and a cross-lane operation in a
    // stage with no guest wave is neither: the encoding is known, every operand resolves, and the
    // stage simply has no lane set to reduce over. Reported as a third mode so a census does not
    // read it as a descriptor defect and go looking for the resource table (#2412, #3135). Set by
    // the emitter at the deciding instruction; consumed by the reject site, which appends it to the
    // one terminal line so the recorded reason names the cause rather than the effect.
    uint32_t stage_reject_pc = UINT32_MAX;
    std::string stage_reject_reason;
    // The first V_MBCNT whose src0 is a general SGPR mask, when the program ALSO builds the
    // canonical all-ones lane-index pair. That mixture disqualifies the flattened-lane vertex model
    // for the whole program, so the reject surfaces at the all-ones instruction -- which is
    // lowerable on its own -- and names a PC that is not the cause. Kept so the reject can say which
    // instruction actually disqualified it (#3135).
    uint32_t vertex_general_mask_mbcnt_pc = UINT32_MAX;
    bool     allow_b32_masks=0;                      // proven Wave32 or byte-exact graphics exception
    bool     ngg_one_lane=0;                         // exact GS_ALLOC_REQ wrapper: one guest lane/invocation
    bool     ngg_logical_lane=0;                     // proven wave64 no-GS producer uses flattened guest lane
    bool     ngg_private_lds=0;                      // exact captured wrapper whose LDS projection is known
    uint32_t ngg_vertex_index_read_pc = UINT32_MAX;   // NGG wave/LDS prologue handoff -> host VertexIndex
    uint32_t ngg_vertex_index_value = 0;
    bool     is_compute=0;                            // true in the compute shell (gates LDS / s_barrier)
    bool     uses_barrier=0;                          // guest or synthesized workgroup barrier emitted
    // Ordinary LDS writes that feed a synthesized float-atomic publication boundary are emitted as
    // atomic exchanges. RDNA serializes indexed same-bank conflicts, while Vulkan ordinary stores
    // from separate invocations would otherwise form a data race before the common-phase barrier.
    // The whole-stream synchronization proof populates exact guest PCs; unrelated LDS stores retain
    // their ordinary lowering.
    std::unordered_set<uint32_t> atomicized_lds_store_pcs;
    // GTA V 0x413ce6000's exact pc153 scalar descriptor-table read is admitted only after the
    // complete dispatch/source/table proof has been repeated at the final compiler boundary.
    // Keeping that authority on the builder, rather than inferring it from a resource field inside
    // emit_alu, prevents a hand-built or stale marker from enabling instruction-local shortcuts.
    bool gta5_selected_sbuffer_dispatch_validated = false;
    uint32_t gta5_selected_sbuffer_soffset = UINT32_MAX;
    bool gta5_cf9200_no_backing_dispatch_validated = false;
    bool indirect_buffer_dispatch_validated = false;
    uint32_t indirect_buffer_binding = UINT32_MAX;
    uint32_t indirect_buffer_source_bytes = 0;
    uint32_t indirect_buffer_slot_count = 0;
    uint32_t indirect_buffer_contract_tag = 0;
    uint32_t indirect_buffer_header_bytes = 0;
    uint32_t indirect_buffer_slot_bytes = 0;
    uint32_t indirect_buffer_atomic_binding = UINT32_MAX;
    uint32_t indirect_buffer_atomic_byte_offset = 0;
    // Dispatch/compiler-boundary authority for generic pointer relocation. The proof identifies
    // exact GLOBAL consumers; the remaining fields describe the already-validated carrier bound at
    // `indirect_pointer_binding`. Version 2's StaticFootprint path needs only the segment directory.
    // Version 3's DescriptorRange path additionally preserves the dynamic source-record identity
    // and Base48 root in per-invocation Function variables across CFG dispatcher cases.
    // Every field deliberately defaults to no authority so a caller which has not repeated the
    // complete proof cannot activate either lowering.
    const IndirectPointerRelocationProof* indirect_pointer_proof = nullptr;
    uint32_t indirect_pointer_binding = UINT32_MAX;
    uint32_t indirect_pointer_source_bytes = 0;
    uint32_t indirect_pointer_record_count = 0;
    uint32_t indirect_pointer_record_directory_byte_offset = 0;
    uint32_t indirect_pointer_segment_count = 0;
    uint32_t indirect_pointer_segment_directory_byte_offset = 0;
    uint32_t indirect_pointer_payload_byte_offset = 0;
    uint32_t indirect_pointer_carrier_bytes = 0;
    uint32_t indirect_pointer_source_stride = 0;
    uint32_t indirect_pointer_source_pointer_byte_offset = 0;
    uint32_t indirect_pointer_source_record_var = 0;
    uint32_t indirect_pointer_source_root_lo_var = 0;
    uint32_t indirect_pointer_source_root_hi_var = 0;
    // S_CSELECT_B64 normally needs both scalar source dwords. A captured GTA V kernel consumes
    // only the selected VCC_LO word and leaves VCC_HI dead on every successor path; the whole-stream
    // liveness proof records that exact exception before emission. Keep it builder-local so recursive
    // phase/CFG emitters see the proof derived from the original complete instruction stream.
    bool cselect_b64_low_only_analysis_done = false;
    std::unordered_set<uint32_t> cselect_b64_low_only_pcs;
    // GTA V also selects one B32 scalar directly into Wave64 VCC_LO and consumes only that low word.
    // The untouched VCC_HI mask is unrepresentable once the pair becomes a mixed data/mask lifetime;
    // retain only sites whose whole-stream proof shows that high half dead before any mask read.
    bool vcc_b32_low_only_analysis_done = false;
    std::unordered_set<uint32_t> vcc_b32_low_only_pcs;
    // A Wave64 B32 scalar write to one physical VCC word can also retain a complete architectural
    // predicate when the dispatcher MUST analysis proves the untouched sibling is scalar data at
    // that exact PC.  The proof is separate from Function-variable presence: dispatcher variables
    // have zero placeholders on paths where no scalar definition reaches the block.
    std::unordered_set<uint32_t> vcc_b32_scalar_pair_pcs;
    // The same dispatcher analysis separately proves that both encoded inputs to a B32 VCC write
    // are scalar words on every path reaching that exact PC. Dispatcher Function variables exist
    // for mask-only lifetimes too, so map membership alone must never select the scalar lowering.
    std::unordered_set<uint32_t> vcc_b32_scalar_result_pcs;
    // Exact structured-CFG consumers whose source pair is a saved Wave64 VCC mask on every
    // incoming path. S_MOV_B64 also materializes ballot words, so emit_alu needs this separate
    // lifetime fact to select the Bool-domain value at the consumer without guessing from maps.
    bool structured_wave64_mask_reduction_analysis_done = false;
    std::unordered_set<uint32_t> structured_wave64_mask_reduction_pcs;
    // Exact Wave64 mask spills retain a Bool view in vgpr_lane_mask_slots. When a CFG MUST proof
    // also identifies which physical ballot half the slot represents, V_READLANE can publish the
    // corresponding scalar dword without guessing from the SGPR number or spill lane.
    std::unordered_map<uint32_t, uint32_t> wave64_mask_readlane_half_for_pc;
    // A mask half restored to an ordinary SGPR may be written back to a spill lane after crossing
    // a dispatcher edge. Keep the exact proven V_WRITELANE sites separate from the SGPR number:
    // physical SGPRs are routinely recycled as scalar data elsewhere in the same shader.
    std::unordered_set<uint32_t> wave64_mask_writelane_alias_pcs;
    bool     declared_subgroup=0, declared_subgroup_vote=0, declared_subgroup_arithmetic=0;
    // When non-zero, the backend promises to create this compute pipeline with an exact required
    // subgroup size equal to the PS5 wave. Native votes/scans are then architecture-exact.
    uint32_t native_subgroup_size=0;
    uint32_t native_storage_format_support=0;
    bool storage_buffer_int64_atomics=false;
    bool packed_r11_storage=true;
    uint32_t compute_pgm_rsrc1=kDefaultComputePgmRsrc1;
    uint32_t compute_min_subgroup_size=0;             // non-semantic backend contract (4/16/32/64)
    uint32_t fragment_required_subgroup_size=0;       // exact guest-wave contract (32 or 64)
    // WHY that width was required, as a bitmask (#2147). The size alone is not actionable: a
    // shader needing 64 for lane IDENTITY can never run at 32, while one needing it only for a
    // width-agnostic branch vote might. The skip diagnostic reported neither, because its
    // `required-ops` field scans for Vote/Arithmetic/Shuffle CAPABILITIES and the lane-id path
    // declares none of them -- so the two cases printed identically.
    uint32_t fragment_wave_reasons=0;
    // SSA provenance for fragment WaveAny results. A vote needs the exact guest-wave width when
    // that particular bool reaches a guest scalar-data consumer. Tracking result ids avoids the
    // false whole-module inference "this shader contains both a vote and S_CSELECT".
    std::unordered_set<uint32_t> fragment_wave_vote_values;
    std::unordered_set<uint32_t> fragment_wave_vote_scalar_consumers;
    std::unordered_map<uint32_t, std::vector<uint32_t>> fragment_wave_vote_dependents;
    std::unordered_map<size_t, uint32_t> fragment_wave_vote_phi_patch_results;
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
    // GLSL.std.450 FindUMsb: bit index of the most significant 1. glslang emits this with a
    // SIGNED result and an unsigned operand, which is the form drivers are exercised on, so
    // mirror it exactly and bitcast back. The result is UNDEFINED when the operand is zero —
    // every caller must exclude that input rather than relying on a -1 that is not specified.
    uint32_t find_umsb(uint32_t a) { uint32_t ri = id(); putv(code, Op_ExtInst, {t_i32, ri, glsl, Glsl_FindUMsb, a}); return i2u(ri); }
    // RDNA V_FFBH_U32 counts zeroes before the highest set bit and returns all ones for zero.
    // FindUMsb returns the bit index instead and is undefined at zero, so keep the conversion and
    // sentinel in one helper shared by ordinary ALU emission and portable dispatcher wave phases.
    uint32_t ffbh_u32(uint32_t a) {
        const uint32_t safe = ibin(Op_BitwiseOr, a, uconst(1));
        const uint32_t leading_zeroes = ibin(Op_ISub, uconst(31), find_umsb(safe));
        return sel(ucmp(Op_INotEqual, a, uconst(0)),
                   leading_zeroes, uconst(0xffffffffu));
    }
    // RDNA V_FFBL_B32 returns the first set-bit index from the LSB, with the same all-ones
    // sentinel for zero. Reversing the dword turns that index into FFBH's leading-zero count and
    // keeps the sentinel exact, so both instructions share one zero-safe FindUMsb lowering.
    uint32_t ffbl_b32(uint32_t a) { return ffbh_u32(iun(Op_BitReverse, a)); }
    // IEEE-754 binary32 ldexp on raw bits, with round-to-nearest-even for a subnormal result.
    // GLSL.std.450 Ldexp is NOT sufficient here: its contract leaves an overflowing product and
    // exp>128 undefined, while a guest exponent is an arbitrary i32. Keep the whole operation in
    // the integer domain so zero signs, infinities and NaN payloads survive exactly, and so every
    // shift remains in SPIR-V's defined [0,31] range. Exponents outside [-512,512] can be clamped
    // before the calculation: every finite nonzero binary32 must overflow above that interval or
    // round to signed zero below it.
    uint32_t ldexp_f32_bits(uint32_t bits, uint32_t exponent) {
        const uint32_t sign = ibin(Op_BitwiseAnd, bits, uconst(0x80000000u));
        const uint32_t magnitude = ibin(Op_BitwiseAnd, bits, uconst(0x7fffffffu));
        const uint32_t raw_exp = ibin(Op_ShiftRightLogical, magnitude, uconst(23));
        const uint32_t fraction = ibin(Op_BitwiseAnd, magnitude, uconst(0x007fffffu));

        // Normalize a subnormal input into a 24-bit significand with bit 23 set. FindUMsb is
        // undefined at zero, so substitute one for that otherwise-dead calculation; signed zero
        // is selected back unchanged at the end.
        const uint32_t fraction_zero = ucmp(Op_IEqual, fraction, uconst(0));
        const uint32_t safe_fraction = sel(fraction_zero, uconst(1), fraction);
        const uint32_t sub_shift = ibin(Op_ISub, uconst(23), find_umsb(safe_fraction));
        const uint32_t sub_significand =
            ibin(Op_ShiftLeftLogical, fraction, sub_shift); // sub_shift is in [1,23]
        const uint32_t normal_significand =
            ibin(Op_BitwiseOr, fraction, uconst(0x00800000u));
        const uint32_t input_subnormal = ucmp(Op_IEqual, raw_exp, uconst(0));
        const uint32_t significand =
            sel(input_subnormal, sub_significand, normal_significand);
        const uint32_t normal_unbiased = ibin(Op_ISub, raw_exp, uconst(127));
        const uint32_t sub_unbiased = ibin(Op_ISub, uconst(static_cast<uint32_t>(-126)),
                                           sub_shift);
        const uint32_t unbiased = sel(input_subnormal, sub_unbiased, normal_unbiased);

        const uint32_t bounded_exponent = sext2(
            Glsl_SMin,
            sext2(Glsl_SMax, exponent, uconst(static_cast<uint32_t>(-512))),
            uconst(512));
        const uint32_t adjusted = ibin(Op_IAdd, unbiased, bounded_exponent);

        const uint32_t normal_exp = ibin(
            Op_ShiftLeftLogical, ibin(Op_IAdd, adjusted, uconst(127)), uconst(23));
        const uint32_t normal_result = ibin(
            Op_BitwiseOr, normal_exp,
            ibin(Op_BitwiseAnd, significand, uconst(0x007fffffu)));

        // For adjusted<-126, shift the normalized 24-bit significand into the subnormal range.
        // The calculations exist in SSA for every input, so clamp the working shift to [1,24]
        // even when the normal/overflow result is ultimately selected. shift>=25 is strictly below
        // half the least subnormal (shift==24 retains the exact halfway tie and its RNE decision).
        const uint32_t under_shift =
            ibin(Op_ISub, uconst(static_cast<uint32_t>(-126)), adjusted);
        const uint32_t safe_under_shift = uext2(
            Glsl_UMax, uconst(1), uext2(Glsl_UMin, under_shift, uconst(24)));
        const uint32_t truncated =
            ibin(Op_ShiftRightLogical, significand, safe_under_shift);
        const uint32_t remainder_mask = ibin(
            Op_ISub, ibin(Op_ShiftLeftLogical, uconst(1), safe_under_shift), uconst(1));
        const uint32_t remainder = ibin(Op_BitwiseAnd, significand, remainder_mask);
        const uint32_t half = ibin(
            Op_ShiftLeftLogical, uconst(1), ibin(Op_ISub, safe_under_shift, uconst(1)));
        const uint32_t remainder_gt_half = ucmp(Op_UGreaterThan, remainder, half);
        const uint32_t remainder_eq_half = ucmp(Op_IEqual, remainder, half);
        const uint32_t truncated_odd = ucmp(
            Op_INotEqual, ibin(Op_BitwiseAnd, truncated, uconst(1)), uconst(0));
        const uint32_t round_up = lor(remainder_gt_half,
                                      land(remainder_eq_half, truncated_odd));
        const uint32_t rounded = ibin(
            Op_IAdd, truncated, sel(round_up, uconst(1), uconst(0)));
        const uint32_t too_small = scmp(Op_SGreaterThanEqual, under_shift, uconst(25));
        const uint32_t subnormal_result = sel(too_small, uconst(0), rounded);

        const uint32_t is_underflow =
            scmp(Op_SLessThan, adjusted, uconst(static_cast<uint32_t>(-126)));
        const uint32_t is_overflow = scmp(Op_SGreaterThan, adjusted, uconst(127));
        uint32_t finite_result = sel(is_underflow, subnormal_result, normal_result);
        finite_result = sel(is_overflow, uconst(0x7f800000u), finite_result);
        finite_result = ibin(Op_BitwiseOr, sign, finite_result);

        const uint32_t is_special = ucmp(Op_IEqual, raw_exp, uconst(255));
        const uint32_t is_zero = ucmp(Op_IEqual, magnitude, uconst(0));
        return sel(is_special, bits, sel(is_zero, bits, finite_result));
    }
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
    bool is_fragment_wave_vote_value(uint32_t value) const {
        return fragment_wave_vote_values.contains(value);
    }
    void mark_fragment_wave_vote_value(uint32_t value) {
        std::vector<uint32_t> pending{value};
        while (!pending.empty()) {
            const uint32_t current = pending.back();
            pending.pop_back();
            if (!fragment_wave_vote_values.insert(current).second) continue;
            if (fragment_wave_vote_scalar_consumers.contains(current))
                fragment_wave_reasons |= kFragmentWaveReasonScalarReduce;
            const auto dependent = fragment_wave_vote_dependents.find(current);
            if (dependent != fragment_wave_vote_dependents.end())
                pending.insert(pending.end(), dependent->second.begin(), dependent->second.end());
        }
    }
    void add_fragment_wave_vote_dependency(uint32_t result, uint32_t source) {
        fragment_wave_vote_dependents[source].push_back(result);
        if (is_fragment_wave_vote_value(source)) mark_fragment_wave_vote_value(result);
    }
    void propagate_fragment_wave_vote(uint32_t result, uint32_t source) {
        add_fragment_wave_vote_dependency(result, source);
    }
    void propagate_fragment_wave_vote(uint32_t result, uint32_t a, uint32_t b_) {
        add_fragment_wave_vote_dependency(result, a);
        add_fragment_wave_vote_dependency(result, b_);
    }
    void mark_fragment_wave_scalar_use(uint32_t value) {
        fragment_wave_vote_scalar_consumers.insert(value);
        if (is_fragment_wave_vote_value(value))
            fragment_wave_reasons |= kFragmentWaveReasonScalarReduce;
    }
    uint32_t bsel(uint32_t cond, uint32_t tval, uint32_t fval) {
        uint32_t r = id();
        put(code, Op_Select, {t_bool, r, cond, tval, fval});
        add_fragment_wave_vote_dependency(r, cond);
        add_fragment_wave_vote_dependency(r, tval);
        add_fragment_wave_vote_dependency(r, fval);
        return r;
    }  // bool-domain select (wave masks)

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
        propagate_fragment_wave_vote(r, v0);
        fragment_wave_vote_phi_patch_results[patch_off] = r;
        return r;
    }
    void patch_phi(size_t patch_off, uint32_t v1, uint32_t l1) {
        code[patch_off] = v1;
        code[patch_off + 1] = l1;
        const auto result = fragment_wave_vote_phi_patch_results.find(patch_off);
        if (result != fragment_wave_vote_phi_patch_results.end())
            propagate_fragment_wave_vote(result->second, v1);
    }
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
        uint32_t value = id();
        put(code, Op_Load, {type, value, var});
        add_fragment_wave_vote_dependency(value, var);
        return value;
    }
    void store_function(uint32_t var, uint32_t value) {
        put(code, Op_Store, {var, value});
        // Function storage can join dynamic paths and loops. Keep an explicit dependency edge so
        // a vote emitted later can propagate back through loads already generated by a dispatcher.
        add_fragment_wave_vote_dependency(var, value);
    }
    uint32_t logical_not(uint32_t value) {
        uint32_t result = id();
        put(code, Op_LogicalNot, {t_bool, result, value});
        propagate_fragment_wave_vote(result, value);
        return result;
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
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonLaneId;
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
        fragment_wave_reasons |= kFragmentWaveReasonWaveAny;
        uint32_t result = id();
        put(code, Op_GroupNonUniformAny,
            {t_bool, result, uconst(Scope_Subgroup), active_bit});
        mark_fragment_wave_vote_value(result);
        return result;
    }
    // Fragment-side ballot (#2412). Same exactness argument as native_wave_ballot_half, but the
    // fragment stage establishes its exact-wave contract differently: it does not carry
    // native_subgroup_size, it DECLARES the width it needs via fragment_required_subgroup_size, which
    // the backend then enforces (or skips the draw). Mirrors fragment_wave_any, which exists for the
    // reduction form of the same problem.
    //
    // This is a REDUCE in #2410's taxonomy -- the bits become guest scalar DATA -- so the wave64
    // requirement it records must NOT be relaxed to a narrower subgroup: half a mask reported as whole
    // is silent wrong data.
    uint32_t fragment_wave_ballot_half(uint32_t mask_bit, uint32_t half) {
        if (!is_fragment || !mask_bit) return 0;
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_ballot) {
            put(caps, Op_Capability, {Cap_GroupNonUniformBallot});
            declared_subgroup_ballot = true;
        }
        fragment_required_subgroup_size = wave_size;
        // Ballot, not vote: the reason bit must say which, because only one of the two can ever be
        // relaxed to a narrower subgroup and the skip diagnostic is where that question gets asked
        // (#2441). This records the same width as before -- the gate keys on
        // fragment_required_subgroup_size, not on the reason -- so behaviour is unchanged.
        fragment_wave_reasons |= kFragmentWaveReasonWaveBallot;
        const uint32_t ballot = id();
        put(code, Op_GroupNonUniformBallot,
            {t_v4u(), ballot, uconst(Scope_Subgroup), mask_bit});
        const uint32_t result = id();
        putv(code, Op_CompositeExtract, {t_u32, result, ballot, half});
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
    // One 32-bit HALF of the guest wave mask, materialised from this lane's bool (#2420).
    // OpGroupNonUniformBallot returns a uvec4 whose .x/.y are lanes 0..31 / 32..63 OF THIS SUBGROUP,
    // so it equals the guest mask only when the subgroup is exactly the guest wave. The caller must
    // establish that; a narrower subgroup would report half a mask as though it were whole, which is
    // silent wrong data rather than a visible reject.
    bool declared_subgroup_ballot = false;
    uint32_t native_wave_ballot_half(uint32_t mask_bit, uint32_t half) {
        if (!native_subgroup_size || !mask_bit) return 0;
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_ballot) {
            put(caps, Op_Capability, {Cap_GroupNonUniformBallot});
            declared_subgroup_ballot = true;
        }
        const uint32_t ballot = id();
        put(code, Op_GroupNonUniformBallot,
            {t_v4u(), ballot, uconst(Scope_Subgroup), mask_bit});
        const uint32_t result = id();
        putv(code, Op_CompositeExtract, {t_u32, result, ballot, half});
        return result;
    }
    uint32_t native_wave_first_active(uint32_t mask_bit) {
        if (!native_subgroup_size) return 0;
        const uint32_t lane = subgroup_local_id();
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        // The exclusive population count is zero only for the first active lane. Exactly one lane
        // therefore contributes its index to the reduction; an empty mask is distinguished by the
        // accompanying wave vote. This is exact only under the enforced guest-size subgroup
        // contract checked by the caller.
        const uint32_t contribution = sel(mask_bit, uconst(1), uconst(0));
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, contribution});
        const uint32_t elected = land(mask_bit, ucmp(Op_IEqual, prefix, uconst(0)));
        const uint32_t selected_lane = sel(elected, lane, uconst(0));
        uint32_t first = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, first, uconst(Scope_Subgroup), GroupOp_Reduce, selected_lane});
        return sel(native_wave_any(mask_bit), first, uconst(0xffffffffu));
    }
    uint32_t native_wave_popcount(uint32_t mask_bit) {
        if (!native_subgroup_size || !mask_bit) return 0;
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t contribution = sel(mask_bit, uconst(1), uconst(0));
        uint32_t result = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, result, uconst(Scope_Subgroup), GroupOp_Reduce, contribution});
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
    // Distinct guest descriptors can name the same allocation. Under the GLSL450 memory model,
    // separate StorageBuffer variables otherwise promise non-aliasing to the SPIR-V consumer. The
    // runtime binding addresses are deliberately absent from the shader cache key, so every external
    // guest-backed buffer declaration must retain the possibility of aliasing. Internal GDS uses its
    // own host allocation and intentionally does not go through this helper.
    void declare_external_storage_buffer(uint32_t pointer_type, uint32_t variable) {
        put(types, Op_Variable, {pointer_type, variable, SC_StorageBuffer});
        put(deco, Op_Decorate, {variable, Dec_Aliased});
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
    // Atomic publication for the diagnostic witness. Plain OpStore is wrong for a record every
    // invocation that reaches the cap writes: with per-invocation values the host can read a record
    // assembled from several invocations, and "last writer wins" silently discards the extremes that
    // make the record mean anything. A device-scope atomic min/max makes the published pair the true
    // extremes over every invocation and workgroup, which is the only reading the report claims.
    void compute_gds_atomic_minmax(uint16_t opcode, uint32_t index, uint32_t value,
                                   uint32_t pred) {
        declare_internal_gds(0, kComputeInternalGdsBinding);
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        uint32_t pointer = id();
        putv(code, Op_AccessChain,
             {t_ptr_gds_u32, pointer, v_internal_gds, uconst(0), index});
        uint32_t old = id();
        put(code, opcode, {t_u32, old, pointer, uconst(Scope_Device),
                           uconst(MemSem_UniformAcqRel), value});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
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
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonDppRow16;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 16u);
    }
    void mark_subgroup_min32() {
        // PERMLANEX16 crosses a pair of 16-lane rows.
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonPermLane32;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 32u);
    }
    void mark_subgroup_min64() {
        // V_READLANE_B32 may address every lane of a wave64.
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonReadLane64;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 64u);
    }
    uint32_t subgroup_shuffle(uint32_t value, uint32_t lane) {
        // Every supported native shuffle at least addresses an architectural quad. Wider row/wave
        // operations raise this contract before calling the common helper.
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonShuffle;
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
    uint32_t subgroup_row_shr_dynamic(uint32_t value, uint32_t active,
                                      uint32_t amount, uint32_t event = 0,
                                      uint32_t* valid_lane = nullptr) {
        mark_subgroup_min16();
        const uint32_t lane = subgroup_local_id();
        const uint32_t row_lane = ibin(Op_BitwiseAnd, lane, uconst(15));
        const uint32_t in_bounds = ucmp(Op_UGreaterThanEqual, row_lane, amount);
        // Keep the shuffle index valid even for row-leading lanes. FI=0 also makes an EXEC-inactive
        // source invalid. The caller uses `valid` to preserve VDST for BOUND_CTRL=0; returning VALUE
        // here is only a safe placeholder for the disabled lane, not its architectural result.
        const uint32_t source_lane = sel(in_bounds,
            ibin(Op_ISub, lane, amount), lane);
        const uint32_t shifted = subgroup_shuffle(value, source_lane);
        const uint32_t source_active = subgroup_shuffle(
            sel(active, uconst(1), uconst(0)), source_lane);
        uint32_t valid = land(in_bounds,
            ucmp(Op_INotEqual, source_active, uconst(0)));
        // A lane-local graphics dispatcher can have adjacent lanes parked at distinct static DPP
        // instructions in the same common phase.  Shuffling only ACTIVE would let one instruction
        // consume a neighbour published by another instruction.  Carry the static event tag through
        // the identical shuffle and require it to match the destination lane's event.
        if (event) {
            const uint32_t source_event = subgroup_shuffle(event, source_lane);
            valid = land(valid, ucmp(Op_IEqual, source_event, event));
        }
        if (valid_lane) *valid_lane = valid;
        return sel(valid, shifted, value);
    }
    uint32_t subgroup_row_shr(uint32_t value, uint32_t active, uint32_t amount,
                              uint32_t* valid_lane = nullptr) {
        return subgroup_row_shr_dynamic(value, active, uconst(amount), 0, valid_lane);
    }
    // `stride` is the XOR applied to the lane id, and must be < 16.
    //
    // ROW_ROR:8 was the first member of this family and is spelled XOR 8 below, because XOR 8 is
    // exactly (row_lane - 8) modulo 16. ROW_XMASK:n is XOR n by definition, so the two controls
    // share one implementation and ROW_ROR:8 is literally ROW_XMASK:8. Generalising the stride is
    // therefore not a widening of the lowering, only of the control values that reach it.
    //
    // The property that makes every stride < 16 exact, and makes it exact independently of the host
    // subgroup width, is that XOR by a value under 16 touches only bits 0..3 -- so the source lane
    // stays inside the same architectural DPP16 row, and every row and subgroup bit above bit 3 is
    // preserved. That is the same argument the ROR:8 form already relied on; it was never specific
    // to 8.
    uint32_t subgroup_row_xor(uint32_t value, uint32_t active, uint32_t stride,
                              uint32_t* valid_lane = nullptr) {
        mark_subgroup_min16();
        const uint32_t lane = subgroup_local_id();
        const uint32_t source_lane = ibin(Op_BitwiseXor, lane, uconst(stride & 15u));
        const uint32_t rotated = subgroup_shuffle(value, source_lane);
        // FI=0 makes an EXEC-inactive source invalid. The one admitted form has BOUND_CTRL=1,
        // whose caller substitutes zero for that invalid source before V_MIN_F32.
        const uint32_t source_active = subgroup_shuffle(
            sel(active, uconst(1), uconst(0)), source_lane);
        const uint32_t valid = ucmp(Op_INotEqual, source_active, uconst(0));
        if (valid_lane) *valid_lane = valid;
        return sel(valid, rotated, value);
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
    uint32_t ds_bpermute_b32(uint32_t address, uint32_t value,
                             uint32_t active, uint32_t offset,
                             uint32_t event = 0) {
        // RDNA2 ISA 12.13.3: ADDR is a byte address and BPERMUTE gathers DATA0 backward from
        // ((ADDR + OFFSET) >> 2) & 31 within each independent 32-lane half. Addition is deliberately
        // performed before the shift so uint32 overflow and unaligned byte addresses match hardware.
        mark_subgroup_min32();
        const uint32_t lane = subgroup_local_id();
        const uint32_t selected = ibin(
            Op_BitwiseAnd,
            ibin(Op_ShiftRightLogical,
                 ibin(Op_IAdd, address, offset), uconst(2)),
            uconst(31));
        const uint32_t source_lane = ibin(
            Op_BitwiseOr, ibin(Op_BitwiseAnd, lane, uconst(~31u)), selected);
        const uint32_t shuffled = subgroup_shuffle(value, source_lane);
        const uint32_t source_active = subgroup_shuffle(
            sel(active, uconst(1), uconst(0)), source_lane);
        uint32_t valid = ucmp(Op_INotEqual, source_active, uconst(0));
        if (event) {
            const uint32_t source_event = subgroup_shuffle(event, source_lane);
            valid = land(valid, ucmp(Op_IEqual, source_event, event));
        }
        return sel(valid, shuffled, uconst(0));
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
        uint32_t r = id();
        put(code, Op_Phi, {type, r, va, la, vb, lb});
        propagate_fragment_wave_vote(r, va, vb);
        return r;
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
    uint32_t t_u64_cache = 0; bool declared_int64 = false, declared_int64_atomics = false;
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
    uint32_t sel64(uint32_t cond, uint32_t yes, uint32_t no) {
        uint32_t r = id(); put(code, Op_Select, {t_u64(), r, cond, yes, no}); return r;
    }
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
    std::unordered_map<uint32_t, bool> tex_binding_arrayed;  // binding -> declared with Arrayed=1
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
        tex_binding_arrayed[binding] = arrayed;
        return true;
    }
    // The sampling coordinate for `binding`, with the array layer appended when the binding was
    // DECLARED arrayed. #325: arrayed-ness is a property of the RESOURCE (a guest 2D_ARRAY T#), not
    // of the instruction, because the uploader chooses the view type from the resource and cannot
    // see which opcode will sample it. So a non-array instruction reaching an array texture must
    // still produce a three-component coordinate -- and layer 0 is exactly the base slice it used
    // to get from the old base-slice 2D view, so behaviour is preserved where it was already right.
    bool tex_is_arrayed(uint32_t binding) {
        auto it = tex_binding_arrayed.find(binding);
        return it != tex_binding_arrayed.end() && it->second;
    }
    uint32_t tex_coord_uv(uint32_t binding, uint32_t u_bits, uint32_t v_bits) {
        uint32_t c = id();
        if (tex_binding_arrayed.count(binding) && tex_binding_arrayed[binding])
            put(code, Op_CompositeConstruct,
                {t_v3f(), c, bcf(u_bits), bcf(v_bits), fconstf(0.0f)});
        else
            put(code, Op_CompositeConstruct, {t_v2f(), c, bcf(u_bits), bcf(v_bits)});
        return c;
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
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
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
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
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
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
        uint32_t res   = id(); put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, bcf(lod_bits)});
        unpack_texture_result(binding, res, out);
    }
    // A real 2D-array explicit-LOD sample. The layer is the third float coordinate; keeping it in
    // SPIR-V makes reflection require a matching 2D-array view instead of silently sampling layer 0.
    // IMAGE_SAMPLE from a 2D_ARRAY texture. The implicit-LOD sibling of the _lod_ form below, and
    // the distinction is not cosmetic: in a fragment stage the LOD comes from quad derivatives, so
    // lowering a plain SAMPLE through the explicit-LOD helper would pin every textured surface to
    // the base level. Outside a fragment stage there are no derivatives, so LOD 0 is the only
    // legal choice -- which is exactly what image_sample_2d already does for the non-array case.
    // PROSPER_FORCE_LAYER=<n>: the constant array layer to substitute, or -1 for off. A malformed
    // or empty value DISABLES the probe rather than forcing layer 0 -- a typo must cost a
    // measurement, never produce a wrong one silently.
    // Announce a substitution, from whichever sampler performed it. Both must announce or the
    // probe produces exactly the null it exists to prevent: a title that samples only through the
    // explicit-LOD helper would otherwise get full substitution with no output at all.
    static void announce_forced_layer(uint32_t binding, int layer) {
        static std::atomic<uint32_t> said{0};
        if (said.fetch_add(1, std::memory_order_relaxed) < 4u) {
            fprintf(stderr, "[force-layer] binding=%u substituted constant layer %d\n",
                    binding, layer);
            fflush(stderr);
        }
    }
    static int forced_array_layer() {
        static const int v = [] {
            const char* e = getenv("PROSPER_FORCE_LAYER");
            if (!e || !*e) return -1;
            char* end = nullptr;
            const long n = strtol(e, &end, 10);
            if (!end || *end || n < 0 || n > 65535) return -1;
            return (int)n;
        }();
        return v;
    }
    void image_sample_2d_array(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                               uint32_t layer_bits, uint32_t out[4]) {
        uint32_t si = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        // #2998: PROSPER_FORCE_LAYER=<n> substitutes a constant array layer, and SAYS SO, which is
        // the point -- it separates "the shader is given the wrong slice" from "the slice it asks
        // for holds the wrong content", two failures that look identical on screen. A probe that
        // could not show its own lever moved would leave a null meaning nothing.
        //
        // This forces guest-visible state, so its output illustrates an investigation and is never
        // acceptance evidence for a rendered frame.
        const int forced_layer = forced_array_layer();
        uint32_t layer_use = bcf(layer_bits);
        if (forced_layer >= 0) {
            layer_use = fconstf((float)forced_layer);
            announce_forced_layer(binding, forced_layer);
        }
        uint32_t coord = id(); put(code, Op_CompositeConstruct,
                                   {t_v3f(), coord, bcf(u_bits), bcf(v_bits), layer_use});
        uint32_t res = id();
        if (is_fragment)
            put(code, Op_ImageSampleImplicitLod, {texture_vec4(binding), res, si, coord});
        else
            put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord,
                                                  ImgOp_Lod, fconstf(0.0f)});
        unpack_texture_result(binding, res, out);
    }
    void image_sample_lod_2d_array(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                   uint32_t layer_bits, uint32_t lod_bits, uint32_t out[4]) {
        uint32_t si = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        // #2998: PROSPER_FORCE_LAYER applies HERE TOO. Covering only the implicit-LOD sampler is
        // the scope error this session already made once and had to withdraw: the census puts most
        // of this title's array events on other opcodes, so a probe on one helper produces a null
        // that means nothing about the rest.
        uint32_t lod_layer = bcf(layer_bits);
        if (forced_array_layer() >= 0) {
            lod_layer = fconstf((float)forced_array_layer());
            announce_forced_layer(binding, forced_array_layer());
        }
        uint32_t coord = id(); put(code, Op_CompositeConstruct,
                                   {t_v3f(), coord, bcf(u_bits), bcf(v_bits), lod_layer});
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
        // Consult the DECLARATION, not just the caller: a binding declared arrayed needs
        // three components whatever opcode reached it, and a caller that did not supply a
        // slice reads the base one (#325). Hoisted so BOTH the sampled and the manual-fetch
        // branches below use it -- they disagreed, and only one was fixed.
        const bool coord_arrayed =
            arrayed || (tex_binding_arrayed.count(binding) && tex_binding_arrayed[binding]);

        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        if (linear_filter) {
            if (!declared_image_query) {
                put(caps, Op_Capability, {Cap_ImageQuery});
                declared_image_query = true;
            }
            uint32_t img = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
            // #325: the DECLARATION, not the caller -- the third of three uses in this function,
            // and the one the first pass missed. Query-size arity is validated exactly, so an
            // ivec2 against an Arrayed image is invalid SPIR-V.
            uint32_t size = id(); put(code, Op_ImageQuerySizeLod,
                                      {coord_arrayed ? t_v3i() : t_v2i(), size, img, uconst(0)});
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
                // #325: follow the DECLARATION, as the sampled branch above does. A caller that
                // reached an arrayed binding without supplying a slice still needs three
                // components, and reads the base one.
                uint32_t coord = id();
                if (coord_arrayed)
                    put(code, Op_CompositeConstruct,
                        {t_v3u_fetch(), coord, tx, ty,
                         arrayed ? i2u(cvt_f2i(bcf(layer_bits))) : uconst(0)});
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
        if (coord_arrayed)
            put(code, Op_CompositeConstruct,
                {t_v3f(), coord, bcf(u_bits), bcf(v_bits),
                 arrayed ? bcf(layer_bits) : fconstf(0.0f)});
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
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
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
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
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
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
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
        // #325: OpImageQuerySizeLod on an Arrayed 2D image yields ivec3 (w, h, layers). Asking
        // for ivec2 is invalid SPIR-V; the width and height are still components 0 and 1.
        const uint32_t q_type = tex_is_arrayed(binding) ? t_v3i() : t_v2i();
        uint32_t size = id(); put(code, Op_ImageQuerySizeLod, {q_type, size, img, uconst(0)});
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
            // #325: an Arrayed 2D image queries as ivec3, its third component being the layer
            // COUNT -- which is exactly what GET_RESINFO's third result means for a 2D_ARRAY T#, so
            // reporting it is right rather than merely legal.
            const bool arrayed_2d = dim == Dim_2D && tex_is_arrayed(binding);
            const uint32_t size_type = (dim == Dim_2D && !arrayed_2d) ? t_v2i() : t_v3i();
            uint32_t size = id(); put(code, Op_ImageQuerySizeLod, {size_type, size, img, bcs(lod_bits)});
            const uint32_t components = (dim == Dim_2D && !arrayed_2d) ? 2u : 3u;
            for (uint32_t c = 0; c < components; c++) {
                uint32_t value = id(); put(code, Op_CompositeExtract, {t_i32, value, size, c});
                out[c] = i2u(value);
            }
        }
        uint32_t levels = id(); put(code, Op_ImageQueryLevels, {t_i32, levels, img});
        out[3] = i2u(levels);
    }
    // image_get_lod: return the sampler-clamped and raw implicit LOD for a hypothetical 2D sample.
    // SPIR-V defines OpImageQueryLod's x/y results in the same order as RDNA2's VDATA[0]/[1], so the
    // two raw FP32 values can be copied directly to the guest VGPRs. Like the hardware instruction,
    // this consumes screen-space derivatives and is therefore fragment-only.
    void image_get_lod_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t out[2]) {
        if (!declared_image_query) { put(caps, Op_Capability, {Cap_ImageQuery}); declared_image_query = true; }
        const uint32_t simg = tex_binding_simg[binding];
        uint32_t si = id(); put(code, Op_Load, {simg, si, tex_var[binding]});
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
        uint32_t lod = id(); put(code, Op_ImageQueryLod, {t_v2f(), lod, si, coord});
        for (uint32_t component = 0; component < 2; ++component) {
            uint32_t value = id();
            put(code, Op_CompositeExtract, {t_f32, value, lod, component});
            out[component] = bcu(value);
        }
    }
    // 2-component uint vector (integer texel coordinates for OpImageFetch).
    uint32_t t_v2u_cache = 0;
    uint32_t t_v2u() { if (!t_v2u_cache) { t_v2u_cache = id(); put(types, Op_TypeVector, {t_v2u_cache, t_u32, 2}); } return t_v2u_cache; }
    uint32_t t_v3u_cache2 = 0;
    uint32_t t_v3u_fetch() {
        if (t_v3u) return t_v3u;   // compute/vertex shells already declare a uvec3 for built-ins
        if (!t_v3u_cache2) { t_v3u_cache2 = id(); put(types, Op_TypeVector, {t_v3u_cache2, t_u32, 3}); }
        return t_v3u_cache2;
    }
    // image_load 2D (image_load): texelFetch the image at the combined sampler's `binding` with INTEGER
    // (x,y) coords (raw VGPR bits). OpImage strips the sampler; OpImageFetch at explicit LOD 0.
    void image_fetch_2d(uint32_t binding, uint32_t x_bits, uint32_t y_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load,  {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
        // #325: an Arrayed image needs a three-component fetch coordinate whatever opcode got
        // here. Layer 0 is the base slice a graphics IMAGE_LOAD used to read through the old
        // base-slice 2D view, so this preserves that behaviour rather than inventing one.
        uint32_t coord = id();
        if (tex_is_arrayed(binding))
            put(code, Op_CompositeConstruct, {t_v3u_fetch(), coord, x_bits, y_bits, uconst(0)});
        else
            put(code, Op_CompositeConstruct, {t_v2u(), coord, x_bits, y_bits});
        uint32_t res   = id(); put(code, Op_ImageFetch, {texture_vec4(binding), res, img, coord, ImgOp_Lod, uconst(0)});
        unpack_texture_result(binding, res, out);
    }

    // Exact sampled IMAGE_LOAD representation for guest 2D_MSAA: the host image is single-sample
    // 2D-array, with each guest sample plane in one layer. The guest's explicit sample coordinate is
    // therefore the third integer coordinate; LOD remains zero because the host view has one level.
    void image_fetch_2d_array(uint32_t binding, uint32_t x_bits, uint32_t y_bits,
                              uint32_t layer_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
        uint32_t coord = id(); put(code, Op_CompositeConstruct,
                                   {t_v3u_fetch(), coord, x_bits, y_bits, layer_bits});
        uint32_t res   = id(); put(code, Op_ImageFetch,
                                   {texture_vec4(binding), res, img, coord,
                                    ImgOp_Lod, uconst(0)});
        unpack_texture_result(binding, res, out);
    }

    // image_load from a 3D texture (integer texel fetch through the combined sampler — DOLL's
    // color-grade LUT, #273): OpImage strips the sampler; OpImageFetch with (x,y,z) integer coords.
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
    bool declared_read_wo_fmt = false, declared_write_wo_fmt = false,
         declared_storage_extended = false, declared_sampled1d = false,
         declared_ms = false, declared_msarray = false;
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
        if ((image_format == ImgFmt_R8ui || image_format == ImgFmt_R16ui ||
             image_format == ImgFmt_Rgba8ui) &&
            !declared_storage_extended) {
            put(caps, Op_Capability, {Cap_StorageImageExtendedFormats});
            declared_storage_extended = true;
        }
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
        {
            // Vulkan's narrow integer storage conversion may saturate a u32 value that does not fit
            // the target, while the guest image-store contract discards the high bits. Make that
            // conversion explicit so typed R8ui/R16ui stay identical to the raw CPU pack fallback
            // for arbitrary shader values, not only values loaded from another narrow surface.
            const uint32_t width_mask =
                (stg_img_format[binding] == ImgFmt_R8ui ||
                 stg_img_format[binding] == ImgFmt_Rgba8ui) ? 0xffu
                : stg_img_format[binding] == ImgFmt_R16ui ? 0xffffu : 0u;
            uint32_t narrowed[4] = {vals[0], vals[1], vals[2], vals[3]};
            if (width_mask)
                for (uint32_t c = 0; c < 4; ++c)
                    narrowed[c] = ibin(Op_BitwiseAnd, vals[c], uconst(width_mask));
            put(code, Op_CompositeConstruct,
                {t_v4u(), texel, narrowed[0], narrowed[1], narrowed[2], narrowed[3]});
        }
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
        // 2D and 2D_ARRAY R32_UINT atomics. `arrayed` is implied by ncoord==3 -- the gate below
        // only reaches here for SQ_DIM_2D (ncoord 2) and SQ_DIM_2D_ARRAY (ncoord 3, x/y/layer).
        const bool arrayed = ncoord == 3;
        if (ncoord != 2 && ncoord != 3) return fallback;
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
        // An arrayed image's OpImageQuerySize yields (width, height, layers); a plain 2D yields
        // (width, height). Querying the wrong arity is a SPIR-V validity error rather than a silent
        // miscompile, so spv_validate catches a mistake here.
        put(code, Op_ImageQuerySize, {arrayed ? t_v3i() : t_v2i(), size, image});
        const uint32_t width_i = id(), height_i = id();
        put(code, Op_CompositeExtract, {t_i32, width_i, size, 0});
        put(code, Op_CompositeExtract, {t_i32, height_i, size, 1});
        uint32_t in_bounds = ucmp(Op_ULessThan, coords[0], i2u(width_i));
        in_bounds = land(in_bounds, ucmp(Op_ULessThan, coords[1], i2u(height_i)));
        if (arrayed) {
            // The layer bound is NOT optional. Vulkan leaves an out-of-bounds image atomic
            // undefined -- robust image access does not cover atomics -- and the 2D case's own
            // comment records that RADV can spend seconds in one before resetting the GPU. An
            // unbounded layer index would be exactly that, so the layer is bounded from the image's
            // own query rather than from a descriptor field the shader cannot see.
            const uint32_t layers_i = id();
            put(code, Op_CompositeExtract, {t_i32, layers_i, size, 2});
            in_bounds = land(in_bounds, ucmp(Op_ULessThan, coords[2], i2u(layers_i)));
        }
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
    // Logical-addressing SPIR-V cannot bitcast a pointer into the ordinary runtime-u32 Block to a
    // u64 pointer. Declare a second, aliased StorageBuffer variable at the SAME descriptor binding.
    // Vulkan binds both declarations to one VkBuffer; ArrayStride=8 gives the qword atomics their
    // natural record index. The narrow caller excludes descriptor arrays.
    uint32_t u64_buf_for_binding(uint32_t binding) {
        if (auto found = cbuf_u64_var.find(binding); found != cbuf_u64_var.end())
            return found->second;
        if (!t_ptr_sb_struct_u64) {
            const uint32_t runtime_array = id(), block = id();
            t_ptr_sb_struct_u64 = id();
            t_ptr_sb_u64 = id();
            put(deco, Op_Decorate, {runtime_array, Dec_ArrayStride, 8});
            put(deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
            put(deco, Op_Decorate, {block, Dec_Block});
            put(types, Op_TypeRuntimeArray, {runtime_array, t_u64()});
            put(types, Op_TypeStruct, {block, runtime_array});
            put(types, Op_TypePointer, {t_ptr_sb_struct_u64, SC_StorageBuffer, block});
            put(types, Op_TypePointer, {t_ptr_sb_u64, SC_StorageBuffer, t_u64()});
        }
        const uint32_t variable = id();
        put(deco, Op_Decorate, {variable, Dec_DescriptorSet, desc_set});
        put(deco, Op_Decorate, {variable, Dec_Binding, binding});
        declare_external_storage_buffer(t_ptr_sb_struct_u64, variable);
        cbuf_u64_var[binding] = variable;
        return variable;
    }
    void mark_cbuf_coherent(uint32_t binding) {
        const uint32_t buf = buf_for_binding(binding);
        if (cbuf_coherent_vars.insert(buf).second)
            put(deco, Op_Decorate, {buf, Dec_Coherent});
    }
    void device_uniform_release_barrier() {
        put(code, Op_MemoryBarrier,
            {uconst(Scope_Device), uconst(MemSem_UniformRelease)});
    }
    // Load one dword (raw bits) from the constant/vertex buffer at descriptor `binding` at dword index
    // `idx` (SMEM). The 1-arg form keeps the legacy slot convention (0 -> binding 2, 1 -> binding 3).
    // THE only place a storage-buffer element pointer is built. There were FOUR identical
    // OpAccessChain sites -- cbuf_load_impl, cbuf_store's unpredicated and predicated branches, and
    // cbuf_atomic_rtn -- and a table-indexed binding needs the leading selector in every one of them.
    //
    // #2474 claimed cbuf_load_impl was "the single place a buffer access chain is built" and used that
    // as its correctness argument. It was false, and the review that caught it found three sites; the
    // fourth turned up while fixing it. For a table-indexed binding the pointee is `OpTypeArray %Block N`,
    // so an unfixed site does not merely read the wrong slot: `uconst(0)` consumes the ARRAY index and
    // `idx` is then a member index into the Block, which has one member. So idx==0 yields a pointer to
    // the runtime array against a declared `t_ptr_sb_u32` (result-type mismatch) and idx!=0 is
    // out-of-range -- an invalid module either way. Loud rather than silent, but only where something
    // validates; on this path nothing does yet, so it surfaces as a failed pipeline and a dropped draw.
    //
    // Hence one function: the sites cannot drift apart again, and a fifth caller inherits the selector
    // rather than a stale comment promising there is only one.
    uint32_t cbuf_element_ptr(uint32_t buf, uint32_t binding, uint32_t idx,
                              uint32_t* array_valid = nullptr) {
        if (array_valid) *array_valid = 0;
        auto arity = cbuf_table_arity.find(binding);
        auto index_sgpr = cbuf_table_index_sgpr.find(binding);
        auto live_selector = cbuf_table_selector_value.find(binding);
        auto selector_block = cbuf_table_selector_block.find(binding);
        uint32_t ptr = id();
        if (arity != cbuf_table_arity.end()) {
            // A live GPU-computed selector takes precedence over the user-SGPR convention: the two
            // are different contracts, and a table whose index the guest derives on the GPU has no
            // push-constant home at all.
            const bool have_live_selector =
                is_compute && arity->second <= 4096u &&
                live_selector != cbuf_table_selector_value.end() && live_selector->second != 0 &&
                // Dominance, see cbuf_table_selector_block. Failing this leaves `have_live_selector`
                // false and, since this mode never sets `table_index_sgpr`, drops through to
                // `invalid_cbuf_array_access` below -- the module is discarded. That is exactly what
                // the same shader does WITHOUT this feature (its MUBUF cannot resolve), so the guard
                // cannot regress anything: it can only decline a case that was never sound.
                selector_block != cbuf_table_selector_block.end() &&
                selector_block->second == cur_block;
            // Only the compute shell declares the user-SGPR push-constant block. Graphics paths do
            // not currently have a runtime source for this selector, so emitting an access there
            // would manufacture type/variable id zero and an invalid module.
            if (!have_live_selector &&
                (!is_compute || arity->second > 4096u ||
                index_sgpr == cbuf_table_index_sgpr.end() ||
                index_sgpr->second >= push_constant_dword_count)) {
                invalid_cbuf_array_access = true;
                putv(code, Op_AccessChain,
                     {t_ptr_sb_u32, ptr, buf, uconst(0), uconst(0), idx});
                return ptr;
            }
            const uint32_t selector = have_live_selector
                ? live_selector->second
                : load_push_constant(index_sgpr->second);
            const uint32_t valid = ucmp(Op_ULessThan, selector, uconst(arity->second));
            const uint32_t safe_selector = sel(valid, selector, uconst(0));
            if (array_valid) *array_valid = valid;
            // NonUniform on both the selector and the pointer, unconditionally. Measured: of
            // PPSA04263's 51 compute launches, 5 are local=256x1x1 -- four waves per workgroup, so four
            // EXECs and four distinct scalar indices, wave-uniform but NOT dynamically uniform across
            // the invocation group. 43 of 51 are one wave per group, which is why the "obviously
            // uniform" reading looks safe and is wrong. `load_push_constant` mints a fresh id per call
            // and does not cache, so these decorations never duplicate onto one id.
            put(deco, Op_Decorate, {safe_selector, Dec_NonUniform});
            putv(code, Op_AccessChain,
                 {t_ptr_sb_u32, ptr, buf, safe_selector, uconst(0), idx});
            put(deco, Op_Decorate, {ptr, Dec_NonUniform});
            return ptr;
        }
        putv(code, Op_AccessChain, {t_ptr_sb_u32, ptr, buf, uconst(0), idx});
        return ptr;
    }
    uint32_t cbuf_load_impl(uint32_t idx, uint32_t binding, bool coherent_access) {
        uint32_t buf = buf_for_binding(binding);
        if (coherent_access) mark_cbuf_coherent(binding);
        uint32_t array_valid = 0;
        uint32_t p = cbuf_element_ptr(buf, binding, idx, &array_valid);
        uint32_t r = id();
        if (coherent_access) put(code, Op_Load, {t_u32, r, p, MemAccess_Volatile});
        else                 put(code, Op_Load, {t_u32, r, p});
        return array_valid ? sel(array_valid, r, uconst(0)) : r;
    }
    uint32_t cbuf_load(uint32_t idx, uint32_t binding = 2, bool coherent_access = false) {
        cbuf_ordinary_accesses.insert(binding);
        return cbuf_load_impl(idx, binding, coherent_access);
    }
    struct U64PairAdd {
        uint32_t lo = 0;
        uint32_t hi = 0;
        uint32_t overflow = 0;
    };
    U64PairAdd add_u64_pair_u32(uint32_t lo, uint32_t hi, uint32_t addend) {
        U64PairAdd result;
        result.lo = ibin(Op_IAdd, lo, addend);
        const uint32_t carry = ucmp(Op_ULessThan, result.lo, lo);
        result.hi = ibin(Op_IAdd, hi, sel(carry, uconst(1), uconst(0)));
        result.overflow = ucmp(Op_ULessThan, result.hi, hi);
        return result;
    }
    uint32_t u64_pair_ule(uint32_t lhs_lo, uint32_t lhs_hi,
                          uint32_t rhs_lo, uint32_t rhs_hi) {
        const uint32_t high_less = ucmp(Op_ULessThan, lhs_hi, rhs_hi);
        const uint32_t high_equal = ucmp(Op_IEqual, lhs_hi, rhs_hi);
        const uint32_t low_less_equal = ucmp(Op_ULessThanEqual, lhs_lo, rhs_lo);
        return lor(high_less, land(high_equal, low_less_equal));
    }
    void declare_indirect_pointer_descriptor_capture() {
        uint32_t pointer_type = 0;
        indirect_pointer_source_record_var = function_var(t_u32, pointer_type);
        indirect_pointer_source_root_lo_var = function_var(t_u32, pointer_type);
        indirect_pointer_source_root_hi_var = function_var(t_u32, pointer_type);
        // UINT32_MAX cannot name a record in the validated source table. It is also the safe
        // fail-closed state for a lane which reaches a consumer without executing the producer.
        store_function(indirect_pointer_source_record_var, uconst(UINT32_MAX));
        store_function(indirect_pointer_source_root_lo_var, uconst(0));
        store_function(indirect_pointer_source_root_hi_var, uconst(0));
    }
    void capture_indirect_pointer_descriptor_source(
            uint32_t record_index, uint32_t root_lo, uint32_t descriptor_word1,
            bool predicated, uint32_t predicate) {
        if (!indirect_pointer_source_record_var ||
            !indirect_pointer_source_root_lo_var ||
            !indirect_pointer_source_root_hi_var)
            return;
        uint32_t root_hi = ibin(
            Op_BitwiseAnd, descriptor_word1, uconst(0xffffu));
        if (predicated) {
            const uint32_t old_record = load_function(
                t_u32, indirect_pointer_source_record_var);
            const uint32_t old_root_lo = load_function(
                t_u32, indirect_pointer_source_root_lo_var);
            const uint32_t old_root_hi = load_function(
                t_u32, indirect_pointer_source_root_hi_var);
            record_index = sel(predicate, record_index, old_record);
            root_lo = sel(predicate, root_lo, old_root_lo);
            root_hi = sel(predicate, root_hi, old_root_hi);
        }
        // Preserve one atomic provenance tuple across EXEC masking. The source MUBUF's inactive
        // destination values may have been recycled since an earlier capture; they must not replace
        // only the root while the old record identity survives.
        store_function(indirect_pointer_source_record_var, record_index);
        store_function(indirect_pointer_source_root_lo_var, root_lo);
        store_function(indirect_pointer_source_root_hi_var, root_hi);
    }
    uint32_t relocated_indirect_carrier_dword(uint32_t selected_byte,
                                               uint32_t valid) {
        // OpSelect does not short-circuit an OpLoad. Select a known in-range carrier address before
        // either load, and avoid index+1 for an aligned dword at the physical end of the binding.
        const uint32_t safe_byte = sel(valid, selected_byte, uconst(0));
        const uint32_t index0 = ibin(Op_ShiftRightLogical, safe_byte, uconst(2));
        const uint32_t shift = ibin(
            Op_ShiftLeftLogical,
            ibin(Op_BitwiseAnd, safe_byte, uconst(3)), uconst(3));
        const uint32_t needs_second = ucmp(Op_INotEqual, shift, uconst(0));
        const uint32_t index1 = sel(
            needs_second, ibin(Op_IAdd, index0, uconst(1)), index0);
        const uint32_t dword0 = cbuf_load(index0, indirect_pointer_binding);
        const uint32_t dword1 = cbuf_load(index1, indirect_pointer_binding);
        const uint32_t lower = ibin(Op_ShiftRightLogical, dword0, shift);
        const uint32_t inverse_shift = ibin(
            Op_BitwiseAnd, ibin(Op_ISub, uconst(32), shift), uconst(31));
        const uint32_t upper = ibin(Op_ShiftLeftLogical, dword1, inverse_shift);
        const uint32_t joined = ibin(
            Op_BitwiseOr, lower, sel(needs_second, upper, uconst(0)));
        return sel(valid, joined, uconst(0));
    }
    // Relocate one proven guest GLOBAL dword through the version-2 carrier. Segment byte_count is
    // the exact guest interval; carrier_bytes includes only the physical zero padding needed for a
    // safe unaligned dword join. An invalid, ambiguous, overflowing, or padding-only address first
    // selects carrier byte zero for memory safety and then returns architectural zero.
    uint32_t relocated_indirect_load_dword(uint32_t address_lo, uint32_t address_hi,
                                            uint32_t immediate_byte_offset) {
        const U64PairAdd access_begin =
            add_u64_pair_u32(address_lo, address_hi, uconst(immediate_byte_offset));
        const U64PairAdd access_end =
            add_u64_pair_u32(access_begin.lo, access_begin.hi, uconst(sizeof(uint32_t)));
        uint32_t selected_byte = uconst(0);
        uint32_t match_count = uconst(0);
        const uint32_t directory_dword =
            indirect_pointer_segment_directory_byte_offset / sizeof(uint32_t);
        constexpr uint32_t kSegmentDwords =
            kIndirectBufferRelocationSegmentBytes / sizeof(uint32_t);
        for (uint32_t segment = 0; segment < indirect_pointer_segment_count; ++segment) {
            const uint32_t entry = directory_dword + segment * kSegmentDwords;
            const uint32_t guest_lo = cbuf_load(
                uconst(entry), indirect_pointer_binding);
            const uint32_t guest_hi = cbuf_load(
                uconst(entry + 1u), indirect_pointer_binding);
            const uint32_t byte_count = cbuf_load(
                uconst(entry + 2u), indirect_pointer_binding);
            const uint32_t packed_byte = cbuf_load(
                uconst(entry + 3u), indirect_pointer_binding);

            const U64PairAdd guest_end =
                add_u64_pair_u32(guest_lo, guest_hi, byte_count);
            uint32_t guest_contains = logical_not(access_begin.overflow);
            guest_contains = land(guest_contains, logical_not(access_end.overflow));
            guest_contains = land(guest_contains, logical_not(guest_end.overflow));
            guest_contains = land(
                guest_contains, ucmp(Op_INotEqual, byte_count, uconst(0)));
            guest_contains = land(
                guest_contains,
                u64_pair_ule(guest_lo, guest_hi, access_begin.lo, access_begin.hi));
            guest_contains = land(
                guest_contains,
                u64_pair_ule(access_end.lo, access_end.hi, guest_end.lo, guest_end.hi));

            // Once containment is true, the interval is at most UINT32_MAX bytes, so the low-word
            // subtraction is the exact residual even when the guest interval crosses 4 GiB.
            const uint32_t residual = ibin(Op_ISub, access_begin.lo, guest_lo);
            const uint32_t candidate = ibin(Op_IAdd, packed_byte, residual);
            const uint32_t candidate_wrapped = ucmp(Op_ULessThan, candidate, packed_byte);
            const uint32_t candidate_end =
                ibin(Op_IAdd, candidate, uconst(sizeof(uint32_t)));
            const uint32_t candidate_end_wrapped =
                ucmp(Op_ULessThan, candidate_end, candidate);
            const uint32_t packed_end = ibin(Op_IAdd, packed_byte, byte_count);
            const uint32_t packed_end_wrapped = ucmp(Op_ULessThan, packed_end, packed_byte);
            uint32_t packed_valid = logical_not(candidate_wrapped);
            packed_valid = land(packed_valid, logical_not(candidate_end_wrapped));
            packed_valid = land(packed_valid, logical_not(packed_end_wrapped));
            packed_valid = land(
                packed_valid,
                ucmp(Op_UGreaterThanEqual, packed_byte,
                     uconst(indirect_pointer_payload_byte_offset)));
            packed_valid = land(
                packed_valid,
                ucmp(Op_ULessThanEqual, packed_end,
                     uconst(indirect_pointer_carrier_bytes)));
            packed_valid = land(
                packed_valid, ucmp(Op_ULessThanEqual, candidate_end, packed_end));

            const uint32_t match = land(guest_contains, packed_valid);
            selected_byte = sel(match, candidate, selected_byte);
            match_count = ibin(
                Op_IAdd, match_count, sel(match, uconst(1), uconst(0)));
        }

        const uint32_t unique = ucmp(Op_IEqual, match_count, uconst(1));
        return relocated_indirect_carrier_dword(selected_byte, unique);
    }
    // DescriptorRange differs from StaticFootprint in one essential way: adjacent source records
    // may describe adjacent or overlapping guest intervals, but an address derived from record A
    // must never borrow record B's authority. Match the captured producer identity against one
    // record-directory entry first, prove the final dword lies in that exact record, and only then
    // translate it through the segment named by that record. All arithmetic is an explicit u32 pair;
    // this path intentionally does not require ShaderInt64.
    uint32_t relocated_indirect_descriptor_load_dword(
            uint32_t address_lo, uint32_t address_hi,
            uint32_t immediate_byte_offset) {
        const U64PairAdd access_begin =
            add_u64_pair_u32(address_lo, address_hi, uconst(immediate_byte_offset));
        const U64PairAdd access_end =
            add_u64_pair_u32(access_begin.lo, access_begin.hi, uconst(sizeof(uint32_t)));
        const uint32_t captured_record = load_function(
            t_u32, indirect_pointer_source_record_var);
        const uint32_t captured_root_lo = load_function(
            t_u32, indirect_pointer_source_root_lo_var);
        const uint32_t captured_root_hi = load_function(
            t_u32, indirect_pointer_source_root_hi_var);

        const uint32_t max_source_index =
            (UINT32_MAX - indirect_pointer_source_pointer_byte_offset) /
            indirect_pointer_source_stride;
        const uint32_t source_index_valid = ucmp(
            Op_ULessThanEqual, captured_record, uconst(max_source_index));
        const uint32_t expected_source_offset = ibin(
            Op_IAdd,
            ibin(Op_IMul, captured_record, uconst(indirect_pointer_source_stride)),
            uconst(indirect_pointer_source_pointer_byte_offset));

        uint32_t selected_byte = uconst(0);
        uint32_t matching_records = uconst(0);
        uint32_t selected_segment_valid = bfalse();
        const uint32_t records_dword =
            indirect_pointer_record_directory_byte_offset / sizeof(uint32_t);
        const uint32_t segments_dword =
            indirect_pointer_segment_directory_byte_offset / sizeof(uint32_t);
        constexpr uint32_t kRecordDwords =
            kIndirectBufferRelocationRecordBytes / sizeof(uint32_t);
        constexpr uint32_t kSegmentDwords =
            kIndirectBufferRelocationSegmentBytes / sizeof(uint32_t);
        for (uint32_t record = 0; record < indirect_pointer_record_count; ++record) {
            const uint32_t entry = records_dword + record * kRecordDwords;
            const uint32_t source_offset = cbuf_load(
                uconst(entry), indirect_pointer_binding);
            const uint32_t segment_index = cbuf_load(
                uconst(entry + 1u), indirect_pointer_binding);
            const uint32_t guest_lo = cbuf_load(
                uconst(entry + 2u), indirect_pointer_binding);
            const uint32_t guest_hi = cbuf_load(
                uconst(entry + 3u), indirect_pointer_binding);
            const uint32_t byte_count = cbuf_load(
                uconst(entry + 4u), indirect_pointer_binding);
            const uint32_t address_kind = cbuf_load(
                uconst(entry + 5u), indirect_pointer_binding);

            const U64PairAdd record_end = add_u64_pair_u32(
                guest_lo, guest_hi, byte_count);
            uint32_t record_contains = logical_not(access_begin.overflow);
            record_contains = land(record_contains, logical_not(access_end.overflow));
            record_contains = land(record_contains, logical_not(record_end.overflow));
            record_contains = land(
                record_contains, ucmp(Op_INotEqual, byte_count, uconst(0)));
            record_contains = land(
                record_contains,
                u64_pair_ule(guest_lo, guest_hi, access_begin.lo, access_begin.hi));
            record_contains = land(
                record_contains,
                u64_pair_ule(access_end.lo, access_end.hi, record_end.lo, record_end.hi));

            uint32_t record_match = source_index_valid;
            record_match = land(
                record_match,
                ucmp(Op_IEqual, source_offset, expected_source_offset));
            record_match = land(
                record_match, ucmp(Op_IEqual, guest_lo, captured_root_lo));
            record_match = land(
                record_match, ucmp(Op_IEqual, guest_hi, captured_root_hi));
            record_match = land(
                record_match,
                ucmp(Op_IEqual, address_kind,
                     uconst(static_cast<uint32_t>(
                         IndirectBufferRelocationRecord::SourceAddressKind::
                             BufferDescriptorBase48))));
            record_match = land(record_match, record_contains);
            matching_records = ibin(
                Op_IAdd, matching_records,
                sel(record_match, uconst(1), uconst(0)));

            const uint32_t segment_index_valid = ucmp(
                Op_ULessThan, segment_index, uconst(indirect_pointer_segment_count));
            const uint32_t safe_segment = sel(
                segment_index_valid, segment_index, uconst(0));
            const uint32_t segment_entry = ibin(
                Op_IAdd, uconst(segments_dword),
                ibin(Op_IMul, safe_segment, uconst(kSegmentDwords)));
            const uint32_t segment_guest_lo = cbuf_load(
                segment_entry, indirect_pointer_binding);
            const uint32_t segment_guest_hi = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(1)),
                indirect_pointer_binding);
            const uint32_t segment_bytes = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(2)),
                indirect_pointer_binding);
            const uint32_t packed_byte = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(3)),
                indirect_pointer_binding);
            const uint32_t reserved_lo = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(4)),
                indirect_pointer_binding);
            const uint32_t reserved_hi = cbuf_load(
                ibin(Op_IAdd, segment_entry, uconst(5)),
                indirect_pointer_binding);

            const U64PairAdd segment_end = add_u64_pair_u32(
                segment_guest_lo, segment_guest_hi, segment_bytes);
            uint32_t segment_valid = segment_index_valid;
            segment_valid = land(
                segment_valid, ucmp(Op_INotEqual, segment_bytes, uconst(0)));
            segment_valid = land(segment_valid, logical_not(segment_end.overflow));
            segment_valid = land(
                segment_valid,
                u64_pair_ule(segment_guest_lo, segment_guest_hi, guest_lo, guest_hi));
            segment_valid = land(
                segment_valid,
                u64_pair_ule(record_end.lo, record_end.hi,
                             segment_end.lo, segment_end.hi));
            segment_valid = land(
                segment_valid, ucmp(Op_IEqual, reserved_lo, uconst(0)));
            segment_valid = land(
                segment_valid, ucmp(Op_IEqual, reserved_hi, uconst(0)));

            // The record is contained by this segment, so low-word subtraction is the exact byte
            // residual. Validate the packed representation independently before selecting it.
            const uint32_t residual = ibin(
                Op_ISub, access_begin.lo, segment_guest_lo);
            const uint32_t candidate = ibin(Op_IAdd, packed_byte, residual);
            const uint32_t candidate_wrapped = ucmp(
                Op_ULessThan, candidate, packed_byte);
            const uint32_t candidate_end = ibin(
                Op_IAdd, candidate, uconst(sizeof(uint32_t)));
            const uint32_t candidate_end_wrapped = ucmp(
                Op_ULessThan, candidate_end, candidate);
            const uint32_t packed_end = ibin(
                Op_IAdd, packed_byte, segment_bytes);
            const uint32_t packed_end_wrapped = ucmp(
                Op_ULessThan, packed_end, packed_byte);
            segment_valid = land(segment_valid, logical_not(candidate_wrapped));
            segment_valid = land(segment_valid, logical_not(candidate_end_wrapped));
            segment_valid = land(segment_valid, logical_not(packed_end_wrapped));
            segment_valid = land(
                segment_valid,
                ucmp(Op_UGreaterThanEqual, packed_byte,
                     uconst(indirect_pointer_payload_byte_offset)));
            segment_valid = land(
                segment_valid,
                ucmp(Op_ULessThanEqual, packed_end,
                     uconst(indirect_pointer_carrier_bytes)));
            segment_valid = land(
                segment_valid,
                ucmp(Op_ULessThanEqual, candidate_end, packed_end));

            const uint32_t select_record = land(record_match, segment_valid);
            selected_byte = sel(select_record, candidate, selected_byte);
            selected_segment_valid = lor(selected_segment_valid, select_record);
        }

        const uint32_t unique = land(
            ucmp(Op_IEqual, matching_records, uconst(1)), selected_segment_valid);
        return relocated_indirect_carrier_dword(selected_byte, unique);
    }
    uint32_t cbuf_load_zero_padded_tail(uint32_t binding,
                                        StorageBufferTailSemantic semantic,
                                        bool coherent_access = false) {
        auto [candidate, inserted] = cbuf_zero_pad_candidates.emplace(binding, semantic);
        if (!inserted && candidate->second != semantic)
            cbuf_ordinary_accesses.insert(binding);
        return cbuf_load_impl(uconst(0), binding, coherent_access);
    }
    // Store one dword `value` to the buffer at descriptor `binding` at dword index `idx` (MUBUF store).
    // When `predicated`, the store is wrapped in a selection merge on `pred` (the per-lane EXEC bool) so
    // inactive lanes do not write — a real conditional store, not a select of a loaded old value.
    void cbuf_store(uint32_t idx, uint32_t value, uint32_t binding, bool predicated,
                    uint32_t pred, bool coherent_access = false) {
        cbuf_ordinary_accesses.insert(binding);
        if (cbuf_table_arity.count(binding)) invalid_cbuf_array_access = true;
        uint32_t buf = buf_for_binding(binding);
        if (coherent_access) mark_cbuf_coherent(binding);
        auto emit = [&]() {
            uint32_t p = cbuf_element_ptr(buf, binding, idx);
            if (coherent_access) put(code, Op_Store, {p, value, MemAccess_Volatile});
            else                 put(code, Op_Store, {p, value});
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
    // Returning atomic RMW on a descriptor-backed storage buffer. BUFFER_ATOMIC_* writes memory and
    // returns the pre-operation value in VDATA. Inactive EXEC lanes neither access the buffer nor
    // clobber VDATA, hence the predicated path joins the old destination through OpPhi.
    uint32_t cbuf_atomic_rtn(uint32_t op, uint32_t idx, uint32_t value, uint32_t binding,
                             bool predicated, uint32_t pred, uint32_t fallback) {
        cbuf_ordinary_accesses.insert(binding);
        if (cbuf_table_arity.count(binding)) invalid_cbuf_array_access = true;
        const uint32_t buf = buf_for_binding(binding);
        auto emit = [&]() {
            uint32_t p = cbuf_element_ptr(buf, binding, idx);
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
    // RDNA2 BUFFER_ATOMIC_FMIN/FMAX and DS_MIN/MAX_F32 use floating min/max bit semantics, but
    // operate on memory for which SPIR-V 1.3 has no core floating-point min/max atomic. Keep the CAS
    // loop and all selection arithmetic in integer space so host-driver float controls cannot alter
    // the guest result. Signaling NaNs are quieted and propagated before ordinary MINNUM/MAXNUM
    // selection. COMPUTE_PGM_RSRC1.FP32_DENORM controls whether subnormal operands compare as
    // signed zero. MIN/MAX still publishes the selected operand's original, non-flushed bits.
    uint32_t atomic_fminmax_bits(uint32_t resident, uint32_t value, bool is_min) {
        const uint32_t denorm_mode =
            (compute_pgm_rsrc1 >>
             prosper::agc::Pm4::COMPUTE_PGM_RSRC1_FP32_DENORM_SHIFT) &
            prosper::agc::Pm4::COMPUTE_PGM_RSRC1_FP32_DENORM_MASK;
        const bool flush_inputs = denorm_mode == 0u || denorm_mode == 2u;
        auto flush_subnormal = [&](uint32_t bits) {
            const uint32_t absolute = ibin(Op_BitwiseAnd, bits, uconst(0x7fffffffu));
            const uint32_t nonzero = ucmp(Op_INotEqual, absolute, uconst(0));
            const uint32_t below_normal = ucmp(Op_ULessThan, absolute, uconst(0x00800000u));
            const uint32_t subnormal = land(nonzero, below_normal);
            const uint32_t signed_zero = ibin(Op_BitwiseAnd, bits, uconst(0x80000000u));
            return sel(subnormal, signed_zero, bits);
        };
        const uint32_t resident_compare = flush_inputs ? flush_subnormal(resident) : resident;
        const uint32_t value_compare = flush_inputs ? flush_subnormal(value) : value;
        auto ordered_key = [&](uint32_t bits) {
            const uint32_t negative = ucmp(
                Op_INotEqual, ibin(Op_BitwiseAnd, bits, uconst(0x80000000u)), uconst(0));
            return sel(negative, iun(Op_Not, bits),
                       ibin(Op_BitwiseXor, bits, uconst(0x80000000u)));
        };
        const uint32_t resident_abs = ibin(
            Op_BitwiseAnd, resident, uconst(0x7fffffffu));
        const uint32_t value_abs = ibin(
            Op_BitwiseAnd, value, uconst(0x7fffffffu));
        const uint32_t resident_nan = ucmp(
            Op_UGreaterThan, resident_abs, uconst(0x7f800000u));
        const uint32_t value_nan = ucmp(
            Op_UGreaterThan, value_abs, uconst(0x7f800000u));
        const uint32_t resident_snan = land(
            resident_nan,
            ucmp(Op_IEqual,
                 ibin(Op_BitwiseAnd, resident, uconst(0x00400000u)), uconst(0)));
        const uint32_t value_snan = land(
            value_nan,
            ucmp(Op_IEqual,
                 ibin(Op_BitwiseAnd, value, uconst(0x00400000u)), uconst(0)));
        const uint32_t ordered = ucmp(
            is_min ? Op_ULessThan : Op_UGreaterThan,
            ordered_key(value_compare), ordered_key(resident_compare));
        const uint32_t numeric = sel(ordered, value, resident);
        const uint32_t resident_number = sel(value_nan, resident, numeric);
        const uint32_t quiet_resident = ibin(
            Op_BitwiseOr, resident, uconst(0x00400000u));
        const uint32_t quiet_value = ibin(
            Op_BitwiseOr, value, uconst(0x00400000u));
        return sel(
            resident_snan, quiet_resident,
            sel(value_snan, quiet_value,
                sel(resident_nan, sel(value_nan, quiet_resident, value), resident_number)));
    }
    uint32_t cbuf_atomic_fminmax_rtn(uint32_t idx, uint32_t value, uint32_t binding,
                                     bool is_min, bool predicated, uint32_t pred,
                                     uint32_t fallback) {
        cbuf_ordinary_accesses.insert(binding);
        if (cbuf_table_arity.count(binding)) invalid_cbuf_array_access = true;
        const uint32_t buf = buf_for_binding(binding);
        auto emit = [&]() {
            const uint32_t pointer = cbuf_element_ptr(buf, binding, idx);
            const uint32_t initial = id();
            put(code, Op_AtomicLoad,
                {t_u32, initial, pointer, uconst(Scope_Device),
                 uconst(MemSem_UniformAcquire)});

            const uint32_t preheader = cur_block;
            const uint32_t header = id(), again = id(), merge = id();
            put(code, Op_Branch, {header});
            put(code, Op_Label, {header}); cur_block = header;
            size_t retry_patch = 0;
            const uint32_t expected = emit_phi2(t_u32, initial, preheader, retry_patch);
            const uint32_t desired = atomic_fminmax_bits(expected, value, is_min);
            const uint32_t observed = id();
            put(code, Op_AtomicCompareExchange,
                {t_u32, observed, pointer, uconst(Scope_Device),
                 uconst(MemSem_UniformAcqRel), uconst(MemSem_UniformAcquire),
                 desired, expected});
            const uint32_t succeeded = ucmp(Op_IEqual, observed, expected);
            put(code, Op_LoopMerge, {merge, again, 0});
            put(code, Op_BranchConditional, {succeeded, merge, again});
            put(code, Op_Label, {again}); cur_block = again;
            put(code, Op_Branch, {header});
            patch_phi(retry_patch, observed, again);
            put(code, Op_Label, {merge}); cur_block = merge;
            return expected;
        };
        if (!predicated) return emit();
        const uint32_t entry = cur_block;
        const uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = emit();
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u32, result, then_end, fallback, entry);
    }
    // One true 64-bit RMW for the exact naturally-strided guest contract. The caller supplies the
    // SPIR-V atomic opcode and original qword record index, and range-checks it before entry.
    uint32_t cbuf_atomic_x2_rtn(uint32_t op, uint32_t index, uint32_t value, uint32_t binding,
                                uint32_t pred, uint32_t fallback) {
        cbuf_ordinary_accesses.insert(binding);
        if (!declared_int64_atomics) {
            put(caps, Op_Capability, {Cap_Int64Atomics});
            declared_int64_atomics = true;
        }
        const uint32_t buf = u64_buf_for_binding(binding);
        auto emit = [&]() {
            const uint32_t pointer = id();
            putv(code, Op_AccessChain,
                 {t_ptr_sb_u64, pointer, buf, uconst(0), index});
            const uint32_t result = id();
            put(code, op,
                {t_u64(), result, pointer, uconst(Scope_Device),
                 uconst(MemSem_UniformAcqRel), value});
            return result;
        };
        const uint32_t entry = cur_block;
        const uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = emit();
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u64(), result, then_end, fallback, entry);
    }
    // RADV currently hangs/reset-poisons the device on Astro Bot's compute R32_UINT image atomic.
    // Compute lowers that exact 2D resource through a detiled storage-buffer view instead. The live
    // backend recognizes the reflected atomic buffer over a StorageImage resource, detiles before
    // dispatch, and tiles the result back afterwards. Graphics retains the native image-atomic path.
    // Bindings this function itself declared, so a REPEAT use of one is a hit rather than a
    // failure. A shader with two image atomics on the same image called here twice: the first
    // declared cbuf_var[binding], and the second was refused by the `cbuf_var.count(binding)` guard
    // -- which rejected the whole shader for the crime of using its own image twice. Measured on
    // Sonic Racing: CrossWorlds, where the resource table has NO entry at the binding in question,
    // so declare_cbufs cannot have been the declarer (#2265).
    //
    // Only OUR bindings are reusable. A binding declared by declare_cbufs from a
    // ConstantBuffer/VertexBuffer resource is still refused: the types happen to match, which is
    // exactly what would make aliasing onto someone else's buffer silent rather than loud.
    std::set<uint32_t> atomic_img_buf_bindings;
    bool declare_compute_atomic_image_buffer(uint32_t binding) {
        if (!is_compute || !t_ptr_sb_struct_u) return false;
        if (atomic_img_buf_bindings.count(binding)) return true;   // ours already; reuse it
        if (cbuf_var.count(binding)) return false;                 // someone else's; refuse
        const uint32_t variable = id();
        put(deco, Op_Decorate, {variable, Dec_DescriptorSet, desc_set});
        put(deco, Op_Decorate, {variable, Dec_Binding, binding});
        declare_external_storage_buffer(t_ptr_sb_struct_u, variable);
        cbuf_var[binding] = variable;
        atomic_img_buf_bindings.insert(binding);
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
    // EXEC-predicated LDS load. The inactive arm must not merely discard a loaded value: its
    // address VGPR retains the old, potentially arbitrary bits when the instruction is masked off,
    // so even forming an AccessChain/OpLoad there can access outside the Workgroup array or race an
    // active lane. Keep the memory access inside the active selection and join the architectural
    // old destination through OpPhi, matching the returning-atomic helper below.
    uint32_t lds_load(uint32_t idx, bool predicated, uint32_t pred, uint32_t fallback) {
        if (!predicated) return lds_load(idx);
        const uint32_t entry = cur_block;
        const uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = lds_load(idx);
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u32, result, then_end, fallback, entry);
    }
    // Store to LDS[idx]; EXEC-predicated (conditional store) under a narrowed mask, like cbuf_store.
    void lds_store(uint32_t idx, uint32_t value, bool predicated, uint32_t pred,
                   bool atomicize = false) {
        if (atomicize) {
            lds_atomic(Op_AtomicExchange, idx, value, predicated, pred);
            return;
        }
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
    // RDNA2 DS_MIN/MAX_F32 quiet and propagate signaling NaNs before numeric selection; a lone quiet
    // NaN yields the number. FP32_DENORM controls whether subnormal inputs compare at their true
    // value or as signed zero, while signed-zero ties choose -0 for min / +0 for max.
    // SPIR-V 1.3 has no core floating atomic min/max, while integer AtomicS/UMin orders the raw bits
    // incorrectly. Implement the architectural operation as a u32 compare-exchange loop instead.
    // The ordering key is monotonic for every non-NaN binary32 bit pattern and therefore does not
    // depend on the host driver's denormal mode. AMD's both-qNaN selection returns the first operand;
    // the resident `*ptr` value is that first operand for an atomic min/max update.
    void lds_atomic_fminmax(uint32_t idx, uint32_t value, bool is_min,
                            bool predicated, uint32_t pred) {
        auto emit = [&]() {
            const uint32_t p = id();
            putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
            const uint32_t initial = id();
            put(code, Op_AtomicLoad,
                {t_u32, initial, p, uconst(Scope_Workgroup), uconst(MemSem_WGAcquire)});

            const uint32_t preheader = cur_block;
            const uint32_t header = id(), again = id(), merge = id();
            put(code, Op_Branch, {header});
            put(code, Op_Label, {header}); cur_block = header;
            size_t retry_patch = 0;
            const uint32_t expected = emit_phi2(t_u32, initial, preheader, retry_patch);
            const uint32_t desired = atomic_fminmax_bits(expected, value, is_min);
            const uint32_t observed = id();
            put(code, Op_AtomicCompareExchange,
                {t_u32, observed, p, uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel),
                 uconst(MemSem_WGAcquire), desired, expected});
            const uint32_t succeeded = ucmp(Op_IEqual, observed, expected);
            put(code, Op_LoopMerge, {merge, again, 0});
            put(code, Op_BranchConditional, {succeeded, merge, again});
            put(code, Op_Label, {again}); cur_block = again;
            put(code, Op_Branch, {header});
            patch_phi(retry_patch, observed, again);
            put(code, Op_Label, {merge}); cur_block = merge;
        };
        if (!predicated) { emit(); return; }
        const uint32_t then = id(), merge = id();
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
    uint32_t cfg_scratch = 0, t_ptr_cfg_u32 = 0, cfg_scratch_dwords = 0;
    bool declare_cfg_scratch(uint32_t dwords) {
        // OpTypeArray is immutable once emitted. A later phased dispatcher may require a wider
        // layout than the first phase; fail closed if its caller did not pre-size from the complete
        // stream instead of emitting an out-of-bounds Workgroup access.
        if (cfg_scratch) return dwords <= cfg_scratch_dwords;
        uint32_t t_arr = id(); put(types, Op_TypeArray, {t_arr, t_u32, uconst(dwords)});
        uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_Workgroup, t_arr});
        cfg_scratch = id(); put(types, Op_Variable, {t_ptr, cfg_scratch, SC_Workgroup});
        t_ptr_cfg_u32 = id(); put(types, Op_TypePointer, {t_ptr_cfg_u32, SC_Workgroup, t_u32});
        cfg_scratch_dwords = dwords;
        return true;
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
    // V_READLANE_B32 at a workgroup-uniform site. Publish every invocation's source value and
    // address the selected lane within this invocation's own guest wave. This is exact at any
    // host subgroup width; unlike a native subgroup shuffle it does not require a Wave64 guest
    // wave to fit inside one Vulkan subgroup. V_READLANE ignores EXEC, so publication is
    // unconditional. Missing lanes in a partial final wave read as zero.
    uint32_t guest_wave_readlane(uint32_t source_value, uint32_t selector) {
        declare_wave_lds();
        uint32_t source_ptr = id();
        putv(code, Op_AccessChain,
             {t_ptr_wg_u32b, source_ptr, lds_wave, linear_localid});
        put(code, Op_Store, {source_ptr, source_value});
        barrier();

        const uint32_t wave_shift = wave_size == 32 ? 5u : 6u;
        const uint32_t wave_base = ibin(
            Op_ShiftLeftLogical,
            ibin(Op_ShiftRightLogical, linear_localid, uconst(wave_shift)),
            uconst(wave_shift));
        const uint32_t lane = ibin(
            Op_BitwiseAnd, selector, uconst(wave_size - 1u));
        const uint32_t index = ibin(Op_IAdd, wave_base, lane);
        const uint32_t zero = uconst(0);
        const uint32_t valid = ucmp(Op_ULessThan, index, uconst(local_count));
        const uint32_t safe_index = sel(valid, index, zero);
        uint32_t result_ptr = id();
        putv(code, Op_AccessChain,
             {t_ptr_wg_u32b, result_ptr, lds_wave, safe_index});
        uint32_t result = id();
        put(code, Op_Load, {t_u32, result, result_ptr});
        barrier();
        return sel(valid, result, zero);
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
        if (rt)
            for (const ShaderResource& resource : rt->resources)
                if (resource.table_index_count != 0u &&
                    (resource.cls == ResourceClass::ConstantBuffer ||
                     resource.cls == ResourceClass::VertexBuffer) &&
                    !valid_shader_buffer_table_contract(resource))
                    invalid_cbuf_array_access = true;
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
        // Bindings 2 and 3 are declared here rather than in the loop below, so a TABLE-INDEXED resource
        // on either of them has to be handled here too (#2472). Before this, both were declared
        // unconditionally as scalar buffers and then seeded into the loop's `seen` set, so an array at
        // binding 2 or 3 was skipped and silently emitted as a single descriptor -- and since Vulkan
        // permits a shader declaring one descriptor against a layout declaring N, that produced no
        // error: the shader read element 0 for every index. Titles put their constant buffers on
        // exactly these two bindings, so it was the common case that failed quietly.
        // Both helpers below must select the SAME resource, so the predicate lives in one place. Two
        // resources may share a binding (`declare_cbufs` does not prevent it), so a class filter on one
        // helper and not the other can take the arity from one and the index SGPR from the other —
        // wrong descriptor index, and quiet, because both values are individually plausible.
        auto table_indexed_here = [](const ShaderResource& r, uint32_t binding) {
            return r.binding == binding && r.table_index_count != 0 &&
                   (r.cls == ResourceClass::ConstantBuffer || r.cls == ResourceClass::VertexBuffer);
        };
        auto table_arity_for = [&](uint32_t binding) -> uint32_t {
            if (!rt) return 0;
            for (const auto& r : rt->resources)
                if (table_indexed_here(r, binding)) return r.table_index_count;
            return 0;
        };
        auto index_sgpr_for = [&](uint32_t binding) -> uint32_t {
            if (!rt) return 0xFFFFFFFFu;
            for (const auto& r : rt->resources)
                if (table_indexed_here(r, binding)) return r.table_index_sgpr;
            return 0xFFFFFFFFu;
        };
        // Declare `binding`'s variable as an array of the Block type, reusing the id already decorated
        // for it, so its set/binding decorations above stay correct.
        auto declare_as_array = [&](uint32_t binding, uint32_t var, uint32_t arity) {
            declare_descriptor_indexing();
            const uint32_t arr = id();
            put(types, Op_TypeArray,
                {arr, t_struct_u, uconst(arity <= 4096u ? arity : 1u)});
            const uint32_t arr_ptr = id();
            put(types, Op_TypePointer, {arr_ptr, SC_StorageBuffer, arr});
            declare_external_storage_buffer(arr_ptr, var);
            cbuf_table_arity[binding] = arity;
            const uint32_t sgpr = index_sgpr_for(binding);
            if (sgpr != 0xFFFFFFFFu) cbuf_table_index_sgpr[binding] = sgpr;
        };
        const uint32_t arity2 = table_arity_for(2), arity3 = table_arity_for(3);
        // NOTE on the fallback: `buf_for_binding` returns `v_cbuf` for a binding it does not know. If
        // binding 2 is an array, that fallback yields an array-typed variable, and an access chain built
        // for a scalar pointee against it is malformed SPIR-V. It is NOT caught at emission: `put`/`putv`
        // (:383, :386) are a word assembler — they push a length word and the raw ids and type-check
        // nothing — so the module is emitted happily. Nor is it caught by `spirv-val`, which gates one
        // representative module per emitter path and has none for this case, since the case has never
        // been constructed. What actually happens is that pipeline creation rejects the module and the
        // draw is dropped. Both branches are hard errors rather than wrong pixels -- index 0 selects a
        // Block, so index 1 indexes into a structure, and SPIR-V requires a struct member index to be an
        // OpConstant, which a computed index is not; in the only constant case it yields a
        // pointer-to-runtime-array against a declared `t_ptr_sb_u32`, a type mismatch. So the direction
        // is still fail-visible, not silently wrong -- but "loud" here means a dropped draw, not a
        // validator, and nothing reports it as a type error at the point the mistake was made.
        if (arity2) declare_as_array(2, v_cbuf, arity2);
        else        declare_external_storage_buffer(t_ptr_sb_struct_u, v_cbuf);
        if (arity3) declare_as_array(3, v_cbuf1, arity3);
        else        declare_external_storage_buffer(t_ptr_sb_struct_u, v_cbuf1);
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
                if (r.table_index_count != 0) {
                    // TABLE-INDEXED binding (#2412 stage 4b): the descriptor is one OF an array,
                    // selected by an index only the GPU knows. The pointee becomes an array of the same
                    // Block type rather than the Block itself, so this binding occupies
                    // `table_index_count` descriptors and an access chain needs a leading index.
                    //
                    // A count of 0 never reaches here -- it means "not table-indexed".
                    //
                    // An IMPLAUSIBLE length emits a runtime array rather than a fixed array of that
                    // length. This is the defensive path for a length that reflection could not read:
                    // #2463 gives such a length its own sentinel (UINT32_MAX), and emitting
                    // `OpTypeArray %Block 4294967295` would ask the driver for a 4-billion-descriptor
                    // binding. The numeric guard catches that sentinel without this branch having to
                    // depend on #2463 being merged first, and catches any other absurd count on the way.
                    declare_descriptor_indexing();
                    const uint32_t arr = id();
                    put(types, Op_TypeArray,
                        {arr, t_struct_u,
                         uconst(r.table_index_count <= 4096u ? r.table_index_count : 1u)});
                    const uint32_t arr_ptr = id();
                    put(types, Op_TypePointer, {arr_ptr, SC_StorageBuffer, arr});
                    declare_external_storage_buffer(arr_ptr, v);
                    cbuf_var[r.binding] = v;
                    cbuf_table_arity[r.binding] = r.table_index_count;
                    if (r.table_index_sgpr != 0xFFFFFFFFu)
                        cbuf_table_index_sgpr[r.binding] = r.table_index_sgpr;
                    continue;
                }
                declare_external_storage_buffer(t_ptr_sb_struct_u, v);
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
    uint32_t land(uint32_t a, uint32_t b_) {
        uint32_t r = id();
        put(code, Op_LogicalAnd, {t_bool, r, a, b_});
        propagate_fragment_wave_vote(r, a, b_);
        return r;
    }
    uint32_t lor(uint32_t a, uint32_t b_) {
        uint32_t r = id();
        put(code, Op_LogicalOr, {t_bool, r, a, b_});
        propagate_fragment_wave_vote(r, a, b_);
        return r;
    }

    void begin(uint32_t input_stride, const ShaderResourceTable* rt = nullptr,
               uint32_t local_x = 64, uint32_t local_y = 1, uint32_t local_z = 1,
               uint32_t hardware_wave_size = 64, uint32_t push_constant_dwords = 0) {
        stride = input_stride;
        push_constant_dword_count = push_constant_dwords;
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
        declare_external_storage_buffer(t_ptr_sb_struct, v_in);
        declare_external_storage_buffer(t_ptr_sb_struct, v_out);
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
    uint32_t invocation_within_extent(uint32_t threads_x, uint32_t threads_y,
                                      uint32_t threads_z) {
        uint32_t within = ucmp(Op_ULessThan, globalid_comp[0], uconst(threads_x));
        within = land(within, ucmp(Op_ULessThan, globalid_comp[1], uconst(threads_y)));
        within = land(within, ucmp(Op_ULessThan, globalid_comp[2], uconst(threads_z)));
        return within;
    }
    void guard_invocation_extent(uint32_t threads_x, uint32_t threads_y, uint32_t threads_z) {
        const uint32_t within = invocation_within_extent(threads_x, threads_y, threads_z);
        const uint32_t active = id();
        invocation_guard_merge = id();
        emit_selmerge(invocation_guard_merge);
        emit_condbranch(within, active, invocation_guard_merge);
        emit_label(active);
    }
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

    // Fragment sample-coverage export (EXP target 8 / VSRC2). Vulkan exposes this as the first
    // element of a BuiltIn SampleMask output array, matching RDNA's 32-bit sample-mask payload.
    uint32_t v_sample_mask = 0;
    uint32_t t_sample_mask = 0;
    void export_sample_mask(uint32_t mask_bits) {
        if (!v_sample_mask) {
            t_sample_mask = id();
            put(types, Op_TypeArray, {t_sample_mask, t_u32, uconst(1)});
            uint32_t t_ptr = id();
            put(types, Op_TypePointer, {t_ptr, SC_Output, t_sample_mask});
            v_sample_mask = id();
            put(types, Op_Variable, {t_ptr, v_sample_mask, SC_Output});
            put(deco, Op_Decorate, {v_sample_mask, Dec_BuiltIn, BI_SampleMask});
            iface.push_back(v_sample_mask);
        }
        const uint32_t value = id();
        put(code, Op_CompositeConstruct, {t_sample_mask, value, mask_bits});
        put(code, Op_Store, {v_sample_mask, value});
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
        else {
            // A tap was REQUESTED and is not available here, which is the silent-degradation case
            // (#2064): tap_vec is set only when the instruction at tap_pc is walked, so a tap_pc
            // AFTER this shader's EXP POS0 leaves it 0 and the real clip position is exported --
            // while the readout still prints "values below are the tapped VGPR". A plausible,
            // well-labelled, completely wrong answer.
            //
            // Warned rather than rejected, and the distinction matters: tap_pc is GLOBAL across
            // stages, so a PC tapped in the pixel shader is legitimately absent from every vertex
            // shader. Refusing here would drop every draw in the frame for a tap that is doing
            // exactly what it was asked to do. The warning names both PCs so a reader can tell the
            // two apart immediately -- "after the export" is the defect, "not in this shader" is not.
            if (tap_pc != 0xFFFFFFFFu) {
                // Atomic, not a plain counter: parallel_draw_worker_execute realizes draws on
                // worker threads and realization recompiles, and I could not establish that
                // the recompile itself runs under ShaderCache's mutex rather than only the
                // lookup. An unverified serialization claim is not worth one relaxed add.
                static std::atomic<unsigned> warned{0};
                if (warned.fetch_add(1, std::memory_order_relaxed) < 8)
                    fprintf(stderr,
                            "[shader-tap] NOT APPLIED at the position export: PROSPER_SHADER_TAP "
                            "pc=%u was not reached before EXP POS0 in this vertex shader, so the "
                            "REAL clip position is exported. If pc=%u is after the export, move it "
                            "earlier; if it belongs to another stage, this line is expected (#2064)\n",
                            tap_pc, tap_pc);
            }
            v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(x), bcf(y), bcf(z), bcf(w)});
        }
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
    // FS: an Input vec4 at Location=attr, rasterizer-interpolated (or Flat if flat_attrs says so, OR
    // if the guest's own PS_INPUT_CNTL declared FLAT_SHADE for this slot -- #3051. Both sources feed
    // the same SPIR-V property and are unioned: flat_attrs recognises flat-ness from the shader's own
    // choice of VINTRP opcode (v_interp_mov), while fragment_interpolation->flat_mask recognises it
    // from the guest's declared semantic, which still applies to an ordinary v_interp_p1/p2 read).
    uint32_t frag_input(uint32_t attr) {
        auto it = in_varying.find(attr); if (it != in_varying.end()) return it->second;
        if (!t_ptr_in_v4f) { t_ptr_in_v4f = id(); put(types, Op_TypePointer, {t_ptr_in_v4f, SC_Input, t_v4f}); }
        uint32_t v = id(); put(types, Op_Variable, {t_ptr_in_v4f, v, SC_Input});
        put(deco, Op_Decorate, {v, Dec_Location, attr});
        const bool flat = flat_attrs.count(attr) != 0 ||
            (fragment_interpolation && attr < 32 &&
             (fragment_interpolation->flat_mask & (1u << attr)) != 0);
        if (flat) put(deco, Op_Decorate, {v, Dec_Flat});
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
    // extension. Input assembly has already decomposed lists/strips/fans into triangles here. A
    // three-vertex PS5 RectList whose attributes come from a vertex buffer also comes through this
    // stage: GFX10 synthesizes its fourth corner after vertex shading, while Vulkan has no RectList
    // topology. The missing post-VS position and every consumed varying are the affine fourth
    // corner P1 + P2 - P0; emitting P0,P1,P2,P3 as a strip preserves the hardware rectangle.
    std::vector<uint32_t> build_interpolation_geometry(
            const FragmentInterpolationLayout& layout, bool capture_geometry_position,
            bool synthesize_rect = false) {
        if ((!layout.requires_geometry && !synthesize_rect) || !layout.valid) return {};

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
            if ((layout.smooth_mask & (1u << attr)) || synthesize_rect)
                attribute_outputs[attr] = id();
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
        put(exec, Op_ExecutionMode, {f_main, EM_OutputVertices, synthesize_rect ? 4u : 3u});
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
        std::array<uint32_t, 32> rect_attribute_values{};
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
            if (synthesize_rect) {
                const uint32_t sum = id();
                put(code, Op_FAdd, {t_v4f, sum,
                                    attribute_values[attr][1], attribute_values[attr][2]});
                rect_attribute_values[attr] = id();
                put(code, Op_FSub, {t_v4f, rect_attribute_values[attr],
                                    sum, attribute_values[attr][0]});
            }
        }

        std::array<uint32_t, 3> positions{};
        for (uint32_t vertex = 0; vertex < 3; ++vertex) {
            const uint32_t input_pointer = id();
            put(code, Op_AccessChain,
                {ptr_in_v4f, input_pointer, input_position, uconst(vertex), uconst(0)});
            positions[vertex] = id();
            put(code, Op_Load, {t_v4f, positions[vertex], input_pointer});
        }
        uint32_t rect_position = 0;
        if (synthesize_rect) {
            const uint32_t sum = id();
            put(code, Op_FAdd, {t_v4f, sum, positions[1], positions[2]});
            rect_position = id();
            put(code, Op_FSub, {t_v4f, rect_position, sum, positions[0]});
        }

        const uint32_t output_vertices = synthesize_rect ? 4u : 3u;
        for (uint32_t vertex = 0; vertex < output_vertices; ++vertex) {
            const uint32_t position = vertex < 3 ? positions[vertex] : rect_position;
            uint32_t output_pointer = id();
            put(code, Op_AccessChain,
                {ptr_out_v4f, output_pointer, output_position, uconst(0)});
            put(code, Op_Store, {output_pointer, position});

            for (uint32_t attr = 0; attr < 32; ++attr) {
                if (attribute_outputs[attr])
                    put(code, Op_Store,
                        {attribute_outputs[attr], vertex < 3
                            ? attribute_values[attr][vertex] : rect_attribute_values[attr]});
                for (uint32_t selector = 0; selector < 3; ++selector)
                    if (parameter_outputs[attr][selector])
                        put(code, Op_Store,
                            {parameter_outputs[attr][selector], parameters[attr][selector]});
            }
            const float i = vertex == 1 || vertex == 3 ? 1.0f : 0.0f;
            const float j = vertex == 2 || vertex == 3 ? 1.0f : 0.0f;
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
        if (invalid_cbuf_array_access) return {};
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
            // A SEPARATE marker, not a wider one: a module cached or captured before #2147
            // carries the size and not this, and a reader must be able to tell 'reasons
            // unknown' from 'reasons none'. Absent and zero are the same number and opposite
            // facts -- a zero would assert that nothing required a width this module demands.
            std::snprintf(marker, sizeof marker, "Prosper.FragmentSubgroupWhy=%u",
                          fragment_wave_reasons);
            words.clear();
            pstr(words, marker);
            putv(debug, Op_ModuleProcessed, words);
        }
        for (const auto& [binding, semantic] : cbuf_zero_pad_candidates) {
            if (cbuf_ordinary_accesses.count(binding)) continue;
            char marker[96];
            const char* token = semantic == StorageBufferTailSemantic::Uint16 ? "u16" :
                                semantic == StorageBufferTailSemantic::Float16 ? "f16" : nullptr;
            if (!token) continue;
            std::snprintf(marker, sizeof marker, "Prosper.StorageBufferZeroPad=%u,%u,2,4,%s",
                          desc_set, binding, token);
            std::vector<uint32_t> words;
            pstr(words, marker);
            putv(debug, Op_ModuleProcessed, words);
        }
        std::vector<uint32_t> m{0x07230203u, 0x00010300u, 0u, next_id, 0u};
        for (auto* s : {&caps, &exts, &extimp, &mem, &entry, &exec, &debug, &deco, &types, &code})
            m.insert(m.end(), s->begin(), s->end());
        return m;
    }
};

// Machine state during recompilation: the VGPR and SGPR files (VGPR/SGPR number -> current SSA bits
// id) and VCC (current bool condition). VGPRs and SGPRs are separate register files; VALU/EXP source
// operands may reference either (SGPR is a valid ALU operand), so both are resolved by operand_bits.
struct RegState {
    std::unordered_map<int, uint32_t> vreg, sreg;
    // The latest VCC SSA value proved identical in every guest lane. A fragment VCCZ branch over
    // that exact value is already scalar and needs no subgroup vote. Tying the proof to the SSA id
    // makes an overwrite or control-flow merge invalidate it automatically.
    uint32_t vcc_wave_uniform = 0;
    // Before the first compact structured construct, map presence means that the scalar value
    // reaches this exact linear path. Branch/loop PHIs can synthesize zero for an absent SGPR and
    // clear this marker. The CFG dispatcher likewise allocates Function variables for every
    // statically observed SGPR and may load a zero placeholder where no scalar lifetime reaches the
    // block. Keep that distinction explicit so a narrow prefix proof cannot weaken either route.
    bool scalar_presence_has_no_placeholders = true;
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
    // One restored physical dword of a Wave64 mask is not a complete B64 predicate. Keep its Bool
    // value out of sreg_bool so an overlapping S_MOV_B64/logical cannot ignore the sibling SGPR.
    // Only exact CFG-proven V_WRITELANE aliases consume this domain; ordinary scalar consumers use
    // the separately materialized ballot dword in sreg.
    std::unordered_map<int, uint32_t> sreg_wave64_mask_half;
    std::unordered_map<int, uint32_t> sreg_wave64_mask_half_index;
    // Wave32 saves occupy exactly one physical SGPR. Track those lifetimes separately so a later
    // scalar-data write can invalidate the bool without also clobbering an unrelated neighbor.
    std::unordered_set<int> sreg_bool_b32;
    std::unordered_map<int, uint32_t> sreg_srt;    // SGPR holding a descriptor -> its user_data/SRT byte offset
                                                   // (descriptor provenance: s_load_dwordx4 tags, s_buffer_load resolves)
    // The DIRECT counterpart of sreg_srt: SGPR word -> the entry-time SGPR word it is an unmodified
    // copy of. An indirect descriptor keeps its provenance across a copy because sreg_srt travels
    // with it; a direct descriptor is keyed only by its REGISTER NUMBER (`by_sgpr_base`), so staging
    // it into a scratch range with s_mov_b32 used to destroy its only key -- the moves make
    // `sreg_range_written` true, which is exactly the guard that suppresses the direct lookup. The
    // compiler idiom that does this is "stage several descriptors, copy the selected one into one
    // SRSRC range" (#1773; #273 added sreg_srt propagation because the INDIRECT form of the same
    // idiom broke DOLL's scene VS).
    //
    // This is a copy alias, not a claim about the descriptor's contents: it says only "these bits
    // are still the bits the driver put in that entry-time register".
    //
    // The load-bearing condition is applied where the alias is ESTABLISHED -- the source must still
    // have been entry-time user data at the moment of the copy. It is deliberately NOT re-checked at
    // consumption: a copy captures bits, so a later write to the SOURCE cannot change what the
    // DESTINATION holds, and re-checking would decline the exact shape #1773 documents (the shader
    // reuses five of the origin words for its second descriptor before the sample). The alias dies
    // when the DESTINATION is written, which record_scalar_write does centrally.
    //
    // Because it is a claim about one register's bits, it is a PER-PATH fact and needs a meet at
    // every control-flow join -- see merge_ud_alias. An alias that held on only one incoming edge
    // would bind a descriptor the other edge never assembled.
    std::unordered_map<int, int> sreg_ud_alias;
    // Immediate S_LOAD_DWORDX16 is typeless: it can be ordinary scalar data, or two adjacent
    // eight-dword T# descriptors. The latter is admitted only after a whole-stream use proof (see
    // proven_smem_x16_descriptor_loads). These are load PCs, not SRT keys: an x16 bundle contains
    // two resources at one byte offset, so every consumer must retain exact-PC provenance.
    std::unordered_set<uint32_t> smem_x16_descriptor_loads;
    bool smem_x16_descriptor_analysis_done = false;
    // Register-offset S_LOAD_DWORDX2 is likewise typeless. GTA V uses it to fetch the first two
    // words of a V#, then replaces/fills the remaining words before an exact-PC buffer consumer.
    // Only PCs certified by the whole-CFG descriptor-use proof may substitute placeholders.
    std::unordered_set<uint32_t> smem_x2_descriptor_fragment_loads;
    bool smem_x2_descriptor_fragment_analysis_done = false;
    uint32_t vcc = 0;
    uint32_t scc = 0;          // scalar condition code (bool); set by s_cmp_*/SCC-writing SOP2, read by s_cselect
    // #2418: does this shader read SCC anywhere? A STATIC property of the decoded stream, computed
    // once before emission, never mutated during it — so it carries none of the lifetime hazards a
    // stateful "pending reduction" marker would. It exists so the fragment stage can re-arm SCC after
    // a mask op (which needs an exact wave vote, and so forces wave64) ONLY for shaders that actually
    // consume it. Re-arming unconditionally is correct but makes every shader that merely saves EXEC
    // pay a subgroup-size requirement it does not need, which gates the draw on a 32-wide device.
    bool reads_scc = false;
    uint32_t exec = 0;         // per-lane execution mask (bool); v_cmpx narrows it, output store honors it
    bool exec_narrowed = false; // true once EXEC is narrowed below all-lanes-on (so VGPR writes predicate)
    // PC-relative EMBEDDED TABLES (#273): load pc -> the table's dwords, resolved by
    // detect_pcrel_tables (an s_getpc_b64-derived V# whose num_records is a known constant). The
    // shader BLOB carries the table; the recompiler folds it into a compile-time constant lookup.
    std::unordered_map<uint32_t, std::vector<uint32_t>> mubuf_pcrel_tables;
    std::unordered_map<uint32_t, std::vector<uint32_t>> smem_pcrel_tables;
    // Typed consumer of the same idiom, admitted only for conversion-free 32-bit formats (#2859).
    std::unordered_map<uint32_t, std::vector<uint32_t>> mtbuf_pcrel_tables;
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

inline bool has_wave64_mask_half_pair(const RegState& rs, int base) {
    const auto low = rs.sreg_wave64_mask_half.find(base);
    const auto high = rs.sreg_wave64_mask_half.find(base + 1);
    const auto low_index = rs.sreg_wave64_mask_half_index.find(base);
    const auto high_index = rs.sreg_wave64_mask_half_index.find(base + 1);
    return low != rs.sreg_wave64_mask_half.end() &&
        high != rs.sreg_wave64_mask_half.end() &&
        low_index != rs.sreg_wave64_mask_half_index.end() &&
        high_index != rs.sreg_wave64_mask_half_index.end() &&
        low_index->second == 0 && high_index->second == 1;
}

// Expire the Bool and physical-half spellings together. The complete-mask alias is erased only
// when its two dedicated halves prove that it came from adjacent-half promotion; unrelated VCC or
// saved-mask lifetimes remain under their existing architectural transfer functions. A B64 mask
// writer may replace the promoted alias after emit_alu, so its new Bool can be preserved explicitly.
inline void expire_wave64_mask_half(RegState& rs, int reg, int preserved_pair = -1) {
    for (int base : {reg - 1, reg}) {
        if (base < 0 || base == preserved_pair || !has_wave64_mask_half_pair(rs, base))
            continue;
        rs.sreg_bool.erase(base);
        rs.sreg_bool_narrowed.erase(base);
    }
    rs.sreg_wave64_mask_half.erase(reg);
    rs.sreg_wave64_mask_half_index.erase(reg);
}

// The same architectural transfer function as expire_wave64_mask_half, for the ORDINARY saved B64
// mask spelling, and deliberately NARROWER than it: a saved wave mask names the bits currently in a
// physical SGPR pair, so a scalar write over the register it is keyed on ends that lifetime. It is
// not provenance attached to the register number forever. (expire_wave64_mask_half expires on either
// word of the pair; only the ROOT word ends it here -- see the comment on expire_saved_b64_mask.)
//
// Without this, `sreg_bool` outlives its register and every later consumer reads a mask the hardware
// no longer holds. Two of them are load-bearing: `operand_bits` rejects an ordinary data read of
// such a word ("a persisted B64 wave mask has no ordinary scalar dword"), and the SOP2 mask lambdas
// would silently substitute the dead mask for a live one. R-Type Delta's sprite vertex shader is the
// worked example (#2783): it saves `s_cselect_b64 s[0:1], exec, 0` in its NGG fetch prologue, then
// rebuilds its PC-relative embedded-table address in that same s[0:1] with `s_getpc_b64` /
// `s_add_u32 s0, lit, s0`, and the whole shader was rejected at that add -- which dropped every
// sprite draw in the title.
//
// `snapshot_saved_b64_masks` must be taken BEFORE emit_alu, because emit_alu materializes the new
// lifetime for the same instruction. The staleness test is exactly "present in the snapshot AND its
// Bool id is unchanged", which is a PROXY for "this instruction did not publish it": every publisher
// either stores a fresh id or is named by `preserved_pair`. Classifying publishers syntactically
// instead is not sufficient -- `scalar_write_is_b64_mask` knows the SOP1/SOP2/VOPC/VOP3B mask
// writers, but the `vgpr_lane_mask_slots` reload in rdna2_emit_alu.cpp republishes a spilled alias
// from v_readlane with no syntactic marker at all. That same reload is the one publisher that could
// in principle re-store an IDENTICAL id (it would have to reload a mask into a register that already
// held that exact mask). If it were reached the alias would be dropped -- and the outcome is
// FAIL-VISIBLE, not silent: `src_mask` resolves a missing `sreg_bool` entry to 0 and every
// Bool-domain consumer then clears `ok` (rdna2_emit_alu.cpp :794, :817, :873, :1058, :1074), so the
// stage rejects. Silent zero is the DATA-domain outcome only. So the residual is bounded by being
// loud rather than by being harmless, and it is the same failure class this function repairs.
//
// VCC (106/107) is deliberately out of scope, and NOT because it is unreachable -- SGPR-kind
// operands really can carry 106/107 (`sgpr()` masks to 7 bits, rdna2_decode.cpp), and SOPK's
// read-modify-write forms read their own destination through `val(in.dst)`, so
// `s_mulk_i32 vcc_lo, imm` does reach operand_bits with an SGPR-kind 106. What makes VCC different
// is that its mask state is mirrored in `rs.vcc`, and expiring `sreg_bool[106]` without a matching
// policy for `rs.vcc` would leave the two spellings disagreeing. That is a separate change with its
// own risk -- it would also alter established behavior for every `s_bfe_u32 vcc_lo` NGG preamble --
// and no observed defect requires it: reaching the reject through VCC additionally needs a writer
// that overwrites VCC_LO while leaving no scalar SSA value behind, since operand_bits consults
// `rs.sreg` first. Tracked in #2804. Wave32 B32 aliases are likewise left to
// record_scalar_write's own (narrower) rules.
struct SavedB64MaskSnapshot {
    // (root, Bool id). At most a few entries: only the roots this one instruction can overwrite.
    std::vector<std::pair<int, uint32_t>> entries;
};

inline void expire_saved_b64_mask(RegState& rs, const SavedB64MaskSnapshot& before, int reg,
                                  int preserved_pair = -1) {
    // The ROOT word only, and this is measured rather than reasoned. Expiring on EITHER word of the
    // pair rejected 19 arms of test_recompile_coverage. Traced on the first of them ("a nested
    // varying-VCC compute CFG preserves spilled EXEC"): its Wave64 EXEC reload keys the
    // reconstructed mask on the LOW word (`v_readlane s14` -> `sreg_bool[14]`, the non-native
    // `vgpr_lane_mask_slots` branch, which erases `sreg_wave64_mask_half` so the
    // `publishes_wave64_mask_half` guard below does not apply) and then writes the HIGH word
    // (`v_readlane s15`). Treating that high-word write as an end-of-life destroyed the alias the
    // very next instruction consumes, and `s_mov_b64 exec, s[14:15]` rejected.
    //
    // So a high-word-only overwrite by unrelated scalar data keeps its established (conservative)
    // behavior. That is a PRE-EXISTING gap, not one introduced here -- before this function nothing
    // ended a B64 alias at all -- and #2783's defect is entirely in the root word.
    const int root = reg;
    if (root < 0 || root > 105 || root == preserved_pair) return;
    const auto mask = rs.sreg_bool.find(root);
    if (mask == rs.sreg_bool.end() || rs.sreg_bool_b32.contains(root)) return;
    // Present AND unchanged across emit_alu -- otherwise this instruction published it itself.
    bool stale = false;
    for (const auto& [snapshot_root, snapshot_id] : before.entries)
        if (snapshot_root == root && snapshot_id == mask->second) stale = true;
    if (!stale) return;
    rs.sreg_bool.erase(root);
    rs.sreg_bool_narrowed.erase(root);
}

// Is SGPR `R` provably DEAD at pc `target` — i.e. redefined before any read on the fall-through, so a
// write to it inside a divergent (execz) block linearized before `target` cannot be observed by later
// code? Sound/conservative: we only scan formats whose complete scalar read/write ranges are decoded,
// including bounded SMEM/MUBUF/MTBUF/MIMG packets. At the first unsupported memory, control-flow,
// interpolation, or unknown instruction we give up and report NOT-dead. Checking value ∈ {R, R-1}
// covers both a 32-bit read of R and a 64-bit pair whose low half is R-1.
// (RE-TAG: divergent-block scalar liveness.)
inline bool sopk_writes_scalar_data(uint32_t opcode) {
    // GFX10 SOPK encodes both read-only compares/waits/register-mode operations and genuine SGPR
    // destinations in the same SDST field. Keep the distinction central so CFG/provenance scans do
    // not mistake s_cmpk's source register for a redefinition.
    return opcode == 0x00 || opcode == 0x02 || opcode == 0x0F ||
           opcode == 0x10 || opcode == 0x12;
}
uint32_t scalar_write_width(const Rdna2Inst& in);

// VOP3B add/subtract without carry-in produces a fresh carry/borrow mask in SDST. Unlike the
// 0x128-0x12a family, this mask does not depend on a tracked src2 mask. Keep this exact predicate
// shared by every Wave32 scalar-width/provenance site so the emitter and CFG dispatcher cannot
// disagree about whether the physical destination is one word or an SGPR pair.
inline bool vop3b_fresh_carry_output(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP3 && in.sdst.kind == OperandKind::SGPR &&
           (in.opcode == 0x30f || in.opcode == 0x310 || in.opcode == 0x319);
}

// Decoder storage has an SDST-shaped field on every VOP3 packet, but only this exact VOP3B
// inventory architecturally writes it. VOP3A lane operations use `dst` for their scalar result;
// treating their overlapping bits as a second destination invents a mask write (notably for GTA's
// v_readlane reconstruction of VCC_LO/HI).
inline bool vop3_writes_mask_sdst(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP3 && in.sdst.kind == OperandKind::SGPR &&
           ((in.opcode >= 0x128 && in.opcode <= 0x12a) ||
            in.opcode == 0x16d || in.opcode == 0x16e || in.opcode == 0x176 ||
            in.opcode == 0x177 || vop3b_fresh_carry_output(in));
}
inline uint32_t scalar_write_width(const Rdna2Inst& in) {
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            if (in.opcode == 0x20) return 0; // s_setpc_b64 reads its decoded "dst" field.
            switch (in.opcode) {
                case 0x04: case 0x08: case 0x0a: case 0x1f: case 0x2d:
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
    if (vop3_writes_mask_sdst(in))
        visit(in.sdst.value, wave32_one_word_masks &&
                              ((in.opcode >= 0x128 && in.opcode <= 0x12a) ||
                              vop3b_fresh_carry_output(in)) ? 1u : 2u);
}

inline SavedB64MaskSnapshot snapshot_saved_b64_masks(const RegState& rs, const Rdna2Inst& in) {
    SavedB64MaskSnapshot snapshot;
    // The widest write form is deliberate: this set only FILTERS what record_scalar_write may
    // expire, and that function applies its own exact `effective_width`, so an extra candidate root
    // here can never widen the erase set.
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        for (uint32_t word = 0; word < width; ++word) {
            const int root = base + static_cast<int>(word);
            if (root > 105 || rs.sreg_bool_b32.contains(root)) continue;
            const auto mask = rs.sreg_bool.find(root);
            if (mask != rs.sreg_bool.end())
                snapshot.entries.emplace_back(root, mask->second);
        }
    }, /*wave32_one_word_masks*/false);
    return snapshot;
}

// True when this explicit scalar destination is written in the per-lane B64 mask domain.  Keep
// this classification independent of the post-emission SSA maps: folded/data writers such as
// s_getpc_b64 intentionally leave no scalar value behind, so absence from `sreg` cannot identify a
// mask write.  VOP3B's SDST is a mask/carry pair even though the instruction's primary VDST is data.
inline bool scalar_write_is_b64_mask(const Rdna2Inst& in, int base) {
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
    return vop3_writes_mask_sdst(in) && in.sdst.value == base;
}

// GTA V's Wave32 post-process kernel selects one scalar dword directly into VCC_LO, then uses
// that same physical word as the implicit predicate of v_cndmask.  The selected dword remains
// ordinary scalar data, but in Wave32 it also contains the complete architectural VCC mask.  Keep
// the admitted packet deliberately exact; the caller still has to prove Wave32 before bridging it.
inline bool is_scalar_cselect_b32_to_vcc_lo(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0a &&
        in.dst.kind == OperandKind::SGPR && in.dst.value == 106 &&
        in.src[0].kind == OperandKind::SGPR &&
        in.src[1].kind == OperandKind::InlineInt && in.src[1].value == 0;
}

// GTA V's Wave32 terrain kernel deliberately uses VCC_HI as an ordinary scalar word. Keep the
// syntactic exception in one place so the dispatcher and emitter cannot disagree about whether the
// physical register starts a mask or scalar-data lifetime. Callers must separately prove an exact
// native Wave32 subgroup before materializing EXEC_LO as a complete scalar dword.
inline bool is_gtav_wave32_vcchi_scalar_packet(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0e &&
        in.dst.kind == OperandKind::SGPR && in.dst.value == 107 &&
        in.src[0].kind == OperandKind::Special && in.src[0].value == 126 &&
        in.src[1].kind == OperandKind::Literal && in.literal == 0x0fffffffu;
}

inline bool allows_compute_scalar_vcc_bridge(const SpirvCompute& b) {
    return b.is_compute && b.allow_b32_masks && b.wave_size == 32;
}

// Meet for the direct-descriptor copy alias at a control-flow join (#1773). An alias asserts what
// the bits in one register ARE, so after a join it holds only if BOTH incoming edges assert the
// same thing; `other` is the alias map of the edge not currently in `rs`.
//
// This must be a meet and not an erase-by-written-set. `then_written` at the if/else joins is a
// SNAPSHOT of the whole `sreg_written` set rather than a delta, and an alias can only exist on a
// register that has been written -- so erasing by it drops every alias inherited from before the
// branch (safe but useless) while keeping exactly the one-armed aliases that are unsound. The
// skipped edge of an if-only region likewise carries the pre-branch aliases, not the arm's.
//
// `other` must be a distinct copy -- this erases from `rs` while iterating `other`, so passing
// `rs.sreg_ud_alias` itself would be undefined. Every call site snapshots into its own `const auto`.
inline void merge_ud_alias(RegState& rs, const std::unordered_map<int, int>& other) {
    for (auto it = rs.sreg_ud_alias.begin(); it != rs.sreg_ud_alias.end(); ) {
        auto edge = other.find(it->first);
        if (edge == other.end() || edge->second != it->second) it = rs.sreg_ud_alias.erase(it);
        else ++it;
    }
}

inline void record_scalar_write(RegState& rs, const Rdna2Inst& in,
                         bool allow_compute_scalar_vcc_bridge,
                         const SavedB64MaskSnapshot& saved_b64_masks_before) {
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
    const bool vop3_b32_mask = vop3_writes_mask_sdst(in) &&
        rs.sreg_bool_b32.contains(in.sdst.value);
    if (vop3_writes_mask_sdst(in)) {
        if (vop3_b32_mask) {
            rs.sreg.erase(in.sdst.value);
            rs.sreg_srt.erase(in.sdst.value);
        } else {
            invalidate_mask_pair(in.sdst.value);
        }
    }

    const bool vopc_b32_write = in.fmt == Rdna2Format::VOPC &&
        !vopc_is_cmpx(in.opcode) && rs.sreg_bool_b32.contains(in.dst.value);

    // DIRECT-descriptor copy alias (#1773). Decide the origin from the state BEFORE the write loop
    // below expires it: `s_mov_b32 sD, sS` may name the same register on both sides, and the loop
    // marks the destination written. Only the one-word scalar move is admitted -- it is the form
    // every observed staging idiom uses, and it cannot be a B64 mask copy. A source that is itself
    // a live mask is excluded outright: those bits are a predicate, not descriptor words.
    int ud_alias_dst = -1, ud_alias_origin = -1;
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x03 &&
        in.dst.kind == OperandKind::SGPR && in.src[0].kind == OperandKind::SGPR &&
        in.dst.value <= 101 && in.src[0].value <= 101 &&
        !rs.sreg_bool.contains(in.src[0].value) &&
        !rs.sreg_bool_b32.contains(in.src[0].value)) {
        const int src = in.src[0].value;
        if (auto chained = rs.sreg_ud_alias.find(src); chained != rs.sreg_ud_alias.end())
            ud_alias_origin = chained->second;      // a copy of a copy still names the origin
        else if (!rs.sreg_written.contains(src))
            ud_alias_origin = src;                  // still the entry-time value the driver supplied
        if (ud_alias_origin >= 0) ud_alias_dst = in.dst.value;
    }

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
        const bool scalar_cselect_vcc_bridge =
            allow_compute_scalar_vcc_bridge && base == 106 &&
            is_scalar_cselect_b32_to_vcc_lo(in) &&
            rs.sreg.contains(106) && rs.sreg_bool_b32.contains(106);
        const bool writes_b32_mask = (effective_width == 1 || vopc_b32_mask) &&
            rs.sreg_bool_b32.contains(base) &&
            (scalar_cselect_vcc_bridge ||
             (!rs.sreg.contains(base) &&
              ((in.fmt == Rdna2Format::SOP1 &&
                (in.opcode == 0x03 || in.opcode == 0x07 || in.opcode == 0x09 ||
                 sop1_opcode_is_emitted_saveexec_b32(in.opcode))) ||
               sop2_b32_mask_domain || vopc_b32_mask || vop3_b32_mask_write)));
        const bool writes_b64_mask = effective_width == 2 && !vopc_b32_mask &&
            scalar_write_is_b64_mask(in, base);
        // The dedicated Wave64 half domain is a spelling of this physical SGPR's current bits,
        // not provenance attached to the register number forever.  V_READLANE clears the old
        // entry before emission and republishes it only for a MUST-proven mask-half reload; every
        // other scalar write ends that lifetime.  This post-emission hook runs before the next
        // instruction, so a later sibling-half reload cannot promote a stale Bool together with
        // the fresh half.
        const bool publishes_wave64_mask_half =
            in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 &&
            effective_width == 1 && base == in.dst.value &&
            rs.sreg_wave64_mask_half.contains(base) &&
            rs.sreg_wave64_mask_half_index.contains(base);
        for (uint32_t word = 0; word < effective_width; ++word) {
            const int reg = base + static_cast<int>(word);
            if (!publishes_wave64_mask_half || reg != base) {
                expire_wave64_mask_half(rs, reg, writes_b64_mask ? base : -1);
                expire_saved_b64_mask(rs, saved_b64_masks_before, reg,
                                      writes_b64_mask ? base : -1);
            }
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
            // Any write ends a copy alias for the register it lands on. This is the one place every
            // scalar write form funnels through, which is why the alias is expired here rather than
            // beside each of the ~50 sites that erase sreg_srt.
            rs.sreg_ud_alias.erase(reg);
        }
    }, vopc_b32_write || vop3_b32_mask);

    if (ud_alias_dst >= 0) rs.sreg_ud_alias[ud_alias_dst] = ud_alias_origin;
}

// emit_alu lives in rdna2_emit_alu.cpp -- 7,676 lines of instruction-family translation that is
// the single largest thing in the recompiler. Its callers stayed behind, so the declaration lives
// here. The DEFAULT ARGUMENTS are stated here and nowhere else: C++ permits them once per scope,
// so the definition carries the same parameters without them.
bool emit_alu(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_exec_update,
              const std::unordered_set<uint32_t>* safe_execz = nullptr, bool allow_smem = false,
              const ShaderResourceTable* rt = nullptr, bool allow_wave = false);

// emit_body and the CFG state machine it drives live in rdna2_emit_cfg.cpp. As with emit_alu, the
// default arguments are stated here and nowhere else.
bool emit_cfg_state_machine(
    SpirvCompute& b, RegState& initial, const std::vector<Rdna2Inst>& ins,
    const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
    bool allow_exec_update, bool allow_smem,
    const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn,
    const uint32_t* code, size_t dwords, uint32_t initial_active,
    bool synchronize_lds_fminmax);

bool emit_body(SpirvCompute& b, RegState& rs, const std::vector<Rdna2Inst>& ins,
               const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
               bool allow_exec_update, bool allow_smem,
               const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn,
               const uint32_t* code = nullptr, size_t dwords = 0,
               const std::unordered_set<uint32_t>* inherited_dead_masks = nullptr,
               bool allow_cfg_dispatcher = true,
               uint32_t initial_dispatch_active = 0,
               bool force_barrier_phases = false,
               bool force_lds_fminmax_dispatcher = false);

}  // namespace prosper::gpu
