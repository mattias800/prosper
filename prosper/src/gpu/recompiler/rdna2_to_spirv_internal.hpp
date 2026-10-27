#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#pragma once
#include "gpu/recompiler/rdna2_reg_state.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_scalar_condition_marks.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/compiler_choices.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_accumulator_cohort.hpp"
#include "gpu/recompiler/rdna2_loop_arbitrary_carry.hpp"
#include "gpu/recompiler/rdna2_diagnostic_attempt.hpp"
#include "gpu/recompiler/rdna2_portable_mask_slots.hpp"

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
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_scalar_exec_scopes.hpp"
#include "gpu/recompiler/rdna2_relative_bank_source.hpp"
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
#include <memory>
#include <mutex>
#include <set>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

class SavedMaskCopyFrame;

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

// While a TerminalRejectCapture (rdna2_to_spirv.hpp) is alive on this thread, every reason recorded
// is also appended here, so a memoized analysis can replay exactly the records a fresh evaluation
// would have made (compute_program_facts.cpp).
inline thread_local std::vector<std::pair<std::string, std::string>>* g_terminal_reject_capture =
    nullptr;

inline void record_terminal_reject_reason(uint64_t program_address, const char* tag,
                                   const char* payload) {
    if (defer_terminal_reject_reason(program_address, tag, payload)) return;
    if (g_terminal_reject_capture) g_terminal_reject_capture->emplace_back(tag, payload);
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
    va_list args;
    va_start(args, format);
    const int body = std::vsnprintf(nullptr, 0, format, args);
    va_end(args);
    if (body < 0) return;
    std::string detail(static_cast<size_t>(body), '\0');
    va_start(args, format);
    std::vsnprintf(detail.data(), detail.size() + 1, format, args);
    va_end(args);
    const bool fabricated = (tag && std::strstr(tag, "hard-placeholder-read")) ||
                            detail.find("hard-placeholder-read") != std::string::npos ||
                            (tag && std::strstr(tag, "unmodelled-scalar-publication")) ||
                            detail.find("unmodelled-scalar-publication") != std::string::npos ||
                            (tag && std::strstr(tag, "fabricated-fresh-mask-source")) ||
                            detail.find("fabricated-fresh-mask-source") != std::string::npos ||
                            detail.find("is fabricated on the back edge") != std::string::npos;
    if (fabricated && diagnostic.placeholder_origins && detail.find("origin=") == std::string::npos)
        detail += " placeholder-roots=[" + *diagnostic.placeholder_origins + "]";
    char* payload = detail.data();
    if (defer_recompile_diagnostic(diagnostic, tag, role, payload)) return;
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
    std::vector<char> line(detail.size() + 512);
    const int written =
        diagnostic.program_address
            ? std::snprintf(line.data(), line.size(), "[%s] %s stage=%s program=0x%llx role=%s\n",
                            tag, payload, recompile_diagnostic_stage_name(diagnostic.stage),
                            static_cast<unsigned long long>(diagnostic.program_address), role)
            : std::snprintf(line.data(), line.size(), "[%s] %s stage=%s program=none role=%s\n",
                            tag, payload, recompile_diagnostic_stage_name(diagnostic.stage), role);
    if (written < 0) return;
    const size_t used = std::min(static_cast<size_t>(written), line.size() - 1);
    if (used == line.size() - 1) { line[line.size() - 2] = '\n'; }
    std::fwrite(line.data(), 1, used, stderr);
}
enum : uint32_t {
    Op_Extension=10, Op_ExtInstImport=11, Op_ExtInst=12, Op_MemoryModel=14, Op_EntryPoint=15, Op_ExecutionMode=16,
    Op_Capability=17, Op_TypeVoid=19, Op_TypeBool=20, Op_TypeInt=21, Op_TypeFloat=22, Op_TypeVector=23,
    Op_TypeRuntimeArray=29, Op_TypeStruct=30, Op_TypePointer=32, Op_TypeFunction=33,
    Op_ConstantTrue=41, Op_ConstantFalse=42, Op_Constant=43, Op_Function=54, Op_FunctionEnd=56, Op_Variable=59,
    Op_Dot=148,
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
enum : uint32_t {
    Glsl_FAbs = 4,
    Glsl_RoundEven = 2,
    Glsl_Trunc = 3,
    Glsl_Floor = 8,
    Glsl_Ceil = 9,
    Glsl_Fract = 10,
    Glsl_Sin = 13,
    Glsl_Cos = 14,
    Glsl_Exp2 = 29,
    Glsl_Log2 = 30,
    Glsl_Sqrt = 31,
    Glsl_InverseSqrt = 32,
    Glsl_FMin = 37,
    Glsl_UMin = 38,
    Glsl_SMin = 39,
    Glsl_FMax = 40,
    Glsl_UMax = 41,
    Glsl_SMax = 42,
    Glsl_PackHalf2x16 = 58,
    Glsl_UnpackHalf2x16 = 62,
    Glsl_FindILsb = 73,   // lowest set bit of a 32-bit word, all ones at zero
    Glsl_FindUMsb = 75,   // bit index of the highest set bit (undefined at zero)
    Glsl_Fma = 50,
    Glsl_NMin = 79,
    Glsl_NMax = 80
};   // NaN-aware min/max: one-NaN operand -> the other operand
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
    // SPV_KHR_float_controls. Declared by declare_float_controls() for every guest module, on a
    // device that has been MEASURED to accept it; see the comment there (#3479, gated in #3561).
    // The five capabilities of that extension are CONSECUTIVE and easy to take one off by:
    // 4464 DenormPreserve, 4465 DenormFlushToZero, 4466 SignedZeroInfNanPreserve,
    // 4467 RoundingModeRTE, 4468 RoundingModeRTZ. The first draft of this line said 4467, and
    // NVIDIA honoured the execution mode anyway while emitting an invalid module -- only
    // `spv_validate` (spirv-val, an instrument that does not share this header) reported it.
    Cap_SignedZeroInfNanPreserve=4466,
    Cap_FloatControls2=6029,
    Cap_StorageBufferArrayNonUniformIndexing=5308,
    Addr_Logical=0, Mem_GLSL450=1, Exec_Vertex=0, Exec_Geometry=3, Exec_Fragment=4, Exec_GLCompute=5,
    EM_Invocations=0, EM_OriginUpperLeft=7, EM_DepthReplacing=12, EM_LocalSize=17, EM_Triangles=22,
    EM_OutputVertices=26, EM_OutputTriangleStrip=29, EM_Xfb=11,   // transform-feedback execution mode
    EM_SignedZeroInfNanPreserve=4461,
    SC_Input=1, SC_UniformConstant=0, SC_Output=3, SC_Function=7, SC_PushConstant=9,
    SC_Image=11, SC_StorageBuffer=12, FC_None=0,
    Dim_1D=0, Dim_2D=1, Dim_3D=2, Dim_Cube=3,   // SPIR-V Dim. (2D coincides with the SQ_RSRC 2D dim value, but distinct.)
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
    Dec_FPFastMathMode=40,
    // Decoration 5300 -- descriptor indexing's NonUniform, unrelated to the GroupNonUniform ops.
    Dec_NonUniform=5300,
    BI_Position=0, BI_Layer=9, BI_FragCoord=15, BI_SampleMask=20, BI_FragDepth=22, BI_HelperInvocation=23,
    BI_WorkgroupId=26, BI_LocalInvocationId=27,
    BI_GlobalInvocationId=28, BI_SubgroupId=40, BI_SubgroupLocalInvocationId=41,
    BI_VertexIndex=42, BI_InstanceIndex=43,
    GroupOp_Reduce=0, GroupOp_ExclusiveScan=2,
};

// Arithmetic contraction constraint (SPIR-V decoration).
enum : uint32_t { Dec_NoContraction = 42 };

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
enum class GuestShaderStage { Standalone, Compute, Vertex, Fragment };
enum class PhysicalExecutionDomain { Invocation, Workgroup };
// Only an entry owning every guest slot may select through cross-worker register state. Native
// subgroup width, the physical compute flag and copying-only scalar taint grant no such authority.
enum class GuestWaveEventDomain { None, Owned64 };

#define PROSPER_GPU_RECOMPILER_SPIRV_COMPUTE_DECLARATION_SCOPE
#include "gpu/recompiler/rdna2_spirv_compute.inl"   // IWYU pragma: export
#undef PROSPER_GPU_RECOMPILER_SPIRV_COMPUTE_DECLARATION_SCOPE

// Cache and live realization derive owned source authority from the original code. A selected
// stream may drop requirements, but must not invent a new admitted read-point PC. Keep omitted
// original PCs classified too, so their bindings cannot become another load's legacy fallback.
inline bool retain_original_owned_raw_wide_proof(
        RegState& rs, const std::vector<uint32_t>& original_pcs) {
    for (uint32_t pc : rs.smem_raw_owned_wide_data_loads)
        if (!std::binary_search(original_pcs.begin(), original_pcs.end(), pc)) return false;
    rs.smem_raw_owned_wide_data_loads.insert(original_pcs.begin(), original_pcs.end());
    return true;
}

inline bool retain_original_owned_nested_wide_proof(
        RegState& rs, const std::vector<RawNestedWideChain>& original) {
    for (const auto& chain : rs.smem_owned_nested_wide_chains)
        if (std::find(original.begin(), original.end(), chain) == original.end()) return false;
    rs.smem_owned_nested_wide_chains = original;
    return true;
}

inline bool retain_original_owned_raw_x2_proof(RegState& rs,
                                               const std::vector<RawNestedWideChain>& original) {
    for (const auto& chain : rs.smem_owned_raw_x2_chains)
        if (std::find(original.begin(), original.end(), chain) == original.end()) return false;
    rs.smem_owned_raw_x2_chains = original;
    for (const auto& chain : original) {
        rs.smem_raw_x2_data_loads.insert(chain.parent_pc);
        rs.smem_raw_x2_data_loads.insert(chain.child_pc);
    }
    return true;
}
// The value a structured emitter seeds a phi or merge with for register `r`: the register's value,
// including genuine ordinary launch data, or the fabricated zero when it is absent.
// CONFIDENCE: HIGH -- operand_bits reads these same producing sreg_input words. An arm-local
// overwrite must preserve that value on the skipped edge, before the merge erases the input alias.
// LoopBlanketCheck separately records which loop seeds were held.
inline uint32_t sreg_seed(SpirvCompute& b, const RegState& rs, int r) {
    const auto it = rs.sreg.find(r);
    if (it != rs.sreg.end()) return it->second;
    if (r >= 0 && r <= 105) {
        const auto input = rs.sreg_input.find(r);
        if (input != rs.sreg_input.end()) return input->second;
    }
    return b.uconst(0);
}
inline uint32_t vreg_seed(SpirvCompute& b, RegState& rs, int r) {
    const auto it = rs.vreg.find(r);
    if (it != rs.vreg.end()) return it->second;
    auto& marks = rs.vreg_source_marks[r];
    marks.placeholder = marks.hard = true;
    marks.blanket_roots.clear();
    return b.uconst(0);
}
inline uint32_t scalar_merge_seed(SpirvCompute& b, RegState& rs, int r) {
    const uint32_t value = sreg_seed(b, rs, r);
    // Equal incoming SSA IDs need no phi. Retain their genuine value before the merge retires
    // sreg_input for any word written on the other edge (including an s_mov sN,sN).
    if (r >= 0 && r <= 105 && !rs.sreg.contains(r) && rs.sreg_input.contains(r)) rs.sreg[r] = value;
    return value;
}
inline bool merge_edge_word_fabricated(const RegState& rs, int r) {
    if (r >= 106 && r <= 124 && !rs.sreg.contains(r)) return true;
    // Match the actual phi input above. Explicit marks still win over genuine launch presence.
    return (!rs.sreg_input.contains(r) && sreg_word_absent_unmasked(rs, r)) ||
           sreg_word_may_be_fabricated(rs, r);
}
// Whether a held word's only mark is a loop header's blanket mark (sreg_loop_blanket): defined
// under that header's assumption, which its back edge checks (LoopBlanketCheck).
inline bool sreg_word_blanket_only(const RegState& rs, int r) {
    return rs.sreg.contains(r) && rs.sreg_loop_blanket.contains(r);
}
// One loop header's assumption about one carried register.
inline uint64_t blanket_root(uint32_t header_pc, int r) {
    return (static_cast<uint64_t>(header_pc) << 16u) | static_cast<uint16_t>(r);
}

// Explicit SCC is a numeric 0/1 source in any decoded operand role; scalar condition/carry
// instructions also consume it implicitly (s_cselect, s_addc/s_subb, s_cmov, s_cmovk -- the
// instruction-effects table owns that list). An SCC branch is not a data read and is not reported.
// A reserved or unknown scalar opcode IS reported: callers use this to demand a defined SCC.
inline bool scalar_reads_scc(const Rdna2Inst& in) {
    return rdna2_operand_reads_scc(in) ||
           rdna2_reads_scc_as_data(in.fmt, in.opcode) != Rdna2Answer::No;
}

// scalar_source_marks() lives in rdna2_cfg_support.hpp, beside the width inventories it reads.

// Set a merged register's marks from its two incoming edges. `rs` is one edge; the caller supplies
// what the other edge held. The memory mark is a union.
// `blanket_roots` keeps the loop header's blanket mark, with these assumptions, when every edge that
// is marked is marked only by it (a merge of such a word with a defined one is defined under the
// same header assumptions).
inline void join_merge_placeholder(RegState& rs, int r, bool either_edge_fabricated,
                                   bool other_edge_memory,
                                   const std::set<uint64_t>* blanket_roots = nullptr) {
    rs.sreg_exec_scopes.erase(r);
    if (either_edge_fabricated)
        rs.sreg_merge_placeholder.insert(r);
    else
        rs.sreg_merge_placeholder.erase(r);
    if (either_edge_fabricated && blanket_roots)
        rs.sreg_loop_blanket[r] = *blanket_roots;
    else
        rs.sreg_loop_blanket.erase(r);
    if (other_edge_memory) rs.sreg_memory_pattern.insert(r);
}

// One structured-merge edge's marks, captured before the other edge is emitted.
struct MergeEdgeMarks {
    std::set<int> fabricated;   // registers this edge may hold as a fabricated zero
    // ...of which only by a loop header's blanket mark, with its assumptions (sreg_loop_blanket)
    std::map<int, std::set<uint64_t>> blanket;
    std::set<int> memory;   // this edge's memory-pattern marks
    std::set<int> unavailable_saved_mask_words;
    bool scc = false;   // this edge's SCC is marked or poisoned (merges as bfalse)
    std::set<uint64_t> scc_blanket;
    // This edge's v_writelane spill-slot marks. A merge keeps only one edge's slot map, so a slot
    // the OTHER edge left marked would otherwise come out clean (#4725 review).
    std::set<std::pair<int, int>> slot_fabricated, slot_memory;
    VectorWordMarks vectors;
    bool vector_full_exec = false;
    // Nonzero when this is the skipped edge of a forward s_cbranch_execz: the EXEC value that is
    // empty in every lane reaching it (confine_execz_skip_marks).
    uint32_t execz_skip_exec = 0;
};
template <class Registers>
MergeEdgeMarks merge_edge_marks(const RegState& rs, const Registers& registers,
                                bool execz_skip = false) {
    MergeEdgeMarks marks;
    for (int r : registers)
        if (merge_edge_word_fabricated(rs, r)) {
            marks.fabricated.insert(r);
            if (sreg_word_blanket_only(rs, r)) marks.blanket[r] = rs.sreg_loop_blanket.at(r);
        }
    marks.memory = rs.sreg_memory_pattern;
    marks.unavailable_saved_mask_words = rs.unavailable_saved_mask_words;
    marks.scc = rs.scc_merge_placeholder || !rs.scc;
    marks.scc_blanket = rs.scc_loop_blanket;
    marks.slot_fabricated = rs.lane_slot_merge_placeholder;
    marks.slot_memory = rs.lane_slot_memory_pattern;
    marks.vectors = rs.vreg_source_marks;
    marks.vector_full_exec = rs.vector_full_exec;
    marks.execz_skip_exec = execz_skip ? rs.exec : 0u;
    return marks;
}
// Join register `r` at a two-edge merge: `rs` is one edge, `other` the captured other edge.
inline void join_merge_edge(RegState& rs, int r, const MergeEdgeMarks& other) {
    const bool here = merge_edge_word_fabricated(rs, r);
    const bool there = other.fabricated.contains(r);
    const bool blanket_only =
        (!here || sreg_word_blanket_only(rs, r)) && (!there || other.blanket.contains(r));
    std::set<uint64_t> roots;
    if (here && blanket_only) roots = rs.sreg_loop_blanket.at(r);
    if (there && blanket_only) roots.insert(other.blanket.at(r).begin(), other.blanket.at(r).end());
    join_merge_placeholder(rs, r, here || there, other.memory.contains(r),
                           blanket_only ? &roots : nullptr);
}
// Join SCC and the spill slots at the same merge (a poisoned SCC edge merges as bfalse, which is
// fabricated too; a slot marked on either edge is marked after it).
inline void join_merge_scc_and_slots(RegState& rs, const MergeEdgeMarks& other) {
    rs.sreg_exec_scopes.clear();   // an edge-local view does not prove the merged generation
    if (other.execz_skip_exec) {
        VectorWordMarks skipped = other.vectors;
        confine_execz_skip_marks(rs, skipped, rs.vreg_source_marks, other.execz_skip_exec);
        join_vector_word_marks(rs.vreg_source_marks, skipped, &rs);
    } else {
        join_vector_word_marks(rs.vreg_source_marks, other.vectors, &rs);
    }
    rs.vector_full_exec = rs.vector_full_exec && other.vector_full_exec;
    rs.unavailable_saved_mask_words.insert(other.unavailable_saved_mask_words.begin(),
                                           other.unavailable_saved_mask_words.end());
    const bool scc_hard = ((rs.scc_merge_placeholder || !rs.scc) && rs.scc_loop_blanket.empty()) ||
                          (other.scc && other.scc_blanket.empty());
    if (scc_hard)
        rs.scc_loop_blanket.clear();
    else
        rs.scc_loop_blanket.insert(other.scc_blanket.begin(), other.scc_blanket.end());
    rs.scc_merge_placeholder = rs.scc_merge_placeholder || !rs.scc || other.scc;
    rs.lane_slot_merge_placeholder.insert(other.slot_fabricated.begin(),
                                          other.slot_fabricated.end());
    rs.lane_slot_memory_pattern.insert(other.slot_memory.begin(), other.slot_memory.end());
}
// The assumption behind sreg_loop_blanket, checked once the back edge exists (#4749 review). A
// header blanket-marks a carried word that was defined on the preheader edge, taking its back-edge
// value as defined too, so lane-bit reads inside the loop may admit it before the body is emitted.
// If such a word leaves the body hard-marked (fabricated by something other than the blanket), the
// assumption was false: every lane-bit read admitted on it since the header is void, and the loop
// exits must not hand the blanket on. Built BEFORE the header seeds its phis, from the preheader
// state: a word held there unmarked, or marked only by an enclosing loop's blanket (defined under
// that loop's assumption, which its own back edge checks; GTA V's workgroup-store kernel carries
// VCC scratch derived from its outer counter into an inner loop), is defined on the preheader edge.
// An absent word is seeded with the fabricated zero, so it is not.
// Assumptions are tracked per (header, carried register): a word derived from blanket words carries
// the union of their roots, and a lane-bit read records the roots it relied on. A violated root only
// voids the reads that depended on it, so a counter recycled as VCC scratch is not refused because
// some unrelated carried word (VCC itself, turned into a mask in the body) is not data on the back
// edge.
struct LoopBlanketCheck {
    // The carried words defined on the preheader edge, each with the enclosing assumptions its
    // preheader value already depends on (empty when it is plainly defined).
    std::map<int, std::set<uint64_t>> carried;
    // Hard vector words whose missing lanes lie outside this preheader EXEC domain. Their header
    // reads remain conditional on each back edge defining the same domain, not just the same ID.
    std::map<int, uint32_t> confined;
    // Vector words whose header keeps the arbitrary property, and whether this loop may hold a
    // defined-preheader word weakly (rdna2_loop_arbitrary_carry.hpp, #5193).
    std::set<int> arbitrary;
    bool weak_allowed = false;
    // Vector words whose header may NOT keep the arbitrary property: those the exit condition of a
    // compute EXEC loop without the narrowing certificate reads (retry_gate).
    std::set<int> no_arbitrary_header;
    uint32_t header_pc = 0;
    template <class Registers>
    LoopBlanketCheck(const RegState& rs, const Registers& registers, uint32_t header,
                     std::set<int> no_arbitrary = {})
        : no_arbitrary_header(std::move(no_arbitrary)), header_pc(header) {
        for (int r : registers) {
            if (!rs.sreg.contains(r)) continue;
            if (!merge_edge_word_fabricated(rs, r))
                carried[r];
            else if (sreg_word_blanket_only(rs, r))
                carried[r] = rs.sreg_loop_blanket.at(r);
        }
    }
    // At the back edge (`rs` is the body's end state): false, with a terminal reject line, when a
    // lane-bit read relied on a carried word whose back-edge value is hard-marked. A violated word
    // that nothing read through only stops the exits from handing its assumption on.
    template <class Exits>
    bool holds(SpirvCompute& b, RegState& rs, Exits& exits) const {
        rs.sreg_exec_scopes.clear();   // exit restoration must establish its own scalar views
        // A carried word is violated when its back-edge value is hard-marked, or blanket-marked on
        // an assumption of this header that is itself violated (s10 = s11 carries s11's fate). An
        // arbitrary back edge holds a defined-preheader vector word weakly (#5193).
        const LoopRootVerdicts verdict =
            loop_root_verdicts(b, rs, header_pc, carried, confined, arbitrary, weak_allowed);
        const std::set<uint64_t>& violated = verdict.violated;
        std::vector<int> diagnostic_words;
        int read_word = -1;
        for (const auto& entry : carried)
            if (read_word < 0 && violated.contains(blanket_root(header_pc, entry.first)) &&
                b.loop_blanket_roots_read.contains(blanket_root(header_pc, entry.first)))
                read_word = entry.first;
        for (const auto& entry : carried)
            if (violated.contains(blanket_root(header_pc, entry.first)) &&
                b.loop_blanket_roots_read.contains(blanket_root(header_pc, entry.first)))
                diagnostic_words.push_back(entry.first);
        PlaceholderRoots diagnostic_roots;
        if (!diagnostic_words.empty()) {
            diagnostic_roots = loop_placeholder_roots(rs, header_pc, diagnostic_words,
                                                      b.placeholder_origin_program.get());
            append_unproved_loop_reads(diagnostic_roots, b, header_pc, diagnostic_words);
            b.diagnostic.placeholder_origins =
                std::make_shared<const std::string>(format_placeholder_origins(diagnostic_roots));
        }
        // Both actual exit edges may still depend on the header's assumptions. Demote the whole
        // assumption when any required root failed; keep its placeholder and unrelated valid roots.
        const auto invalid = [&](const auto& roots) {
            return std::any_of(roots.begin(), roots.end(),
                               [&](uint64_t root) { return violated.contains(root); });
        };
        // Scalar data holds no arbitrary value: a weak root is as invalid as a violated one there.
        const auto scalar_invalid = [&](const auto& roots) {
            return invalid(roots) || std::any_of(roots.begin(), roots.end(), [&](uint64_t root) {
                       return verdict.weak.contains(root);
                   });
        };
        const auto sanitize = [&](auto& scalar, auto& vectors) {
            std::erase_if(scalar, [&](const auto& word) { return scalar_invalid(word.second); });
            for (auto& [reg, source] : vectors) {
                strip_arbitrary_outside_roots(source, violated);
                retire_vector_word_roots(source, violated);
                weaken_vector_word_roots(source, verdict.weak);
            }
        };
        sanitize(exits.blanket, exits.vectors);
        sanitize(rs.sreg_loop_blanket, rs.vreg_source_marks);
        if (scalar_invalid(exits.scc_blanket)) exits.scc_blanket.clear();
        if (scalar_invalid(rs.scc_loop_blanket)) rs.scc_loop_blanket.clear();
        if (read_word < 0) return true;
        if (admit_arbitrary_placeholder_read(b, header_pc, ArbitraryPlaceholderRead::LoopCarry,
                                             diagnostic_roots))
            return true;
        log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                 "loop-carried %c%d is fabricated on the back edge but was read as "
                                 "lane bits in the loop (header pc=%u)",
                                 read_word >= kVectorBlanketBase ? 'v' : 's',
                                 read_word >= kVectorBlanketBase
                                     ? (read_word - kVectorBlanketBase) %
                                           kArbitraryOutsideRootOffset
                                     : read_word,
                                 header_pc);
        return false;
    }
};

// Every loop-carried SGPR, and SCC, is marked at its loop header: the phi takes the preheader value
// (the fabricated zero when absent) and the back edge, whose marks are unknown until the body has
// been emitted. A write from definite inputs clears the mark.
// A spill slot the loop writes (a constant-lane v_writelane in [lo, hi)) is loop-carried the same
// way and is marked at the header too.
template <class Registers>
void mark_loop_carried(RegState& rs, const Registers& carried, const std::vector<Rdna2Inst>& ins,
                       uint32_t lo, uint32_t hi, LoopBlanketCheck& blanket) {
    rs.sreg_exec_scopes.clear();   // incoming views do not authenticate new loop-header PHIs
    mark_vector_loop_carried(rs, ins, lo, hi, blanket.carried, blanket.confined, &blanket.arbitrary,
                             blanket.no_arbitrary_header);
    blanket.weak_allowed = loop_may_hold_weakly(ins, lo, hi);
    // A full preheader does not prove a full backedge. Clear the independent proof; ordinary
    // entry-iteration full writes still follow exec_narrowed and initialize their old generation.
    for (const auto& in : ins)
        if (in.pc >= lo && in.pc < hi && rdna2_instruction_may_change_exec(in))
            rs.vector_full_exec = false;
    for (int r : carried) {
        join_merge_placeholder(rs, r, true, false);
        if (const auto it = blanket.carried.find(r); it != blanket.carried.end()) {
            auto& roots = rs.sreg_loop_blanket[r];
            roots = it->second;   // the enclosing assumptions, plus this header's own
            roots.insert(blanket_root(lo, r));
        }
    }
    rs.scc_merge_placeholder = true;
    rs.scc_loop_blanket.clear();
    for (const Rdna2Inst& in : ins)
        if (in.pc >= lo && in.pc < hi && in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361 &&
            in.src[1].kind == OperandKind::InlineInt)
            rs.lane_slot_merge_placeholder.insert({in.dst.value, in.src[1].value});
}
// The loop's marks at the END OF ITS CHECK BLOCK, where the exit leaves: a condition-region
// register exits with the value it had there, so it must exit with the mark it had there too, not
// with whatever the body's later writes left (#4711 review, finding 1).
struct LoopCheckMarks {
    std::set<int> fabricated, memory;
    std::set<std::pair<int, int>> slot_fabricated, slot_memory;
    std::map<int, std::set<uint64_t>> blanket;   // the check block's sreg_loop_blanket
    VectorWordMarks vectors;
    bool vector_full_exec = false;
    bool scc = false;
    std::set<uint64_t> scc_blanket;
};
inline LoopCheckMarks loop_check_marks(const RegState& rs) {
    return {rs.sreg_merge_placeholder,
            rs.sreg_memory_pattern,
            rs.lane_slot_merge_placeholder,
            rs.lane_slot_memory_pattern,
            rs.sreg_loop_blanket,
            rs.vreg_source_marks,
            rs.vector_full_exec,
            rs.scc_merge_placeholder || !rs.scc,
            rs.scc_loop_blanket};
}
// The spill slots at a loop exit: `rs` holds the body end's slot map, so union in the check
// block's marks (a slot marked on either is marked after the loop).
inline void mark_loop_exit_slots(RegState& rs, const LoopCheckMarks& check,
                                 const LoopCheckMarks* body = nullptr) {
    if (body) rs.vreg_source_marks = body->vectors;
    join_vector_word_marks(rs.vreg_source_marks, check.vectors);
    rs.vector_full_exec = rs.vector_full_exec && check.vector_full_exec;
    rs.lane_slot_merge_placeholder.insert(check.slot_fabricated.begin(),
                                          check.slot_fabricated.end());
    rs.lane_slot_memory_pattern.insert(check.slot_memory.begin(), check.slot_memory.end());
}
// At the loop exit: a body-only register leaves as the (marked) header phi; a condition-region
// register leaves with its check-block marks. An optional body snapshot supplies the direct-break
// edge, captured before any exit register is published. SCC leaves as the header phi.
inline void mark_loop_exit(RegState& rs, int r, bool written_in_condition,
                           const LoopCheckMarks& check, const LoopCheckMarks* body = nullptr) {
    const bool fabricated = !written_in_condition || check.fabricated.contains(r) ||
                            (body && body->fabricated.contains(r));
    const bool memory = check.memory.contains(r) || (body && body->memory.contains(r));
    // Both snapshots have been checked against the backedge. A hard input on either edge must
    // stay hard; roots on a valid soft input are requirements, never alternative proofs.
    const auto kept = check.blanket.find(r);
    std::set<uint64_t> roots;
    if (kept != check.blanket.end() &&
        (!body || !body->fabricated.contains(r) || body->blanket.contains(r))) {
        roots = kept->second;
        if (body && body->blanket.contains(r))
            roots.insert(body->blanket.at(r).begin(), body->blanket.at(r).end());
    }
    join_merge_placeholder(rs, r, fabricated, false, roots.empty() ? nullptr : &roots);
    if (memory)
        rs.sreg_memory_pattern.insert(r);
    else
        rs.sreg_memory_pattern.erase(r);
    rs.scc_merge_placeholder = true;
}

inline bool ordinary_portable_wave64_masks(const SpirvCompute& b) {
    return b.is_compute && b.wave_size == 64 && !b.native_subgroup_size &&
           !b.is_fragment_packet() && !b.ngg_workgroup_shell && !b.ngg_one_lane;
}

struct SavedB64MaskSnapshot {
    // Instruction-order architectural writes only; load_state does not use this snapshot.
    bool portable_numeric_replacement = false;
    // Agreement proof for this exact READFIRSTLANE input, captured before emission.
    bool scalar_first_lane_agrees = false;
    // Closed, once-through fragment local-index route; no general broadcast authority.
    bool scalar_local_index_waterfall = false;
    // (root, Bool id). At most a few entries: only the roots this one instruction can overwrite.
    std::vector<std::pair<int, uint32_t>> entries;
    // The instruction's input marks, taken before emission with the masks (#4706).
    ScalarSourceMarks source_marks;
    RelativeBankPublication relative_bank_publication;
};

inline void expire_saved_b64_mask(RegState& rs, const SavedB64MaskSnapshot& before, int reg,
                                  int preserved_pair = -1, bool mask_slot_reload = false) {
    // CONFIDENCE: HIGH -- either numeric half replacement ends this complete Wave64 generation.
    // The established portable mask-slot spelling is separate: this leaf neither authenticates
    // its physical halves nor changes that reload path's historical alias policy (#4781).
    for (int root : {reg - 1, reg}) {
        if (root != reg && !rs.native_saved_mask_words &&
            (!before.portable_numeric_replacement || mask_slot_reload))
            continue;
        if (root < 0 || root > 105 || root == preserved_pair ||
            (root != reg && rs.portable_mask_half_types.words.contains(root)))
            continue;
        const auto mask = rs.sreg_bool.find(root);
        if (mask == rs.sreg_bool.end() || rs.sreg_bool_b32.contains(root)) continue;
        bool stale = false;
        for (const auto& [snapshot_root, snapshot_id] : before.entries)
            if (snapshot_root == root && snapshot_id == mask->second) stale = true;
        if (!stale) continue;
        rs.sreg_bool.erase(root);
        rs.sreg_bool_narrowed.erase(root);
    }
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
    // not mistake s_cmpk's source register for a redefinition. The instruction-effects table owns
    // the answer; an encoding it does not know is reported as a writer.
    const Rdna2InstructionEffects& effects = rdna2_instruction_effects(Rdna2Format::SOPK, opcode);
    return !effects.known() || effects.sdst_dwords != 0;
}
uint32_t scalar_write_width(const Rdna2Inst& in);
bool rdna2_may_write_guest_memory(const Rdna2Inst& in);

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
        case Rdna2Format::SOP2:
        case Rdna2Format::SOPK: {
            // The instruction-effects table owns the scalar ALU destination widths (a conditional
            // s_cmov_b64 still counts both words). An encoding it does not know is charged a pair,
            // the widest scalar ALU destination.
            const Rdna2InstructionEffects& effects = rdna2_instruction_effects(in);
            return effects.known() ? effects.sdst_dwords : 2u;
        }
        case Rdna2Format::SMEM: {
            // What SDATA receives (the effects table): a load's dwords, the s_memtime /
            // s_memrealtime pair, a gfx10.1 atomic's returned value when GLC asks for it. Nothing
            // for a store, a probe, cache control or a reserved encoding.
            return rdna2_smem_destination_dwords(in);
        }
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP3: return rdna2_lane_read_writes_sgpr(in) ? 1u : 0u;
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

// snapshot_saved_b64_masks() lives in rdna2_cfg_support.hpp (it reads scalar_source_marks()).

inline bool wqm_has_numeric_destination(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x09 || in.opcode == 0x0a) &&
        in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 &&
        in.dst.value <= (in.opcode == 0x0a ? 104 : 105) &&
        (in.opcode == 0x09 || (in.dst.value & 1) == 0);
}

inline bool wqm_has_exact_legacy_inline_mask_source(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP1 || (in.opcode != 0x09 && in.opcode != 0x0a) ||
        in.src[0].kind != OperandKind::InlineInt) return false;
    if (in.opcode == 0x09) return in.src[0].value == 0 || in.src[0].value == -1;
    // Preserve previously exact saved-mask consumers, not every old identity case. An inline
    // B64 integer is sign-extended, so its high word is already quad-closed. Its low word must be
    // invariant under nibble widening too (e.g. 0, 15, -1, -16). DATA/SCC support for these legacy
    // mask spellings is separate work; do not silently drop the predicate that their users need.
    const uint32_t bits = static_cast<uint32_t>(in.src[0].value);
    const uint32_t any = bits | (bits >> 1) | (bits >> 2) | (bits >> 3);
    return ((any & 0x11111111u) * 15u) == bits;
}

inline bool wqm_has_intrinsic_numeric_source(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x09 || in.opcode == 0x0a) &&
        !wqm_has_exact_legacy_inline_mask_source(in) &&
        (in.src[0].kind == OperandKind::InlineInt || in.src[0].kind == OperandKind::Literal ||
         (in.opcode == 0x09 && in.src[0].kind == OperandKind::InlineFloat));
}

// True when this explicit scalar destination is written in the per-lane B64 mask domain.  Keep
// this classification independent of the post-emission SSA maps: folded/data writers such as
// s_getpc_b64 intentionally leave no scalar value behind, so absence from `sreg` cannot identify a
// mask write.  VOP3B's SDST is a mask/carry pair even though the instruction's primary VDST is data.
inline bool scalar_write_is_b64_mask(const Rdna2Inst& in, int base) {
    if (in.fmt == Rdna2Format::SOP1 && in.dst.value == base) {
        switch (in.opcode) {
            case 0x0a:
                return !(wqm_has_numeric_destination(in) &&
                         wqm_has_intrinsic_numeric_source(in));
            case 0x04: case 0x08:
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

// Is `r` carrying a LIVE entry-M0 token (#3133)? Set membership alone is not the predicate: the
// token is the ABSENCE of a value, so a register that has since been given a real word is an
// ordinary tracked scalar with a stale set entry. `operand_bits` asks exactly this question, and
// every control-flow join has to ask the same one or the two disagree about what the register is.
inline bool entry_m0_live(const RegState& rs, int r) {
    return rs.sreg_entry_m0.contains(r) && !rs.sreg.contains(r);
}

// Join for the entry-M0 token at a two-edge scalar merge (#3133). `rs` holds one incoming edge;
// `other_edge_token` is whether the OTHER edge had a live token for `r`.
//
// This exists because the token is the ABSENCE of a value, and that is what makes a join dangerous
// rather than merely imprecise. Every scalar merge in the CFG emitter reads its inputs through a
// `sget` lambda that renders an absent scalar as `uconst(0)`, and then stores the resulting phi back
// into `rs.sreg` -- so without this, a save inside a branch arm comes out of the merge as an
// ordinary TRACKED zero. That is the fabricated word the token exists to withhold, and it also
// silences the `operand_bits` barrier for the rest of the shader, since that barrier is
// `token AND untracked`.
//
// The join is a UNION, not the intersection `merge_ud_alias` uses, and the asymmetry is deliberate.
// A copy alias is a positive claim ("these bits are still the driver's"), so one edge failing to
// assert it must drop it. The token is the opposite: it is a claim that the register has NO usable
// data value, and one edge holding entry-M0 is exactly as unusable as both. The tempting third
// option -- drop the register to plain untracked -- is the one that must NOT be taken: an untracked
// ordinary SGPR is not an error in this model, `operand_bits` returns `uconst(0)` for it with `ok`
// left true, so "untracked" would fabricate the same zero silently instead of rejecting. Measured,
// not assumed (`rdna2_alu_support.hpp`, the SGPR case's final `return b.uconst(0)`).
//
// Returns true when the caller should emit its ordinary value phi. When it returns false, `rs`
// already holds the correct post-join state and NO phi may be emitted for `r`:
//
//   * a live token on EITHER edge -> the register carries the token out of the join and holds no
//     value. A later read rejects loudly, which is right: on one path the word is entry-M0 and on
//     the other it is whatever the register held before, and neither is data this model can supply.
//     A later ordinary write starts a fresh lifetime and ends the token, as it does anywhere else.
//     Marking a maybe-token register token-live can only over-reject, never fabricate -- and the one
//     consumer that is allowed to eat a token, `s_mov_b32 m0, sSRC`, returns M0 to UNTRACKED, which
//     is the strictest state M0 has and the one every movrel/ADDTID/ds_append guard demands.
//   * a live token on NEITHER edge -> the ordinary phi, plus any stale set membership dropped so it
//     cannot over-reject a legitimate later read of the merged value.
inline bool join_entry_m0(RegState& rs, int r, bool other_edge_token) {
    if (entry_m0_live(rs, r) || other_edge_token) {
        rs.sreg.erase(r);                 // the other edge's word cannot speak for this one
        rs.sreg_entry_m0.insert(r);
        return false;
    }
    rs.sreg_entry_m0.erase(r);
    return true;
}

// Post-emission clients know the actual mask width. The generic inventory may reserve a pair,
// but a B32 compare/carry leaves its neighbor's numeric value and every provenance mark intact.
inline uint32_t emitted_scalar_write_width(const RegState& rs, const Rdna2Inst& in, int base,
                                           uint32_t inventory_width) {
    const bool mask_destination =
        (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) && base == in.dst.value) ||
        (vop3_writes_mask_sdst(in) && base == in.sdst.value);
    return mask_destination && rs.sreg_bool_b32.contains(base) ? 1u : inventory_width;
}

// Apply one emitted instruction's input marks to what it wrote (#4706): every destination word, a
// v_writelane spill slot, and SCC when a scalar instruction may have written it.
inline void propagate_merge_placeholder(RegState& rs, const Rdna2Inst& in,
                                        const ScalarSourceMarks& marks,
                                        bool scalar_first_lane_agrees = false,
                                        bool scalar_local_index_waterfall = false,
                                        const RelativeBankPublication& relative_bank = {}) {
    record_vector_word_marks_preserving_cohort(rs, in, marks);
    if (relative_bank.cohort) publish_relative_bank_word(rs, in, marks, relative_bank);
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
        if (in.dst.kind != OperandKind::VGPR || in.src[1].kind != OperandKind::InlineInt) return;
        const std::pair<int, int> slot{in.dst.value, in.src[1].value};
        if (marks.placeholder)
            rs.lane_slot_merge_placeholder.insert(slot);
        else
            rs.lane_slot_merge_placeholder.erase(slot);
        if (marks.memory)
            rs.lane_slot_memory_pattern.insert(slot);
        else
            rs.lane_slot_memory_pattern.erase(slot);
        return;
    }
    // A word this instruction left in the MASK domain (a compare's or a mask logical's Bool, with
    // no data view) is a ballot of this wave: whatever its inputs were, it is consumed per lane
    // like every other mask, and a later data read of it materializes the exact ballot word. So
    // it carries neither mark. Kena's f1d1baa8 compares SMEM-loaded s101 into s[36:37], spills the
    // mask and reloads it for its pc308 projection; a memory mark copied onto that ballot refused
    // the program live on main after #4725.
    // A compare (VOPC, or a VOP3-encoded compare or carry-out) writes a per-lane predicate even
    // when the emitter also keeps a uniform data view of it: Kena's pc226 is
    // `v_cmp_gt_f32_sdwa s[36:37], s101, -2.0`, two scalar operands, s101 from SMEM.
    const bool compare_writer =
        in.fmt == Rdna2Format::VOPC ||
        (in.fmt == Rdna2Format::VOP3 && (in.opcode < 0x100 || vop3_writes_mask_sdst(in)));
    const auto mask_word = [&](int r) {
        if (compare_writer) return true;
        if (rs.sreg.contains(r)) return false;
        if ((r == 106 || r == 107) && rs.vcc) return true;
        return rs.sreg_bool.contains(r) ||
               (r > 0 && rs.sreg_bool.contains(r - 1) && !rs.sreg_bool_b32.contains(r - 1));
    };
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        const uint32_t effective_width = emitted_scalar_write_width(rs, in, base, width);
        for (int r = base; r < base + static_cast<int>(effective_width); ++r) {
            const bool ballot = mask_word(r);
            if (marks.placeholder && !ballot)
                rs.sreg_merge_placeholder.insert(r);
            else
                rs.sreg_merge_placeholder.erase(r);
            if (marks.placeholder && !marks.hard && !ballot)
                rs.sreg_loop_blanket[r] = marks.blanket_roots;
            else
                rs.sreg_loop_blanket.erase(r);
            if (marks.memory && !ballot)
                rs.sreg_memory_pattern.insert(r);
            else
                rs.sreg_memory_pattern.erase(r);
            record_scalar_exec_scope(rs, in, r, ballot, marks, scalar_first_lane_agrees,
                                     scalar_local_index_waterfall);
        }
    });
    // SCC: a marked input marks an operation which may rewrite it. Only a compare from definite
    // inputs clears it here; other scalar writers retain the old mark conservatively.
    const bool scalar_alu = in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                            in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SOPC;
    // S_MOV_B32/B64 copy marked numeric data without writing SCC. Keep the destination's marks
    // above, but neither poison nor refresh the prior condition. CONFIDENCE: HIGH -- RDNA2 70648
    // section 12.3 gives only the destination assignment for these moves, unlike SCC writers.
    const bool move_preserves_scc =
        in.fmt == Rdna2Format::SOP1 &&
        (in.opcode == kSop1OpcodeMovB32 || in.opcode == kSop1OpcodeMovB64);
    if (rs.vector_wave_size == 32 || rs.track_scc_definition) {
        record_wave32_scalar_condition(rs, in, marks);
        return;
    }
    if (scalar_alu && marks.placeholder && !move_preserves_scc)
        rs.scc_merge_placeholder = true;
    else if (in.fmt == Rdna2Format::SOPC)
        rs.scc_merge_placeholder = false;
}

inline void record_scalar_write(RegState& rs, const Rdna2Inst& in,
                         bool allow_compute_scalar_vcc_bridge,
                         const SavedB64MaskSnapshot& saved_b64_masks_before) {
    propagate_merge_placeholder(rs, in, saved_b64_masks_before.source_marks,
                                saved_b64_masks_before.scalar_first_lane_agrees,
                                saved_b64_masks_before.scalar_local_index_waterfall,
                                saved_b64_masks_before.relative_bank_publication);
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        for (uint32_t w = 0; w < width; ++w)
            rs.unavailable_saved_mask_words.erase(base + static_cast<int>(w));
    });
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
        const uint32_t effective_width = emitted_scalar_write_width(rs, in, base, width);
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
        // Raw-SGPR WQM is classified from the emitter's explicit numeric result, not the
        // syntax-only mask inventory. The numeric lowering clears every overlapping Bool alias
        // before publishing both new words; the Boolean WQM path erases both scalar words.
        const bool numeric_wqm_write = wqm_has_numeric_destination(in) &&
            rs.sreg.contains(base) &&
            (in.opcode == 0x09 || rs.sreg.contains(base + 1)) &&
            !rs.sreg_bool.contains(base);
        const bool writes_b64_mask = effective_width == 2 && !vopc_b32_mask &&
            !numeric_wqm_write && scalar_write_is_b64_mask(in, base);
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
        // emit_alu's V_READLANE erases its destination's Bool unless it reloaded a mask slot into
        // it, so a Bool there now is this instruction's. See expire_saved_b64_mask.
        const bool publishes_reloaded_mask = in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 &&
                                             base == in.dst.value && rs.sreg_bool.contains(base);
        const bool publishes_portable_half =
            rs.portable_mask_half_types.words.contains(base) &&
            (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 ||
             in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32);
        // Wave32 compute (the only stage the VCC bridge flag admits): every mask is one word, so a
        // mask reloaded into a register that held a B32 mask stays one. Dropping the marker here
        // also dropped the Bool, and a data read then took the untracked SGPR's silent 0 (#4607).
        // Within one block: the Wave32 dispatcher still loses the reload at an edge (#4613).
        const bool keeps_b32_reload = publishes_reloaded_mask && allow_compute_scalar_vcc_bridge &&
                                      rs.sreg_bool_b32.contains(base);
        for (uint32_t word = 0; word < effective_width; ++word) {
            const int reg = base + static_cast<int>(word);
            if (!publishes_wave64_mask_half || reg != base) {
                expire_wave64_mask_half(rs, reg, writes_b64_mask ? base : -1);
                expire_saved_b64_mask(
                    rs, saved_b64_masks_before, reg,
                    writes_b64_mask || publishes_reloaded_mask || publishes_portable_half ? base
                                                                                          : -1,
                    publishes_reloaded_mask);
            }
            if (!rs.sreg_bool_b32.contains(reg) ||
                ((writes_b32_mask || keeps_b32_reload) && reg == base))
                continue;
            rs.sreg_bool_b32.erase(reg);
            if (!writes_b64_mask || reg != base) {
                rs.sreg_bool.erase(reg);
                rs.sreg_bool_narrowed.erase(reg);
                if (reg == 106) rs.vcc = 0;
            }
        }
        if (!writes_b32_mask && !keeps_b32_reload) rs.sreg_bool_b32.erase(base);
        // VCC_LO's mask is mirrored in rs.vcc, which every implicit-VCC consumer (v_cndmask,
        // vccz/vccnz, carry-in) reads. The reload published only the Bool, so mirror it, and drop
        // the uniformity proof of the compare it replaces (#4607 review).
        if (keeps_b32_reload && base == 106) {
            rs.vcc = rs.sreg_bool.at(106);
            rs.vcc_wave_uniform = 0;
        }
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

// Exact-PC complete owned windows are a separate admission from resource-table SMEM.
bool emit_owned_raw_window(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, uint32_t words,
                           bool& ok);
bool owned_or_portable_readfirstlane_enabled(const SpirvCompute& b);
bool owned_or_portable_readfirstlane_candidate(const SpirvCompute& b, const Rdna2Inst& in);

// emit_alu lives in rdna2_emit_alu.cpp -- 7,676 lines of instruction-family translation that is
// the single largest thing in the recompiler. Its callers stayed behind, so the declaration lives
// here. The DEFAULT ARGUMENTS are stated here and nowhere else: C++ permits them once per scope,
// so the definition carries the same parameters without them.
bool emit_alu(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_exec_update,
              const std::unordered_set<uint32_t>* safe_execz = nullptr, bool allow_smem = false,
              const ShaderResourceTable* rt = nullptr, bool allow_wave = false);

// emit_body and the CFG state machine it drives live in rdna2_emit_cfg.cpp. As with emit_alu, the
// default arguments are stated here and nowhere else.
struct PacketVgprDefinedness;
struct FragmentPacketMaskRequirements;
class NativeBarrierHalfProof;
bool emit_cfg_state_machine(
    SpirvCompute& b, RegState& initial, const std::vector<Rdna2Inst>& ins,
    const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt, bool allow_exec_update,
    bool allow_smem, const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn,
    const uint32_t* code, size_t dwords, uint32_t initial_active = 0,
    bool synchronize_lds_fminmax = false,
    const std::function<int(RegState&, const Rdna2Inst&)>& packet_instruction = {},
    PacketVgprDefinedness* packet_definedness = nullptr,
    const FragmentPacketMaskRequirements* packet_masks = nullptr,
    NativeBarrierHalfProof* barrier_halves = nullptr);

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

std::unordered_set<uint32_t> rdna2_proven_smem_pointer_loads(const std::vector<Rdna2Inst>&);
bool emit_owned_raw_x2(SpirvCompute&, RegState&, const Rdna2Inst&, const ShaderResourceTable*,
                       bool&);
void end_smem_vcc_data_lifetime(RegState&, const Rdna2Inst&, uint32_t);
}  // namespace prosper::gpu
