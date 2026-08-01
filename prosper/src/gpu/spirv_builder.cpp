// spirv_builder.cpp — see spirv_builder.hpp. Emits SPIR-V by assembling the logical module sections
// in the order the spec requires (capabilities, memory model, entry point, exec modes, decorations,
// types/constants/globals, function bodies). Instruction word 0 = (wordCount<<16) | opcode.
#include "spirv_builder.hpp"
#include <cstring>

namespace prosper::gpu {
namespace {

// SPIR-V opcodes.
enum : uint32_t {
    Op_MemoryModel=14, Op_EntryPoint=15, Op_ExecutionMode=16, Op_Capability=17,
    Op_TypeVoid=19, Op_TypeInt=21, Op_TypeFloat=22, Op_TypeVector=23,
    Op_TypeRuntimeArray=29, Op_TypeStruct=30, Op_TypePointer=32, Op_TypeFunction=33,
    Op_Constant=43, Op_Function=54, Op_FunctionEnd=56, Op_Variable=59,
    Op_Load=61, Op_Store=62, Op_AccessChain=65, Op_Decorate=71, Op_MemberDecorate=72,
    Op_CompositeExtract=81, Op_FAdd=129, Op_FMul=133, Op_Any=154, Op_INotEqual=171,
    Op_ULessThan=176, Op_AtomicExchange=229, Op_SelectionMerge=247, Op_Label=248,
    Op_Branch=249, Op_BranchConditional=250, Op_Return=253,
};
// Enumerants.
enum : uint32_t {
    Cap_Shader=1, Addr_Logical=0, Mem_GLSL450=1, Exec_GLCompute=5, EM_LocalSize=17,
    SC_Input=1, SC_PushConstant=9, SC_StorageBuffer=12, FC_None=0,
    Dec_Block=2, Dec_ArrayStride=6, Dec_BuiltIn=11, Dec_Binding=33, Dec_DescriptorSet=34, Dec_Offset=35,
    BI_GlobalInvocationId=28,
};

struct Emitter {
    std::vector<uint32_t> caps, mem, entry, exec, deco, types, code;
    uint32_t next_id = 1;
    uint32_t id() { return next_id++; }

    static void put(std::vector<uint32_t>& s, uint32_t op, std::initializer_list<uint32_t> ops) {
        s.push_back(((uint32_t)(ops.size() + 1) << 16) | op);
        for (uint32_t o : ops) s.push_back(o);
    }
    // Instruction whose operands are a prebuilt vector (for variadic ops like EntryPoint/AccessChain).
    static void putv(std::vector<uint32_t>& s, uint32_t op, const std::vector<uint32_t>& ops) {
        s.push_back(((uint32_t)(ops.size() + 1) << 16) | op);
        s.insert(s.end(), ops.begin(), ops.end());
    }

    std::vector<uint32_t> assemble() {
        std::vector<uint32_t> m;
        m.push_back(0x07230203u);   // magic
        m.push_back(0x00010300u);   // version 1.3
        m.push_back(0u);            // generator
        m.push_back(next_id);       // bound
        m.push_back(0u);            // schema
        for (auto* s : {&caps, &mem, &entry, &exec, &deco, &types, &code})
            m.insert(m.end(), s->begin(), s->end());
        return m;
    }
};

// Pack a C string into SPIR-V literal words (little-endian, NUL-terminated, word-padded).
void push_string(std::vector<uint32_t>& v, const char* s) {
    size_t len = std::strlen(s);
    for (size_t i = 0; i <= len; i += 4) {
        uint32_t w = 0;
        for (size_t k = 0; k < 4; k++) { size_t j = i + k; if (j <= len) w |= (uint32_t)(uint8_t)s[j] << (8 * k); }
        v.push_back(w);
    }
}
uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

}  // namespace

std::vector<uint32_t> build_compute_scale_bias(float scale, float bias) {
    Emitter e;

    // Type/constant/global ids.
    uint32_t t_void = e.id(), t_fn = e.id(), t_f32 = e.id(), t_u32 = e.id(), t_v3u = e.id();
    uint32_t t_ptr_in_v3u = e.id(), v_gid = e.id();
    uint32_t t_rta = e.id(), t_struct = e.id(), t_ptr_sb_struct = e.id();
    uint32_t v_in = e.id(), v_out = e.id(), t_ptr_sb_f32 = e.id();
    uint32_t c_u0 = e.id(), c_scale = e.id(), c_bias = e.id();
    uint32_t f_main = e.id(), lbl = e.id();

    // Capabilities + memory model.
    Emitter::put(e.caps, Op_Capability, {Cap_Shader});
    Emitter::put(e.mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});

    // Entry point: GLCompute main, interface = [gid] (Input; SPIR-V 1.3 lists only In/Out).
    { std::vector<uint32_t> o{Exec_GLCompute, f_main}; push_string(o, "main"); o.push_back(v_gid);
      Emitter::putv(e.entry, Op_EntryPoint, o); }
    Emitter::put(e.exec, Op_ExecutionMode, {f_main, EM_LocalSize, 64, 1, 1});

    // Decorations.
    Emitter::put(e.deco, Op_Decorate, {v_gid, Dec_BuiltIn, BI_GlobalInvocationId});
    Emitter::put(e.deco, Op_Decorate, {t_rta, Dec_ArrayStride, 4});
    Emitter::put(e.deco, Op_MemberDecorate, {t_struct, 0, Dec_Offset, 0});
    Emitter::put(e.deco, Op_Decorate, {t_struct, Dec_Block});
    Emitter::put(e.deco, Op_Decorate, {v_in, Dec_DescriptorSet, 0});
    Emitter::put(e.deco, Op_Decorate, {v_in, Dec_Binding, 0});
    Emitter::put(e.deco, Op_Decorate, {v_out, Dec_DescriptorSet, 0});
    Emitter::put(e.deco, Op_Decorate, {v_out, Dec_Binding, 1});

    // Types / constants / globals.
    Emitter::put(e.types, Op_TypeVoid, {t_void});
    Emitter::put(e.types, Op_TypeFunction, {t_fn, t_void});
    Emitter::put(e.types, Op_TypeFloat, {t_f32, 32});
    Emitter::put(e.types, Op_TypeInt, {t_u32, 32, 0});
    Emitter::put(e.types, Op_TypeVector, {t_v3u, t_u32, 3});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_in_v3u, SC_Input, t_v3u});
    Emitter::put(e.types, Op_Variable, {t_ptr_in_v3u, v_gid, SC_Input});
    Emitter::put(e.types, Op_TypeRuntimeArray, {t_rta, t_f32});
    Emitter::put(e.types, Op_TypeStruct, {t_struct, t_rta});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_sb_struct, SC_StorageBuffer, t_struct});
    Emitter::put(e.types, Op_Variable, {t_ptr_sb_struct, v_in, SC_StorageBuffer});
    Emitter::put(e.types, Op_Variable, {t_ptr_sb_struct, v_out, SC_StorageBuffer});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_sb_f32, SC_StorageBuffer, t_f32});
    Emitter::put(e.types, Op_Constant, {t_u32, c_u0, 0});
    Emitter::put(e.types, Op_Constant, {t_f32, c_scale, fbits(scale)});
    Emitter::put(e.types, Op_Constant, {t_f32, c_bias, fbits(bias)});

    // Function body: gidx = gid.x; b[gidx] = a[gidx]*scale + bias.
    Emitter::put(e.code, Op_Function, {t_void, f_main, FC_None, t_fn});
    Emitter::put(e.code, Op_Label, {lbl});
    uint32_t ld_gid = e.id(); Emitter::put(e.code, Op_Load, {t_v3u, ld_gid, v_gid});
    uint32_t gidx   = e.id(); Emitter::put(e.code, Op_CompositeExtract, {t_u32, gidx, ld_gid, 0});
    uint32_t p_a    = e.id(); Emitter::putv(e.code, Op_AccessChain, {t_ptr_sb_f32, p_a, v_in, c_u0, gidx});
    uint32_t a      = e.id(); Emitter::put(e.code, Op_Load, {t_f32, a, p_a});
    uint32_t mul    = e.id(); Emitter::put(e.code, Op_FMul, {t_f32, mul, a, c_scale});
    uint32_t res    = e.id(); Emitter::put(e.code, Op_FAdd, {t_f32, res, mul, c_bias});
    uint32_t p_b    = e.id(); Emitter::putv(e.code, Op_AccessChain, {t_ptr_sb_f32, p_b, v_out, c_u0, gidx});
    Emitter::put(e.code, Op_Store, {p_b, res});
    Emitter::put(e.code, Op_Return, {});
    Emitter::put(e.code, Op_FunctionEnd, {});

    return e.assemble();
}

std::vector<uint32_t> build_compute_compare_uvec4() {
    Emitter e;

    const uint32_t t_void = e.id(), t_fn = e.id(), t_bool = e.id(), t_u32 = e.id();
    const uint32_t t_v3u = e.id(), t_v4u = e.id(), t_v4bool = e.id();
    const uint32_t t_ptr_in_v3u = e.id(), v_gid = e.id();
    const uint32_t t_rta = e.id(), t_buffer = e.id(), t_ptr_buffer = e.id();
    const uint32_t v_a = e.id(), v_b = e.id();
    const uint32_t t_ptr_buffer_u32 = e.id(), t_ptr_buffer_v4u = e.id();
    // The changed-flag buffer is its own block. It is bound as a single 4-byte range (one dword per
    // compare target — see live_compute.cpp's compare_flags descriptor), so it is NOT the uvec4
    // runtime array a[]/b[] use. Declaring it as one carried a type-checking defect for the flag's
    // access chain (#1711) and described a 16-byte-stride array over a 4-byte binding.
    //
    // That second half is a correctness bug, not only a conformance one, which is why this is a new
    // block rather than one more index on the old chain. live_compute.cpp requires and enables
    // `robustBufferAccess`. Under the old declaration the flag's runtime array had ArrayStride 16
    // over a 4-byte bound range, so its robust length is floor(4/16) = 0 and element 0 is OUT OF
    // BOUNDS: a conformant driver is permitted to discard the OpAtomicExchange entirely, leaving
    // the comparison's "results differ" flag stuck at zero. That flag is what live_compute.cpp
    // reads back as `gpu_result_unchanged`, and a false "unchanged" makes it SKIP writing the
    // dispatch's result back to guest memory altogether — not merely skip a baseline update.
    // Adding a component index would have kept that property. One uint in its own block removes it.
    const uint32_t t_flag = e.id(), t_ptr_flag = e.id(), v_flag = e.id();
    const uint32_t t_push = e.id(), t_ptr_push = e.id(), v_push = e.id();
    const uint32_t t_ptr_push_u32 = e.id();
    // SPIR-V requires non-aggregate constants to be unique, so the Device scope (1) and the
    // "no semantics" mask (0) reuse the same two %uint constants rather than declaring duplicates.
    // spirv-val does not enforce uniqueness and drivers accept it, which is precisely why a module
    // this file emits should not rely on that.
    const uint32_t c_u0 = e.id(), c_u1 = e.id();
    const uint32_t c_scope_device = c_u1, c_semantics_none = c_u0;
    const uint32_t f_main = e.id(), lbl_entry = e.id(), lbl_compare = e.id();
    const uint32_t lbl_different = e.id(), lbl_equal = e.id(), lbl_done = e.id();

    Emitter::put(e.caps, Op_Capability, {Cap_Shader});
    Emitter::put(e.mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
    {
        std::vector<uint32_t> o{Exec_GLCompute, f_main};
        push_string(o, "main");
        o.push_back(v_gid);
        Emitter::putv(e.entry, Op_EntryPoint, o);
    }
    Emitter::put(e.exec, Op_ExecutionMode, {f_main, EM_LocalSize, 256, 1, 1});

    Emitter::put(e.deco, Op_Decorate, {v_gid, Dec_BuiltIn, BI_GlobalInvocationId});
    Emitter::put(e.deco, Op_Decorate, {t_rta, Dec_ArrayStride, 16});
    Emitter::put(e.deco, Op_MemberDecorate, {t_buffer, 0, Dec_Offset, 0});
    Emitter::put(e.deco, Op_Decorate, {t_buffer, Dec_Block});
    Emitter::put(e.deco, Op_MemberDecorate, {t_flag, 0, Dec_Offset, 0});
    Emitter::put(e.deco, Op_Decorate, {t_flag, Dec_Block});
    for (const auto [variable, binding] :
         {std::pair{v_a, 0u}, std::pair{v_b, 1u}, std::pair{v_flag, 2u}}) {
        Emitter::put(e.deco, Op_Decorate, {variable, Dec_DescriptorSet, 0});
        Emitter::put(e.deco, Op_Decorate, {variable, Dec_Binding, binding});
    }
    Emitter::put(e.deco, Op_MemberDecorate, {t_push, 0, Dec_Offset, 0});
    Emitter::put(e.deco, Op_Decorate, {t_push, Dec_Block});

    Emitter::put(e.types, Op_TypeVoid, {t_void});
    Emitter::put(e.types, Op_TypeFunction, {t_fn, t_void});
    Emitter::put(e.types, Op_TypeInt, {t_u32, 32, 0});
    // OpTypeBool is opcode 20. Keep the tiny emitter's enum limited to opcodes shared by users.
    Emitter::put(e.types, 20, {t_bool});
    Emitter::put(e.types, Op_TypeVector, {t_v3u, t_u32, 3});
    Emitter::put(e.types, Op_TypeVector, {t_v4u, t_u32, 4});
    Emitter::put(e.types, Op_TypeVector, {t_v4bool, t_bool, 4});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_in_v3u, SC_Input, t_v3u});
    Emitter::put(e.types, Op_Variable, {t_ptr_in_v3u, v_gid, SC_Input});
    Emitter::put(e.types, Op_TypeRuntimeArray, {t_rta, t_v4u});
    Emitter::put(e.types, Op_TypeStruct, {t_buffer, t_rta});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_buffer, SC_StorageBuffer, t_buffer});
    Emitter::put(e.types, Op_Variable, {t_ptr_buffer, v_a, SC_StorageBuffer});
    Emitter::put(e.types, Op_Variable, {t_ptr_buffer, v_b, SC_StorageBuffer});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_buffer_u32, SC_StorageBuffer, t_u32});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_buffer_v4u, SC_StorageBuffer, t_v4u});
    Emitter::put(e.types, Op_TypeStruct, {t_flag, t_u32});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_flag, SC_StorageBuffer, t_flag});
    Emitter::put(e.types, Op_Variable, {t_ptr_flag, v_flag, SC_StorageBuffer});
    Emitter::put(e.types, Op_TypeStruct, {t_push, t_u32});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_push, SC_PushConstant, t_push});
    Emitter::put(e.types, Op_Variable, {t_ptr_push, v_push, SC_PushConstant});
    Emitter::put(e.types, Op_TypePointer, {t_ptr_push_u32, SC_PushConstant, t_u32});
    Emitter::put(e.types, Op_Constant, {t_u32, c_u0, 0});
    Emitter::put(e.types, Op_Constant, {t_u32, c_u1, 1});

    Emitter::put(e.code, Op_Function, {t_void, f_main, FC_None, t_fn});
    Emitter::put(e.code, Op_Label, {lbl_entry});
    const uint32_t ld_gid = e.id();
    Emitter::put(e.code, Op_Load, {t_v3u, ld_gid, v_gid});
    const uint32_t index = e.id();
    Emitter::put(e.code, Op_CompositeExtract, {t_u32, index, ld_gid, 0});
    const uint32_t p_count = e.id();
    Emitter::putv(e.code, Op_AccessChain, {t_ptr_push_u32, p_count, v_push, c_u0});
    const uint32_t count = e.id();
    Emitter::put(e.code, Op_Load, {t_u32, count, p_count});
    const uint32_t in_range = e.id();
    Emitter::put(e.code, Op_ULessThan, {t_bool, in_range, index, count});
    Emitter::put(e.code, Op_SelectionMerge, {lbl_done, 0});
    Emitter::put(e.code, Op_BranchConditional, {in_range, lbl_compare, lbl_done});

    Emitter::put(e.code, Op_Label, {lbl_compare});
    const uint32_t p_a = e.id();
    Emitter::putv(e.code, Op_AccessChain, {t_ptr_buffer_v4u, p_a, v_a, c_u0, index});
    const uint32_t a = e.id();
    Emitter::put(e.code, Op_Load, {t_v4u, a, p_a});
    const uint32_t p_b = e.id();
    Emitter::putv(e.code, Op_AccessChain, {t_ptr_buffer_v4u, p_b, v_b, c_u0, index});
    const uint32_t b = e.id();
    Emitter::put(e.code, Op_Load, {t_v4u, b, p_b});
    const uint32_t component_differences = e.id();
    Emitter::put(e.code, Op_INotEqual, {t_v4bool, component_differences, a, b});
    const uint32_t different = e.id();
    Emitter::put(e.code, Op_Any, {t_bool, different, component_differences});
    Emitter::put(e.code, Op_SelectionMerge, {lbl_equal, 0});
    Emitter::put(e.code, Op_BranchConditional, {different, lbl_different, lbl_equal});

    Emitter::put(e.code, Op_Label, {lbl_different});
    // Fold baseline maintenance into the comparison. Identical results issue no large transfer;
    // changed words update themselves while the flag preserves the CPU-visible reduction.
    Emitter::put(e.code, Op_Store, {p_b, a});
    const uint32_t p_flag = e.id();
    // One index: member 0 of the flag block, which IS the uint the atomic exchanges. Indexing
    // v_flag as if it were the uvec4 array walked struct -> runtime array -> uvec4 and then claimed
    // a uint pointer for the result, which spirv-val rejects (#1711).
    Emitter::putv(e.code, Op_AccessChain, {t_ptr_buffer_u32, p_flag, v_flag, c_u0});
    const uint32_t old_flag = e.id();
    Emitter::put(e.code, Op_AtomicExchange,
                 {t_u32, old_flag, p_flag, c_scope_device, c_semantics_none, c_u1});
    Emitter::put(e.code, Op_Branch, {lbl_equal});

    Emitter::put(e.code, Op_Label, {lbl_equal});
    Emitter::put(e.code, Op_Branch, {lbl_done});
    Emitter::put(e.code, Op_Label, {lbl_done});
    Emitter::put(e.code, Op_Return, {});
    Emitter::put(e.code, Op_FunctionEnd, {});

    return e.assemble();
}

} // namespace prosper::gpu
