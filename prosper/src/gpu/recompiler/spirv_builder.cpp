// spirv_builder.cpp — see spirv_builder.hpp. Emits SPIR-V by assembling the logical module sections
// in the order the spec requires (capabilities, memory model, entry point, exec modes, decorations,
// types/constants/globals, function bodies). Instruction word 0 = (wordCount<<16) | opcode.
#include "gpu/recompiler/spirv_builder.hpp"
#include <cstring>

namespace prosper::gpu {
namespace {

// SPIR-V opcodes.
enum : uint32_t {
    Op_MemoryModel=14, Op_EntryPoint=15, Op_ExecutionMode=16, Op_Capability=17,
    Op_TypeVoid=19, Op_TypeBool=20, Op_TypeInt=21, Op_TypeFloat=22, Op_TypeVector=23,
    Op_TypeArray=28, Op_TypeRuntimeArray=29, Op_TypeStruct=30, Op_TypePointer=32, Op_TypeFunction=33,
    Op_Constant=43, Op_Function=54, Op_FunctionEnd=56, Op_Variable=59,
    Op_Load=61, Op_Store=62, Op_AccessChain=65, Op_Decorate=71, Op_MemberDecorate=72,
    Op_CompositeExtract=81, Op_IAdd=128, Op_FAdd=129, Op_ISub=130, Op_IMul=132,
    Op_FMul=133, Op_UDiv=134, Op_UMod=137, Op_Any=154, Op_LogicalAnd=167, Op_Select=169, Op_INotEqual=171,
    Op_ShiftRightLogical=194, Op_ShiftLeftLogical=196, Op_BitwiseOr=197,
    Op_BitwiseXor=198, Op_BitwiseAnd=199, Op_BitCount=205,
    Op_ULessThan=176, Op_AtomicExchange=229, Op_Phi=245, Op_SelectionMerge=247, Op_Label=248,
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

std::vector<uint32_t> build_compute_detile_rgba16f() {
  Emitter e;
  // Integer-only conversion: the source's half bits, including NaNs/subnormals,
  // are preserved by raw buffer loads rather than native float16 operations.
  const auto void_t = e.id(), fn_t = e.id(), bool_t = e.id(), uint_t = e.id(),
             v3_t = e.id();
  const auto input_ptr = e.id(), gid = e.id(), array_t = e.id(),
             block_t = e.id();
  const auto block_ptr = e.id(), word_ptr = e.id(), source = e.id(),
             output = e.id();
  const auto main = e.id(), entry = e.id(), body = e.id(), done = e.id();
  Emitter::put(e.caps, Op_Capability, {Cap_Shader});
  Emitter::put(e.mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
  std::vector<uint32_t> ep{Exec_GLCompute, main};
  push_string(ep, "main");
  ep.push_back(gid);
  Emitter::putv(e.entry, Op_EntryPoint, ep);
  Emitter::put(e.exec, Op_ExecutionMode, {main, EM_LocalSize, 128, 1, 1});
  Emitter::put(e.deco, Op_Decorate, {gid, Dec_BuiltIn, BI_GlobalInvocationId});
  Emitter::put(e.deco, Op_Decorate, {array_t, Dec_ArrayStride, 4});
  Emitter::put(e.deco, Op_Decorate, {block_t, Dec_Block});
  Emitter::put(e.deco, Op_MemberDecorate, {block_t, 0, Dec_Offset, 0});
  for (auto [variable, binding] :
       {std::pair{source, 0u}, std::pair{output, 1u}}) {
    Emitter::put(e.deco, Op_Decorate, {variable, Dec_DescriptorSet, 0});
    Emitter::put(e.deco, Op_Decorate, {variable, Dec_Binding, binding});
  }
  Emitter::put(e.types, Op_TypeVoid, {void_t});
  Emitter::put(e.types, Op_TypeFunction, {fn_t, void_t});
  Emitter::put(e.types, Op_TypeBool, {bool_t});
  Emitter::put(e.types, Op_TypeInt, {uint_t, 32, 0});
  Emitter::put(e.types, Op_TypeVector, {v3_t, uint_t, 3});
  Emitter::put(e.types, Op_TypePointer, {input_ptr, SC_Input, v3_t});
  Emitter::put(e.types, Op_Variable, {input_ptr, gid, SC_Input});
  Emitter::put(e.types, Op_TypeRuntimeArray, {array_t, uint_t});
  Emitter::put(e.types, Op_TypeStruct, {block_t, array_t});
  Emitter::put(e.types, Op_TypePointer, {block_ptr, SC_StorageBuffer, block_t});
  Emitter::put(e.types, Op_TypePointer, {word_ptr, SC_StorageBuffer, uint_t});
  Emitter::put(e.types, Op_Variable, {block_ptr, source, SC_StorageBuffer});
  Emitter::put(e.types, Op_Variable, {block_ptr, output, SC_StorageBuffer});
  std::vector<std::pair<uint32_t, uint32_t>> constants;
  auto constant = [&](uint32_t value) {
    for (auto [v, id] : constants)
      if (v == value)
        return id;
    const auto id = e.id();
    constants.emplace_back(value, id);
    Emitter::put(e.types, Op_Constant, {uint_t, id, value});
    return id;
  };
  auto binary = [&](uint32_t op, uint32_t a, uint32_t b) {
    const auto id = e.id();
    Emitter::put(e.code, op, {uint_t, id, a, b});
    return id;
  };
  auto less = [&](uint32_t a, uint32_t b) {
    const auto id = e.id();
    Emitter::put(e.code, Op_ULessThan, {bool_t, id, a, b});
    return id;
  };
  auto select = [&](uint32_t condition, uint32_t yes, uint32_t no) {
    const auto id = e.id();
    Emitter::put(e.code, Op_Select, {uint_t, id, condition, yes, no});
    return id;
  };
  auto load = [&](uint32_t variable, uint32_t index) {
    const auto pointer = e.id(), value = e.id();
    Emitter::put(e.code, Op_AccessChain,
                 {word_ptr, pointer, variable, constant(0), index});
    Emitter::put(e.code, Op_Load, {uint_t, value, pointer});
    return value;
  };
  Emitter::put(e.code, Op_Function, {void_t, main, FC_None, fn_t});
  Emitter::put(e.code, Op_Label, {entry});
  uint32_t parameters[4]{};
  for (uint32_t i = 0; i < 4; ++i)
    parameters[i] = load(source, constant(i));
  const auto id3 = e.id(), index = e.id();
  Emitter::put(e.code, Op_Load, {v3_t, id3, gid});
  Emitter::put(e.code, Op_CompositeExtract, {uint_t, index, id3, 0});
  const auto in_range = less(index, parameters[3]);
  Emitter::put(e.code, Op_SelectionMerge, {done, 0});
  Emitter::put(e.code, Op_BranchConditional, {in_range, body, done});
  Emitter::put(e.code, Op_Label, {body});
  const auto width = parameters[0], height = parameters[1],
             face_words = parameters[2];
  const auto x = binary(Op_UMod, index, width),
             row = binary(Op_UDiv, index, width);
  const auto y = binary(Op_UMod, row, height),
             face = binary(Op_UDiv, row, height);
  const auto blocks_per_row = binary(
      Op_ShiftRightLogical, binary(Op_IAdd, width, constant(127)), constant(7));
  const auto block =
      binary(Op_IAdd,
             binary(Op_IMul, binary(Op_ShiftRightLogical, y, constant(6)),
                    blocks_per_row),
             binary(Op_ShiftRightLogical, x, constant(7)));
  uint32_t byte_offset = constant(0);
  for (uint32_t bit = 3; bit < 16; ++bit) {
    const auto equation = load(source, constant(bit + 4));
    const auto xm = binary(Op_BitwiseAnd, equation, constant(65535));
    const auto ym = binary(Op_ShiftRightLogical, equation, constant(16));
    // Parity of the XOR is parity(x & mask) XOR parity(y & mask).
    const auto coordinate = binary(Op_BitwiseXor, binary(Op_BitwiseAnd, x, xm),
                                   binary(Op_BitwiseAnd, y, ym));
    const auto population = e.id();
    Emitter::put(e.code, Op_BitCount, {uint_t, population, coordinate});
    byte_offset = binary(Op_BitwiseOr, byte_offset,
                         binary(Op_ShiftLeftLogical,
                                binary(Op_BitwiseAnd, population, constant(1)),
                                constant(bit)));
  }
  const auto address = binary(
      Op_IAdd, constant(20),
      binary(Op_IAdd, binary(Op_IMul, face, face_words),
             binary(Op_IAdd, binary(Op_ShiftLeftLogical, block, constant(14)),
                    binary(Op_ShiftRightLogical, byte_offset, constant(2)))));
  const uint32_t raw[2] = {load(source, address),
                           load(source, binary(Op_IAdd, address, constant(1)))};
  uint32_t packed = constant(0);
  for (uint32_t channel = 0; channel < 4; ++channel) {
    const auto half = binary(Op_BitwiseAnd,
                             binary(Op_ShiftRightLogical, raw[channel / 2],
                                    constant((channel % 2) * 16)),
                             constant(65535));
    // All evaluated shifts stay in range, even for values subsequently rejected
    // by OpSelect. Only exponents 6..14 use the arithmetic result.
    const auto exponent =
        binary(Op_BitwiseAnd, binary(Op_ShiftRightLogical, half, constant(10)),
               constant(15));
    const auto shift = binary(Op_ISub, constant(25), exponent);
    const auto numerator = binary(
        Op_IAdd,
        binary(Op_IMul,
               binary(Op_IAdd, binary(Op_BitwiseAnd, half, constant(1023)),
                      constant(1024)),
               constant(255)),
        binary(Op_ShiftLeftLogical, constant(1),
               binary(Op_ISub, shift, constant(1))));
    auto value =
        select(less(half, constant(0x3c00)),
               binary(Op_ShiftRightLogical, numerator, shift), constant(255));
    value = select(less(half, constant(0x7c00)), value, constant(0));
    value = select(less(half, constant(0x1800)), constant(0), value);
    packed = binary(Op_BitwiseOr, packed,
                    binary(Op_ShiftLeftLogical, value, constant(channel * 8)));
  }
  const auto destination = e.id();
  Emitter::put(e.code, Op_AccessChain,
               {word_ptr, destination, output, constant(0), index});
  Emitter::put(e.code, Op_Store, {destination, packed});
  Emitter::put(e.code, Op_Branch, {done});
  Emitter::put(e.code, Op_Label, {done});
  Emitter::put(e.code, Op_Return, {});
  Emitter::put(e.code, Op_FunctionEnd, {});
  return e.assemble();
}

std::vector<uint32_t> build_compute_retile_words(RetileShaderKind kind) {
    const bool volume = kind == RetileShaderKind::Volume3D;
    const bool paired = kind == RetileShaderKind::Paired16Array;
    Emitter e;
    const auto void_t = e.id(), fn_t = e.id(), bool_t = e.id(), uint_t = e.id();
    const auto v3_t = e.id(), input_ptr = e.id(), gid = e.id();
    const auto array_t = e.id(), block_t = e.id(), block_ptr = e.id(), word_ptr = e.id();
    const auto source = e.id(), output = e.id(), push_array = e.id(), push_t = e.id();
    const auto push_ptr = e.id(), push_word = e.id(), push = e.id();
    const auto main = e.id(), entry = e.id(), body = e.id(), done = e.id();
    const auto read_pixel = e.id(), pixel_done = e.id();
    Emitter::put(e.caps, Op_Capability, {Cap_Shader});
    Emitter::put(e.mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
    std::vector<uint32_t> ep{Exec_GLCompute, main};
    push_string(ep, "main"); ep.push_back(gid);
    Emitter::putv(e.entry, Op_EntryPoint, ep);
    Emitter::put(e.exec, Op_ExecutionMode, {main, EM_LocalSize, 128, 1, 1});
    Emitter::put(e.deco, Op_Decorate, {gid, Dec_BuiltIn, BI_GlobalInvocationId});
    for (auto array : {array_t, push_array})
        Emitter::put(e.deco, Op_Decorate, {array, Dec_ArrayStride, 4});
    for (auto block : {block_t, push_t}) {
        Emitter::put(e.deco, Op_Decorate, {block, Dec_Block});
        Emitter::put(e.deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
    }
    for (auto [variable, binding] : {std::pair{source, 0u}, std::pair{output, 1u}}) {
        Emitter::put(e.deco, Op_Decorate, {variable, Dec_DescriptorSet, 0});
        Emitter::put(e.deco, Op_Decorate, {variable, Dec_Binding, binding});
    }
    Emitter::put(e.types, Op_TypeVoid, {void_t});
    Emitter::put(e.types, Op_TypeFunction, {fn_t, void_t});
    Emitter::put(e.types, Op_TypeBool, {bool_t});
    Emitter::put(e.types, Op_TypeInt, {uint_t, 32, 0});
    Emitter::put(e.types, Op_TypeVector, {v3_t, uint_t, 3});
    Emitter::put(e.types, Op_TypePointer, {input_ptr, SC_Input, v3_t});
    Emitter::put(e.types, Op_Variable, {input_ptr, gid, SC_Input});
    Emitter::put(e.types, Op_TypeRuntimeArray, {array_t, uint_t});
    Emitter::put(e.types, Op_TypeStruct, {block_t, array_t});
    Emitter::put(e.types, Op_TypePointer, {block_ptr, SC_StorageBuffer, block_t});
    Emitter::put(e.types, Op_TypePointer, {word_ptr, SC_StorageBuffer, uint_t});
    Emitter::put(e.types, Op_Variable, {block_ptr, source, SC_StorageBuffer});
    Emitter::put(e.types, Op_Variable, {block_ptr, output, SC_StorageBuffer});
    std::vector<std::pair<uint32_t, uint32_t>> constants;
    auto constant = [&](uint32_t value) {
        for (auto [v, id] : constants) if (v == value) return id;
        const auto id = e.id(); constants.emplace_back(value, id);
        Emitter::put(e.types, Op_Constant, {uint_t, id, value}); return id;
    };
    Emitter::put(e.types, Op_TypeArray, {push_array, uint_t, constant(26)});
    Emitter::put(e.types, Op_TypeStruct, {push_t, push_array});
    Emitter::put(e.types, Op_TypePointer, {push_ptr, SC_PushConstant, push_t});
    Emitter::put(e.types, Op_TypePointer, {push_word, SC_PushConstant, uint_t});
    Emitter::put(e.types, Op_Variable, {push_ptr, push, SC_PushConstant});
    auto binary = [&](uint32_t op, uint32_t a, uint32_t b) {
        const auto id = e.id(); Emitter::put(e.code, op, {uint_t, id, a, b}); return id;
    };
    auto load = [&](uint32_t variable, uint32_t pointer_type, uint32_t index) {
        const auto pointer = e.id(), value = e.id();
        Emitter::put(e.code, Op_AccessChain,
                     {pointer_type, pointer, variable, constant(0), index});
        Emitter::put(e.code, Op_Load, {uint_t, value, pointer}); return value;
    };
    Emitter::put(e.code, Op_Function, {void_t, main, FC_None, fn_t});
    Emitter::put(e.code, Op_Label, {entry});
    uint32_t params[6]{};
    for (uint32_t i = 0; i < 6; ++i) params[i] = load(push, push_word, constant(i));
    uint32_t depth = 0, block_depth = 0, blocks_y = 0, block_shift = constant(14);
    if (volume) {
        depth = load(push, push_word, constant(22));
        block_depth = load(push, push_word, constant(23));
        blocks_y = load(push, push_word, constant(24));
        block_shift = load(push, push_word, constant(25));
    }
    uint32_t linear_layer_words = 0, tiled_layer_words = 0;
    if (paired) {
        depth = load(push, push_word, constant(22));
        linear_layer_words = load(push, push_word, constant(23));
        tiled_layer_words = load(push, push_word, constant(24));
    }
    const auto id3 = e.id(), word_x = e.id(), y = e.id();
    uint32_t z = 0;
    Emitter::put(e.code, Op_Load, {v3_t, id3, gid});
    Emitter::put(e.code, Op_CompositeExtract, {uint_t, word_x, id3, 0});
    Emitter::put(e.code, Op_CompositeExtract, {uint_t, y, id3, 1});
    if (volume || paired) {
        z = e.id();
        Emitter::put(e.code, Op_CompositeExtract, {uint_t, z, id3, 2});
    }
    const auto row_words = paired ? binary(Op_ShiftRightLogical, params[0], constant(1))
        : binary(Op_ShiftLeftLogical, params[0], params[2]);
    const auto padded_row_words = binary(Op_ShiftLeftLogical, params[5],
        paired ? binary(Op_ISub, params[3], constant(1))
               : binary(Op_IAdd, params[3], params[2]));
    const auto height_mask = binary(Op_ISub,
        binary(Op_ShiftLeftLogical, constant(1), params[4]), constant(1));
    const auto padded_height = binary(Op_ShiftLeftLogical,
        binary(Op_ShiftRightLogical, binary(Op_IAdd, params[1], height_mask), params[4]), params[4]);
    const auto x_valid = e.id(), y_valid = e.id(), valid = e.id();
    Emitter::put(e.code, Op_ULessThan, {bool_t, x_valid, word_x, padded_row_words});
    Emitter::put(e.code, Op_ULessThan, {bool_t, y_valid, y, padded_height});
    Emitter::put(e.code, Op_LogicalAnd, {bool_t, valid, x_valid, y_valid});
    auto in_bounds = valid;
    if (paired) {
        const auto layer_valid = e.id(); in_bounds = e.id();
        Emitter::put(e.code, Op_ULessThan, {bool_t, layer_valid, z, depth});
        Emitter::put(e.code, Op_LogicalAnd, {bool_t, in_bounds, valid, layer_valid});
    }
    Emitter::put(e.code, Op_SelectionMerge, {done, 0});
    Emitter::put(e.code, Op_BranchConditional, {in_bounds, body, done});
    Emitter::put(e.code, Op_Label, {body});
    const auto x = paired ? binary(Op_ShiftLeftLogical, word_x, constant(1))
                          : binary(Op_ShiftRightLogical, word_x, params[2]);
    const auto component_mask = binary(Op_ISub,
        binary(Op_ShiftLeftLogical, constant(1), params[2]), constant(1));
    const auto component = paired ? constant(0) : binary(Op_BitwiseAnd, word_x, component_mask);
    auto block_row = binary(Op_ShiftRightLogical, y, params[4]);
    if (volume)
        block_row = binary(Op_IAdd, block_row,
            binary(Op_IMul, binary(Op_ShiftRightLogical, z, block_depth), blocks_y));
    const auto block = binary(Op_IAdd, binary(Op_IMul, block_row, params[5]),
        binary(Op_ShiftRightLogical, x, params[3]));
    uint32_t byte_offset = binary(Op_ShiftLeftLogical, component, constant(2));
    for (uint32_t bit = 2; bit < 16; ++bit) {
        const auto equation = load(push, push_word, constant(bit + 6));
        uint32_t coordinate;
        if (volume) {
            const auto xy = binary(Op_BitwiseXor,
                binary(Op_BitwiseAnd, x, binary(Op_BitwiseAnd, equation, constant(255))),
                binary(Op_BitwiseAnd, y, binary(Op_BitwiseAnd,
                    binary(Op_ShiftRightLogical, equation, constant(8)), constant(255))));
            coordinate = binary(Op_BitwiseXor, xy,
                binary(Op_BitwiseAnd, z, binary(Op_ShiftRightLogical, equation, constant(16))));
        } else {
            coordinate = binary(Op_BitwiseXor,
                binary(Op_BitwiseAnd, x, binary(Op_BitwiseAnd, equation, constant(65535))),
                binary(Op_BitwiseAnd, y, binary(Op_ShiftRightLogical, equation, constant(16))));
        }
        const auto population = e.id();
        Emitter::put(e.code, Op_BitCount, {uint_t, population, coordinate});
        byte_offset = binary(Op_BitwiseOr, byte_offset,
            binary(Op_ShiftLeftLogical, binary(Op_BitwiseAnd, population, constant(1)), constant(bit)));
    }
    auto address = binary(Op_IAdd, binary(Op_ShiftLeftLogical, block, block_shift),
        binary(Op_ShiftRightLogical, byte_offset, constant(2)));
    if (paired) address = binary(Op_IAdd, address, binary(Op_IMul, z, tiled_layer_words));
    // Padding has no source texel. Branch before the load; selecting zero AFTER
    // an out-of-range read would still violate the source descriptor's bounds.
    const auto pixel_x = e.id(), pixel_y = e.id();
    auto pixel_valid = e.id();
    Emitter::put(e.code, Op_ULessThan, {bool_t, pixel_x, word_x, row_words});
    Emitter::put(e.code, Op_ULessThan, {bool_t, pixel_y, y, params[1]});
    Emitter::put(e.code, Op_LogicalAnd, {bool_t, pixel_valid, pixel_x, pixel_y});
    if (volume) {
        const auto pixel_z = e.id(), all = e.id();
        Emitter::put(e.code, Op_ULessThan, {bool_t, pixel_z, z, depth});
        Emitter::put(e.code, Op_LogicalAnd, {bool_t, all, pixel_valid, pixel_z});
        pixel_valid = all;
    }
    Emitter::put(e.code, Op_SelectionMerge, {pixel_done, 0});
    Emitter::put(e.code, Op_BranchConditional, {pixel_valid, read_pixel, pixel_done});
    Emitter::put(e.code, Op_Label, {read_pixel});
    const auto row = volume ? binary(Op_IAdd, binary(Op_IMul, z, params[1]), y) : y;
    auto linear = binary(Op_IAdd, binary(Op_IMul, row, row_words), word_x);
    if (paired) linear = binary(Op_IAdd, linear, binary(Op_IMul, z, linear_layer_words));
    const auto loaded = load(source, word_ptr, linear);
    Emitter::put(e.code, Op_Branch, {pixel_done});
    Emitter::put(e.code, Op_Label, {pixel_done});
    const auto value = e.id(), destination = e.id();
    Emitter::put(e.code, Op_Phi, {uint_t, value, loaded, read_pixel, constant(0), body});
    Emitter::put(e.code, Op_AccessChain, {word_ptr, destination, output, constant(0), address});
    Emitter::put(e.code, Op_Store, {destination, value});
    Emitter::put(e.code, Op_Branch, {done});
    Emitter::put(e.code, Op_Label, {done});
    Emitter::put(e.code, Op_Return, {});
    Emitter::put(e.code, Op_FunctionEnd, {});
    return e.assemble();
}

std::vector<uint32_t> build_compute_rgba8_to_packed10() {
    Emitter e;
    const auto void_t = e.id(), fn_t = e.id(), uint_t = e.id(), bool_t = e.id();
    const auto vec_t = e.id(), input_ptr = e.id(), gid = e.id();
    const auto array_t = e.id(), block_t = e.id(), block_ptr = e.id(), word_ptr = e.id();
    const auto words = e.id(), push_t = e.id(), push_ptr = e.id(), push_word = e.id();
    const auto push = e.id(), main = e.id(), entry = e.id(), body = e.id(), done = e.id();
    Emitter::put(e.caps, Op_Capability, {Cap_Shader});
    Emitter::put(e.mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
    std::vector<uint32_t> ep{Exec_GLCompute, main};
    push_string(ep, "main"); ep.push_back(gid);
    Emitter::putv(e.entry, Op_EntryPoint, ep);
    Emitter::put(e.exec, Op_ExecutionMode, {main, EM_LocalSize, 128, 1, 1});
    Emitter::put(e.deco, Op_Decorate, {gid, Dec_BuiltIn, BI_GlobalInvocationId});
    Emitter::put(e.deco, Op_Decorate, {array_t, Dec_ArrayStride, 4});
    for (auto block : {block_t, push_t}) {
        Emitter::put(e.deco, Op_Decorate, {block, Dec_Block});
        Emitter::put(e.deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
    }
    Emitter::put(e.deco, Op_Decorate, {words, Dec_DescriptorSet, 0});
    Emitter::put(e.deco, Op_Decorate, {words, Dec_Binding, 0});
    Emitter::put(e.types, Op_TypeVoid, {void_t});
    Emitter::put(e.types, Op_TypeFunction, {fn_t, void_t});
    Emitter::put(e.types, Op_TypeInt, {uint_t, 32, 0});
    Emitter::put(e.types, Op_TypeBool, {bool_t});
    Emitter::put(e.types, Op_TypeVector, {vec_t, uint_t, 3});
    Emitter::put(e.types, Op_TypePointer, {input_ptr, SC_Input, vec_t});
    Emitter::put(e.types, Op_Variable, {input_ptr, gid, SC_Input});
    Emitter::put(e.types, Op_TypeRuntimeArray, {array_t, uint_t});
    Emitter::put(e.types, Op_TypeStruct, {block_t, array_t});
    Emitter::put(e.types, Op_TypePointer, {block_ptr, SC_StorageBuffer, block_t});
    Emitter::put(e.types, Op_TypePointer, {word_ptr, SC_StorageBuffer, uint_t});
    Emitter::put(e.types, Op_Variable, {block_ptr, words, SC_StorageBuffer});
    Emitter::put(e.types, Op_TypeStruct, {push_t, uint_t});
    Emitter::put(e.types, Op_TypePointer, {push_ptr, SC_PushConstant, push_t});
    Emitter::put(e.types, Op_TypePointer, {push_word, SC_PushConstant, uint_t});
    Emitter::put(e.types, Op_Variable, {push_ptr, push, SC_PushConstant});
    auto constant = [&](uint32_t value) {
        auto id = e.id();
        Emitter::put(e.types, Op_Constant, {uint_t, id, value});
        return id;
    };
    auto binary = [&](uint32_t op, uint32_t a, uint32_t b) {
        auto id = e.id();
        Emitter::put(e.code, op, {uint_t, id, a, b});
        return id;
    };
    Emitter::put(e.code, Op_Function, {void_t, main, FC_None, fn_t});
    Emitter::put(e.code, Op_Label, {entry});
    const auto xyz = e.id(), index = e.id(), count_ptr = e.id(), count = e.id(), valid = e.id();
    Emitter::put(e.code, Op_Load, {vec_t, xyz, gid});
    Emitter::put(e.code, Op_CompositeExtract, {uint_t, index, xyz, 0});
    Emitter::put(e.code, Op_AccessChain, {push_word, count_ptr, push, constant(0)});
    Emitter::put(e.code, Op_Load, {uint_t, count, count_ptr});
    Emitter::put(e.code, Op_ULessThan, {bool_t, valid, index, count});
    Emitter::put(e.code, Op_SelectionMerge, {done, 0});
    Emitter::put(e.code, Op_BranchConditional, {valid, body, done});
    Emitter::put(e.code, Op_Label, {body});
    const auto texel_ptr = e.id(), rgba = e.id();
    Emitter::put(e.code, Op_AccessChain, {word_ptr, texel_ptr, words, constant(0), index});
    Emitter::put(e.code, Op_Load, {uint_t, rgba, texel_ptr});
    auto packed = constant(0);
    for (uint32_t c = 0; c < 4; ++c) {
        const auto byte = binary(Op_BitwiseAnd,
            binary(Op_ShiftRightLogical, rgba, constant(c * 8)), constant(255));
        // Nearest integer to byte * scale / 255. The odd denominator has no ties.
        const auto q = binary(Op_UDiv, binary(Op_IAdd,
            binary(Op_IMul, byte, constant(c == 3 ? 3 : 1023)), constant(127)), constant(255));
        packed = binary(Op_BitwiseOr, packed,
            binary(Op_ShiftLeftLogical, q, constant(c * 10)));
    }
    Emitter::put(e.code, Op_Store, {texel_ptr, packed});
    Emitter::put(e.code, Op_Branch, {done});
    Emitter::put(e.code, Op_Label, {done});
    Emitter::put(e.code, Op_Return, {});
    Emitter::put(e.code, Op_FunctionEnd, {});
    return e.assemble();
}

} // namespace prosper::gpu
