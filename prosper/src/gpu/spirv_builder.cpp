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
    Op_CompositeExtract=81, Op_FAdd=129, Op_FMul=133, Op_Label=248, Op_Return=253,
};
// Enumerants.
enum : uint32_t {
    Cap_Shader=1, Addr_Logical=0, Mem_GLSL450=1, Exec_GLCompute=5, EM_LocalSize=17,
    SC_Input=1, SC_StorageBuffer=12, FC_None=0,
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

} // namespace prosper::gpu
