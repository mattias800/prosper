// Fragment and vertex stage entry declarations and function setup.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {
void SpirvCompute::begin_fragment(const ShaderResourceTable* rt, uint32_t color_mask) {
    bool with_cbufs = rt != nullptr;
    t_void = id();
    t_fn = id();
    t_f32 = id();
    t_u32 = id();
    t_i32 = id();
    t_bool = id();
    t_v4f = id();
    uint32_t t_ptr_out = id();
    for (uint32_t mrt = 0; mrt < v_color.size(); ++mrt)
        if (color_mask & (1u << mrt)) v_color[mrt] = id();
    f_main = id();
    uint32_t lbl = id();
    glsl = id();
    put(caps, Op_Capability, {Cap_Shader});
    declare_float_controls(f_main);
    {
        std::vector<uint32_t> o{glsl};
        pstr(o, "GLSL.std.450");
        putv(extimp, Op_ExtInstImport, o);
    }
    put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
    is_fragment = true;
    guest_stage = GuestShaderStage::Fragment;
    physical_domain = PhysicalExecutionDomain::Invocation;
    desc_set =
        1;   // PS resources live in descriptor set 1 (VS owns set 0) — no cross-stage binding collision
    exec_model = Exec_Fragment;
    for (uint32_t output : v_color)
        if (output) iface.push_back(output);   // EntryPoint deferred
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
    declare_color_outputs(t_ptr_out);
    if (with_cbufs)
        declare_cbufs(
            rt);   // only when the shader has memory ops (keeps no-op renders binding-free)
    put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
    put(code, Op_Label, {lbl});
    cur_block = lbl;
    function_var_insert = code.size();
}

void SpirvCompute::begin_vertex(const ShaderResourceTable* rt) {
    bool with_cbufs = rt != nullptr;
    t_void = id();
    t_fn = id();
    t_f32 = id();
    t_u32 = id();
    t_i32 = id();
    t_bool = id();
    t_v4f = id();
    uint32_t t_ptr_in_i32 = id();
    v_vid = id();
    v_iid = id();
    uint32_t t_pv = id(), t_ptr_out_pv = id();
    v_pos = id();
    t_ptr_out_v4f = id();
    f_main = id();
    uint32_t lbl = id();
    glsl = id();
    put(caps, Op_Capability, {Cap_Shader});
    declare_float_controls(f_main);
    {
        std::vector<uint32_t> o{glsl};
        pstr(o, "GLSL.std.450");
        putv(extimp, Op_ExtInstImport, o);
    }
    put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
    is_vertex = true;
    guest_stage = GuestShaderStage::Vertex;
    physical_domain = PhysicalExecutionDomain::Invocation;
    exec_model = Exec_Vertex;
    iface = {v_vid, v_iid, v_pos};   // EntryPoint deferred to finish()
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
    put(code, Op_Label, {lbl});
    cur_block = lbl;
    function_var_insert = code.size();
}
}   // namespace prosper::gpu
