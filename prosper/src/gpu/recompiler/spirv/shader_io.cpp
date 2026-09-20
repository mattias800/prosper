// Generated move, not new code: `SpirvCompute`'s method bodies, lifted out of
// gpu/recompiler/rdna2_to_spirv_internal.hpp. outline_methods.py checked each body byte for byte.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

void     SpirvCompute::store_output(uint32_t bits) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_f32, p, v_out, uconst(0), gidx});
        put(code, Op_Store, {p, bcf(bits)});
    }

void     SpirvCompute::store_output_pred(uint32_t bits, uint32_t exec_bool) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_f32, p, v_out, uconst(0), gidx});
        uint32_t old = id(); put(code, Op_Load, {t_f32, old, p});
        uint32_t sel = id(); put(code, Op_Select, {t_f32, sel, exec_bool, bcf(bits), old});
        put(code, Op_Store, {p, sel});
    }

void SpirvCompute::export_color(uint32_t mrt, uint32_t r, uint32_t g, uint32_t bl, uint32_t a) {
        if (mrt >= v_color.size() || !v_color[mrt]) return;
        // Fragment I/O tap (PROSPER_FS_TAP): if an intermediate was snapshotted at the tapped PC, store THAT
        // as the MRT0 colour instead of the shader's real colour, so the rendered frame visualises the value.
        uint32_t v;
        if (tap_vec && mrt == 0) { v = tap_vec; }
        else { v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(r), bcf(g), bcf(bl), bcf(a)}); }
        put(code, Op_Store, {v_color[mrt], v});
    }

void SpirvCompute::export_depth(uint32_t z_bits) {
        if (!v_fragdepth) {
            uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_Output, t_f32});
            v_fragdepth = id(); put(types, Op_Variable, {t_ptr, v_fragdepth, SC_Output});
            put(deco, Op_Decorate, {v_fragdepth, Dec_BuiltIn, BI_FragDepth});
            put(exec, Op_ExecutionMode, {f_main, EM_DepthReplacing});
            iface.push_back(v_fragdepth);
        }
        put(code, Op_Store, {v_fragdepth, bcf(z_bits)});
    }

void SpirvCompute::export_sample_mask(uint32_t mask_bits) {
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

void SpirvCompute::export_position(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
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

uint32_t SpirvCompute::frag_input(uint32_t attr) {
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

uint32_t SpirvCompute::interp_read(uint32_t attr, uint32_t chan) {
        uint32_t v = frag_input(attr);
        uint32_t vec = id(); put(code, Op_Load, {t_v4f, vec, v});
        uint32_t e = id(); put(code, Op_CompositeExtract, {t_f32, e, vec, chan}); return bcu(e);
    }

uint32_t SpirvCompute::interp_parameter(uint32_t attr, uint32_t chan, uint32_t selector) {
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

uint32_t SpirvCompute::system_interpolation_component(uint32_t field, uint32_t component) {
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

uint32_t SpirvCompute::fragcoord_var() {
        if (v_fragcoord) return v_fragcoord;
        if (!t_ptr_in_v4f) { t_ptr_in_v4f = id(); put(types, Op_TypePointer, {t_ptr_in_v4f, SC_Input, t_v4f}); }
        v_fragcoord = id(); put(types, Op_Variable, {t_ptr_in_v4f, v_fragcoord, SC_Input});
        put(deco, Op_Decorate, {v_fragcoord, Dec_BuiltIn, BI_FragCoord});
        iface.push_back(v_fragcoord); return v_fragcoord;
    }

uint32_t SpirvCompute::fragcoord_component(uint32_t component) {
        uint32_t value = id(); put(code, Op_Load, {t_v4f, value, fragcoord_var()});
        uint32_t scalar = id(); put(code, Op_CompositeExtract, {t_f32, scalar, value, component});
        return bcu(scalar);
    }

uint32_t SpirvCompute::vtx_output(uint32_t loc) {
        auto it = out_varying.find(loc); if (it != out_varying.end()) return it->second;
        uint32_t v = id(); put(types, Op_Variable, {t_ptr_out_v4f, v, SC_Output});
        put(deco, Op_Decorate, {v, Dec_Location, loc});
        out_varying[loc] = v; iface.push_back(v); return v;
    }

void SpirvCompute::export_param(uint32_t loc, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
        uint32_t v = vtx_output(loc);
        uint32_t vec = id(); putv(code, Op_CompositeConstruct, {t_v4f, vec, bcf(x), bcf(y), bcf(z), bcf(w)});
        put(code, Op_Store, {v, vec});
    }

}  // namespace prosper::gpu
