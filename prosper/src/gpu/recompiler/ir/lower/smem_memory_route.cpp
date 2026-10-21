// Pure textual relocation: tools/refactor/extract_case_routes.py.
#include "gpu/recompiler/ir/lower/smem_case_routes.hpp"
#include "gpu/recompiler/ir/lower/smem_index_lowering.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
namespace prosper::gpu::ir {
// NOLINTBEGIN(concurrency-mt-unsafe): existing diagnostic reads retain their timing.
bool SmemCaseRoutes::memory(uint32_t n) {
    // SOFFSET handling. Immediate-only loads encode SOFFSET = SGPR_NULL (125). A register
    // SOFFSET adds an SGPR-computed byte offset:
    //  * a descriptor-only s_load (x4/x8 = V#/T#) with a computed offset can be resolved
    //    by the front end at the exact fetch PC, leaving placeholder words here. Typeless
    //    register-offset x4/x8 loads whose words also feed ordinary ALU require real
    //    backing in every stage and reject below when that backing is not proven.
    //    Uncharted's full-resolution compute writer has several such numeric loads.
    //  * an s_buffer_load (0x8..0xC) with a TRACKED scalar offset is a computed constant-
    //    buffer read (DOLL's bloom-combine PS: per-tap weights at `vcc_lo = 16*(tap/2)`
    //    inside its counted loop, #273) — model it as a DYNAMIC dword index into the
    //    resolved cbuf binding: idx = (soffset + imm) >> 2. An UNTRACKED offset register
    //    (a runtime user-SGPR we have no value for) still rejects — never fold as 0.
    const bool soff_null = (in.src[1].kind == OperandKind::Special && in.src[1].value == 125);
    uint32_t soff_bits = 0;
    bool soff_dyn = false;
    // Numeric register-offset x4/x8 words require real backing in every stage. Descriptor
    // assembly alone is excluded by the full-stream derived-use classification; its exact
    // consumer-PC route remains valid without treating descriptor placeholders as data.
    // The bounded immediate current-byte proof (#3956) is stage-independent and unchanged.
    const bool raw_wide_register_offset_data = rt && (in.opcode == 0x2 || in.opcode == 0x3) &&
                                               !soff_null &&
                                               rs.smem_raw_wide_data_loads.contains(in.pc);
    const ShaderResource* register_source = rt ? rt->by_fetch_pc(in.pc) : nullptr;
    bool owned_nested_source = false;
    if (!b.is_compute) {
        for (const auto& chain : rs.smem_owned_nested_wide_chains) {
            if (chain.parent_pc != in.pc && chain.child_pc != in.pc) continue;
            owned_nested_source = true;
            if (!rt ||
                !owned_nested_snapshot_at(*rt, chain.parent_pc, chain.parent_bytes,
                                          compiler_resource_has_host_data) ||
                !owned_nested_snapshot_at(*rt, chain.child_pc, chain.child_bytes,
                                          compiler_resource_has_host_data)) {
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr, "[smem-reject] pc=%u reason=nested-wide-requires-owned-chain\n",
                            in.pc);
                ok = false;
                return true;
            }
        }
    }
    const bool owned_wide_source = rs.smem_raw_owned_wide_data_loads.contains(in.pc);
    // SOURCE replay retains opaque backing presence separately from readable bytes.
    // These admission checks consume only shape/presence; actual byte readers use the
    // checked resource gateway, and draw execution retains its real ownership checks.
    const ShaderResource* owned_wide_binding =
        owned_wide_source && rt ? owned_raw_snapshot_at(*rt, in.pc, n * sizeof(uint32_t),
                                                        compiler_resource_has_host_data)
                                : nullptr;
    if (owned_wide_source && !owned_wide_binding) {
        if (getenv("PROSPER_DBG"))
            fprintf(stderr, "[smem-reject] pc=%u reason=raw-wide-read-requires-owned-backing\n",
                    in.pc);
        ok = false;
        return true;
    }
    // A source word admitted by the wide-offset proof belongs to this exact realization.
    // A malformed replay table must fail before ordinary scalar-load resource fallback.
    if (!owned_wide_source && rs.smem_raw_offset_scalar_source_pcs.contains(in.pc) &&
        (!register_source || register_source->fetch_pc != in.pc ||
         !valid_owned_raw_snapshot_resource(*register_source, n * sizeof(uint32_t),
                                            compiler_resource_has_host_data(*register_source)))) {
        if (getenv("PROSPER_DBG"))
            fprintf(stderr, "[smem-reject] pc=%u reason=raw-offset-scalar-requires-owned-backing\n",
                    in.pc);
        ok = false;
        return true;
    }
    // This is scalar-address data at Wave32 or Wave64: the full-stream proof requires
    // explicit reaching scalar definitions and excludes intervening implicit VCC writers.
    // No lane-mask projection or uninitialized VCC value supplies the selected address.
    const bool backed_register_wide =
        raw_wide_register_offset_data && rs.smem_raw_register_wide_data_loads.contains(in.pc) &&
        register_source &&
        valid_raw_register_snapshot_resource(*register_source,
                                             compiler_resource_has_host_data(*register_source)) &&
        register_source->fetch_pc == in.pc && register_source->size == n * 4u &&
        // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): SMEM width n is at most 16.
        shader_resource_buffer_binding_bytes(*register_source) >= n * 4u;
    // The bounded current-byte proof below covers an unchanged ENTRY pointer. A pointer
    // loaded into SGPRs by this shader can still name numeric x4/x8 data, but its load is
    // absent from that proof. No stage may turn numeric child bytes into descriptor zeros.
    const bool nested_raw_wide_data = soff_null && sreg_range_written(rs, in.src[0].value, 2);
    if (!b.is_compute && !rt && nested_raw_wide_data && (in.opcode == 0x2 || in.opcode == 0x3) &&
        rs.smem_raw_wide_data_loads.contains(in.pc)) {
        // A missing table is absence of ownership, including chains excluded by the
        // full-program proof. It must not select the legacy descriptor-zero route.
        ok = false;
        return true;
    }
    if (rt && (in.opcode == 0x2 || in.opcode == 0x3) &&
        rs.smem_raw_wide_data_loads.contains(in.pc) &&
        (!soff_null || (soff_null && rs.smem_raw_immediate_wide_data_loads.contains(in.pc)) ||
         nested_raw_wide_data)) {
        // Keep the existing immediate descriptor route for loads outside the bounded
        // current-byte proof. Some established routed tables carry only the resulting
        // resource, without a source binding at this load PC. The may-use census still
        // reports numeric/uncertain reads; it cannot turn that legacy route into a
        // current-byte observation. Register-offset data requires backing in every stage.
        const ShaderResource* exact = rt->by_fetch_pc(in.pc);
        const bool backed_immediate_wide =
            owned_nested_source || owned_wide_binding ||
            (soff_null && !nested_raw_wide_data && (n == 4u || n == 8u) &&
             rs.smem_raw_immediate_wide_data_loads.contains(in.pc) && exact &&
             exact->fetch_pc == in.pc && exact->cls == ResourceClass::ConstantBuffer &&
             shader_resource_buffer_binding_bytes(*exact) >=
                 static_cast<uint64_t>(in.literal) + n * sizeof(uint32_t));
        const bool backed_nested_wide =
            owned_nested_source ||
            (b.is_compute && nested_raw_wide_data &&
             rs.smem_raw_nested_wide_data_loads.contains(in.pc) && exact &&
             exact->fetch_pc == in.pc && exact->cls == ResourceClass::ConstantBuffer &&
             exact->nested_raw_snapshot_admitted && compiler_resource_has_host_data(*exact) &&
             exact->host_data_size >= exact->size &&
             shader_resource_buffer_binding_bytes(*exact) >=
                 static_cast<uint64_t>(in.literal) + n * sizeof(uint32_t));
        if (!backed_immediate_wide && !backed_nested_wide && !backed_register_wide) {
            if (getenv("PROSPER_DBG"))
                fprintf(stderr,
                        "[smem-reject] pc=%u reason=raw-wide-data-requires-backing "
                        "op=0x%x\n",
                        in.pc, in.opcode);
            ok = false;
            return true;
        }
    }
    if (!soff_null && !backed_register_wide) {
        if (rt && (in.opcode == 0x2 || in.opcode == 0x3)) {
            // An x4/x8 DESCRIPTOR load with a non-null soffset leaves here with placeholders
            // and NO SRT tag, so every MIMG that consumes one of its descriptors reports
            // srt_tag=NONE and cannot resolve. Say so when asked: whether the soffset is a
            // TRACKED scalar decides whether an effective offset could be computed at all,
            // and that is the question any fix starts from.
            if (getenv("PROSPER_DBG")) {
                bool tracked = false;
                uint32_t val = 0;
                if (in.src[1].kind == OperandKind::SGPR ||
                    (in.src[1].kind == OperandKind::Special && in.src[1].value >= 106 &&
                     in.src[1].value <= 123)) {
                    auto it = rs.sreg.find(in.src[1].value);
                    tracked = it != rs.sreg.end();
                    if (tracked) val = it->second;
                } else if (in.src[1].kind == OperandKind::InlineInt) {
                    tracked = true;
                    val = (uint32_t)in.src[1].value;
                }
                fprintf(stderr,
                        "[smem-untagged] pc=%u op=0x%x dst=s%d src1-kind=%d src1=%d "
                        "tracked=%d val=%u imm=0x%x\n",
                        in.pc, in.opcode, in.dst.value, static_cast<int>(in.src[1].kind),
                        in.src[1].value, (int)tracked, val, in.literal);
                // Answers the tempting follow-up before anyone spends a build on it: if
                // the soffset were a literal the effective SRT offset would be
                // `soffset + imm` and the bundle could be tagged exactly as the
                // null-soffset form is. Measured on PPSA05684 it is NOT -- the offset is
                // computed, so there is no static tag to attach and constant folding is
                // not the route to these descriptors.
                uint32_t lit = 0;
                auto srt_it = rs.sreg_srt.find(in.src[1].value);
                fprintf(stderr, "[smem-untagged]   soffset: builder-constant=%s srt_tag=%s\n",
                        (tracked && b.uconst_literal(val, &lit)) ? "YES" : "no",
                        srt_it == rs.sreg_srt.end()
                            ? "none"
                            : ("0x" + std::to_string(srt_it->second)).c_str());
            }
            for (uint32_t k = 0; k < n; k++) {
                const int dst = in.dst.value + static_cast<int>(k);
                rs.sreg[dst] = b.uconst(0);
                rs.sreg_srt.erase(dst);   // a new descriptor cannot inherit an older tag
            }
            return true;
        }
        bool tracked = false;
        // A register SOFFSET means the same thing on a raw `s_load_*` as on an
        // `s_buffer_load_*`: a scalar byte offset added to the base. The gate admitted only
        // the buffer opcodes, so a raw load with a perfectly TRACKED offset register was
        // refused without the tracking ever being attempted (#3616). The raw form resolves
        // through the same provenance the buffer form does -- the front half publishes a
        // raw-pointer resource at the load's exact PC and `by_fetch_pc` finds it below --
        // and the dword index is `(soffset + imm) >> 2` either way. An UNTRACKED offset
        // still rejects, for both forms: never fold an unknown runtime offset as 0.
        //
        // Only `s_load_dword` (opcode 0x0), because that is exactly as far as the
        // provenance reaches. The fold publishes raw-pointer ConstantBuffers for
        // single-dword loads and use-proved immediate x2 scalar observations. A dynamic
        // SOFFSET on x2/x4/x8/x16 has no such binding; admitting it here would let it
        // fall to an unrelated binding-2 buffer. The bound is the fold's own.
        if (in.opcode == 0x0 || (in.opcode >= 0x8 && in.opcode <= 0xC)) {
            if (in.src[1].kind == OperandKind::SGPR ||
                (in.src[1].kind == OperandKind::Special && in.src[1].value >= 106 &&
                 in.src[1].value <= 123)) {
                auto it = rs.sreg.find(in.src[1].value);
                if (it != rs.sreg.end()) {
                    soff_bits = it->second;
                    tracked = true;
                }
            } else if (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0) {
                soff_bits = b.uconst((uint32_t)in.src[1].value);
                tracked = true;
            }
        }
        if (!tracked) {
            if (getenv("PROSPER_DBG"))
                fprintf(stderr,
                        "[smem-reject] pc=%u reason=untracked-soffset op=0x%x "
                        "src1-kind=%d src1=%d\n",
                        in.pc, in.opcode, static_cast<int>(in.src[1].kind), in.src[1].value);
            ok = false;
            return true;
        }
        soff_dyn = true;
    } else if ((int32_t)in.literal < 0) {
        ok = false;
        return true;
    }   // negative imm-only would wrap
    // A raw-offset scalar source's snapshot holds the words at the EFFECTIVE address
    // (immediate included), so it is read from index zero like the owned sources (#3135).
    const bool raw_offset_scalar_source = rs.smem_raw_offset_scalar_source_pcs.contains(in.pc);
    uint32_t base_idx = (soff_dyn || backed_register_wide || owned_wide_source ||
                         owned_nested_source || raw_offset_scalar_source)
                            ? 0
                            : in.literal >> 2;
    // Descriptor provenance: pick which bound constant buffer via the resource table, routing this
    // load to that buffer's OWN binding (N-buffer model) — so Unity's several constant buffers
    // (per-draw transform, per-frame, …) don't collapse onto one. For s_buffer_load, SBASE
    // (src[0]) is the V#: resolve it by an earlier s_load's SRT tag (indirect) or directly by its
    // user-data SGPR index (the V# was placed in SGPRs by the driver). Default binding 2.
    uint32_t binding = 2;
    bool cbuf_resolved = false;
    const ShaderResource* cbuf_resource = nullptr;
    if (rt) {
        const ShaderResource* res = nullptr;
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
        if (res) {
            binding = res->binding;
            cbuf_resolved = true;
            cbuf_resource = res;
        }
    }
    if (getenv("PROSPER_CBUFLOG"))
        fprintf(stderr,
                "[cbuf] pc=%u s_buffer_load x%u src0=s%d off=0x%x(dw%u) dyn=%d -> binding=%u %s\n",
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
    // GTA V packs two adjacent T# descriptors into one immediate S_LOAD_DWORDX16. Admit
    // only the load PCs certified by the whole-stream proof above. The values are resource
    // provenance, not scalar data used by the emitted module, so placeholders are exact;
    // unlike x4/x8, DO NOT attach the load's one SRT offset to either half. Both resources
    // are key-less and every MIMG consumer resolves by its own fetch_pc.
    if (rt && !cbuf_resolved && in.opcode == kSmemOpcodeLoadDwordX16 &&
        rs.smem_x16_descriptor_loads.contains(in.pc)) {
        for (uint32_t k = 0; k < n; ++k) {
            rs.sreg[in.dst.value + static_cast<int>(k)] = b.uconst(0);
            rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
        }
        // A shared tag here aliases both halves. Keep the invariant executable so the
        // paired-resource regression fails under a same-site tag mutation.
        uint32_t accidental_tag = 0;
        if (sreg_srt_range_tag(rs, in.dst.value, 8, accidental_tag) ||
            sreg_srt_range_tag(rs, in.dst.value + 8, 8, accidental_tag)) {
            ok = false;
        }
        return true;
    }
    // A data load with a runtime table may retain the legacy binding-2 convention only when
    // that table actually binds a buffer there. VS/compute tables start at binding 2 and use
    // this path legitimately; PS tables start at 32, where keeping the fallback would emit an
    // interface the renderer cannot satisfy (#719).
    if (rt && !cbuf_resolved) {
        const auto fallback = std::find_if(
            rt->resources.begin(), rt->resources.end(), [&](const ShaderResource& resource) {
                return resource.binding == 2 &&
                       // Preserve the historical binding-2 convention for other resources.
                       // A newly admitted raw x2 data resource belongs to its own load;
                       // borrowing it here would silently make an unrelated load succeed.
                       (resource.fetch_pc == in.pc ||
                        !rs.smem_raw_x2_data_loads.contains(resource.fetch_pc) &&
                            !rs.smem_raw_offset_scalar_source_pcs.contains(resource.fetch_pc) &&
                            !rs.smem_raw_owned_wide_data_loads.contains(resource.fetch_pc) &&
                            !resource.owned_nested_snapshot_bytes &&
                            !rs.smem_raw_register_wide_data_loads.contains(resource.fetch_pc) &&
                            !rs.smem_raw_immediate_wide_data_loads.contains(resource.fetch_pc)) &&
                       (resource.cls == ResourceClass::ConstantBuffer ||
                        resource.cls == ResourceClass::VertexBuffer);
            });
        const bool fallback_bound = fallback != rt->resources.end();
        if (!fallback_bound) {
            if (getenv("PROSPER_DBG"))
                fprintf(stderr,
                        "[smem-reject] pc=%u reason=unresolved-cbuf op=0x%x "
                        "src0=s%d dyn=%d\n",
                        in.pc, in.opcode, in.src[0].value, (int)soff_dyn);
            ok = false;
            return true;
        }
        cbuf_resource = &*fallback;
    }
    // RDNA2 scalar-buffer loads use m_size=(STRIDE==0 ? 1 : NUM_RECORDS) dwords. STRIDE is
    // only a bounds-mode selector here; it does not scale scalar addresses. Preserve the
    // explicit scalar count: size/stride cannot reconstruct it after the binding is staged
    // to the independent dword-addressable span (and is outright shorter when STRIDE < 4).
    // For every component select index zero before OpAccessChain, then select architectural
    // zero after the safe load; OpSelect does not make an already-OOB Vulkan load harmless.
    const bool scalar_buffer_load = in.opcode >= 0x8u && in.opcode <= 0xCu;
    const uint32_t scalar_bound_dwords =
        scalar_buffer_load && cbuf_resource ? cbuf_resource->scalar_buffer_dword_count : 0u;
    const bool scalar_bound_known = scalar_bound_dwords != 0u;
    if (scalar_bound_known && shader_resource_buffer_binding_bytes(*cbuf_resource) <
                                  static_cast<uint64_t>(scalar_bound_dwords) * sizeof(uint32_t)) {
        ok = false;
        return true;
    }
    auto bounded_cbuf_load = [&](uint32_t dword_index) {
        if (!scalar_bound_known) return b.cbuf_load(dword_index, binding);
        if (scalar_bound_dwords == 0u) return b.uconst(0u);
        const uint32_t in_bounds = b.ucmp(Op_ULessThan, dword_index, b.uconst(scalar_bound_dwords));
        const uint32_t safe_index = b.sel(in_bounds, dword_index, b.uconst(0u));
        const uint32_t loaded = b.cbuf_load(safe_index, binding);
        return b.sel(in_bounds, loaded, b.uconst(0u));
    };
    // #2481: this exact instruction is the producer of a runtime-selected descriptor table.
    // Its SOFFSET is the byte offset of the chosen record, so the element index is
    // SOFFSET / record-stride. Publish it as the live selector for that array binding before
    // the consumer runs; the consumer is a later instruction on the same path, so the value
    // dominates it. `cbuf_element_ptr` still range-checks the index against the declared
    // arity, so an out-of-range runtime selector reads element zero rather than out of
    // bounds -- and the front half proved every declared element is a valid descriptor.
    if (rt && soff_dyn) {
        for (const ShaderResource& resource : rt->resources) {
            if (resource.table_selector_mode != BufferTableSelectorMode::DynamicSbufferByteOffset ||
                resource.table_load_pc != in.pc || resource.table_index_count == 0u ||
                resource.table_entry_stride == 0u)
                continue;
            b.cbuf_table_selector_value[resource.binding] =
                ir::lower_smem_table_selector(b, soff_bits, resource.table_entry_stride);
            b.cbuf_table_selector_block[resource.binding] = b.cur_block;
            if (getenv("PROSPER_DBG"))
                fprintf(stderr, "[selected-table] producer pc=%u binding=%u stride=%u arity=%u\n",
                        in.pc, resource.binding, resource.table_entry_stride,
                        resource.table_index_count);
        }
    }
    if (soff_dyn) {
        // Dynamic dword index: (soffset + signed imm) >> 2 (uint add == two's-complement add).
        uint32_t idx0 = ir::lower_smem_dynamic_index(b, soff_bits, in);
        for (uint32_t k = 0; k < n; k++) {
            uint32_t kidx = ir::lower_smem_dynamic_component(b, idx0, k);
            rs.sreg[in.dst.value + (int)k] = bounded_cbuf_load(kidx);
            rs.sreg_srt.erase(in.dst.value + (int)k);   // data load: drop any stale descriptor tag
        }
        end_smem_vcc_data_lifetime(rs, in, n);
        return true;
    }
    for (uint32_t k = 0; k < n; k++)
        rs.sreg[in.dst.value + (int)k] = bounded_cbuf_load(b.uconst(base_idx + k));
    // A wide scalar load is a descriptor fetch — tag its dest SGPRs with the SRT offset so a
    // later buffer/image op using them resolves to the right resource (provenance). x4 = V#/S#
    // (buffers, samplers), x8 = T# (textures, 8 dwords).
    if (rt && (in.opcode == 0x2 || in.opcode == 0x3) &&
        !rs.smem_raw_immediate_wide_data_loads.contains(in.pc)) {
        for (uint32_t k = 0; k < n; k++) rs.sreg_srt[in.dst.value + (int)k] = in.literal;
    } else {
        // Scalar data loads overwrite the destination; they do not carry descriptor identity.
        for (uint32_t k = 0; k < n; k++) rs.sreg_srt.erase(in.dst.value + (int)k);
    }
    end_smem_vcc_data_lifetime(rs, in, n);
    return true;
}
// NOLINTEND(concurrency-mt-unsafe)
}   // namespace prosper::gpu::ir
