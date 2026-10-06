// gpu_executor_diagnostics.cpp — PROSPER_COMPUTELOG diagnostics moved out of
// gpu_executor.cpp (file-size ratchet). Diagnostic-only: stderr output, no guest-
// visible behavior. Shares read_user_sgprs/kUserSgprs via gpu_execute.hpp.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace prosper::gpu {

void diagnose_compute_dispatches(const GpuState& st, uint64_t submit_no) {
    const char* enabled = getenv("PROSPER_COMPUTELOG");
    const char* dim_env = getenv("PROSPER_COMPUTELOG_DIM");
    // PROSPER_SRTDUMP arms the shader-resource-table dump below and NOTHING else. It needs the same
    // per-dispatch walk as COMPUTELOG, which is why it enters this function, but it must not turn on
    // COMPUTELOG's per-dispatch prose -- that prose is what makes a COMPUTELOG run on this title's
    // gameplay route unreadable, and it carries a 4 KiB FNV hash of the shader code per dispatch.
    //
    // `prose` is what the two PRE-EXISTING switches ask for, and it gates the tail exactly as they
    // did before this diagnostic existed. Getting this wrong is not cosmetic: the first version of
    // this switch let `srt_only` un-gate the tail, so an "SRTDUMP-only" run emitted 352,940
    // `[compute]` lines it never claimed to, and the resulting log was read as if it were the small
    // targeted one the comment promised.
    const char* srt_env = getenv("PROSPER_SRTDUMP");
    const bool srt_only = srt_env && *srt_env;
    const bool prose = (enabled && *enabled) || (dim_env && *dim_env);
    if (!prose && !srt_only) return;

    uint32_t want_w = 0, want_h = 0;
    if (dim_env && *dim_env && sscanf(dim_env, "%ux%u", &want_w, &want_h) != 2) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[compute] invalid PROSPER_COMPUTELOG_DIM='%s' (expected WxH)\n", dim_env);
        }
        want_w = want_h = 0;
    }

    namespace P = prosper::agc::Pm4;
    auto rd = [](const RegisterFile& regs, uint32_t off) {
        auto it = regs.find(off); return it == regs.end() ? 0u : it->second;
    };
    size_t matched = 0;
    for (size_t i = 0; i < st.dispatches.size(); ++i) {
        const auto& d = st.dispatches[i];
        const GpuState& ds = d.state ? *d.state : st;
        const ComputeLaunchDimensions launch = resolve_compute_launch(d);
        const uint64_t code_addr = compute_dispatch_code_addr(st, d);
        const auto* hdr = static_cast<const AgcShaderHeader*>(prosper_agc_shader_header_for_code(code_addr));

        uint32_t range_start = 0;
        if (hdr && hdr->specials && guest_readable((uint64_t)(uintptr_t)hdr->specials,
                                                   sizeof(AgcShaderSpecials))) {
            uint32_t s = hdr->specials->user_data_range_start;
            uint32_t e = hdr->specials->user_data_range_end;
            if (s < kUserSgprs && e > s && e <= 2 * kUserSgprs) range_start = s;
        }

        ShaderResourceTable table;
        uint32_t sgprs[kUserSgprs] = {};
        // `user_data` can be NON-NULL yet point at unmapped guest memory -- build_shader_resources
        // probes for exactly this and documents the observed case (#713, PPSA02664), returning an
        // empty table rather than faulting. TWO readers below dereference it, in sibling scopes, so
        // the probe is hoisted here and shared: a diagnostic that SIGSEGVs where the renderer does
        // not turns an investigation into a crash report about the instrument.
        const AgcShaderUserData* ud = hdr ? hdr->user_data : nullptr;
        const bool ud_ok = ud && guest_readable((uint64_t)(uintptr_t)ud, sizeof(AgcShaderUserData));
        if (hdr) {
            read_user_sgprs(ds.sh, P::COMPUTE_USER_DATA_0 + range_start, sgprs);
            table = build_shader_resources(*hdr, sgprs, kUserSgprs, 0);
            assign_convention_bindings(table, 2);

            // SRT CONTENTS. `build_shader_resources` reads sharps and the EUD and has no
            // shader-resource-table path, and every SRT-declaring header in a routed GTA V gameplay
            // run declares neither -- `sharps={0,0,0,0} eud=0` on 138,034 of 138,034 headers across
            // 88 of 88 programs (#2705). So the AGC-header path resolves nothing for them and the
            // table's own bytes are the only way to see what they describe. Nothing dumps them; this
            // does.
            //
            // WHAT THIS DOES NOT MEAN, because an earlier version of this comment said it and it is
            // false: prosper is NOT blind to this channel. `add_compute_buffer_resources` (:5665,
            // called :7570) const-folds descriptors loaded with `s_load_dwordx4/x8 sN, s[ptr:ptr+1],
            // <imm>` from a user-data table -- see the SrtUse contract in gpu_execute.hpp -- and it
            // fires on these very programs, e.g. `[compute-table] program 0x205b657200 …
            // addr=0x20037cf620 stride=32 srt=0x10`. The gap is narrower and stranger than "no path
            // exists": the recovered key set starts at 0x10 and byte offset 0 never appears in it,
            // while the shaders do load descriptors there (#2757).
            //
            // Recorded at this length because the false version travelled: it was written here, then
            // into a PR body, and would have been the wording the next agent inherited -- from code,
            // which outlives the PR that a correction lives in.
            //
            // Deciding that needs the table's own bytes, and nothing dumps them. This does, and
            // deliberately dumps CANDIDATES rather than naming one pointer as the SRT: its position
            // in the user-data block is not established, and a diagnostic that picks one and labels
            // it "the SRT" would manufacture the fact it exists to measure. Every readable dword
            // pair in the window is shown with the first `srt_size_dw` dwords behind it. Whichever
            // holds descriptors will be evident; if none does, that is equally the answer.
            if (srt_only && ud_ok && ud->srt_size_dw) {
                const uint32_t want = ud->srt_size_dw;
                const uint32_t shown = want < 64 ? want : 64;   // srt_size_dw is a uint16
                fprintf(stderr, "[srtdump] code=0x%llx srt_size_dw=%u (showing %u) "
                                "sharps={%u,%u,%u,%u} eud=%u\n",
                        (unsigned long long)code_addr, want, shown,
                        ud->sharp_resource_count[0], ud->sharp_resource_count[1],
                        ud->sharp_resource_count[2], ud->sharp_resource_count[3], ud->eud_size_dw);
                bool any = false;
                for (uint32_t k = 0; k + 1 < kUserSgprs; k++) {
                    const uint64_t raw = (uint64_t)sgprs[k] | ((uint64_t)sgprs[k + 1] << 32);
                    if (raw <= 0x10000) continue;
                    uint64_t addr = 0;
                    for (uint64_t mask : {~uint64_t{0}, uint64_t{0xFFFFFFFFFFFF},
                                          uint64_t{0xFFFFFFFFFF}}) {
                        const uint64_t cand = raw & mask;
                        if (cand > 0x10000 && guest_readable(cand, shown * 4u)) { addr = cand; break; }
                    }
                    if (!addr) continue;
                    any = true;
                    fprintf(stderr, "[srtdump]   dw%u -> 0x%llx:", k, (unsigned long long)addr);
                    const uint32_t* words = reinterpret_cast<const uint32_t*>(addr);
                    uint32_t nonzero = 0;
                    for (uint32_t w = 0; w < shown; w++) {
                        fprintf(stderr, " %08x", words[w]);
                        nonzero += words[w] != 0;
                    }
                    fprintf(stderr, "  (nz=%u/%u)\n", nonzero, shown);
                }
                if (!any)
                    fprintf(stderr, "[srtdump]   no readable pointer in the user-data window\n");
            }
        }

        // A compute shader can carry only an inline direct type-1 V# and no sharp descriptors. Dump
        // its metadata and bound SGPRs once per program if resource decoding still returns empty.
        // This turns the next unsupported layout into a reproducible decode problem instead of
        // another blind `resources=0` investigation.
        if (hdr && table.resources.empty() && enabled && *enabled) {
            static std::set<uint64_t> logged_empty;
            if (logged_empty.insert(code_addr).second) {
                fprintf(stderr, "[compute] empty-resource metadata code=0x%llx type=%u ud=%p%s",
                        (unsigned long long)code_addr, hdr->type, (const void*)ud,
                        ud && !ud_ok ? " (UNMAPPED)" : "");
                if (ud_ok) {
                    fprintf(stderr, " eud=%u srt=%u direct_count=%u sharp={%u,%u,%u,%u}",
                            ud->eud_size_dw, ud->srt_size_dw, ud->direct_resource_count,
                            ud->sharp_resource_count[0], ud->sharp_resource_count[1],
                            ud->sharp_resource_count[2], ud->sharp_resource_count[3]);
                }
                fprintf(stderr, "\n[compute]   user_sgprs:");
                for (uint32_t s = 0; s < kUserSgprs; ++s) fprintf(stderr, " %08x", sgprs[s]);
                fprintf(stderr, "\n");

                if (ud_ok && ud->direct_resource_offset && ud->direct_resource_count) {
                    fprintf(stderr, "[compute]   direct offsets:");
                    for (uint16_t t = 0; t < ud->direct_resource_count && t < 16; ++t)
                        fprintf(stderr, " [%u]=%u", t, ud->direct_resource_offset[t]);
                    fprintf(stderr, "\n");
                    const uint32_t reg = ud->direct_resource_count > 1 ? ud->direct_resource_offset[1] : 0xffffu;
                    if (reg != 0xffffu && reg + 4 <= kUserSgprs) {
                        const DecodedBufferDescriptor d = decode_buffer_descriptor(&sgprs[reg]);
                        fprintf(stderr, "[compute]   type1 V# reg=%u base=0x%llx stride=%u records=%u "
                                        "size=%u fmt=%u comps=%u\n",
                                reg, (unsigned long long)d.base, d.stride, d.num_records, d.size_bytes,
                                (unsigned)d.format, d.num_components);
                    }
                }
            }
        }

        bool dim_match = !want_w || !want_h;
        if (!dim_match) {
            for (const auto& r : table.resources)
                if (r.width == want_w && r.height == want_h) { dim_match = true; break; }
        }
        if (!dim_match) continue;
        matched++;
        // Everything past here is COMPUTELOG/COMPUTELOG_DIM's output, including the 4 KiB code hash.
        // An SRTDUMP-only run has already printed what it came for.
        if (!prose) continue;

        uint64_t code_hash = 1469598103934665603ull;
        if (code_addr && guest_readable(code_addr, 4096)) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(code_addr));
            for (size_t n = 0; n < 4096; ++n) { code_hash ^= p[n]; code_hash *= 1099511628211ull; }
        } else {
            code_hash = 0;
        }
        fprintf(stderr,
                "[compute] submit=%llu dispatch=%zu threads=%ux%ux%u local=%ux%ux%u "
                "groups=%ux%ux%u modifier=0x%llx "
                "code=0x%llx hash4k=%016llx header=%s resources=%zu\n",
                (unsigned long long)submit_no, i,
                launch.threads_x, launch.threads_y, launch.threads_z,
                launch.local_x, launch.local_y, launch.local_z,
                launch.groups_x, launch.groups_y, launch.groups_z,
                (unsigned long long)d.modifier, (unsigned long long)code_addr,
                (unsigned long long)code_hash, hdr ? "yes" : "no", table.resources.size());
        for (const auto& r : table.resources) {
            fprintf(stderr,
                    "[compute]   cls=%u binding=%u addr=0x%llx size=%u dims=%ux%u "
                    "fmt=%u comps=%u tile=%u sgpr=%u srt=0x%x\n",
                    (unsigned)r.cls, r.binding, (unsigned long long)r.gpu_addr, r.size,
                    r.width, r.height, (unsigned)r.format, r.num_components, r.tile_mode,
                    r.sgpr_base, r.srt_offset);
        }
    }

    if (want_w && want_h && !st.dispatches.empty() && matched == 0 && enabled && enabled[0] == 'a') {
        fprintf(stderr, "[compute] submit=%llu dispatches=%zu: no resource matched %ux%u\n",
                (unsigned long long)submit_no, st.dispatches.size(), want_w, want_h);
    }
}

} // namespace prosper::gpu
