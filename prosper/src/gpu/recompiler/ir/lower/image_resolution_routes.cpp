// Mechanical relocation of unresolved MIMG reporting and diagnostic handling.
#include "gpu/recompiler/ir/lower/image_resolution_routes.hpp"
#include "gpu/recompiler/compiler_choices.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>

namespace prosper::gpu::ir {
std::optional<bool> ImageResolutionRoutes::route() {
    if (!res) {
        // Resolution-failure diagnostic: which provenance step failed for this image op.
        //
        // Ungated, and deduped per (pc, srsrc). It fires only when an image op has already
        // failed to resolve, so its volume is bounded by the defect it reports. Behind
        // PROSPER_DBG it was unreachable in practice on any routed boot.
        //
        // #3143: this report used to sit downstream of the PROSPER_MIMG_SOFT early return
        // below, so arming that switch to see whether softening changes a frame ALSO deleted
        // the one line that names which descriptor failed -- the switch was least informative
        // exactly when it was armed for its intended purpose ("softening changed nothing"
        // became indistinguishable from "there was nothing here to soften"). It now runs
        // before either the soft or the reject path decides what to DO about the failure, so
        // both keep the same descriptor identification; the soft path still gets its own
        // confirmation line below ([mimg-soft]) once the constant is written.
        static std::mutex mimg_mutex;
        // Keyed by PROGRAM as well as (pc, srsrc): every shader has a pc 16, so a
        // pc-only key lets the first program to reach one silence all later programs and
        // attribute its line to a shader the reader is not looking at.
        static std::set<std::tuple<uint64_t, uint32_t, int>> mimg_reported;
        bool first_report = false;
        {
            std::lock_guard<std::mutex> lock(mimg_mutex);
            first_report =
                mimg_reported.emplace(b.diagnostic.program_address, in.pc, in.src[1].value).second;
        }
        report(first_report);

        // NOLINTNEXTLINE(bugprone-optional-value-conversion): Preserve the original bool-return route protocol.
        if (auto routed = soften()) return *routed;
        if (!first_report) {
            ok = false;
            return true;
        }
    }
    return std::nullopt;
}

void ImageResolutionRoutes::available(
    const std::function<const char*(const ShaderResource*)>& cls_name) {
    // #3126: say what WAS available, not only what failed. "key_res=null pc_res=null"
    // tells you the lookups missed; it does not tell you whether the table holds the
    // descriptor under a different key, holds it classified as something else, or does
    // not hold it at all -- and those are three different bugs. Print every resource's
    // class and provenance so the miss can be diagnosed from one line instead of a
    // rebuild. Bounded, and printed once per site regardless of whether the site is
    // then softened or rejected (#3143).
    //
    // Print EVERY key a resource carries, not one of them. The old form printed
    // `srt=0x%x` whenever sgpr_base was unset, so a resource keyed only by fetch_pc
    // -- which is how the const-fold publishes a seed/table-recovered T# -- rendered
    // as `srt=0xffffffff` and read as "this entry has no lookup key at all". Four of
    // the nine resources in Stray's failing pixel stage printed that way while being
    // perfectly well keyed, just not by a key this reader could see (#3126).
    std::string avail;
    size_t shown = 0;
    for (const ShaderResource& r : rt->resources) {
        if (shown++ >= 16) {
            avail += " ...";
            break;
        }
        char one[128];
        char keys[80];
        int used = 0;
        if (r.sgpr_base != 0xFFFFFFFFu)
            used += snprintf(keys + used, sizeof keys - (size_t)used, " s%u", r.sgpr_base);
        if (r.srt_offset != 0xFFFFFFFFu)
            used += snprintf(keys + used, sizeof keys - (size_t)used, " srt=0x%x", r.srt_offset);
        if (r.fetch_pc != 0xFFFFFFFFu)
            used += snprintf(keys + used, sizeof keys - (size_t)used, " pc=%u", r.fetch_pc);
        if (!used) snprintf(keys, sizeof keys, " unkeyed");
        snprintf(one, sizeof one, " [b%u %s%s]", r.binding, cls_name(&r), keys);
        avail += one;
    }
    fprintf(stderr, "[mimg-unresolved]   available:%s\n",
            avail.empty() ? " (none)" : avail.c_str());
}

void ImageResolutionRoutes::report(bool& first_report) {
    if (first_report) {
        uint32_t srt_tag = 0;
        const bool has_srt_tag = sreg_srt_range_tag(rs, in.src[1].value, 8, srt_tag);
        const ShaderResource* pk = has_srt_tag ? rt->by_srt_offset(srt_tag) : nullptr;
        const ShaderResource* pp = rt->by_fetch_pc(in.pc);
        // Report the copy-alias step too. Without it a staged descriptor (#1773) and a
        // genuinely absent one print the identical line, which is the output-vs-input
        // confusion that cost #1590 several sessions -- name which provenance route failed.
        int ud_origin = 0;
        const bool has_ud_alias = sreg_range_ud_alias(rs, in.src[1].value, 8, ud_origin);
        const ShaderResource* pa = has_ud_alias ? rt->by_sgpr_base(ud_origin) : nullptr;
        // sgpr_res: the DIRECT user-data route's own outcome, which this line did not
        // report and which is the ONLY route that can fire for the commonest failure
        // shape -- `written=0`, i.e. the shader never touched SRSRC, so by construction
        // there is no s_load tag and no copy alias for the other three fields to carry.
        // For such a site `srt_tag=NONE key_res=null ud_alias=NONE alias_res=null` is a
        // restatement of `written=0`, not evidence: those routes did not run. Five sites
        // on Stray (PPSA02101, #3126) printed exactly that quadruple while three of them
        // had a resource sitting at the requested SGPR the whole time -- of the wrong
        // class, discarded in silence. Print the class-BLIND hit so "nothing is here"
        // and "something is here and it is a cbuf" stop looking identical. #3126/#1634.
        //
        // ...and say whether that route was CONSULTED, which is the same trap inverted
        // (trap 254). The SGPR route is guarded by `!sreg_range_written` above, so on a
        // `written=1` site it never runs -- and printing a bare `sgpr_res=tex` there
        // tells the reader a Texture was found at the requested SGPR and rejected, when
        // in fact nothing consulted it and the descriptor is not the entry user data at
        // all. That is live on this title: `0x30131d0000` pc=77 is `written=1`. Print the
        // status and the contents, never the contents alone.
        const bool srsrc_written = sreg_range_written(rs, in.src[1].value, 8);
        const ShaderResource* ps_direct = rt->by_sgpr_base(in.src[1].value);
        auto cls_name = [](const ShaderResource* r) {
            if (!r) return "null";
            switch (r->cls) {
                case ResourceClass::Texture: return "tex";
                case ResourceClass::StorageImage: return "simg";
                case ResourceClass::ConstantBuffer: return "cbuf";
                case ResourceClass::VertexBuffer: return "vbuf";
                case ResourceClass::Sampler: return "samp";
            }
            return "other-cls";
        };
        const char* need = image_requirement == ImageResourceRequirement::StorageOnly   ? "storage"
                           : image_requirement == ImageResourceRequirement::SampledOnly ? "sampled"
                                                                                        : "either";
        // `not-consulted(<cls>)` keeps BOTH facts on the line: that the route did not
        // run, and what happens to sit at that SGPR anyway. Dropping the second half
        // would trade one misreading for a different one.
        const std::string sgpr_res = srsrc_written
                                         ? std::string("not-consulted(") + cls_name(ps_direct) + ")"
                                         : std::string(cls_name(ps_direct));
        // Keep the original diagnostic literal intact for the relocation token proof.
        // clang-format off
        fprintf(stderr, "[mimg-unresolved] program=0x%llx pc=%u op=0x%02x storage=%d need=%s srsrc=s%d srt_tag=%s0x%x key_res=%s pc_res=%s ud_alias=%s%d alias_res=%s sgpr_res=%s written=%d (%zu res)\n",
                (unsigned long long)b.diagnostic.program_address,
                in.pc, in.opcode, (int)storage_only_op, need,
                in.src[1].value, has_srt_tag ? "" : "NONE ",
                has_srt_tag ? srt_tag : 0u,
                cls_name(pk), cls_name(pp),
                has_ud_alias ? "s" : "NONE ", has_ud_alias ? ud_origin : 0,
                cls_name(pa), sgpr_res.c_str(),
                srsrc_written ? 1 : 0,
                rt->resources.size());
        // clang-format on
        available(std::cref(cls_name));
    }
}

std::optional<bool> ImageResolutionRoutes::soften() {
    // PROSPER_MIMG_SOFT=1 -- diagnostic only, and deliberately WRONG output.
    //
    // Placed BEFORE the reject gate below on purpose. That gate fires once per
    // (program, pc, srsrc); downstream of it, only the FIRST compile of each site would be
    // softened and every later one -- reachable through ordinary shader-cache eviction --
    // would take the reject path unchanged. The recorded result would then be
    // indistinguishable from "nothing was softened", which is exactly the reading this
    // diagnostic exists to produce. When an image
    // op cannot resolve its descriptor, write a constant instead of failing the stage, so a
    // frame can be seen that would otherwise be discarded entirely. It answers one question
    // nothing else here can: how much of the picture is riding on THIS resolution failure.
    // On Stray it showed the answer was "none" -- the composite compiled and the frame did
    // not change -- which is what redirected #3126 away from the recompiler.
    //
    // Never acceptable in a run that produces progression evidence: every unresolved sample
    // reads a flat value.
    if (!res && compiler_choice(CompilerChoice::MimgSoft) != 0) {
        const uint32_t soft = b.uconst(fbits(0.5f));
        uint32_t comps = 0;
        for (uint32_t m = 0; m < 4; ++m)
            if (in.mimg_dmask & (1u << m)) ++comps;
        if (!comps) comps = 1;
        for (uint32_t k = 0; k < comps; ++k) {
            const int dst = in.dst.value + static_cast<int>(k);
            if (dst < 0) break;
            const uint32_t old = vreg_old(b, rs, dst);
            rs.vreg[dst] = soft;
            predicate_write(b, rs, dst, old);
        }
        static std::mutex smx;
        static std::set<uint64_t> sseen;
        std::lock_guard<std::mutex> lk(smx);
        if (sseen.insert(((uint64_t)b.diagnostic.program_address << 16) | in.pc).second)
            fprintf(stderr, "[mimg-soft] program=0x%llx pc=%u -> constant %u comps\n",
                    (unsigned long long)b.diagnostic.program_address, in.pc, comps);
        return true;
    }
    return std::nullopt;
}

}   // namespace prosper::gpu::ir
