// Original PC-relative table format proofs.
#include "gpu/recompiler/rdna2_cfg_pcrel_tables.hpp"
#include "gpu/recompiler/rdna2_cfg_pcrel_parts.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
void cfg_pcrel_scalar_load(const prosper::gpu::Rdna2Inst& in,
                           std::unordered_map<int, uint64_t>& pcoff, std::unordered_set<int>& pchi,
                           std::unordered_map<int, uint32_t>& kconst,
                           std::unordered_map<int, uint32_t>& fact_pc,
                           std::vector<uint32_t>& br_targets, size_t& dwords,
                           size_t*& required_dwords, PcrelTables& out, const uint32_t*& code,
                           const std::function<void(int)>& kill) {
    switch (in.fmt) {
        case Rdna2Format::SMEM: {
            // The SGPRs SDATA may receive (the effects table); a reserved encoding keeps the
            // old one-register kill.
            const uint32_t n =
                rdna2_instruction_effects(in).known() ? rdna2_smem_destination_dwords(in) : 1u;
            // Scalar-buffer form of the same compiler idiom: SBASE is the getpc-built V#, while
            // SOFFSET selects a row from the embedded table. Inspect it before killing SDATA facts.
            const int sb = in.src[0].value;
            if (smem_opcode_is_buffer_load(in.opcode) && pcoff.count(sb) && pchi.count(sb + 1) &&
                kconst.count(sb + 2) && kconst.count(sb + 3)) {
                const uint32_t nrec = kconst[sb + 2];
                const uint64_t off = pcoff[sb];
                uint32_t newest = 0;
                for (int r : {sb, sb + 1, sb + 2, sb + 3}) {
                    auto fact = fact_pc.find(r);
                    if (fact != fact_pc.end() && fact->second > newest) newest = fact->second;
                }
                bool entered = false;
                for (uint32_t target : br_targets)
                    if (target > newest && target <= in.pc) {
                        entered = true;
                        break;
                    }
                const bool supported_soffset =
                    (in.src[1].kind == OperandKind::Special && in.src[1].value == 125) ||
                    in.src[1].kind == OperandKind::SGPR ||
                    (in.src[1].kind == OperandKind::Special && in.src[1].value >= 106 &&
                     in.src[1].value <= 123) ||
                    (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0);
                if (!entered && supported_soffset && static_cast<int32_t>(in.literal) >= 0 &&
                    !(off & 3u) && !(nrec & 3u) && nrec && nrec <= 1024 &&
                    off / 4 + nrec / 4 <= dwords) {
                    if (required_dwords)
                        *required_dwords =
                            std::max(*required_dwords, static_cast<size_t>(off / 4 + nrec / 4));
                    out.smem[in.pc] =
                        std::vector<uint32_t>(code + off / 4, code + off / 4 + nrec / 4);
                }
            }
            for (uint32_t k = 0; k < n; k++) kill(in.dst.value + (int)k);
            break;
        }
        default: break;
    }
}

void cfg_pcrel_typed_load(const prosper::gpu::Rdna2Inst& in,
                          std::unordered_map<int, uint64_t>& pcoff, std::unordered_set<int>& pchi,
                          std::unordered_map<int, uint32_t>& kconst,
                          std::unordered_map<int, uint32_t>& fact_pc,
                          std::vector<uint32_t>& br_targets, size_t& dwords,
                          size_t*& required_dwords, PcrelTables& out, const uint32_t*& code) {
    switch (in.fmt) {
        case Rdna2Format::MTBUF: {
            // Same proof as the MUBUF case below, with two extra obligations: the typed format
            // must be a pure 32-bit-per-component pass-through (so the fold copies dwords, not
            // converted texels), and TFE must be off (its trailing status word is not modelled).
            // Per RDNA2 ISA Table 31, TBUFFER_LOAD_FORMAT_* always uses DST SEL = identity,
            // ignoring the descriptor's DST_SEL field entirely (see test_rdna2_to_spirv.cpp:4261).
            if (in.opcode > 0x03u || in.mtbuf_tfe) break;   // typed LOADS only, no status
            const uint32_t components = in.opcode + 1u;
            if (!rdna2_buffer_format_is_raw_dwords(in.mtbuf_format, components)) break;
            const int sb = in.src[1].value;
            if (!pcoff.count(sb) || !pchi.count(sb + 1) || !kconst.count(sb + 2)) break;
            const uint32_t nrec = kconst[sb + 2];
            const bool idxen = (in.literal >> 13) & 1u;
            const bool soff0 = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                               (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
            const uint64_t off = pcoff[sb];
            // Entry soundness. A fact established at pc F is invalid at the load if control can
            // ENTER the stream at a branch target T with F < T <= load_pc, because on that path
            // F never executed. Every contributing register carries that obligation, so the
            // bound is the OLDEST fact, not the newest: a maximum protects only the last
            // register written and silently admits a branch that entered after an earlier one.
            // (The MUBUF and SMEM cases below still use a maximum and have the same hole —
            // #2862. Deliberately not changed here: this is the typed path's own predicate, and
            // widening the fix to the untyped ones needs its own evidence and its own arms.)
            uint32_t oldest = UINT32_MAX;
            bool entered = false;
            for (int r : {sb, sb + 1, sb + 2}) {
                auto it = fact_pc.find(r);
                if (it == fact_pc.end()) {
                    entered = true;
                    break;
                }   // no proven fact at all
                if (it->second < oldest) oldest = it->second;
            }
            if (!entered)
                for (uint32_t t : br_targets)
                    if (t > oldest && t <= in.pc) {
                        entered = true;
                        break;
                    }
            if (entered) break;
            if (idxen || !soff0 || (off & 3u) || (nrec & 3u) || nrec == 0 || nrec > 1024 ||
                off / 4 + nrec / 4 > dwords)
                break;
            if (required_dwords)
                *required_dwords =
                    std::max(*required_dwords, static_cast<size_t>(off / 4 + nrec / 4));
            out.mtbuf[in.pc] = std::vector<uint32_t>(code + off / 4, code + off / 4 + nrec / 4);
            break;
        }
        default: break;
    }
}

void cfg_pcrel_raw_load(const prosper::gpu::Rdna2Inst& in, std::unordered_map<int, uint64_t>& pcoff,
                        std::unordered_set<int>& pchi, std::unordered_map<int, uint32_t>& kconst,
                        std::unordered_map<int, uint32_t>& fact_pc,
                        std::vector<uint32_t>& br_targets, size_t& dwords, size_t*& required_dwords,
                        PcrelTables& out, const uint32_t*& code) {
    switch (in.fmt) {
        case Rdna2Format::MUBUF: {
            if (in.opcode < 0xCu || in.opcode > 0xFu) break;   // raw loads only
            const int sb = in.src[1].value;   // SRSRC base SGPR of the V# quad
            if (!pcoff.count(sb) || !pchi.count(sb + 1) || !kconst.count(sb + 2)) break;
            const uint32_t nrec = kconst[sb + 2];   // V# word2 = num_records (bytes here)
            const bool offen = (in.literal >> 12) & 1u, idxen = (in.literal >> 13) & 1u;
            const bool soff0 = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                               (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
            const uint64_t off = pcoff[sb];
            // Entry soundness: no branch may enter between the newest contributing fact and here.
            uint32_t newest = 0;
            for (int r : {sb, sb + 1, sb + 2}) {
                auto it = fact_pc.find(r);
                if (it != fact_pc.end() && it->second > newest) newest = it->second;
            }
            bool entered = false;
            for (uint32_t t : br_targets)
                if (t > newest && t <= in.pc) {
                    entered = true;
                    break;
                }
            if (entered) break;
            if (idxen || !soff0 || (off & 3u) || (nrec & 3u) || nrec == 0 || nrec > 1024 ||
                off / 4 + nrec / 4 > dwords)
                break;   // unprovable / out of window -> reject
            if (required_dwords)
                *required_dwords =
                    std::max(*required_dwords, static_cast<size_t>(off / 4 + nrec / 4));
            out.mubuf[in.pc] = std::vector<uint32_t>(code + off / 4, code + off / 4 + nrec / 4);
            (void)offen;   // offen just adds the runtime index
            break;
        }
        default: break;
    }
}

}   // namespace prosper::gpu
