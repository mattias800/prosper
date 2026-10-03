#include "gpu/recompiler/fragment_packet_export_timing.hpp"
#include "gpu/recompiler/fragment_packet_exports.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <bitset>
#include <map>

namespace prosper::gpu {
const char* fragment_packet_export_timing_gap(const std::vector<Rdna2Inst>& ins, uint32_t& pc) {
    // AMD RDNA2 ISA 70648, indexed official section11.4/page100: bus grant reads EXEC and
    // VGPR data; EXPCNT is decremented after the last transfer. Section12.5/page124 explicitly
    // makes ordinary S_ENDPGM execute WAITCNT0 and WAITCNT_VSCNT0. EXP.DONE is NOT a drain.
    // STATUS page15 restricts SKIP_EXPORT to VS: no fabricated PS launch-status value is needed.
    // Sources: https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content
    // Only indexed sections were retrieved; no retained full PDF is claimed.
    struct Pending {
        std::bitset<256> words;
        bool exec = false, reached = false;
    };
    pc = UINT32_MAX;
    if (ins.empty()) return "packet-export-timing-inventory-incomplete";
    std::map<uint32_t, size_t> indices;
    for (size_t i = 0; i < ins.size(); ++i) indices.emplace(ins[i].pc, i);
    std::vector<Pending> entry(ins.size());
    entry.front().reached = true;
    const auto join = [&](size_t next, const Pending& value) {
        entry[next].reached = true;
        entry[next].words |= value.words;
        entry[next].exec |= value.exec;
    };
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!entry[i].reached) continue;
        const auto& in = ins[i];
        pc = in.pc;
        auto pending = entry[i];
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        if (branch && (branch_target(in) <= in.pc || !indices.contains(branch_target(in))))
            return "packet-export-timing-control-unimplemented";
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0c && in.simm16 == 0) {
            pending.words.reset();
            pending.exec = false;
        } else if (in.is_end) {
            if (in.fmt != Rdna2Format::SOPP || in.opcode != 1)
                return "packet-export-timing-terminal-unimplemented";
            continue;   // exact ordinary END implicitly drains all exports before termination
        } else {
            bool writes_exec = false;
            for_each_scalar_write(in, [&](int reg, uint32_t width) {
                writes_exec |= reg <= 127 && reg + static_cast<int>(width) > 126;
            });
            writes_exec |= in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeAndSaveexecB64;
            writes_exec |= in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode);
            if (pending.exec && writes_exec) return "packet-export-exec-overwrite-before-wait";
            for (uint32_t word = 0; word < rdna2_vgpr_write_count(in); ++word) {
                const int reg = in.dst.value + static_cast<int>(word);
                if (reg < 0 || reg > 255) return "packet-export-timing-register-invalid";
                if (pending.words.test(reg)) return "packet-export-source-overwrite-before-wait";
            }
            if (in.fmt == Rdna2Format::EXP) {
                pending.exec = true;   // null/control exports also observe EXEC
                const auto mask = fragment_packet_export_source_mask(in.exp_en, in.exp_compr);
                for (uint32_t word = 0; word < 4; ++word) {
                    if (!(mask & (1u << word))) continue;
                    if (word >= in.n_src || in.src[word].kind != OperandKind::VGPR ||
                        in.src[word].value < 0 || in.src[word].value > 255)
                        return "packet-export-timing-source-form-unimplemented";
                    pending.words.set(in.src[word].value);
                }
            }
        }
        // Union pending obligations on EVERY conditional predecessor. Supplied EXEC/SCC never
        // prune a path, and WAIT on just one arm cannot certify the other arm's bus completion.
        if (branch) join(indices.at(branch_target(in)), pending);
        if ((!branch || in.opcode != 2) && i + 1 < ins.size()) join(i + 1, pending);
    }
    pc = UINT32_MAX;
    return nullptr;
}
}   // namespace prosper::gpu
