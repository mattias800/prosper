#include "gpu/recompiler/fragment_packet_vgpr_requirements.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <algorithm>

namespace prosper::gpu {
uint64_t FragmentPacketVgprRequirements::retained_bytes() const {
    using ReadMap = decltype(reads);
    constexpr uint64_t alignment = alignof(ReadMap::value_type);
    // Cache-budget estimate, NOT an exact allocator census. Count the complete aligned pair
    // (key/vector padding included) plus a deliberately generous eight-pointer allowance for
    // tree links, bookkeeping and allocation overhead, then align the complete estimate.
    constexpr uint64_t node_allowance = 8 * sizeof(void*);
    constexpr uint64_t node_estimate =
        ((sizeof(ReadMap::value_type) + node_allowance + alignment - 1) / alignment) * alignment;
    uint64_t bytes = sizeof(*this) + rejection.capacity() + 1 + masks.retained_bytes() +
                     scalar_reads.retained_bytes();
    for (const auto& [pc, accesses] : reads)
        bytes += node_estimate + accesses.capacity() * sizeof(FragmentPacketVgprAccess);
    return bytes;
}

FragmentPacketVgprRequirements
fragment_packet_vgpr_requirements(const std::vector<uint32_t>& code,
                                  const std::vector<Rdna2Inst>& ins,
                                  FragmentPacketExportObservation observation) {
    FragmentPacketVgprRequirements result;
    result.source_words = &code;
    result.masks = fragment_packet_mask_requirements(code, ins);
    result.scalar_reads = fragment_packet_scalar_read_requirements(code, ins);
    if (code.empty() || code.size() > 4096 || ins.empty() || !ins.back().is_end ||
        ins.back().pc + ins.back().len_dwords != code.size()) {
        result.rejection = "packet-vgpr-program-inventory-incomplete";
        return result;
    }
    std::map<uint32_t, size_t> indices;
    for (size_t i = 0; i < ins.size(); ++i) indices.emplace(ins[i].pc, i);
    std::vector<std::bitset<256>> writers(ins.size());
    std::vector<bool> reached(ins.size());
    reached.front() = true;
    const auto meet = [&](size_t next, const std::bitset<256>& defined) {
        if (!reached[next]) {
            reached[next] = true;
            writers[next] = defined;
        } else
            writers[next] &= defined;
    };
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        const auto read = [&](int reg, FragmentPacketVgprRead kind) {
            if (reg < 0 || reg > 255) {
                result.rejection = "packet-vgpr-register-out-of-range";
                return;
            }
            result.storage.set(reg);
            result.reads[in.pc].push_back({static_cast<uint32_t>(reg), kind});
            if (reached[i] && !writers[i].test(reg)) result.possible_entry.set(reg);
        };
        // This is the CURRENT packet's bounded full-dword instruction domain, not a generic
        // decoder operand-field proof. New partial/RMW/dynamic forms must extend this inventory.
        const bool vector = in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 ||
                            in.fmt == Rdna2Format::VOP3 || in.fmt == Rdna2Format::VOPC ||
                            in.fmt == Rdna2Format::VINTRP || in.fmt == Rdna2Format::MIMG ||
                            in.fmt == Rdna2Format::EXP;
        if (in.has_modifier || in.has_sdwa || in.has_dpp || in.clamp || in.omod ||
            std::any_of(std::begin(in.src_abs), std::end(in.src_abs), [](bool v) { return v; }) ||
            std::any_of(std::begin(in.src_neg), std::end(in.src_neg), [](bool v) { return v; }) ||
            (!vector && in.fmt != Rdna2Format::SOP1 && in.fmt != Rdna2Format::SOP2 &&
             in.fmt != Rdna2Format::SOPK && in.fmt != Rdna2Format::SOPP &&
             in.fmt != Rdna2Format::SMEM))
            result.rejection = "packet-vgpr-read-form-unimplemented";
        // RCP/RSQ/SQRT have the same one-source/full-dword storage contract. Their numerical,
        // launch and exception admission remains exclusively in the resource services preflight.
        if ((in.fmt == Rdna2Format::VOP1 && in.opcode != 1 && in.opcode != 0x2a &&
             in.opcode != 0x2e && in.opcode != 0x33) ||
            (in.fmt == Rdna2Format::VOP2 && in.opcode != 3 && in.opcode != 8) ||
            (in.fmt == Rdna2Format::VOP3 && in.opcode != 0x360 && in.opcode != 0x365 &&
             in.opcode != 0x366) ||
            (in.fmt == Rdna2Format::VOPC && !((in.opcode >= 0xc0 && in.opcode <= 0xc7) ||
                                              (in.opcode >= 0xd0 && in.opcode <= 0xd7))) ||
            (in.fmt == Rdna2Format::VINTRP && in.opcode > 1) ||
            (in.fmt == Rdna2Format::EXP && in.exp_compr &&
             observation == FragmentPacketExportObservation::LegacyRaw))
            result.rejection = "packet-vgpr-read-form-unimplemented";
        for (uint32_t src = 0; src < in.n_src; ++src) {
            if (in.fmt == Rdna2Format::EXP &&
                !(fragment_packet_export_source_mask(in.exp_en, in.exp_compr) & (1u << src)))
                continue;
            if (in.src[src].kind != OperandKind::VGPR) continue;
            const uint32_t width = in.fmt == Rdna2Format::MIMG && src == 0
                                       ? (in.opcode == 0x24   ? 3u
                                          : in.opcode == 0x27 ? 2u
                                                              : 0u)
                                       : 1u;
            if (!width) result.rejection = "packet-vgpr-read-form-unimplemented";
            const auto kind = in.fmt == Rdna2Format::EXP ? FragmentPacketVgprRead::RawExport
                              : in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360
                                  ? FragmentPacketVgprRead::SelectedPeer
                                  : FragmentPacketVgprRead::Direct;
            for (uint32_t word = 0; word < width; ++word) read(in.src[src].value + word, kind);
        }
        if (in.fmt == Rdna2Format::VINTRP && in.opcode == 1)
            read(in.dst.value, FragmentPacketVgprRead::InterpolationPrevious);
        auto defined = writers[i];
        for (uint32_t word = 0; word < rdna2_vgpr_write_count(in); ++word) {
            const int reg = in.dst.value + static_cast<int>(word);
            if (reg < 0 || reg > 255)
                result.rejection = "packet-vgpr-register-out-of-range";
            else {
                result.storage.set(reg);
                defined.set(reg);
            }
        }
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        if (branch && (branch_target(in) <= in.pc || !indices.contains(branch_target(in)))) {
            result.rejection = "packet-vgpr-control-flow-unimplemented";
            continue;
        }
        if (!reached[i] || in.is_end) continue;
        if (branch) meet(indices.at(branch_target(in)), defined);
        if ((!branch || in.opcode != 2) && i + 1 < ins.size()) meet(i + 1, defined);
    }
    return result;
}
FragmentPacketVgprRequirements
fragment_packet_vgpr_requirements(const std::vector<uint32_t>& code) {
    if (code.empty() || code.size() > 4096) {
        FragmentPacketVgprRequirements result;
        result.source_words = &code;
        result.rejection = "packet-vgpr-program-inventory-incomplete";
        return result;
    }
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    return fragment_packet_vgpr_requirements(code, ins);
}
}   // namespace prosper::gpu
