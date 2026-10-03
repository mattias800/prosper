#include "gpu/recompiler/fragment_packet_definedness.hpp"
#include <string_view>

namespace prosper::gpu {
void PacketVgprDefinedness::begin(SpirvCompute& b, const std::map<int, uint32_t>& columns) {
    uint32_t ptr_bool = 0, ptr_uint = 0;
    for (uint32_t reg = 0; reg < 256; ++reg)
        if (requirements.storage.test(reg)) {
            const auto variable = b.function_var(b.t_bool, ptr_bool);
            valid_vars.emplace(reg, variable);
            const auto column = columns.find(reg);
            const auto valid =
                column == columns.end()
                    ? b.bfalse()
                    : b.ucmp(Op_INotEqual,
                             b.load_input(static_cast<uint32_t>(columns.size()) + column->second),
                             b.uconst(0));
            b.store_function(variable, valid);
        }
    failure_pc = b.function_var(b.t_u32, ptr_uint);
    failure_reg = b.function_var(b.t_u32, ptr_uint);
    failure_kind = b.function_var(b.t_u32, ptr_uint);
    peer_valid = b.function_var(b.t_bool, ptr_bool);
    peer_pc = b.function_var(b.t_u32, ptr_uint);
    peer_reg = b.function_var(b.t_u32, ptr_uint);
    b.store_function(failure_pc, b.uconst(UINT32_MAX));
    b.store_function(failure_reg, b.uconst(UINT32_MAX));
    b.store_function(failure_kind, b.uconst(0));
    b.store_function(peer_valid, b.bfalse());
    b.store_function(peer_pc, b.uconst(UINT32_MAX));
    b.store_function(peer_reg, b.uconst(UINT32_MAX));
}
void PacketVgprDefinedness::fail(SpirvCompute& b, uint32_t condition, uint32_t pc, uint32_t reg,
                                 uint32_t kind) {
    const auto old_kind = b.load_function(b.t_u32, failure_kind);
    const auto first = b.land(condition, b.ucmp(Op_IEqual, old_kind, b.uconst(0)));
    b.store_function(failure_pc, b.sel(first, pc, b.load_function(b.t_u32, failure_pc)));
    b.store_function(failure_reg, b.sel(first, reg, b.load_function(b.t_u32, failure_reg)));
    b.store_function(failure_kind, b.sel(first, kind, old_kind));
}
void PacketVgprDefinedness::instruction(SpirvCompute& b, const RegState& state,
                                        const Rdna2Inst& in) {
    if (const auto found = requirements.reads.find(in.pc); found != requirements.reads.end())
        for (const auto& read : found->second) {
            const auto valid = b.load_function(b.t_bool, valid_vars.at(read.reg));
            if (read.kind == FragmentPacketVgprRead::SelectedPeer) {
                // This source is NOT read in each EXEC-active lane. The existing uniformly reached
                // READLANE phase must test the SELECTED peer, including EXEC-off peer lanes.
                b.store_function(peer_valid, valid);
                b.store_function(peer_pc, b.uconst(in.pc));
                b.store_function(peer_reg, b.uconst(read.reg));
                continue;
            }
            // Existing raw-EXP ABI observes enabled payload even EXEC-off. Preserve that stronger
            // observation contract: an absent inactive payload is a named failure, not a zero.
            const auto consumed =
                read.kind == FragmentPacketVgprRead::RawExport ? b.btrue() : state.exec;
            fail(b, b.land(consumed, b.logical_not(valid)), b.uconst(in.pc), b.uconst(read.reg),
                 b.uconst(static_cast<uint32_t>(read.kind)));
        }
    // Read all OLD sources (P2's VDST too) before establishing any overlapping new definition.
    // This inventory admits only full-dword EXEC-predicated vector writes, never partial/RMW ones.
    for (uint32_t word = 0; word < rdna2_vgpr_write_count(in); ++word) {
        const auto variable = valid_vars.at(in.dst.value + static_cast<int>(word));
        b.store_function(variable,
                         b.ucmp(Op_LogicalOr, state.exec, b.load_function(b.t_bool, variable)));
    }
}
void PacketVgprDefinedness::publish_peer(SpirvCompute& b, uint32_t pending) {
    // A tag prevents a stale/different READLANE event from supplying peer authority. No barrier
    // here: caller publishes beside values, then uses the EXISTING uniform service barrier.
    const auto tag =
        b.ibin(Op_ShiftLeftLogical, b.ibin(Op_IAdd, b.load_function(b.t_u32, peer_pc), b.uconst(1)),
               b.uconst(1));
    const auto metadata = b.ibin(
        Op_BitwiseOr, tag, b.sel(b.load_function(b.t_bool, peer_valid), b.uconst(1), b.uconst(0)));
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(peer_scratch_base), b.linear_localid),
                        b.sel(pending, metadata, b.uconst(0)));
}
void PacketVgprDefinedness::consume_peer(SpirvCompute& b, uint32_t pending, uint32_t index) {
    const auto metadata = b.cfg_scratch_load(b.ibin(Op_IAdd, b.uconst(peer_scratch_base), index));
    const auto pc = b.load_function(b.t_u32, peer_pc);
    const auto expected = b.ibin(
        Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, b.ibin(Op_IAdd, pc, b.uconst(1)), b.uconst(1)),
        b.uconst(1));
    fail(b, b.land(pending, b.ucmp(Op_INotEqual, metadata, expected)), pc,
         b.load_function(b.t_u32, peer_reg),
         b.uconst(static_cast<uint32_t>(FragmentPacketVgprRead::SelectedPeer)));
}
void PacketVgprDefinedness::finish(SpirvCompute& b, uint32_t offset) {
    b.store_output_word(b.uconst(kFragmentPacketVgprStatusMagic), kFragmentPacketVgprStatusWords,
                        offset, 0);
    b.store_output_word(b.load_function(b.t_u32, failure_pc), kFragmentPacketVgprStatusWords,
                        offset + 1, 0);
    b.store_output_word(b.load_function(b.t_u32, failure_reg), kFragmentPacketVgprStatusWords,
                        offset + 2, 0);
    b.store_output_word(b.load_function(b.t_u32, failure_kind), kFragmentPacketVgprStatusWords,
                        offset + 3, 0);
}

bool emit_cfg_readlane_phase(SpirvCompute& b, PacketVgprDefinedness* definedness,
                             uint32_t pending_var, uint32_t source_var, uint32_t selector_var,
                             uint32_t destination_var, const std::set<int>& destinations,
                             const std::map<int, uint32_t>& scalar_vars,
                             const std::map<int, uint32_t>& mask_vars,
                             const std::map<int, uint32_t>& mask_half_vars, uint32_t vcc_var) {
    const auto zero = b.uconst(0), no = b.bfalse();
    // Every invocation publishes its VGPR value unconditionally; READLANE ignores EXEC. The
    // uniform common region and both original barriers remain shared by ALL physical workers.
    const uint32_t pending = b.load_function(b.t_bool, pending_var);
    const uint32_t source = b.load_function(b.t_u32, source_var);
    b.cfg_scratch_store(b.linear_localid, source);
    if (definedness) definedness->publish_peer(b, pending);
    b.barrier();
    const uint32_t shift = b.uconst(b.wave_size == 32 ? 5u : 6u);
    const uint32_t wave_base =
        b.ibin(Op_ShiftLeftLogical, b.ibin(Op_ShiftRightLogical, b.linear_localid, shift), shift);
    const uint32_t lane =
        b.ibin(Op_BitwiseAnd, b.load_function(b.t_u32, selector_var), b.uconst(b.wave_size - 1u));
    const uint32_t index = b.ibin(Op_IAdd, wave_base, lane);
    const uint32_t valid = b.ucmp(Op_ULessThan, index, b.uconst(b.local_count));
    const uint32_t safe_index = b.sel(valid, index, zero);
    const uint32_t result = b.sel(valid, b.cfg_scratch_load(safe_index), zero);
    if (definedness) definedness->consume_peer(b, pending, safe_index);
    const uint32_t dst = b.load_function(b.t_u32, destination_var);
    for (int reg : destinations) {
        const auto destination = scalar_vars.find(reg);
        if (destination == scalar_vars.end()) return false;
        const uint32_t selected =
            b.land(pending, b.ucmp(Op_IEqual, dst, b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, destination->second);
        b.store_function(destination->second, b.sel(selected, result, old));
    }
    for (const auto& kv : mask_vars) {
        if (!destinations.contains(kv.first)) continue;
        const uint32_t selected =
            b.land(pending, b.ucmp(Op_IEqual, dst, b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    for (const auto& kv : mask_half_vars) {
        if (!destinations.contains(kv.first)) continue;
        const uint32_t selected =
            b.land(pending, b.ucmp(Op_IEqual, dst, b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    if (destinations.contains(106)) {
        const uint32_t selected = b.land(pending, b.ucmp(Op_IEqual, dst, b.uconst(106u)));
        const uint32_t bit =
            b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, result, b.uconst(1u)), zero);
        const uint32_t old = b.load_function(b.t_bool, vcc_var);
        b.store_function(vcc_var, b.bsel(selected, bit, old));
    }
    b.barrier();
    return true;
}

FragmentPacketResult decode_fragment_packet(const FragmentPacketProgram& program,
                                            std::span<const uint32_t> words, bool completed) {
    FragmentPacketResult result;
    const auto reject = [&](const char* reason) {
        result.rejection = reason;
        std::fprintf(stderr, "[fragment-packet-reject] reason=%s lane=%u pc=%u vgpr=%u read=%u\n",
                     reason, result.lane, result.pc, result.reg, result.kind);
        return result;
    };
    const uint64_t exports = uint64_t(program.exports_per_lane) * 64 * kFragmentPacketExportWords;
    if (!completed) return reject("packet-completion-or-host-availability-unproved");
    if (program.spirv.size() < 5 || program.spirv[0] != 0x07230203u || !program.rejection.empty() ||
        !program.exports_per_lane || program.exports_per_lane > 64 ||
        words.size() != program.output_words.size() || words.size() < exports)
        return reject("packet-output-abi-invalid");
    bool marked = false;
    for (size_t pc = 5; pc < program.spirv.size();) {
        const auto count = program.spirv[pc] >> 16;
        if (!count || count > program.spirv.size() - pc) return reject("packet-module-abi-invalid");
        if ((program.spirv[pc] & 0xffffu) == Op_ModuleProcessed && count > 1) {
            const auto* text = reinterpret_cast<const char*>(program.spirv.data() + pc + 1);
            const auto* end =
                static_cast<const char*>(std::memchr(text, 0, (count - 1) * sizeof(uint32_t)));
            if (!end) return reject("packet-module-abi-invalid");
            marked |= std::string_view(text, static_cast<size_t>(end - text)) ==
                      kPacketVgprValidityMarker;
        }
        pc += count;
    }
    if (marked != (program.vgpr_status_offset != UINT32_MAX) ||
        (!marked && !program.vgpr_failure_sites.empty()))
        return reject("packet-vgpr-status-abi-invalid");
    if (program.vgpr_status_offset != UINT32_MAX) {
        if (program.vgpr_status_offset < exports ||
            uint64_t(program.vgpr_status_offset) + 64 * kFragmentPacketVgprStatusWords !=
                words.size() ||
            program.vgpr_failure_sites.empty())
            return reject("packet-vgpr-status-abi-invalid");
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto offset = program.vgpr_status_offset + lane * kFragmentPacketVgprStatusWords;
            const FragmentPacketProgram::VgprFailureSite site{words[offset + 1], words[offset + 2],
                                                              words[offset + 3]};
            if (words[offset] != kFragmentPacketVgprStatusMagic ||
                (!site.kind && (site.pc != UINT32_MAX || site.reg != UINT32_MAX)) ||
                (site.kind &&
                 std::find(program.vgpr_failure_sites.begin(), program.vgpr_failure_sites.end(),
                           site) == program.vgpr_failure_sites.end()))
                return reject("packet-vgpr-status-record-invalid");
            if (site.kind && !result.kind) {
                result.lane = lane;
                result.pc = site.pc;
                result.reg = site.reg;
                result.kind = site.kind;
            }
        }
        result.vgpr_status_validated = true;
        if (result.kind)
            return reject(result.kind == static_cast<uint32_t>(FragmentPacketVgprRead::SelectedPeer)
                              ? "packet-vgpr-selected-peer-unavailable"
                          : result.kind == static_cast<uint32_t>(FragmentPacketVgprRead::RawExport)
                              ? "packet-vgpr-raw-export-unavailable"
                              : "packet-vgpr-read-before-definition");
    }
    for (uint64_t offset = 0; offset < exports; offset += kFragmentPacketExportWords) {
        if (!words[offset]) {
            for (uint32_t field = 1; field < kFragmentPacketExportWords; ++field)
                if (words[offset + field]) return reject("packet-unreached-export-invalid");
            continue;
        }
        if (words[offset] != 1 || words[offset + 1] > 1 || words[offset + 2] > 1 ||
            words[offset + 4] > 15 || words[offset + 5] || words[offset + 6] > 1 ||
            words[offset + 7] > 1 ||
            (words[offset + 3] >= kFragmentColorOutputs && words[offset + 3] != 8 &&
             words[offset + 3] != 9) ||
            (words[offset + 3] == 8 && (!words[offset + 4] || (words[offset + 4] & ~5u))) ||
            (words[offset + 3] == 9 && words[offset + 4]))
            return reject("packet-export-record-invalid");
        for (uint32_t channel = 0; channel < 4; ++channel)
            if (!(words[offset + 4] & (1u << channel)) && words[offset + 8 + channel])
                return reject("packet-disabled-export-payload-invalid");
    }
    result.exports.assign(words.begin(), words.begin() + static_cast<size_t>(exports));
    return result;
}
}   // namespace prosper::gpu
