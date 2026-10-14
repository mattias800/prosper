#include "gpu/recompiler/rdna2_wave32_active_data.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>
namespace prosper::gpu {
bool tracks_wave32_active_data(const SpirvCompute& b) {
    // CURRENT lane-wise DATA needs the actual native width, not full launch participation.
    // In particular this must not authorize physical mask words or SCC reductions from a ballot.
    const auto& launch = b.guest_compute_launch;
    const uint64_t xy = uint64_t{launch.local[0]} * launch.local[1];
    if (!launch.local[2] || xy > UINT32_MAX / launch.local[2]) return false;
    const uint64_t count = xy * launch.local[2];
    return b.is_compute && !b.is_fragment && !b.is_vertex && !b.is_fragment_packet() &&
           !b.ngg_workgroup_shell && !b.ngg_one_lane && b.wave_size == 32 &&
           b.native_subgroup_size == 32 && b.guest_stage == GuestShaderStage::Compute &&
           b.physical_domain == PhysicalExecutionDomain::Workgroup &&
           launch.wave_size == b.wave_size && launch.local == b.guest_local_size && count &&
           count % 32 == 0 && count == b.local_count;
}

void meet_wave32_active_data(Wave32ActiveData& into, const Wave32ActiveData& incoming) {
    std::erase_if(into.current, [&](int word) { return !incoming.current.contains(word); });
    meet_exec_spill_equality(into.exec_spill, incoming.exec_spill);
    into.vcc_produced = into.vcc_produced && incoming.vcc_produced;
    into.control = into.control && incoming.control;
    if (!into.control) into.current.clear();
}
void advance_wave32_active_data(Wave32ActiveData& data, const Rdna2Inst& in,
                                const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                const std::vector<int>& vector_writes, int mask_write,
                                bool genuine_sources) {
    // CONFIDENCE: HIGH -- the shell's initial VCC is storage, not an architectural input.
    // Lose this new recovery at the input read, before a whole-EXEC numeric relay can erase
    // the old MAY mark. Later writers/compares cannot repair that unproved dependency.
    bool reads_vcc = (in.fmt == Rdna2Format::SOPP && (in.opcode == 6 || in.opcode == 7)) ||
                     (in.fmt == Rdna2Format::VOP2 &&
                      (in.opcode == 1 || (in.opcode >= 0x28 && in.opcode <= 0x2a)));
    bool ordinary_vcc_words = false;
    for (uint32_t source = 0; source < in.n_src; ++source) {
        const auto& operand = in.src[source];
        if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) continue;
        const auto width = scalar_alu_source_words(in, source);
        if (operand.value > 107 || operand.value < 0 ||
            uint64_t{static_cast<uint32_t>(operand.value)} + width <= 106)
            continue;
        reads_vcc = true;
        // A genuine Wave32 low predicate does not publish ordinary numeric/physical VCC words.
        // Only the actual low-mask operand of these vector forms uses that predicate service.
        const bool low_mask = in.fmt == Rdna2Format::VOP3 && source == 2 &&
                              (in.opcode == 0x101 || (in.opcode >= 0x128 && in.opcode <= 0x12a)) &&
                              operand.value == 106;
        ordinary_vcc_words |= !low_mask;
    }
    const auto implicit_words = scalar_implicit_destination_read_width(in);
    if (implicit_words && in.dst.value >= 0 && in.dst.value <= 107 &&
        uint64_t{static_cast<uint32_t>(in.dst.value)} + implicit_words > 106) {
        reads_vcc = true;
        ordinary_vcc_words = true;
    }
    if (reads_vcc && (!data.vcc_produced || ordinary_vcc_words)) {
        data.control = false;
        data.current.clear();
    }
    genuine_sources = genuine_sources && data.control;
    for (const auto& [base, width] : scalar_writes)
        if (base <= 107 && base + static_cast<int>(width) > 106) data.vcc_produced = false;
    if (mask_write == 106 || mask_write == 107) data.vcc_produced = false;
    if (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2a)
        data.vcc_produced = false;   // an implicit carry output needs its own input-mask proof

    if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
        (in.dst.kind == OperandKind::SGPR ? in.dst.value : 106) == 106)
        data.vcc_produced = genuine_sources;
    // CONFIDENCE: HIGH -- the inline spill lowering keeps raw slot keys, but RDNA2 Wave32
    // aliases selectors modulo32. Retire every physical alias before publishing a raw write.
    // A read through a different raw spelling gets no equality: that transport is not lowered
    // by the same stored slot. Dynamic/out-of-range writes retain the helper's all-slot kill.
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361 && in.n_src >= 2 &&
        in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 && in.src[1].value < 64)
        std::erase_if(data.exec_spill.slots, [&](const auto& slot) {
            return slot.first.first == in.dst.value &&
                   (slot.first.second & 31) == (in.src[1].value & 31);
        });
    const bool unchanged =
        advance_exec_spill_equality(data.exec_spill, in, scalar_writes, vector_writes, mask_write);
    if (rdna2_instruction_may_change_exec(in) && !unchanged) {
        // CMPX only removes lanes. Every other unproved write may enable lanes with missing DATA.
        if (in.fmt != Rdna2Format::VOPC || !vopc_is_cmpx(in.opcode)) data.current.clear();
    }
    if (vector_write_leaves_old_word(in, false)) return;
    // Only ordinary lane-wise arithmetic publishes this new fact. Memory, selected peers, old
    // accumulators and partial destinations retain their existing independent admission proofs.
    const bool valu =
        in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 || in.fmt == Rdna2Format::VOP3;
    const bool complete = valu && vector_write_completes_active_word(in);
    for (int word : vector_writes) {
        if (!genuine_sources)
            data.current.erase(word);
        else if (complete)
            data.current.insert(word);
    }
}
}   // namespace prosper::gpu
