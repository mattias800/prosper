#include "gpu/recompiler/rdna2_wave32_dispatch_words.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <set>
#include <utility>
#include <vector>
namespace prosper::gpu {
Wave32DispatchWords::Wave32DispatchWords(std::size_t count)
    : masks(count), ambiguous(count), pairs(count), ambiguous_pairs(count),
      reachable(count, false) {}
bool analyze_wave32_dispatch_words(
    SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const RegState& initial,
    const std::vector<uint32_t>& starts, const std::vector<std::vector<uint32_t>>& successors,
    bool proven_wave32_masks, bool compute_scalar_vcc_bridge, Wave32DispatchWords& words,
    std::set<int>& static_mask_keys,
    std::unordered_set<uint32_t>& proven_saved_mask_pair_compare_pcs,
    const std::function<std::array<int, 2>(const Rdna2Inst&)>& saved_mask_pair_compare_sources) {
    auto& b32_mask_in = words.masks;
    auto& b32_mask_ambiguous_in = words.ambiguous;
    auto& b64_mask_in = words.pairs;
    auto& b64_mask_ambiguous_in = words.ambiguous_pairs;
    auto& b32_mask_reachable = words.reachable;
    if (b.allow_b32_masks && !starts.empty()) {
        b32_mask_in.front().insert(initial.sreg_bool_b32.begin(), initial.sreg_bool_b32.end());
        for (const auto& mask : initial.sreg_bool) {
            if (!initial.sreg_bool_b32.contains(mask.first) && mask.first != 106 &&
                mask.first != 107)
                b64_mask_in.front().insert(mask.first);
        }
        // `vcc` is the implicit VCC_LO mask even when no instruction has needed an explicit
        // sreg_bool[106] alias yet. Preserve that valid entry lifetime in the same physical-domain
        // analysis as explicit Wave32 masks. A zero SSA id instead means the caller currently has
        // ordinary scalar data in the VCC words, so it deliberately does not seed the mask domain.
        if (initial.vcc) b32_mask_in.front().insert(106);
        b32_mask_reachable.front() = true;
        auto implicit_vcc_mask_source = [](const Rdna2Inst& in) -> int {
            if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07))
                return 106;   // s_cbranch_vccz/nz
            if (in.fmt == Rdna2Format::VOP2 &&
                (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a)))
                return 106;   // e32 cndmask and carry-in/out forms use implicit VCC
            if (in.fmt == Rdna2Format::VOP3 &&
                (in.opcode == 0x101 || (in.opcode >= 0x128 && in.opcode <= 0x12a))) {
                const Operand& mask = in.src[2];
                if ((mask.kind == OperandKind::SGPR || mask.kind == OperandKind::Special) &&
                    (mask.value == 106 || mask.value == 107))
                    return mask.value;
            }
            if (in.fmt == Rdna2Format::VOP3 && (in.opcode == 0x365 || in.opcode == 0x366)) {
                const Operand& mask = in.src[0];
                if ((mask.kind == OperandKind::SGPR || mask.kind == OperandKind::Special) &&
                    (mask.value == 106 || mask.value == 107))
                    return mask.value;
            }
            return -1;
        };
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> masks = b32_mask_in[block];
            std::set<int> ambiguous = b32_mask_ambiguous_in[block];
            std::set<int> b64_masks = b64_mask_in[block];
            std::set<int> b64_ambiguous = b64_mask_ambiguous_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;

                const auto compare_sources = saved_mask_pair_compare_sources(in);
                if (compare_sources[0] >= 0) {
                    const bool proven = b64_masks.contains(compare_sources[0]) &&
                                        b64_masks.contains(compare_sources[1]) &&
                                        !b64_ambiguous.contains(compare_sources[0]) &&
                                        !b64_ambiguous.contains(compare_sources[1]);
                    if (proven)
                        proven_saved_mask_pair_compare_pcs.insert(in.pc);
                    else
                        proven_saved_mask_pair_compare_pcs.erase(in.pc);
                }

                // The dispatcher may use a false backing value for a physical VCC lifetime that is
                // provably dead. Do not let that implementation placeholder become observable: every
                // instruction whose ISA encoding requires a VCC mask must see one on all incoming
                // paths. Explicit SGPR operands remain governed by the data/mask-domain checks below.
                const int implicit_vcc = implicit_vcc_mask_source(in);
                if (implicit_vcc >= 0 &&
                    (!masks.contains(implicit_vcc) || ambiguous.contains(implicit_vcc))) {
                    log_recompile_diagnostic(
                        b.diagnostic, b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
                        "terminal", "pc=%u reason=missing-wave32-vcc-mask", in.pc);
                    return false;
                }

                bool reads_ambiguous = false;
                for (uint32_t source = 0; source < in.n_src; ++source) {
                    if ((in.src[source].kind == OperandKind::SGPR ||
                         in.src[source].kind == OperandKind::Special)) {
                        const int reg = in.src[source].value;
                        reads_ambiguous |= ambiguous.contains(reg);
                        for (int base : b64_ambiguous)
                            reads_ambiguous |= reg == base || reg == base + 1;
                    }
                }
                if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07) &&
                    ambiguous.contains(106))
                    reads_ambiguous = true;
                if (reads_ambiguous) {
                    log_recompile_diagnostic(
                        b.diagnostic, b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
                        "terminal", "pc=%u reason=wave32-ambiguous-mask-read", in.pc);
                    return false;
                }

                bool writes_b32_mask = false;
                auto register_mask = [&](const Operand& source) {
                    // EXEC_HI is the architectural zero dword in Wave32, not the second half of
                    // a live mask. Keep this transfer classification aligned with emit_alu's
                    // scalar materialization so an ordinary value derived from EXEC_HI survives
                    // dispatcher save/reload boundaries as scalar data.
                    return (source.kind == OperandKind::Special && source.value == 126) ||
                           ((source.kind == OperandKind::SGPR ||
                             source.kind == OperandKind::Special) &&
                            (masks.contains(source.value) || b64_masks.contains(source.value)));
                };
                auto source_mask = [&](const Operand& source) {
                    return register_mask(source) || source.kind == OperandKind::InlineInt ||
                           source.kind == OperandKind::Literal;
                };
                if (in.fmt == Rdna2Format::SOP1 &&
                    (in.opcode == 0x03 || in.opcode == 0x07 || in.opcode == 0x09 ||
                     sop1_opcode_is_emitted_saveexec_b32(in.opcode))) {
                    const bool source_is_mask =
                        register_mask(in.src[0]) ||
                        (in.dst.value == 126 && in.src[0].kind == OperandKind::InlineInt);
                    writes_b32_mask =
                        source_is_mask && in.dst.value != 127 &&
                        (!sop1_opcode_is_emitted_saveexec_b32(in.opcode) || in.dst.value != 126);
                }
                if (in.fmt == Rdna2Format::SOP2 &&
                    (in.opcode == 0x0a ||
                     (in.opcode >= 0x0e && in.opcode <= 0x1c && (in.opcode & 1u) == 0))) {
                    const bool scalar_vcc_bridge =
                        compute_scalar_vcc_bridge && is_scalar_cselect_b32_to_vcc_lo(in);
                    const bool scalar_vcchi_packet = compute_scalar_vcc_bridge &&
                                                     b.native_subgroup_size == 32 &&
                                                     is_gtav_wave32_vcchi_scalar_packet(in);
                    const bool mask_domain = in.dst.value == 126 || in.src[0].value == 126 ||
                                             in.src[1].value == 126 ||
                                             (((in.src[0].kind == OperandKind::SGPR ||
                                                in.src[0].kind == OperandKind::Special) &&
                                               masks.contains(in.src[0].value))) ||
                                             (((in.src[1].kind == OperandKind::SGPR ||
                                                in.src[1].kind == OperandKind::Special) &&
                                               masks.contains(in.src[1].value)));
                    writes_b32_mask =
                        !scalar_vcchi_packet &&
                        (scalar_vcc_bridge || (mask_domain && source_mask(in.src[0]) &&
                                               source_mask(in.src[1]) && in.dst.value != 127));
                }

                int b32_write_reg = writes_b32_mask ? in.dst.value : -1;
                const bool wave32_b64_vcc_mask_write =
                    in.fmt == Rdna2Format::SOP2 && in.dst.value == 106 && in.opcode >= 0x0f &&
                    in.opcode <= 0x1d && (in.opcode & 1u) == 1 && source_mask(in.src[0]) &&
                    source_mask(in.src[1]);
                if (wave32_b64_vcc_mask_write) b32_write_reg = 106;
                if (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 && in.opcode <= 0x12a &&
                    in.sdst.kind == OperandKind::SGPR) {
                    const Operand& carry_in = in.src[2];
                    if (register_mask(carry_in)) b32_write_reg = in.sdst.value;
                } else if (vop3b_fresh_carry_output(in)) {
                    b32_write_reg = in.sdst.value;
                }

                // B64 mask-shaped instructions are not sufficient to prove a live mask lifetime:
                // s_mov_b64 and the logical family also move ordinary 64-bit scalar data. Require
                // their sources to be masks in the current path state. Saveexec always writes OLD
                // EXEC to its explicit destination, while BFM and VOP3B construct fresh masks.
                bool writes_b64_mask = false;
                if (in.fmt == Rdna2Format::SOP1 && in.dst.value <= 105) {
                    if (in.opcode == 0x04 || in.opcode == 0x08 || in.opcode == 0x0a)
                        writes_b64_mask = source_mask(in.src[0]);
                    else if ((in.opcode >= 0x24 && in.opcode <= 0x2b) || in.opcode == 0x37 ||
                             in.opcode == 0x38)
                        writes_b64_mask = true;
                }
                if (in.fmt == Rdna2Format::SOP2 && in.dst.value <= 105) {
                    if (in.opcode == 0x25)
                        writes_b64_mask = true;
                    else if (in.opcode == 0x0b ||
                             (in.opcode >= 0x0f && in.opcode <= 0x1d && (in.opcode & 1u) == 1))
                        writes_b64_mask = source_mask(in.src[0]) && source_mask(in.src[1]);
                }
                const bool vop3_b32_carry =
                    in.fmt == Rdna2Format::VOP3 &&
                    ((in.opcode >= 0x128 && in.opcode <= 0x12a) || vop3b_fresh_carry_output(in));
                // rdna2_decode populates `sdst` only for its explicit ten-opcode VOP3B whitelist:
                // add/sub carry, div-scale flag, and 64-bit multiply-add carry outputs. Each SDST
                // is a per-lane scalar mask/flag; ordinary VOP3A scalar data never appears here
                // (v_readlane uses `dst`, not `sdst`).
                if (vop3_writes_mask_sdst(in) && !vop3_b32_carry && in.sdst.value <= 105)
                    writes_b64_mask = true;
                int b64_write_reg =
                    writes_b64_mask ? (in.fmt == Rdna2Format::VOP3 ? in.sdst.value : in.dst.value)
                                    : -1;

                auto erase_b64_overlapping = [&](int base, uint32_t width) {
                    auto erase = [&](std::set<int>& values) {
                        for (auto it = values.begin(); it != values.end();) {
                            const int mask_base = *it;
                            if (base < mask_base + 2 && mask_base < base + static_cast<int>(width))
                                it = values.erase(it);
                            else
                                ++it;
                        }
                    };
                    erase(b64_masks);
                    erase(b64_ambiguous);
                };

                for_each_scalar_write(
                    in,
                    [&](int base, uint32_t width) {
                        const bool one_word_write = base == b32_write_reg;
                        const uint32_t effective_width = one_word_write ? 1u : width;
                        erase_b64_overlapping(base, effective_width);
                        for (uint32_t word = 0; word < effective_width; ++word) {
                            const int reg = base + static_cast<int>(word);
                            if (!one_word_write || reg != b32_write_reg) {
                                masks.erase(reg);
                                ambiguous.erase(reg);
                            }
                        }
                        if (one_word_write && b32_write_reg != 126) {
                            masks.insert(b32_write_reg);
                            ambiguous.erase(b32_write_reg);
                        }
                    },
                    proven_wave32_masks);

                if (b64_write_reg >= 0) {
                    for (int word = 0; word < 2; ++word) {
                        masks.erase(b64_write_reg + word);
                        ambiguous.erase(b64_write_reg + word);
                    }
                    b64_masks.insert(b64_write_reg);
                    b64_ambiguous.erase(b64_write_reg);
                }

                // Every non-CMPX VOPC destination is a one-word mask in proven Wave32, including
                // explicit ordinary SGPRs. CMPX writes EXEC only and establishes no VCC lifetime.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
                    const int destination = in.dst.kind == OperandKind::SGPR ? in.dst.value : 106;
                    erase_b64_overlapping(destination, 1);
                    masks.insert(destination);
                    ambiguous.erase(destination);
                }
            }

            for (uint32_t successor : successors[block]) {
                if (!b32_mask_reachable[successor]) {
                    b32_mask_reachable[successor] = true;
                    b32_mask_in[successor] = masks;
                    b32_mask_ambiguous_in[successor] = ambiguous;
                    b64_mask_in[successor] = b64_masks;
                    b64_mask_ambiguous_in[successor] = b64_ambiguous;
                    pending.push_back(successor);
                } else {
                    std::set<int> joined_masks;
                    std::set<int> joined_ambiguous = b32_mask_ambiguous_in[successor];
                    std::set_intersection(b32_mask_in[successor].begin(),
                                          b32_mask_in[successor].end(), masks.begin(), masks.end(),
                                          std::inserter(joined_masks, joined_masks.end()));
                    for (int reg : b32_mask_in[successor])
                        if (!masks.contains(reg)) joined_ambiguous.insert(reg);
                    for (int reg : masks)
                        if (!b32_mask_in[successor].contains(reg)) joined_ambiguous.insert(reg);
                    joined_ambiguous.insert(ambiguous.begin(), ambiguous.end());
                    for (int reg : joined_ambiguous) joined_masks.erase(reg);
                    std::set<int> joined_b64_masks;
                    std::set<int> joined_b64_ambiguous = b64_mask_ambiguous_in[successor];
                    std::set_intersection(b64_mask_in[successor].begin(),
                                          b64_mask_in[successor].end(), b64_masks.begin(),
                                          b64_masks.end(),
                                          std::inserter(joined_b64_masks, joined_b64_masks.end()));
                    for (int reg : b64_mask_in[successor])
                        if (!b64_masks.contains(reg)) joined_b64_ambiguous.insert(reg);
                    for (int reg : b64_masks)
                        if (!b64_mask_in[successor].contains(reg)) joined_b64_ambiguous.insert(reg);
                    joined_b64_ambiguous.insert(b64_ambiguous.begin(), b64_ambiguous.end());
                    for (int reg : joined_b64_ambiguous) joined_b64_masks.erase(reg);
                    if (joined_masks != b32_mask_in[successor] ||
                        joined_ambiguous != b32_mask_ambiguous_in[successor] ||
                        joined_b64_masks != b64_mask_in[successor] ||
                        joined_b64_ambiguous != b64_mask_ambiguous_in[successor]) {
                        b32_mask_in[successor] = std::move(joined_masks);
                        b32_mask_ambiguous_in[successor] = std::move(joined_ambiguous);
                        b64_mask_in[successor] = std::move(joined_b64_masks);
                        b64_mask_ambiguous_in[successor] = std::move(joined_b64_ambiguous);
                        pending.push_back(successor);
                    }
                }
            }
        }
        for (const auto& masks : b32_mask_in)
            for (int reg : masks)
                if (reg <= 107) static_mask_keys.insert(reg);
        for (const auto& masks : b64_mask_in)
            for (int reg : masks)
                if (reg <= 105) static_mask_keys.insert(reg);
    }

    return true;
}
}   // namespace prosper::gpu
