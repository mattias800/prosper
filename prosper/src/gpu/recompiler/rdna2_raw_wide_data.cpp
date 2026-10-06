// Distinguish descriptor provenance from scalar data in typeless raw x4/x8 SMEM loads.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
// A raw wide load backed by a dispatch-time CPU snapshot cannot observe an earlier shader write
// to that same guest allocation. Without an alias proof, keep potentially writing instructions
// on the ordinary unresolved route. Read-only operations remain eligible.
//
// One answer for "may this instruction write guest memory": the ISA-table-pinned classifier
// (rdna2_decode.cpp, exhaustively checked by test_may_write_memory). This used to keep its own
// MIMG reader list (0x00/0x0e/0x27/0x2f), grown one opcode at a time, so a pure read such as
// image_sample_l (0x24) counted as a writer here alone. Kena's NGG vertex programs sample after a
// register-offset descriptor load, and that phantom write forced the load to need backing it
// could never prove (#4422). MUBUF/MTBUF/SMEM/FLAT already agreed; unlisted opcodes stay writers.
bool rdna2_may_write_guest_memory(const Rdna2Inst& in) {
    return rdna2_instruction_may_write_memory(in);
}

namespace {

struct RawWideState {
    size_t index;
    uint16_t live;
};

// One load's live words, tracked along every direct branch until overwrite or exit. An uncertain
// control edge or ordinary scalar read is a refusal to replace those words with zero. This does
// not certify descriptor provenance; resource resolution still decides whether a consumer works.
class RawWideLifetime {
public:
    // The first instruction that stopped each walk, for diagnostics only (#4499). A walk that
    // reports only "failed" makes every widening of it a guess: three different follow-ups hide
    // behind one `needs-backing=1`. Never read by the classification itself.
    struct Blocker {
        uint32_t pc = UINT32_MAX;
        const char* kind = "none";
    };
    const Blocker& backing_blocker() const { return backing_blocker_; }
    const Blocker& numeric_blocker() const { return numeric_blocker_; }

    RawWideLifetime(const std::vector<Rdna2Inst>& instructions,
                    const std::unordered_map<uint32_t, size_t>& pc_indices,
                    size_t load_index, uint32_t word_count)
        : ins(instructions), by_pc(pc_indices), start(load_index),
          first(instructions[load_index].dst.value), words(word_count) {}

    bool requires_backing() const {
        const auto blocked = [&](uint32_t pc, const char* kind) {
            backing_blocker_ = {pc, kind};
            return true;
        };
        if (first + static_cast<int>(words) > 106)
            return blocked(ins[start].pc, "destination-above-s105");
        std::vector<RawWideState> pending;
        std::unordered_set<uint64_t> visited;
        if (start + 1 < ins.size())
            pending.push_back({start + 1, static_cast<uint16_t>((1u << words) - 1u)});
        while (!pending.empty()) {
            const RawWideState state = pending.back();
            pending.pop_back();
            if (!state.live || state.index >= ins.size() || state.index == start) continue;
            const uint64_t key = (static_cast<uint64_t>(state.index) << 8u) | state.live;
            if (!visited.insert(key).second) continue;
            const Rdna2Inst& in = ins[state.index];
            // s_movrels_b32 is not in this set: reads_data below sees its source range.
            if (in.fmt == Rdna2Format::Unknown || !in.len_dwords ||
                rdna2_may_write_unnamed_register_or_leave_cfg(in))
                return blocked(in.pc, "unknown-or-indirect-control");
            if (in.is_end) continue;
            if (reads_data(in, state.live)) return blocked(in.pc, "data-read");
            const uint16_t live = kill_written_words(in, state.live);
            if (live && !enqueue_successors(in, state.index, live, pending))
                return blocked(in.pc, "unmodelled-control");
        }
        return false;
    }

    bool has_numeric_reader_or_uncertain_path() const {
        const auto blocked = [&](uint32_t pc, const char* kind) {
            numeric_blocker_ = {pc, kind};
            return true;
        };
        // Scalar data is MAY provenance (OR at joins). A separate MUST fact identifies exact
        // fresh mask roots used by the emitter's Bool consumers (AND at joins). A compare can
        // replace that Bool without proving that both physical scalar words were overwritten.
        // Keep the possible high-word numeric dependency alive for every ordinary data reader.
        struct State {
            size_t index;
            std::bitset<128> regs;
            bool scc;
            std::bitset<128> masks;
        };
        if (start + 1 >= ins.size()) return false;
        // A destination that reaches past s105 loads VCC, M0 or EXEC themselves. Those are
        // read by instructions that never name them, and the walk below starts from an EXEC
        // that does not depend on the load; neither holds for such a load, so it is numeric
        // without a walk. requires_backing() already answers the same way.
        if (first + static_cast<int>(words) > 106)
            return blocked(ins[start].pc, "destination-above-s105");
        State initial{start + 1, {}, false, {}};
        initial.masks = saved_exec_masks_at_load();
        initial.masks.set(126); // entry EXEC cannot depend on this subsequent, unreplayed load
        for (uint32_t word = 0; word < words; ++word)
            initial.regs.set(static_cast<size_t>(first + static_cast<int>(word)));
        std::vector<State> pending{initial};
        std::vector<std::array<std::bitset<128>, 2>> seen(ins.size());
        std::vector<std::array<std::bitset<128>, 2>> seen_masks(ins.size());
        std::vector<std::array<bool, 2>> seen_any(ins.size());
        const auto mask_register = [](const Operand& operand) {
            if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special)
                return -1;
            return operand.value >= 0 &&
                (operand.value <= 105 || operand.value == 106 || operand.value == 126)
                ? operand.value : -1;
        };
        const auto supported_compare = [](const Rdna2Inst& in) {
            // Match emit_alu's exact effective compare families. CMPX publishes EXEC only;
            // its SDST-shaped decoder field cannot manufacture a saved-mask lifetime.
            const uint32_t op = vopc_is_cmpx(in.opcode) ? in.opcode - 0x10u : in.opcode;
            return in.fmt == Rdna2Format::VOPC &&
                (op <= 0x0fu || (op >= 0x81u && op <= 0x86u) || op == 0x88u ||
                 (op >= 0x89u && op <= 0x8eu) || (op >= 0xa1u && op <= 0xa6u) ||
                 (op >= 0xa9u && op <= 0xaeu) || (op >= 0xc1u && op <= 0xc6u) ||
                 (op >= 0xc9u && op <= 0xceu) || (op >= 0xe1u && op <= 0xe6u));
        };
        // What an instruction does to a derived SCC. Replaces: SCC is rewritten, so it is derived
        // exactly when this instruction read something derived. Keeps: SCC is untouched. Unknown:
        // it may be either, and a taint has to survive that -- a write is assumed for what it
        // adds and not for what it would clear. The old rule cleared on "not known to keep",
        // which ended a compare-on-a-loaded-word at s_ff1_i32_b32, at the relative moves and at
        // two of the three SOP2 packs, all of which leave SCC alone.
        enum class SccEffect { Keeps, Replaces, Unknown };
        auto scc_effect = [](const Rdna2Inst& in) {
            if (in.fmt == Rdna2Format::SOPC) return SccEffect::Replaces;
            if (in.fmt == Rdna2Format::SOP1) {
                if (sop1_opcode_writes_scc(in.opcode)) return SccEffect::Replaces;
                return sop1_opcode_leaves_scc_unmodified(in.opcode) ? SccEffect::Keeps
                                                                    : SccEffect::Unknown;
            }
            if (in.fmt == Rdna2Format::SOPK)
                return (in.opcode >= kSopkOpcodeCmpkFirst && in.opcode <= kSopkOpcodeCmpkLast) ||
                               in.opcode == kSopkOpcodeAddkI32
                           ? SccEffect::Replaces
                           : SccEffect::Keeps;
            if (in.fmt == Rdna2Format::SOP2) {
                // gfx10 SOP2: cselect reads SCC; BFM, MUL, the three packs and MUL_HI leave it.
                if (in.opcode == 0x0au || in.opcode == 0x0bu || in.opcode == kSop2OpcodeBfmB32 ||
                    in.opcode == kSop2OpcodeBfmB64 || in.opcode == 0x26u ||
                    (in.opcode >= 0x32u && in.opcode <= 0x36u))
                    return SccEffect::Keeps;
                // add/sub/min/max, the logicals and shifts, BFE, ABSDIFF, LSHLn_ADD.
                if (in.opcode <= 0x09u || (in.opcode >= 0x0eu && in.opcode <= 0x23u) ||
                    (in.opcode >= 0x27u && in.opcode <= 0x2au) || in.opcode == 0x2cu ||
                    (in.opcode >= 0x2eu && in.opcode <= 0x31u))
                    return SccEffect::Replaces;
                return SccEffect::Unknown;
            }
            return SccEffect::Keeps;
        };
        size_t processed = 0;
        while (!pending.empty()) {
            State state = std::move(pending.back());
            pending.pop_back();
            if ((!state.regs.any() && !state.scc) || state.index >= ins.size()) continue;
            // Control came back to the load itself. One descriptor is resolved per fetch PC, so
            // a load that can observe DIFFERENT bytes on a later execution cannot keep using the
            // first observation: that stays uncertain. A load that provably reads the same bytes
            // every time is handled below, once it has been walked like any other instruction.
            if (state.index == start && !replay_observes_same_bytes())
                return blocked(ins[start].pc, "load-re-executed");
            const size_t slot = state.scc ? 1u : 0u;
            if (seen_any[state.index][slot] &&
                (state.regs & ~seen[state.index][slot]).none() &&
                (seen_masks[state.index][slot] & ~state.masks).none()) continue;
            if (seen_any[state.index][slot])
                seen_masks[state.index][slot] &= state.masks;
            else
                seen_masks[state.index][slot] = state.masks;
            seen_any[state.index][slot] = true;
            seen[state.index][slot] |= state.regs;
            state.regs = seen[state.index][slot];
            state.masks = seen_masks[state.index][slot];
            if (++processed > 32768) return blocked(ins[state.index].pc, "walk-budget");
            const Rdna2Inst& in = ins[state.index];
            if (in.fmt == Rdna2Format::Unknown || !in.len_dwords)
                return blocked(in.pc, "unknown-instruction");
            if (in.is_end) continue;
            // Indirect control, a subvector loop, or an M0-relative move. The range this used
            // to test for the last of those, 0x28..0x2a, is three B64 saveexec forms: they were
            // refused here for nothing, and the real relative moves were not refused at all.
            // s_movrels_b32 is not in this set either. It is an ordinary scalar derivation here:
            // source_width() gives it every register from its base up to s105, so a loaded word
            // in that range taints the destination, and a later numeric reader of the
            // destination stops the walk. It cannot end a derived SCC (scc_effect above), and its
            // index, M0, is never a derived word when control gets here: see
            // derived-value-enters-m0.
            if (rdna2_may_write_unnamed_register_or_leave_cfg(in))
                return blocked(in.pc, "unmodelled-control-or-relative-sgpr");

            if (state.scc && in.fmt == Rdna2Format::SOPP &&
                (in.opcode == 0x04u || in.opcode == 0x05u))
                return blocked(in.pc, "scc-branch-on-derived-value");
            // Readers of VCC that never name it (#4527): the e32 select and carry-in forms and the
            // vccz/vccnz branches, the same list sgpr_dead_at_merge keeps. They consume the pair as
            // a lane mask, so they are harmless exactly when VCC IS one: a fresh compare into VCC,
            // or a transfer from an independent mask root, which is what masks[106] records. With
            // a derived word in the pair and no such fact, the "mask" is the load's own bytes --
            // `s_mov_b64 vcc, s[16:17]` straight after the load, then a select or a branch.
            //
            // Wave width does not enter into it. In Wave64 a fresh compare wrote both words. In
            // Wave32 it wrote only vcc_lo, a derived word may survive in vcc_hi, and these readers
            // do not look at vcc_hi there.
            if ((state.regs.test(106) || state.regs.test(107)) && !state.masks.test(106) &&
                ((in.fmt == Rdna2Format::VOP2 &&
                  (in.opcode == 0x01u || (in.opcode >= 0x28u && in.opcode <= 0x2au))) ||
                 (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06u || in.opcode == 0x07u))))
                return blocked(in.pc, "implicit-vcc-reader");
            if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x08u || in.opcode == 0x09u) &&
                !state.masks.test(126))
                return blocked(in.pc, "exec-branch-on-dependent-exec");
            const auto independent_mask = [&](const Operand& operand) {
                if (operand.kind == OperandKind::InlineInt)
                    return operand.value == 0 || operand.value == -1;
                const int root = mask_register(operand);
                return root >= 0 && state.masks.test(static_cast<size_t>(root));
            };
            const bool mask_logical = in.fmt == Rdna2Format::SOP2 &&
                (in.opcode == 0x0fu || in.opcode == 0x11u || in.opcode == 0x13u ||
                 in.opcode == 0x15u || in.opcode == 0x17u || in.opcode == 0x19u ||
                 in.opcode == 0x1bu || in.opcode == 0x1du);
            // The unary B64 mask transfers: S_MOV, S_NOT and S_WQM. From an independent mask each
            // produces an independent mask, and the emitter keeps all three in the Bool domain
            // (into VCC as well: all three update the VCC the branches read).
            // S_WQM is the one that matters: nearly every pixel shader opens with
            // `s_wqm_b64 exec, exec`, and while only S_MOV was listed, that instruction left EXEC
            // "dependent" for every load fetched ABOVE it -- the walk starts at the load with
            // EXEC independent, so a load below the prologue never saw it. For those early loads
            // no later compare counted as fresh and every consumer of a recycled pair was a
            // numeric reader (#4555: GTA V's V# loads at pc 4, above the prologue at pc 9).
            const bool mask_move = in.fmt == Rdna2Format::SOP1 && (in.opcode == kSop1OpcodeMovB64 ||
                                                                   in.opcode == kSop1OpcodeNotB64 ||
                                                                   in.opcode == kSop1OpcodeWqmB64);
            const bool mask_saveexec = in.fmt == Rdna2Format::SOP1 &&
                in.opcode >= kSop1OpcodeAndSaveexecB64 &&
                in.opcode <= kSop1OpcodeXnorSaveexecB64;
            const bool independent_transfer = mask_register(in.dst) >= 0 &&
                ((mask_logical && independent_mask(in.src[0]) &&
                                  independent_mask(in.src[1])) ||
                 (mask_move && independent_mask(in.src[0])) ||
                 (mask_saveexec && independent_mask(in.src[0]) && state.masks.test(126)));
            const bool implicit_scc_read = state.scc &&
                ((in.fmt == Rdna2Format::SOP2 &&
                  (in.opcode == 0x04u || in.opcode == 0x05u ||
                   in.opcode == 0x0au || in.opcode == 0x0bu)) ||
                 (in.fmt == Rdna2Format::SOP1 &&
                  (in.opcode == kSop1OpcodeCmovB32 ||
                   in.opcode == kSop1OpcodeCmovB64)) ||
                 (in.fmt == Rdna2Format::SOPK &&
                  in.opcode == kSopkOpcodeCmovkI32));
            const uint32_t copy_words = in.fmt == Rdna2Format::SOP1 &&
                in.opcode == kSop1OpcodeMovB32 ? 1u :
                in.fmt == Rdna2Format::SOP1 &&
                in.opcode == kSop1OpcodeMovB64 ? 2u : 0u;
            const bool plain_copy = copy_words && in.dst.kind == OperandKind::SGPR &&
                (in.src[0].kind == OperandKind::SGPR ||
                 in.src[0].kind == OperandKind::Special);
            bool derived_read = implicit_scc_read;
            if (!plain_copy) {
                const uint32_t implicit = scalar_implicit_destination_read_width(in);
                for (uint32_t k = 0; k < implicit; ++k)
                    if (in.dst.value >= 0 && in.dst.value + static_cast<int>(k) < 128 &&
                        state.regs.test(static_cast<size_t>(in.dst.value + k)))
                        derived_read = true;
                for (uint32_t source = 0; source < in.n_src; ++source) {
                    const Operand& operand = in.src[source];
                    // v_cndmask's condition was already exempt here when it is an independent
                    // mask root. The carry-in of the three VOP3B add/sub-with-carry forms is the
                    // same kind of operand: the emitter takes it as a Bool and refuses an
                    // untracked one (rdna2_emit_alu.cpp), so from an independent root the Bool it
                    // reads is the fresh compare's and no loaded word is observed.
                    if (in.fmt == Rdna2Format::VOP3 && source == 2u &&
                        (in.opcode == 0x101u || (in.opcode >= 0x128u && in.opcode <= 0x12au)) &&
                        independent_mask(operand))
                        continue;
                    if (operand.kind == OperandKind::Special && operand.value == 253) {
                        derived_read |= state.scc;
                        continue;
                    }
                    if ((operand.kind != OperandKind::SGPR &&
                         !(operand.kind == OperandKind::Special &&
                           operand.value >= 106 && operand.value <= 127)) ||
                        descriptor_source(in, source)) continue;
                    for (uint32_t k = 0; k < source_width(in, source); ++k)
                        if (operand.value >= 0 &&
                            operand.value + static_cast<int>(k) < 128 &&
                            state.regs.test(static_cast<size_t>(operand.value + k)))
                            derived_read = true;
                }
            }
            const bool scalar_result = in.fmt == Rdna2Format::SOP1 ||
                in.fmt == Rdna2Format::SOP2 || in.fmt == Rdna2Format::SOPK;
            if (derived_read && !scalar_result && in.fmt != Rdna2Format::SOPC)
                return blocked(in.pc, "numeric-reader");
            if (derived_read && scalar_result &&
                (in.dst.kind != OperandKind::SGPR ||
                 (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeSetregB32) ||
                 rdna2_instruction_may_change_exec(in)) &&
                !independent_transfer)
                return blocked(in.pc, "derived-value-leaves-scalar-data");

            // Evaluate inputs before expiring overlapping roots. Both siblings are invalidated
            // conservatively: a saved Bool must never authorize a later physical data lifetime.
            const int compare_root = in.dst.value == 126 ? -1 : mask_register(in.dst);
            const bool fresh_compare = supported_compare(in) &&
                !vopc_is_cmpx(in.opcode) && compare_root >= 0 &&
                !derived_read && state.masks.test(126);
            // The emitter narrows EXEC with CMPX's explicit comparison. Independent inputs
            // preserve an already independent EXEC, without overwriting SDST/VCC or proving
            // anything about their numeric high words.
            const bool independent_cmpx = supported_compare(in) &&
                vopc_is_cmpx(in.opcode) && !derived_read && state.masks.test(126);
            const bool fresh_exec = independent_cmpx || (independent_transfer &&
                (mask_saveexec || mask_register(in.dst) == 126));
            // The e32 add/sub-with-carry forms write their carry-out to VCC without naming it,
            // so no writer inventory reports it. The emitter builds it as it builds a compare
            // result, a Bool with the bit of an inactive lane written 0, out of the two operands,
            // the carry-in (the VCC it replaces) and EXEC. A derived operand has already stopped
            // the walk as a numeric reader, so VCC is kept as an independent root afterwards
            // only when it was one before and EXEC is one.
            // Only the compare was handled here, so a VCC that held a saved EXEC kept that fact
            // across a carry-out formed under an EXEC nothing vouched for (#4574); the pass that
            // seeds a load already ended the save there.
            const bool carry_out =
                in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28u && in.opcode <= 0x2au;
            const bool root_carry = carry_out && state.masks.test(126) && state.masks.test(106);
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (uint32_t k = 0; k < width; ++k) {
                    const int reg = base + static_cast<int>(k);
                    if (reg >= 0 && reg < 128) state.masks.reset(static_cast<size_t>(reg));
                    if (reg > 0 && reg <= 128) state.masks.reset(static_cast<size_t>(reg - 1));
                }
            });
            if ((in.fmt == Rdna2Format::VOPC && in.dst.value == 106 && !vopc_is_cmpx(in.opcode)) ||
                carry_out)
                state.masks.reset(106);
            if (rdna2_instruction_may_change_exec(in)) state.masks.reset(126);
            if (fresh_compare) state.masks.set(static_cast<size_t>(compare_root));
            if (root_carry) state.masks.set(106);
            if (independent_transfer) state.masks.set(static_cast<size_t>(mask_register(in.dst)));
            if (fresh_exec) state.masks.set(126);

            std::array<bool, 2> copied{};
            if (plain_copy)
                for (uint32_t k = 0; k < copy_words; ++k)
                    copied[k] = (k == 0 && state.scc &&
                                 in.src[0].kind == OperandKind::Special &&
                                 in.src[0].value == 253) ||
                        (in.src[0].value >= 0 &&
                         in.src[0].value + static_cast<int>(k) < 128 &&
                         state.regs.test(static_cast<size_t>(in.src[0].value + k)));
            std::bitset<128> produced;
            const bool conditional_write =
                (in.fmt == Rdna2Format::SOP1 &&
                 (in.opcode == kSop1OpcodeCmovB32 ||
                  in.opcode == kSop1OpcodeCmovB64)) ||
                (in.fmt == Rdna2Format::SOPK &&
                 in.opcode == kSopkOpcodeCmovkI32);
            const bool definite_scalar_write =
                !conditional_write &&
                (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                 in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SMEM ||
                 (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) ||
                 // The two lane reads write their SGPR whatever EXEC is. Only v_readlane was
                 // listed, so a register v_readfirstlane had just replaced kept its old mark.
                 (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02u) ||
                 (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360u));
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (uint32_t k = 0; k < width; ++k) {
                    const int reg = base + static_cast<int>(k);
                    if (reg < 0 || reg >= 128) continue;
                    if (definite_scalar_write)
                        state.regs.reset(static_cast<size_t>(reg));
                    if (derived_read && scalar_result &&
                        (in.dst.kind == OperandKind::SGPR || independent_transfer) &&
                        base == in.dst.value)
                        produced.set(static_cast<size_t>(reg));
                }
            }, /*wave32_one_word_masks*/true);
            if (in.fmt == Rdna2Format::VOPC && in.dst.value == 106 &&
                !vopc_is_cmpx(in.opcode))
                state.regs.reset(106); // only the guaranteed VCC low word, no width assumption
            state.regs |= produced;
            // The load itself, reached again round a back-edge, reading the same bytes as before
            // (checked on entry). It was processed like any other instruction on the way here:
            // its destination words were expired as a definite scalar write, and what it writes
            // now is the same observation this walk started from. Those words become derived
            // again and the walk runs to a fixed point -- MAY words only grow, and the MUST mask
            // roots have already been narrowed by everything executed on the way round, entry
            // EXEC included.
            // Every return to the load used to answer "uncertain". A per-light loop reloads its
            // V# from the resource table on each iteration and only ever hands it to
            // s_buffer_load as SBASE; that load then needed backing with no reader anywhere, as
            // soon as the cheaper walk above was unsure about one of its words.
            if (state.index == start)
                for (uint32_t word = 0; word < words; ++word)
                    state.regs.set(static_cast<size_t>(first) + word);
            if (plain_copy)
                for (uint32_t k = 0; k < copy_words; ++k)
                    if (copied[k] && in.dst.value >= 0 &&
                        in.dst.value + static_cast<int>(k) < 128)
                        state.regs.set(static_cast<size_t>(in.dst.value + k));
            // M0 is read by instructions that never name it: the relative moves (s_movrels_b32
            // is walked through above, and v_movrels_b32 gets no special treatment here), the
            // LDS and append/consume forms, and s_sendmsg. A loaded word that reaches M0 is
            // consumed as a number by whichever of them runs next, and no operand scan can see
            // that, so it is counted here, where the word goes in.
            if (state.regs.test(124)) return blocked(in.pc, "derived-value-enters-m0");
            // EXEC is read by nearly everything and named by nothing: every lane write below
            // is predicated by it, and a compare run under it folds it into the mask it
            // produces. Every other way a derived word can reach EXEC is stopped above (the
            // scalar forms as derived-value-leaves-scalar-data, a v_cmpx on a derived operand as
            // numeric-reader), but a plain s_mov takes the copy path, where nothing is a reader.
            // It marked EXEC and went on, and the mark did nothing: the lanes written under
            // that EXEC were not readers, and a mask formed under it carried no mark at all, so
            // it could outlive the walk (#4574). So the copy itself is the numeric use, as it is
            // for M0.
            // A transfer from an independent root is not one. Its high word may still be the
            // load's on a 32-lane view, and the copy marks EXEC's accordingly, but what the
            // emitter installs is the root's Bool.
            if (plain_copy && !independent_transfer)
                for (uint32_t k = 0; k < copy_words; ++k)
                    if (copied[k] && (in.dst.value + static_cast<int>(k) == 126 ||
                                      in.dst.value + static_cast<int>(k) == 127))
                        return blocked(in.pc, "derived-value-enters-exec");
            if (const SccEffect effect = scc_effect(in); effect == SccEffect::Replaces)
                state.scc = derived_read;
            else if (effect == SccEffect::Unknown)
                state.scc = state.scc || derived_read;
            if (!state.regs.any() && !state.scc) continue;

            auto enqueue = [&](size_t next) {
                // MAY words grow and MUST mask roots shrink at joins, including backedges.
                // The finite worklist must converge within the cap above. A path back to the
                // load is followed like any other; the load re-derives its words above.
                pending.push_back({next, state.regs, state.scc, state.masks});
            };
            if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
                const int64_t target_pc = static_cast<int64_t>(in.pc) +
                    in.len_dwords + in.simm16;
                if (target_pc < 0 || target_pc > UINT32_MAX)
                    return blocked(in.pc, "branch-target-out-of-range");
                const auto target = by_pc.find(static_cast<uint32_t>(target_pc));
                if (target == by_pc.end()) {
                    if (target_pc <= ins.back().pc)
                        return blocked(in.pc, "branch-target-mid-instruction");
                } else {
                    enqueue(target->second);
                }
                if (in.opcode == kSoppOpcodeBranch) continue;
            } else if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
                       in.opcode != 0x0au && in.opcode != 0x10u &&
                       in.opcode != 0x16u && in.opcode != 0x17u) {
                return blocked(in.pc, "unmodelled-control");
            }
            if (state.index + 1 < ins.size()) enqueue(state.index + 1);
        }
        return false;
    }

    bool has_reader_after_bypassed_load(size_t entry) const {
        std::vector<RawWideState> pending{{entry,
            static_cast<uint16_t>((1u << words) - 1u)}};
        std::unordered_set<uint64_t> visited;
        while (!pending.empty()) {
            const RawWideState state = pending.back();
            pending.pop_back();
            if (!state.live || state.index >= ins.size()) continue;
            const uint64_t key = (static_cast<uint64_t>(state.index) << 8u) | state.live;
            if (!visited.insert(key).second) continue;
            if (visited.size() > 8192) return true;
            const Rdna2Inst& in = ins[state.index];
            if (in.fmt == Rdna2Format::Unknown || !in.len_dwords) return true;
            if (in.is_end) continue;
            const uint32_t implicit = scalar_implicit_destination_read_width(in);
            if (implicit && overlaps(in.dst.value, implicit, state.live)) return true;
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const Operand& operand = in.src[source];
                if ((operand.kind == OperandKind::SGPR ||
                     (operand.kind == OperandKind::Special &&
                      operand.value >= 106 && operand.value <= 124)) &&
                    overlaps(operand.value, source_width(in, source), state.live))
                    return true;
            }
            const uint16_t live = kill_written_words(in, state.live);
            if (live && !enqueue_successors(in, state.index, live, pending)) return true;
        }
        return false;
    }

private:
    const std::vector<Rdna2Inst>& ins;
    const std::unordered_map<uint32_t, size_t>& by_pc;
    size_t start;
    int first;
    uint32_t words;
    mutable Blocker backing_blocker_, numeric_blocker_;

    // Whether every execution of this load reads the same bytes: an immediate offset from a base
    // pair that no instruction ever writes, in a program that cannot write guest memory at all.
    // Deliberately whole-program and path-insensitive -- being wrong about "same" would hand a
    // later iteration an earlier iteration's descriptor. A register SOFFSET never qualifies,
    // whatever its value: the offset is exactly what a loop changes.
    //
    // "No instruction ever writes" is only as good as the writer inventory, so everything that
    // inventory is known not to see through refuses outright, wherever it sits in the program:
    // rdna2_may_write_unnamed_register_or_leave_cfg (calls, indirect transfers, subvector loops,
    // M0-relative moves other than s_movrels_b32) and any SMEM instruction other than a plain
    // load. s_movrels_b32 reads through M0 and cannot change the base pair, so it is not on this
    // list; the walks see its read range instead. That is the known list, not a
    // proof of completeness; an instruction for_each_scalar_write misreports and this does not
    // name would be a hole here.
    bool replay_observes_same_bytes() const {
        const Rdna2Inst& load = ins[start];
        if (load.src[1].kind != OperandKind::Special || load.src[1].value != 125 ||
            load.src[0].kind != OperandKind::SGPR || load.src[0].value < 0 ||
            load.src[0].value + 1 > 105)
            return false;
        for (const Rdna2Inst& in : ins) {
            if (in.fmt == Rdna2Format::Unknown || rdna2_may_write_guest_memory(in)) return false;
            if (rdna2_may_write_unnamed_register_or_leave_cfg(in)) return false;
            // An SMEM instruction that is not one of the ten plain loads has no entry in the
            // writer inventory at all (scalar_write_width answers 0), s_memtime's SDATA pair
            // included. None compiles today; refuse rather than rely on that.
            if (in.fmt == Rdna2Format::SMEM && !scalar_write_width(in)) return false;
            bool writes_base = false;
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                if (base <= load.src[0].value + 1 &&
                    base + static_cast<int>(width) > load.src[0].value)
                    writes_base = true;
            });
            if (writes_base) return false;
        }
        return true;
    }

    std::bitset<128> saved_exec_masks_at_load() const {
        // Only actual mask saves, on every path that reaches this load from above, may seed its
        // Bool facts. Their value predates this load; no scalar bits, wave width or zero are
        // inferred.
        //
        // A MUST dataflow over the instructions before the load, run to a fixed point. It used
        // to be one forward pass that returned nothing at the first backward branch, so a loop
        // anywhere above a load cost it every seed; and it accepted a save only into s0..s104,
        // while compilers also park EXEC in VCC (`s_mov_b64 vcc, exec` ... `s_mov_b64 exec,
        // vcc` around a guarded sample). Both together kept a GTA V program on the owned-wave
        // path (#4555).
        //
        // The pass only sees instructions before the load, and ignores a branch that leaves
        // that range, as the sweep it replaces did. So a path that jumps over the load and
        // reaches it from below can arrive with a seeded pair holding something else. A seed
        // asserts one thing, that the pair does not depend on this load. On the first arrival
        // that holds outright: everything was written before the load ran. On a later arrival
        // it holds only as far as every value that depends on the load carries a mark while it
        // lives, because then a pair rewritten from such a value is a write the walk reaches,
        // and it drops the fact there. The one known way round that was a plain copy of loaded
        // words into EXEC: a compare run under that EXEC yields a mask with no mark, which
        // could be moved into a seeded pair after the walk had stopped and come back round a
        // loop as an independent EXEC (#4574). The walk now stops at that copy, and ends a
        // VCC-held root at a carry-out formed under an EXEC that is not one.
        std::vector<std::bitset<128>> incoming(start + 1);
        std::vector<bool> reached(start + 1);
        reached[0] = true;
        std::vector<size_t> pending{0};
        size_t steps = 0;
        while (!pending.empty()) {
            const size_t index = pending.back();
            pending.pop_back();
            if (index >= start) continue;   // the load is where the facts are read
            if (++steps > 65536) return {};
            const auto& in = ins[index];
            if (in.fmt == Rdna2Format::Unknown || !in.len_dwords ||
                rdna2_may_write_unnamed_register_or_leave_cfg(in))
                return {};
            auto masks = incoming[index];
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (uint32_t k = 0; k < width; ++k) {
                    const int reg = base + static_cast<int>(k);
                    if (reg >= 0 && reg < 128) masks.reset(static_cast<size_t>(reg));
                    if (reg > 0 && reg <= 128) masks.reset(static_cast<size_t>(reg - 1));
                }
            });
            // VCC has writers the explicit inventory does not report: a compare with the
            // implicit destination, and the carry-out of the e32 add/sub-with-carry forms.
            if ((in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) && in.dst.value == 106) ||
                (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28u && in.opcode <= 0x2au))
                masks.reset(106);
            if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 &&
                (in.dst.value <= 104 || in.dst.value == 106) &&
                in.src[0].kind == OperandKind::Special && in.src[0].value == 126)
                masks.set(static_cast<size_t>(in.dst.value));
            const auto enter = [&](size_t next) {
                if (next > start) return;
                const std::bitset<128> joined = reached[next] ? incoming[next] & masks : masks;
                if (reached[next] && joined == incoming[next]) return;
                incoming[next] = joined;
                reached[next] = true;
                pending.push_back(next);
            };
            if (in.is_end) continue;
            if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
                const int64_t target_pc = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
                if (target_pc < 0 || target_pc > UINT32_MAX) return {};
                const auto target = by_pc.find(static_cast<uint32_t>(target_pc));
                if (target == by_pc.end()) {
                    if (target_pc <= ins.back().pc) return {};
                } else {
                    enter(target->second);
                }
                if (in.opcode == kSoppOpcodeBranch) continue;
            } else if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
                       in.opcode != 0x0au && in.opcode != 0x10u &&
                       in.opcode != 0x16u && in.opcode != 0x17u) return {};
            enter(index + 1);
        }
        auto masks = reached[start] ? incoming[start] : std::bitset<128>{};
        // The load itself overwrites data and invalidates an overlapping saved pair.
        for_each_scalar_write(ins[start], [&](int base, uint32_t width) {
            for (uint32_t k = 0; k < width; ++k) {
                const int reg = base + static_cast<int>(k);
                if (reg >= 0 && reg < 128) masks.reset(static_cast<size_t>(reg));
                if (reg > 0 && reg <= 128) masks.reset(static_cast<size_t>(reg - 1));
            }
        });
        return masks;
    }

    bool overlaps(int base, uint32_t width, uint16_t live) const {
        if (base < 0) return false;
        for (uint32_t word = 0; word < words; ++word) {
            const int reg = first + static_cast<int>(word);
            if ((live & (1u << word)) && base <= reg &&
                reg < base + static_cast<int>(width)) return true;
        }
        return false;
    }

    static uint32_t source_width(const Rdna2Inst& in, uint32_t source) {
        if (in.fmt == Rdna2Format::MIMG) {
            if (source == 1) return 8;
            if (source == 2) return 4;
        }
        if ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) && source == 1)
            return 4;
        if (in.fmt == Rdna2Format::SMEM && source == 0)
            return in.opcode >= 8u ? 4u : 2u;
        if (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
            in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::VOP3 ||
            in.fmt == Rdna2Format::VOPC) {
            const uint32_t width = scalar_alu_source_words(in, source);
            return width == UINT32_MAX ? 0u : width;
        }
        return 1;
    }

    static bool descriptor_source(const Rdna2Inst& in, uint32_t source) {
        if (in.fmt == Rdna2Format::MIMG) return source == 1 || source == 2;
        if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF)
            return source == 1;
        return in.fmt == Rdna2Format::SMEM && in.opcode >= 8u && source == 0;
    }

    bool reads_data(const Rdna2Inst& in, uint16_t live) const {
        const uint32_t implicit_width = scalar_implicit_destination_read_width(in);
        if (implicit_width && overlaps(in.dst.value, implicit_width, live)) return true;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const Operand& operand = in.src[source];
            if (operand.kind != OperandKind::SGPR &&
                !(operand.kind == OperandKind::Special &&
                  operand.value >= 106 && operand.value <= 124)) continue;
            if (overlaps(operand.value, source_width(in, source), live) &&
                !descriptor_source(in, source)) return true;
        }
        return false;
    }

    uint16_t kill_written_words(const Rdna2Inst& in, uint16_t live) const {
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            if (base < 0) return;
            for (uint32_t word = 0; word < words; ++word)
                if (base <= first + static_cast<int>(word) &&
                    first + static_cast<int>(word) < base + static_cast<int>(width))
                    live &= static_cast<uint16_t>(~(1u << word));
        }, /*wave32_one_word_masks*/true);
        return live;
    }

    bool enqueue_successors(const Rdna2Inst& in, size_t index, uint16_t live,
                            std::vector<RawWideState>& pending) const {
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target_pc = static_cast<int64_t>(in.pc) +
                static_cast<int64_t>(in.len_dwords) + in.simm16;
            if (target_pc < 0 || target_pc > UINT32_MAX) return false;
            const auto target = by_pc.find(static_cast<uint32_t>(target_pc));
            if (target == by_pc.end()) {
                // A branch beyond the decoded program ends this path. A missing internal target
                // is unproven and cannot retain placeholder authority.
                if (target_pc <= ins.back().pc) return false;
            } else {
                pending.push_back({target->second, live});
            }
            if (in.opcode == kSoppOpcodeBranch) return true;
        } else if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
                   in.opcode != 0x0au && in.opcode != 0x10u &&
                   in.opcode != 0x16u && in.opcode != 0x17u) {
            // 0x10/0x16/0x17 have no register/control effect in the emitter. All other unknown
            // transfers might enter a reader outside this decoded path.
            return false;
        }
        if (index + 1 < ins.size()) pending.push_back({index + 1, live});
        return true;
    }
};

} // namespace

std::vector<uint32_t> rdna2_raw_wide_data_loads(const std::vector<Rdna2Inst>& ins) {
    std::vector<uint32_t> data_loads;
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t index = 0; index < ins.size(); ++index)
        by_pc.emplace(ins[index].pc, index);
    const bool has_guest_write = std::any_of(ins.begin(), ins.end(),
                                             rdna2_may_write_guest_memory);
    for (size_t index = 0; index < ins.size(); ++index) {
        const Rdna2Inst& load = ins[index];
        if (load.fmt != Rdna2Format::SMEM ||
            (load.opcode != 0x2u && load.opcode != 0x3u) ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            load.dst.value + (load.opcode == 0x2u ? 4 : 8) > 128)
            continue;
        const uint32_t words = load.opcode == 0x2u ? 4u : 8u;
        const RawWideLifetime lifetime(ins, by_pc, index, words);
        const bool register_offset = load.src[1].kind != OperandKind::Special ||
            load.src[1].value != 125;
        // Scalar arithmetic may assemble a descriptor even when SOFFSET is a register.
        // The derived-use proof must find an actual non-descriptor observer before forbidding
        // the established exact-fetch-PC descriptor route. Copies and descriptor patches alone
        // are provenance, not scalar data observed by the emitted module. A shader-side memory
        // write could invalidate the front half's dispatch-time descriptor snapshot, so keep
        // newly admitted register-offset patches fail-visible until an alias proof exists.
        if (lifetime.requires_backing() &&
            (lifetime.has_numeric_reader_or_uncertain_path() ||
             (register_offset && has_guest_write)))
            data_loads.push_back(load.pc);
    }
    return data_loads;
}

// The same two walks as rdna2_raw_wide_data_loads, reporting where each one stopped. One row per
// load the classifier calls numeric data; a load it clears has no row. Diagnostics only.
std::vector<RawWideLoadDiagnosis>
rdna2_raw_wide_data_load_diagnoses(const std::vector<Rdna2Inst>& ins) {
    std::vector<RawWideLoadDiagnosis> rows;
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t index = 0; index < ins.size(); ++index) by_pc.emplace(ins[index].pc, index);
    const bool has_guest_write = std::any_of(ins.begin(), ins.end(), rdna2_may_write_guest_memory);
    for (size_t index = 0; index < ins.size(); ++index) {
        const Rdna2Inst& load = ins[index];
        if (load.fmt != Rdna2Format::SMEM || (load.opcode != 0x2u && load.opcode != 0x3u) ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            load.dst.value + (load.opcode == 0x2u ? 4 : 8) > 128)
            continue;
        const RawWideLifetime lifetime(ins, by_pc, index, load.opcode == 0x2u ? 4u : 8u);
        if (!lifetime.requires_backing()) continue;
        const bool numeric = lifetime.has_numeric_reader_or_uncertain_path();
        const bool register_offset =
            load.src[1].kind != OperandKind::Special || load.src[1].value != 125;
        if (!numeric && !(register_offset && has_guest_write)) continue;
        RawWideLoadDiagnosis row;
        row.load_pc = load.pc;
        row.backing_pc = lifetime.backing_blocker().pc;
        row.backing_kind = lifetime.backing_blocker().kind;
        row.numeric_pc = numeric ? lifetime.numeric_blocker().pc : load.pc;
        row.numeric_kind =
            numeric ? lifetime.numeric_blocker().kind : "register-offset-with-guest-memory-write";
        rows.push_back(row);
    }
    return rows;
}

// A small, deliberately stricter subset of the above refusal population can use a current-byte
// buffer. The predicate above reports uncertainty as "needs backing"; it must never itself grant
// admission. Here the entire decoded program has only valid forward edges, and the raw pointer is
// an unchanged entry pair. A load then observes one dispatch-local upload on every visit.
// `pointer_until_read` ends the entry-pointer lifetime at the load instead of the program end. Only
// the register-offset path asks for it: its load reads the fold's exact per-PC snapshot and never
// the base register again, and forward-only control means the load runs at most once, so only
// writes BEFORE it can change which bytes it reads (#4578 follow-up). The strict immediate set and
// the owned read points keep their own lifetimes, so their consumers are unchanged.
static std::vector<uint32_t> proven_immediate_wide_data_loads(const std::vector<Rdna2Inst>& ins,
                                                              bool owned_read_point,
                                                              bool pointer_until_read = false) {
    std::vector<uint32_t> proven;
    if (ins.empty()) return proven;
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i)
        if (ins[i].fmt == Rdna2Format::Unknown || !ins[i].len_dwords ||
            !by_pc.emplace(ins[i].pc, i).second) return proven;
    for (const Rdna2Inst& in : ins) {
        if (rdna2_may_write_unnamed_register_or_leave_cfg(in)) return proven;
        if (in.fmt != Rdna2Format::SOPP || in.is_end) continue;
        if (sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            if (target <= static_cast<int64_t>(in.pc) || target > UINT32_MAX ||
                !by_pc.contains(static_cast<uint32_t>(target))) return proven;
        } else if (!sopp_is_noop(in) && in.opcode != kSoppOpcodeBarrier &&
                   // GS_ALLOC_REQ reads M0 to request parameter-cache space; it changes no
                   // scalar value or guest source bytes (RDNA2 ISA 12.5.1). The emitter already
                   // supports this NGG prelude. Other messages remain outside this proof.
                   !(in.opcode == 0x10u && in.simm16 == 9))
            return proven;
    }
    const auto needs_backing = rdna2_raw_wide_data_loads(ins);
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& load = ins[i];
        const uint32_t words = load.opcode == 0x2u ? 4u : 8u;
        if (load.fmt != Rdna2Format::SMEM ||
            (load.opcode != 0x2u && load.opcode != 0x3u) ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            load.dst.value + static_cast<int>(words) > 106 ||
            load.src[0].kind != OperandKind::SGPR || load.src[0].value < 0 ||
            load.src[0].value + 1 > 105 ||
            load.src[1].kind != OperandKind::Special || load.src[1].value != 125 ||
            static_cast<int32_t>(load.literal) < 0 || (load.literal & 3u) ||
            !std::binary_search(needs_backing.begin(), needs_backing.end(), load.pc))
            continue;
        bool stable_entry_pointer = true;
        const RawWideLifetime lifetime(ins, by_pc, i, words);
        for (const Rdna2Inst& in : ins) {
            // Forward-only control permits a textual prefix scan: any earlier global/image
            // store might alias the raw source after the CPU upload was captured.
            if ((owned_read_point || in.pc < load.pc) && rdna2_may_write_guest_memory(in))
                stable_entry_pointer = false;
            // A predecessor may bypass the load only if every bypass path overwrites all
            // would-be loaded words before reading them. This admits branch-to-after-region
            // shapes without allowing a joined reader to see a fabricated definition.
            if (in.pc < load.pc && in.fmt == Rdna2Format::SOPP &&
                sopp_opcode_is_direct_branch(in.opcode)) {
                const int64_t target = static_cast<int64_t>(in.pc) +
                    in.len_dwords + in.simm16;
                if (target > static_cast<int64_t>(load.pc) &&
                    lifetime.has_reader_after_bypassed_load(
                        by_pc.at(static_cast<uint32_t>(target))))
                    stable_entry_pointer = false;
            }
            if ((!owned_read_point && !pointer_until_read) || in.pc < load.pc)
                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    if (base >= 0 && base <= load.src[0].value + 1 &&
                        base + static_cast<int>(width) > load.src[0].value)
                        stable_entry_pointer = false;
                });
            if (!stable_entry_pointer) break;
        }
        if (stable_entry_pointer && lifetime.has_numeric_reader_or_uncertain_path())
            proven.push_back(load.pc);
    }
    return proven;
}

std::vector<uint32_t> rdna2_proven_raw_immediate_wide_data_loads(
        const std::vector<Rdna2Inst>& ins) {
    return proven_immediate_wide_data_loads(ins, false);
}

std::vector<uint32_t>
rdna2_proven_raw_register_wide_entry_loads(const std::vector<Rdna2Inst>& ins) {
    // Reuse the entry-pointer, forward-CFG, bypass-reader and guest-write proofs. Only the
    // candidate's addressing mode changes here; its loaded-word lifetime is unchanged.
    auto immediate = ins;
    for (auto& load : immediate)
        if (load.fmt == Rdna2Format::SMEM &&
            (load.opcode == 0x2u || load.opcode == 0x3u))
            load.src[1] = {OperandKind::Special, 125};
    return proven_immediate_wide_data_loads(immediate, false, true);
}

std::vector<uint32_t>
rdna2_proven_raw_register_wide_data_loads(const std::vector<Rdna2Inst>& ins,
                                          std::vector<uint32_t>* scalar_source_pcs) {
    if (scalar_source_pcs) scalar_source_pcs->clear();
    const auto entry_proven = rdna2_proven_raw_register_wide_entry_loads(ins);
    const auto owned_parents = proven_immediate_wide_data_loads(ins, true);
    std::vector<uint32_t> proven;
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& load = ins[i];
        const auto& offset = load.src[1];
        const bool ordinary = offset.kind == OperandKind::SGPR &&
                              offset.value >= 0 && offset.value <= 105;
        const bool vcc = offset.kind == OperandKind::Special &&
                         (offset.value == 106 || offset.value == 107);
        if ((!ordinary && !vcc) ||
            !std::binary_search(entry_proven.begin(), entry_proven.end(), load.pc))
            continue;
        // The compact fold can omit implicit VALU mask writes. Authenticate the entire reaching
        // scalar-definition chain against the FULL stream before trusting a concrete fold value.
        std::bitset<128> needed;
        needed.set(static_cast<size_t>(offset.value));
        bool scalar_prefix = true;
        std::vector<uint32_t> sources;
        for (size_t j = i; j-- > 0;) {
            const auto& writer = ins[j];
            // This first subset does not select between incoming scalar definitions. A later
            // branch remains allowed by the shared lifetime proof, including a bypass overwrite.
            if (writer.fmt == Rdna2Format::SOPP &&
                sopp_opcode_is_direct_branch(writer.opcode)) {
                scalar_prefix = false;
                break;
            }
            const bool valu = writer.fmt == Rdna2Format::VOP1 ||
                              writer.fmt == Rdna2Format::VOP2 ||
                              writer.fmt == Rdna2Format::VOP3 ||
                              writer.fmt == Rdna2Format::VOPC;
            // Conservative for implicit writers too: no VALU that MAY write VCC may intervene in a
            // live VCC dependency. cannot_write_vcc admits only encodings it can show leave VCC
            // alone: VOP1/VOP2 other than the carry chain and v_readfirstlane into VCC, and VOP3A
            // without a mask SDST other than v_readlane into VCC. VOPC's implicit dst, the VOP2
            // carry chain, a VOP3 mask SDST naming VCC and a lane read into VCC still stop the
            // walk. Kena's NGG vertex programs schedule a VOP3 v_cndmask_b32 with an explicit SGPR
            // mask between `s_and_b32 vcc_lo, ...` and `s_load_dwordx4 ..., vcc_lo` (#4422). Any
            // other scalar write of a needed register is still reported by for_each_scalar_write
            // below. An earlier VALU before its scalar replacement has no such dependency.
            if (valu && (needed.test(106) || needed.test(107)) && !cannot_write_vcc(writer)) {
                scalar_prefix = false;
                break;
            }
            std::bitset<128> writes;
            for_each_scalar_write(writer, [&](int base, uint32_t width) {
                for (uint32_t k = 0; k < width; ++k)
                    if (base + static_cast<int>(k) >= 0 &&
                        base + static_cast<int>(k) < 128)
                        writes.set(static_cast<size_t>(base + static_cast<int>(k)));
            });
            if ((writes & needed).none()) continue;
            // A complete immediate parent supplies the SAME observed words to the fold and
            // emitted stage. Later writes to its entry pointer cannot retarget this earlier read.
            // The parent proof still rejects every potentially aliasing guest write in the shader.
            if (std::binary_search(owned_parents.begin(), owned_parents.end(), writer.pc)) {
                needed &= ~writes;
                sources.push_back(writer.pc);
                // The exact parent already proves its entry pointer and reaching definition.
                // Once every selector word resolves, unrelated earlier control is irrelevant.
                // The child entry-pointer authority remains required by entry_proven above.
                if (needed.none()) break;
                continue;
            }
            // A bounded immediate raw x1 or x2 fetch is a latched scalar value. Its entry pointer
            // needs to survive only UNTIL this read, unlike the wide source pointer, whose
            // full-program lifetime is authenticated above. GTA overwrites this source pair
            // after the read while preserving the loaded scalar that supplies SOFFSET. UE4's
            // vertex-factory fetch loads its index pair with s_load_dwordx2 (Kena, #4578); the
            // owned snapshot then carries both words, since the emitted load writes both.
            const bool immediate_scalar_read =
                writer.fmt == Rdna2Format::SMEM && (writer.opcode == 0u || writer.opcode == 1u) &&
                writer.dst.kind == OperandKind::SGPR && writer.dst.value >= 0 &&
                writer.dst.value + static_cast<int>(writer.opcode) <= 105 &&
                writer.src[0].kind == OperandKind::SGPR && writer.src[0].value >= 0 &&
                writer.src[0].value < 105 && writer.src[1].kind == OperandKind::Special &&
                writer.src[1].value == 125 && writer.literal == 0u;
            if (immediate_scalar_read) {
                bool entry_at_read = true;
                for (size_t prefix = 0; prefix < j && entry_at_read; ++prefix) {
                    const auto& before = ins[prefix];
                    // Control is forward-only (entry_proven). A branch before the source matters
                    // only if it lands after the source and at or before the load: then the load
                    // can run without this read. One landing at or before the source joins
                    // ahead of it; one landing past the load skips both, which the wide load's
                    // own bypass proof covers. Every earlier instruction is still checked for
                    // guest writes and source-pointer writes, on every path.
                    bool lands_inside = false;
                    if (before.fmt == Rdna2Format::SOPP &&
                        sopp_opcode_is_direct_branch(before.opcode)) {
                        const int64_t target =
                            static_cast<int64_t>(before.pc) + before.len_dwords + before.simm16;
                        lands_inside = target > static_cast<int64_t>(writer.pc) &&
                                       target <= static_cast<int64_t>(load.pc);
                    }
                    if (rdna2_may_write_guest_memory(before) || lands_inside ||
                        rdna2_may_write_unnamed_register_or_leave_cfg(before)) {
                        entry_at_read = false;
                        break;
                    }
                    for_each_scalar_write(before, [&](int base, uint32_t width) {
                        if (base >= 0 && base <= writer.src[0].value + 1 &&
                            base + static_cast<int>(width) > writer.src[0].value)
                            entry_at_read = false;
                    });
                }
                if (!entry_at_read) {
                    scalar_prefix = false;
                    break;
                }
                // The existing exact-PC x1 resource supplies the emitted numeric value;
                // realization still requires mapped current bytes for this draw's pointer.
                needed &= ~writes;
                sources.push_back(writer.pc);
                // As for an owned parent: once every selector word resolves, earlier control is
                // irrelevant. A word still needed keeps the walk, and its branch refusal, going.
                if (needed.none()) break;
                continue;
            }
            // Scalar moves and B32 arithmetic are per-draw data, rather than masks or conditional
            // definitions. Concrete operands/opcode support are checked again by the live fold.
            const bool scalar_writer =
                (writer.fmt == Rdna2Format::SOP1 && writer.opcode == 0x03u) ||
                (writer.fmt == Rdna2Format::SOP2 && scalar_write_width(writer) == 1u &&
                 writer.opcode != 0x0au && writer.opcode != 0x04u && writer.opcode != 0x05u) ||
                (writer.fmt == Rdna2Format::SOPK && writer.opcode == 0x00u);
            if (!scalar_writer) {
                scalar_prefix = false;
                break;
            }
            needed &= ~writes;
            for (uint32_t k = 0; k < writer.n_src; ++k) {
                const auto& source = writer.src[k];
                if (source.kind == OperandKind::SGPR ||
                    (source.kind == OperandKind::Special &&
                     (source.value == 106 || source.value == 107))) {
                    if (source.value < 0 || source.value > 107) scalar_prefix = false;
                    else needed.set(static_cast<size_t>(source.value));
                } else if (source.kind == OperandKind::VGPR ||
                           (source.kind == OperandKind::Special && source.value != 125)) {
                    scalar_prefix = false;
                }
            }
            if (!scalar_prefix) break;
        }
        // Ordinary entry SGPRs are resolved from this draw. VCC cannot be an uninitialized entry
        // value: its complete scalar definition must have been found above.
        if (scalar_prefix && !needed.test(106) && !needed.test(107)) {
            proven.push_back(load.pc);
            if (scalar_source_pcs)
                scalar_source_pcs->insert(scalar_source_pcs->end(), sources.begin(), sources.end());
        }
    }
    if (scalar_source_pcs) {
        std::sort(scalar_source_pcs->begin(), scalar_source_pcs->end());
        scalar_source_pcs->erase(std::unique(scalar_source_pcs->begin(), scalar_source_pcs->end()),
                                scalar_source_pcs->end());
    }
    return proven;
}

std::vector<uint32_t> rdna2_owned_raw_wide_data_loads(const std::vector<Rdna2Inst>& ins) {
    const auto strict = rdna2_proven_raw_immediate_wide_data_loads(ins);
    const auto read_points = proven_immediate_wide_data_loads(ins, true);
    std::vector<uint32_t> owned;
    std::set_difference(read_points.begin(), read_points.end(), strict.begin(), strict.end(),
                        std::back_inserter(owned));
    std::vector<uint32_t> scalar_sources;
    rdna2_proven_raw_register_wide_data_loads(ins, &scalar_sources);
    for (uint32_t pc : scalar_sources)
        if (std::binary_search(read_points.begin(), read_points.end(), pc))
            owned.push_back(pc);
    std::sort(owned.begin(), owned.end());
    owned.erase(std::unique(owned.begin(), owned.end()), owned.end());
    return owned;
}

std::vector<uint32_t> rdna2_raw_wave_wide_data_loads(const std::vector<Rdna2Inst>& ins) {
    if (std::none_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return in.fmt == Rdna2Format::VOP1 && in.opcode == 2u;
        }))
        return {};
    const auto numeric = rdna2_raw_wide_data_loads(ins);
    const auto immediate = rdna2_proven_raw_immediate_wide_data_loads(ins);
    const auto ordinary = rdna2_proven_raw_register_wide_data_loads(ins);
    const auto nested = rdna2_proven_raw_nested_wide_data_loads(ins);
    std::vector<uint32_t> required;
    for (uint32_t pc : numeric)
        if (!std::binary_search(immediate.begin(), immediate.end(), pc) &&
            !std::binary_search(ordinary.begin(), ordinary.end(), pc) &&
            !std::binary_search(nested.begin(), nested.end(), pc))
            required.push_back(pc);
    // MAY population, not an admission certificate: independently proved ordinary paths keep
    // their own owners; every remaining numeric load coexisting with a guest-wave read needs
    // logical-wave admission. Unknown instructions and certificate budgets never erase this.
    return required;
}

std::vector<RawWaveWideCertificate>
rdna2_raw_wave_wide_certificates(const std::vector<Rdna2Inst>& ins) {
    // Bounded abstract interpretation, not the emitter's copying-only lane-local taint. Unknown
    // reaching definitions, implicit SCC dependencies and unsigned wrap cannot acquire a bound.
    // Forward joins union BOTH definitions; a loop must reach a finite fixed point without a cap
    // changing guest execution. The budget below declines a proof, never truncates execution/data.
    if (ins.empty() || ins.size() > 512 || !ins.back().is_end) return {};
    struct Value {
        bool uniform = false;
        uint32_t lo = 0, hi = UINT32_MAX, alignment = 1;
        std::vector<uint32_t> events, definitions;
        bool event_defined = false;   // MUST, unlike the MAY dependency inventory
        bool operator==(const Value&) const = default;
    };
    struct State {
        std::array<Value, 106> scalar;
        Value scc;
        bool operator==(const State&) const = default;
    };
    const auto unite = [](std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
        a.insert(a.end(), b.begin(), b.end());
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    };
    const auto join = [&](Value a, const Value& b) {
        a.uniform &= b.uniform;
        a.lo = std::min(a.lo, b.lo);
        a.hi = std::max(a.hi, b.hi);
        a.alignment = std::min(a.alignment, b.alignment);
        unite(a.events, b.events);
        unite(a.definitions, b.definitions);
        a.event_defined &= b.event_defined;
        return a;
    };
    const auto constant = [](uint32_t bits) {
        return Value{true, bits, bits, bits ? (bits & (0u - bits)) : 0x80000000u, {}, {}};
    };
    std::unordered_map<uint32_t, size_t> by_pc;
    std::vector<std::vector<size_t>> edges(ins.size());
    std::vector<uint32_t> controls;
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (!in.len_dwords || in.fmt == Rdna2Format::Unknown || rdna2_may_write_guest_memory(in) ||
            in.fmt == Rdna2Format::DS || !by_pc.emplace(in.pc, i).second ||
            (i && ins[i - 1].pc + ins[i - 1].len_dwords != in.pc))
            return {};
        if (rdna2_may_write_unnamed_register_or_leave_cfg(in))
            return {};   // indirect control cannot be represented by the complete edge inventory
    }
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end) continue;
        bool fallthrough = true;
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = int64_t(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX || !by_pc.contains(uint32_t(target))) return {};
            edges[i].push_back(by_pc.at(uint32_t(target)));
            fallthrough = in.opcode != kSoppOpcodeBranch;
            controls.push_back(in.pc);
        } else if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) && in.opcode != 0x10u &&
                   in.opcode != 0x16u && in.opcode != 0x17u)
            return {};
        if (rdna2_instruction_may_change_exec(in)) controls.push_back(in.pc);
        if (fallthrough) {
            if (i + 1 == ins.size()) return {};
            edges[i].push_back(i + 1);
        }
    }
    std::sort(controls.begin(), controls.end());
    controls.erase(std::unique(controls.begin(), controls.end()), controls.end());
    std::vector<State> incoming(ins.size());
    std::vector<bool> reached(ins.size(), false);
    for (auto& value : incoming[0].scalar) value.uniform = true;   // explicit wave entry words
    incoming[0].scc = Value{true, 0, 1, 1, {}, {}};
    reached[0] = true;
    std::vector<size_t> pending{0};
    size_t transfers = 0;
    bool unknown_scc_control = false;
    while (!pending.empty()) {
        if (++transfers > ins.size() * 64u) return {};
        const size_t index = pending.back();
        pending.pop_back();
        const auto& in = ins[index];
        State next = incoming[index];
        const auto read = [&](const Operand& operand) -> Value {
            if (operand.kind == OperandKind::InlineInt) return constant(uint32_t(operand.value));
            if (operand.kind == OperandKind::Literal) return constant(in.literal);
            if (operand.kind == OperandKind::Special && operand.value == 125) return constant(0);
            if (operand.kind == OperandKind::Special && operand.value == 253) return next.scc;
            if (operand.kind == OperandKind::SGPR && operand.value >= 0 && operand.value < 106)
                return next.scalar[operand.value];
            return {};
        };
        Value result;
        bool modeled = false, writes_scc = false, preserves_scc = false;
        uint32_t arithmetic_carry_max = 1;
        if (in.fmt == Rdna2Format::VOP1 && in.opcode == 2u && in.src[0].kind == OperandKind::VGPR &&
            !in.has_sdwa && !in.has_dpp && !in.has_modifier && !in.src_abs[0] && !in.src_neg[0] &&
            !in.clamp && !in.omod) {
            result = Value{true, 0, UINT32_MAX, 1, {in.pc}, {in.pc}};
            result.event_defined = true;
            modeled = true;
            preserves_scc = true;
        } else if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32) {
            result = read(in.src[0]);
            modeled = true;
            preserves_scc = true;
        } else if (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeMovkI32) {
            result = constant(uint32_t(int32_t(in.simm16)));
            modeled = true;
            preserves_scc = true;
        } else if (in.fmt == Rdna2Format::SOP2 && scalar_write_width(in) == 1u) {
            const Value a = read(in.src[0]), c = read(in.src[1]);
            if (a.uniform && c.uniform) {
                result = join(a, c);
                result.event_defined = a.event_defined || c.event_defined;
                switch (in.opcode) {
                    case kSop2OpcodeAndB32: {
                        const Value* mask = c.lo == c.hi ? &c : a.lo == a.hi ? &a : nullptr;
                        if (!mask) break;
                        result.lo = 0;
                        result.hi = mask->lo;
                        result.alignment = mask->lo ? (mask->lo & (0u - mask->lo)) : 0x80000000u;
                        modeled = true;
                        writes_scc = true;
                        break;
                    }
                    case 0x1eu:
                    case 0x20u:
                        if (c.lo != c.hi) break;
                        if (in.opcode == 0x1eu) {
                            const uint32_t shift = c.lo & 31u;
                            if (uint64_t(a.hi) << shift > UINT32_MAX) break;
                            result.lo = a.lo << shift;
                            result.hi = a.hi << shift;
                            result.alignment = uint32_t(
                                std::min(uint64_t(a.alignment) << shift, uint64_t(0x80000000u)));
                        } else {
                            result.lo = a.lo >> (c.lo & 31u);
                            result.hi = a.hi >> (c.lo & 31u);
                            result.alignment = std::max(1u, a.alignment >> (c.lo & 31u));
                        }
                        modeled = true;
                        writes_scc = true;
                        break;
                    case 0x27u: {
                        if (c.lo != c.hi) break;
                        const uint32_t offset = c.lo & 31u, width = (c.lo >> 16u) & 0x7fu;
                        if (width > 32u || width > 32u - offset) break;
                        result.lo = 0;
                        result.hi = width == 32u ? UINT32_MAX : (1u << width) - 1u;
                        result.alignment = 1;
                        modeled = true;
                        writes_scc = true;
                        break;
                    }
                    case kSop2OpcodeAddU32:
                    case kSop2OpcodeAddcU32: {
                        const Value carry =
                            in.opcode == kSop2OpcodeAddcU32 ? next.scc : constant(0);
                        if (!carry.uniform) break;
                        if (uint64_t(a.hi) + c.hi + carry.hi > UINT32_MAX) {
                            // Wrapping arithmetic is still wave-uniform. Preserve that control
                            // fact, but it supplies no finite address interval without a later
                            // authentic mask/extract. Never authorize an unchecked wrapped span.
                            result.lo = 0;
                            result.hi = UINT32_MAX;
                            result.alignment = 1;
                        } else {
                            result.lo = a.lo + c.lo + carry.lo;
                            result.hi = a.hi + c.hi + carry.hi;
                            arithmetic_carry_max = 0;
                        }
                        result.alignment = std::min(result.alignment, carry.alignment);
                        unite(result.events, carry.events);
                        unite(result.definitions, carry.definitions);
                        result.event_defined |= carry.event_defined;
                        modeled = true;
                        writes_scc = true;
                        break;
                    }
                    case kSop2OpcodeCselectB32:
                        if (!next.scc.uniform) break;
                        unite(result.events, next.scc.events);
                        unite(result.definitions, next.scc.definitions);
                        result.event_defined =
                            next.scc.event_defined || (a.event_defined && c.event_defined);
                        modeled = true;
                        preserves_scc = true;
                        break;
                    default: break;
                }
            }
        }
        if (modeled) unite(result.definitions, {in.pc});
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                if (base + int(word) >= 0 && base + int(word) < 106)
                    next.scalar[base + int(word)] = modeled && width == 1u ? result : Value{};
        });
        // Close the implicit SCC dependency independently of ordinary SGPR writes. Integer CMP
        // and the admitted arithmetic have wave-uniform operands; unmodeled definitions poison it.
        if (in.fmt == Rdna2Format::SOPC) {
            const Value a = read(in.src[0]), c = read(in.src[1]);
            next.scc = join(a, c);
            next.scc.uniform = a.uniform && c.uniform && in.opcode <= 0x0bu;
            next.scc.event_defined = a.event_defined || c.event_defined;
            next.scc.lo = 0;
            next.scc.hi = 1;
            next.scc.alignment = 1;
            unite(next.scc.definitions, {in.pc});
        } else if (writes_scc) {
            next.scc = result;
            // Checked ADD/ADDC cannot carry; a widened wrapping result retains both carry cases.
            next.scc.lo = 0;
            next.scc.hi = (in.opcode == kSop2OpcodeAddU32 || in.opcode == kSop2OpcodeAddcU32)
                              ? arithmetic_carry_max
                              : 1;
            next.scc.alignment = 1;
        } else if (!preserves_scc && (in.fmt == Rdna2Format::SOP2 || in.fmt == Rdna2Format::SOPK ||
                                      (in.fmt == Rdna2Format::SOP1 &&
                                       !sop1_opcode_leaves_scc_unmodified(in.opcode)))) {
            next.scc = {};
        }
        if (in.fmt == Rdna2Format::SOPP &&
            (in.opcode == kSoppOpcodeCbranchScc0 || in.opcode == kSoppOpcodeCbranchScc0 + 1u) &&
            !next.scc.uniform)
            unknown_scc_control = true;
        for (size_t target : edges[index]) {
            State merged = next;
            if (reached[target]) {
                for (size_t reg = 0; reg < merged.scalar.size(); ++reg)
                    merged.scalar[reg] = join(incoming[target].scalar[reg], next.scalar[reg]);
                merged.scc = join(incoming[target].scc, next.scc);
                if (target <= index)
                    for (size_t reg = 0; reg < merged.scalar.size(); ++reg) {
                        auto& value = merged.scalar[reg];
                        const auto& previous = incoming[target].scalar[reg];
                        // Finite conservative widening closes unrelated loop counters. MAY event/
                        // definition closure and MUST event presence remain independently joined.
                        // A selector must still regain a complete aligned finite interval at its load.
                        if (value.lo < previous.lo) {
                            value.lo = 0;
                            value.alignment = 1;
                        }
                        if (value.hi > previous.hi) {
                            value.hi = UINT32_MAX;
                            value.alignment = 1;
                        }
                    }
            }
            if (!reached[target] || merged != incoming[target]) {
                incoming[target] = std::move(merged);
                reached[target] = true;
                pending.push_back(target);
            }
        }
    }
    if (unknown_scc_control) return {};
    const auto numeric = rdna2_raw_wide_data_loads(ins);
    std::vector<RawWaveWideCertificate> certificates;
    for (size_t index = 0; index < ins.size(); ++index) {
        const auto& load = ins[index];
        const uint32_t bytes = load.opcode == 2u ? 16u : 32u;
        if (!reached[index] || load.fmt != Rdna2Format::SMEM ||
            (load.opcode != 2u && load.opcode != 3u) || load.dst.kind != OperandKind::SGPR ||
            load.dst.value < 0 || load.dst.value + int(bytes / 4u) > 106 ||
            load.src[0].kind != OperandKind::SGPR || load.src[0].value < 0 ||
            load.src[0].value > 104 || load.src[1].kind != OperandKind::SGPR ||
            load.src[1].value < 0 || load.src[1].value > 105 ||
            !std::binary_search(numeric.begin(), numeric.end(), load.pc))
            continue;
        bool stable_base = true;
        for (const auto& in : ins)
            for_each_scalar_write(in, [&](int reg, uint32_t width) {
                if (reg < load.src[0].value + 2 && load.src[0].value < reg + int(width))
                    stable_base = false;
            });
        const Value& offset = incoming[index].scalar[load.src[1].value];
        const int32_t immediate = int32_t(load.literal);
        if (!stable_base || !offset.uniform || !offset.event_defined || offset.events.empty() ||
            offset.hi == UINT32_MAX || offset.alignment < 4u || (load.literal & 3u) ||
            int64_t(offset.lo) + immediate < 0)
            continue;
        certificates.push_back({load.pc, uint32_t(load.src[0].value), uint32_t(load.src[1].value),
                                bytes, immediate, offset.lo, offset.hi, offset.alignment,
                                offset.events, offset.definitions, controls});
    }
    return certificates;
}

std::vector<uint32_t> rdna2_proven_raw_nested_wide_data_loads(
        const std::vector<Rdna2Inst>& ins) {
    std::vector<uint32_t> proven;
    if (ins.empty()) return proven;
    const auto parents = rdna2_proven_raw_immediate_wide_data_loads(ins);
    const auto needs_backing = rdna2_raw_wide_data_loads(ins);
    if (parents.empty() || needs_backing.empty()) return proven;
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i) {
        if (ins[i].fmt == Rdna2Format::Unknown || !ins[i].len_dwords ||
            !by_pc.emplace(ins[i].pc, i).second) return {};
    }
    for (const Rdna2Inst& in : ins) {
        if (in.fmt != Rdna2Format::SOPP || in.is_end) continue;
        if (sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = static_cast<int64_t>(in.pc) +
                in.len_dwords + in.simm16;
            if (target <= static_cast<int64_t>(in.pc) || target > UINT32_MAX ||
                !by_pc.contains(static_cast<uint32_t>(target))) return {};
        } else if (!sopp_is_noop(in) && in.opcode != kSoppOpcodeBarrier) {
            return {};
        }
    }
    for (size_t child_index = 0; child_index < ins.size(); ++child_index) {
        const Rdna2Inst& child = ins[child_index];
        const int child_width = child.opcode == 0x2u ? 4 : 8;
        if (child.fmt != Rdna2Format::SMEM ||
            (child.opcode != 0x2u && child.opcode != 0x3u) ||
            child.dst.kind != OperandKind::SGPR ||
            // Numeric snapshots write ordinary SGPRs, not the emitter's typed VCC mask.
            child.dst.value < 0 || child.dst.value > 106 - child_width ||
            child.src[0].kind != OperandKind::SGPR ||
            child.src[0].value < 0 || child.src[0].value + 1 >= 106 ||
            child.src[1].kind != OperandKind::Special || child.src[1].value != 125 ||
            static_cast<int32_t>(child.literal) < 0 || (child.literal & 3u) ||
            !std::binary_search(needs_backing.begin(), needs_backing.end(), child.pc))
            continue;
        size_t parent_index = ins.size();
        for (size_t i = 0; i < child_index; ++i) {
            const Rdna2Inst& candidate = ins[i];
            if (candidate.fmt == Rdna2Format::SMEM &&
                (candidate.opcode == 0x2u || candidate.opcode == 0x3u) &&
                candidate.dst.kind == OperandKind::SGPR &&
                candidate.dst.value == child.src[0].value &&
                std::binary_search(parents.begin(), parents.end(), candidate.pc))
                parent_index = i;
        }
        if (parent_index == ins.size()) continue;
        bool stable = true;
        const RawWideLifetime child_lifetime(ins, by_pc, child_index,
                                             child.opcode == 0x2u ? 4u : 8u);
        for (size_t i = 0; i < child_index && stable; ++i) {
            const Rdna2Inst& preceding = ins[i];
            if (i > parent_index)
                for_each_scalar_write(preceding, [&](int base, uint32_t width) {
                    if (base >= 0 && base < child.src[0].value + 2 &&
                        base + static_cast<int>(width) > child.src[0].value)
                        stable = false;
                });
            if (i < parent_index && preceding.fmt == Rdna2Format::SOPP &&
                sopp_opcode_is_direct_branch(preceding.opcode)) {
                const int64_t target = static_cast<int64_t>(preceding.pc) +
                    preceding.len_dwords + preceding.simm16;
                if (target > static_cast<int64_t>(ins[parent_index].pc) &&
                    target <= static_cast<int64_t>(child.pc)) stable = false;
            }
            // A branch around the child can join a numeric reader with no child definition.
            // The parent-dominance check above does not cover this second edge.
            if (preceding.fmt == Rdna2Format::SOPP &&
                sopp_opcode_is_direct_branch(preceding.opcode)) {
                const int64_t target = static_cast<int64_t>(preceding.pc) +
                    preceding.len_dwords + preceding.simm16;
                if (target > static_cast<int64_t>(child.pc) &&
                    child_lifetime.has_reader_after_bypassed_load(
                        by_pc.at(static_cast<uint32_t>(target)))) stable = false;
            }
        }
        if (stable) proven.push_back(child.pc);
    }
    return proven;
}

std::vector<uint32_t> rdna2_raw_nested_numeric_loads(const std::vector<Rdna2Inst>& ins) {
    const auto numeric = rdna2_raw_wide_data_loads(ins);
    std::vector<uint32_t> result;
    for (size_t index = 0; index < ins.size(); ++index) {
        const auto& load = ins[index];
        if (load.fmt != Rdna2Format::SMEM || load.src[0].kind != OperandKind::SGPR ||
            load.src[1].kind != OperandKind::Special || load.src[1].value != 125 ||
            !std::binary_search(numeric.begin(), numeric.end(), load.pc)) continue;
        bool written = false;
        for (size_t before = 0; before < index; ++before)
            for_each_scalar_write(ins[before], [&](int base, uint32_t width) {
                written |= base < load.src[0].value + 2 &&
                           load.src[0].value < base + static_cast<int>(width);
            });
        if (written) result.push_back(load.pc);
    }
    return result;
}

std::vector<RawNestedWideChain> rdna2_owned_nested_wide_chains(
        const std::vector<Rdna2Inst>& ins) {
    if (std::any_of(ins.begin(), ins.end(), rdna2_may_write_guest_memory)) return {};
    const auto children = rdna2_proven_raw_nested_wide_data_loads(ins);
    const auto parents = rdna2_proven_raw_immediate_wide_data_loads(ins);
    std::vector<RawNestedWideChain> chains;
    for (uint32_t pc : children) {
        const auto child = std::find_if(ins.begin(), ins.end(),
            [pc](const Rdna2Inst& in) { return in.pc == pc; });
        if (child == ins.end()) return {};
        const Rdna2Inst* parent = nullptr;
        for (auto it = ins.begin(); it != child; ++it)
            if (it->fmt == Rdna2Format::SMEM &&
                (it->opcode == 0x2u || it->opcode == 0x3u) &&
                it->dst.kind == OperandKind::SGPR &&
                it->dst.value == child->src[0].value &&
                std::binary_search(parents.begin(), parents.end(), it->pc)) parent = &*it;
        if (!parent) return {};
        // One bounded hop only. The effective snapshots are short; offsets cannot authenticate
        // a giant allocation, negative addressing, or a later pointer chain.
        const uint32_t parent_bytes = parent->opcode == 0x2u ? 16u : 32u;
        const uint32_t child_bytes = child->opcode == 0x2u ? 16u : 32u;
        if (uint64_t(parent->literal) + parent_bytes > 0x100000u ||
            uint64_t(child->literal) + child_bytes > 0x100000u) continue;
        chains.push_back({parent->pc, child->pc, parent_bytes, child_bytes,
                          parent->literal, child->literal});
    }
    return chains;
}

} // namespace prosper::gpu
