#include "gpu/recompiler/rdna2_saved_mask_words.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include <bit>

namespace prosper::gpu {
namespace {
bool stops_entry_participation(const Rdna2Inst& in) {
    // Keep this an allowlist for scalar control. Unknown escapes, calls, indirect PC writes,
    // kill/early-end, and every branch stop the proof, even when a structurizer linearizes them.
    if (in.is_end || in.fmt == Rdna2Format::Unknown || rdna2_escapes_decoded_effects(in))
        return true;
    if (in.fmt == Rdna2Format::SOPP)
        return in.opcode != 0x00 && in.opcode != 0x0a && in.opcode != 0x0c && in.opcode != 0x20 &&
               in.opcode != 0x21 && in.opcode != 0x22;
    if (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeSetregB32) return true;
    return in.fmt == Rdna2Format::EXP;
}

bool definite_word(const RegState& rs, int word) {
    return rs.sreg.contains(word) && !sreg_word_may_be_fabricated(rs, word) &&
           !rs.unavailable_saved_mask_words.contains(word);
}

bool same_original_instruction(const Rdna2Inst& in, const Rdna2Inst& original) {
    return !in.synthetic_terminator && in == original;
}

bool has_prior_partial_control(const SpirvCompute& b) {
    for (size_t n = 0; n < b.code.size();) {
        const uint32_t width = b.code[n] >> 16;
        if (!width || width > b.code.size() - n) return true;
        const uint32_t op = b.code[n] & 0xffffu;
        if (op == Op_BranchConditional || op == Op_Switch || op == Op_LoopMerge || op == Op_Kill)
            return true;
        n += width;
    }
    return false;
}

bool numeric_pair_source(const Rdna2Inst& in) {
    // MOV/NOT/WQM/save-EXEC and mask reductions can consume a Bool without reading two DATA
    // dwords. Numeric CSELECT, arithmetic, compares and vector operands cannot assume that domain.
    if (in.fmt == Rdna2Format::SOP1)
        return in.opcode != kSop1OpcodeMovB64 && in.opcode != 0x08 && in.opcode != 0x0a &&
               !(in.opcode >= kSop1OpcodeAndSaveexecB64 &&
                 in.opcode <= kSop1OpcodeXnorSaveexecB64) &&
               in.opcode != kSop1OpcodeAndn1SaveexecB64 &&
               in.opcode != kSop1OpcodeOrn1SaveexecB64 && in.opcode != kSop1OpcodeBcnt1I32B64 &&
               in.opcode != kSop1OpcodeFf1I32B64 && in.opcode != kSop1OpcodeFlbitI32B64 &&
               in.opcode != kSop1OpcodeQuadmaskB64;
    return in.fmt != Rdna2Format::SOP2 || in.opcode < 0x0f || in.opcode > 0x1d ||
           (in.opcode & 1u) == 0;
}

std::set<uint32_t> reconverged_entry_pcs(const std::vector<Rdna2Inst>& ins) {
    if (ins.empty()) return {};
    const size_t exit = ins.size();
    std::map<uint32_t, size_t> at_pc;
    for (size_t i = 0; i < exit; ++i) at_pc.emplace(ins[i].pc, i);
    uint32_t first_reentry = UINT32_MAX;
    for (const auto& in : ins) {
        if (in.fmt == Rdna2Format::Unknown || rdna2_escapes_decoded_effects(in)) return {};
        if (in.fmt != Rdna2Format::SOPP || in.is_end) continue;
        const bool branch = in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09);
        if (!branch) {
            if (stops_entry_participation(in)) return {};
            continue;
        }
        const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
        if (target < 0 || target > UINT32_MAX || !at_pc.contains(static_cast<uint32_t>(target)))
            return {};
        if (target <= in.pc) first_reentry = std::min(first_reentry, static_cast<uint32_t>(target));
    }
    // Forward edges form a DAG. The dominators of the synthetic exit are precisely the sites
    // every entry-to-exit path visits. Unsafe effects and backwards edges lead to that exit;
    // an arm which ends, kills, escapes, or bypasses a producer therefore cannot grant it proof.
    std::vector<std::vector<size_t>> predecessors(exit + 1);
    std::vector<size_t> parent(exit + 1, SIZE_MAX), depth(exit + 1);
    const size_t levels = std::bit_width(exit);
    std::vector<std::vector<size_t>> ancestor(levels, std::vector<size_t>(exit + 1));
    parent[0] = 0;
    const auto common_ancestor = [&](size_t a, size_t c) {
        if (depth[a] < depth[c]) std::swap(a, c);
        const size_t difference = depth[a] - depth[c];
        for (size_t level = 0; level < levels; ++level)
            if ((difference >> level) & 1u) a = ancestor[level][a];
        if (a == c) return a;
        for (size_t level = levels; level-- > 0;)
            if (ancestor[level][a] != ancestor[level][c]) {
                a = ancestor[level][a];
                c = ancestor[level][c];
            }
        return parent[a];
    };
    for (size_t i = 0; i <= exit; ++i) {
        if (i) {
            for (size_t incoming : predecessors[i])
                parent[i] = parent[i] == SIZE_MAX ? incoming : common_ancestor(parent[i], incoming);
            if (parent[i] == SIZE_MAX) continue;
            depth[i] = depth[parent[i]] + 1;
            ancestor[0][i] = parent[i];
            for (size_t level = 1; level < levels; ++level)
                ancestor[level][i] = ancestor[level - 1][ancestor[level - 1][i]];
        }
        if (i == exit) break;
        const auto& in = ins[i];
        const bool branch = in.fmt == Rdna2Format::SOPP &&
                            (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09));
        if (branch) {
            const uint32_t target =
                static_cast<uint32_t>(static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16);
            predecessors[target > in.pc ? at_pc.at(target) : exit].push_back(i);
            if (in.opcode != 0x02) predecessors[i + 1].push_back(i);
        } else
            predecessors[stops_entry_participation(in) ? exit : i + 1].push_back(i);
    }
    std::set<uint32_t> result;
    if (parent[exit] == SIZE_MAX) return result;
    for (size_t i = parent[exit];; i = parent[i]) {
        if (ins[i].pc < first_reentry) result.insert(ins[i].pc);
        if (!i) break;
    }
    return result;
}
}   // namespace

bool uses_native_saved_mask_words(const SpirvCompute& b) {
    return b.wave_size == 64 && (b.is_fragment || (b.is_compute && b.native_subgroup_size == 64 &&
                                                   !b.is_fragment_packet()));
}

bool saved_mask_producer_has_full_participation(const SpirvCompute& b, const Rdna2Inst& in) {
    if (!uses_native_saved_mask_words(b) ||
        !(b.saved_mask_entry_prefix_pcs.contains(in.pc) ||
          (!b.saved_mask_in_dispatcher && b.saved_mask_reconverged_entry_pcs.contains(in.pc))) ||
        in.pc >= b.saved_mask_original_words.size())
        return false;
    auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                     b.saved_mask_original_words.size() - in.pc);
    original.pc = in.pc;
    return same_original_instruction(in, original);
}

bool validate_saved_mask_copy_words(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in) {
    if (!uses_native_saved_mask_words(b) || in.dst.value <= 104 ||
        !b.saved_mask_word_roots.contains(in.dst.value) || in.src[0].kind != OperandKind::SGPR ||
        !rs.sreg_bool.contains(in.src[0].value))
        return true;
    if (definite_word(rs, in.src[0].value) && definite_word(rs, in.src[0].value + 1)) return true;
    // CONFIDENCE: HIGH -- this exact copy reuses the current B64 predicate, with no DATA
    // consumer of its output generation. Earlier reads and later definite replacements cannot
    // demand this copy's physical words. No ballot or numeric authority follows from the proof.
    if (!rs.sreg_bool_b32.contains(in.src[0].value) && native_vcc_copy_is_bool_only(b, in))
        return true;
    b.stage_reject_pc = in.pc;
    b.stage_reject_reason = "saved-mask-copy-words-unavailable";
    return false;
}

bool saved_mask_write_retains_words(const SpirvCompute& b, const Rdna2Inst& in, int root) {
    return root >= 0 && root <= 104 && saved_mask_producer_has_full_participation(b, in) &&
           b.saved_mask_word_roots.contains(root) &&
           !(in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
             in.src[0].kind == OperandKind::SGPR);
}

bool saved_mask_mov_defines_words(const SpirvCompute& b, const Rdna2Inst& in, bool scalar_result,
                                  bool source_is_mask) {
    return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
           (in.src[0].kind == OperandKind::InlineInt || scalar_result ||
            (b.is_compute && b.wave_size == 64 && b.native_subgroup_size == 64 && source_is_mask &&
             in.src[0].kind != OperandKind::SGPR &&
             (in.dst.value > 104 || saved_mask_producer_has_full_participation(b, in))));
}

bool saved_mask_logical_defines_words(const SpirvCompute& b, const Rdna2Inst& in,
                                      bool scalar_sources) {
    if (!is_native_b64_logical(b, in)) return false;
    // Do not let a wider generic scalar inventory authenticate unsupported operand spellings.
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const auto& source = in.src[k];
        const bool supported =
            source.kind == OperandKind::InlineInt ||
            (source.kind == OperandKind::Literal && in.has_literal) ||
            (source.kind == OperandKind::SGPR && source.value >= 0 && source.value <= 104) ||
            (source.kind == OperandKind::Special && (source.value == 106 || source.value == 107 ||
                                                     source.value == 126 || source.value == 127));
        if (!supported) return false;
    }
    return native_logical_data_demanded(b, in) &&
           (scalar_sources || native_logical_producer_has_full_participation(b, in));
}

void retire_saved_mask_facts(const SpirvCompute& b, std::set<int>& masks, std::set<int>& ambiguous,
                             int base, uint32_t width) {
    for (auto it = masks.begin(); it != masks.end();) {
        const int mask_base = *it;
        if (base < mask_base + 2 && mask_base < base + static_cast<int>(width)) {
            if (uses_native_saved_mask_words(b) && mask_base <= 104 &&
                !(base <= mask_base && base + static_cast<int>(width) >= mask_base + 2))
                ambiguous.insert(mask_base);
            it = masks.erase(it);
        } else
            ++it;
    }
    // A one-word scalar write kills the definite-mask fact for the physical pair, but
    // it resolves only the addressed half of an ambiguous pair. Keep the ambiguity
    // until one definite write covers both halves; otherwise the untouched word could
    // be reloaded from the wrong Function-variable domain. The separate scalar-word
    // MUST facts below can validate exactly the overwritten word while complete-pair
    // consumers remain fail-closed until both halves are definitely scalar.
    for (auto it = ambiguous.begin(); it != ambiguous.end();) {
        const int mask_base = *it;
        if (base <= mask_base && base + static_cast<int>(width) >= mask_base + 2)
            it = ambiguous.erase(it);
        else
            ++it;
    }
}

void load_missing_saved_mask_words(RegState& rs, const std::set<int>& roots,
                                   const std::set<int>& defined_words) {
    if (!rs.native_saved_mask_words) return;
    for (int root : roots)
        if (root >= 0 && root <= 104)
            for (int word : {root, root + 1})
                if (!defined_words.contains(word)) rs.unavailable_saved_mask_words.insert(word);
}

void initialize_saved_mask_word_proof(SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                      const uint32_t* code, size_t dwords, uint32_t initial_active,
                                      const RegState* initial) {
    if (b.saved_mask_word_analysis_done) {
        // A phase refers to the original bytes. Neither a reused pointer/PC nor a changed literal,
        // truncated stream or altered source can inherit another program's participation proof.
        bool same_stream = code && dwords == b.saved_mask_original_words.size() &&
                           std::equal(b.saved_mask_original_words.begin(),
                                      b.saved_mask_original_words.end(), code);
        if (same_stream) {
            std::vector<Rdna2Inst> original;
            rdna2_walk(b.saved_mask_original_words.data(), b.saved_mask_original_words.size(),
                       original);
            same_stream = ins.size() == original.size();
            for (size_t i = 0; same_stream && i < ins.size(); ++i)
                same_stream = same_original_instruction(ins[i], original[i]);
        }
        if (!same_stream || !native_vcc_copy_entry_masks_match(b, initial)) {
            b.native_bool_only_vcc_copy_pcs.clear();
            b.fragment_bool_only_vcc_copy_pcs.clear();
        }
        if (!same_stream || initial_active || b.invocation_guard_merge ||
            has_prior_partial_control(b)) {
            b.saved_mask_entry_prefix_pcs.clear();
            b.saved_mask_reconverged_entry_pcs.clear();
            b.native_logical_uniform_entry_pcs.clear();
        }
        return;
    }
    b.saved_mask_word_analysis_done = true;
    if (!uses_native_saved_mask_words(b) || !code || !dwords) return;
    b.saved_mask_original_words.assign(code, code + dwords);
    std::vector<Rdna2Inst> original;
    rdna2_walk(code, dwords, original);
    initialize_native_logical_data_demand(b, original, initial);
    // The incoming state must be the original entry, not the state of a recursive slice which
    // happens to contain the same MOV PC. A later reinitialization may invalidate, never reseed.
    if (ins.size() != original.size() ||
        !std::equal(ins.begin(), ins.end(), original.begin(), same_original_instruction)) {
        b.native_bool_only_vcc_copy_pcs.clear();
        b.fragment_bool_only_vcc_copy_pcs.clear();
    }
    // Physical reads and partial replacements require retained words. Mask-only saves keep their
    // existing subgroup-width contract; a shader which never observes DATA needs no new ballot.
    std::set<int> demanded;
    const auto demand_word = [&](int word) {
        if (word >= 0 && word <= 107) {
            demanded.insert(word);
            if (word) demanded.insert(word - 1);
        }
    };
    for (const auto& in : original) {
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            if (width != 1) return;
            const uint32_t next = in.pc + in.len_dwords;
            if (base >= 0 && base <= 104 && !sgpr_dead_at_merge(original, next, base + 1))
                demanded.insert(base);
            if (base > 0 && base <= 105 && !sgpr_dead_at_merge(original, next, base - 1))
                demanded.insert(base - 1);
        });
        for (uint32_t k = 0; k < in.n_src; ++k)
            if ((in.src[k].kind == OperandKind::SGPR ||
                 (in.src[k].kind == OperandKind::Special &&
                  (in.src[k].value == 106 || in.src[k].value == 107))) &&
                (scalar_alu_source_words(in, k) == 1 ||
                 (scalar_alu_source_words(in, k) == 2 && numeric_pair_source(in)))) {
                demand_word(in.src[k].value);
                if (scalar_alu_source_words(in, k) == 2) demand_word(in.src[k].value + 1);
            }
    }
    // Track logical destinations separately: a dispatcher Bool reload with no current scalar
    // MUST word must not trigger the Special-VCC late-ballot fallback at a later DATA read.
    for (const auto& in : original)
        if (is_native_b64_logical(b, in)) b.native_logical_word_roots.insert(in.dst.value);
    // Copies and logical DATA consumers need their original physical source words too.
    for (bool changed = true; changed;) {
        changed = false;
        for (const auto& in : original) {
            const bool copy = in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64;
            if ((!copy && !is_native_b64_logical(b, in)) || !demanded.contains(in.dst.value))
                continue;
            for (uint32_t k = 0; k < in.n_src; ++k)
                if (in.src[k].kind == OperandKind::SGPR ||
                    (in.src[k].kind == OperandKind::Special &&
                     (in.src[k].value == 106 || in.src[k].value == 107)))
                    changed |= demanded.insert(in.src[k].value).second;
        }
    }
    b.saved_mask_word_roots = std::move(demanded);
    if (initial_active || b.invocation_guard_merge || has_prior_partial_control(b)) return;
    // Never manufacture entry participation from a sliced phase or a rewritten control stream.
    if (ins.size() != original.size()) return;
    for (size_t i = 0; i < ins.size(); ++i)
        if (!same_original_instruction(ins[i], original[i])) return;
    initialize_native_logical_uniform_entry_proof(b, original);
    b.saved_mask_reconverged_entry_pcs = reconverged_entry_pcs(original);
    if (b.saved_mask_reconverged_entry_pcs.empty()) return;
    // CONFIDENCE: HIGH -- a ballot observes participating invocations (Vulkan subgroup contract).
    // A structured producer must postdominate entry through forward reconverging control, with no
    // earlier exit or reentry. The dispatcher retains only the branch-free entry prefix: guest
    // reconvergence alone does not prove the same host dispatcher iteration/dynamic ballot instance.
    // allow_wave/wave_ok and an exact subgroup width are not this participation proof.
    uint32_t prefix_end = UINT32_MAX;
    for (const auto& in : original)
        if (stops_entry_participation(in)) {
            prefix_end = in.pc;
            break;
        }
    // A backwards edge can revisit a lexical prefix after lanes have left. Such sites are not
    // entry-only producers. Indirect PC/control escapes cannot prove where execution reenters.
    for (const auto& in : original) {
        if (in.fmt == Rdna2Format::SOPP &&
            (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09))) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            if (target >= 0 && target < prefix_end) prefix_end = static_cast<uint32_t>(target);
        }
        if ((in.fmt == Rdna2Format::SOP1 && in.opcode >= kSop1OpcodeSetpcB64 &&
             in.opcode <= kSop1OpcodeRfeB64) ||
            (in.fmt == Rdna2Format::SOPK &&
             (in.opcode == kSopkOpcodeCallB64 || in.opcode == kSopkOpcodeSubvectorLoopBegin ||
              in.opcode == kSopkOpcodeSubvectorLoopEnd)))
            prefix_end = 0;
    }
    for (const auto& in : original)
        if (in.pc < prefix_end) b.saved_mask_entry_prefix_pcs.insert(in.pc);
}

bool saved_mask_spill_uses_data(const RegState& rs, int word) {
    return rs.native_saved_mask_words && definite_word(rs, word);
}

void prepare_saved_mask_word_write(const SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    rs.native_saved_mask_words = uses_native_saved_mask_words(b);
    if (!uses_native_saved_mask_words(b)) return;
    std::set<int> written;
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        for (uint32_t w = 0; w < width; ++w) written.insert(base + static_cast<int>(w));
    });
    for (const auto& [root, mask] : rs.sreg_bool) {
        if (root < 0 || root > 104 || rs.sreg_bool_b32.contains(root)) continue;
        for (int word : {root, root + 1})
            if (!written.contains(word) && written.contains(word == root ? root + 1 : root) &&
                !definite_word(rs, word))
                rs.unavailable_saved_mask_words.insert(word);
    }
}

void retain_saved_mask_words(SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    if (!uses_native_saved_mask_words(b)) return;
    if (is_native_b64_logical(b, in)) {
        record_native_logical_word_gaps(b, rs, in);
        return;   // its pre-write source/proof lowering already decided whether DATA exists
    }
    for_each_scalar_write(in, [&](int root, uint32_t width) {
        if (width != 2 || root < 0 || root > 104 || rs.sreg_bool_b32.contains(root) ||
            !rs.sreg_bool.contains(root) || !scalar_write_is_b64_mask(in, root) ||
            !b.saved_mask_word_roots.contains(root))
            return;
        if (definite_word(rs, root) && definite_word(rs, root + 1)) return;
        // A saved SGPR Bool may describe lanes which no longer execute this copy. Only its
        // already-retained words are a copy source; even a prefix copy must not reinvent them.
        const bool older_copy = in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                                in.src[0].kind == OperandKind::SGPR;
        if (older_copy || rs.sreg_merge_placeholder.contains(root) ||
            rs.sreg_merge_placeholder.contains(root + 1) ||
            !saved_mask_producer_has_full_participation(b, in)) {
            for (int word : {root, root + 1})
                if (!definite_word(rs, word)) rs.unavailable_saved_mask_words.insert(word);
            return;
        }
        const uint32_t mask = rs.sreg_bool.at(root);
        for (uint32_t half = 0; half < 2; ++half) {
            const int word = root + static_cast<int>(half);
            rs.sreg[word] = b.is_fragment ? b.fragment_wave_ballot_half(mask, half)
                                          : b.native_wave_ballot_half(mask, half);
            rs.sreg_merge_placeholder.erase(word);
            rs.sreg_memory_pattern.erase(word);
            rs.unavailable_saved_mask_words.erase(word);
        }
    });
}

uint32_t refuse_unavailable_saved_mask_word(SpirvCompute& b, const Rdna2Inst& in, bool* ok) {
    if (ok) *ok = false;
    b.stage_reject_pc = in.pc;
    b.stage_reject_reason = "saved-mask-physical-word-unavailable";
    return b.uconst(0);
}
}   // namespace prosper::gpu
