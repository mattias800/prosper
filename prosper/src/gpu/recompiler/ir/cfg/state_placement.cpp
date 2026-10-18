// state_placement.cpp -- the phi sets of every structured join; see state_placement.hpp and the
// "State placement" section of AGENTS.md (the inventory rules and where they differ).
#include "gpu/recompiler/ir/cfg/state_placement.hpp"

#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <algorithm>
#include <span>
#include <string>

namespace prosper::gpu::ir::cfg {
namespace {

using Kind = Region::Kind;

void add_words(std::set<int>& out, int base, uint32_t words) {
    for (uint32_t word = 0; word < words; ++word) out.insert(base + static_cast<int>(word));
}

// VOP1, VOP2, VOP3, VOP3P. The lane reads write their SGPR; v_movreld (the one VOP1 row whose
// destination alone is M0-relative) may write any VGPR from VDST up. Otherwise VOP1, VOP2 and
// VOP3P charge VDST alone and VOP3 charges the table's VDST width (rdna2_vgpr_write_count).
void add_valu_writes(const Rdna2Inst& in, const Rdna2InstructionEffects& effects, int max_vgpr,
                     PhiSets& out) {
    if (rdna2_lane_read_writes_sgpr(in)) {
        out.sgprs.insert(in.dst.value);
        return;
    }
    if (in.fmt == Rdna2Format::VOP3) {
        add_words(out.vgprs, in.dst.value, rdna2_vgpr_write_count(in));
        return;
    }
    if (in.fmt == Rdna2Format::VOP1 && effects.m0_relative == Rdna2M0Relative::Destination) {
        for (int reg = in.dst.value; reg <= max_vgpr; ++reg) out.vgprs.insert(reg);
        return;
    }
    out.vgprs.insert(in.dst.value);
}

void add_sop2_writes(const Rdna2Inst& in, PhiSets& out) {
    add_words(out.sgprs, in.dst.value, sop2_phi_words(in));
}

void add_memory_writes(const Rdna2Inst& in, bool status_word, PhiSets& out) {
    add_words(out.vgprs, in.dst.value, rdna2_vgpr_write_count(in));
    if (!status_word) return;
    if (const int status = rdna2_tfe_status_vgpr(in); status >= 0) out.vgprs.insert(status);
}

bool constant_lane_writelane(const Rdna2Inst& in) {
    return rdna2_instruction_effects(in).lane_flow == Rdna2LaneFlow::WriteLane &&
           in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 &&
           in.src[1].value <= 63;
}

class Placer {
public:
    Placer(const GuestCfg& cfg, const RegionTree& tree, const std::vector<Rdna2Inst>& ins)
        : cfg_(cfg), tree_(tree), ins_(ins), max_vgpr_(shader_max_vgpr(ins)) {
        // Lane slots stop at the first s_endpgm, as LaneSlotLoopCarry's scan always did.
        const auto end =
            std::find_if(ins.begin(), ins.end(), [](const auto& in) { return in.is_end; });
        first_end_ = static_cast<uint32_t>(end - ins.begin());
    }
    StatePlacement run();

private:
    // Instructions of blocks [first, end), as indices into `ins`.
    std::pair<uint32_t, uint32_t> span(uint32_t first, uint32_t end) const {
        if (first >= end || end > cfg_.blocks.size()) return {0, 0};
        return {cfg_.blocks[first].first, cfg_.blocks[end - 1].last + 1};
    }
    void add_range(uint32_t first, uint32_t end, PhiSets& out) const {
        const auto [lo, hi] = span(first, end);
        for (uint32_t i = lo; i < hi && i < ins_.size(); ++i)
            add_phi_writes(ins_[i], max_vgpr_, out);
    }
    void add_arm(uint32_t region, PhiSets& out) const {
        if (region == kNone || region >= tree_.regions.size()) return;
        add_range(tree_.regions[region].first_block, tree_.regions[region].end_block, out);
    }
    void place_if(const Region& r, StatePlacement& out) const;
    void place_loop(const Region& r, StatePlacement& out) const;

    const GuestCfg& cfg_;
    const RegionTree& tree_;
    const std::vector<Rdna2Inst>& ins_;
    int max_vgpr_ = 0;
    uint32_t first_end_ = 0;
};

void Placer::place_if(const Region& r, StatePlacement& out) const {
    IfPlacement placed;
    placed.branch_pc = ins_[cfg_.blocks[r.header].last].pc;
    // IfThen {then}; IfElse {then, else}. The then-arm's s_branch and a break arm's back edge
    // are branches, which write nothing.
    for (uint32_t arm : r.children) add_arm(arm, placed.merge);
    out.ifs.push_back(std::move(placed));
}

void Placer::place_loop(const Region& r, StatePlacement& out) const {
    LoopPlacement placed;
    placed.header_pc = cfg_.blocks[r.header].pc;
    add_range(r.first_block, r.continue_block + 1, placed.carried);
    add_range(r.first_block, r.exit_branch + 1, placed.condition);
    const auto [lo, hi] = span(r.first_block, r.continue_block + 1);
    for (uint32_t i = lo; i < hi && i < first_end_; ++i)
        if (constant_lane_writelane(ins_[i]))
            placed.lane_slots.insert({ins_[i].dst.value, ins_[i].src[1].value});
    out.loops.push_back(std::move(placed));
}

StatePlacement Placer::run() {
    StatePlacement out;
    if (tree_.root == kNone || tree_.root >= tree_.regions.size() ||
        tree_.regions[tree_.root].kind == Kind::Unstructured)
        return out;
    std::vector<uint32_t> pending{tree_.root};
    std::vector<bool> seen(tree_.regions.size(), false);
    while (!pending.empty()) {
        const uint32_t id = pending.back();
        pending.pop_back();
        if (id == kNone || id >= tree_.regions.size() || seen[id]) continue;
        seen[id] = true;
        const Region& r = tree_.regions[id];
        if (r.kind == Kind::IfThen || r.kind == Kind::IfElse) place_if(r, out);
        if (r.kind == Kind::Loop || r.kind == Kind::CountedLoop) place_loop(r, out);
        pending.insert(pending.end(), r.children.begin(), r.children.end());
    }
    std::sort(out.ifs.begin(), out.ifs.end(),
              [](const auto& a, const auto& b) { return a.branch_pc < b.branch_pc; });
    std::sort(out.loops.begin(), out.loops.end(),
              [](const auto& a, const auto& b) { return a.header_pc < b.header_pc; });
    return out;
}

thread_local StatePlacementCensus* g_census = nullptr;

std::string regs_only(const PhiSets& a, const PhiSets& b) {
    std::string text;
    for (int reg : a.vgprs)
        if (!b.vgprs.contains(reg)) text += (text.empty() ? "v" : ",v") + std::to_string(reg);
    for (int reg : a.sgprs)
        if (!b.sgprs.contains(reg)) text += (text.empty() ? "s" : ",s") + std::to_string(reg);
    return text;
}

std::string slots_only(const LaneSlotSet& a, const LaneSlotSet& b) {
    std::string text;
    for (const auto& [vgpr, lane] : a)
        if (!b.contains({vgpr, lane}))
            text += (text.empty() ? "v" : ",v") + std::to_string(vgpr) + "[" +
                    std::to_string(lane) + "]";
    return text;
}

std::string difference_text(const std::string& emitter_only, const std::string& placed_only) {
    std::string text;
    if (!emitter_only.empty()) text = "emitter-only=" + emitter_only;
    if (!placed_only.empty()) text += (text.empty() ? "" : " ") + ("placed-only=" + placed_only);
    return text;
}

}   // namespace

void add_phi_writes(const Rdna2Inst& in, int max_vgpr, PhiSets& out) {
    const Rdna2InstructionEffects& effects = rdna2_instruction_effects(in);
    switch (in.fmt) {
        case Rdna2Format::VINTRP:
            if (!rdna2_vintrp_is_p1(in)) out.vgprs.insert(in.dst.value);
            return;
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
        case Rdna2Format::VOP3:
        case Rdna2Format::VOP3P: add_valu_writes(in, effects, max_vgpr, out); return;
        case Rdna2Format::DS: add_memory_writes(in, false, out); return;
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF:
        case Rdna2Format::MIMG:
        case Rdna2Format::FLAT: add_memory_writes(in, true, out); return;
        case Rdna2Format::SOP1: add_words(out.sgprs, in.dst.value, sop1_phi_words(in)); return;
        case Rdna2Format::SOPK:
            if (!effects.known() || effects.sdst_dwords != 0) out.sgprs.insert(in.dst.value);
            return;
        case Rdna2Format::SOP2: add_sop2_writes(in, out); return;
        case Rdna2Format::SMEM:
            // A reserved encoding keeps a one-register clobber.
            add_words(out.sgprs, in.dst.value,
                      effects.known() ? rdna2_smem_destination_dwords(in) : 1u);
            return;
        default: return;   // VOPC/SOPC write VCC/SCC/EXEC, which have their own phis
    }
}

// One word, except a 64-bit result that is scalar data (the table's pair destination whose sources
// are not two lane-mask pairs, or s_cselect_b64). A 64-bit bitwise op of two pairs (s_and_b64, ...)
// charges its low word: its value lives in the mask domain. A 64-bit shift into EXEC lives in the
// mask domain only and charges nothing. An encoding the table does not know charges one word.
// CONFIDENCE: HIGH -- every pair row's lowering writes both words of an ordinary SGPR destination
// (ir/lower/sop2_b64_routes.cpp; s_bfm_b64 through record_s_bfm_b64_scalar_data). #5436: the scan's
// own opcode list charged the pair for s_cselect/s_lshl/s_lshr_b64 only, and dropped the high word
// of s_ashr_i64, s_bfe_u64, s_bfe_i64 and s_bfm_b64 at every merge.
uint32_t sop2_phi_words(const Rdna2Inst& in) {
    const Rdna2InstructionEffects& effects = rdna2_instruction_effects(in);
    if (!effects.known() || effects.sdst_dwords != 2) return 1;
    const bool mask_pair_op = effects.scc_read == Rdna2SccRead::None &&
                              effects.ssrc_dwords[0] == 2 && effects.ssrc_dwords[1] == 2;
    const bool shift_into_exec = effects.ssrc_dwords[0] == 2 && effects.ssrc_dwords[1] == 1 &&
                                 (in.dst.value == 126 || in.dst.value == 127);
    return mask_pair_op ? 1u : (shift_into_exec ? 0u : 2u);
}

// One word, except a 64-bit result its lowering writes as scalar data in both words: the pair.
// The table's pair rows split three ways:
// - Scalar data (the pair): s_mov_b64, s_bitreplicate_b64_b32, s_getpc_b64 and the rows the
//   lowering refuses (s_cmov_b64, s_brev_b64, s_bitset*_b64, s_movrels_b64, s_swappc_b64).
//   s_mov_b64 overwrites both words in every lowering (ir/lower/sop1_b64_mask_routes.cpp): it
//   copies an SGPR pair, an inline constant or a literal, and a saved lane mask (EXEC, VCC)
//   either publishes both ballot words or retires both. s_bitreplicate_b64_b32 publishes both
//   words (sop1_data_routes.cpp); s_getpc_b64 retires both.
// - The mask domain (the low word, as SOP2's 64-bit bitwise ops of two pairs): the SAVEEXEC and
//   WREXEC families (D = a lane mask; both words are retired, the value lives in sreg_bool), and
//   the SCC-writing unary ops of a pair, s_not_b64, s_wqm_b64 and s_quadmask_b64. s_not_b64 is
//   mask-only; the numeric s_wqm_b64 route and s_quadmask_b64 also write both data words, which
//   this charge does not see (ir/cfg/AGENTS.md records the exception; #5441).
// - Into EXEC: the low word. s_mov_b64 and s_bitreplicate_b64_b32 into EXEC write rs.exec only.
// An M0-relative destination (s_movreld_b64, refused) charges the SDST field. An encoding the
// table does not know, and every one-word or no-word row, charges one word, as before.
// CONFIDENCE: HIGH for s_mov_b64 and s_bitreplicate_b64_b32 into an ordinary SGPR pair (#5439: an
// execution witness drops the high word at an if merge with the one-word charge).
uint32_t sop1_phi_words(const Rdna2Inst& in) {
    const Rdna2InstructionEffects& effects = rdna2_instruction_effects(in);
    if (!effects.known() || effects.sdst_dwords != 2) return 1;
    if (effects.exec != Rdna2ExecEffect::None) return 1;
    if (effects.m0_relative == Rdna2M0Relative::Destination ||
        effects.m0_relative == Rdna2M0Relative::Both)
        return 1;
    const bool mask_unary =
        (effects.implicit_writes & kRdna2ImplicitScc) != 0 && effects.ssrc_dwords[0] == 2;
    if (mask_unary) return 1;
    return in.dst.value == 126 || in.dst.value == 127 ? 1u : 2u;
}

StatePlacement place_state(const GuestCfg& cfg, const RegionTree& tree,
                           const std::vector<Rdna2Inst>& ins) {
    return Placer(cfg, tree, ins).run();
}

StatePlacement place_state(const std::vector<Rdna2Inst>& ins, const RegionPolicy& policy) {
    const GuestCfg cfg = build_guest_cfg(std::span<const Rdna2Inst>(ins), SplitPolicy{});
    return place_state(cfg, build_region_tree(cfg, ins, policy), ins);
}

const IfPlacement* StatePlacement::if_at(uint32_t branch_pc) const {
    const auto it = std::lower_bound(ifs.begin(), ifs.end(), branch_pc,
                                     [](const auto& p, uint32_t pc) { return p.branch_pc < pc; });
    return it != ifs.end() && it->branch_pc == branch_pc ? &*it : nullptr;
}

const LoopPlacement* StatePlacement::loop_at(uint32_t header_pc) const {
    const auto it = std::lower_bound(loops.begin(), loops.end(), header_pc,
                                     [](const auto& p, uint32_t pc) { return p.header_pc < pc; });
    return it != loops.end() && it->header_pc == header_pc ? &*it : nullptr;
}

const char* phi_site_name(PhiSite site) {
    switch (site) {
        case PhiSite::IfMerge: return "if-merge";
        case PhiSite::LoopCarried: return "loop-carried";
        case PhiSite::LoopCondition: return "loop-condition";
        case PhiSite::LaneSlots: return "lane-slots";
    }
    return "?";
}

void take_placed_if_equal(const PhiSets* placed, PhiSite site, uint32_t pc, std::set<int>& vgprs,
                          std::set<int>& sgprs) {
    StatePlacementCensus* census = g_census;
    if (census && census->withhold()) return;
    const bool equal = placed && placed->vgprs == vgprs && placed->sgprs == sgprs;
    if (census) {
        using V = PlacementSiteRecord::Verdict;
        census->record({site, pc, !placed ? V::Absent : (equal ? V::Equal : V::Differs),
                        placed && !equal ? difference_text(regs_only({vgprs, sgprs}, *placed),
                                                           regs_only(*placed, {vgprs, sgprs}))
                                         : ""});
    }
    if (!equal) return;   // the guard: anything but an exact match keeps the emitter's sets
    vgprs = placed->vgprs;
    sgprs = placed->sgprs;
}

void take_placed_if_equal(const LaneSlotSet* placed, uint32_t pc, LaneSlotSet& slots) {
    StatePlacementCensus* census = g_census;
    if (census && census->withhold()) return;
    const bool equal = placed && *placed == slots;
    if (census) {
        using V = PlacementSiteRecord::Verdict;
        census->record(
            {PhiSite::LaneSlots, pc, !placed ? V::Absent : (equal ? V::Equal : V::Differs),
             placed && !equal
                 ? difference_text(slots_only(slots, *placed), slots_only(*placed, slots))
                 : ""});
    }
    if (!equal) return;   // the guard
    slots = *placed;
}

StatePlacementCensus::StatePlacementCensus(bool withhold)
    : previous_(g_census), withhold_(withhold) {
    g_census = this;
}

StatePlacementCensus::~StatePlacementCensus() {
    g_census = previous_;
}

StatePlacementCensus* StatePlacementCensus::active() {
    return g_census;
}

}   // namespace prosper::gpu::ir::cfg
