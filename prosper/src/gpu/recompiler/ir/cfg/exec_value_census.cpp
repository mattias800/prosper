// exec_value_census.cpp -- the step-7a shadow census; see exec_value_census.hpp and the "EXEC
// values" section of AGENTS.md.
#include "gpu/recompiler/ir/cfg/exec_value_census.hpp"

#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <algorithm>

namespace prosper::gpu::ir::cfg {
namespace {

thread_local ExecValueCensus* g_census = nullptr;

// Kept per census; the counts stay complete past it.
constexpr size_t kMaxWeakerCases = 200;

bool reload_involved(const ExecValues& values, uint32_t value) {
    for (int hop = 0; hop < 64 && value < values.values.size(); ++hop) {
        if (values.values[value].kind == ExecValue::Kind::DispatchReload) return true;
        value = values.values[value].same_as;
    }
    return false;
}

}   // namespace

const char* exec_compare_name(ExecCompare compare) {
    switch (compare) {
        case ExecCompare::Equal: return "equal";
        case ExecCompare::StaticStronger: return "static-stronger";
        case ExecCompare::StaticWeaker: return "static-weaker";
        case ExecCompare::Unmapped: return "unmapped";
        case ExecCompare::CrossContext: return "cross-context";
    }
    return "?";
}

size_t ExecCensusTally::total() const {
    size_t sum = 0;
    for (size_t count : counts) sum += count;
    return sum;
}

ExecValueCensus::ExecValueCensus() : previous_(g_census) {
    g_census = this;
}

ExecValueCensus::~ExecValueCensus() {
    g_census = previous_;
}

ExecValueCensus* ExecValueCensus::active() {
    return g_census;
}

void ExecValueCensus::begin_program(const GuestCfg& cfg, std::span<const Rdna2Inst> ins,
                                    bool dispatcher) {
    auto context = std::make_unique<Context>();
    context->ins.assign(ins.begin(), ins.end());
    for (size_t i = 0; i < context->ins.size(); ++i)
        context->index_of_pc.emplace(context->ins[i].pc, static_cast<uint32_t>(i));
    context->values = build_exec_values(cfg, context->ins, {dispatcher});
    context->dispatcher = dispatcher;
    contexts_.push_back(std::move(context));
}

void ExecValueCensus::note(uint32_t id, uint32_t context, uint32_t value, int location,
                           bool after) {
    if (!current_ || id == 0 || value == kNone) return;
    auto& list = current_->handles[id];
    const ExecValues& values = contexts_[context]->values;
    bool alias = false;
    for (const Handle& h : list) {
        if (h.context != context) continue;
        if (values.representative(h.value) == values.representative(value)) return;
        alias = true;
    }
    aliases_ += alias;
    list.push_back({context, value, pc_, location, after});
}

// Builder identity: a SpirvCompute allocates ids upward, so a smaller next id at the same address
// is a new builder, whose ids mean nothing the old one's did.
void ExecValueCensus::observe(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                              bool after) {
    Ids& ids = builders_[&b];
    if (b.next_id < ids.next_id) ids.handles.clear();
    ids.next_id = b.next_id;
    current_ = &ids;
    pc_ = in.pc;
    if (contexts_.empty()) {
        misses_ += !after;
        return;
    }
    const uint32_t context = static_cast<uint32_t>(contexts_.size() - 1);
    const Context& ctx = *contexts_.back();
    const auto found = ctx.index_of_pc.find(in.pc);
    if (found == ctx.index_of_pc.end() || !(ctx.ins[found->second] == in)) {
        misses_ += !after;
        return;
    }
    const uint32_t index = found->second;
    if (!after) {
        note(rs.exec, context, ctx.values.exec_before[index], kExecLocation, false);
        return;
    }
    note(rs.exec, context, ctx.values.exec_after[index], kExecLocation, true);
    for (const auto& [location, value] : ctx.values.defined[index]) {
        if (location == kExecLocation) continue;
        if (location == kVccLocation) note(rs.vcc, context, value, location, true);
        const auto saved = rs.sreg_bool.find(location);
        if (saved != rs.sreg_bool.end()) note(saved->second, context, value, location, true);
    }
}

const std::vector<ExecValueCensus::Handle>* ExecValueCensus::handles(uint32_t id) const {
    if (!current_ || id == 0) return nullptr;
    const auto found = current_->handles.find(id);
    return found == current_->handles.end() ? nullptr : &found->second;
}

// "id%N=" then each static value the id was seen holding, with where it was first seen.
std::string ExecValueCensus::describe(uint32_t id, const std::vector<Handle>& list) const {
    std::string out = "%" + std::to_string(id) + "=";
    bool first = true;
    for (const Handle& h : list) {
        const Context& ctx = *contexts_[h.context];
        const ExecValue& v = ctx.values.values[h.value];
        const uint32_t pc = v.inst != kNone ? ctx.ins[v.inst].pc : kNone;
        if (!first) out += ',';
        first = false;
        out += std::string(exec_value_kind_name(v.kind)) + "#" + std::to_string(h.value);
        if (pc != kNone) {
            const Rdna2Inst& in = ctx.ins[v.inst];
            const char* name = rdna2_instruction_effects(in.fmt, in.opcode).name;
            out += "@pc" + std::to_string(pc) + "(" + (name ? name : "?") + " " +
                   std::to_string(in.dst.value) + "," + std::to_string(in.src[0].value) + "," +
                   std::to_string(in.src[1].value) + ")";
        }
        if (v.block != kNone && pc == kNone) out += "@block" + std::to_string(v.block);
        out += ctx.dispatcher ? "/dispatcher" : "/structured";
        out += "[seen " + std::string(h.after ? "after" : "before") + " pc" +
               std::to_string(h.seen_pc) + " s" + std::to_string(h.location) + "]";
    }
    return out;
}

// The kinds the static proof passes through, reader first, and whether a reload or a copy
// stands in for a SPIR-V id the emitter replaced.
std::string ExecValueCensus::proof_shape(const Handle& reader, const Handle& owner,
                                         const std::vector<uint32_t>& path) const {
    const ExecValues& values = contexts_[reader.context]->values;
    std::string out;
    if (reload_involved(values, reader.value) || reload_involved(values, owner.value))
        out = "reload:";
    else if (values.representative(reader.value) != reader.value ||
             values.representative(owner.value) != owner.value)
        out = "copy:";
    if (path.size() == 1) return out + "same-value";
    for (size_t i = 0; i < path.size(); ++i) {
        if (i) out += '>';
        out += exec_value_kind_name(values.values[path[i]].kind);
    }
    return out;
}

void ExecValueCensus::tally(ExecCensusTally& t, ExecCompare compare, const char* site,
                            std::string detail) {
    ++t.counts[static_cast<size_t>(compare)];
    if (compare == ExecCompare::StaticStronger) ++t.stronger[detail];
    if (compare == ExecCompare::StaticWeaker && weaker_.size() < kMaxWeakerCases)
        weaker_.push_back({site, pc_, std::move(detail)});
}

// The SPIR-V proof a static-weaker case lacks: each id from reader to owner through
// exec_subset_of, with the static values each was seen holding.
std::string ExecValueCensus::spirv_chain(const RegState& rs, uint32_t reader,
                                         uint32_t owner) const {
    std::string out;
    uint32_t id = reader;
    for (int hop = 0; hop < 16 && id; ++hop) {
        const auto* list = handles(id);
        out += hop ? " < " : "";
        out += list ? describe(id, *list) : "%" + std::to_string(id) + "=unmapped";
        if (id == owner) break;
        const auto parent = rs.exec_subset_of.find(id);
        if (parent == rs.exec_subset_of.end()) break;
        id = parent->second;
    }
    return out;
}

void ExecValueCensus::compare_within(const RegState& rs, uint32_t reader, uint32_t owner,
                                     bool spirv) {
    if (!current_ || reader == 0 || owner == 0) return;
    const auto* readers = handles(reader);
    const auto* owners = handles(owner);
    if (!readers || !owners) {
        const std::string side =
            !readers ? (!owners ? "unmapped:both" : "unmapped:reader") : "unmapped:owner";
        if (!within_.example.contains(side))
            within_.example[side] = "pc=" + std::to_string(pc_) +
                                    " spirv=" + (spirv ? "within" : "not") +
                                    " chain: " + spirv_chain(rs, reader, owner);
        ++within_.unmapped[side];
        tally(within_, ExecCompare::Unmapped, "within", "");
        return;
    }
    bool shared = false, proved = false;
    std::string shape;
    for (const Handle& r : *readers) {
        for (const Handle& o : *owners) {
            if (r.context != o.context) continue;
            shared = true;
            std::vector<uint32_t> path;
            if (!contexts_[r.context]->values.within(r.value, o.value, &path)) continue;
            proved = true;
            shape = proof_shape(r, o, path);
            break;
        }
        if (proved) break;
    }
    if (!shared) {
        tally(within_, ExecCompare::CrossContext, "within", "");
    } else if (proved == spirv) {
        tally(within_, ExecCompare::Equal, "within", "");
    } else if (proved) {
        if (!within_.example.contains(shape))
            within_.example[shape] = "pc=" + std::to_string(pc_) +
                                     " reader=" + describe(reader, *readers) +
                                     " owner=" + describe(owner, *owners);
        tally(within_, ExecCompare::StaticStronger, "within", shape);
    } else {
        tally(within_, ExecCompare::StaticWeaker, "within",
              "reader=" + describe(reader, *readers) + " owner=" + describe(owner, *owners) +
                  " spirv-chain: " + spirv_chain(rs, reader, owner));
    }
}

// The emitter reads complement_in[value]; the static pass answers for each static value the id
// holds. A SPIR-V fact the static pass lacks, or one whose parts the static fact does not name,
// is weaker.
void ExecValueCensus::compare_complement(const RegState& rs, uint32_t value) {
    if (!current_ || value == 0) return;
    const auto* list = handles(value);
    if (!list) {
        tally(complement_, ExecCompare::Unmapped, "complement", "");
        return;
    }
    const auto spirv = rs.exec_complement_in.find(value);
    const bool spirv_has = spirv != rs.exec_complement_in.end();
    const auto names = [&](uint32_t id, const Handle& h, uint32_t static_value) {
        const auto* ids = handles(id);
        if (!ids) return false;
        const ExecValues& values = contexts_[h.context]->values;
        return std::any_of(ids->begin(), ids->end(), [&](const Handle& other) {
            return other.context == h.context &&
                   values.representative(other.value) == values.representative(static_value);
        });
    };
    bool static_has = false, same_parts = false;
    for (const Handle& h : *list) {
        const ExecComplement fact = contexts_[h.context]->values.complement(h.value);
        if (!fact.present()) continue;
        static_has = true;
        same_parts = same_parts || (spirv_has && names(spirv->second.first, h, fact.whole) &&
                                    names(spirv->second.second, h, fact.part));
    }
    if (spirv_has == static_has && (!spirv_has || same_parts)) {
        tally(complement_, ExecCompare::Equal, "complement", "");
    } else if (static_has && !spirv_has) {
        if (!complement_.example.contains("complement"))
            complement_.example["complement"] =
                "pc=" + std::to_string(pc_) + " value=" + describe(value, *list);
        tally(complement_, ExecCompare::StaticStronger, "complement", "complement");
    } else {
        tally(complement_, ExecCompare::StaticWeaker, "complement",
              std::string(static_has ? "parts differ " : "no static fact ") +
                  describe(value, *list));
    }
}

void exec_census_program(const GuestCfg& cfg, std::span<const Rdna2Inst> ins, bool dispatcher) {
    if (g_census) g_census->begin_program(cfg, ins, dispatcher);
}

void exec_census_within(const RegState& rs, uint32_t reader, uint32_t owner, bool spirv) {
    if (g_census) g_census->compare_within(rs, reader, owner, spirv);
}

void exec_census_complement(const RegState& rs, uint32_t value) {
    if (g_census) g_census->compare_complement(rs, value);
}

ExecCensusSuspend::ExecCensusSuspend() : suspended_(g_census) {
    g_census = nullptr;
}

ExecCensusSuspend::~ExecCensusSuspend() {
    g_census = suspended_;
}

ExecCensusObservation::ExecCensusObservation(const SpirvCompute& b, const RegState& rs,
                                             const Rdna2Inst& in)
    : b_(b), rs_(rs), in_(in), active_(g_census != nullptr) {
    if (active_) g_census->observe(b_, rs_, in_, false);
}

ExecCensusObservation::~ExecCensusObservation() {
    if (active_ && g_census) g_census->observe(b_, rs_, in_, true);
}

}   // namespace prosper::gpu::ir::cfg
