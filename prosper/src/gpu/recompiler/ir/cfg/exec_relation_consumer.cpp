// Static EXEC answers are consumed only on equality; see exec_relation_consumer.hpp.
#include "gpu/recompiler/ir/cfg/exec_relation_consumer.hpp"
#include "gpu/recompiler/ir/cfg/exec_values.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace prosper::gpu::ir::cfg {
namespace {
thread_local ExecRelationScope* g_scope = nullptr;
thread_local ExecRelationCapture* g_capture = nullptr;

template <class T>
T take_equal(std::optional<T> answer, T by_id, ExecRelationCounts* counts) {
    if (counts) {
        if (!answer)
            ++counts->unmapped;
        else if (*answer == by_id)
            ++counts->equal;
        else
            ++counts->differs;
    }
    return answer && *answer == by_id ? *answer : by_id;
}
}   // namespace

ExecRelationCapture::ExecRelationCapture(Mode mode) : mode(mode), previous_(g_capture) {
    g_capture = this;
}
ExecRelationCapture::~ExecRelationCapture() {
    g_capture = previous_;
}
ExecRelationCapture* ExecRelationCapture::active() {
    return g_capture;
}

ExecRelationScope::ExecRelationScope(const SpirvCompute& builder)
    : builder_(builder), previous_(g_scope) {
    g_scope = this;
}
ExecRelationScope::~ExecRelationScope() {
    g_scope = previous_;
}
ExecRelationScope* ExecRelationScope::active() {
    return g_scope;
}

void ExecRelationScope::begin_program(const GuestCfg& cfg, std::span<const Rdna2Inst> ins,
                                      bool dispatcher) {
    ins_.assign(ins.begin(), ins.end());
    index_of_pc_.clear();
    value_of_id_.clear();
    for (size_t i = 0; i < ins_.size(); ++i)
        index_of_pc_.emplace(ins_[i].pc, static_cast<uint32_t>(i));
    values_ = build_exec_values(cfg, ins_, {dispatcher});
}

void ExecRelationScope::note(uint32_t id, uint32_t value) {
    if (!id || value == kNone) return;
    value = values_.representative(value);
    const auto [it, inserted] = value_of_id_.emplace(id, value);
    if (!inserted && it->second != value) it->second = kNone;
}

void ExecRelationScope::observe(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                bool after) {
    if (&b != &builder_) return;
    const auto found = index_of_pc_.find(in.pc);
    if (found == index_of_pc_.end() || !(ins_[found->second] == in)) return;
    const uint32_t index = found->second;
    note(rs.exec, after ? values_.exec_after[index] : values_.exec_before[index]);
    if (!after) return;
    for (const auto& [location, value] : values_.defined[index]) {
        if (location == kExecLocation) continue;
        if (location == kVccLocation) note(rs.vcc, value);
        const auto saved = rs.sreg_bool.find(location);
        if (saved != rs.sreg_bool.end()) note(saved->second, value);
    }
}

uint32_t ExecRelationScope::value_of(uint32_t id) const {
    const auto found = value_of_id_.find(id);
    return found == value_of_id_.end() ? kNone : found->second;
}

uint32_t ExecRelationScope::id_of(uint32_t value) const {
    const uint32_t representative = values_.representative(value);
    for (const auto& [id, mapped] : value_of_id_)
        if (mapped != kNone && mapped == representative) return id;
    return 0;
}

std::optional<bool> ExecRelationScope::within(uint32_t reader, uint32_t owner) const {
    const uint32_t r = value_of(reader), o = value_of(owner);
    if (r == kNone || o == kNone) return std::nullopt;
    return values_.within(r, o);
}

std::optional<ExecPartition> ExecRelationScope::complement(uint32_t value) const {
    const uint32_t mapped = value_of(value);
    if (mapped == kNone) return std::nullopt;
    const ExecComplement fact = values_.complement(mapped);
    if (!fact.present()) return ExecPartition{};
    const uint32_t whole = id_of(fact.whole), part = id_of(fact.part);
    if (!whole || !part) return std::nullopt;
    return ExecPartition{{whole, part}};
}

bool guarded_exec_within(uint32_t reader, uint32_t owner, bool by_id) {
    std::optional<bool> answer = g_scope ? g_scope->within(reader, owner) : std::nullopt;
    if (g_capture && g_capture->mode == ExecRelationCapture::Mode::Withhold) answer.reset();
    if (g_capture && g_capture->mode == ExecRelationCapture::Mode::Corrupt && answer)
        answer = !*answer;
    return take_equal(answer, by_id, g_capture ? &g_capture->within : nullptr);
}

ExecPartition guarded_exec_complement(uint32_t value, ExecPartition by_id) {
    std::optional<ExecPartition> answer = g_scope ? g_scope->complement(value) : std::nullopt;
    if (g_capture && g_capture->mode == ExecRelationCapture::Mode::Withhold) answer.reset();
    if (g_capture && g_capture->mode == ExecRelationCapture::Mode::Corrupt && answer)
        answer = *answer ? ExecPartition{} : ExecPartition{{0u, 0u}};
    return take_equal(answer, by_id, g_capture ? &g_capture->complement : nullptr);
}

}   // namespace prosper::gpu::ir::cfg
