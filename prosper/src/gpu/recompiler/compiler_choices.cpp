#include "gpu/recompiler/compiler_choices.hpp"
#include <stdexcept>

namespace prosper::gpu {
namespace { thread_local CompilerChoiceScope* active = nullptr; }
CompilerChoiceScope::CompilerChoiceScope(CompilerChoiceTrace& recording)
    : previous_(active), recording_(&recording) { active = this; }
CompilerChoiceScope::CompilerChoiceScope(const CompilerChoiceTrace& replay)
    : previous_(active), replay_(&replay) { active = this; }
CompilerChoiceScope::CompilerChoiceScope(const CompilerChoiceSet& supplied)
    : previous_(active), supplied_(&supplied) { active = this; }
CompilerChoiceScope::~CompilerChoiceScope() { active = previous_; }
void CompilerChoiceScope::finish_replay() const {
    if (!replay_ || !replay_->error.empty() || position_ != replay_->reads.size())
        throw std::runtime_error("compiler choice trace not completely consumed");
}
bool replay_compiler_choice(CompilerChoice choice, int64_t& value) {
    if (!active || !active->replay_) return false;
    const auto& trace = *active->replay_;
    if (!trace.error.empty() || active->position_ >= trace.reads.size() ||
        trace.reads[active->position_].choice != choice)
        throw std::runtime_error("compiler choice trace diverged: unknown or missing semantic read");
    value = trace.reads[active->position_++].value;
    return true;
}
bool supplied_compiler_choice(CompilerChoice choice, int64_t& value) {
    // The innermost scope that answers wins; a recording-only scope does not answer.
    for (auto* scope = active; scope; scope = scope->previous_) {
        if (scope->replay_) return false;   // replay owns the sequence; it is asked first
        if (!scope->supplied_) continue;
        if (const auto v = scope->supplied_->get(choice)) {
            value = *v;
            return true;
        }
    }
    return false;
}
void record_compiler_choice(CompilerChoice choice, int64_t value) {
    // A nested producing observer must not hide semantic reads from its outer observer. Stop at
    // replay: an independent fidelity check is not another live compiler invocation.
    for (auto* scope = active; scope && !scope->replay_; scope = scope->previous_) {
        if (!scope->recording_) continue;
        auto& trace = *scope->recording_;
        try {
            if (trace.reads.size() >= 65536) {
                trace.error = "compiler-choice-budget";
                continue;
            }
            trace.reads.push_back({choice, value});
        } catch (const std::bad_alloc&) {
            // A recording allocation is observational; never prevent the live semantic read.
            try {
                trace.error = "compiler-choice-allocation-failed";
            } catch (const std::bad_alloc&) {
                continue;   // Even failure to record a diagnostic reason cannot skip a live read.
            }
        }
    }
}
} // namespace prosper::gpu
