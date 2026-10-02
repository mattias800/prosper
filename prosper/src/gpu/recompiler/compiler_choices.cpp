#include "gpu/recompiler/compiler_choices.hpp"
#include <stdexcept>

namespace prosper::gpu {
namespace { thread_local CompilerChoiceScope* active = nullptr; }
CompilerChoiceScope::CompilerChoiceScope(CompilerChoiceTrace& recording)
    : previous_(active), recording_(&recording) { active = this; }
CompilerChoiceScope::CompilerChoiceScope(const CompilerChoiceTrace& replay)
    : previous_(active), replay_(&replay) { active = this; }
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
void record_compiler_choice(CompilerChoice choice, int64_t value) {
    if (!active || !active->recording_) return;
    auto& trace = *active->recording_;
    if (trace.reads.size() >= 65536) {
        trace.error = "compiler-choice-budget";
        return;
    }
    try { trace.reads.push_back({choice, value}); }
    catch (const std::bad_alloc&) { trace.error = "compiler-choice-allocation-failed"; }
}
} // namespace prosper::gpu
