#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace prosper::gpu {

// Versioned semantic reads, not an environment snapshot. A repeated read is a separate event;
// replay must consume the same sequence or fail closed. Observational logging is deliberately absent.
enum class CompilerChoice : uint32_t {
    FloatControls,
    FragmentTapPc,
    ForcedArrayLayer,
    VccScalarMerge,
    MimgSoft,
    NormalizeSamplerCoordinates,
    VertexTapPc,
    ForcePositionW,
};
// ADR 0006 / spec GPU-1: the values a recompile REQUEST carries for the switch-type choices. An entry
// that is set answers its choice with no live read; an unset entry falls through to the choice's own
// live read, which is what an entry point that has not migrated yet still does. Built from the
// environment by the caller (gpu/execute), never by the recompiler.
struct CompilerChoiceSet {
    static constexpr size_t kCount = static_cast<size_t>(CompilerChoice::ForcePositionW) + 1;
    std::array<std::optional<int64_t>, kCount> value{};
    void set(CompilerChoice choice, int64_t v) { value[static_cast<size_t>(choice)] = v; }
    std::optional<int64_t> get(CompilerChoice choice) const {
        return value[static_cast<size_t>(choice)];
    }
    bool operator==(const CompilerChoiceSet&) const = default;
};
struct CompilerChoiceRead {
    CompilerChoice choice = CompilerChoice::FloatControls;
    int64_t value = 0;
    bool operator==(const CompilerChoiceRead&) const = default;
};
struct CompilerChoiceTrace {
    std::vector<CompilerChoiceRead> reads;
    std::string error;
};
class CompilerChoiceScope {
public:
    explicit CompilerChoiceScope(CompilerChoiceTrace& recording);
    explicit CompilerChoiceScope(const CompilerChoiceTrace& replay);
    // The request's own values (ADR 0006): answers every set choice without a live read.
    explicit CompilerChoiceScope(const CompilerChoiceSet& supplied);
    ~CompilerChoiceScope();
    CompilerChoiceScope(const CompilerChoiceScope&) = delete;
    CompilerChoiceScope& operator=(const CompilerChoiceScope&) = delete;
    void finish_replay() const;
private:
    friend bool replay_compiler_choice(CompilerChoice, int64_t&);
    friend bool supplied_compiler_choice(CompilerChoice, int64_t&);
    friend void record_compiler_choice(CompilerChoice, int64_t);
    CompilerChoiceScope* previous_ = nullptr;
    CompilerChoiceTrace* recording_ = nullptr;
    const CompilerChoiceTrace* replay_ = nullptr;
    const CompilerChoiceSet* supplied_ = nullptr;
    size_t position_ = 0;
};
bool replay_compiler_choice(CompilerChoice choice, int64_t& value);
bool supplied_compiler_choice(CompilerChoice choice, int64_t& value);
void record_compiler_choice(CompilerChoice choice, int64_t value);
template<class Read> int64_t compiler_choice(CompilerChoice choice, Read&& live_read) {
    int64_t value = 0;
    if (replay_compiler_choice(choice, value)) return value;
    if (supplied_compiler_choice(choice, value)) {
        record_compiler_choice(choice, value);   // an outer recording still sees the semantic read
        return value;
    }
    value = live_read();
    record_compiler_choice(choice, value);
    return value;
}

// The four switch-type choices an emitter reads without a lambda: a request-supplied value, else
// the replayed value, else the environment adapter (compiler_choice_environment.cpp). Kept to those
// four on purpose; FloatControls and the taps still use the lambda form above.
int64_t environment_compiler_choice(CompilerChoice choice);
// Every switch-type choice's value from the environment, for a caller to hand to a request. Reading
// the environment is the caller's act, not the recompiler's (ADR 0006).
CompilerChoiceSet compiler_choices_from_environment();
inline int64_t compiler_choice(CompilerChoice choice) {
    return compiler_choice(choice, [choice] { return environment_compiler_choice(choice); });
}

} // namespace prosper::gpu
