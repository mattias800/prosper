#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {

// Versioned semantic reads, not an environment snapshot. A repeated read is a separate event;
// replay must consume the same sequence or fail closed. Observational logging is deliberately absent.
enum class CompilerChoice : uint32_t {
    FloatControls, FragmentTapPc, ForcedArrayLayer, VccScalarMerge,
    MimgSoft, NormalizeSamplerCoordinates,
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
    ~CompilerChoiceScope();
    CompilerChoiceScope(const CompilerChoiceScope&) = delete;
    CompilerChoiceScope& operator=(const CompilerChoiceScope&) = delete;
    void finish_replay() const;
private:
    friend bool replay_compiler_choice(CompilerChoice, int64_t&);
    friend void record_compiler_choice(CompilerChoice, int64_t);
    CompilerChoiceScope* previous_ = nullptr;
    CompilerChoiceTrace* recording_ = nullptr;
    const CompilerChoiceTrace* replay_ = nullptr;
    size_t position_ = 0;
};
bool replay_compiler_choice(CompilerChoice choice, int64_t& value);
void record_compiler_choice(CompilerChoice choice, int64_t value);
template<class Read> int64_t compiler_choice(CompilerChoice choice, Read&& live_read) {
    int64_t value = 0;
    if (replay_compiler_choice(choice, value)) return value;
    value = live_read();
    record_compiler_choice(choice, value);
    return value;
}

} // namespace prosper::gpu
