#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {
struct Rdna2Inst;
enum class ShaderProgramStage : uint8_t;

// Cold facts about the ORIGINAL stage, before folding/dead-code or native reflection. A caller
// must alias this result to the same immutable owner as source_words. This value alone grants
// neither a read point nor source/entry/attachment/currentness authority.
struct OriginalGraphicsStageEffects {
    const std::vector<uint32_t>* source_words = nullptr;
    ShaderProgramStage stage{};
    uint32_t consumed_dwords = 0, instruction_count = 0;
    bool architectural_end = false;
    bool guest_memory_reads = false;
    bool attachment_exports = false;   // original PS EXP presence, not completed attachment writes
    std::string rejection;

    bool known_read_only() const { return architectural_end && rejection.empty(); }
    uint64_t retained_bytes() const { return rejection.capacity() + 1; }
};

// Complete straight-line VS/PS effects only. All control transfers and unimplemented effects
// refuse; an END/Unknown prefix is never mistaken for the whole original program. No warm call
// is needed when the producer already retains these exact immutable cold facts.
OriginalGraphicsStageEffects original_graphics_stage_effects(const std::vector<uint32_t>& original,
                                                             const std::vector<Rdna2Inst>& full,
                                                             ShaderProgramStage stage);
}   // namespace prosper::gpu
