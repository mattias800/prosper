// exec_value_census.hpp -- the shadow check of IR_CONTROL_FLOW_PLAN.md step 7a: the static EXEC
// relations (exec_values.hpp) against the SPIR-V-id relations the emitter uses today.
//
// While a census is active on this thread, the emitter's hooks report three things to it
// (the same hooks independently feed the guarded consumer):
//   - the program each route is about to emit (place_state for the structured route,
//     adopt_guest_cfg_blocks for the dispatcher, whose blocks are its cases), for which the census
//     builds the static values;
//   - around every emit_alu, the SPIR-V ids in EXEC, VCC and the saved masks, which it maps to the
//     static value in that location before or after that instruction;
//   - every exec_within call and every read of exec_complement_in, which it answers again from the
//     static relations and classifies.
// Nothing here is read by emission, so the census cannot change an emitted byte.
#pragma once

#include "gpu/recompiler/ir/cfg/exec_values.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
}   // namespace prosper::gpu

namespace prosper::gpu::ir::cfg {

// Equal: both relations answer the same. StaticStronger: only the static relation proves it
// (expected where a dispatcher reload or a fresh SPIR-V id dropped a fact). StaticWeaker: only the
// SPIR-V relation proves it, which is a defect of exec_values unless explained. Unmapped: an id
// no hook saw in a location. CrossContext: the ids come from two different programs' values.
enum class ExecCompare : uint8_t { Equal, StaticStronger, StaticWeaker, Unmapped, CrossContext };
inline constexpr size_t kExecCompareKinds = 5;
const char* exec_compare_name(ExecCompare compare);

struct ExecCensusTally {
    size_t counts[kExecCompareKinds] = {};
    std::map<std::string, size_t> stronger;   // by the shape of the static proof
    std::map<std::string, size_t> unmapped;   // by the side no hook saw
    std::map<std::string, std::string> example;   // the first case of each shape
    size_t total() const;
};

struct ExecCensusCase {
    const char* site = "";   // "within" or "complement"
    uint32_t pc = 0;   // the instruction being emitted
    std::string detail;
};

class ExecValueCensus {
public:
    ExecValueCensus();
    ~ExecValueCensus();
    ExecValueCensus(const ExecValueCensus&) = delete;
    ExecValueCensus& operator=(const ExecValueCensus&) = delete;
    static ExecValueCensus* active();

    const ExecCensusTally& within() const { return within_; }
    const ExecCensusTally& complement() const { return complement_; }
    const std::vector<ExecCensusCase>& weaker() const { return weaker_; }
    // Observations where an id already mapped to a static value sat in a location the static
    // pass gives a different value (a copy the static pass does not model as one).
    size_t aliases() const { return aliases_; }
    size_t programs() const { return contexts_.size(); }
    // emit_alu calls whose instruction is not in the current program (no program yet, or a pc or
    // instruction the program does not have): nothing they hold is mapped.
    size_t misses() const { return misses_; }

    void begin_program(const GuestCfg& cfg, std::span<const Rdna2Inst> ins, bool dispatcher);
    void observe(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in, bool after);
    void compare_within(const RegState& rs, uint32_t reader, uint32_t owner, bool spirv);
    void compare_complement(const RegState& rs, uint32_t value);

private:
    struct Context {
        std::vector<Rdna2Inst> ins;
        std::unordered_map<uint32_t, uint32_t> index_of_pc;
        ExecValues values;
        bool dispatcher = false;
    };
    struct Handle {
        uint32_t context = 0, value = 0;
        uint32_t seen_pc = 0;   // where the id was first seen holding this value
        int location = 0;   // and in which mask location
        bool after = false;   // after (true) or before that instruction
    };
    struct Ids {
        uint32_t next_id = 0;
        std::unordered_map<uint32_t, std::vector<Handle>> handles;
    };
    void note(uint32_t id, uint32_t context, uint32_t value, int location, bool after);
    const std::vector<Handle>* handles(uint32_t id) const;
    std::string describe(uint32_t id, const std::vector<Handle>& handles) const;
    std::string spirv_chain(const RegState& rs, uint32_t reader, uint32_t owner) const;
    std::string proof_shape(const Handle& reader, const Handle& owner,
                            const std::vector<uint32_t>& path) const;
    void tally(ExecCensusTally& tally, ExecCompare compare, const char* site, std::string detail);

    std::vector<std::unique_ptr<Context>> contexts_;
    std::map<const void*, Ids> builders_;
    Ids* current_ = nullptr;
    uint32_t pc_ = 0;
    ExecCensusTally within_, complement_;
    std::vector<ExecCensusCase> weaker_;
    size_t aliases_ = 0;
    size_t misses_ = 0;
    ExecValueCensus* previous_ = nullptr;
};

// The emitter's hooks. They also feed the guarded step-7b consumer when an emission scope exists.
void exec_census_program(const GuestCfg& cfg, std::span<const Rdna2Inst> ins, bool dispatcher);
void exec_census_within(const RegState& rs, uint32_t reader, uint32_t owner, bool spirv);
void exec_census_complement(const RegState& rs, uint32_t value);

// Hides the census from a walk that is not an emission route (recompile_coverage's linear pass,
// which ignores control flow, so its EXEC ids follow no program point).
class ExecCensusSuspend {
public:
    ExecCensusSuspend();
    ~ExecCensusSuspend();
    ExecCensusSuspend(const ExecCensusSuspend&) = delete;
    ExecCensusSuspend& operator=(const ExecCensusSuspend&) = delete;

private:
    ExecValueCensus* suspended_;
};

// Reports the masks before an emit_alu call and, when it is destroyed, after it.
class ExecCensusObservation {
public:
    ExecCensusObservation(const SpirvCompute& b, const RegState& rs, const Rdna2Inst& in);
    ~ExecCensusObservation();
    ExecCensusObservation(const ExecCensusObservation&) = delete;
    ExecCensusObservation& operator=(const ExecCensusObservation&) = delete;

private:
    const SpirvCompute& b_;
    const RegState& rs_;
    const Rdna2Inst& in_;
    bool active_;
};

}   // namespace prosper::gpu::ir::cfg
