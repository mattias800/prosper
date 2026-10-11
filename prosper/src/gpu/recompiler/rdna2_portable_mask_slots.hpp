// Ordinary portable Wave64 mask-slot words retain a predicate and its architectural half.
// These are symbolic projections, never scalar DATA or permission to ballot departed lanes.
// CONFIDENCE: HIGH — project-owned production modules expose both half projections separately.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;

struct PortableMaskHalfTypes {
    std::map<int, uint32_t> words;
    std::map<std::pair<int, int>, uint32_t> slots;
    // MAY facts prevent a one-arm/role-conflicting half from looking like a complete Bool.
    std::set<int> may_words;
    std::set<std::pair<int, int>> may_slots;
    std::set<int> complete;
    bool operator==(const PortableMaskHalfTypes&) const = default;
};
struct PortableMaskHalfProgram {
    std::map<uint32_t, PortableMaskHalfTypes> before, after;
};
class PortableMaskHalfBodyScope {
public:
    PortableMaskHalfBodyScope(SpirvCompute&, const RegState&, const std::vector<Rdna2Inst>&);
    ~PortableMaskHalfBodyScope();

private:
    SpirvCompute& b_;
    std::shared_ptr<const PortableMaskHalfProgram> previous_;
};
// Emits only the named symbolic spill/reload and B32-copy forms. For other instructions it
// authenticates symbolic reads, temporarily composes complete B64 source predicates, and then
// restores the unchanged per-word carriers. Every architectural overwrite ends the old role.
class PortableMaskHalfInstruction {
public:
    PortableMaskHalfInstruction(SpirvCompute&, RegState&, const Rdna2Inst&, bool&);
    ~PortableMaskHalfInstruction();
    bool handled() const { return handled_; }

private:
    RegState& rs_;
    std::map<int, uint32_t> restore_;
    std::set<int> written_;
    bool handled_ = false;
};
void add_portable_mask_half_keys(const SpirvCompute&, std::set<int>&);
bool portable_mask_half_at_entry(const SpirvCompute&, uint32_t pc, int word);
void load_portable_mask_half_types(const SpirvCompute&, RegState&, uint32_t pc);
} // namespace prosper::gpu
