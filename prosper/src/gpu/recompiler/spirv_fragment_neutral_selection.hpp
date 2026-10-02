#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

enum class FragmentNeutralType { Other, Boolean, Int32, Float32 };

struct FragmentNeutralValue {
    FragmentNeutralType type = FragmentNeutralType::Other;
    uint32_t opcode = 0;
    uint32_t extended_opcode = 0;
    std::vector<uint32_t> operands;
    // Ordinary scalar constants, never specialization/undefined values. Float bits supply only
    // finite-domain authority, never an exact floating export identity.
    bool constant = false;
    uint32_t literal = 0;
    bool stable_input = false; // caller-certified builtins, not arbitrary Location inputs
    bool explicit_transport = false; // exact caller-validated per-op FPFastMathMode None
};

struct FragmentNeutralExport {
    uint32_t skipped = 0;
    uint32_t executed = 0;
};

// Internal graph from a valid module carrying the complete guest-wave (including helpers) vote
// contract, not an arbitrary host subgroup Any. The caller proves a single common-entry, one-arm,
// straight-line selection, sole Any controller use, no escaping SSA/CFG and no module-wide
// rendezvous effects. Body IDs are in execution order; exports include EVERY live merge Phi.
// The caller excludes FPFastMathDefault and all per-op environments except exact transport None.
// Transport None supplies no stable/uniform/address authority. No arbitrary Input
// load is a definedness seed. Frozen leaves are independently defined values;
// this certificate must never be fed back into address/load admission.
struct FragmentNeutralSelection {
    uint32_t predicate = 0;
    bool preserve_f32 = false;
    std::unordered_map<uint32_t, FragmentNeutralValue> values;
    std::vector<uint32_t> body;
    std::vector<FragmentNeutralExport> exports;
};

// Certifies replacing the controller Any(P) with TRUE, NOT P or FALSE. Only all-false waves gain
// executions. Every extra operation must be UB-free; poison-producing *values* are permitted only
// when masked away before an observable export. Nonpoison is not the same as stable/defined:
// false AND undefined can become stable false, but false AND poison cannot.
bool prove_fragment_neutral_selection(const FragmentNeutralSelection& selection,
                                     const std::unordered_set<uint32_t>& frozen_leaves);

} // namespace prosper::gpu
