// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"

namespace prosper::gpu {


// Does THIS module write the trip-bound witness?
//
// Derived from the compiled artifact, not from process history. An earlier revision kept a global
// set of program addresses that had ever emitted one, which cannot express the contract the host
// needs: the set was monotonic and keyed only by address, so once a program emitted under one phase,
// recompiling the SAME address under a phase it does not have still answered "instrumented" -- and
// the host would then read and clear guest-visible dwords no shader in the current module writes.
//
// Reading the module removes the whole class: the answer is a property of the bytes the backend is
// about to run, so it cannot be stale, cannot be defeated by a shader-cache hit, and needs no
// invalidation. The witness's first field is published by an atomic through an OpAccessChain onto the
// internal GDS binding at kComputeTripWitnessDword, which nothing else emits.
bool spirv_writes_trip_witness(const std::vector<uint32_t>& spirv) {
    // FAIL CLOSED, and "well formed" means the exact signature this function relies on -- not merely
    // that the words parse. The result authorizes the host to write guest-visible GDS, so every step
    // that could be true by accident has to be pinned:
    //
    //   * a decorated ID must actually name an OpVariable (a decoration can outlive its target);
    //   * a candidate must not carry conflicting DescriptorSet/Binding values;
    //   * the instructions consumed must have their EXACT operand counts -- a truncated OpAtomicUMax
    //     whose declared length happens to end at the module boundary passes any `word + len` check;
    //   * the access chain must have the shape the builder emits, not merely enough operands.
    //
    // This predicate consumes prosper's own generator output, so accepting exactly that canonical
    // form and refusing everything else is both sufficient and the conservative choice. It does not
    // attempt general SPIR-V validation, and it is not a substitute for spirv-val.
    constexpr uint32_t kSpirvMagic = 0x07230203u;
    if (spirv.size() < 5 || spirv[0] != kSpirvMagic) return false;

    constexpr uint32_t OpDecorate = 71, OpVariable = 59, OpConstant = 43, OpAccessChain = 65,
                       OpAtomicUMax = 239;
    constexpr uint32_t DecorationBinding = 33, DecorationDescriptorSet = 34;
    // Exact word counts for the forms consumed below (opcode word included).
    constexpr uint32_t kDecorateLiteralWords = 4;      // OpDecorate target decoration literal
    constexpr uint32_t kVariableWords = 4;             // result-type result storage-class, NO init
    constexpr uint32_t kAccessChainWords = 6;          // result-type result base member-0 slot
    constexpr uint32_t kAtomicUMaxWords = 7;           // result-type result pointer scope sem value

    std::set<uint32_t> variables;
    std::map<uint32_t, uint32_t> descriptor_set, binding;
    std::set<uint32_t> conflicting;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t op = spirv[word] & 0xffffu, len = spirv[word] >> 16;
        if (!len || word + len > spirv.size()) return false;   // truncated stream: fail closed
        // EXACT, like every other instruction consumed as proof. The builder emits the internal-GDS
        // variable as four words with no initializer; a longer OpVariable is a different declaration
        // and this predicate has no business reasoning about it.
        if (op == OpVariable && len == kVariableWords) {
            variables.insert(spirv[word + 2]);
        } else if (op == OpDecorate && len == kDecorateLiteralWords) {
            const uint32_t target = spirv[word + 1], value = spirv[word + 3];
            auto record = [&](std::map<uint32_t, uint32_t>& into) {
                const auto existing = into.find(target);
                if (existing == into.end()) into.emplace(target, value);
                else if (existing->second != value) conflicting.insert(target);
            };
            if (spirv[word + 2] == DecorationDescriptorSet) record(descriptor_set);
            else if (spirv[word + 2] == DecorationBinding) record(binding);
        }
        word += len;
    }

    std::set<uint32_t> witness_variables;
    for (uint32_t id : variables) {
        if (conflicting.count(id)) continue;
        const auto set_it = descriptor_set.find(id);
        const auto binding_it = binding.find(id);
        if (set_it != descriptor_set.end() && set_it->second == 0u &&
            binding_it != binding.end() && binding_it->second == kComputeInternalGdsBinding)
            witness_variables.insert(id);
    }
    if (witness_variables.empty()) return false;

    std::set<uint32_t> zero_constants, slot_constants, witness_pointers;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t op = spirv[word] & 0xffffu, len = spirv[word] >> 16;
        if (!len || word + len > spirv.size()) return false;
        if (op == OpConstant && len == 4) {
            if (spirv[word + 3] == 0u) zero_constants.insert(spirv[word + 2]);
            else if (spirv[word + 3] == kComputeTripWitnessDword)
                slot_constants.insert(spirv[word + 2]);
        } else if (op == OpAccessChain && len == kAccessChainWords &&
                   witness_variables.count(spirv[word + 3]) &&
                   zero_constants.count(spirv[word + 4]) &&
                   slot_constants.count(spirv[word + 5])) {
            witness_pointers.insert(spirv[word + 2]);
        } else if (op == OpAtomicUMax && len == kAtomicUMaxWords &&
                   witness_pointers.count(spirv[word + 3])) {
            return true;
        }
        word += len;
    }
    return false;
}

namespace {

struct SpirvInst { uint32_t op; size_t at; uint32_t len; };

// ---------------------------------------------------------------------------------------------
// A shared SPIR-V word model. Following every trailing word as an SSA id treats LITERALS as ids,
// and a literal that happens to equal a result id fabricates a dataflow edge: measured, that made
// a vote "reach" an output through OpExtInst, whose word 4 is a GLSL instruction NUMBER in the
// same numeric range as low result ids. It produced 8 false positives across 49 real modules.
bool spirv_has_result(uint32_t op) {
    switch (op) {
        case Op_Store: case Op_Branch: case Op_BranchConditional:
        case Op_SelectionMerge: case Op_LoopMerge: case Op_Return: case Op_Kill:
        case Op_Decorate: case Op_MemberDecorate: case Op_Switch:
            return false;
        default: return true;
    }
}

// Type instructions carry their result at word 1 (they have no type operand).
uint32_t spirv_result_word(uint32_t op) {
    switch (op) {
        case Op_TypeVoid: case Op_TypeBool: case Op_TypeInt: case Op_TypeFloat:
        case Op_TypeVector: case 24 /*OpTypeMatrix*/: case Op_TypeImage:
        case 26 /*OpTypeSampler*/:
        case Op_TypeSampledImage: case Op_TypeArray: case Op_TypeRuntimeArray:
        case Op_TypeStruct: case Op_TypePointer: case Op_TypeFunction:
        case Op_ExtInstImport:
            return 1;
        default: return 2;
    }
}

bool spirv_is_id_operand(uint32_t op, uint32_t i) {
    switch (op) {
        case Op_ExtInst:        return i == 3 || i >= 5;          // set id, LITERAL instr, args
        case Op_Constant:       return false;                     // value words
        case Op_CompositeExtract: return i == 3;                  // then literal indices
        case 82 /*OpCompositeInsert*/: return i == 3 || i == 4;
        case 79 /*OpVectorShuffle*/: return i == 3 || i == 4;     // then literal components
        case Op_ImageSampleImplicitLod:
        case Op_ImageSampleExplicitLod:
        case Op_ImageFetch: case Op_ImageGather:
            return i == 3 || i == 4 || i >= 6;                    // word 5 is a LITERAL mask
        case Op_TypeInt: case Op_TypeFloat: return false;
        case Op_TypeVector: case 24 /*OpTypeMatrix*/: case Op_TypeImage: return i == 2;
        case Op_TypePointer:    return i == 3;                    // word 2 is LITERAL storage
        case Op_Variable:       return i >= 4;                    // word 3 is LITERAL storage
        default:                return i >= 3;
    }
}

// Values that differ between the invocations of one draw, and so cannot carry uniformity. Listing
// the DIVERGENT sources rather than the uniform ones is deliberate: an opcode nobody thought about
// then has to earn uniformity through its operands instead of inheriting it, so an unknown case
// fails towards "not proven" rather than towards "safe".
bool spirv_op_is_divergent_source(uint32_t op) {
    switch (op) {
        case Op_ImageSampleImplicitLod: case Op_ImageSampleExplicitLod:
        case Op_ImageSampleDrefImplicitLod: case Op_ImageSampleDrefExplicitLod:
        case Op_ImageFetch: case Op_ImageGather: case Op_ImageRead:
        case Op_ImageQuerySizeLod: case Op_ImageQuerySize: case Op_ImageQueryLod:
        case Op_ImageQueryLevels:
        case Op_DPdx: case Op_DPdy:
        case 209 /*OpFwidth*/: case 210 /*OpDPdxFine*/: case 211 /*OpDPdyFine*/:
        case 212 /*OpFwidthFine*/: case 213 /*OpDPdxCoarse*/: case 214 /*OpDPdyCoarse*/:
        case 215 /*OpFwidthCoarse*/:
        case Op_AtomicLoad: case Op_AtomicExchange: case Op_AtomicCompareExchange:
        case Op_AtomicIAdd: case Op_AtomicISub: case Op_AtomicSMin: case Op_AtomicUMin:
        case Op_AtomicSMax: case Op_AtomicUMax: case Op_AtomicAnd: case Op_AtomicOr:
        case Op_AtomicXor:
        case 57 /*OpFunctionCall*/:
            return true;
        default:
            // Every subgroup instruction: its result depends on which lanes share the group, which
            // is the very thing under question here.
            return op >= 333 && op <= 366;
    }
}

bool spirv_op_is_constant(uint32_t op) {
    switch (op) {
        case 41 /*OpConstantTrue*/: case 42 /*OpConstantFalse*/: case Op_Constant:
        case 44 /*OpConstantComposite*/: case 45 /*OpConstantSampler*/:
        case 46 /*OpConstantNull*/:
        case 48 /*OpSpecConstantTrue*/: case 49 /*OpSpecConstantFalse*/:
        case 50 /*OpSpecConstant*/: case 51 /*OpSpecConstantComposite*/:
        case 52 /*OpSpecConstantOp*/:
            return true;
        default: return false;
    }
}

// Atomics, named rather than taken as the numeric window `227..242`: the float and flag forms sit
// outside it. ONE predicate, because the observable check and the leaving-scan were two spellings of
// this set and had already drifted apart once.
bool spirv_op_is_atomic(uint32_t op) {
    return (op >= Op_AtomicLoad && op <= Op_AtomicXor) ||
           op == 318 /*OpAtomicFlagTestAndSet*/ || op == 319 /*OpAtomicFlagClear*/ ||
           op == 5614 /*OpAtomicFMinEXT*/ || op == 5615 /*OpAtomicFMaxEXT*/ ||
           op == 6035 /*OpAtomicFAddEXT*/;
}

// Every way one pointer is derived from another. Missing one splits a single slot into two, which
// loses the dependence between a store through one id and a load through another.
bool spirv_op_derives_pointer(uint32_t op) {
    return op == Op_AccessChain || op == 66 /*OpInBoundsAccessChain*/ ||
           op == 67 /*OpPtrAccessChain*/ || op == 70 /*OpInBoundsPtrAccessChain*/ ||
           op == 83 /*OpCopyObject*/;
}

bool spirv_storage_is_uniform(uint32_t sc, bool module_writes_memory) {
    if (sc == SC_StorageBuffer) return !module_writes_memory;   // a UAV this draw writes is not
    return sc == 2 /*Uniform*/ || sc == SC_PushConstant || sc == SC_UniformConstant;
}

}  // namespace

bool fragment_spirv_wave_width_independent(const std::vector<uint32_t>& spirv) {
    // Unparseable is not proven: a module we cannot read has not been shown to be width-independent,
    // and the whole point of this predicate is that its POSITIVE answer is load-bearing.
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return false;

    std::vector<SpirvInst> insts;
    for (size_t off = 5; off < spirv.size();) {
        const uint32_t len = spirv[off] >> 16;
        const uint32_t op = spirv[off] & 0xffffu;
        if (!len || len > spirv.size() - off) return false;   // truncated: not proven, see above
        insts.push_back({op, off, len});
        off += len;
    }

    // --- the votes ------------------------------------------------------------------------------
    std::vector<std::pair<uint32_t, uint32_t>> votes;   // (result, operand)
    std::unordered_set<uint32_t> ballots;               // results whose WIDTH is the answer
    for (const SpirvInst& in : insts) {
        if (in.op == Op_GroupNonUniformAny && in.len >= 5) {
            votes.push_back({spirv[in.at + 2], spirv[in.at + 4]});
        } else if (in.op == Op_GroupNonUniformBallot && in.len >= 5) {
            votes.push_back({spirv[in.at + 2], spirv[in.at + 4]});
            ballots.insert(spirv[in.at + 2]);
        }
    }
    if (votes.empty()) return true;                     // nothing width-dependent to prove

    // --- storage classes ------------------------------------------------------------------------
    std::unordered_map<uint32_t, uint32_t> var_storage;
    std::unordered_set<uint32_t> outputs, locals;
    for (const SpirvInst& in : insts) {
        if (in.op != Op_Variable || in.len < 4) continue;
        const uint32_t storage = spirv[in.at + 3];
        var_storage[spirv[in.at + 2]] = storage;
        if (storage == SC_Output) outputs.insert(spirv[in.at + 2]);
        else if (storage == SC_Function) locals.insert(spirv[in.at + 2]);
    }
    // Access chains inherit their base's class, transitively. Two passes over the instruction list
    // are not enough in general, so this runs to a fixed point.
    for (bool changed = true; changed;) {
        changed = false;
        for (const SpirvInst& in : insts) {
            if (!spirv_op_derives_pointer(in.op) || in.len < 4) continue;
            const uint32_t base = spirv[in.at + 3], result = spirv[in.at + 2];
            if (outputs.count(base) && outputs.insert(result).second) changed = true;
            if (locals.count(base) && locals.insert(result).second) changed = true;
        }
    }

    // The VARIABLE a pointer roots in. Taint through memory has to be keyed on the slot, not on the
    // pointer id: the emitter mints a fresh OpAccessChain per guest-scratch access, so a
    // read-modify-write of one slot goes through two different ids and a store/load pair keyed by id
    // drops the dependence between them entirely.
    std::unordered_map<uint32_t, uint32_t> pointer_root;
    for (const SpirvInst& in : insts)
        if (in.op == Op_Variable && in.len >= 3) pointer_root[spirv[in.at + 2]] = spirv[in.at + 2];
    // Every way one pointer is derived from another. OpCopyObject and the InBounds/Ptr chain forms
    // each split one slot into two if they are not followed.
    for (bool changed = true; changed;) {
        changed = false;
        for (const SpirvInst& in : insts) {
            if (!spirv_op_derives_pointer(in.op) || in.len < 4) continue;
            const auto base = pointer_root.find(spirv[in.at + 3]);
            if (base == pointer_root.end()) continue;
            if (pointer_root.emplace(spirv[in.at + 2], base->second).second) changed = true;
        }
    }
    // An UNKNOWN pointer must not root at itself. That was an identity fallback dressed as a
    // lookup: a pointer the map has never seen became its own slot, so a store through one id and a
    // load through another lost the dependence between them -- exactly the defect the root map was
    // added to fix, reintroduced by its own default. Unknown now means "could be any local", which
    // is conservative on both sides. (The asymmetry that hid this: an unrooted pointer was already
    // safe on the STORE side, because a pointer not in `locals` reads as an observable effect.)
    const auto pointer_is_known = [&](uint32_t ptr) { return pointer_root.count(ptr) != 0; };
    const auto root_of = [&](uint32_t ptr) {
        const auto it = pointer_root.find(ptr);
        return it == pointer_root.end() ? ptr : it->second;
    };

    // What counts as leaving the shader. A store through anything that is not a Function-storage
    // pointer is observable -- a colour attachment, a storage buffer, an image. Restricting this to
    // Output would make a UAV write invisible to the analysis, which is the same shape of blind spot
    // as the control-dependence one, one level down.
    const auto is_observable_effect = [&](const SpirvInst& in) {
        switch (in.op) {
            case Op_Store:
                return in.len >= 3 && !locals.count(spirv[in.at + 1]);
            case 63 /*OpCopyMemory*/:
                return in.len >= 3 && !locals.count(spirv[in.at + 1]);
            case 64 /*OpCopyMemorySized*/:
                return in.len >= 4 && !locals.count(spirv[in.at + 1]);
            case Op_ImageWrite: case Op_Kill:
            case 4416 /*OpTerminateInvocation*/: case 5380 /*OpDemoteToHelperInvocation*/:
            case Op_ControlBarrier: case Op_MemoryBarrier:
                return true;
            // Atomics both read and write memory the rest of the draw can see. Named explicitly
            // rather than as the numeric window `227..242`: the float and flag atomics sit outside
            // it, so a vote-guarded OpAtomicFAddEXT admitted while the byte-identical module using
            // OpAtomicIAdd refused. One opcode number was the whole difference.
            default:
                return spirv_op_is_atomic(in.op);
        }
    };

    // --- arm (a): the vote's operand is wave-uniform ---------------------------------------------
    // Any(P) reduces P over whatever lanes the group holds. When P takes the same value in every
    // invocation of the draw, that reduction IS P -- at 64 lanes, at 32, at any width. So a vote
    // over a provably uniform operand answers identically however the hardware groups lanes, and
    // nothing downstream of it can move. This is the arm that carries the corpus: the guest's
    // `s_cbranch_execz`-style tests are overwhelmingly comparisons of constant-buffer scalars.
    //
    // One qualification, and it is the difference between a predicate that happens to be right on
    // the shaders in hand and one whose argument holds: "uniform" must mean the same value in EVERY
    // invocation of this draw. A read-only constant buffer satisfies that. A storage buffer this
    // shader also WRITES does not -- one invocation's store changes what another's load returns --
    // so when the module writes buffer or image memory at all, StorageBuffer loads stop counting.
    // Measured, 0 of the 49 corpus modules write any, so the guard costs nothing today.
    // Defined in terms of is_observable_effect rather than as its own list. These are the same
    // question at two scopes -- "can the rest of the draw see this?" versus "can this shader read it
    // back?" -- and keeping them as separate lists let them drift twice: this one kept the
    // `227..242` atomic window after the other moved to a named set, and never learned about
    // OpCopyMemory at all. A write to memory is an observable effect that is not a store to a colour
    // output, which nothing reads back.
    bool writes_memory = false;
    for (const SpirvInst& in : insts) {
        if (!is_observable_effect(in)) continue;
        // Observable and yet writes NOTHING a later load could read back: ending the invocation,
        // or ordering other threads' accesses. Measured -- folding these in cost Evergate 10 of its
        // 11 provable modules, because a module containing a barrier stopped being able to treat
        // any constant buffer as uniform.
        switch (in.op) {
            case Op_Kill: case 4416 /*OpTerminateInvocation*/:
            case 5380 /*OpDemoteToHelperInvocation*/:
            case Op_ControlBarrier: case Op_MemoryBarrier:
                continue;
            default: break;
        }
        // A store to a colour output is observable but is not memory this shader reads back.
        if ((in.op == Op_Store || in.op == 63 /*OpCopyMemory*/ || in.op == 64) &&
            in.len >= 2 && outputs.count(spirv[in.at + 1]))
            continue;
        writes_memory = true;
        break;
    }
    // Pointers that root in a read-only uniform storage class, and the access chains over them.
    // Kept SEPARATE from the value set on purpose: a pointer's uniformity says something about
    // where it points, and a load's uniformity is about what is stored there. Conflating the two
    // made a Function local with a uniform initializer -- an initializer is an id operand, so the
    // generic rule marked the variable uniform -- lend that uniformity to values written into it
    // per lane. Constructed by hand; the corpus has no instance, which is exactly why.
    std::unordered_set<uint32_t> uniform_ptrs;
    for (const SpirvInst& in : insts)
        if (in.op == Op_Variable && in.len >= 4 &&
            spirv_storage_is_uniform(spirv[in.at + 3], writes_memory))
            uniform_ptrs.insert(spirv[in.at + 2]);

    std::unordered_set<uint32_t> uniform;
    for (const SpirvInst& in : insts)
        if (spirv_op_is_constant(in.op) && in.len >= 3) uniform.insert(spirv[in.at + 2]);
    for (bool changed = true; changed;) {
        changed = false;
        for (const SpirvInst& in : insts) {
            if (!spirv_has_result(in.op)) continue;
            const uint32_t rw = spirv_result_word(in.op);
            if (in.len <= rw) continue;
            const uint32_t result = spirv[in.at + rw];
            if (uniform.count(result)) continue;
            if (in.op == Op_GroupNonUniformAny) {
                // A vote's own result is uniform once its operand is: an all-lanes reduction of a
                // value every lane already agrees on. Chained votes resolve by fixed point.
                if (in.len >= 5 && uniform.count(spirv[in.at + 4])) {
                    uniform.insert(result);
                    changed = true;
                }
                continue;
            }
            if (spirv_op_is_divergent_source(in.op)) continue;
            if (in.op == Op_Load) {
                // ONLY from a pointer that roots in a read-only uniform class. Never from "this
                // pointer value is uniform": a Function local's pointer is perfectly uniform and
                // what it holds is whatever was last stored into it, which may be per lane.
                if (in.len >= 4 && uniform_ptrs.count(spirv[in.at + 3])) {
                    uniform.insert(result);
                    changed = true;
                }
                continue;
            }
            if (in.op == Op_AccessChain) {
                // A chain inherits its base's class, and is a uniform POINTER only when every
                // index is uniform too -- a per-lane index selects a different element per lane.
                if (in.len >= 4 && uniform_ptrs.count(spirv[in.at + 3]) &&
                    !uniform_ptrs.count(result)) {
                    bool indices_uniform = true;
                    for (uint32_t i = 4; i < in.len; ++i)
                        if (!uniform.count(spirv[in.at + i])) { indices_uniform = false; break; }
                    if (indices_uniform) { uniform_ptrs.insert(result); changed = true; }
                }
                continue;
            }
            if (in.op == Op_Variable) continue;   // a pointer, never a value; see uniform_ptrs
            if (in.op == Op_Phi) {
                // IDENTICAL incoming values, not merely uniform ones. A phi's whole job is to
                // choose between its edges, so "every value is uniform" says nothing about the
                // result when the branch doing the choosing is divergent:
                //
                //     p = divergent;  if (p) {} else {};  q = phi(true, false);  a = Any(q)
                //
                // both constants are uniform, and q is exactly p. Admitting that vote lets a
                // 64-lane wave and two 32-lane groups answer differently, which is the defect this
                // predicate exists to prevent. Identical ids are safe for the opposite reason:
                // which edge ran cannot matter when every edge carries the same value.
                //
                // Proving the SELECTION uniform would admit more, and is left undone rather than
                // guessed: measured over the 49-module corpus this restriction costs nothing.
                bool all = in.len > 4;
                for (uint32_t i = 5; i + 1 < in.len; i += 2)
                    if (spirv[in.at + i] != spirv[in.at + 3]) { all = false; break; }
                if (all && uniform.count(spirv[in.at + 3])) {
                    uniform.insert(result);
                    changed = true;
                }
                continue;
            }
            bool any_operand = false, all_uniform = true;
            for (uint32_t i = 1; i < in.len; ++i) {
                if (i == rw || !spirv_is_id_operand(in.op, i)) continue;
                any_operand = true;
                if (!uniform.count(spirv[in.at + i])) { all_uniform = false; break; }
            }
            if (any_operand && all_uniform) { uniform.insert(result); changed = true; }
        }
    }

    // --- the CFG, for arm (b) -------------------------------------------------------------------
    std::unordered_map<uint32_t, std::vector<SpirvInst>> blocks;
    std::unordered_map<uint32_t, std::vector<uint32_t>> succ;
    std::unordered_map<uint32_t, uint32_t> merge_of, branch_cond;
    uint32_t current = 0;
    for (const SpirvInst& in : insts) {
        if (in.op == Op_Label && in.len >= 2) {
            current = spirv[in.at + 1];
            blocks[current];
            continue;
        }
        if (!current) continue;
        blocks[current].push_back(in);
        if ((in.op == Op_SelectionMerge || in.op == Op_LoopMerge) && in.len >= 2) {
            merge_of[current] = spirv[in.at + 1];
        } else if (in.op == Op_Branch && in.len >= 2) {
            succ[current] = {spirv[in.at + 1]};
        } else if (in.op == Op_BranchConditional && in.len >= 4) {
            branch_cond[current] = spirv[in.at + 1];
            succ[current] = {spirv[in.at + 2], spirv[in.at + 3]};
        } else if (in.op == Op_Switch && in.len >= 3) {
            // The SELECTOR is a branch condition too. Modelling only the successors left a
            // vote-tainted switch with no control dependence at all, and the CFG emitter really
            // does build one -- on the guest PC (rdna2_emit_cfg.cpp's dispatcher).
            branch_cond[current] = spirv[in.at + 1];
            std::vector<uint32_t> targets{spirv[in.at + 2]};
            for (uint32_t i = 4; i < in.len; i += 2) targets.push_back(spirv[in.at + i]);
            succ[current] = targets;
        }
    }

    // --- arm (b): the vote cannot influence a colour output --------------------------------------
    // Data flow alone is NOT enough, and believing it was is the defect this replaced. A vote used
    // only as a branch condition never flows into a store, yet it decides WHICH store runs: with
    // EXEC true in the upper half and false in the lower, a 64-lane any() takes the branch where two
    // independent 32-lane groups disagree, and the constant the taken block writes lands in one half
    // only. Measured over the same 49 modules, the value-reachability question clears 49 of 49 while
    // this one clears 1 -- the earlier predicate was not incomplete at the margin, it was blind to
    // the whole population.
    const auto influences_output = [&](uint32_t seed) {
        std::unordered_set<uint32_t> tainted{seed}, tainted_ptrs;
        // A bounded fixed point, and the bound is conservative on purpose: falling out of the loop
        // still growing means the closure was NOT complete, so the honest answer is "it might" --
        // returning false there would admit a module on an unfinished analysis.
        bool settled = false;
        for (int round = 0; round < 32 && !settled; ++round) {
            for (bool changed = true; changed;) {           // data flow to a fixed point
                changed = false;
                for (const SpirvInst& in : insts) {
                    if (in.op == Op_Store && in.len >= 3) {
                        const uint32_t raw = spirv[in.at + 1], val = spirv[in.at + 2];
                        if (tainted.count(val)) {
                            if (!pointer_is_known(raw)) {          // unknown: could be any of them
                                for (uint32_t local : locals)
                                    if (tainted_ptrs.insert(local).second) changed = true;
                            } else if (locals.count(root_of(raw)) &&
                                       tainted_ptrs.insert(root_of(raw)).second) {
                                changed = true;
                            }
                        }
                        continue;
                    }
                    if ((in.op == 63 /*OpCopyMemory*/ || in.op == 64 /*OpCopyMemorySized*/) &&
                        in.len >= 3) {
                        // Target <- Source, both pointers. A tainted source slot makes the target
                        // slot tainted; no OpLoad/OpStore pair appears, which is why the closure
                        // could not see it.
                        const uint32_t dst = spirv[in.at + 1], src = spirv[in.at + 2];
                        const bool src_tainted = pointer_is_known(src)
                            ? tainted_ptrs.count(root_of(src)) != 0
                            : !tainted_ptrs.empty();
                        if (src_tainted) {
                            if (!pointer_is_known(dst)) {
                                for (uint32_t local : locals)
                                    if (tainted_ptrs.insert(local).second) changed = true;
                            } else if (locals.count(root_of(dst)) &&
                                       tainted_ptrs.insert(root_of(dst)).second) {
                                changed = true;
                            }
                        }
                        continue;
                    }
                    if (in.op == Op_Load && in.len >= 4) {
                        const uint32_t raw = spirv[in.at + 3];
                        const bool reads_tainted = pointer_is_known(raw)
                            ? tainted_ptrs.count(root_of(raw)) != 0
                            : !tainted_ptrs.empty();               // unknown: could be any of them
                        if (reads_tainted && tainted.insert(spirv[in.at + 2]).second)
                            changed = true;
                        continue;
                    }
                    if (!spirv_has_result(in.op)) continue;
                    const uint32_t rw = spirv_result_word(in.op);
                    if (in.len <= rw) continue;
                    const uint32_t result = spirv[in.at + rw];
                    if (tainted.count(result)) continue;
                    for (uint32_t i = 1; i < in.len; ++i) {
                        if (i == rw || !spirv_is_id_operand(in.op, i)) continue;
                        if (!tainted.count(spirv[in.at + i])) continue;
                        tainted.insert(result);
                        changed = true;
                        break;
                    }
                }
            }
            for (const SpirvInst& in : insts) {              // a tainted value leaving the shader
                if (in.op == Op_Store && in.len >= 3 && !locals.count(spirv[in.at + 1]) &&
                    tainted.count(spirv[in.at + 2]))
                    return true;
                if (in.op == Op_ImageWrite && in.len >= 4 &&
                    (tainted.count(spirv[in.at + 2]) || tainted.count(spirv[in.at + 3])))
                    return true;
                // EVERY id operand, not word 3. Word 3 is an atomic's POINTER; the value it
                // writes is further along and its index differs by opcode, so checking one word
                // let a vote-derived value be atomically written to a UAV in straight-line code --
                // the scan fired and read the wrong word. Testing all of them needs no per-opcode
                // value index and cannot drift as opcodes are added.
                if (spirv_op_is_atomic(in.op)) {
                    for (uint32_t i = 3; i < in.len; ++i)
                        if (tainted.count(spirv[in.at + i])) return true;
                }
                // OpCopyMemory moves a value without an OpLoad/OpStore pair, so a tainted local
                // copied straight out was invisible here even though is_observable_effect knew the
                // opcode. Source is word 2; a tainted SOURCE leaving through a non-local target is
                // the same event as a tainted store.
                if ((in.op == 63 /*OpCopyMemory*/ || in.op == 64 /*OpCopyMemorySized*/) &&
                    in.len >= 3 && !locals.count(spirv[in.at + 1])) {
                    // The SOURCE is a POINTER, so its taint lives in tainted_ptrs -- a slot is
                    // tainted, not a value id. Testing `tainted` here made the arm inert, which is
                    // what its own fixture caught.
                    const uint32_t src = spirv[in.at + 2];
                    if (pointer_is_known(src) ? tainted_ptrs.count(root_of(src)) != 0
                                              : !tainted_ptrs.empty())
                        return true;
                }
            }

            bool grew = false;                               // then control dependence
            for (const auto& entry : branch_cond) {
                if (!tainted.count(entry.second)) continue;
                const uint32_t head = entry.first;
                const auto merge_it = merge_of.find(head);
                if (merge_it == merge_of.end())
                    return true;   // unstructured: the region has no bound we can trust
                const uint32_t merge = merge_it->second;
                // Every block strictly inside the construct executes only because the vote said so.
                std::unordered_set<uint32_t> inside;
                std::vector<uint32_t> stack;
                if (succ.count(head)) stack = succ[head];
                while (!stack.empty()) {
                    const uint32_t b = stack.back();
                    stack.pop_back();
                    if (b == merge || !blocks.count(b) || !inside.insert(b).second) continue;
                    if (succ.count(b))
                        for (uint32_t s : succ[b]) stack.push_back(s);
                }
                for (uint32_t b : inside) {
                    for (const SpirvInst& in : blocks[b]) {
                        if (is_observable_effect(in)) return true;
                        // A store into a local inside the region is control-dependent too: whether
                        // it happened at all is the vote's answer, so every later load of that
                        // local carries the vote.
                        if (in.op == Op_Store && in.len >= 3) {
                            const uint32_t raw = spirv[in.at + 1];
                            if (!pointer_is_known(raw)) {
                                for (uint32_t local : locals)
                                    if (tainted_ptrs.insert(local).second) grew = true;
                            } else if (locals.count(root_of(raw)) &&
                                       tainted_ptrs.insert(root_of(raw)).second) {
                                grew = true;
                            }
                        }
                    }
                }
                const auto merge_blk = blocks.find(merge);
                if (merge_blk == blocks.end()) continue;
                for (const SpirvInst& in : merge_blk->second) {
                    if (in.op != Op_Phi || in.len < 5) continue;
                    const uint32_t result = spirv[in.at + 2];
                    if (tainted.count(result)) continue;
                    bool differs = false;
                    for (uint32_t i = 5; i + 1 < in.len; i += 2)
                        if (spirv[in.at + i] != spirv[in.at + 3]) { differs = true; break; }
                    if (differs) { tainted.insert(result); grew = true; }
                }
            }
            settled = !grew;
        }
        return !settled;
    };

    for (const auto& vote : votes) {
        // A ballot is never cleared by uniformity: its result is a per-lane bit MASK whose width is
        // the subgroup's, so a 32-lane ballot reports half a mask as though it were whole however
        // uniform the value being balloted. Only deadness can clear one.
        if (!ballots.count(vote.first) && uniform.count(vote.second)) continue;
        if (!influences_output(vote.first)) continue;
        return false;
    }
    return true;
}

namespace {

// A no-GS NGG program is split into two machine-code allocations by the guest compiler: the
// logical vertex producer writes one compact per-vertex LDS record, then a compiler-generated NGG
// wrapper culls/compacts primitives and exports fields from that record.  Vulkan's vertex stage
// already launches exactly the logical draw vertices and performs primitive assembly itself.  When
// both sides of this ABI can be proven from the machine code, execute only the producer and export
// the same LDS fields directly.  This avoids pretending that Function-private LDS can communicate
// between independent Vulkan vertex invocations.

} // namespace

} // namespace prosper::gpu
