// Reads at arbitrary-lane addresses (#5193, decision on the issue). Which memory reads may hand
// ARBITRARY result lanes on (ScalarSourceMarks::arbitrary), the LDS index bound that makes a plain
// LDS read one of them, and the loop roots an address needs DEFINED rather than merely free of hard
// lanes (consumed by rdna2_loop_arbitrary_carry.cpp). Compiler facts only: nothing here grants
// resource, entry-register or memory authority.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <set>

namespace prosper::gpu {

struct IndirectPointerRelocationProof;
struct RegState;
struct ScalarSourceMarks;
struct SpirvCompute;

// The bounds-checked, side-effect-free reads whose out-of-range result prosper reproduces, so an
// unconstrained address yields a value hardware could have loaded: ordinary buffer and typed-buffer
// loads and image reads (robustBufferAccess / robustImageAccess, rdna2_to_spirv_internal.hpp), and
// the plain per-lane LDS reads, whose index lds_index_inside_if_noted() bounds. Stores, atomics,
// GDS, LDS-mode buffer loads, FLAT, ADDTID and the cross-lane LDS forms are not among them.
bool arbitrary_read_form(const Rdna2Inst& in);
// Condition 1 of the decision: no descriptor word (V#, T#, the S# of a sampling form) and no
// scalar address input of `in` may be fabricated. LDS reads have none.
bool arbitrary_read_resource_genuine(const RegState& rs, const Rdna2Inst& in);
// The CFG dispatcher's form of the same resource condition, on its MUST scalar words and MAY
// relays (rdna2_spilled_mask_halves.cpp): "same hardware, same answer" (#5039).
// `relocation` is the builder's indirect-pointer proof, when it has one.
bool dispatcher_arbitrary_read(const Rdna2Inst& in, const std::set<int>& scalar_words,
                               const std::set<int>& relayed, bool definite_scc,
                               const IndirectPointerRelocationProof* relocation);
// An image read whose actual address words (the MIMG source proof) hold no hard lane, through a
// genuine resource: its active lanes are defined or arbitrary.
bool mimg_result_may_be_arbitrary(const RegState& rs, const Rdna2Inst& in);
// Emission hook, called once per instruction before its lowering with the instruction's input
// marks (snapshot_saved_b64_masks), which its mark transfer then applies. Decides whether this LDS
// read bounds its index, records the loop roots an address read relies on being defined
// (SpirvCompute::loop_blanket_roots_strong), and admits a relocated GLOBAL read -- bounded by its
// relocation proof, which only the builder holds -- at an arbitrary address.
void note_memory_address(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                         ScalarSourceMarks& source);
// The in-range predicate for one LDS dword index when the current read was noted as bounded, else
// 0. SpirvCompute::lds_load selects index 0 and the result zero outside it.
uint32_t lds_index_inside_if_noted(SpirvCompute& b, uint32_t index);

}   // namespace prosper::gpu
