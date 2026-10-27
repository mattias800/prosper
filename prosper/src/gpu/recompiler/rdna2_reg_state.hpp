// Register state and its mark helpers; private to the recompiler.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_portable_mask_slots.hpp"
#include "gpu/recompiler/rdna2_native_image_results.hpp"
#include <cstdint>
#include <memory>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
namespace prosper::gpu {

// Machine state during recompilation: the VGPR and SGPR files (VGPR/SGPR number -> current SSA bits
// id) and VCC (current bool condition). VGPRs and SGPRs are separate register files; VALU/EXP source
// operands may reference either (SGPR is a valid ALU operand), so both are resolved by operand_bits.
// True when the scalar register currently holds a value prosper knows is NOT wave-uniform (#3596).
// Defined beside RegState so every uniformity predicate can ask the same question the same way; the
// lookup is on the register's CURRENT SSA value, so an overwrite clears it without bookkeeping.
bool scalar_is_lane_local(const struct RegState& rs, int sgpr);

// Seed RegState::smem_pointer_loads from the decoded stream (see proven_smem_pointer_loads in
// rdna2_emit_cfg.cpp). Every emission entry must call this: the proof used to be seeded only inside
// the CFG dispatcher, so a shader emitted straight-line carried an empty set and its SRT pointer
// loads reached the constant-buffer path and rejected the shader (#3616).
struct RegState;
void seed_smem_pointer_provenance(RegState& rs, const std::vector<Rdna2Inst>& ins);

struct PlaceholderOriginProgram;

struct RegState {
    // Original launch/dependency inventory; admission consumes its typed roots, never log text.
    mutable std::shared_ptr<const PlaceholderOriginProgram> placeholder_origin_program;
    PortableMaskHalfTypes portable_mask_half_types;
    std::unordered_map<int, uint32_t> vreg, sreg;
    VectorWordMarks vreg_source_marks;
    MimgSourceProof mimg_sources;
    NativeInterpolationPublication native_interpolation_publication;
    bool vector_full_exec =
        false;   // provenance fact; dispatcher runtime predication stays enabled
    bool vector_native_interpolation = false;
    uint32_t vector_wave_size = 64;
    // The latest VCC SSA value proved identical in every guest lane. A fragment VCCZ branch over
    // that exact value is already scalar and needs no subgroup vote. Tying the proof to the SSA id
    // makes an overwrite or control-flow merge invalidate it automatically.
    uint32_t vcc_wave_uniform = 0;
    // SSA values that are NOT wave-uniform despite living in a scalar register (#3596).
    //
    // prosper models one guest lane per invocation and has no cross-lane reduction, so
    // `v_readfirstlane_b32` is lowered as THIS lane's source value -- it is annotated
    // SPECULATIVE(confidence: med) at that site for exactly this reason. The result lands in
    // `sreg`, and every uniformity predicate downstream treats "it is in a scalar register" as
    // "it broadcasts one value to the whole wave". For a readfirstlane result that is false.
    //
    // It matters because a fragment `s_cbranch_vccz` whose VCC derives from such a value would
    // otherwise pass the wave-uniformity proof and SKIP its wave vote, branching on the lane's own
    // bit -- wrong pixels, no reject, no diagnostic.
    //
    // Keyed on the SSA VALUE rather than the register number, which makes copy propagation free:
    // `s_mov_b32` lowers as `d = a`, so a tainted value stays tainted through any number of scalar
    // moves, and an overwrite installs a fresh id that is naturally clean.
    //
    // Numeric S_FF1_I32_B32 also carries known source taint to its fresh result (#4060). General
    // ALU propagation remains incomplete, and the gap is wider than a round-trip through a
    // dispatcher variable: `s_add_u32 s3, s0, s5` launders taint by producing a clean fresh id,
    // and so does an `s_cselect_b32` on SCC. SCC itself
    // is a third route -- a compare on tainted operands sets SCC, and SCC read as a VOPC source
    // returns from `scalar_data_operand` one line BEFORE the lane-local check. So this closes the
    // direct and copy-laundered shapes, not the class. #3606 carries the propagation work.
    std::unordered_set<uint32_t> lane_local_scalars;
    // Before the first compact structured construct, map presence means that the scalar value
    // reaches this exact linear path. Branch/loop PHIs can synthesize zero for an absent SGPR and
    // clear this marker. The CFG dispatcher likewise allocates Function variables for every
    // statically observed SGPR and may load a zero placeholder where no scalar lifetime reaches the
    // block. Keep that distinction explicit so a narrow prefix proof cannot weaken either route.
    bool scalar_presence_has_no_placeholders = true;
    // The per-register form of that fact (#4706). An SGPR is here when its current value may be
    // the structured emitter's fabricated zero on some path: an if-merge or loop-header phi took
    // `sget()`'s `uconst(0)` from an edge where the register had no value, or the value was computed
    // from such a register (including through SCC, a destination's own old bits, or a source
    // that was never written). A write whose every input is definite clears the mark.
    // `lane_slot_merge_placeholder` carries the same mark through a v_writelane/v_readlane spill.
    // In the CFG dispatcher, `load_state` drops every SGPR word the Wave64 MUST analysis cannot
    // prove, which covers SGPRs; spill slots carry no such fact, so it marks every reloaded data
    // slot instead. Consumers that turn scalar DATA into lane bits consult the marks (mask()).
    std::set<int> sreg_merge_placeholder;
    // Which marks are ONLY the loop header's blanket mark (mark_loop_carried marks every loop-carried
    // SGPR before the body exists) on a word that was DEFINED on the preheader edge, or derived only
    // from such words (a write from them, or a merge where every marked edge is blanket-only). Any
    // other cause erases the entry. The lane-bit guard accepts these: refusing them made a
    // VCC-as-scratch write from a loop counter drop VCC's lane view, and a loop exit that could not
    // name VCC (Kena's per-cone loop). The assumption that the back-edge value is defined too is
    // checked at the back edge (LoopBlanketCheck), which refuses the program when a word read as
    // lane bits on it was fabricated there (#4714, #4749 review). Each entry names the assumptions
    // (blanket_root: header pc and carried register) the word depends on.
    std::map<int, std::set<uint64_t>> sreg_loop_blanket;
    // A READFIRSTLANE/SALU generation's existing active-domain view. The global marks above
    // remain authoritative outside it; scalar writes and control-flow joins expire the view.
    struct ScalarExecScope {
        uint32_t value = 0;
        ScalarSourceMarks source;
    };
    std::map<int, ScalarExecScope> sreg_exec_scopes;
    std::set<std::pair<int, int>> lane_slot_merge_placeholder;
    // Whether a DATA read of an ordinary SGPR that holds nothing (absent from `sreg` and
    // `sreg_input`, no mask covering it) is the fabricated zero `operand_bits` reads it as. True in
    // every emitting shell: compute seeds its whole launch state, so absence is "never written";
    // fragment models no user data, so its absent words read as 0 and count as never-written
    // (#4725). recompile_coverage() clears it: its census seeds no launch state at all and discards
    // its code, so a stage's launch word there (Messenger's VS `s_and_b32 vcc_hi, s3, ...`) is not
    // a fabricated word (#4714 review). Merge edges ignore it: `sget()` always supplies uconst(0).
    bool absent_sgpr_reads_fabricated_zero = true;
    // Same propagation, second fact: the SGPR's bits came from MEMORY (an SMEM load, or scalar ALU
    // over one). Such a word is not a ballot of this wave, so projecting it onto host lanes would
    // select different pixels than the PS5 does whenever the pattern is not uniform.
    std::set<int> sreg_memory_pattern;
    std::set<std::pair<int, int>> lane_slot_memory_pattern;
    // SCC computed from a marked word (or poisoned, or a loop/merge phi of either): a later
    // s_cselect/s_cmov/s_addc/s_subb carries the mark into its destination even with clean
    // explicit sources.
    bool scc_merge_placeholder = false;
    // Scoped Wave64 fragment scalar-VCC admission needs an actual SCC producer, just as Wave32
    // does. Other Wave64 paths retain their established entry/condition policy.
    bool track_scc_definition = false;
    std::set<uint64_t> scc_loop_blanket;
    int max_vgpr = 0;   // highest statically referenced VGPR in this shader
    // Immutable raw user-data words for DIRECT descriptors. They stay absent from `sreg` so
    // descriptor provenance can still distinguish driver input from a shader overwrite, while
    // scalar ALU may read their real bits (Astro copies V#.word2 into M0 as an LDS base).
    std::unordered_map<int, uint32_t> sreg_input;
    // SGPRs definitely written by shader instructions along at least one path to this point. This
    // provenance is independent of `sreg`: an unrepresentable write deliberately erases its SSA
    // value, but must still invalidate an entry-time direct descriptor stored in that register.
    std::unordered_set<int> sreg_written;
    // At a Wave64 CFG terminal, these scalar words have a MUST reaching definition on every
    // incoming path. Unlike sreg presence, this excludes Function-variable zero placeholders.
    // Only the terminal snapshot sets this; it is consumed at the next barrier phase's entry.
    std::unordered_set<int> terminal_wave64_scalar_words;
    // Scalar registers currently holding the ENTRY value of M0 -- the value the driver left before
    // this shader wrote M0 -- as an OPAQUE token rather than as data (#3133). Membership is not a
    // value: the register has no `sreg` entry, so every ordinary consumer still rejects it exactly
    // as an untracked read does. Only `s_mov_b32 m0, sSRC` may consume it, and consuming it returns
    // M0 to untracked. That is what makes the containment structural instead of a promise: there is
    // no fabricated number anywhere for arithmetic, a V_WRITELANE source, or an LDS index to pick up.
    std::unordered_set<int> sreg_entry_m0;
    // Native saved masks must retain DATA before participants leave. An absent untouched word
    // after a physical overwrite is not the generic uninitialized-SGPR zero.
    bool native_saved_mask_words = false;
    std::set<int> unavailable_saved_mask_words;
    std::unordered_map<int, uint32_t>
        sreg_bool;   // SGPR (pairs) holding a saved per-lane mask (bool id)
    std::unordered_map<int, bool>
        sreg_bool_narrowed;   // was EXEC narrowed when this mask was saved? (restores it)
    // One restored physical dword of a Wave64 mask is not a complete B64 predicate. Keep its Bool
    // value out of sreg_bool so an overlapping S_MOV_B64/logical cannot ignore the sibling SGPR.
    // Only exact CFG-proven V_WRITELANE aliases consume this domain; ordinary scalar consumers use
    // the separately materialized ballot dword in sreg.
    std::unordered_map<int, uint32_t> sreg_wave64_mask_half;
    std::unordered_map<int, uint32_t> sreg_wave64_mask_half_index;
    // Wave32 saves occupy exactly one physical SGPR. Track those lifetimes separately so a later
    // scalar-data write can invalidate the bool without also clobbering an unrelated neighbor.
    std::unordered_set<int> sreg_bool_b32;
    std::unordered_map<int, uint32_t>
        sreg_srt;   // SGPR holding a descriptor -> its user_data/SRT byte offset
    // (descriptor provenance: s_load_dwordx4 tags, s_buffer_load resolves)
    // The DIRECT counterpart of sreg_srt: SGPR word -> the entry-time SGPR word it is an unmodified
    // copy of. An indirect descriptor keeps its provenance across a copy because sreg_srt travels
    // with it; a direct descriptor is keyed only by its REGISTER NUMBER (`by_sgpr_base`), so staging
    // it into a scratch range with s_mov_b32 used to destroy its only key -- the moves make
    // `sreg_range_written` true, which is exactly the guard that suppresses the direct lookup. The
    // compiler idiom that does this is "stage several descriptors, copy the selected one into one
    // SRSRC range" (#1773; #273 added sreg_srt propagation because the INDIRECT form of the same
    // idiom broke DOLL's scene VS).
    //
    // This is a copy alias, not a claim about the descriptor's contents: it says only "these bits
    // are still the bits the driver put in that entry-time register".
    //
    // The load-bearing condition is applied where the alias is ESTABLISHED -- the source must still
    // have been entry-time user data at the moment of the copy. It is deliberately NOT re-checked at
    // consumption: a copy captures bits, so a later write to the SOURCE cannot change what the
    // DESTINATION holds, and re-checking would decline the exact shape #1773 documents (the shader
    // reuses five of the origin words for its second descriptor before the sample). The alias dies
    // when the DESTINATION is written, which record_scalar_write does centrally.
    //
    // Because it is a claim about one register's bits, it is a PER-PATH fact and needs a meet at
    // every control-flow join -- see merge_ud_alias. An alias that held on only one incoming edge
    // would bind a descriptor the other edge never assembled.
    std::unordered_map<int, int> sreg_ud_alias;
    // Immediate S_LOAD_DWORDX16 is typeless: it can be ordinary scalar data, or two adjacent
    // eight-dword T# descriptors. The latter is admitted only after a whole-stream use proof (see
    // proven_smem_x16_descriptor_loads). These are load PCs, not SRT keys: an x16 bundle contains
    // two resources at one byte offset, so every consumer must retain exact-PC provenance.
    std::unordered_set<uint32_t> smem_x16_descriptor_loads;
    bool smem_x16_descriptor_analysis_done = false;
    // Immediate S_LOAD_DWORDX2 that loads a DESCRIPTOR-TABLE POINTER rather than data: the shader
    // chases an SRT pointer to reach the table the real V#/T#/S# lives in. Admitted by a
    // whole-stream use proof (proven_smem_pointer_loads); load PCs, not SRT keys.
    std::unordered_set<uint32_t> smem_pointer_loads;
    // Raw x2 loads with direct scalar observations. Their exact-PC resource must not turn an
    // unrelated unresolved scalar load into a binding-2 fallback success.
    std::unordered_set<uint32_t> smem_raw_x2_data_loads;
    std::unordered_set<uint32_t> smem_raw_immediate_wide_data_loads;
    std::unordered_set<uint32_t> smem_raw_register_wide_data_loads;
    std::unordered_set<uint32_t> smem_raw_offset_scalar_source_pcs;
    std::unordered_set<uint32_t> smem_raw_owned_wide_data_loads;
    std::unordered_set<uint32_t> smem_raw_nested_wide_data_loads;
    std::vector<RawNestedWideChain> smem_owned_nested_wide_chains;
    std::vector<RawNestedWideChain> smem_owned_raw_x2_chains;
    std::unordered_set<uint32_t> smem_raw_wide_data_loads;
    bool smem_pointer_analysis_done = false;
    // Register-offset S_LOAD_DWORDX2 is likewise typeless. GTA V uses it to fetch the first two
    // words of a V#, then replaces/fills the remaining words before an exact-PC buffer consumer.
    // Only PCs certified by the whole-CFG descriptor-use proof may substitute placeholders.
    std::unordered_set<uint32_t> smem_x2_descriptor_fragment_loads;
    bool smem_x2_descriptor_fragment_analysis_done = false;
    uint32_t vcc = 0;
    uint32_t scc =
        0;   // scalar condition code (bool); set by s_cmp_*/SCC-writing SOP2, read by s_cselect
    // #2418: does this shader read SCC anywhere? A STATIC property of the decoded stream, computed
    // once before emission, never mutated during it — so it carries none of the lifetime hazards a
    // stateful "pending reduction" marker would. It exists so the fragment stage can re-arm SCC after
    // a mask op (which needs an exact wave vote, and so forces wave64) ONLY for shaders that actually
    // consume it. Re-arming unconditionally is correct but makes every shader that merely saves EXEC
    // pay a subgroup-size requirement it does not need, which gates the draw on a 32-wide device.
    bool reads_scc = false;
    uint32_t exec =
        0;   // per-lane execution mask (bool); v_cmpx narrows it, output store honors it
    bool exec_narrowed =
        false;   // true once EXEC is narrowed below all-lanes-on (so VGPR writes predicate)
    // Lane-set facts about EXEC SSA values: child -> superset it was derived from (v_cmpx computes
    // EXEC = EXEC & cmp). Each entry is a fact about one immutable SSA value, so it holds on every
    // path that reaches that value.
    std::map<uint32_t, uint32_t> exec_subset_of;
    // Complement facts about EXEC SSA values: child -> {whole, part} when the child was computed as
    // whole & ~part and part was already proved a subset of whole, so part | child == whole. An
    // if/else lowering's second arm runs under such a child of the saved outer mask.
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> exec_complement_in;
    // PC-relative EMBEDDED TABLES (#273): load pc -> the table's dwords, resolved by
    // detect_pcrel_tables (an s_getpc_b64-derived V# whose num_records is a known constant). The
    // shader BLOB carries the table; the recompiler folds it into a compile-time constant lookup.
    std::unordered_map<uint32_t, std::vector<uint32_t>> mubuf_pcrel_tables;
    std::unordered_map<uint32_t, std::vector<uint32_t>> smem_pcrel_tables;
    // Typed consumer of the same idiom, admitted only for conversion-free 32-bit formats (#2859).
    std::unordered_map<uint32_t, std::vector<uint32_t>> mtbuf_pcrel_tables;
    // SCALAR-SPILL VGPR (#273 — DOLL's big post PS): the compiler packs excess wave-uniform scalars
    // into one VGPR's lanes via `v_writelane_b32 vN, sX, <const lane>` and reads them back with
    // `v_readlane_b32 sY, vN, <const lane>`. Per-invocation each (vgpr, lane) slot is just a named
    // scalar: vgpr -> lane -> SSA id. A vgpr used as a spill array must never be read as ordinary
    // per-lane data — operand_bits rejects that (fail-visible), keeping the model honest.
    std::unordered_map<int, std::unordered_map<int, uint32_t>> vgpr_lane_slots;
    // EXEC/VCC/saved-mask spills use the same fixed lanes but remain bool-domain values.
    std::unordered_map<int, std::unordered_map<int, uint32_t>> vgpr_lane_mask_slots;
    // An ordinary VALU write ends a spill-array lifetime. Keep a tombstone so a later static
    // v_readlane cannot reinterpret the recycled per-lane VGPR as a generic subgroup shuffle.
    // A new v_writelane starts a fresh spill lifetime and clears the tombstone.
    std::unordered_set<int> invalidated_vgpr_lane_slots;
    // LDS ADDTID per-lane spill slots (#273 — DOLL's title post PSes): `s_movk m0, K;
    // ds_write_addtid_b32 vN` spills THIS lane's vN to LDS[M0 + offset + tid*4], and the matching
    // `ds_read_addtid_b32` reloads it. Per-invocation the slot is one value, keyed by the M0 SSA id
    // + the instruction offset (uconst is interned, so equal constant M0s share one id). A read of a
    // never-written slot rejects (fail-visible).
    std::unordered_map<uint64_t, uint32_t> lds_addtid;
    // The provenance of each slot's spilled word (addtid_spill_marks), so the reload carries the
    // spilled word's missing lanes instead of looking genuine (#5183).
    std::unordered_map<uint64_t, ScalarSourceMarks> lds_addtid_marks;
    // One successful native sample result, consumed by its own word-mark transfer.
    NativeImageResultPublication native_image_result_publication;
};
struct LoopBlanketCheck;
template <class Registers>
void mark_loop_carried(RegState& rs, const Registers& carried, const std::vector<Rdna2Inst>& ins,
                       uint32_t lo, uint32_t hi, LoopBlanketCheck& blanket);
bool scalar_is_lane_local(const RegState& rs, int sgpr);
bool has_wave64_mask_half_pair(const RegState& rs, int base);
void expire_wave64_mask_half(RegState& rs, int reg, int preserved_pair = -1);
bool sreg_word_absent_unmasked(const RegState& rs, int r);
bool sreg_word_may_be_fabricated(const RegState& rs, int r);
bool b32_source_word_defined(const RegState& rs, const Operand& source);
bool b32_sources_defined(const RegState& rs, const Rdna2Inst& in);
}   // namespace prosper::gpu
