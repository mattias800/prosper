#ifndef PROSPER_GPU_RECOMPILER_SPIRV_COMPUTE_MEMBER_SCOPE
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"   // IWYU pragma: export
#else
#ifndef PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_VALUE_OPS_INL
#define PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_VALUE_OPS_INL
// Class-scope fragment: preserve the mechanically moved member bytes.
// clang-format off

    uint32_t id() { return next_id++; }
    static void put(std::vector<uint32_t>& s, uint32_t op, std::initializer_list<uint32_t> o) {
        s.push_back(((uint32_t)(o.size() + 1) << 16) | op); for (uint32_t x : o) s.push_back(x);
    }
    static void putv(std::vector<uint32_t>& s, uint32_t op, const std::vector<uint32_t>& o) {
        s.push_back(((uint32_t)(o.size() + 1) << 16) | op); s.insert(s.end(), o.begin(), o.end());
    }
    void pstr(std::vector<uint32_t>& v, const char* s);
    uint32_t fconst(float f);
    uint32_t uconst(uint32_t v);
    // Reverse of uconst: the literal behind an id, when that id IS one of our u32 constants.
    //
    // Only ids this builder minted through uconst() are answerable, which is the point: a value that
    // came out of arithmetic has no literal and must report false rather than a plausible number.
    // The cache is small (one entry per distinct constant in the module) and this is called at
    // descriptor-resolution sites, not per instruction.
    bool uconst_literal(uint32_t id_, uint32_t* out) const;
    // VGPRs are modeled as raw 32-bit VALUES (uint). Float ops bitcast their operands uint->float and
    // bitcast the result back to uint; integer ops operate on the bits directly. This matches the
    // hardware's untyped VGPRs and lets float and integer instructions share the same register file.
    uint32_t bcf(uint32_t u) { uint32_t r = id(); put(code, Op_Bitcast, {t_f32, r, u}); decorate_float_transport(r); return r; }
    uint32_t bcu(uint32_t f) { uint32_t r = id(); put(code, Op_Bitcast, {t_u32, r, f}); decorate_float_transport(r); return r; }
    uint32_t fconstf(float f) { uint32_t b = fbits(f); auto it = fconst_cache.find(b); if (it != fconst_cache.end()) return it->second;
        uint32_t c = id(); put(types, Op_Constant, {t_f32, c, b}); fconst_cache[b] = c; return c; }
    // Float binary op on bit-operands -> bit-result.
    uint32_t fbin(uint32_t op, uint32_t a, uint32_t b) { uint32_t rf = id(); put(code, op, {t_f32, rf, bcf(a), bcf(b)}); return bcu(rf); }
    uint32_t fbin_no_contraction(uint32_t op, uint32_t a, uint32_t b);
    uint32_t fma(uint32_t a, uint32_t b, uint32_t c);
    // Integer binary op on bit-operands -> bit-result.
    uint32_t ibin(uint32_t op, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, op, {t_u32, r, a, b}); return r; }
    // Integer unary op on a bit-operand -> bit-result (e.g. OpNot).
    uint32_t iun(uint32_t op, uint32_t a) { uint32_t r = id(); put(code, op, {t_u32, r, a}); return r; }
    // GLSL.std.450 uint ext-instruction (UMin/UMax) on bit-operands -> bit-result (no bitcast).
    uint32_t uext2(uint32_t inst, uint32_t a, uint32_t b) { uint32_t r = id(); putv(code, Op_ExtInst, {t_u32, r, glsl, inst, a, b}); return r; }
    // GLSL.std.450 FindILsb scans all 32 operand bits; zero returns -1. Preserve its signed
    // result as raw uint bits for the guest scalar register file.
    uint32_t find_ilsb(uint32_t a);
    // GLSL.std.450 FindUMsb: bit index of the most significant 1. glslang emits this with a
    // SIGNED result and an unsigned operand, which is the form drivers are exercised on, so
    // mirror it exactly and bitcast back. The result is UNDEFINED when the operand is zero —
    // every caller must exclude that input rather than relying on a -1 that is not specified.
    uint32_t find_umsb(uint32_t a) { uint32_t ri = id(); putv(code, Op_ExtInst, {t_i32, ri, glsl, Glsl_FindUMsb, a}); return i2u(ri); }
    // RDNA V_FFBH_U32 counts zeroes before the highest set bit and returns all ones for zero.
    // FindUMsb returns the bit index instead and is undefined at zero, so keep the conversion and
    // sentinel in one helper shared by ordinary ALU emission and portable dispatcher wave phases.
    uint32_t ffbh_u32(uint32_t a);
    // RDNA V_FFBL_B32 returns the first set-bit index from the LSB, with the same all-ones
    // sentinel for zero. Reversing the dword turns that index into FFBH's leading-zero count and
    // keeps the sentinel exact, so both instructions share one zero-safe FindUMsb lowering.
    uint32_t ffbl_b32(uint32_t a) { return ffbh_u32(iun(Op_BitReverse, a)); }
    // IEEE-754 binary32 ldexp on raw bits, with round-to-nearest-even for a subnormal result.
    // GLSL.std.450 Ldexp is NOT sufficient here: its contract leaves an overflowing product and
    // exp>128 undefined, while a guest exponent is an arbitrary i32. Keep the whole operation in
    // the integer domain so zero signs, infinities and NaN payloads survive exactly, and so every
    // shift remains in SPIR-V's defined [0,31] range. Exponents outside [-512,512] can be clamped
    // before the calculation: every finite nonzero binary32 must overflow above that interval or
    // round to signed zero below it.
    uint32_t ldexp_f32_bits(uint32_t bits, uint32_t exponent);
    // GLSL.std.450 float ext-instructions on bit-operands -> bit-result.
    uint32_t fext1(uint32_t inst, uint32_t a) { uint32_t r = id(); putv(code, Op_ExtInst, {t_f32, r, glsl, inst, bcf(a)}); return bcu(r); }
    uint32_t fext2(uint32_t inst, uint32_t a, uint32_t b) { uint32_t r = id(); putv(code, Op_ExtInst, {t_f32, r, glsl, inst, bcf(a), bcf(b)}); return bcu(r); }
    uint32_t frcp(uint32_t a) { uint32_t rf = id(); put(code, Op_FDiv, {t_f32, rf, fconstf(1.0f), bcf(a)}); return bcu(rf); }
    // The NEG source modifier is a floating-point negation = raw sign-bit toggle (ISA sec 6.2:
    // "Negate input"). Lowering it as 0.0f - x lost the sign of zero (FSub(+0,+0) = +0 where
    // hardware yields -0, observable via v_rcp -> wrong-signed Inf, packed sign bits, min/max +-0
    // ordering) and did not deterministically flip NaN signs. A bit-XOR is exact for every input
    // (+-0, NaN, denormals) on every driver, unlike OpFNegate which lacks that guarantee without
    // SignedZeroInfNanPreserve. The unpacked-f16 paths run in the f32 domain, so this applies there too.
    uint32_t fneg(uint32_t a) { return ibin(Op_BitwiseXor, a, uconst(0x80000000u)); }
    // CLAMP-modifier saturate to [0,1] with the hardware's DX10_CLAMP NaN rule: a NaN result clamps
    // to 0 (graphics shaders run with MODE.DX10_CLAMP set). NMax(NaN, 0) = 0 then NMin(0, 1) = 0
    // gives that for free; the previous FMin/FMax chain left a NaN result driver-defined.
    uint32_t clamp01(uint32_t x) {
        return fext2(Glsl_NMin, fext2(Glsl_NMax, x, bcu(fconstf(0.0f))), bcu(fconstf(1.0f)));
    }
    uint32_t cvt_u2f(uint32_t u) { uint32_t rf = id(); put(code, Op_ConvertUToF, {t_f32, rf, u}); return bcu(rf); }   // uint -> float bits
    // v_cvt_u32_f32 SATURATES: NaN -> 0, negative -> 0, >= 2^32 -> 0xFFFFFFFF. A bare OpConvertFToU
    // has an undefined result out of range (#135), so clamp first. FMin/FMax are themselves
    // NaN-undefined, so NaN is routed to 0 explicitly. 4294967040.0f (0x4F7FFFFF) is the largest
    // float < 2^32 — the next float IS 2^32, so the clamp is exact for every in-range input;
    // inputs >= 2^32 (incl. +inf) select UINT_MAX.
    uint32_t cvt_f2u(uint32_t bits);
    // Float ordered compare on bit-operands -> bool (for VCC). select() picks bits by a bool condition.
    uint32_t fcmp(uint32_t cmpop, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, cmpop, {t_bool, r, bcf(a), bcf(b)}); return r; }
    // Guest VOPC facts, not facts about arbitrary SPIR-V float comparisons. The emitter records
    // only successful full-EXEC, nonexception-signaling DWORD F32 compares. Key and value are
    // immutable result/raw-word SSA ids: register reuse, reloads and Phi never inherit them.
    std::unordered_map<uint32_t, uint32_t> guest_f32_zero_equal_words;
    std::unordered_map<uint32_t, uint32_t> guest_f32_abs_normal_le_words;
    // Returns zero unless bound is an owned constant positive finite NORMAL binary32 encoding.
    // The comparison is invariant under either guest input-denorm mode and uses integer bits only.
    uint32_t f32_abs_normal_le_bits(uint32_t word, uint32_t bound);
    uint32_t f32_nonzero_bits(uint32_t word);
    uint32_t bfalse() { if (!bconst_false) { bconst_false = id(); put(types, Op_ConstantFalse, {t_bool, bconst_false}); } return bconst_false; }
    uint32_t sel(uint32_t cond, uint32_t tval, uint32_t fval) { uint32_t r = id(); put(code, Op_Select, {t_u32, r, cond, tval, fval}); return r; }
    bool is_fragment_wave_vote_value(uint32_t value) const {
        return fragment_wave_vote_values.contains(value);
    }
    void mark_fragment_wave_vote_value(uint32_t value);
    void add_fragment_wave_vote_dependency(uint32_t result, uint32_t source);
    void propagate_fragment_wave_vote(uint32_t result, uint32_t source) {
        add_fragment_wave_vote_dependency(result, source);
    }
    void propagate_fragment_wave_vote(uint32_t result, uint32_t a, uint32_t b_);
    void mark_fragment_wave_scalar_use(uint32_t value);
    uint32_t bsel(uint32_t cond, uint32_t tval, uint32_t fval);  // bool-domain select (wave masks)

    // --- Structured control-flow primitives (for counted-loop reconstruction) ---
    uint32_t cur_block = 0;   // label id of the block currently being emitted (predecessor for OpPhi)
    void emit_label(uint32_t l)          { put(code, Op_Label, {l}); cur_block = l; }
    void emit_branch(uint32_t t)         { put(code, Op_Branch, {t}); }
    void emit_condbranch(uint32_t c, uint32_t t, uint32_t f) { put(code, Op_BranchConditional, {c, t, f}); }
    void emit_loopmerge(uint32_t m, uint32_t c)              { put(code, Op_LoopMerge, {m, c, 0}); }
    // OpPhi with two predecessors; the back-edge (value,label) is patched later via patch_phi(). Returns
    // the phi result id and sets `patch_off` to the code index of the placeholder back-edge value word.
    uint32_t emit_phi2(uint32_t type, uint32_t v0, uint32_t l0, size_t& patch_off);
    void patch_phi(size_t patch_off, uint32_t v1, uint32_t l1);
    void emit_selmerge(uint32_t m) { put(code, Op_SelectionMerge, {m, 0}); }   // structured if (before condbranch)
    void emit_switch(uint32_t selector, uint32_t fallback,
                     const std::vector<std::pair<uint32_t, uint32_t>>& cases);
    uint32_t function_var(uint32_t type, uint32_t& ptr_type);
    void declare_guest_scratch(const StaticScratchLayout& layout);
    uint32_t guest_scratch_load_word(uint32_t index);
    void guest_scratch_store_word(uint32_t index, uint32_t value,
                                  bool predicated, uint32_t pred);
    uint32_t guest_scratch_load_bits(int32_t byte_offset, uint32_t bits, bool sign_extend);
    void guest_scratch_store_bits(int32_t byte_offset, uint32_t bits, uint32_t value,
                                  bool predicated, uint32_t pred);
    uint32_t load_function(uint32_t type, uint32_t var);
    void store_function(uint32_t var, uint32_t value);
    uint32_t logical_not(uint32_t value);
    uint32_t subgroup_local_id();
    // Exact scalar wave vote for fragment control flow and mask reductions. A guest scalar branch
    // observes one 64-bit EXEC/VCC value for the complete hardware wave; branching directly on this
    // invocation's bool would instead create 64 independent pixel branches. Mark the module with the
    // same arithmetic capability used by fragment MBCNT so the backend enforces subgroup size 64,
    // and declare Vote for OpGroupNonUniformAny itself.
    uint32_t fragment_wave_any(uint32_t active_bit);
    // Fragment-side ballot (#2412). Same exactness argument as native_wave_ballot_half, but the
    // fragment stage establishes its exact-wave contract differently: it does not carry
    // native_subgroup_size, it DECLARES the width it needs via fragment_required_subgroup_size, which
    // the backend then enforces (or skips the draw). Mirrors fragment_wave_any, which exists for the
    // reduction form of the same problem.
    //
    // This is a REDUCE in #2410's taxonomy -- the bits become guest scalar DATA -- so the wave64
    // requirement it records must NOT be relaxed to a narrower subgroup: half a mask reported as whole
    // is silent wrong data.
    uint32_t fragment_wave_ballot_half(uint32_t mask_bit, uint32_t half);
    uint32_t subgroup_id();
    uint32_t fragment_mbcnt(uint32_t mask_bit, uint32_t acc_bits, bool lo);
    // Numeric MBCNT counts physical source bits, not participating invocations. This lowering
    // only needs the current invocation's architectural lane position and is valid in divergence.
    uint32_t numeric_mbcnt(uint32_t source_bits, uint32_t acc_bits, bool lo);
    uint32_t native_wave_any(uint32_t value);
    // One 32-bit HALF of the guest wave mask, materialised from this lane's bool (#2420).
    // OpGroupNonUniformBallot returns a uvec4 whose .x/.y are lanes 0..31 / 32..63 OF THIS SUBGROUP,
    // so it equals the guest mask only when the subgroup is exactly the guest wave. The caller must
    // establish that; a narrower subgroup would report half a mask as though it were whole, which is
    // silent wrong data rather than a visible reject.
    bool declared_subgroup_ballot = false;
    uint32_t native_wave_ballot_half(uint32_t mask_bit, uint32_t half);
    uint32_t native_wave_first_active(uint32_t mask_bit);
    uint32_t native_wave_popcount(uint32_t mask_bit);
    uint32_t native_compute_mbcnt(uint32_t mask_bit, uint32_t acc_bits, uint32_t lo);
    uint32_t helper_invocation();
    // Distinct guest descriptors can name the same allocation. Under the GLSL450 memory model,
    // separate StorageBuffer variables otherwise promise non-aliasing to the SPIR-V consumer. The
    // runtime binding addresses are deliberately absent from the shader cache key, so every external
    // guest-backed buffer declaration must retain the possibility of aliasing. Internal GDS uses its
    // own host allocation and intentionally does not go through this helper.
    void declare_external_storage_buffer(uint32_t pointer_type, uint32_t variable);
    void declare_internal_gds(uint32_t set = 1, uint32_t binding = 0);
    // Atomic publication for the diagnostic witness. Plain OpStore is wrong for a record every
    // invocation that reaches the cap writes: with per-invocation values the host can read a record
    // assembled from several invocations, and "last writer wins" silently discards the extremes that
    // make the record mean anything. A device-scope atomic min/max makes the published pair the true
    // extremes over every invocation and workgroup, which is the only reading the report claims.
    void compute_gds_atomic_minmax(uint16_t opcode, uint32_t index, uint32_t value,
                                   uint32_t pred);
    void compute_gds_store(uint32_t index, uint32_t value, bool predicated, uint32_t pred);
    // The read that pairs with it: one dword of the same device-global buffer. Not predicated;
    // the caller keeps the old register value on an inactive lane, as for every VGPR write.
    uint32_t compute_gds_load(uint32_t index);
    uint32_t compute_gds_atomic_rtn(uint32_t op, uint32_t index, uint32_t value);
    // Native-subgroup GDS append/consume is one device-global atomic per hardware wave. Fragment
    // helper invocations participate in subgroup operations but cannot consume guest counter slots;
    // compute has no helper lanes. Callers use this only when one host subgroup is one guest wave.
    uint32_t native_gds_append(uint32_t index, uint32_t exec_bit, bool consume);
    bool declared_subgroup_shuffle = false;
    void mark_subgroup_min16();
    void mark_subgroup_min32();
    void mark_subgroup_min64();
    uint32_t subgroup_shuffle(uint32_t value, uint32_t lane);
    uint32_t subgroup_row_shr_dynamic(uint32_t value, uint32_t active, uint32_t amount,
                                      uint32_t event = 0, uint32_t* valid_lane = nullptr);
    uint32_t subgroup_row_shr(uint32_t value, uint32_t active, uint32_t amount,
                              uint32_t* valid_lane = nullptr) {
        return subgroup_row_shr_dynamic(value, active, uconst(amount), 0, valid_lane);
    }
    // `stride` is the XOR applied to the lane id, and must be < 16.
    //
    // ROW_ROR:8 was the first member of this family and is spelled XOR 8 below, because XOR 8 is
    // exactly (row_lane - 8) modulo 16. ROW_XMASK:n is XOR n by definition, so the two controls
    // share one implementation and ROW_ROR:8 is literally ROW_XMASK:8. Generalising the stride is
    // therefore not a widening of the lowering, only of the control values that reach it.
    //
    // The property that makes every stride < 16 exact, and makes it exact independently of the host
    // subgroup width, is that XOR by a value under 16 touches only bits 0..3 -- so the source lane
    // stays inside the same architectural DPP16 row, and every row and subgroup bit above bit 3 is
    // preserved. That is the same argument the ROR:8 form already relied on; it was never specific
    // to 8.
    uint32_t subgroup_row_xor(uint32_t value, uint32_t active, uint32_t stride,
                              uint32_t* valid_lane = nullptr);
    uint32_t subgroup_quad_permute(uint32_t value, uint32_t ctrl, uint32_t active);
    bool ds_swizzle_source_lane(uint32_t offset, uint32_t* source_lane);
    uint32_t ds_bpermute_b32(uint32_t address, uint32_t value, uint32_t active, uint32_t offset,
                             uint32_t event = 0);
    // Called only at the portable compute dispatcher's common Workgroup rendezvous.
    uint32_t portable_ds_bpermute_b32(uint32_t address, uint32_t value,
                                      uint32_t active, uint32_t offset,
                                      uint32_t event, uint32_t metadata_base);
    // Common Workgroup rendezvous only. Returns this logical lane's widened Bool and whole64 SCC.
    // The caller owns mask destination/lifetime writes. No raster topology is inferred here.
    std::pair<uint32_t, uint32_t> packet_wqm_b64(
        uint32_t source, uint32_t pending, uint32_t event, uint32_t result_slot);
    // Fragment discard: OpKill any lane where `alive` is false, survivors fall through to `mergeL`. Used to
    // lower an EXP done under a narrowed EXEC (an alpha-test / WQM discard — the surviving lanes are the
    // ones that pass the kill test) to a real per-invocation discard. OpKill is a block terminator, so the
    // kill block has no back-edge to the merge; the merge's sole predecessor is the conditional branch.
    void discard_unless(uint32_t alive);
    // OpPhi with two fully-known predecessors (both edges' values available now) — for an if/merge join.
    uint32_t emit_phi_2way(uint32_t type, uint32_t va, uint32_t la, uint32_t vb, uint32_t lb);
    // Signed-int helpers: bits are bitcast to i32, the op runs, and the i32 result is bitcast to bits.
    uint32_t bcs(uint32_t u) { uint32_t r = id(); put(code, Op_Bitcast, {t_i32, r, u}); return r; }   // bits -> i32
    uint32_t i2u(uint32_t i) { uint32_t r = id(); put(code, Op_Bitcast, {t_u32, r, i}); return r; }   // i32 -> bits
    // High 32 bits of a 32x32 multiply (v_/s_mul_hi_*), via the {lo,hi} struct of OpU/SMulExtended.
    uint32_t t_u32pair = 0, t_i32pair = 0;
    uint32_t umul_hi(uint32_t a, uint32_t b_);
    uint32_t smul_hi(uint32_t a, uint32_t b_);
    uint32_t sbin(uint32_t op, uint32_t a, uint32_t b) { uint32_t ri = id(); put(code, op, {t_i32, ri, bcs(a), bcs(b)}); return i2u(ri); }
    uint32_t sext2(uint32_t inst, uint32_t a, uint32_t b) { uint32_t ri = id(); putv(code, Op_ExtInst, {t_i32, ri, glsl, inst, bcs(a), bcs(b)}); return i2u(ri); }
    // v_cvt_i32_f32 SATURATES: NaN -> 0, clamp to [INT_MIN, INT_MAX] (#135). Do not emit
    // OpConvertFToS: SPIR-V leaves invalid conversions undefined, and Metal also produces a
    // backend-specific result near the signed limits even after a float clamp (#686). Convert a
    // bounded absolute magnitude through the working unsigned path, then restore two's-complement
    // sign. 2^31 is representable by u32, so every conversion is defined on all backends.
    uint32_t cvt_f2i(uint32_t bits);
    uint32_t cvt_i2f(uint32_t bits) { uint32_t rf = id(); put(code, Op_ConvertSToF, {t_f32, rf, bcs(bits)}); return bcu(rf); }
    // Integer compares -> bool. scmp treats operands as signed, ucmp as unsigned.
    uint32_t scmp(uint32_t op, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, op, {t_bool, r, bcs(a), bcs(b)}); return r; }
    uint32_t ucmp(uint32_t op, uint32_t a, uint32_t b) { uint32_t r = id(); put(code, op, {t_bool, r, a, b}); return r; }
    // Bitfield extract (base, offset, count) -> bits. Unsigned and signed variants.
    // Clamp so Offset+Count <= 32 (#455). SPIR-V OpBitField*Extract is UNDEFINED when off+cnt exceeds
    // the 32-bit operand width, but s_bfe's 7-bit width field (0..127) and v_bfe's off+cnt (up to 62)
    // can exceed it. RDNA2 reads bits past the source MSB as 0 (unsigned) / sign (signed), i.e. the
    // effective count is min(cnt, 32-off) — reproduce that instead of emitting UB (the constant-offset
    // callers already satisfy off+cnt<=32, so the umin/isub fold to no-ops there).
    uint32_t bfe_clamp_cnt(uint32_t offc, uint32_t cnt) { return uext2(Glsl_UMin, cnt, ibin(Op_ISub, uconst(32), offc)); }
    uint32_t bfe_u(uint32_t base, uint32_t off, uint32_t cnt) { uint32_t offc = uext2(Glsl_UMin, off, uconst(32)); uint32_t r = id(); put(code, Op_BitFieldUExtract, {t_u32, r, base, offc, bfe_clamp_cnt(offc, cnt)}); return r; }
    uint32_t bfe_s(uint32_t base, uint32_t off, uint32_t cnt) { uint32_t offc = uext2(Glsl_UMin, off, uconst(32)); uint32_t ri = id(); put(code, Op_BitFieldSExtract, {t_i32, ri, bcs(base), offc, bfe_clamp_cnt(offc, cnt)}); return i2u(ri); }
    // Lazily declared 64-bit uint (+ Int64 capability), for s_bfe_u64 etc. that operate on SGPR pairs.
    uint32_t t_u64_cache = 0; bool declared_int64 = false, declared_int64_atomics = false;
    uint32_t t_u64() { if (!t_u64_cache) { if (!declared_int64) { put(caps, Op_Capability, {Cap_Int64}); declared_int64 = true; }
                       t_u64_cache = id(); put(types, Op_TypeInt, {t_u64_cache, 64, 0}); } return t_u64_cache; }
    // A 64-bit uint constant needs two value words. Shift amounts MUST be u64 here: a u32 shift operand on
    // a u64 base is mishandled by some drivers (llvmpipe), silently dropping the high half.
    uint32_t uconst64(uint64_t v);
    uint32_t u64_from_lohi(uint32_t lo, uint32_t hi);
    uint32_t u64_shift(uint32_t op, uint32_t value, uint32_t amount);
    uint32_t bfe_u64(uint32_t base64, uint32_t off, uint32_t cnt);
    uint32_t u64_lo(uint32_t v64) { uint32_t r = id(); put(code, Op_UConvert, {t_u32, r, v64}); return r; }  // truncate low 32
    uint32_t u64_hi(uint32_t v64) { uint32_t s = id(); put(code, Op_ShiftRightLogical, {t_u64(), s, v64, uconst64(32)});
        uint32_t r = id(); put(code, Op_UConvert, {t_u32, r, s}); return r; }
    uint32_t sel64(uint32_t cond, uint32_t yes, uint32_t no) {
        uint32_t r = id(); put(code, Op_Select, {t_u64(), r, cond, yes, no}); return r;
    }
    uint32_t u64_bit(uint32_t v64, uint32_t bit);
    // Lazily declared 2-float vector type (types are emitted as a block before code, so on-demand is safe).
    uint32_t t_v2f_cache = 0;
    uint32_t t_v2f() { if (!t_v2f_cache) { t_v2f_cache = id(); put(types, Op_TypeVector, {t_v2f_cache, t_f32, 2}); } return t_v2f_cache; }
    // pack two f32 (raw VGPR bits) into a dword of two f16 halves (src0->low, src1->high). Uses SPIR-V
    // PackHalf2x16, which is round-to-nearest-even — correct for the RTE store path (pack_half_lo).
    uint32_t pack_half2x16(uint32_t a, uint32_t b);
    // v_cvt_pkrtz_f16_f32 is round-toward-ZERO. PackHalf2x16 is round-to-nearest-even — within range that
    // differs by <=1 ULP (accepted), but at the f16 OVERFLOW boundary RTE yields +/-Inf where RTZ clamps
    // to the max finite f16 (+/-65504). Clamp each source to [-65504, 65504] before packing so an HDR
    // value above the f16 range becomes 65504 (matching RTZ's saturate) instead of an Inf/NaN that then
    // propagates through blending/compositing (#452). The RTE store path (pack_half_lo) is unaffected.
    uint32_t pack_half2x16_rtz(uint32_t a, uint32_t b);
    // Vertex-attribute unpacking (buffer_load_format_* with a packed data format). Each returns the
    // component as raw float VGPR bits, matching how the hardware presents a format load to the shader.
    //   unpack_norm: extract a `bits`-wide field at `bit_off` of `dword`, convert to float, divide by
    //   `norm` (255/127/65535/32767) — the UNORM/SNORM normalization; SNORM is clamped to >= -1.0.
    uint32_t unpack_norm(uint32_t dword, uint32_t bit_off, uint32_t bits, bool is_signed, float norm);
    // unpack_half: extract one of the two f16 halves packed in `dword` (which=0 low, 1 high) -> float bits.
    uint32_t unpack_half(uint32_t dword, uint32_t which);
    // One channel of a compressed (COMPR) colour export: half `which` of `dword`, decoded as
    // SPI_SHADER_COL_FORMAT says it was packed, and delivered as the bits the MRT's output class
    // expects (#4703). Defined in spirv/fragment_color_export.cpp.
    uint32_t compressed_export_channel(uint32_t dword, uint32_t which, ColorExportFormat format,
                                       FragmentOutputClass output_class);
    // GFX10's packed 10/11-bit vertex-float fields use binary16's five-bit exponent and a shortened
    // mantissa, without a sign bit. Widening the complete field left by 4 (11-bit) or 5 (10-bit)
    // produces the exact low-half binary16 encoding, including subnormals, infinity, and NaN.
    uint32_t unpack_ufloat(uint32_t dword, uint32_t bit_off, uint32_t bits);
    // Round a 24-bit significand right by a dynamic amount using round-to-nearest-even. SPIR-V
    // shifts by the word width are undefined, so clamp the executed shift and select zero for the
    // large-shift case (the significand is then strictly below halfway).
    uint32_t round_shift_even_u32(uint32_t value, uint32_t shift);
    // Exact inverse of unpack_ufloat for R11G11B10 stores. This mirrors float_to_f11/f10 without
    // going through binary16, which would double-round values near the shortened-mantissa ties.
    uint32_t pack_ufloat(uint32_t bits, uint32_t mantissa_bits);
    // Inverse of unpack_norm: pack a float VGPR (bits) into a `bits`-wide UNORM/SNORM integer field:
    // clamp to [0,1] (unsigned) or [-1,1] (signed), scale by `norm`, round-to-nearest-even, mask to width.
    uint32_t pack_norm(uint32_t fbits, uint32_t bits, bool is_signed, float norm);
    // Pack a float into the low 16 bits as an f16 (inverse of unpack_half low half).
    uint32_t pack_half_lo(uint32_t fbits) {
        return ibin(Op_BitwiseAnd, pack_half2x16(fbits, bcu(fconstf(0.0f))), uconst(0xFFFFu));
    }
// clang-format on
#endif
#endif
