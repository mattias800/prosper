// test_entry_m0_dispatcher — the entry-M0 token must survive a CFG-DISPATCHER block edge (#3203).
//
// Why this file exists, and why it is not in test_rdna2_spirv_struct.cpp with the other #3133 arms.
//
// #3136 contains the entry-M0 token at all seven places the recompiler writes `rs.sreg` from merge
// machinery. Six of them are structured-emitter joins and each has an arm in
// test_rdna2_spirv_struct.cpp that reddens when it is disabled. The seventh is the CFG DISPATCHER,
// and it had no arm at all: an independent reviewer disabled it in isolation and the whole suite
// stayed green (#3203).
//
// The dispatcher is the one that matters most, because of HOW it loses the token. It persists every
// referenced SGPR in a Function variable and reloads it at each block entry through a `sget` that
// renders an absent scalar as `uconst(0)` — so an absent scalar comes back as ORDINARY TRACKED DATA
// one indirection further away than a phi. `rdna2_emit_cfg.cpp`'s `load_state` therefore re-arms the
// token on the MAY set (every register any `s_mov_b32 sX, m0` in the stream can leave holding it)
// at every block entry, dropping the reloaded placeholder:
//
//     for (int reg : entry_m0_may_hold) { state.sreg.erase(reg); state.sreg_entry_m0.insert(reg); }
//
// Without those two lines the token becomes a tracked zero at the first block boundary it crosses,
// and `operand_bits`' SGPR case ends in `return b.uconst(0)` with `ok` left TRUE — so the leak does
// not reject, it silently reads as zero. A fabricated word then feeds whatever the shader does with
// the driver's M0. Silent-wrong is exactly the failure the token exists to prevent, and it is the
// one path where a regression produced no signal anywhere.
//
// THE ROUTE IS THE POINT. Every program here is compute, is portable (`native_subgroup_size == 0`),
// contains one structured forward SCC if, and contains a `v_readlane_b32` whose source is not a
// writelane spill array. That combination is `portable_compute_cfg_readlane` in `emit_body`, and it
// is the one dispatcher entry that does NOT fall back to the structurizer on a clean reject:
//
//     if (allow_cfg_dispatcher && portable_compute_cfg_readlane) { ...
//         const int emitted = try_cfg_dispatcher();
//         if (emitted > 0) return true;
//         return false;                      // <-- no second attempt
//     }
//
// So for these programs the dispatcher is the ONLY emitter. A compile can only be the dispatcher
// accepting, and a reject can only be the dispatcher rejecting — the structured `join_entry_m0`
// cannot stand in for either, which is what makes this test discriminate the seventh site from the
// six already covered. Both halves of that are asserted rather than assumed: the compiling controls
// must contain an `OpSwitch` (only the dispatcher emits one), and the rejecting arm's recorded
// terminal reason must carry the `cfg-recompile-reject` tag, which only `emit_cfg_state_machine`
// writes.
//
// #3308 EXTENDS this file with the other half of the same contract. #3203 pinned that the token
// must SURVIVE a dispatcher edge; it said nothing about where the token may first appear, and
// `load_state` re-armed the whole-stream MAY set at EVERY block entry -- the program's own ENTRY
// block included, where no instruction has run and no save can have executed. Stray's compute
// program `0x300e390000` saves M0 into s14 at pc157, and s14 is also the compute stage's
// workgroup-id X, so `v_lshl_add_u32 v11, s14, 3, v0` at pc4 -- the shader's global-thread-index
// computation, four dwords in -- read a token for a save 153 dwords AHEAD of it and the dispatch
// was skipped (#3126). The re-arm is now an entry-rooted forward MAY dataflow, and arms D and E
// below pin both directions of it: an entry-block read compiles, and a read past the save still
// rejects at that read.
//
// Every arm reports independently instead of returning at the first failure. The ctest binary the
// other #3133 arms live in returns at its first `[FAIL]`, so a mutation run there cannot tell which
// arms moved; #3136's author had to build a throwaway probe harness for exactly that. This one is
// the harness, kept in the tree.
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Whether the module contains an instruction with the given opcode.
static bool has_opcode(const std::vector<uint32_t>& spv, uint32_t opcode) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t wc = spv[i] >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if ((spv[i] & 0xffffu) == opcode) return true;
        i += wc;
    }
    return false;
}

static bool has(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

constexpr uint32_t kOpSwitch = 251;

// A distinct non-zero program address per arm: `record_terminal_reject_reason` early-returns on a
// zero address, and the reasons are keyed by address, so sharing one would let a later arm's verdict
// overwrite an earlier arm's.
static std::vector<uint32_t> compile(const uint32_t* code, size_t dwords, uint64_t program) {
    ComputeShaderConfig config;          // portable Wave64: native_subgroup_size stays 0
    return recompile_compute(code, dwords, nullptr, config,
                             {RecompileDiagnosticStage::Compute, program});
}

int main() {
    printf("== test_entry_m0_dispatcher ==\n");

    // ---------------------------------------------------------------------------------------------
    // The four programs are a minimal pair set: pc0-pc2 and pc4-pc7 are byte-identical across the
    // first three, and only the single dword at pc3 changes. Nothing about the control flow, the
    // dispatcher route, the block partition or the post-edge read differs between them, so the
    // verdicts cannot diverge for any reason other than what that one instruction does to s0.
    //
    //   pc0: s_mov_b32 s0, 5            a real value on the edge that skips the arm
    //   pc1: s_cmp_eq_u32 s4, 0
    //   pc2: s_cbranch_scc0 -> pc4      one structured forward if  => Fs is non-empty
    //   pc3: <the variable>             the arm body: a save, or an ordinary write
    //   pc4: v_readlane_b32 s2, v0, 0   portable readlane          => the dispatcher route, no fallback
    //   pc6: v_add_nc_u32 v1, s0, v0    reads s0 PAST the block edge
    //   pc7: s_endpgm
    //
    // pc3 and pc6 are in different basic blocks, and a portable compute stream leaves
    // `direct_dispatch` false, so each block is its own dispatch case with its own `load_state`.
    // s0 therefore crosses a dispatcher edge between the write and the read in every program here.

    // THE ARM. `s_mov_b32 s0, m0` tokenises s0 in the arm block; the read at pc6 is in the next
    // block. The re-arm re-establishes the token at that block's entry, so the read rejects. Without
    // it the read resolves to the Function variable's reloaded zero and the program COMPILES —
    // silently, with a fabricated word standing in for the driver's M0.
    const uint32_t save_then_read_across_edge[] = {
        0xBE800385u,                            // pc0: s_mov_b32 s0, 5
        0xbf068004u,                            // pc1: s_cmp_eq_u32 s4, 0
        0xbf840001u,                            // pc2: s_cbranch_scc0 -> pc4
        0xBE80037Cu,                            // pc3: s_mov_b32 s0, m0     (the save)
        0xd7600002u, 0x00010100u,               // pc4: v_readlane_b32 s2, v0, 0
        0x4A020000u,                            // pc6: v_add_nc_u32 v1, s0, v0
        0xbf810000u,                            // pc7: s_endpgm
    };
    // CONTROL A — the same program with an ordinary scalar write in place of the save. This is what
    // proves the region shape, the readlane route and the cross-block scalar read are all
    // representable: without it the arm's reject could be any of those three failing instead of the
    // token, and the arm would pass for the wrong reason.
    const uint32_t ordinary_write_across_edge[] = {
        0xBE800385u, 0xbf068004u, 0xbf840001u,
        0xBE800387u,                            // pc3: s_mov_b32 s0, 7      (ordinary data)
        0xd7600002u, 0x00010100u,
        0x4A020000u,
        0xbf810000u,
    };
    // CONTROL B — a real save, into a register nothing reads. `entry_m0_may_hold` is {s3}, so the
    // re-arm loop RUNS at every block entry; it simply has nothing to say about s0. The read of s0
    // across the same edge must still compile.
    //
    // This is the control that separates "the re-arm is register-precise" from "a program containing
    // an entry-M0 save is rejected wholesale by the dispatcher". If it were the latter, the arm above
    // would redden nothing when the re-arm was deleted and this test would be a tautology wearing a
    // reject.
    const uint32_t save_elsewhere_read_across_edge[] = {
        0xBE800385u, 0xbf068004u, 0xbf840001u,
        0xBE83037Cu,                            // pc3: s_mov_b32 s3, m0     (save into an unread reg)
        0xd7600002u, 0x00010100u,
        0x4A020000u,                            // pc6: reads s0, which holds ordinary data
        0xbf810000u,
    };
    // CONTROL C — the permissive half, and the reason the dispatcher re-arms rather than rejecting
    // any block whose MAY set is non-empty: a save and its restore CONTAINED IN ONE dispatch block
    // still compile. The token re-armed at that block's entry is consumed by the restore, which
    // returns M0 to untracked — the state every movrel / ADDTID / ds_append guard demands.
    const uint32_t save_and_restore_in_one_block[] = {
        0xBE800385u,                            // pc0: s_mov_b32 s0, 5
        0xbf068004u,                            // pc1: s_cmp_eq_u32 s4, 0
        0xbf840001u,                            // pc2: s_cbranch_scc0 -> pc4
        0x7e040281u,                            // pc3: v_mov_b32 v2, 1      (arm body, not a save)
        0xd7600002u, 0x00010100u,               // pc4: v_readlane_b32 s2, v0, 0
        0xBE80037Cu,                            // pc6: s_mov_b32 s0, m0     (save)
        0xBEFC0300u,                            // pc7: s_mov_b32 m0, s0     (restore, SAME block)
        0xbf810000u,                            // pc8: s_endpgm
    };

    // ARM D (#3308) — THE ENTRY BLOCK. Same region shape and same dispatcher route as the arm
    // above, with the read moved to pc0, AHEAD of the save at pc3 in both program order and every
    // control-flow path. s0 gets no prior write on purpose: that is the Stray shape, where the
    // token-bearing register is an inbound launch word (workgroup-id X) the shader reads before it
    // ever reuses it. No save has executed at pc0 on any path, so the token cannot be live there and
    // the read must resolve. With the whole-stream re-arm it did not: `entry_m0_may_hold` is {s0}
    // and the entry block's own `load_state` stamped it, so the program was refused at pc=0.
    const uint32_t entry_read_before_any_save[] = {
        0x4A020000u,                            // pc0: v_add_nc_u32 v1, s0, v0   (ENTRY-block read)
        0xbf068004u,                            // pc1: s_cmp_eq_u32 s4, 0
        0xbf840001u,                            // pc2: s_cbranch_scc0 -> pc4
        0xBE80037Cu,                            // pc3: s_mov_b32 s0, m0          (the save, LATER)
        0xd7600002u, 0x00010100u,               // pc4: v_readlane_b32 s2, v0, 0
        0xbf810000u,                            // pc6: s_endpgm
    };
    // ARM E (#3308) — the direction that keeps arm D from being a licence to delete the token. The
    // same program with one extra read of s0 AFTER the save's block. The entry read at pc0 must
    // still resolve and the post-save read at pc6 must still reject, so the program is refused and
    // the refusal names pc=6. Asserting only `empty()` here would pass for a fix that reverted to
    // rejecting at pc=0, which is the defect; asserting only arm D would pass for a fix that deleted
    // the re-arm outright, which is #3203's silent-zero regression. Neither arm is sufficient alone.
    const uint32_t entry_read_then_post_save_read[] = {
        0x4A020000u,                            // pc0: v_add_nc_u32 v1, s0, v0   (ENTRY-block read)
        0xbf068004u,                            // pc1: s_cmp_eq_u32 s4, 0
        0xbf840001u,                            // pc2: s_cbranch_scc0 -> pc4
        0xBE80037Cu,                            // pc3: s_mov_b32 s0, m0          (the save)
        0xd7600002u, 0x00010100u,               // pc4: v_readlane_b32 s2, v0, 0
        0x4A060000u,                            // pc6: v_add_nc_u32 v3, s0, v0   (past the edge)
        0xbf810000u,                            // pc7: s_endpgm
    };

    const auto arm = compile(save_then_read_across_edge,
                             std::size(save_then_read_across_edge), 0x32030001ull);
    const auto control_a = compile(ordinary_write_across_edge,
                                   std::size(ordinary_write_across_edge), 0x32030002ull);
    const auto control_b = compile(save_elsewhere_read_across_edge,
                                   std::size(save_elsewhere_read_across_edge), 0x32030003ull);
    const auto control_c = compile(save_and_restore_in_one_block,
                                   std::size(save_and_restore_in_one_block), 0x32030004ull);
    const auto arm_d = compile(entry_read_before_any_save,
                               std::size(entry_read_before_any_save), 0x32030005ull);
    const auto arm_e = compile(entry_read_then_post_save_read,
                               std::size(entry_read_then_post_save_read), 0x32030006ull);

    // The controls first: an arm whose controls have not been read is a reject with no denominator.
    CHECK(!control_a.empty(),
          "control A: the same region with an ordinary scalar write compiles");
    CHECK(has_opcode(control_a, kOpSwitch),
          "control A: it lowered through the CFG DISPATCHER (the module contains an OpSwitch)");
    CHECK(!control_b.empty(),
          "control B: an entry-M0 save into an unread register does not block the region");
    CHECK(has_opcode(control_b, kOpSwitch),
          "control B: a program CONTAINING a save still lowers through the CFG dispatcher");
    CHECK(!control_c.empty(),
          "control C: a save and its restore inside ONE dispatch block still compile");

    // The arm. Two independent statements, because either alone is weaker than it looks: an empty
    // result says the program was refused but not by whom, and a recorded reason says who refused
    // but is a diagnostic string. Together they say the CFG dispatcher refused the read of a
    // token-live register.
    CHECK(arm.empty(),
          "#3203: an entry-M0 token cannot cross a CFG-DISPATCHER block edge into the data domain");
    const std::string reason = last_terminal_reject_reason(0x32030001ull);
    CHECK(has(reason, "cfg-recompile-reject"),
          "#3203: the refusal came from the CFG dispatcher itself, not from a structured join");
    CHECK(has(reason, "pc=6"),
          "#3203: it refused at the READ past the block edge, not at the save or the region shape");
    CHECK(has(reason, "mode=unresolved-operand"),
          "#3203: it refused because the operand did not resolve -- the token barrier, not a "
          "missing lowering");
    if (fails) printf("  [info] arm reject reason: '%s'\n", reason.c_str());

    // Arms D and E (#3308).
    CHECK(!arm_d.empty(),
          "#3308: a read at the ENTRY block of a register saved LATER compiles -- no save has "
          "executed there on any path, so no token can be live");
    CHECK(has_opcode(arm_d, kOpSwitch),
          "#3308: arm D lowered through the CFG DISPATCHER (the module contains an OpSwitch)");
    CHECK(arm_e.empty(),
          "#3308: the token still rejects a read past the save's block -- the entry-block fix is a "
          "narrowing, not a deletion");
    const std::string reason_e = last_terminal_reject_reason(0x32030006ull);
    CHECK(has(reason_e, "cfg-recompile-reject"),
          "#3308: arm E's refusal came from the CFG dispatcher itself");
    CHECK(has(reason_e, "pc=6"),
          "#3308: arm E refused at the READ PAST THE SAVE, not at the entry-block read at pc=0");
    if (fails) printf("  [info] arm E reject reason: '%s'\n", reason_e.c_str());

    // Measured mutation signatures, so a future reader can tell a real regression from a fixture
    // that drifted. Every one built cleanly (rc=0) before being believed.
    //
    //   delete the re-arm entirely          -> the four arm lines redden, five controls stay green,
    //                                          reject reason is EMPTY (nothing refused at all), and
    //                                          the other 335 registered tests still pass.
    //   `state.sreg.erase(reg)` with no      -> identical signature. "Untracked" is not a safe
    //   `sreg_entry_m0.insert(reg)`             middle ground: it fabricates the same zero silently.
    //   re-arm the whole-stream MAY set at    -> 3 fails (rc=1): arm D's two lines, and arm E's
    //   every block entry (the pre-#3308          `pc=` line, whose reason reads
    //   behaviour)                                `pc=0 words=4a020000 fmt=6 op=0x25` -- the
    //                                             ENTRY-block read refused for a save three dwords
    //                                             ahead of it. The four #3203 arms and all five
    //                                             controls stay green, which is why #3203 alone
    //                                             could not see this.
    //   stamp NO tokens at any block entry    -> 7 fails (rc=1): every #3203 arm plus arm E. Arm D
    //                                             passes, correctly -- nothing rejects when the
    //                                             token is gone. The two signatures are disjoint in
    //                                             both directions, so neither arm is vacuous and
    //                                             neither substitutes for the other.
    //   re-arm EVERY scalar, not the MAY set -> the OPPOSITE signature. Controls B and C redden, the
    //                                          arm still rejects -- but at `pc=1`, the `s_cmp` that
    //                                          reads s4, not the read the contract is about. That is
    //                                          why the reject reason's `pc=` is asserted and not
    //                                          just the emptiness: without it this mutation passes
    //                                          the arm while refusing the wrong instruction.
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
