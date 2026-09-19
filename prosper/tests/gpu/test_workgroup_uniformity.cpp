// Workgroup barrier admission must prove every wave takes the same branch. These are CPU-only
// counterexamples: dispatching a shader whose waves disagree around a barrier is not a safe test.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int failures = 0;
static int checks = 0;
static void check(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::printf("FAIL: %s\n", message);
    }
}

static bool uniform(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    // Every fixture ends with a conditional branch over s_barrier to s_endpgm.
    return scc_branch_is_workgroup_uniform(ins, static_cast<uint32_t>(code.size() - 3));
}

int main() {
    // Exercise both the new entry-prefix walk (a block boundary before the compare) and the
    // local-block walk. They must agree about implicit operands and source widths.
    for (bool separate_block : {false, true}) {
        const uint32_t boundary = separate_block ? 0xbf820000u : 0xbf800000u;
        for (uint32_t operation : {0xbe9c1b80u, 0xbe9c1d80u, 0xbe9c0580u}) {
            const std::vector<uint32_t> code = {
                0x7e380500u,             // v_readfirstlane_b32 s28,v0: may differ across waves
                operation,              // bitset0/bitset1/cmov_b32 s28,0: reads old s28
                0xf420050eu, 0xfa000000u,// s_buffer_load_dword s20,s[28:31],0
                0xbf8c0000u,             // s_waitcnt 0
                boundary,               // branch to next instruction, or nop
                0xbf061480u,             // s_cmp_eq_u32 0,s20
                0xbf850001u,             // s_cbranch_scc1 -> end
                0xbf8a0000u,             // s_barrier
                0xbf810000u,             // s_endpgm
            };
            check(!uniform(code), separate_block
                ? "entry-prefix implicit destination must not disappear from the proof"
                : "local implicit destination must not disappear from the proof");
        }
        // A full overwrite really does remove the varying dependency. Rejecting every scalar
        // instruction or retaining an obsolete writer would fail this positive control.
        std::vector<uint32_t> overwritten = {
            0x7e380500u,                 // v_readfirstlane_b32 s28,v0
            0xbe9c0380u,                 // s_mov_b32 s28,0 (no implicit destination read)
            0xf420050eu, 0xfa000000u,
            0xbf8c0000u, boundary,
            0xbf061480u, 0xbf850001u, 0xbf8a0000u, 0xbf810000u,
        };
        check(uniform(overwritten), "full scalar overwrite severs the varying dependency");

        std::vector<uint32_t> high_word = {
            0xbe9c0380u,                 // s_mov_b32 s28,0 (uniform low half)
            0x7e3a0500u,                 // v_readfirstlane_b32 s29,v0 (varying high half)
            0xbe9e101cu,                 // s_bcnt1_i32_b64 s30,s[28:29]: 32-bit dst, 64-bit src
            0xf4200506u, 0x3c000000u,    // s_buffer_load_dword s20,s[12:15],s30
            0xbf8c0000u, boundary,
            0xbf061480u, 0xbf850001u, 0xbf8a0000u, 0xbf810000u,
        };
        // E.g. high halves 15 and 255 produce byte offsets 4 and 8, which may load different
        // branch selectors. The uniform low half alone says nothing about the result.
        check(!uniform(high_word), separate_block
            ? "entry-prefix B64 source high word must be proved uniform"
            : "local B64 source high word must be proved uniform");
        // Conditional B64 writes must invalidate BOTH destination words. Otherwise the proof
        // skips the high-word write and incorrectly finds the earlier uniform s29 definition.
        std::vector<uint32_t> conditional_pair = {
            0xbe9d0380u,                 // s_mov_b32 s29,0
            0x7e080500u,                 // v_readfirstlane_b32 s4,v0
            0xbf060480u,                 // s_cmp_eq_u32 0,s4: wave-dependent SCC
            0xbe9c0600u,                 // s_cmov_b64 s[28:29],s[0:1]
            0xf4200506u, 0x3a000000u,    // s_buffer_load_dword s20,s[12:15],s29
            0xbf8c0000u, boundary,
            0xbf061480u, 0xbf850001u, 0xbf8a0000u, 0xbf810000u,
        };
        check(!uniform(conditional_pair), separate_block
            ? "entry-prefix conditional B64 high-word write invalidates the old value"
            : "local conditional B64 high-word write invalidates the old value");
        conditional_pair[3] = 0xbe9c0400u; // unconditional s_mov_b64 from uniform launch pair
        check(uniform(conditional_pair), "unconditional B64 overwrite remains uniform");

        high_word[1] = 0xbe9d038fu;       // s_mov_b32 s29,15: now both halves are uniform
        check(uniform(high_word), "B64 count with two uniform source words remains admitted");
    }

    // Keep the PR's intended indirect-descriptor shape: the inner load sees the descriptor
    // before a later scratch overwrite, and the branch is in a different basic block.
    const std::vector<uint32_t> indirect = {
        0xf4280706u, 0xfa000000u,         // s_buffer_load_dwordx4 s[28:31],s[12:15],0
        0xbf8c0000u,
        0xf420050eu, 0xfa000000u,         // s_buffer_load_dword s20,s[28:31],0
        0xbf8c0000u,
        0xbe9c0380u,                     // later scratch reuse of s28
        0xbf820000u,
        0xbf061480u, 0xbf850001u, 0xbf8a0000u, 0xbf810000u,
    };
    check(uniform(indirect), "entry descriptor indirection survives later scratch reuse");

    // A syntactically branch-free prefix need not execute only once. A later backedge can
    // re-enter its load with an address changed by another wave-local instruction.
    for (bool overwrite_before_load : {false, true}) {
        std::vector<uint32_t> loop;
        if (overwrite_before_load)
            loop.push_back(0xbe9c0380u);  // uniform first visit, bypassed by the backedge
        const uint32_t load_pc = static_cast<uint32_t>(loop.size());
        const uint32_t tail[] = {
            0xf420050eu, 0xfa000000u, 0xbf8c0000u, 0xbf820000u,
            0xbf061480u, 0xbf850001u, 0xbf8a0000u,
            0x7e380500u,                 // s28 becomes wave-dependent for the next iteration
            0xbf82fff7u,                 // s_branch -9: back to load, bypassing the overwrite
            0xbf810000u,
        };
        loop.insert(loop.end(), std::begin(tail), std::end(tail));
        auto loop_uniform = [&] {
            std::vector<Rdna2Inst> ins;
            rdna2_walk(loop.data(), loop.size(), ins);
            return scc_branch_is_workgroup_uniform(ins, load_pc + 5);
        };
        check(!loop_uniform(), overwrite_before_load
            ? "backedge can bypass an explicit uniform prefix definition"
            : "backedge can replace apparently untouched launch data");
        loop[load_pc + 8] = 0xbe80201cu; // s_setpc_b64 s[28:29]: unknown target
        check(!loop_uniform(), "unresolved indirect edge cannot establish an entry prefix");
        loop[load_pc + 8] = 0xbf800000u; // no backedge: scratch reuse is after the only load
        check(loop_uniform(), "post-load scratch reuse alone does not poison the prefix");
    }

    const std::vector<uint32_t> narrowed_exec = {
        0xbefe04c1u, 0x7da40080u, 0xbf880001u, 0x06020000u,
        0xbe800385u, 0xbf060085u, 0xbf850001u, 0xbf8a0000u, 0xbf810000u,
    };
    check(uniform(narrowed_exec), "uniform SCC does not require full EXEC");
    std::vector<Rdna2Inst> narrowed_ins;
    rdna2_walk(narrowed_exec.data(), narrowed_exec.size(), narrowed_ins);
    check(!vcc_branch_is_workgroup_uniform(narrowed_ins, 6),
          "VCC still requires full EXEC at the same site");

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
