// test_traversal_chase — the EXEC-driven walk that chases a REAL buffer, executed on real Vulkan.
//
// WHY THIS EXISTS. `test_cfg_trip_bound` proves the control-flow half of this shape and says so in
// its own header: "Its body decrements the index instead of chasing a buffer, so the guest's data is
// out of the picture and only the control-flow lowering is on trial." That leaves the other half
// untested — the combination of the EXEC-narrowing loop AND a MUBUF load through a V#, which is the
// shape GTA V's 0x413dc6700 actually runs:
//
//     pc88  v_mov_b32     v2, s22          ; publish the current depth
//     pc89  v_cmpx_ne_u32 exec, 0, v1      ; EXEC = (index != 0)  <-- the ONLY exit
//     pc90  s_cbranch_execz -> 98
//     pc91  buffer_load_dword v1, v1, ...  ; follow the link      <-- what the other test omits
//     pc93  s_add_i32     s22, s22, 1      ; depth++
//     pc95  v_bfe_u32     v1, v1, 3, 27    ; next index, from bits [3:29] of the record
//     pc97  s_branch      -> 88
//
// That program hangs the GPU into a driver reset (#2542). The traversal has exactly two exits — a
// zero link, and an index walking past NUM_RECORDS (RDNA2 returns 0 for an out-of-range buffer load,
// which then trips the cmpx). Nothing bounds its depth, so if EITHER exit is lowered wrongly the loop
// cannot end, and that is indistinguishable from outside from "the guest's table is cyclic".
//
// The kernel below is a HAND-BUILT positive instance of that class. It is not derived from any
// capture: the three non-trivial encodings are derived from the RDNA2 field layouts (MUBUF
// enc|op|idxen and vaddr|vdata|srsrc|soffset; VOP3 enc|op|vdst and src0|src1|src2), and the branch
// displacements are computed as target-(pc+1). Deriving them independently is deliberate — a fixture
// lifted from the shader under test inherits whatever is wrong with that shader.
//
// The link array is supplied by the test, so the DATA is known-acyclic by construction and only the
// lowering is on trial. That is the separation no measurement on the live title can make, because
// there a wrong lowering and a cyclic table are both live at once.
//
// WHAT THIS DOES NOT COVER, stated so the pass is not read as more than it is. The buffer here is
// bound by the TEST HARNESS, so this pins the recompiler's half of the out-of-bounds contract: the
// walk is lowered correctly and an out-of-range index reads zero. It does NOT exercise
// `live_compute.cpp`'s `binding_bytes` computation, which is what makes the bound descriptor range
// equal the guest V#'s `num_records x stride` on a real dispatch. A regression THERE would not be
// caught here. That remains the open coverage gap on #2795.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/compute_runner.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// v1 arrives as a float (the harness loads storage-buffer floats into v0..vN) and leaves as one, so
// the walk is bracketed by converts — the same convention test_cfg_trip_bound uses.
static const uint32_t kChaseWalk[] = {
    0x7e020f01u,                //  0: v_cvt_u32_f32   v1, v1           ; index = (uint)input[1]
    0xbe960380u,                //  1: s_mov_b32       s22, 0           ; depth = 0
    0xbeea047eu,                //  2: s_mov_b64       s[106:107], exec  ; save the incoming mask
    0x7e040216u,                //  3: v_mov_b32       v2, s22          ; LOOP HEAD: publish depth
    0x7daa0280u,                //  4: v_cmpx_ne_u32   exec, 0, v1      ; narrow EXEC; the only exit
    0xbf880007u,                //  5: s_cbranch_execz -> 13            ; 13-(5+1) = 7
    0xe0302000u, 0x80000101u,   //  6: buffer_load_dword v1, v1, s[0:3], 0 idxen
    0xbf8c3f70u,                //  8: s_waitcnt
    0x81168116u,                //  9: s_add_i32       s22, s22, 1      ; depth++
    0xd5480001u, 0x026d0701u,   // 10: v_bfe_u32       v1, v1, 3, 27    ; next index from [3:29]
    0xbf82fff6u,                // 12: s_branch        -> 3             ; 3-(12+1) = -10
    0xbefe046au,                // 13: s_mov_b64       exec, s[106:107] ; restore so the store is live
    0x7e040d02u,                // 14: v_cvt_f32_u32   v2, v2
    0xbf810000u,                // 15: s_endpgm
};

namespace {

constexpr uint32_t kLanes = 64;

// The link record encodes its successor in bits [3:29] — the `v_bfe_u32 v1, v1, 3, 27` above — so a
// successor k is stored as k<<3. The low three bits are flags in the guest's format; setting them
// here would be a stronger test of the extract, and is deliberately left to the caller.
uint32_t link_to(uint32_t successor) { return successor << 3; }

// Recompile with a resource table so the MUBUF resolves. Binding 2 is the runner's `cbuf`.
std::vector<uint32_t> chase_module() {
    ShaderResourceTable table;
    ShaderResource links{};
    links.cls = ResourceClass::ConstantBuffer;
    links.format = DataFormat::Uint32;
    links.num_components = 1;
    links.binding = 2;                 // compute_runner binds `cbuf` here
    links.stride = 4;                  // one dword per record, as the guest's V# declares
    links.size = kLanes * 4;           // NUM_RECORDS * stride
    links.sgpr_base = 0;               // the load names s[0:3]
    table.resources.push_back(links);
    return recompile_valu(kChaseWalk, std::size(kChaseWalk), /*num_inputs*/2, /*out_vgpr*/2, &table);
}

// Run the walk over `links` and return each lane's reported depth.
std::vector<float> walk(const std::vector<uint32_t>& module_spv,
                        const std::vector<uint32_t>& links) {
    std::vector<float> input(kLanes * 2, 0.0f);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        input[lane * 2 + 1] = static_cast<float>(lane);   // lane i starts at index i
    return prosper::test::run_compute(module_spv, input, kLanes, kLanes, links);
}

uint32_t mismatches(const std::vector<float>& got, const std::vector<uint32_t>& want) {
    if (got.size() != want.size()) return 0xFFFFFFFFu;
    uint32_t bad = 0;
    for (size_t i = 0; i < want.size(); ++i)
        if (std::fabs(got[i] - static_cast<float>(want[i])) > 0.5f) ++bad;
    return bad;
}

}  // namespace

int main() {
    printf("== test_traversal_chase ==\n");

    const std::vector<uint32_t> spv = chase_module();
    CHECK(!spv.empty() && spv[0] == 0x07230203u,
          "the buffer-chasing EXEC walk recompiles to a SPIR-V module");
    if (spv.empty()) { printf("== FAIL: recompile produced nothing ==\n"); return 1; }

    // The load must actually be EMITTED. A V# that decoded to NUM_RECORDS=0 would fold the load to a
    // constant zero, and then every arm below would pass for the wrong reason: the walk would exit
    // after one step no matter what the links say. This is the check that the lever moved.
    bool has_access_chain = false;
    for (size_t i = 5; i + 1 < spv.size(); ++i) {
        const uint32_t op = spv[i] & 0xFFFFu;
        if (op == 65u /* OpAccessChain */) { has_access_chain = true; break; }
    }
    CHECK(has_access_chain,
          "the module contains a real access chain -- the load was not folded to a constant");

    // --- Arm 1: a descending acyclic chain. Lane i walks i steps. -------------------------------
    // The expectation is DISTINCT per lane on purpose: a lowering that ignored the per-lane mask, or
    // one whose load always returned the same value, would produce a constant depth and fail here.
    {
        std::vector<uint32_t> links(kLanes, 0u);
        for (uint32_t i = 1; i < kLanes; ++i) links[i] = link_to(i - 1);
        std::vector<uint32_t> want(kLanes);
        for (uint32_t i = 0; i < kLanes; ++i) want[i] = i;

        const std::vector<float> got = walk(spv, links);
        CHECK(got.size() == kLanes, "the chasing walk RUNS TO COMPLETION on real Vulkan");
        const uint32_t bad = mismatches(got, want);
        if (bad && bad != 0xFFFFFFFFu)
            for (uint32_t i = 0; i < kLanes; ++i)
                if (std::fabs(got[i] - static_cast<float>(i)) > 0.5f) {
                    printf("  first wrong lane=%u got=%g expected=%u\n", i, got[i], i);
                    break;
                }
        CHECK(bad == 0, "every lane reports the depth its own chain length implies");
    }

    // --- Arm 2: every link points at 0, so every non-zero lane stops after ONE step. -------------
    // Same kernel, same bindings, different DATA. This is what separates "the walk follows the
    // buffer" from "the walk happens to terminate": arm 1 alone is also satisfied by a lowering that
    // decrements, which is precisely what the other test's kernel does.
    {
        const std::vector<uint32_t> links(kLanes, link_to(0));
        std::vector<uint32_t> want(kLanes, 1u);
        want[0] = 0;                                   // lane 0 never enters the body

        const std::vector<float> got = walk(spv, links);
        CHECK(got.size() == kLanes, "the one-step walk runs to completion");
        CHECK(mismatches(got, want) == 0,
              "depths follow the LINK DATA, not the loop shape (all-zero links -> depth 1)");
    }

    // --- Arm 3: the out-of-range exit, which is the one HARDWARE supplies. -----------------------
    // Every in-range link points FAR past NUM_RECORDS, so the walk leaves the buffer and must then be
    // ended by the bounds rule rather than by a stored zero. RDNA2 returns 0 for an out-of-range
    // buffer load; the next cmpx sees 0 and the lane exits.
    //
    // The expected depth is **2**, not 1, and the difference is the whole point of the arm:
    //   step 1  reads links[i], which is IN range and hands back the out-of-range successor
    //   step 2  reads links[out-of-range], which must come back 0  <-- the contract under test
    // A depth of 1 would mean the first load already returned 0 (so the buffer was never really
    // read), and a walk that never ended would mean the out-of-range load returned neighbouring
    // memory. Only 2 is consistent with the console's behaviour. Measured: 0, 2, 2, ..., 2.
    //
    // This is the exit prosper delegates ENTIRELY to robustBufferAccess -- `num_records` appears
    // nowhere in the MUBUF emitter, so the recompiler issues no record-count check of its own and the
    // whole contract rests on the bound descriptor range agreeing with the V# (#2795, #2796). If that
    // agreement ever broke, this arm is what would notice, and it would notice by HANGING rather than
    // by returning a wrong number -- which is exactly how it presents on a live title.
    {
        const std::vector<uint32_t> links(kLanes, link_to(kLanes + 1000u));
        std::vector<uint32_t> want(kLanes, 2u);
        want[0] = 0;                                   // lane 0 never enters the body

        const std::vector<float> got = walk(spv, links);
        CHECK(got.size() == kLanes, "the out-of-range walk runs to completion (does not hang)");
        if (got.size() == kLanes)
            printf("  [diag] out-of-range depths: lane0=%g lane1=%g lane2=%g lane63=%g\n",
                   got[0], got[1], got[2], got[63]);
        CHECK(mismatches(got, want) == 0,
              "an index past NUM_RECORDS reads 0 and ends the walk in exactly two steps");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
