// QUAD_PERM must use genuine current guest words, the selected peer's EXEC, and local SRC1.
// Zero-seeded Function storage is neither initialization evidence nor a lane-value oracle.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/compute_quad_or.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/rdna2_compute_quad_init.hpp"
#include "gpu/recompiler/rdna2_dpp_quad_or.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>

namespace fixture = prosper::test::compute_quad_or;
using namespace prosper::gpu;
namespace {
struct Entry {
    SpirvCompute b;
    RegState rs;
    Entry() {
        b.begin(16, nullptr, fixture::lanes, 1, 1, 64, 0);
        b.native_logical_uniform_launch = true;
        b.guest_compute_complete_groups = compute_dispatch_has_complete_groups(fixture::config());
        rs.exec = b.btrue();
        rs.vreg[0] = b.localid_comp[0];
    }
    std::set<uint32_t> proof(const std::vector<uint32_t>& code, uint32_t active = 0) {
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        return proven_compute_quad_or_sources(b, rs, ins, code.data(), code.size(), active);
    }
};
void compiles(fixture::Shape shape, uint32_t native = 0, uint32_t lanes = fixture::lanes,
              uint32_t wave = 64) {
    const auto code = fixture::program(shape);
    const auto addr = fixture::address();
    const auto config = fixture::config(native, lanes, wave);
    const auto module = fixture::compile(code, fixture::table(lanes), config, addr);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(addr);
    if (!native) {
        bpermute_oracle::Interpreter vm(module);
        const auto output =
            vm.run_buffers(lanes, {{3, std::vector<uint32_t>(lanes, fixture::sentinel)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(output, fixture::expected(shape, lanes));
    }
}
} // namespace
TEST(ComputeQuadOr, PlainSourcesAndDistinctLocalAddend) {
    compiles(fixture::Shape::Plain);
}
TEST(ComputeQuadOr, GenuineComplementaryExecDefinitions) {
    compiles(fixture::Shape::Complement);
}
TEST(ComputeQuadOr, DestinationAndPeerExecAreIndependent) {
    compiles(fixture::Shape::Masked);
}
TEST(ComputeQuadOr, DifferentEventsAtDifferentWavePcs) {
    compiles(fixture::Shape::DistinctSites);
}
TEST(ComputeQuadOr, UnlaunchedFinalQuadPeerContributesZero) {
    compiles(fixture::Shape::Plain, 0, 65);
}
TEST(ComputeQuadOr, Native64UsesTheSameCommonService) {
    for (auto shape : {fixture::Shape::Plain, fixture::Shape::Complement, fixture::Shape::Masked,
                       fixture::Shape::DistinctSites})
        compiles(shape, 64);
}
TEST(ComputeQuadOr, PortableWave32UsesTheSameLogicalQuadService) {
    for (auto shape : {fixture::Shape::Plain, fixture::Shape::Complement, fixture::Shape::Masked,
                       fixture::Shape::DistinctSites})
        compiles(shape, 0, fixture::lanes, 32);
}
TEST(ComputeQuadOr, CompleteRawLaunchAuthenticatesWritersBeforeExecRestore) {
    const auto code = fixture::restored_entry_program();
    // With PARTIAL_TG_EN clear, the partial-count fields have no entry-mask authority.
    for (bool has_partial_counts : {false, true}) {
        auto config = fixture::config(0, 64);
        if (has_partial_counts) config.dispatch_launch.num_thread[0] |= 31u << 16u;
        ASSERT_TRUE(compute_dispatch_has_complete_groups(config));
        const auto addr = fixture::address();
        const auto module = fixture::compile(code, fixture::table(64), config, addr);
        ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(addr);
        bpermute_oracle::Interpreter vm(module);
        const auto output =
            vm.run_buffers(64, {{3, std::vector<uint32_t>(64, fixture::sentinel)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(output, std::vector<uint32_t>(64, 273u));
    }
}
TEST(ComputeQuadOr, PartialUnknownAndMismatchedLaunchesRefuseAfterExecRestore) {
    const auto code = fixture::restored_entry_program();
    for (int mutation = 0; mutation < 10; ++mutation) {
        SCOPED_TRACE(mutation);
        auto config = fixture::config(0, 64);
        auto& raw = config.dispatch_launch;
        if (mutation == 0) {
            // Same normalized local/threads as the complete control, but only 31 guest lanes
            // start active. S_MOV_B64 EXEC,-1 cannot repair the earlier absent vector writes.
            raw.initiator |= 2u;
            raw.num_thread = {(31u << 16u) | 64u, (1u << 16u) | 1u, (1u << 16u) | 1u};
        }
        if (mutation == 1) raw = {};
        if (mutation == 2) raw.known &= ~ComputeDispatchLaunchState::kNumThreadX;
        if (mutation == 3) raw.known &= ~ComputeDispatchLaunchState::kInitiator;
        if (mutation == 4) raw.initiator |= 1u << 5u;
        if (mutation == 5) raw.num_thread[0] = 65;
        if (mutation == 6) raw.dimensions[0] = 2;
        if (mutation == 7) raw.initiator |= 1u << 15u;
        if (mutation == 8) config.exact_thread_extent = true;
        if (mutation == 9) raw.initiator = 0;
        EXPECT_FALSE(compute_dispatch_has_complete_groups(config));
        const auto addr = fixture::address();
        EXPECT_TRUE(fixture::compile(code, fixture::table(64), config, addr).empty());
        EXPECT_NE(last_terminal_reject_reason(addr).find("dpp-quad-or-entry-launch-unproven"),
                  std::string::npos)
            << last_terminal_reject_reason(addr);
    }
}
TEST(ComputeQuadOr, HostRoutingAndFullExecDoNotAuthenticateGuestEntry) {
    const auto code = fixture::restored_entry_program();
    Entry entry;
    ASSERT_EQ(entry.proof(code).size(), 1u);
    entry.b.guest_compute_complete_groups = false;
    EXPECT_TRUE(entry.proof(code).empty());
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    ComputeDppQuadOrService service;
    EXPECT_FALSE(service.prepare(entry.b, entry.rs, ins, code.data(), code.size(), 0));
    EXPECT_STREQ(service.rejection(), "dpp-quad-or-entry-launch-unproven");
    // This new fact is an OR-service premise, not an admission change for unrelated code.
    const std::vector<uint32_t> unrelated{0xbf810000u};
    ComputeDppQuadOrService no_quad;
    rdna2_walk(unrelated.data(), unrelated.size(), ins = {});
    EXPECT_TRUE(no_quad.prepare(entry.b, entry.rs, ins, unrelated.data(), unrelated.size(), 0));
}
TEST(ComputeQuadOr, UnsupportedFetchBoundsMasksAndModifiersRemainRefused) {
    for (int form = 0; form < 5; ++form) {
        auto p = fixture::program();
        std::vector<Rdna2Inst> ins;
        rdna2_walk(p.data(), p.size(), ins);
        for (const auto& in : ins)
            if (compute_dpp_quad_or(in)) {
                auto& modifiers = p[in.pc + 1];
                if (form == 0) modifiers |= 1u << 18u;   // FI1
                if (form == 1) modifiers &= ~(1u << 19u);   // BC0
                if (form == 2) modifiers &= ~(1u << 28u);   // sparse row mask
                if (form == 3) modifiers &= ~(1u << 24u);   // sparse bank mask
                if (form == 4) modifiers |= 1u << 20u;   // SRC0_NEG
            }
        const auto addr = fixture::address();
        EXPECT_TRUE(fixture::compile(p, fixture::table(), fixture::config(), addr).empty())
            << "unproved DPP form " << form;
    }
    compiles(fixture::Shape::Plain);
}
TEST(ComputeQuadOr, MissingPartialAndMismatchedGenerationsRefuseBeforeEmission) {
    for (auto shape :
         {fixture::Shape::MissingArm, fixture::Shape::NewGeneration, fixture::Shape::Absent,
          fixture::Shape::Partial, fixture::Shape::Cyclic, fixture::Shape::TooManyGenerations}) {
        SCOPED_TRACE(int(shape));
        const auto p = fixture::program(shape);
        const auto addr = fixture::address();
        Entry e;
        EXPECT_TRUE(e.proof(p).empty());
        EXPECT_TRUE(fixture::compile(p, fixture::table(), fixture::config(), addr).empty());
        EXPECT_NE(last_terminal_reject_reason(addr).find("dpp-quad-or-source-uninitialized"),
                  std::string::npos);
    }
}
TEST(ComputeQuadOr, WidthStorageAndForgedLaunchDoNotAuthenticateEntry) {
    const auto p = fixture::program(fixture::Shape::Complement);
    for (int mutation = 0; mutation < 8; ++mutation) {
        Entry e;
        if (mutation == 0) e.b.native_logical_uniform_launch = false;
        if (mutation == 1) e.rs.vreg[0] = e.b.uconst(0);
        if (mutation == 2) e.rs.vreg_source_marks[0].hard = true;
        if (mutation == 3) e.rs.exec_narrowed = true;
        if (mutation == 4) e.b.ngg_workgroup_shell = true;
        if (mutation == 5) e.b.invocation_guard_merge = 1;
        if (mutation == 6) e.rs.exec = e.b.bfalse();
        if (mutation == 7) e.rs.exec = e.b.uconst(1);
        EXPECT_TRUE(e.proof(p).empty()) << mutation;
    }
    Entry good;
    EXPECT_EQ(good.proof(p).size(), 2u);
    EXPECT_TRUE(good.proof(p, good.b.btrue()).empty());
}
TEST(ComputeQuadOr, MarkedLocalComponentsCannotSeedInitializedData) {
    for (int component : {1, 2}) {
        // Read the entry component before any guest writer can replace its generation.
        const std::vector<uint32_t> p{0x380600fau | (uint32_t(component) << 9u),
                                      0xff08b100u | uint32_t(component), 0xbf810000u};
        Entry e;
        e.rs.vreg[component] = e.b.localid_comp[component];
        e.rs.vreg[3] = e.b.localid_comp[0];
        EXPECT_EQ(e.proof(p).size(), 1u);
        e.rs.vreg_source_marks[component].memory = true;
        EXPECT_TRUE(e.proof(p).empty());
    }
}
TEST(ComputeQuadOr, AlteredDecodedSourceAndRecursiveSliceCannotReuseProof) {
    const auto p = fixture::program();
    std::vector<Rdna2Inst> ins;
    rdna2_walk(p.data(), p.size(), ins);
    Entry e;
    ASSERT_EQ(proven_compute_quad_or_sources(e.b, e.rs, ins, p.data(), p.size(), 0).size(), 2u);
    ins.erase(ins.begin());
    EXPECT_TRUE(proven_compute_quad_or_sources(e.b, e.rs, ins, p.data(), p.size(), 0).empty());
    rdna2_walk(p.data(), p.size(), ins = {});
    for (auto& in : ins)
        if (compute_dpp_quad_or(in)) {
            in.src[0].value = 90;
            break;
        }
    EXPECT_TRUE(proven_compute_quad_or_sources(e.b, e.rs, ins, p.data(), p.size(), 0).empty());
}
TEST(ComputeQuadOr, ExecIgnoringWriteLaneRetiresInitializationUntilFullReplacement) {
    auto program = fixture::program();
    std::vector<Rdna2Inst> ins;
    rdna2_walk(program.data(), program.size(), ins);
    const auto site = std::find_if(ins.begin(), ins.end(), compute_dpp_quad_or);
    ASSERT_NE(site, ins.end());
    // Dynamic WRITELANE replaces its selected physical lane even with empty EXEC. Neither
    // an unproved scalar payload nor a subsequent plain VGPR copy authenticates that lane.
    const std::vector<uint32_t> prefix{0xbe800381u, 0xbefe0480u, 0xd7610001u, 0x0000005au,
                                       0xbefe04c1u, 0x7e0a0301u, 0x7e020305u};
    program.insert(program.begin() + site->pc, prefix.begin(), prefix.end());
    Entry entry;
    EXPECT_TRUE(entry.proof(program).empty());
    const auto rejected = fixture::address();
    EXPECT_TRUE(fixture::compile(program, fixture::table(), fixture::config(64), rejected).empty());
    EXPECT_NE(last_terminal_reject_reason(rejected).find("dpp-quad-or-source-uninitialized"),
              std::string::npos);
    // A later full genuine writer terminates this unproved generation without laundering it.
    program.insert(program.begin() + site->pc + prefix.size(), {0x7e0202ffu, 42u});
    Entry replaced;
    EXPECT_EQ(replaced.proof(program).size(), 2u);
    const auto admitted = fixture::address();
    EXPECT_FALSE(fixture::compile(program, fixture::table(), fixture::config(64), admitted).empty())
        << last_terminal_reject_reason(admitted);
}
