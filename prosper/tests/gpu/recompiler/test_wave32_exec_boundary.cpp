// A masked word stays genuine inside the same guest EXEC after an authenticated spill/restore.
// Losing that domain rejects inactive comparisons; inventing it after a widening writes bad data.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/compute_coverage_inputs.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {
using namespace prosper::gpu;
namespace coverage_inputs = prosper::test::coverage_inputs;
ComputeShaderConfig profile() {
    ComputeShaderConfig result;
    result.wave_size = 32;
    result.native_subgroup_size = 32;
    return result; // original Local64 default: two complete logical Wave32 waves; no raw fields
}
ShaderResourceTable table(bool original) {
    ShaderResource buffer{};
    buffer.cls = original ? ResourceClass::VertexBuffer : ResourceClass::ConstantBuffer;
    buffer.binding = 3;
    buffer.sgpr_base = 8;
    buffer.stride = original ? 16 : 4;
    buffer.format = original ? DataFormat::Float32 : DataFormat::Uint32;
    buffer.num_components = original ? 4 : 1;
    ShaderResourceTable result;
    result.resources.push_back(buffer);
    return result;
}
std::vector<uint32_t> original_code() {
    const auto& head = coverage_inputs::compute_cfg_dispatch_saved_mask_not_boundary;
    auto result = std::vector<uint32_t>(head.begin(), head.end());
    result.insert(result.end(), std::begin(coverage_inputs::dispatch_tail),
                  std::end(coverage_inputs::dispatch_tail));
    return result;
}
std::vector<uint32_t> store_code(bool restore_full) {
    const auto& head = coverage_inputs::compute_cfg_dispatch_saved_mask_not_boundary;
    std::vector<uint32_t> result{0x7e0602a5u, 0x7e100300u}; // v3=37; v8=original index
    if (restore_full) result.insert(result.end(), {0x7e000280u, 0x7e020280u});
    result.insert(result.end(), head.begin(), head.end());
    result.insert(result.end(), {
                                    0xbe840380u, // genuine bounded counter s4=0
                                    0x7e000280u,
                                    0x7e020300u, // masked CURRENT definitions
                                    0xd7610013u,
                                    0x00014a7eu, // WRITELANE v19,EXEC_LO,37
                                    0xd7610013u,
                                    0x0001507fu, // WRITELANE v19,EXEC_HI,40
                                    0xd760000eu,
                                    0x00014b13u, // READLANE s14,v19,37
                                    0xd760000fu,
                                    0x00015113u, // READLANE s15,v19,40
                                    0xbefe040eu,   // restore both actual CURRENT words
                                });
    if (restore_full) result.push_back(0xbefe03c1u);
    const uint32_t header = result.size();
    result.insert(result.end(),
                  {
                      0x7da40080u,   // CMPX_EQ_I32 0,v0: CURRENT may only shrink
                      0x7d020200u,   // CMP_EQ s0,v1 after the actual dispatcher reload
                      0xbf860004u,   // VCCZ skips counter body to store/END
                      0x80048104u,   // s_add_u32 s4,s4,1
                      0xbf0a8204u,   // CMP_GE s4,2
                      0xbf850001u,   // completed counter exits before the backedge
                      0xbf820000u | (static_cast<uint32_t>(-7) & 0xffffu),   // ordinary backedge
                      0xe0702000u,
                      0x80020308u,   // store v3 at original v8 index
                      0xbf810000u,
                  });
    EXPECT_EQ(header, result.size() - 10);
    return result;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, bool original, uint64_t address) {
    const auto resources = table(original);
    auto module = recompile_compute(code.data(), code.size(), &resources, profile(),
                                    {RecompileDiagnosticStage::Compute, address});
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (const char* directory = std::getenv("PROSPER_WAVE32_BOUNDARY_SPV_DIRECTORY")) {
        if (!module.empty()) {
            std::ofstream output(std::filesystem::path(directory) /
                                     (std::to_string(address) + ".spv"),
                                 std::ios::binary);
            output.write(reinterpret_cast<const char*>(module.data()),
                         static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
            EXPECT_TRUE(output.good());
        }
    }
    return module;
}
void observes(const std::vector<uint32_t>& code, uint32_t expected, uint64_t address) {
    const auto module = compile(code, false, address);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bool dispatcher = false;
    for (std::size_t at = 5; at < module.size();) {
        const auto size = module[at] >> 16;
        ASSERT_NE(size, 0u);
        ASSERT_LE(at + size, module.size());
        dispatcher |= (module[at] & 0xffffu) == 251;
        at += size;
    }
    EXPECT_TRUE(dispatcher) << "the producing module must exercise the actual CFG boundary";
    std::vector<uint32_t> result(64, 0xdeadbeefu);
    for (uint32_t subgroup = 0; subgroup < 2; ++subgroup) {
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 32;
        vm.native32_isolated_local64 = true;
        vm.native32_subgroup_index = subgroup;
        result = vm.run(32, result);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
    }
    ASSERT_EQ(result.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(result[lane], expected) << "lane=" << lane;
}
}   // namespace
TEST(Wave32ExecBoundary, TheOriginalFixtureKeepsItsBytesAndLaunchFields) {
    const auto code = original_code();
    const auto config = profile();
    EXPECT_EQ(config.local_x, 64u);
    EXPECT_EQ(config.wave_size, 32u);
    EXPECT_EQ(config.native_subgroup_size, 32u);
    EXPECT_FALSE(config.exact_thread_extent);
    EXPECT_TRUE(config.user_sgprs.empty());
    EXPECT_FALSE(compile(code, true, 0x5138000001ull).empty());
}
TEST(Wave32ExecBoundary, AnEmptyCurrentRestoreLeavesTheActualBufferUntouched) {
    observes(store_code(false), 0xdeadbeefu, 0x5138000002ull);
}
TEST(Wave32ExecBoundary, TheFirstHeaderConsumerKeepsItsReloadedCurrentOwner) {
    auto code = store_code(false);
    const auto cmpx = std::find(code.begin(), code.end(), 0x7da40080u);
    ASSERT_NE(cmpx, code.end());
    code.erase(cmpx);
    const auto back = std::find(code.begin(), code.end(), 0xbf82fff9u);
    ASSERT_NE(back, code.end());
    *back = 0xbf82fffau;   // the first header instruction is now the genuine v1 comparison
    observes(code, 0xdeadbeefu, 0x5138000004ull);
}
TEST(Wave32ExecBoundary, GenuineEntryDataAndAFullRestoreActuallyWriteTheBuffer) {
    const auto code = store_code(true);
    const auto producer = rdna2_decode_one(code.data(), code.size());
    ASSERT_EQ(producer.fmt, Rdna2Format::VOP1);
    ASSERT_EQ(producer.opcode, 1u);
    ASSERT_EQ(producer.dst.value, 3);
    ASSERT_EQ(producer.src[0].kind, OperandKind::InlineInt);
    ASSERT_EQ(producer.src[0].value, 37);
    observes(code, 37u, 0x5138000003ull);
}

namespace {
// v1 in the lanes no writer covers: hard on every route. s30 is not a user SGPR, so hardware never
// loads it. (A never-written v1 would not do: under #5185 it holds an arbitrary value hardware
// could hold, which a fresh predicate may read on any route, #5039.)
std::vector<uint32_t> hard_old_v1(std::vector<uint32_t> code) {
    code.insert(code.begin(), 0x7e02021eu);   // v_mov_b32 v1, s30
    return code;
}
void missing(const std::vector<uint32_t>& code, uint64_t address, bool original = false) {
    EXPECT_TRUE(compile(code, original, address).empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    const auto compare = std::find_if(code.begin(), code.end(), [](uint32_t word) {
        return word == 0x7d020200u || word == 0x7d840200u;
    });
    ASSERT_NE(compare, code.end());
    EXPECT_NE(reason.find("pc=" + std::to_string(compare - code.begin()) + " "), std::string::npos)
        << reason;
}
void before_read(std::vector<uint32_t>& code, std::initializer_list<uint32_t> effect) {
    const auto at = std::find(code.begin(), code.end(), 0xd760000eu);
    ASSERT_NE(at, code.end());
    code.insert(at, effect);
}
bpermute_oracle::Interpreter isolated(const std::vector<uint32_t>& module, uint32_t subgroup = 0) {
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 32;
    vm.native32_isolated_local64 = true;
    vm.native32_subgroup_index = subgroup;
    return vm;
}
}   // namespace
TEST(Wave32ExecBoundary, MissingVectorProducerStaysRefusedAtTheOriginalConsumer) {
    auto code = original_code();
    code[10] = 0xbf800000u;
    missing(hard_old_v1(code), 0x5138000100ull, true);
}
TEST(Wave32ExecBoundary, MissingWriterSourceCannotBorrowCurrent) {
    auto code = store_code(false);
    *std::find(code.begin(), code.end(), 0x7e020300u) = 0x7e02021eu;
    missing(code, 0x5138000101ull);
}
TEST(Wave32ExecBoundary, AFullRestoreCannotReviveDataOnlyWrittenInEmptyExec) {
    auto code = store_code(false);
    const auto at = std::find(code.begin(), code.end(), 0x7da40080u);
    ASSERT_NE(at, code.end());
    code.insert(at, 0xbefe03c1u);
    missing(hard_old_v1(code), 0x5138000102ull);
}
TEST(Wave32ExecBoundary, SwappedTransportWordsCannotPreserveCurrent) {
    auto code = store_code(false);
    std::iter_swap(std::find(code.begin(), code.end(), 0x00014b13u),
                   std::find(code.begin(), code.end(), 0x00015113u));
    missing(hard_old_v1(code), 0x5138000103ull);
}
TEST(Wave32ExecBoundary, PhysicalLowSlotAliasEndsTheActualEquality) {
    auto code = store_code(false);
    before_read(code, {0xd7610013u, 0x00010a81u});   // lane5 overwrites physical37&31
    missing(hard_old_v1(code), 0x5138000104ull);
}
TEST(Wave32ExecBoundary, PhysicalHighSlotAliasEndsTheActualEquality) {
    auto code = store_code(false);
    before_read(code, {0xd7610013u, 0x00011081u});   // lane8 overwrites physical40&31
    missing(hard_old_v1(code), 0x5138000105ull);
}
TEST(Wave32ExecBoundary, ChangedExecGenerationCannotUseTheOldSlotTransport) {
    auto code = store_code(false);
    before_read(code, {0x7da40080u});
    missing(hard_old_v1(code), 0x5138000106ull);
}
TEST(Wave32ExecBoundary, MissingSourceOnTheBackedgeCannotBorrowThePreheader) {
    auto code = store_code(false);
    const auto at = std::find(code.begin(), code.end(), 0x80048104u);
    ASSERT_NE(at, code.end());
    code.insert(at, 0x7e02021eu);
    // The insertion lies inside the loop: keep its VCC exit and scalar exit/backedge targets exact.
    const auto exit = std::find(code.begin(), code.end(), 0xbf860004u);
    ASSERT_NE(exit, code.end());
    *exit = 0xbf860005u;
    const auto back = std::find(code.begin(), code.end(), 0xbf82fff9u);
    ASSERT_NE(back, code.end());
    *back = 0xbf82fff8u;
    missing(code, 0x5138000107ull);
}
TEST(Wave32ExecBoundary, EachNativeSubgroupWritesOnlyItsOwnActualLocal64Slots) {
    const auto module = compile(store_code(true), false, 0x5138000200ull);
    ASSERT_FALSE(module.empty());
    auto vm = isolated(module);
    const auto result = vm.run(32, std::vector<uint32_t>(64, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(result.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(result[lane], lane < 32 ? 37u : 0xdeadbeefu);
}
TEST(Wave32ExecBoundary, TheSecondSubgroupCannotBeInventedWithoutOracleOptIn) {
    const auto module = compile(store_code(true), false, 0x5138000201ull);
    ASSERT_FALSE(module.empty());
    auto vm = isolated(module, 1);
    vm.native32_isolated_local64 = false;
    EXPECT_TRUE(vm.run(32, std::vector<uint32_t>(64)).empty());
    EXPECT_EQ(vm.error, "missing isolated native32 opt-in");
}
TEST(Wave32ExecBoundary, AnOutOfRangeSubgroupIsRejectedByTheOracle) {
    const auto module = compile(store_code(true), false, 0x5138000202ull);
    auto vm = isolated(module, 2);
    EXPECT_TRUE(vm.run(32, std::vector<uint32_t>(64)).empty());
    EXPECT_EQ(vm.error, "unproved isolated native32 Local64 instance");
}
TEST(Wave32ExecBoundary, OracleLocalSizeMustMatchTheActualSixtyFourWorkerModule) {
    const auto module = compile(store_code(true), false, 0x5138000203ull);
    auto vm = isolated(module);
    vm.local_sizes[vm.main] = {32, 1, 1};
    EXPECT_TRUE(vm.run(32, std::vector<uint32_t>(64)).empty());
    EXPECT_EQ(vm.error, "unproved isolated native32 Local64 instance");
}
TEST(Wave32ExecBoundary, OracleCannotFlattenTwoSubgroupsIntoOneCollective) {
    const auto module = compile(store_code(true), false, 0x5138000204ull);
    auto vm = isolated(module);
    EXPECT_TRUE(vm.run(64, std::vector<uint32_t>(64)).empty());
    EXPECT_EQ(vm.error, "unproved isolated native32 Local64 instance");
}
TEST(Wave32ExecBoundary, OracleDeclinesEvenUnusedWorkgroupSharedStorage) {
    const auto module = compile(store_code(true), false, 0x5138000205ull);
    auto vm = isolated(module);
    ASSERT_FALSE(vm.globals.empty());
    auto shared = vm.globals.front();
    shared.a[2] = 4;
    vm.globals.push_back(shared);
    EXPECT_TRUE(vm.run(32, std::vector<uint32_t>(64)).empty());
    EXPECT_EQ(vm.error, "isolated native32 shared storage");
}
TEST(Wave32ExecBoundary, OracleDeclinesEvenAnUnreachedWorkgroupBarrier) {
    const auto module = compile(store_code(true), false, 0x5138000206ull);
    auto vm = isolated(module);
    vm.ins.push_back({224, {0, 0, 0}});
    EXPECT_TRUE(vm.run(32, std::vector<uint32_t>(64)).empty());
    EXPECT_EQ(vm.error, "isolated native32 barrier or atomic");
}
TEST(Wave32ExecBoundary, OracleDeclinesAStorageReadThatCouldCommunicateBetweenSubgroups) {
    auto code = store_code(true);
    const auto at = std::find(code.begin(), code.end(), 0xe0702000u);
    ASSERT_NE(at, code.end());
    code.insert(at, {0xe0302000u, 0x80020408u});
    const auto module = compile(code, false, 0x5138000207ull);
    ASSERT_FALSE(module.empty());
    auto vm = isolated(module);
    EXPECT_TRUE(vm.run(32, std::vector<uint32_t>(64)).empty());
    EXPECT_EQ(vm.error, "isolated native32 storage read");
}
TEST(Wave32ExecBoundary, OracleRejectsAStoreToAnotherSubgroupsOrLanesSlot) {
    auto code = store_code(true);
    *std::find(code.begin(), code.end(), 0x80020308u) = 0x80020300u;   // all lanes would race slot0
    const auto module = compile(code, false, 0x5138000208ull);
    ASSERT_FALSE(module.empty());
    auto vm = isolated(module);
    EXPECT_TRUE(vm.run(32, std::vector<uint32_t>(64)).empty());
    EXPECT_EQ(vm.error, "isolated native32 cross-subgroup store");
}

TEST(Wave32ExecBoundary, ApproximateReadfirstCannotAuthenticateThisNewDataRecovery) {
    auto code = store_code(false);
    code.insert(code.begin(), 0x7e400500u);   // s32=this lane fallback
    missing(hard_old_v1(code), 0x5138000300ull);
}
TEST(Wave32ExecBoundary, ARealScalarMoveDoesNotDisableTheMatchedRecovery) {
    auto code = store_code(false);
    code.insert(code.begin(), 0xbea00380u);   // genuine s32=0
    observes(code, 0xdeadbeefu, 0x5138000301ull);
}

namespace {
std::vector<uint32_t> scalar_relay_loop(bool readfirst) {
    auto code = store_code(false);
    // Keep the actual CFG loop/transport, but enable a genuine nonempty subset instead of
    // the empty-prefix fixture. READFIRST has no exact native32 service on this route.
    std::fill(code.begin() + 2, code.begin() + 9, 0xbf800000u);
    code[2] = readfirst ? 0x7e400500u : 0xbea00381u;   // s32 = selected first / genuine1
    code[3] = 0x7daa0080u;   // EXEC &= local-ID !=0
    code[4] = 0xbe800380u;   // genuine s0=0 for the original s0/v1 comparison
    code[10] = 0xbf800000u;   // retain genuine local-ID, do not replace it with0
    code[11] = 0x7e020220u;   // masked v1 = s32
    *std::find(code.begin(), code.end(), 0x7d020200u) = 0x7d840200u;   // exact integer comparison
    *std::find(code.begin(), code.end(), 0x7da40080u) = 0x7daa0080u;
    *std::find(code.begin(), code.end(), 0xbf860004u) = 0xbf860006u;   // VCCZ skips the store
    return code;
}
}   // namespace
TEST(Wave32ExecBoundary, AnApproximateScalarRelayCannotAuthenticateANonemptyCurrentWord) {
    missing(hard_old_v1(scalar_relay_loop(true)), 0x5138000302ull);
}
TEST(Wave32ExecBoundary, TheMatchedRealScalarValueProducesTheActualEmptySink) {
    observes(scalar_relay_loop(false), 0xdeadbeefu, 0x5138000303ull);
}

TEST(Wave32ExecBoundary, ACurrentTransportInsideTheHeaderKeepsItsActualInstructionOwner) {
    auto code = store_code(false);
    const auto compare = std::find(code.begin(), code.end(), 0x7d020200u);
    ASSERT_NE(compare, code.end());
    code.insert(compare, {0xd7610013u, 0x00014a7eu, 0xd7610013u, 0x0001507fu, 0xd760000eu,
                          0x00014b13u, 0xd760000fu, 0x00015113u, 0xbefe040eu});
    const auto back = std::find(code.begin(), code.end(), 0xbf82fff9u);
    ASSERT_NE(back, code.end());
    *back = 0xbf82fff0u;   // backedge still enters the original CMPX
    observes(code, 0xdeadbeefu, 0x5138000400ull);
}

TEST(Wave32ExecBoundary, UnprovidedImplicitVccCannotPublishTheNewCurrentDataFact) {
    auto code = store_code(false);
    *std::find(code.begin(), code.end(), 0x7e020300u) = 0x02020680u;   // CNDMASK v1,0,v3,initialVCC
    missing(hard_old_v1(code), 0x5138000401ull);
}

TEST(Wave32ExecBoundary, UnprovidedImplicitCarryCannotPublishTheNewCurrentDataFact) {
    auto code = store_code(false);
    *std::find(code.begin(), code.end(), 0x7e020300u) =
        0x50020680u;   // ADD_CO_CI v1,0,v3,initialVCC
    missing(hard_old_v1(code), 0x5138000402ull);
}
TEST(Wave32ExecBoundary, ARealNumericWordKeepsTheMatchedCurrentDataRoute) {
    auto code = store_code(false);
    *std::find(code.begin(), code.end(), 0x7e020300u) =
        0x7e0202a5u;   // v1=37, actual complete numeric writer
    observes(code, 0xdeadbeefu, 0x5138000403ull);
}

namespace {
std::vector<uint32_t> explicit_mask_writer(bool carry, bool genuine) {
    auto code = store_code(false);
    const auto at = std::find(code.begin(), code.end(), 0x7e020300u);
    EXPECT_NE(at, code.end());
    const auto index = at - code.begin();
    code.erase(at);
    code.insert(code.begin() + index, {carry ? 0xd5286a01u : 0xd1010001u, 0x01aa0680u});
    if (genuine) {
        code[index + 1] =
            0x00820680u;   // explicit genuine SRC2 s[32:33], not the prefix's overwritten s20/s21
        code.insert(code.begin(), {0xbe800380u, 0x7d840080u, 0xbea00480u, 0x7e020280u});
    }
    // The matched control supplies real mask DATA words and a genuine pre-entry v1 word.
    // It pins unchanged ordinary mask lowering; this new numeric proof does not grant mask DATA.
    return code;
}
std::vector<uint32_t> scc_source_writer(bool carry, bool genuine) {
    auto code = store_code(false);
    *std::find(code.begin(), code.end(), 0x7e020300u) = 0x7e020220u;   // masked v1=s32
    code.insert(code.begin(), carry ? 0x82208080u : 0x85208180u);   // ADDC / CSELECT reads SCC
    if (genuine)
        code.insert(code.begin(), {0xbe800380u, 0xbf060000u});   // actual scalar SCC producer
    return code;
}
std::vector<uint32_t> scc_branch_writer(bool genuine) {
    auto code = store_code(false);
    const uint32_t arms[]{0xbf840002u, 0x7e020281u, 0xbf820001u, 0x7e020280u};
    code.insert(code.begin(), std::begin(arms), std::end(arms));
    if (genuine) code.insert(code.begin(), {0xbe800380u, 0xbf060000u});
    return code;
}
}   // namespace
TEST(Wave32ExecBoundary, ExplicitCndmaskCannotUseUnprovidedVccToPublishCurrentData) {
    missing(explicit_mask_writer(false, false), 0x5138000500ull);
}
TEST(Wave32ExecBoundary, ExplicitCarryCannotUseUnprovidedVccToPublishCurrentData) {
    const auto code = explicit_mask_writer(true, false);
    EXPECT_TRUE(compile(code, false, 0x5138000501ull).empty());
    const auto reason = last_terminal_reject_reason(0x5138000501ull);
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=11 "), std::string::npos) << reason;
}
TEST(Wave32ExecBoundary, RealMaskWordsAndGenuineEntryDataKeepTheOrdinaryCndmaskControl) {
    observes(explicit_mask_writer(false, true), 0xdeadbeefu, 0x5138000502ull);
}
TEST(Wave32ExecBoundary, RealMaskWordsAndGenuineEntryDataKeepTheOrdinaryCarryControl) {
    observes(explicit_mask_writer(true, true), 0xdeadbeefu, 0x5138000503ull);
}
TEST(Wave32ExecBoundary, CselectCannotLaunderUnprovidedEntrySccThroughTheNewVectorWord) {
    missing(scc_source_writer(false, false), 0x5138000504ull);
}
TEST(Wave32ExecBoundary, AddcCannotLaunderUnprovidedEntrySccThroughTheNewVectorWord) {
    missing(scc_source_writer(true, false), 0x5138000505ull);
}
TEST(Wave32ExecBoundary, ARealScalarCompareSuppliesTheCselectControl) {
    observes(scc_source_writer(false, true), 0xdeadbeefu, 0x5138000506ull);
}
TEST(Wave32ExecBoundary, ARealScalarCompareSuppliesTheCarryControl) {
    observes(scc_source_writer(true, true), 0xdeadbeefu, 0x5138000507ull);
}
TEST(Wave32ExecBoundary, AScalarBranchNeedsRealControlEvenWhenBothArmsWriteTheWord) {
    const auto module = compile(scc_branch_writer(false), false, 0x5138000508ull);
    EXPECT_TRUE(module.empty());
    const auto reason = last_terminal_reject_reason(0x5138000508ull);
    EXPECT_NE(reason.find("reject"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=0 "), std::string::npos) << reason;
}
TEST(Wave32ExecBoundary, ARealScalarCompareKeepsBothLiteralBranchInputsGenuine) {
    observes(scc_branch_writer(true), 0xdeadbeefu, 0x5138000509ull);
}

namespace {
std::vector<uint32_t> vcc_control_code(bool produced) {
    // CMPX_NE0 narrows EXEC without producing VCC; literal writers retain distinct branch arms.
    std::vector<uint32_t> result{
        0xbe840380u, 0x7e0602a5u, 0x7e100300u, 0xbe800380u, 0x7daa0080u, 0xbf860002u, 0x7e020281u,
        0xbf820001u, 0x7e020280u, 0xd7610013u, 0x00014a7eu, 0xd7610013u, 0x0001507fu, 0xd760000eu,
        0x00014b13u, 0xd760000fu, 0x00015113u, 0xbefe040eu, 0x7d840200u, 0xbf860006u, 0x80048104u,
        0xbf0a8204u, 0xbf850001u, 0xbf82fffau, 0xe0702000u, 0x80020308u, 0xbf810000u};
    if (produced) result.insert(result.begin() + 5, 0x7d840080u);   // genuine CMP_EQ0,v0
    return result;
}
}   // namespace
TEST(Wave32ExecBoundary, UnprovidedVccCannotChooseNewCurrentDataWriters) {
    missing(hard_old_v1(vcc_control_code(false)), 0x5138000320ull);
}
TEST(Wave32ExecBoundary, ARealVccCompareKeepsTheMatchedCurrentDataWriters) {
    const auto code = vcc_control_code(true);
    const auto cmpx = rdna2_decode_one(code.data() + 4, code.size() - 4);
    ASSERT_EQ(cmpx.fmt, Rdna2Format::VOPC);
    ASSERT_TRUE(vopc_is_cmpx(cmpx.opcode));
    const auto module = compile(code, false, 0x5138000321ull);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x5138000321ull);
    std::vector<uint32_t> result(64, 0xdeadbeefu);
    for (uint32_t subgroup = 0; subgroup < 2; ++subgroup) {
        auto vm = isolated(module, subgroup);
        result = vm.run(32, result);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
    }
    ASSERT_EQ(result.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(result[lane], lane == 0 ? 0xdeadbeefu : 37u) << "lane=" << lane;
}

namespace {
std::vector<uint32_t> vcc_relay_control_code(bool explicit_mask, bool real_move) {
    auto code = vcc_control_code(false);
    if (explicit_mask && !real_move) {
        code.insert(code.begin() + 3, {0xd1010002u, 0x01aa0680u});
        code.insert(code.begin() + 7, 0x7d840480u);   // genuine-looking CMP_EQ0,v2
    } else {
        code.insert(code.begin() + 3, real_move ? 0x7e040280u : 0x02040680u);
        code.insert(code.begin() + 6, 0x7d840480u);
    }
    return code;
}
}   // namespace
TEST(Wave32ExecBoundary, AFullExecImplicitMaskRelayCannotAuthenticateLaterControl) {
    missing(hard_old_v1(vcc_relay_control_code(false, false)), 0x5138000330ull);
}
TEST(Wave32ExecBoundary, AFullExecExplicitMaskRelayCannotAuthenticateLaterControl) {
    const auto code = vcc_relay_control_code(true, false);
    EXPECT_TRUE(compile(code, false, 0x5138000331ull).empty());
    const auto reason = last_terminal_reject_reason(0x5138000331ull);
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=7 "), std::string::npos) << reason;
}
TEST(Wave32ExecBoundary, AnIndependentLiteralAuthenticatesTheMatchedRelayControl) {
    observes(vcc_relay_control_code(false, true), 0xdeadbeefu, 0x5138000332ull);
}

TEST(Wave32ExecBoundary, AWholeExecVectorCopyCannotEraseTheUnprovidedMaskDependency) {
    auto code = vcc_relay_control_code(false, false);
    code.insert(code.begin() + 4, 0x7e080302u);   // v4=v2 while EXEC is still full
    *std::find(code.begin(), code.end(), 0x7d840480u) = 0x7d840880u;
    missing(hard_old_v1(code), 0x5138000333ull);
}
TEST(Wave32ExecBoundary, OverwritingTheRelayDoesNotRepairTheUnprovedReadHistory) {
    auto code = vcc_relay_control_code(false, false);
    code.insert(code.begin() + 4,
                0x7e040280u);   // independent overwrite after the unsupported read
    missing(hard_old_v1(code), 0x5138000334ull);
}
TEST(Wave32ExecBoundary, TheMatchedOtherLiteralActuallyWritesTheActiveBuffer) {
    auto code = vcc_relay_control_code(false, true);
    code[3] = 0x7e040281u;
    const auto module = compile(code, false, 0x5138000335ull);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x5138000335ull);
    std::vector<uint32_t> result(64, 0xdeadbeefu);
    for (uint32_t subgroup = 0; subgroup < 2; ++subgroup) {
        auto vm = isolated(module, subgroup);
        result = vm.run(32, result);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
    }
    ASSERT_EQ(result.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(result[lane], lane == 0 ? 0xdeadbeefu : 37u) << "lane=" << lane;
}

TEST(Wave32ExecBoundary, AnActuallyProducedImplicitMaskKeepsTheMatchedNumericRelay) {
    auto code = vcc_relay_control_code(false, false);
    code.insert(code.begin() + 3,
                0x7d840080u);   // actual VCC producer before the full-EXEC CNDMASK
    observes(code, 0xdeadbeefu, 0x5138000336ull);
}
