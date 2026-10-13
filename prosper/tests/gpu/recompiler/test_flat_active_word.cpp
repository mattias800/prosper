// A complete FLAT load defines its destination for the lanes of its actual EXEC. A fresh
// comparison may consume those lanes, but a missing address or restored inactive lanes cannot
// authenticate a predicate. These tests pin the public compiler, not device execution.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;

std::vector<uint32_t> program() {
    return {
        0x7e02020cu, // v1 = genuine pointer low s12
        0x7e04020du, // v2 = genuine pointer high s13
        0xbe84047eu, // save the complete EXEC in s[4:5]
        0x7dea0080u, // narrow EXEC by comparing actual local-ID v0
        0xdc308000u, 0x037d0001u, // global_load_dword v3, v[1:2]
        0x7d280680u, // compare the loaded v3
        0xbf810000u,
    };
}

struct Compilation {
    std::vector<uint32_t> module;
    std::string reason;
};

Compilation compile(const std::vector<uint32_t>& code, bool dispatcher = false,
                    uint32_t native_width = 64, bool partial = false) {
    ComputeShaderConfig config;
    config.user_sgprs.assign(14, 0);
    config.user_sgprs[12] = 0x100000u;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = native_width;
    config.force_exchange_dispatcher = dispatcher;
    prosper::test::complete_compute_launch(config);
    if (partial) {
        config.exact_thread_extent = true;
        config.threads_x = 151;
        config.dispatch_launch.known = 0;
    }
    ShaderResourceTable table;
    ShaderResource window;
    window.cls = ResourceClass::ConstantBuffer;
    window.format = DataFormat::Uint32;
    window.num_components = 1;
    window.binding = 3;
    window.gpu_addr = 0x100000u;
    window.size = 256;
    window.stride = 4;
    window.flat_base_sgpr = 12;
    // Every control keeps the original packet PC unless it inserts an instruction before it.
    for (std::size_t pc = 0; pc + 1 < code.size(); ++pc) {
        if ((code[pc] & 0xfc000000u) == 0xdc000000u || (code[pc] & 0xfc000000u) == 0xe0000000u ||
            (code[pc] & 0xfc000000u) == 0xe8000000u) {
            window.fetch_pc = static_cast<uint32_t>(pc);
            break;
        }
    }
    table.resources.push_back(window);
    static uint64_t next = 0x5060000100ull;
    const uint64_t address = next++;
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

void dispatcher_module(const Compilation& result) {
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bool switch_found = false;
    for (std::size_t word = 5; word < result.module.size();) {
        const auto count = result.module[word] >> 16;
        ASSERT_NE(count, 0u);
        ASSERT_LE(word + count, result.module.size());
        switch_found |= (result.module[word] & 0xffffu) == 251; // OpSwitch
        word += count;
    }
    EXPECT_TRUE(switch_found);
}

void missing(const Compilation& result, uint32_t pc, bool dispatcher = false) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(
        result.reason.find(dispatcher ? "mode=unresolved-operand" : "fabricated-fresh-mask-source"),
        std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reason;
}
} // namespace

TEST(FlatActiveWord, CompleteLoadedActiveLanesCanSupplyTheComparison) {
    const auto result = compile(program(), true);
    dispatcher_module(result);
}

TEST(FlatActiveWord, RestoringInactiveLanesDoesNotDefineTheirLoadedWord) {
    auto code = program();
    code.insert(code.begin() + 6, 0xbefe0404u); // restore saved full EXEC before the comparison
    missing(compile(code, true), 7, true);
}

TEST(FlatActiveWord, MissingLowAddressCannotSupplyTheComparison) {
    auto code = program();
    code[0] = 0xbf800000u;
    missing(compile(code, true), 6, true);
}

TEST(FlatActiveWord, MissingHighAddressCannotSupplyTheComparison) {
    auto code = program();
    code[1] = 0xbf800000u;
    missing(compile(code, true), 6, true);
}

TEST(FlatActiveWord, ActiveAddressWordsSurviveARealDispatcherEdge) {
    const std::vector<uint32_t> code{
        0xbe84047eu, // save full EXEC before the addresses exist
        0x7dea0080u, // narrow using genuine v0
        0x7e02020cu, 0x7e04020du, // define both pointer words under the narrowed EXEC
        0xbf060000u, 0xbf850001u, 0xbf800000u, // genuine SCC and a conditional CFG edge
        0xdc308000u, 0x037d0001u, // load v3 in the same active domain
        0x7d280680u, 0xbf810000u,
    };
    const auto result = compile(code, true);
    dispatcher_module(result);
}

TEST(FlatActiveWord, WideningAtTheDispatcherEdgeDoesNotCreateAddressWords) {
    const std::vector<uint32_t> code{
        0xbe84047eu, 0x7dea0080u, 0x7e02020cu, 0x7e04020du, 0xbf060000u, 0xbf850001u, 0xbf800000u,
        0xbefe0404u, // restore the wider mask whose inactive pointer words never existed
        0xdc308000u, 0x037d0001u, 0x7d280680u, 0xbf810000u,
    };
    missing(compile(code, true), 10, true);
}

namespace {
std::vector<uint32_t> both_arm_addresses() {
    return {
        0xbe84047eu, 0x7dea0080u, 0x7e02020cu, 0xbf060000u,
        0xbf850002u, // both scalar successors reach the load
        0x7e04020du, 0xbf820001u,
        0x7e04020du, // both arms independently define the current high pointer word
        0xdc308000u, 0x037d0001u, 0x7d280680u, 0xbf810000u,
    };
}

std::vector<uint32_t> backedge_address(bool genuine) {
    return {
        0xbe84047eu,
        0x7dea0080u,
        0x7e02020cu,
        0x7e04020du,
        0xbe860380u, // genuine scalar loop counter s6=0
        0xdc308000u,
        0x037d0001u,
        0x7d280680u,
        genuine ? 0x7e04020du : 0x7e040309u, // replace v2 from genuine s13 or absent v9
        0x80068106u,
        0xbf0a8206u,
        0xbf85fff9u, // counter++; counter<2; backedge -> load
        0xbf810000u,
    };
}
} // namespace

TEST(FlatActiveWord, BothPredecessorsSupplyCurrentAddressWords) {
    dispatcher_module(compile(both_arm_addresses(), true));
}

TEST(FlatActiveWord, OnePredecessorCannotSupplyTheJoinedAddress) {
    auto code = both_arm_addresses();
    code[7] = 0xbf800000u;
    missing(compile(code, true), 10, true);
}

TEST(FlatActiveWord, ASecondNarrowingRetainsCurrentAddressWords) {
    auto code = both_arm_addresses();
    code.insert(code.begin() + 8, 0x7dea0080u);
    dispatcher_module(compile(code, true));
}

TEST(FlatActiveWord, RestoringTheSavedCurrentMaskRetainsAddressWords) {
    auto code = both_arm_addresses();
    code.insert(code.begin() + 8, {0xbe86047eu, 0x7dea0080u, 0xbefe0406u});
    dispatcher_module(compile(code, true));
}

TEST(FlatActiveWord, GenuineBackedgeReplacementRetainsAddressWords) {
    dispatcher_module(compile(backedge_address(true), true));
}

TEST(FlatActiveWord, MissingBackedgeReplacementCannotReviveOldAddressWords) {
    missing(compile(backedge_address(false), true), 7, true);
}

TEST(FlatActiveWord, PortableDispatcherRetainsCurrentAddressWords) {
    dispatcher_module(compile(both_arm_addresses(), true, 0));
}

TEST(FlatActiveWord, PortableDispatcherRejectsMissingJoinedAddress) {
    auto code = both_arm_addresses();
    code[7] = 0xbf800000u;
    missing(compile(code, true, 0), 10, true);
}

TEST(FlatActiveWord, PartialNativeLaunchKeepsOnlyActualCurrentLanes) {
    dispatcher_module(compile(both_arm_addresses(), true, 64, true));
}

TEST(FlatActiveWord, PartialNativeLaunchCannotDefineAMissingWord) {
    auto code = both_arm_addresses();
    code[7] = 0xbf800000u;
    missing(compile(code, true, 64, true), 10, true);
}

namespace {
std::vector<uint32_t> saved_backedge_address(bool genuine) {
    return {
        0xbe84047eu, 0x7dea0080u, 0x7e02020cu, 0x7e04020du,
        0xbe86047eu, 0xbe880380u, // save current EXEC in s[6:7], genuine s8=0
        0xdc308000u, 0x037d0001u, 0x7d280680u, genuine ? 0x7e04020du : 0x7e040309u,
        0xbefe0406u, // this identical mask value must not revive an overwritten DATA generation
        0x80088108u, 0xbf0a8208u, 0xbf85fff8u, 0xbf810000u,
    };
}
} // namespace

TEST(FlatActiveWord, CurrentSavedMaskKeepsGenuineBackedgeReplacement) {
    dispatcher_module(compile(saved_backedge_address(true), true));
}

TEST(FlatActiveWord, CurrentSavedMaskCannotReviveMissingBackedgeReplacement) {
    missing(compile(saved_backedge_address(false), true), 8, true);
}

TEST(FlatActiveWord, LaterEntryWriteRefreshesItsPreviouslySavedMask) {
    const std::vector<uint32_t> code{
        0xbe84047eu, 0x7dea0080u, 0xbefe0404u, // return to exactly the actual initial EXEC
        0x7e02020cu, 0x7e04020du, // address words now cover that entry mask
        0x7dea0080u, 0xbefe0404u, // save predates the address, but the mask value is unchanged
        0xbf060000u, 0xbf850001u, 0xbf800000u, 0xdc308000u, 0x037d0001u, 0x7d280680u, 0xbf810000u,
    };
    dispatcher_module(compile(code, true));
}

namespace {
std::vector<uint32_t> entry_both_arms() {
    return {
        0xbe84047eu, 0x7e02020cu, 0xbf060000u, 0xbf850002u, 0x7e04020du, 0xbf820001u, 0x7e04020du,
        0x7dea0080u, 0xbefe0404u, 0xdc308000u, 0x037d0001u, 0x7d280680u, 0xbf810000u,
    };
}
} // namespace

TEST(FlatActiveWord, BothEntryArmsRefreshTheOldSavedMask) {
    dispatcher_module(compile(entry_both_arms(), true));
}

TEST(FlatActiveWord, OneEntryArmCannotRefreshTheOldSavedMask) {
    auto code = entry_both_arms();
    code[6] = 0xbf800000u;
    missing(compile(code, true), 11, true);
}

namespace {
std::vector<uint32_t> buffer_address() {
    return {
        0xbe84047eu, 0x7dea0080u, 0x7e020280u, // genuine active offset v1=0
        0xe0301000u, 0x80030301u, // buffer_load_dword v3,v1,s[12:15],0 offen
        0xbf060000u, 0xbf850001u, 0xbf800000u, 0x7d280680u, 0xbf810000u,
    };
}
} // namespace

TEST(FlatActiveWord, BoundBufferAlsoRequiresTheActualCurrentAddressWord) {
    dispatcher_module(compile(buffer_address(), true));
}

TEST(FlatActiveWord, ABufferBindingCannotInventTheMissingAddressWord) {
    auto code = buffer_address();
    code[2] = 0xbf800000u;
    missing(compile(code, true), 8, true);
}

TEST(FlatActiveWord, GenuineScalarBufferOffsetIsRetained) {
    auto code = buffer_address();
    code[4] = 0x0c030301u; // SOFFSET=s12, independently genuine scalar DATA
    dispatcher_module(compile(code, true));
}

TEST(FlatActiveWord, MissingScalarBufferOffsetIsNotCurrentData) {
    auto code = buffer_address();
    code[4] = 0x14030301u; // absent SOFFSET=s20
    missing(compile(code, true), 8, true);
}

TEST(FlatActiveWord, FusedShiftAddDefinesItsActualActiveAddressWord) {
    auto code = program();
    code.erase(code.begin());
    code.insert(code.begin() + 3, {0xd7460001u, 0x0401000cu}); // v1=(s12<<0)+v0
    dispatcher_module(compile(code, true));
}

TEST(FlatActiveWord, FusedShiftAddCannotInventItsMissingScalarSource) {
    auto code = program();
    code.erase(code.begin());
    code.insert(code.begin() + 3, {0xd7460001u, 0x04010014u}); // absent s20
    missing(compile(code, true), 7, true);
}

namespace {
std::vector<uint32_t> saveexec_address(uint32_t opcode, bool parent_addresses) {
    std::vector<uint32_t> code{
        0xbe84047eu, // retain the actual parent EXEC
        0x7dea0080u, // active subset A from genuine local-ID v0
        0x7e02020cu,
        0x7e04020du, // pointer words exist only under A
        0xbe860004u | (opcode << 8), // s_*_saveexec_b64 s[6:7],s[4:5]
        0xdc308000u,
        0x037d0001u,
        0x7d280680u,
        0xbf810000u,
    };
    if (parent_addresses) {
        code[2] = code[3] = 0xbf800000u;
        code.insert(code.begin(), {0x7e02020cu, 0x7e04020du});
    }
    return code;
}
} // namespace

TEST(FlatActiveWord, AndSaveexecKeepsTheOldActiveAddressDomain) {
    dispatcher_module(compile(saveexec_address(0x24, false), true));
}

TEST(FlatActiveWord, Andn2SaveexecCannotBorrowTheOldActiveAddressDomain) {
    // ANDN2 activates parent & ~A. Neither address word was defined in those lanes.
    missing(compile(saveexec_address(0x27, false), true), 7, true);
}

TEST(FlatActiveWord, Andn2SaveexecRetainsGenuineParentAddressWords) {
    dispatcher_module(compile(saveexec_address(0x27, true), true));
}

TEST(FlatActiveWord, AliasingAndn2SaveexecCannotBorrowTheOldActiveAddressDomain) {
    auto code = saveexec_address(0x27, false);
    code[4] = 0xbe842704u; // destination aliases S0; the old parent must be read first
    missing(compile(code, true), 7, true);
}

namespace {
std::vector<uint32_t> whole_entry_destination(unsigned destination, unsigned missing_word) {
    auto code = program();
    if (missing_word) code[missing_word - 1] = 0xbf800000u;
    code[5] = (destination << 24u) | 0x007d0001u;
    code[6] = 0x7d280080u | (destination << 9u);
    if (destination == 3) code.insert(code.begin(), 0x7e060281u);
    return code;
}

void accepted(const Compilation& result, bool dispatcher) {
    if (dispatcher)
        dispatcher_module(result);
    else
        ASSERT_FALSE(result.module.empty()) << result.reason;
}

std::vector<uint32_t> whole_entry_buffer(bool typed, bool missing_address) {
    return {
        0x7e060281u, // whole-entry genuine old v3
        missing_address ? 0xbf800000u : 0x7e020280u,
        0xbe84047eu,
        0x7dea0080u,
        typed ? 0xe8b01000u : 0xe0301000u,
        0x80030301u,
        0x7d280680u,
        0xbf810000u,
    };
}
} // namespace

TEST(FlatActiveWord, WholeEntryDestinationStillRequiresTheLowAddressWord) {
    for (bool dispatcher : {false, true})
        for (unsigned destination : {3u, 0u}) {
            const auto code = whole_entry_destination(destination, 1);
            missing(compile(code, dispatcher), static_cast<uint32_t>(code.size() - 2), dispatcher);
        }
}

TEST(FlatActiveWord, WholeEntryDestinationStillRequiresTheHighAddressWord) {
    for (bool dispatcher : {false, true})
        for (unsigned destination : {3u, 0u}) {
            const auto code = whole_entry_destination(destination, 2);
            missing(compile(code, dispatcher), static_cast<uint32_t>(code.size() - 2), dispatcher);
        }
}

TEST(FlatActiveWord, WholeEntryDestinationKeepsARealAddressLoad) {
    for (bool dispatcher : {false, true})
        for (unsigned destination : {3u, 0u})
            accepted(compile(whole_entry_destination(destination, 0), dispatcher), dispatcher);
}

TEST(FlatActiveWord, OverlappingDestinationReadsTheOldGenuineAddressFirst) {
    for (bool dispatcher : {false, true})
        for (unsigned destination : {1u, 2u})
            accepted(compile(whole_entry_destination(destination, 0), dispatcher), dispatcher);
}

TEST(FlatActiveWord, OverlappingDestinationCannotCreateTheOtherAddressWord) {
    for (bool dispatcher : {false, true})
        for (unsigned destination : {1u, 2u}) {
            const auto code = whole_entry_destination(destination, destination == 1 ? 2 : 1);
            missing(compile(code, dispatcher), static_cast<uint32_t>(code.size() - 2), dispatcher);
        }
}

TEST(FlatActiveWord, RelayingTheNewLoadResultRetainsItsCurrentSourceValidity) {
    for (bool dispatcher : {false, true})
        for (unsigned missing_word : {0u, 1u, 2u}) {
            auto code = whole_entry_destination(3, missing_word);
            code.insert(code.end() - 2, 0x7e080303u); // v4 = the new loaded v3
            code[code.size() - 2] = 0x7d280880u;
            const auto result = compile(code, dispatcher);
            if (missing_word)
                missing(result, static_cast<uint32_t>(code.size() - 2), dispatcher);
            else
                accepted(result, dispatcher);
        }
}

TEST(FlatActiveWord, RelayedAddressMustHaveACurrentOriginalSource) {
    for (bool dispatcher : {false, true})
        for (bool supplied : {false, true}) {
            auto code = whole_entry_destination(3, 0);
            code[1] = 0x7e020309u; // address low v1 = v9, before the load writes v3
            code.insert(code.begin(), supplied ? 0x7e12020cu : 0xbf800000u);
            const auto result = compile(code, dispatcher);
            if (supplied)
                accepted(result, dispatcher);
            else
                missing(result, static_cast<uint32_t>(code.size() - 2), dispatcher);
        }
}

TEST(FlatActiveWord, DispatcherEdgeKeepsTheNewLoadSourceObligation) {
    for (unsigned missing_word : {0u, 1u, 2u}) {
        auto code = whole_entry_destination(3, missing_word);
        code.insert(code.end() - 2, {0xbf060000u, 0xbf850001u, 0xbf800000u});
        const auto result = compile(code, true);
        if (missing_word)
            missing(result, static_cast<uint32_t>(code.size() - 2), true);
        else
            dispatcher_module(result);
    }
}

TEST(FlatActiveWord, BufferOverwriteKeepsTheGenuineCurrentAddress) {
    for (bool dispatcher : {false, true})
        for (bool typed : {false, true})
            accepted(compile(whole_entry_buffer(typed, false), dispatcher), dispatcher);
}

TEST(FlatActiveWord, BufferOverwriteCannotBorrowItsGenuineOldDestination) {
    for (bool dispatcher : {false, true})
        for (bool typed : {false, true})
            missing(compile(whole_entry_buffer(typed, true), dispatcher), 6, dispatcher);
}

TEST(FlatActiveWord, BufferDispatcherEdgeRetainsTheCurrentAddressObligation) {
    for (bool typed : {false, true})
        for (bool missing_address : {false, true}) {
            auto code = whole_entry_buffer(typed, missing_address);
            code.insert(code.end() - 2, {0xbf060000u, 0xbf850001u, 0xbf800000u});
            const auto result = compile(code, true);
            if (missing_address)
                missing(result, 9, true);
            else
                dispatcher_module(result);
        }
}

TEST(FlatActiveWord, BothSelectedBufferAddressWordsMustBeCurrentBeforeOverwrite) {
    for (bool dispatcher : {false, true})
        for (bool typed : {false, true})
            for (unsigned destination : {3u, 0u})
                for (unsigned missing_word : {0u, 1u, 2u}) {
                    std::vector<uint32_t> code{
                        missing_word == 1 ? 0xbf800000u : 0x7e020280u,
                        missing_word == 2 ? 0xbf800000u : 0x7e040280u,
                        0xbe84047eu,
                        0x7dea0080u,
                        typed ? 0xe8b03000u : 0xe0303000u,
                        0x80030001u | (destination << 8u),
                        0x7d280080u | (destination << 9u),
                        0xbf810000u,
                    };
                    if (destination == 3) code.insert(code.begin(), 0x7e060281u);
                    const auto result = compile(code, dispatcher);
                    if (missing_word)
                        missing(result, static_cast<uint32_t>(code.size() - 2), dispatcher);
                    else
                        accepted(result, dispatcher);
                }
}

TEST(FlatActiveWord, AnUnusedBufferAddressFieldIsNotAMissingInput) {
    for (bool dispatcher : {false, true})
        for (bool typed : {false, true}) {
            auto code = whole_entry_buffer(typed, true);
            code[4] &= ~0x3000u; // neither OFFEN nor IDXEN consumes VADDR
            code[5] = 0x80030309u; // an absent v9 in the unused encoded field
            accepted(compile(code, dispatcher), dispatcher);
        }
}

namespace {
struct ScalarOffsetCase {
    bool typed;
    bool dispatcher;
    int target;
    bool supplied;
    bool edge;
};

std::vector<uint32_t> scalar_offset_buffer(const ScalarOffsetCase& arm) {
    std::vector<uint32_t> code{
        0x7e060281u, // whole-entry genuine old v3
        0x7e020280u, // genuine current VADDR v1
        0xbe800300u | (static_cast<uint32_t>(arm.target) << 16u) |
            (arm.supplied ? 0x80u : 0x16u), // copy zero or absent s22 into the scalar offset
        0xbe84047eu,
        0x7dea0080u,
        arm.typed ? 0xe8b01000u : 0xe0301000u,
        (static_cast<uint32_t>(arm.target) << 24u) | 0x00030301u,
        0x7d280680u,
        0xbf810000u,
    };
    if (arm.edge) code.insert(code.end() - 2, {0xbf060000u, 0xbf850001u, 0xbf800000u});
    return code;
}

class FlatActiveWordScalarOffset : public testing::TestWithParam<ScalarOffsetCase> {};

std::vector<ScalarOffsetCase> scalar_offset_cases() {
    std::vector<ScalarOffsetCase> cases;
    for (bool typed : {false, true})
        for (bool dispatcher : {false, true})
            for (int target : {20, 108, 124}) // ordinary s20, ttmp0 and M0
                for (bool supplied : {false, true})
                    for (bool edge : {false, true})
                        cases.push_back({typed, dispatcher, target, supplied, edge});
    return cases;
}
} // namespace

TEST_P(FlatActiveWordScalarOffset, CurrentNumericOffsetDefinesTheNewLoadResult) {
    const auto& arm = GetParam();
    const auto code = scalar_offset_buffer(arm);
    const auto result = compile(code, arm.dispatcher);
    if (arm.supplied)
        accepted(result, arm.dispatcher);
    else
        missing(result, static_cast<uint32_t>(code.size() - 2), arm.dispatcher);
}

INSTANTIATE_TEST_SUITE_P(BufferLoad, FlatActiveWordScalarOffset,
                         testing::ValuesIn(scalar_offset_cases()),
                         [](const testing::TestParamInfo<ScalarOffsetCase>& info) {
                             const auto& arm = info.param;
                             return std::string(arm.typed ? "Mtbuf" : "Mubuf") +
                                    (arm.dispatcher ? "Dispatcher" : "Structured") +
                                    (arm.target == 20    ? "S20"
                                     : arm.target == 108 ? "Ttmp0"
                                                         : "M0") +
                                    (arm.supplied ? "Supplied" : "Absent") +
                                    (arm.edge ? "Edge" : "Direct");
                         });

TEST(FlatActiveWord, ABufferNullOffsetHasItsArchitecturalZeroValue) {
    for (bool typed : {false, true})
        for (bool dispatcher : {false, true}) {
            auto code = scalar_offset_buffer({typed, dispatcher, 124, true, true});
            code[2] = 0xbf800000u; // no scalar data producer needed for SGPR_NULL
            code[6] = 0x7d030301u;
            accepted(compile(code, dispatcher), dispatcher);
        }
}

TEST(FlatActiveWord, ABufferOffsetRetainsExactNativeMaskMaterialization) {
    for (bool typed : {false, true})
        for (bool dispatcher : {false, true})
            for (uint32_t source : {106u, 107u, 126u, 127u}) {
                auto code = scalar_offset_buffer({typed, dispatcher, 124, true, false});
                code[2] = 0xbf800000u;
                code[6] = (source << 24u) | 0x00030301u;
                // CMPX supplies actual VCC/EXEC before the offset is read.
                accepted(compile(code, dispatcher), dispatcher);
            }
}

TEST(FlatActiveWord, ABufferSccOffsetUsesItsCurrentNumericCondition) {
    for (bool typed : {false, true})
        for (bool dispatcher : {false, true})
            for (bool supplied : {false, true})
                for (bool edge : {false, true}) {
                    auto code = scalar_offset_buffer({typed, dispatcher, 124, true, edge});
                    code[2] = supplied ? 0xbf068080u : 0xbf068016u;   // SCC from zero or absent s22
                    code[6] = 0xfd030301u;   // SCC is consumed as a numeric 0/1 offset
                    const auto result = compile(code, dispatcher);
                    if (supplied)
                        accepted(result, dispatcher);
                    else
                        missing(result, static_cast<uint32_t>(code.size() - 2), dispatcher);
                }
}

TEST(FlatActiveWord, ABufferSccOffsetCannotUseTheUnproducedEntryCondition) {
    for (bool typed : {false, true})
        for (bool dispatcher : {false, true})
            for (bool edge : {false, true}) {
                auto code = scalar_offset_buffer({typed, dispatcher, 124, true, edge});
                code[2] = 0xbf800000u;
                code[6] = 0xfd030301u;
                missing(compile(code, dispatcher), static_cast<uint32_t>(code.size() - 2),
                        dispatcher);
            }
}

TEST(FlatActiveWord, ABufferSccOffsetRequiresAGenuineConditionOnBothIncomingArms) {
    for (bool typed : {false, true})
        for (bool dispatcher : {false, true})
            for (bool supplied : {false, true}) {
                auto code = scalar_offset_buffer({typed, dispatcher, 124, true, false});
                code.erase(code.begin() + 2);
                code.insert(
                    code.begin() + 2,
                    {
                        0xbf060000u,
                        0xbf850002u,   // genuine incoming branch condition
                        0xbf068080u,
                        0xbf820001u,   // one arm computes genuine SCC
                        supplied ? 0xbf068080u : 0xbf068016u,   // the other arm needs actual s22
                    });
                code[10] = 0xfd030301u;
                const auto result = compile(code, dispatcher);
                if (supplied)
                    accepted(result, dispatcher);
                else
                    // A dispatcher refuses the unavailable joined SCC at the offset read;
                    // structured emission retains its mark until the fresh comparison.
                    missing(result, static_cast<uint32_t>(code.size() - (dispatcher ? 4 : 2)),
                            dispatcher);
            }
}
