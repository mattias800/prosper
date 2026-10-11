// #4945: a mapped null-store fixture still needs real synthetic launch words before physical
// mask DATA can be retained. Store-marker discovery and compilation are separate contracts.
#include "fixtures/gta5_guarded_raw_store_fixture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {

struct Input {
    alignas(256) std::array<uint32_t, 64> srt{};
    alignas(256) std::array<uint32_t, 256> input{}, output{};
    alignas(256) std::array<uint32_t, 16> scalar{};
    std::vector<uint32_t> code{prosper::test::kGta5GuardedRawStoreProgram.begin(),
                               prosper::test::kGta5GuardedRawStoreProgram.end()};
    std::array<uint32_t, 7> seed{};
    ShaderResourceTable table;
    std::vector<SrtUse> uses;

    Input() {
        const auto put_address = [](uint32_t* words, const void* pointer) {
            const auto address = reinterpret_cast<uintptr_t>(pointer);
            words[0] = static_cast<uint32_t>(address);
            words[1] = static_cast<uint32_t>(uint64_t{address} >> 32u);
        };
        put_address(srt.data() + 0x90u / 4u, input.data());
        put_address(srt.data() + 0x98u / 4u, output.data());
        put_address(srt.data() + 0xc0u / 4u, scalar.data());
        srt[0xc0u / 4u + 1u] |= 4u << 16u;
        srt[0xc0u / 4u + 2u] = scalar.size();
        srt[0xc0u / 4u + 3u] = 0x00016204u;
        put_address(seed.data(), srt.data());
        seed[4] = 151u;
        seed[5] = 4u;
    }

    void discover() {
        table = {};
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Uint32;
        resource.num_components = 1;
        resource.gpu_addr = reinterpret_cast<uintptr_t>(srt.data());
        resource.size = sizeof(srt);
        table.resources.push_back(resource);
        uses =
            add_compute_buffer_resources(table, code.data(), code.size(), seed.data(), seed.size());
        assign_convention_bindings(table, 2);
    }

    ComputeShaderConfig config() const { return prosper::test::guarded_raw_store_config(seed); }

    size_t markers() const {
        return std::count_if(table.resources.begin(), table.resources.end(),
                             is_proven_null_guarded_raw_store);
    }

    void expect_markers() const {
        EXPECT_EQ(markers(), 3u);
        for (uint32_t pc : {74u, 76u, 78u}) {
            const auto* resource = table.by_fetch_pc(pc);
            ASSERT_NE(resource, nullptr) << "store PC " << pc;
            EXPECT_TRUE(is_proven_null_guarded_raw_store(*resource)) << "store PC " << pc;
        }
        EXPECT_EQ(std::count_if(uses.begin(), uses.end(),
                                [](const SrtUse& use) {
                                    return use.proven_null_guarded_raw_store &&
                                           !use.zero_record_raw;
                                }),
                  3u);
    }
};

struct Result {
    std::vector<uint32_t> module;
    std::string reason;
};

Result compile(const Input& input, const ComputeShaderConfig& config) {
    static std::atomic<uint64_t> next_address{0x4945000000ull};
    const uint64_t address = next_address.fetch_add(0x100u);
    auto module = recompile_compute(input.code.data(), input.code.size(), &input.table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

void expect_launch_refusal(const Input& input, const ComputeShaderConfig& config) {
    EXPECT_FALSE(compute_dispatch_has_complete_groups(config));
    input.expect_markers();
    const auto result = compile(input, config);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("native-logical-data-words-unavailable"), std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("pc=46 "), std::string::npos) << result.reason;
}

} // namespace

TEST(GuardedStoreLaunch, CompleteSyntheticLaunchCompilesAllThreeMarkers) {
    Input input;
    input.discover();
    input.expect_markers();
    const auto config = input.config();
    EXPECT_TRUE(compute_dispatch_has_complete_groups(config));
    const auto result = compile(input, config);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_TRUE(validate_spirv_descriptor_interface(result.module, &input.table, 0,
                                                    SpirvShaderStage::Compute, false)
                    .ok());
}

TEST(GuardedStoreLaunch, UnknownRawLaunchPreservesMarkersButRefusesPhysicalData) {
    Input input;
    input.discover();
    auto config = input.config();
    config.dispatch_launch = {};
    expect_launch_refusal(input, config);
}

TEST(GuardedStoreLaunch, PartialFlagSameGeometryRefusesPhysicalData) {
    Input input;
    input.discover();
    auto config = input.config();
    config.dispatch_launch.initiator |= 2u;
    expect_launch_refusal(input, config);
}

TEST(GuardedStoreLaunch, EachMissingRawWordRefusesPhysicalData) {
    Input input;
    input.discover();
    for (uint32_t bit = 1u; bit <= ComputeDispatchLaunchState::kDimZ; bit <<= 1u) {
        auto config = input.config();
        config.dispatch_launch.known &= ~bit;
        SCOPED_TRACE(bit);
        expect_launch_refusal(input, config);
    }
}

TEST(GuardedStoreLaunch, ChangedEmptyExecGuardCannotAcquireStoreMarkers) {
    Input input;
    input.code[47] = 0x88fe126au; // S_OR_B64 instead of S_AND_B64.
    input.discover();
    EXPECT_EQ(input.markers(), 0u);
    EXPECT_FALSE(rdna2_gta5_null_guarded_raw_store_dispatch(input.code.data(), input.code.size(),
                                                            input.seed.data(), input.seed.size()));
    EXPECT_TRUE(compile(input, input.config()).module.empty());
}

TEST(GuardedStoreLaunch, EntryAfterGuardCannotAcquireStoreMarkers) {
    Input input;
    input.code[24] = 0xbf820017u; // S_BRANCH to PC48, bypassing the mask identity.
    input.discover();
    EXPECT_EQ(input.markers(), 0u);
    EXPECT_FALSE(rdna2_gta5_null_guarded_raw_store_dispatch(input.code.data(), input.code.size(),
                                                            input.seed.data(), input.seed.size()));
    EXPECT_TRUE(compile(input, input.config()).module.empty());
}

} // namespace prosper::gpu
