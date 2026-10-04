// #4302: actual guarded mappings reach cold, warm, cache-off and diagnostic consumers.
// These are CPU source/module oracles, not Vulkan execution or guest launch authority.
#include "fixtures/test_scratch.h"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/shader_source_window.hpp"
#include "gpu/resources/fold_reader.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "host/memory/guest_memory_map.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

namespace {
class PageWindow {
public:
    PageWindow() {
#ifdef _WIN32
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        page_ = info.dwPageSize;
        bytes_ = static_cast<uint8_t*>(
            VirtualAlloc(nullptr, page_ * 3, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) return;
        page_ = static_cast<size_t>(page);
        void* allocation =
            mmap(nullptr, page_ * 3, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (allocation != MAP_FAILED) bytes_ = static_cast<uint8_t*>(allocation);
#endif
        if (bytes_) prosper::host::notify_guest_page_protection_changed();
    }
    ~PageWindow() {
        if (!bytes_) return;
#ifdef _WIN32
        VirtualFree(bytes_, 0, MEM_RELEASE);
#else
        munmap(bytes_, page_ * 3);
#endif
        prosper::host::notify_guest_page_protection_changed();
    }
    PageWindow(const PageWindow&) = delete;
    PageWindow& operator=(const PageWindow&) = delete;
    bool valid() const { return bytes_ && page_ >= 256 && !(page_ % 256); }
    bool guard_middle() {
#ifdef _WIN32
        DWORD old = 0;
        const bool result = VirtualProtect(bytes_ + page_, page_, PAGE_NOACCESS, &old) != 0;
#else
        const bool result = mprotect(bytes_ + page_, page_, PROT_NONE) == 0;
#endif
        if (result) prosper::host::notify_guest_page_protection_changed();
        return result;
    }
    uint32_t* tail(size_t words) { return reinterpret_cast<uint32_t*>(bytes_ + page_) - words; }
    uint64_t middle() const { return uint64_t(uintptr_t(bytes_ + page_)); }
    size_t page_dwords() const { return page_ / sizeof(uint32_t); }

private:
    uint8_t* bytes_ = nullptr;
    size_t page_ = 0;
};

class ShaderSourceWindow : public testing::Test {
protected:
    void SetUp() override {
        clear_shader_recompile_cache();
        clear_shader_analysis_cache();
        clear_shader_decode_cache();
    }
};

struct NoResourceReader : FoldReader {
    bool probe(FoldProbe, uint32_t, uint64_t, uint32_t) override {
        ADD_FAILURE() << "these original instructions have no resource probe";
        return false;
    }
    uint32_t word(uint32_t, uint64_t) override {
        ADD_FAILURE() << "these original instructions have no resource word";
        return 0;
    }
    void prefix(uint32_t, uint64_t, void*, uint32_t) override {
        ADD_FAILURE() << "these original instructions have no resource prefix";
    }
};

constexpr std::array fragment = {
    0x7e000280u,
    0xf8001803u,
    0u,
    0xbf810000u,
};
alignas(256) constexpr uint32_t vertex[] = {
    0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
    0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};

GpuState state_for(uint64_t pixel_address) {
    GpuState state;
    const auto program = [&](uint32_t lo, uint32_t hi, uint64_t address) {
        state.sh[lo] = static_cast<uint32_t>(address >> 8);
        state.sh[hi] = static_cast<uint32_t>((address >> 40) & 0xffu);
    };
    program(P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, uint64_t(uintptr_t(vertex)));
    program(P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, pixel_address);
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[P::CB_TARGET_MASK] = 0xf;
    state.cx[P::CB_COLOR_CONTROL] = P::CB_COLOR_CONTROL_MODE_NORMAL
                                    << P::CB_COLOR_CONTROL_MODE_SHIFT;
    return state;
}

// Real AGC registration retains this process-lifetime header. SDK-relative register pointers work
// in both low-address non-PIE tests and ordinary high-address hosts.
struct RegisteredBlob {
    AgcShaderHeader header{};
    ShaderReg registers[2]{};
};
bool register_program(RegisteredBlob& blob, const uint32_t* code, uint32_t bytes) {
    prosper::register_agc_hle();
    const auto create = prosper::Hle::lookup("f3dg2CSgRKY");
    if (!create) return false;
    blob.header = {};
    blob.registers[0] = {P::SPI_SHADER_PGM_LO_PS, 0};
    blob.registers[1] = {P::SPI_SHADER_PGM_HI_PS, 0};
    blob.header.file_header = 0x34333231u;
    blob.header.version = 0x18;
    blob.header.type = 1;
    blob.header.shader_size = bytes;
    blob.header.sh_registers = reinterpret_cast<const void*>(uintptr_t(blob.registers) -
                                                             uintptr_t(&blob.header.sh_registers));
    blob.header.num_sh_registers = 2;
    void* result = nullptr;
    return create(uint64_t(uintptr_t(&result)), uint64_t(uintptr_t(&blob.header)),
                  uint64_t(uintptr_t(code)), 0, 0, 0) == 0 &&
           result == &blob.header;
}

std::vector<std::vector<uint32_t>> dumped_words() {
    std::vector<std::vector<uint32_t>> result;
    const auto directory = refused_shader_dump_directory();
    if (directory.empty()) return result;
    for (const auto& file : std::filesystem::directory_iterator(directory)) {
        if (file.path().extension() != ".bin") continue;
        std::ifstream input(file.path(), std::ios::binary);
        const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
        EXPECT_EQ(bytes.size() % sizeof(uint32_t), 0u);
        std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
        if (!bytes.empty())
            std::memcpy(words.data(), bytes.data(), words.size() * sizeof(uint32_t));
        result.push_back(std::move(words));
    }
    return result;
}
} // namespace

TEST_F(ShaderSourceWindow, ReadablePrefixStopsAtTheFirstHole) {
    PageWindow pages;
    ASSERT_TRUE(pages.valid());
    auto* code = pages.tail(2);
    code[0] = code[1] = 0xbf800000u; // S_NOP, not an early-END escape from the page edge.
    ASSERT_TRUE(pages.guard_middle());
    const uint64_t address = uint64_t(uintptr_t(code));
    const size_t request = pages.page_dwords() * 2 + 2;
    ASSERT_TRUE(guest_readable(pages.middle() + pages.page_dwords() * 4, 4));
    EXPECT_EQ(shader_source_dwords(address, request), 2u);
    const auto analysis = acquire_shader_analysis(code, request);
    ASSERT_TRUE(analysis);
    EXPECT_EQ(analysis->code, (std::vector<uint32_t>{0xbf800000u, 0xbf800000u}));
    NoResourceReader reader;
    EXPECT_TRUE(resolve_dynamic_fetch(code, request, nullptr, 0, 0, nullptr, UINT32_MAX, nullptr,
                                      nullptr, 0, &reader)
                    .empty());
    EXPECT_EQ(reader.decoded_dwords, 2u);
    EXPECT_EQ(fragment_consumed_attribute_mask_cached(code, request), 0u);
}

TEST_F(ShaderSourceWindow, TruncatedEncodingDoesNotInventMissingOperandsOrPoisonTheCache) {
    PageWindow pages;
    ASSERT_TRUE(pages.valid());
    auto* code = pages.tail(1);
    ASSERT_TRUE(pages.guard_middle());
    // Original EXP, SMEM and literal V_MOV each require a second physical dword.
    for (const uint32_t first : {0xf8001803u, 0xf4000000u, 0x7e0002ffu}) {
        SCOPED_TRACE(first);
        *code = first;
        const auto analysis = acquire_shader_analysis(code, 17);
        ASSERT_TRUE(analysis);
        EXPECT_TRUE(analysis->code.empty());
        NoResourceReader reader;
        EXPECT_TRUE(resolve_dynamic_fetch(code, 17, nullptr, 0, 0, nullptr, UINT32_MAX, nullptr,
                                          nullptr, 0, &reader)
                        .empty());
        EXPECT_EQ(reader.decoded_dwords, 0u);
        const auto compiled =
            recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment, code, 17);
        EXPECT_TRUE(!compiled || compiled->empty());
        EXPECT_EQ(shader_analysis_cache_stats().entries, 0u);
        EXPECT_EQ(shader_decode_cache_stats().entries, 0u);
    }
    *code = 0xbf810000u;
    const auto valid = acquire_shader_analysis(code, 17);
    ASSERT_TRUE(valid);
    EXPECT_EQ(valid->code, (std::vector<uint32_t>{0xbf810000u}));
}

TEST_F(ShaderSourceWindow, RegisteredExtentIsCheckedBeforeApplyingTheCallerBudget) {
    PageWindow pages;
    ASSERT_TRUE(pages.valid());
    auto* code = pages.tail(1);
    *code = 0xbf810000u;
    ASSERT_TRUE(pages.guard_middle());
    AgcShaderHeader header{};
    header.code = code;
    header.shader_size = 8;
    const uint64_t address = uint64_t(uintptr_t(code));
    EXPECT_EQ(shader_source_dwords(address, 1, &header), 0u);
    header.shader_size = 4;
    EXPECT_EQ(shader_source_dwords(address, 17, &header), 1u);
    header.code = vertex;
    EXPECT_EQ(shader_source_dwords(address, 1, &header), 0u);
    EXPECT_EQ(shader_source_dwords(
                  address, 1, reinterpret_cast<const AgcShaderHeader*>(uintptr_t(pages.middle()))),
              0u);
    EXPECT_EQ(shader_source_dwords(0, 1), 0u);
    EXPECT_EQ(shader_source_dwords(UINT64_MAX - 3, 2), 0u);
    EXPECT_EQ(shader_source_dwords(address + 1, 1), 0u);
}

TEST_F(ShaderSourceWindow, GenuineShortRegistrationAndHeaderlessCodeKeepTheNativeModule) {
    PageWindow pages;
    ASSERT_TRUE(pages.valid());
    auto* code = pages.tail(fragment.size());
    std::copy(fragment.begin(), fragment.end(), code);
    ASSERT_TRUE(pages.guard_middle());
    const auto expected = recompile_fragment(fragment.data(), fragment.size());
    ASSERT_FALSE(expected.empty());
    const auto cold =
        recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment, code, 17);
    const auto warm =
        recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment, code, 17);
    ASSERT_TRUE(cold);
    ASSERT_TRUE(warm);
    EXPECT_EQ(*cold, expected);
    EXPECT_EQ(*warm, expected);
    EXPECT_EQ(shader_source_dwords(uint64_t(uintptr_t(code)), 17), fragment.size());
    // An aligned real registration publishes only sixteen bytes, not the caller's artificial 4 KiB.
    static RegisteredBlob blob;
    alignas(256) static const uint32_t registered[] = {
        0x7e000280u,
        0xf8001803u,
        0u,
        0xbf810000u,
    };
    ASSERT_TRUE(register_program(blob, registered, sizeof(registered)));
    ASSERT_EQ(blob.header.code, registered);
    ASSERT_EQ(prosper_agc_shader_header_for_code(uint64_t(uintptr_t(registered))), &blob.header);
    EXPECT_EQ(native_shader_source_dwords(uint64_t(uintptr_t(registered)), 1024), fragment.size());
    DrawItem draw;
    ASSERT_TRUE(realize_draw_item(state_for(uint64_t(uintptr_t(registered))), nullptr, 3,
                                  std::size(vertex), false, draw));
    EXPECT_EQ(draw.fs_words(), expected);
}

TEST_F(ShaderSourceWindow, MissingPhysicalProgramRefusesBeforeAnyNativeAnalysis) {
    PageWindow pages;
    ASSERT_TRUE(pages.valid());
    ASSERT_TRUE(pages.guard_middle());
    // Actual bound PGM registers, no synthetic point, code-free hint or bank authority.
    const auto before_analysis = shader_analysis_cache_stats();
    const auto before_decode = shader_decode_cache_stats();
    DrawItem draw;
    OperationRealizationFailure failure;
    EXPECT_FALSE(realize_draw_item(state_for(pages.middle()), nullptr, 3, std::size(vertex), false,
                                   draw, &failure));
    EXPECT_EQ(failure.reason, RealizationFailureReason::MissingProgram);
    EXPECT_TRUE(draw.fs_words().empty());
    EXPECT_FALSE(draw.fragment_draw_inputs);
    ASSERT_EQ(failure.stages.size(), 2u);
    EXPECT_EQ(failure.stages[1].program_addr, pages.middle());
    EXPECT_FALSE(failure.stages[1].recompiled);
    EXPECT_EQ(shader_analysis_cache_stats().misses, before_analysis.misses);
    EXPECT_EQ(shader_decode_cache_stats().misses, before_decode.misses);
}

TEST_F(ShaderSourceWindow, RealRegistrationCannotHideAnUnreadableTailBehindAShortBudget) {
    // Registration owns the original mapping for the process, just as the normal AGC registry
    // retains its header; a recycled VA must not accidentally select this deliberately bad record.
    static PageWindow pages;
    static RegisteredBlob blob;
    ASSERT_TRUE(pages.valid());
    auto* code = pages.tail(64);
    std::fill_n(code, 64, 0xbf800000u);
    std::copy(fragment.begin(), fragment.end(), code);
    ASSERT_TRUE(pages.guard_middle());
    ASSERT_TRUE(register_program(blob, code, 65 * sizeof(uint32_t)));
    ASSERT_EQ(blob.header.code, code);
    EXPECT_EQ(native_shader_source_dwords(uint64_t(uintptr_t(code)), 1), 0u);
    DrawItem draw;
    OperationRealizationFailure failure;
    EXPECT_FALSE(realize_draw_item(state_for(uint64_t(uintptr_t(code))), nullptr, 3,
                                   std::size(vertex), false, draw, &failure));
    EXPECT_EQ(failure.reason, RealizationFailureReason::MissingProgram);
    EXPECT_TRUE(draw.fs_words().empty());
}

TEST_F(ShaderSourceWindow, RefusalEvidenceRetainsOnlyTheAuthenticReadablePrefix) {
    PageWindow pages;
    ASSERT_TRUE(pages.valid());
    auto* code = pages.tail(2);
    code[0] = code[1] = 0xbf800000u;
    ASSERT_TRUE(pages.guard_middle());
    reset_refused_shader_dump_for_test(
        (prosper_test::test_scratch_dir() / "shader-source-window").string());
    RefusedDrawShaders source{};
    source.ps_address = source.es_address = uint64_t(uintptr_t(code));
    source.ps_failed = true;
    source.max_dwords = 17;
    note_refused_draw_shaders(source);
    EXPECT_EQ(dumped_words(), (std::vector<std::vector<uint32_t>>{{code[0], code[1]}}));
    EXPECT_EQ(refused_shader_dump_stats().hashed_dwords, 2u);
    const auto before = shader_analysis_cache_stats();
    source.ps_address = pages.middle();
    note_refused_draw_shaders(source);
    EXPECT_EQ(shader_analysis_cache_stats().misses, before.misses);
    EXPECT_EQ(refused_shader_dump_stats().content_records, 1u);
}
