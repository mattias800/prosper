#include "gpu/recompiler/rdna2_native_image_results.hpp"
// #5497: the admitted native 0xa0 sample writes complete dwords in its active cohort.
// A conservative opcode-effects fallback must not turn those results into fabricated masks.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "bpermute_spirv_oracle.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
ShaderResourceTable resources() {
    ShaderResource texture;
    texture.cls = ResourceClass::Texture;
    texture.binding = 4;
    texture.img_dim = 1;
    texture.width = texture.height = texture.declared_mip_levels = 1;
    texture.format = DataFormat::Float32;
    texture.num_components = 4;
    texture.sgpr_base = 8;
    texture.sampler_sgpr_base = 16;
    ShaderResourceTable table;
    table.resources.push_back(texture);
    return table;
}
std::vector<uint32_t> program() {
    return {
        0x7e000f00u,                 // v0 = uint(FragCoord.x)
        0x7e0c02ffu, 0x40400000u,    // old v6 = 3.0
        0x7e0e023cu,                 // old v7 = absent s60
        0x7e0802ffu, 0x3f000000u,    // v4 = 0.5, genuine coordinate in every lane
        0x7e0a02ffu, 0x3f000000u,    // v5 = 0.5, genuine coordinate in every lane
        0x7da6009fu,                 // EXEC = (31 <= v0), both Wave64 halves
        0xf0800309u, 0x00820604u,    // admitted 0xa0 2D sample -> v[6:7]
        0x7d840f07u,                 // fresh VCC = (v7 == v7)
        0x02100e80u,                 // v8 = VCC ? v7 : 0
        0xf800180fu, 0x08080808u,    // export v8
        0xbf810000u,
    };
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address,
                              const ShaderResourceTable& table) {
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 1u << 8;
    return recompile_fragment(code.data(), code.size(), &table, &inputs, UINT32_MAX, nullptr, false,
                              {RecompileDiagnosticStage::Fragment, address});
}
// Compute transport uses the same guest operations and native-fragment word policy. The single
// owned mip makes its explicit LOD-0 sample equal to the fragment's implicit-LOD sample for every
// derivative. The oracle supports only these exact-center nearest samples, not general filtering.
std::vector<uint32_t> emitted_body(const std::vector<uint32_t>& code, uint32_t& refused_pc) {
    auto table = resources();
    SpirvCompute b;
    b.native_subgroup_size = 64;
    b.begin(5, &table, 64, 1, 1, 64, 0, true, true);
    RegState rs;
    rs.exec = b.btrue();
    rs.vcc = rs.scc = b.bfalse();
    rs.vreg[0] = b.cvt_u2f(b.localid_comp[0]);
    rs.max_vgpr = 8;
    std::vector<Rdna2Inst> instructions;
    if (rdna2_walk(code.data(), code.size(), instructions) != code.size()) return {};
    seed_vector_word_definedness(rs, instructions, true, 64);
    for (const auto& in : instructions) {
        if (in.fmt == Rdna2Format::EXP || in.is_end) break;
        prepare_saved_mask_word_write(b, rs, in);
        const auto before = snapshot_saved_b64_masks(b, rs, in);
        bool ok = true;
        if (!emit_alu(b, rs, in, ok, true, nullptr, true, &table) || !ok) {
            refused_pc = in.pc;
            return {};
        }
        record_scalar_write(rs, in, false, before);
        retain_saved_mask_words(b, rs, in);
    }
    b.store_output_word(rs.vreg.at(6), 5, 0);
    b.store_output_word(rs.vreg.at(7), 5, 1, rs.exec);
    b.store_output_word(rs.vreg.at(8), 5, 2, rs.exec);
    b.store_output_word(b.sel(rs.exec, b.uconst(1), b.uconst(0)), 5, 3);
    b.store_output_word(b.sel(rs.vcc, b.uconst(1), b.uconst(0)), 5, 4);
    return b.finish();
}
void evaluate(const std::vector<uint32_t>& code) {
    uint32_t refused = UINT32_MAX;
    const auto module = emitted_body(code, refused);
    ASSERT_FALSE(module.empty()) << "refused at PC " << refused;
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.sampled_images[4].push_back({1, 1, {{{0x3f800000u, 0x40000000u, 0u, 0u}}}});
    const auto output =
        vm.run_buffers(64, {{1, std::vector<uint32_t>(std::size_t{64} * 5, 0xdeadbeefu)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(output.size(), 64u * 5);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto at = static_cast<size_t>(lane) * 5;
        const uint32_t second = lane < 31 ? 0xdeadbeefu : 0x40000000u;
        EXPECT_EQ(output[at], lane < 31 ? 0x40400000u : 0x3f800000u) << lane;
        EXPECT_EQ(output[at + 1], second) << lane;
        EXPECT_EQ(output[at + 2], second) << lane;
        EXPECT_EQ(output[at + 3], lane >= 31 ? 1u : 0u) << lane;
        EXPECT_EQ(output[at + 4], lane >= 31 ? 1u : 0u) << lane;
    }
}
} // namespace
TEST(NativeImageResults, TheSecondSelectedDwordReachesTheFreshMaskAndExport) {
    EXPECT_FALSE(compile(program(), 0x5497060100ull, resources()).empty())
        << last_terminal_reject_reason(0x5497060100ull);
}
TEST(NativeImageResults, OwnedTexelsReachBothWaveHalvesAndKeepInactiveWords) {
    evaluate(program());
}
TEST(NativeImageResults, TheOrdinarySampleControlHasTheSameValueContract) {
    auto code = program();
    code[9] &= ~1u; // architectural base sample 0x20
    EXPECT_FALSE(compile(code, 0x5497060200ull, resources()).empty());
    evaluate(code);
}
TEST(NativeImageResults, MissingCoordinatesStillRefuse) {
    auto code = program();
    code[6] = 0x7e0a023cu; // v5 = absent s60
    code[7] = 0xbf800000u;
    EXPECT_TRUE(compile(code, 0x5497060300ull, resources()).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0x5497060300ull).find("fabricated-fresh-mask-source pc=11"),
        std::string::npos);
}
TEST(NativeImageResults, ExpandedExecCannotObserveAnUnwrittenSampleLane) {
    auto code = program();
    code.insert(code.begin() + 11, 0xbefe04c1u);
    EXPECT_TRUE(compile(code, 0x5497060400ull, resources()).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0x5497060400ull).find("fabricated-fresh-mask-source pc=12"),
        std::string::npos);
}
TEST(NativeImageResults, ADescriptorRefusalCannotSupplyTheSampledWord) {
    EXPECT_TRUE(compile(program(), 0x5497060500ull, ShaderResourceTable{}).empty());
}

namespace {
struct PublicationHarness {
    SpirvCompute b;
    RegState rs;
    ShaderResourceTable table = resources();
    Rdna2Inst sample;
    SavedB64MaskSnapshot before;
    PublicationHarness() {
        b.wave_size = 64;
        b.begin_fragment(&table);
        rs.exec = b.btrue();
        rs.vcc = rs.scc = b.bfalse();
        rs.vreg[0] = b.uconst(0x42200000u); // float 40, independent genuine EXEC input
        auto code = program();
        std::vector<Rdna2Inst> instructions;
        rdna2_walk(code.data(), code.size(), instructions);
        seed_vector_word_definedness(rs, instructions, true, 64);
        for (const auto& in : instructions) {
            if (in.pc > 9) break;
            prepare_saved_mask_word_write(b, rs, in);
            before = snapshot_saved_b64_masks(b, rs, in);
            bool ok = true;
            const bool handled = emit_alu(b, rs, in, ok, true, nullptr, true, &table);
            EXPECT_TRUE(handled && ok) << in.pc;
            if (!handled || !ok) return;
            if (in.pc == 9) {
                sample = in;
                return; // hold the actual publication before its ordinary mark transfer
            }
            record_scalar_write(rs, in, false, before);
            retain_saved_mask_words(b, rs, in);
        }
    }
};
} // namespace

TEST(NativeImageResults, ARealPublicationNamesEveryResultGenerationAndIsConsumedOnce) {
    PublicationHarness h;
    EXPECT_EQ(h.rs.native_image_result_publication.count, 2u);
    EXPECT_EQ(h.rs.native_image_result_publication.values[0], h.rs.vreg.at(6));
    EXPECT_EQ(h.rs.native_image_result_publication.values[1], h.rs.vreg.at(7));
    EXPECT_TRUE(consume_native_image_result(h.rs, h.sample));
    EXPECT_FALSE(consume_native_image_result(h.rs, h.sample));
}
TEST(NativeImageResults, TheGlobalOpcodeAndWriteInventoriesRemainConservative) {
    PublicationHarness h;
    EXPECT_TRUE(rdna2_instruction_may_write_memory(h.sample));
    EXPECT_FALSE(vector_write_completes_active_word(h.sample));
    EXPECT_TRUE(consume_native_image_result(h.rs, h.sample));
}
TEST(NativeImageResults, ChangedContextCannotConsumeAnEarlierPublication) {
    for (unsigned mutation = 0; mutation < 15; ++mutation) {
        PublicationHarness h;
        ASSERT_NE(h.rs.native_image_result_publication.exec, 0u);
        switch (mutation) {
            case 0: ++h.sample.pc; break;
            case 1: ++h.sample.dst.value; break;
            case 2: h.rs.exec = h.b.btrue(); break;
            case 3: h.rs.vreg[7] = h.b.uconst(0); break;
            case 4: h.rs.mimg_sources.pc = UINT32_MAX; break;
            case 5: h.rs.mimg_sources.defined = false; break;
            case 6: h.rs.mimg_sources.words.clear(); break;
            case 7: h.rs.unavailable_saved_mask_words.insert(8); break;
            case 8: h.rs.sreg_merge_placeholder.insert(16); break;
            case 9: h.rs.vector_wave_size = 32; break;
            case 10: h.rs.vector_native_interpolation = false; break;
            case 11: h.sample.words[0] ^= 1u; break;
            case 12: h.rs.native_image_result_publication.count = 5; break;
            case 13: h.rs.native_image_result_publication.values[0] = h.b.uconst(0); break;
            case 14: h.rs.exec_narrowed = false; break;
        }
        EXPECT_FALSE(consume_native_image_result(h.rs, h.sample)) << mutation;
        EXPECT_EQ(h.rs.native_image_result_publication.exec, 0u) << mutation;
    }
}
TEST(NativeImageResults, UnsupportedFormsCannotCaptureAResult) {
    for (unsigned mutation = 0; mutation < 9; ++mutation) {
        PublicationHarness h;
        switch (mutation) {
            case 0: h.sample.mimg_dmask = 0; break;
            case 1: h.sample.mimg_dim = 2; break;
            case 2: h.sample.mimg_d16 = true; break;
            case 3: h.sample.mimg_tfe = true; break;
            case 4: h.sample.mimg_lwe = true; break;
            case 5: h.sample.mimg_r128 = true; break;
            case 6: h.sample.mimg_reserved = true; break;
            case 7: h.sample.dst.value = 255; break;
            case 8: h.rs.vreg.erase(7); break;
        }
        record_native_image_result(h.rs, h.sample);
        EXPECT_EQ(h.rs.native_image_result_publication.exec, 0u) << mutation;
        EXPECT_FALSE(consume_native_image_result(h.rs, h.sample)) << mutation;
    }
}
TEST(NativeImageResults, AnotherInstructionExpiresTheHeldPublication) {
    PublicationHarness h;
    auto other = h.sample;
    other.fmt = Rdna2Format::VOP1;
    other.opcode = 0;
    EXPECT_FALSE(consume_native_image_result(h.rs, other));
    EXPECT_FALSE(consume_native_image_result(h.rs, h.sample));
}
TEST(NativeImageResults, AFailedDescriptorReadExpiresTheHeldPublication) {
    PublicationHarness h;
    h.table.resources.clear();
    bool ok = true;
    EXPECT_TRUE(emit_alu(h.b, h.rs, h.sample, ok, true, nullptr, true, &h.table));
    EXPECT_FALSE(ok);
    EXPECT_EQ(h.rs.native_image_result_publication.exec, 0u);
}
TEST(NativeImageResults, OrdinaryTransferKeepsInactiveMarksAndConsumesThePublication) {
    PublicationHarness h;
    record_scalar_write(h.rs, h.sample, false, h.before);
    EXPECT_EQ(h.rs.native_image_result_publication.exec, 0u);
    const auto& word = h.rs.vreg_source_marks.at(7);
    EXPECT_TRUE(word.placeholder && word.hard && word.memory);
    EXPECT_FALSE(word.arbitrary);
    EXPECT_EQ(word.outside_exec, h.rs.exec);
    const uint32_t compare_word = 0x7d840f07u;
    auto compare = rdna2_decode_one(&compare_word, 1);
    compare.pc = 11;
    EXPECT_TRUE(fresh_mask_sources_defined(h.b, h.rs, compare));
    h.rs.exec = h.b.btrue();
    h.rs.exec_narrowed = false;
    EXPECT_FALSE(fresh_mask_sources_defined(h.b, h.rs, compare));
}
