#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_packet_mask_entry_fixture.hpp"
#include "gpu/execute/fragment_packet_preparation.hpp"
#include "gpu/recompiler/raster_quad_collector.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
using namespace prosper::gpu;
namespace f = prosper::test::fragment_mask_entry;
void retain(const FragmentPacketProgram& program, const std::string& name) {
    // Diagnostic CPU-test-only original SOURCE retention. No default artifacts, guest selector,
    // compiler substitution, live admission or inference that factory modules are these modules.
    const char* root = std::getenv("PROSPER_MASK_ENTRY_SPV_DIRECTORY");
    if (!root || !*root) return;
    std::filesystem::create_directories(root);
    std::ofstream file(std::filesystem::path(root) / (name + ".spv"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(program.spirv.data()),
               static_cast<std::streamsize>(program.spirv.size() * 4));
    file.close();
    EXPECT_TRUE(bool(file)) << "actual SOURCE retention";
}
FragmentPacketMaskRequirements facts(const FragmentInvocationPacket& packet) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(packet.guest_code.data(), packet.guest_code.size(), ins);
    return fragment_packet_mask_requirements(packet.guest_code, ins);
}
void evaluate(const f::Case& c) {
    SCOPED_TRACE(c.name);
    const auto program = recompile_fragment_packet(c.packet);
    ASSERT_FALSE(program.spirv.empty()) << program.rejection;
    retain(program, c.name);   // retain produced SOURCE even if VM/consumer evaluation fails
    bpermute_oracle::Interpreter vm(program.spirv);
    const auto words = vm.run_packet(program.input_words, program.output_words);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const auto result = decode_fragment_packet(program, words, true);
    ASSERT_TRUE(result.rejection.empty()) << result.rejection;
    EXPECT_EQ(result.exports, c.expected) << "all64 independent original raw EXP sinks";
}
bool gap(const FragmentPacketPreparation& prepared, const char* value) {
    return std::find(prepared.unmet.begin(), prepared.unmet.end(), value) != prepared.unmet.end();
}
}   // namespace
TEST(FragmentPacketMaskEntry, ActualSourceDemandsOnlyReachingInitialValues) {
    reset_float_controls_support_for_test();
    const auto cases = f::cases();
    ASSERT_EQ(cases.size(), f::kCases);
    uint32_t evaluations = 0;
    for (const auto& c : cases) {
        evaluate(c);
        ++evaluations;
    }
    EXPECT_EQ(evaluations, f::kCases);
    std::fprintf(stderr, "[mask-entry-source] evaluation_attempts=%u expected=%u\n", evaluations,
                 f::kCases);
}
TEST(FragmentPacketMaskEntry, AbsenceCannotBorrowZeroStorageOrPruneABypass) {
    auto p = f::base();
    f::finish(p, 0);
    p.exec_available = false;
    p.exec_mask = 0;   // a populated internal wire slot is not guest presence
    EXPECT_EQ(recompile_fragment_packet(p).rejection, "packet-entry-exec-unavailable");
    p = f::old_scc(false).packet;
    p.scc_available = false;
    EXPECT_EQ(recompile_fragment_packet(p).rejection, "packet-entry-scc-unavailable");
    p = f::cmpx_vcc(false).packet;
    p.vcc_available = false;
    const auto cmpx = facts(p);
    EXPECT_EQ(cmpx.demanded, kPacketInitialExec | kPacketInitialVcc);
    EXPECT_EQ(recompile_fragment_packet(p).rejection, "packet-entry-vcc-unavailable");
    p = f::joined_vcc(true).packet;
    p.guest_code[2] = 0xbf840003u;   // SCC0 bypasses BOTH complete writers; actual SCC1 does not
    const auto bypass = facts(p);
    EXPECT_TRUE(bypass.demanded & kPacketInitialVcc);
    EXPECT_EQ(bypass.first_read_pc[1], 6u);
    EXPECT_EQ(recompile_fragment_packet(p).rejection, "packet-entry-vcc-unavailable")
        << "supplied SCC1 cannot discard the structural SCC0 predecessor";
    for (uint32_t missing_word : {20u, 21u}) {
        p = f::scalar_vcc(0, true).packet;   // present zero is valid; an absent half is not
        std::erase_if(p.sgprs, [&](const auto& value) { return value.first == missing_word; });
        const auto refused = recompile_fragment_packet(p);
        EXPECT_TRUE(refused.spirv.empty());
        EXPECT_EQ(refused.rejection, "packet-sgpr-read-before-definition")
            << "opposite supplied VCC cannot manufacture missing s" << missing_word;
    }
    // Partial physical VCC definitions are still outside the executable packet DATA domain;
    // nevertheless the demanded-state proof must not count a low-only writer as a whole pair.
    p = f::base();
    f::fp::smov(p.guest_code, 106, 0);
    p.guest_code.push_back(0xbf860000u);
    f::finish(p, 0);
    EXPECT_EQ(facts(p).demanded, kPacketInitialExec | kPacketInitialVcc);
    EXPECT_EQ(facts(p).first_read_pc[1], 2u);
}
TEST(FragmentPacketMaskEntry, MissingPeerAndInactiveRawPayloadStillRefuse) {
    auto c = f::peer_before_exec();
    for (auto& column : c.packet.vgprs)
        if (column.reg == 8) column.available_mask &= ~(uint64_t(1) << 40);
    auto program = recompile_fragment_packet(c.packet);
    ASSERT_FALSE(program.spirv.empty()) << program.rejection;
    retain(program, "selected_peer_unavailable");
    bpermute_oracle::Interpreter vm(program.spirv);
    auto result = decode_fragment_packet(
        program, vm.run_packet(program.input_words, program.output_words), true);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(result.rejection, "packet-vgpr-selected-peer-unavailable");
    EXPECT_EQ(result.kind, uint32_t(FragmentPacketVgprRead::SelectedPeer));
    EXPECT_EQ(result.pc, 0u);
    EXPECT_EQ(result.reg, 8u);
    EXPECT_TRUE(result.exports.empty());
    c = f::exec_writer(0, true);
    c.packet.vgprs.clear();   // unchanged raw EXP observes enabled payload even though EXEC is zero
    program = recompile_fragment_packet(c.packet);
    ASSERT_FALSE(program.spirv.empty()) << program.rejection;
    retain(program, "inactive_raw_payload_unavailable");
    bpermute_oracle::Interpreter inactive(program.spirv);
    result = decode_fragment_packet(
        program, inactive.run_packet(program.input_words, program.output_words), true);
    EXPECT_TRUE(inactive.error.empty()) << inactive.error;
    EXPECT_EQ(result.rejection, "packet-vgpr-raw-export-unavailable");
    EXPECT_EQ(result.kind, uint32_t(FragmentPacketVgprRead::RawExport));
    EXPECT_TRUE(result.exports.empty());
}
TEST(FragmentPacketMaskEntry, LegacyBundledAndExplicitFullAvailabilityStayIdentical) {
    auto legacy = f::cases().front().packet;
    legacy.mask_state_available = true;
    const auto old = recompile_fragment_packet(legacy);
    ASSERT_FALSE(old.spirv.empty()) << old.rejection;
    legacy.mask_state_available = false;
    legacy.exec_available = legacy.vcc_available = legacy.scc_available = true;
    const auto explicit_full = recompile_fragment_packet(legacy);
    ASSERT_FALSE(explicit_full.spirv.empty()) << explicit_full.rejection;
    EXPECT_EQ(old.spirv, explicit_full.spirv);
    EXPECT_EQ(old.input_words, explicit_full.input_words);
    EXPECT_EQ(old.output_words, explicit_full.output_words);
    retain(old, "legacy_bundled_full");
    retain(explicit_full, "explicit_full_same_legacy_bytes");

    // The independently admitted owned stage domain includes READFIRST, which is outside this
    // PR's forward mask inventory. Keep its genuine full profile, never infer absence is unused.
    auto wider = f::base();
    wider.mask_state_available = true;
    f::fd::column(wider, 8, 0x71330008u);
    wider.guest_code = {0x7e080508u};   // original READFIRST s4,v8
    f::finish(wider, 4);
    const auto expected = f::fd::expected(UINT64_MAX, 0x71330008u);
    evaluate({"wider_owned_bundled_full", wider, expected});
    const auto bundled_wider = recompile_fragment_packet(wider);
    ASSERT_FALSE(bundled_wider.spirv.empty()) << bundled_wider.rejection;
    EXPECT_EQ(bundled_wider.initial_mask_availability, 7u);
    EXPECT_EQ(bundled_wider.demanded_initial_masks, 7u)
        << "conservative wider-domain compatibility requirement, not a narrow mask proof";
    wider.mask_state_available = false;
    wider.exec_available = wider.vcc_available = wider.scc_available = true;
    evaluate({"wider_owned_explicit_full", wider, expected});
    const auto explicit_wider = recompile_fragment_packet(wider);
    ASSERT_FALSE(explicit_wider.spirv.empty()) << explicit_wider.rejection;
    EXPECT_EQ(bundled_wider.spirv, explicit_wider.spirv);
    EXPECT_EQ(bundled_wider.input_words, explicit_wider.input_words);
    EXPECT_EQ(bundled_wider.output_words, explicit_wider.output_words);
    wider.scc_available = false;
    EXPECT_EQ(recompile_fragment_packet(wider).rejection,
              "packet-mask-instruction-effects-unimplemented");

    // The stage-local proof consumes individual presence independently of its live flag witness.
    // A present zero is supplied, whereas entry_flags_observed=false cannot bless offline storage.
    auto partial = f::old_scc(false).packet;
    std::string refusal;
    EXPECT_TRUE(complete_graphics_packet_locals(partial, refusal)) << refusal;
    const auto scc_gap =
        "stage-input-scc-uninitialized:pc=" + std::to_string(facts(partial).first_read_pc[2]);
    partial.scc_available = false;
    EXPECT_FALSE(complete_graphics_packet_locals(partial, refusal));
    EXPECT_EQ(refusal, scc_gap);
    partial.scc_available = true;
    EXPECT_FALSE(complete_graphics_packet_locals(partial, refusal, false));
    EXPECT_EQ(refusal, scc_gap);
    partial = f::new_scc(false).packet;
    EXPECT_TRUE(complete_graphics_packet_locals(partial, refusal, false)) << refusal;
    partial.exec_available = false;
    EXPECT_FALSE(complete_graphics_packet_locals(partial, refusal));
    EXPECT_EQ(refusal, "stage-input-original-unavailable:pc=4294967295")
        << "missing EXEC cannot become certain zero execution in the inherited stage proof";
}
TEST(FragmentPacketMaskEntry, CachedCodeConsumesDifferentSuppliedDataAndExactPresenceProfile) {
    const auto c = f::cases().front();
    auto prototype = f::wave_input(c);
    const auto kernel =
        std::make_shared<const FragmentPacketKernel>(recompile_fragment_packet_kernel(prototype));
    ASSERT_FALSE(kernel->program.packet.spirv.empty()) << kernel->program.packet.rejection;
    auto opposite = prototype;
    opposite.invocation.exec_mask = 0;
    opposite.invocation.vcc_mask = ~prototype.invocation.vcc_mask;
    opposite.invocation.scc = !prototype.invocation.scc;
    const auto rebuilt = recompile_fragment_packet_kernel(opposite);
    EXPECT_EQ(kernel->program.packet.spirv, rebuilt.program.packet.spirv)
        << "supplied values and absent unused bytes are DATA, not compiler specialization";
    retain(kernel->program.packet, "cached_mask_profile_original");
    retain(rebuilt.program.packet, "cached_mask_profile_opposite");
    const std::vector<FragmentResourcePacket> waves{prototype, opposite, prototype};
    const auto batch = pack_fragment_packet_waves(kernel, waves);
    ASSERT_TRUE(batch.rejection.empty()) << batch.rejection;
    auto words = batch.output_words;
    for (uint32_t wave : {2u, 0u, 1u}) {
        bpermute_oracle::Interpreter vm(kernel->program.packet.spirv);
        words = vm.run_buffers(
            64, {{0, batch.input_words}, {1, words}, {2, batch.authority->words()}}, 1, wave);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
    }
    const auto result =
        decode_fragment_packet_waves(batch, words, true, prototype.device.device_identity);
    ASSERT_TRUE(result.rejection.empty()) << result.rejection;
    ASSERT_EQ(result.exports.size(), 3u);
    EXPECT_EQ(result.exports[0], c.expected);
    EXPECT_EQ(result.exports[1], f::fd::expected(0, 0x42230011u));
    EXPECT_EQ(result.exports[2], c.expected);
    auto missing = waves;
    missing[1].invocation.exec_available = false;
    EXPECT_EQ(pack_fragment_packet_waves(kernel, missing).rejection,
              "packet-wave-demanded-initial-mask-unavailable");
    missing = waves;
    missing[1].invocation.scc_available = true;
    EXPECT_EQ(pack_fragment_packet_waves(kernel, missing).rejection,
              "packet-wave-initial-mask-profile-mismatch");
}
TEST(FragmentPacketMaskEntry, ShippingPreparationUsesExactCachedProgramRequirements) {
    const auto prepare = [&](const f::Case& c) {
        auto inputs = std::make_shared<RasterQuadInputs>();
        inputs->raw_code = std::make_shared<const std::vector<uint32_t>>(c.packet.guest_code);
        inputs->source_fs = std::make_shared<const std::vector<uint32_t>>(
            std::initializer_list<uint32_t>{0x07230203u});
        inputs->raw_matches_producing_source = true;
        inputs->vgpr_requirements = std::make_shared<const FragmentPacketVgprRequirements>(
            fragment_packet_vgpr_requirements(*inputs->raw_code));
        return inputs;
    };
    auto inputs = prepare(f::cases().front());
    const auto only_exec = prepare_fragment_packet_inputs(inputs, true);
    ASSERT_TRUE(only_exec->mask_requirements);
    EXPECT_EQ(only_exec->mask_requirements.get(), &inputs->vgpr_requirements->masks);
    EXPECT_EQ(only_exec->mask_requirements->source_words, inputs->raw_code.get());
    EXPECT_GE(inputs->vgpr_requirements->retained_bytes(),
              sizeof(FragmentPacketVgprRequirements) +
                  only_exec->mask_requirements->entries.capacity() *
                      sizeof(FragmentPacketMaskRequirements::Entry));
    EXPECT_TRUE(gap(*only_exec, "packet-entry-exec-value-unproved"));
    EXPECT_FALSE(gap(*only_exec, "packet-entry-vcc-value-unproved"));
    EXPECT_FALSE(gap(*only_exec, "packet-entry-scc-value-unproved"));
    inputs = prepare(f::exec_writer(0, true));
    const auto no_masks = prepare_fragment_packet_inputs(inputs, true);
    ASSERT_TRUE(no_masks->mask_requirements);
    EXPECT_EQ(no_masks->mask_requirements->demanded, 0u);
    EXPECT_FALSE(gap(*no_masks, "packet-entry-mask-abi-unproved"));
    EXPECT_FALSE(no_masks->ready);
    EXPECT_TRUE(gap(*no_masks, "packet-logical64-composition-unproved"));
    EXPECT_TRUE(gap(*no_masks, "packet-ordered-export-commit-unimplemented"));
    const auto wrong_source = prepare_fragment_packet_inputs(inputs, false);
    EXPECT_FALSE(wrong_source->mask_requirements);
    EXPECT_TRUE(gap(*wrong_source, "packet-entry-vgpr-and-mask-abi-unproved"));
    inputs->vgpr_requirements.reset();
    const auto absent_analysis = prepare_fragment_packet_inputs(inputs, true);
    EXPECT_FALSE(absent_analysis->mask_requirements);
    EXPECT_TRUE(gap(*absent_analysis, "packet-vgpr-program-requirements-unavailable"));
    inputs = prepare(f::exec_writer(0, true));
    auto foreign = std::make_shared<FragmentPacketVgprRequirements>(*inputs->vgpr_requirements);
    foreign->masks.source_words = nullptr;
    inputs->vgpr_requirements = foreign;
    const auto absent_mask_owner = prepare_fragment_packet_inputs(inputs, true);
    EXPECT_FALSE(absent_mask_owner->mask_requirements);
    EXPECT_TRUE(gap(*absent_mask_owner, "packet-mask-program-requirements-unavailable"));
}
