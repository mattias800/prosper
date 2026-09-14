#include "gpu/resources/fold_control_plan.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <string_view>

using namespace prosper::gpu;
static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s\n", m); ++failures; } } while (0)

static Rdna2Inst inst(uint32_t pc, uint32_t op, int16_t displacement = 0) {
    Rdna2Inst in{};
    in.pc = pc; in.len_dwords = 1; in.fmt = Rdna2Format::SOPP;
    in.opcode = op; in.simm16 = displacement;
    return in;
}

struct Fixture {
    alignas(16) uint32_t data[32]{};
    alignas(16) uint32_t constants[40]{};
    alignas(16) uint32_t descriptor[4]{};
    uint32_t seed[12]{};
    std::vector<uint32_t> code{0xBF860003u, 0xF4300404u, 0xFA000060u,
        0xBF820004u, 0xF4080108u, 0xFA000000u, 0xE0002000u, 0x80010100u, 0xBF810000u};
    Fixture() {
        constants[24] = 0x3f800000u;
        const auto cb = uint64_t(uintptr_t(constants));
        const auto table = uint64_t(uintptr_t(descriptor));
        const auto vb = uint64_t(uintptr_t(data));
        seed[0] = uint32_t(cb); seed[1] = uint32_t(cb >> 32) | (16u << 16);
        seed[2] = 10;
        seed[8] = uint32_t(table); seed[9] = uint32_t(table >> 32);
        descriptor[0] = uint32_t(vb); descriptor[1] = uint32_t(vb >> 32) | (16u << 16);
        descriptor[2] = 8; descriptor[3] = 0;
    }
    bool resolves(uint32_t records) {
        auto result = resolve_dynamic_fetch(code.data(), code.size(), seed, 12, 8);
        return result.size() == 1 && result[0].desc.base == uint64_t(uintptr_t(data)) &&
            result[0].desc.num_records == records;
    }
};

int main(int argc, char** argv) {
    // These hand-built PCs isolate control metadata from decoder and interpreter behavior.
    std::vector<Rdna2Inst> stream{inst(0, 6, 3), inst(1, 0), inst(3, 2, 8),
        inst(4, 4, 3), inst(5, 0), inst(7, 2, 4), inst(8, 0)};
    auto plan = build_fold_control_plan(stream);
    CHECK(plan.cfg_known && plan.snapshot_count == 2, "two independently qualified targets");
    CHECK(plan.steps[0].save_slot == 0 && plan.steps[3].restore_slot == 0 &&
          plan.steps[3].save_slot == 1 && plan.steps[6].restore_slot == 1,
          "a target may restore and then save at the same instruction");
    CHECK(plan.steps[3].reset_zero_mip && plan.steps[6].reset_zero_mip,
          "both target blocks reset zero-mip provenance");
    auto changed = stream;
    changed[2].opcode = 0;
    CHECK(build_fold_control_plan(changed).steps[3].restore_slot == UINT32_MAX,
          "fall-through prevents exclusive restoration");
    changed = stream; changed[2].pc = 2;
    CHECK(build_fold_control_plan(changed).steps[3].restore_slot == UINT32_MAX,
          "physical gap prevents restoration even when retained instructions are adjacent");
    changed = stream; changed.push_back(inst(9, 2, -6));
    CHECK(build_fold_control_plan(changed).steps[3].restore_slot == UINT32_MAX,
          "backward second predecessor disqualifies the first target");
    changed = stream; changed[1].fmt = Rdna2Format::SOP1; changed[1].opcode = 0x20;
    auto indirect = build_fold_control_plan(changed);
    CHECK(!indirect.cfg_known && indirect.snapshot_count == 0 &&
          indirect.steps[0].reset_zero_mip && indirect.steps.back().reset_zero_mip,
          "indirect transfer refuses finite-CFG proofs for every instruction");
    // The branch targets PC4, which was compacted out. PC5 must nevertheless start a new block.
    auto gap = build_fold_control_plan({inst(0, 6, 3), inst(1, 0), inst(3, 0), inst(5, 0)});
    CHECK(gap.steps[3].reset_zero_mip, "compacted-out first block instruction still resets proof");

    static_assert(sizeof(FoldControlStep) == 16, "instruction facts fit existing plan storage");
    Rdna2Inst mip{};
    mip.fmt = Rdna2Format::MIMG; mip.opcode = 1; mip.len_dwords = 2;
    mip.src[0] = {OperandKind::VGPR, 253};
    mip.mimg_unorm = mip.mimg_glc = true; mip.mimg_dmask = 1; mip.mimg_dim = 1;
    auto exec = inst(1, 0); exec.fmt = Rdna2Format::VOPC; exec.opcode = 0xd2;
    auto transfer = inst(0, 0); transfer.fmt = Rdna2Format::SOP1; transfer.opcode = 0x20;
    auto facts = build_fold_control_plan({transfer, exec, mip});
    CHECK(!facts.cfg_known && facts.steps[1].changes_exec && facts.steps[2].zero_mip_vgpr == 255,
          "instruction facts after an indirect transfer survive CFG refusal");
    exec.opcode = 0xc2;
    mip.src[0].value = 254;
    facts = build_fold_control_plan({exec, mip});
    CHECK(!facts.steps[0].changes_exec && facts.steps[1].zero_mip_vgpr == UINT16_MAX,
          "ordinary compare preserves EXEC and out-of-range mip operand has no shape");
    exec.sdst = {OperandKind::Special, 126};
    mip.opcode = 9; mip.mimg_nsa = 1; mip.len_dwords = 3; mip.words[2] = 0xff00;
    facts = build_fold_control_plan({exec, mip});
    CHECK(facts.steps[0].changes_exec && facts.steps[1].zero_mip_vgpr == 255,
          "explicit scalar EXEC destination and NSA mip operand retain their distinct facts");
    mip.mimg_tfe = true;
    CHECK(build_fold_control_plan({mip}).steps[0].zero_mip_vgpr == UINT16_MAX,
          "unsupported mip modifier refuses the cached shape");

    // Reuse the exact code address while changing instruction facts and live register values.
    uint32_t mip_seed[28]{};
    const uint32_t image[8] = {0x055c0100u, 0xc1400000u, 0x001fc01fu, 0x91b00204u,
                              0u, 0x00700000u, 0u, 0u};
    std::copy(std::begin(image), std::end(image), mip_seed + 20);
    uint32_t mip_code[] = {0x7e040207u, 0xbe9e0380u, 0xf0043108u, 0x00050000u, 0xbf810000u};
    const auto mip_proof = [&](bool expected) {
        std::vector<SrtUse> uses;
        resolve_dynamic_fetch(mip_code, std::size(mip_code), mip_seed, 28, 0, &uses);
        return uses.size() == 1 && uses[0].use_pc == 2 && uses[0].proven_zero_mip == expected;
    };
    CHECK(mip_proof(true) && mip_proof(true), "warm mip plan keeps current zero-register proof");
    mip_seed[7] = 1;
    CHECK(mip_proof(false), "unchanged cached code never caches the mip register value");
    mip_seed[7] = 0;
    mip_code[1] = 0x7da40100u; // implicit EXEC write between v2=zero and its use
    CHECK(mip_proof(false), "same-address EXEC mutation invalidates the expected zero proof");
    mip_code[1] = 0xbe9e0380u;
    CHECK(mip_proof(true), "restoring exact code restores its independent zero proof");
    mip_code[3] = 0x00050001u; // coordinates now start at v1, so mip is v3
    CHECK(mip_proof(false), "same-address mip-operand mutation selects an unproven register");
    mip_code[3] = 0x00050000u;
    CHECK(mip_proof(true), "A-B-A mip-operand mutation restores the original proof");

    Fixture f;
    clear_shader_decode_cache();
    const auto before = fold_control_plan_builds.load();
    CHECK(f.resolves(8), "real fold restores the entry descriptor pointer");
    const auto cold = fold_control_plan_builds.load();
    CHECK(cold > before, "real cold fold constructs control metadata");
    CHECK(f.resolves(8), "warm fold retains the expected resource");
    const bool cached = std::getenv("PROSPER_NO_FOLD_CONTROL_CACHE") == nullptr;
    CHECK(cached ? fold_control_plan_builds.load() == cold : fold_control_plan_builds.load() > cold,
          "warm cached path reuses metadata; explicit opt-out actually reconstructs it");
    f.descriptor[2] = 4;
    CHECK(f.resolves(4), "unchanged pointers reread current descriptor words");
    alignas(16) uint32_t second[4];
    std::copy(std::begin(f.descriptor), std::end(f.descriptor), std::begin(second));
    second[2] = 6;
    const auto pointer = uint64_t(uintptr_t(second));
    f.seed[8] = uint32_t(pointer); f.seed[9] = uint32_t(pointer >> 32);
    CHECK(f.resolves(6), "warm metadata follows a rewritten entry pointer");
    const auto builds = fold_control_plan_builds.load();
    f.code[3] = 0xBF800000u;
    CHECK(!f.resolves(6), "same-address branch rewrite changes the independently expected result");
    CHECK(fold_control_plan_builds.load() > builds, "same-address changed code reconstructs metadata");
    f.code[3] = 0xBF820004u;
    CHECK(f.resolves(6), "A-B-A code mutation restores correct branch semantics");

    // Separate seeds and descriptor storage share exactly one immutable code stream. Concurrent
    // eviction may drop the map owner, but must not invalidate a fold already borrowing it.
    Fixture other;
    other.descriptor[2] = 3;
    bool left = true, right = true;
    const auto worker = [&](Fixture& inputs, uint32_t count, bool& ok) {
        for (int n = 0; n < 200; ++n) {
            auto out = resolve_dynamic_fetch(f.code.data(), f.code.size(), inputs.seed, 12, 8);
            ok = ok && out.size() == 1 && out[0].desc.base == uint64_t(uintptr_t(inputs.data)) &&
                out[0].desc.num_records == count;
        }
    };
    std::thread a(worker, std::ref(f), 6, std::ref(left));
    std::thread b(worker, std::ref(other), 3, std::ref(right));
    for (int n = 0; n < 200; ++n) clear_shader_decode_cache();
    a.join(); b.join();
    CHECK(left && right, "parallel folds with cache clears preserve independent resources");

    if (argc > 1) {
        const bool long_stream = std::string_view(argv[1]) == "benchmark-long";
        if (long_stream) f.code.insert(f.code.begin(), 1500, 0xbe9e0380u);
        CHECK(f.resolves(6), "benchmark validates and warms its selected stream");
        const int calls = long_stream ? 10000 : 100000;
        const auto start = std::chrono::steady_clock::now();
        for (int n = 0; n < calls; ++n) if (!f.resolves(6)) ++failures;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("fold benchmark: calls=%d total_ms=%.3f cached=%d\n", calls, ms, int(cached));
    }
    std::printf("fold control plan: %d failures\n", failures);
    return failures ? 1 : 0;
}
