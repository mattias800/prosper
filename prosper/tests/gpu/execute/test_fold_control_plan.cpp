#include "gpu/resources/fold_control_plan.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>

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
    std::array<uint32_t, 9> code{0xBF860003u, 0xF4300404u, 0xFA000060u,
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

int main(int argc, char**) {
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
        constexpr int calls = 100000;
        const auto start = std::chrono::steady_clock::now();
        for (int n = 0; n < calls; ++n) if (!f.resolves(6)) ++failures;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("fold benchmark: calls=%d total_ms=%.3f cached=%d\n", calls, ms, int(cached));
    }
    std::printf("fold control plan: %d failures\n", failures);
    return failures ? 1 : 0;
}
