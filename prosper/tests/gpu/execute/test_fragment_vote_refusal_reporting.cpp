// Exercise the shipped backend reporting route, not just a diagnostic formatter. Refused
// modules never create a fragment pipeline; the clear remains the unchanged render result.
#include "fixtures/render_runner.h"
#include "fixtures/spirv_fragment_vote_execution.hpp"
#include "fixtures/spirv_fragment_vote_fixtures.hpp"
#include "fixtures/test_scratch.h"
#include <cstdio>
#include <string>
#ifdef _WIN32
#include <io.h>
#define reporting_fileno _fileno
#define reporting_dup _dup
#define reporting_dup2 _dup2
#define reporting_close _close
#else
#include <unistd.h>
#define reporting_fileno fileno
#define reporting_dup dup
#define reporting_dup2 dup2
#define reporting_close close
#endif

namespace {
int failures = 0, checks = 0;
void check(bool ok, const char* name) {
    ++checks;
    failures += !ok;
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", name);
}
}
int main() {
    namespace backend = prosper::test;
    namespace f = prosper::test::fragment_vote_execution;
    namespace inputs = prosper::test::fragment_votes;
    namespace perf = prosper::diagnostics::perf;
    using prosper::gpu::FragmentWavePolicy;
    const auto& context = backend::render_vk_ctx();
    if (!context.ok) {
        std::puts("SKIP: Vulkan backend unavailable");
        return 77;
    }
    if (context.subgroup_size_control && context.min_subgroup_size <= 64 &&
        context.max_subgroup_size >= 64 &&
        (context.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (context.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (context.subgroup_operations & VK_SUBGROUP_FEATURE_VOTE_BIT)) {
        std::puts("SKIP: native fragment Wave64 bypasses this refusal/lowering route");
        return 77;
    }
    std::vector<uint32_t> words;
    for (const auto& fixture : inputs::fixtures())
        if (fixture.name == "safe_then_unsafe_vote_rolls_back") words = fixture.words;
    if (words.empty()) return 2;
    const auto captured_source = words;
    const auto first = inputs::find(words, 335);
    size_t second = first + (words[first] >> 16);
    while (second < words.size() && (words[second] & 0xffffu) != 335) second += words[second] >> 16;
    check(second < words.size() && words[second + 2] == 96 && words[second + 4] == 51,
          "project-owned specimen has an earlier proved vote and a later live unproved vote");
    const auto source = std::make_shared<const std::vector<uint32_t>>(words);
    const auto vertex = f::edge_vertex(2, 2);
    const auto draw = [&](uint64_t identity, uint64_t program, FragmentWavePolicy policy) {
        backend::BackendDraw d;
        d.vs = vertex; d.fs_shared = source; d.fs_identity = identity; d.fs_guest_addr = program;
        d.vcount = 3; d.fragment_wave_policy = policy;
        for (uint32_t set = 0; set < 2; ++set) {
            backend::FrameResource cb; cb.binding = 2; cb.set = set; d.R.push_back(cb);
            backend::FrameResource vb; vb.binding = 3; vb.set = set; d.R.push_back(vb);
        }
        backend::FrameResource values; values.binding = 0; values.dwords = {0, 0x3f800000u};
        d.R.push_back(values);
        const auto pixels = backend::render_draws_rgba({d}, f::width, f::height);
        bool clear = pixels.size() == f::width * f::height * 4;
        if (clear) for (size_t i = 0; i < pixels.size(); i += 4)
            clear &= pixels[i] == 0 && pixels[i + 1] == 0 && pixels[i + 2] == 255 && pixels[i + 3] == 255;
        check(clear, "actual refused draw preserves the backend clear rather than executing a narrow vote");
    };
    FILE* captured = std::fopen(
        prosper_test::test_scratch_file("fragment-vote-reporting-stderr.log").c_str(), "w+b");
    std::fflush(stderr);
    const int saved = reporting_dup(reporting_fileno(stderr));
    if (!captured || saved < 0 || reporting_dup2(reporting_fileno(captured), reporting_fileno(stderr)) < 0) {
        if (captured) std::fclose(captured);
        if (saved >= 0) reporting_close(saved);
        return 2;
    }
    const auto count = [] { return perf::ledger().counters[static_cast<size_t>(perf::Counter::Wave64FragmentSubgroup)].load(); };
    const auto before = count();
    draw(0x40711001, 0x40710001, FragmentWavePolicy::ProvenVotes);
    draw(0x40711001, 0x40710001, FragmentWavePolicy::ProvenVotes);
    {
        const perf::SuppressDrawDropCounting capture;
        draw(0x40711002, 0x40710002, FragmentWavePolicy::ProvenVotes);
    }
    draw(0x40711002, 0x40710002, FragmentWavePolicy::ProvenVotes);
    draw(0, 0x40710003, FragmentWavePolicy::ProvenVotes);
    draw(0x40711004, 0x40710004, FragmentWavePolicy::Strict);
    std::fflush(stderr);
    if (reporting_dup2(saved, reporting_fileno(stderr)) < 0) return 2;
    reporting_close(saved);
    std::rewind(captured);
    std::string log;
    char buffer[512];
    while (const size_t n = std::fread(buffer, 1, sizeof buffer, captured)) log.append(buffer, n);
    std::fclose(captured);
    // Capturing the observer must not hide Vulkan-layer messages from the CI validation scan.
    std::fwrite(log.data(), 1, log.size(), stderr);
    const auto line = [&](const char* program) {
        const auto at = log.find(std::string("[wave64-unsupported] stage=fragment program=") + program + " ");
        return at == std::string::npos ? std::string{} : log.substr(at, log.find('\n', at) - at);
    };
    const std::string expected = "lowering=unproved-vote vote-source-word=" + std::to_string(second) +
        " vote-result-id=96 vote-predicate-id=51 predicate-def-op=186";
    check(line("0x40710001").find(expected) != std::string::npos,
          "actual cold memo refusal reports the exact first failed source vote");
    check(line("0x40710002").find(expected) != std::string::npos,
          "actual warm memo retains details across suppressed cold capture");
    check(line("0x40710003").find(expected) != std::string::npos,
          "actual zero-identity uncached result survives until final refusal announcement");
    check(line("0x40710004").find("lowering=not-attempted") != std::string::npos,
          "actual Strict route does not pretend a proof was attempted");
    const auto occurrence = log.find("identity=0x40711001");
    check(occurrence != std::string::npos && log.find("identity=0x40711001", occurrence + 1) == std::string::npos &&
          count() == before + 5, "actual backend dedup and capture suppression preserve refused-use counts");
    check(*source == captured_source, "all production routes preserve the complete shared SOURCE bytes");
    std::printf("== %s: %d checks, %d failures ==\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
