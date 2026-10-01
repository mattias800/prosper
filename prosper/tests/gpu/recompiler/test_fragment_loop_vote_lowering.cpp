#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#include "gpu/recompiler/spirv_fragment_uniform_trace.hpp"
#include "../../fixtures/spirv_fragment_vote_fixtures.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string_view>

namespace {
int failures = 0;
void check(bool ok, const char* name) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", name);
    failures += !ok;
}
void graph_controls() {
    using namespace prosper::gpu;
    FragmentUniformTrace trace;
    trace.entry_block = 1;
    trace.successors = {{1, {2}}, {2, {3, 4}}, {3, {2}}, {4, {}}};
    trace.values = {{10, {2, {5, 11}, {1, 3}}}, {11, {3, {10, 6}, {}}},
                    {12, {2, {10, 6}, {}}}};
    trace.selectors = {12};
    const std::unordered_set<uint32_t> leaves{5, 6};
    check(prove_fragment_uniform_trace(trace, leaves, {12}).size() == 3,
          "initialized recurrence and its selector ground together");
    auto bad = trace;
    bad.values.at(10).operands[0] = 99;
    check(prove_fragment_uniform_trace(bad, leaves, {12}).empty(),
          "unknown initializer refuses the entire certificate");
    bad = trace;
    bad.values.at(10).operands[0] = 11;
    check(prove_fragment_uniform_trace(bad, leaves, {12}).empty(),
          "a constant somewhere in a cycle is not a real initializer");
    bad = trace;
    bad.values.at(10).phi_predecessors = {3, 3};
    check(prove_fragment_uniform_trace(bad, leaves, {12}).empty(),
          "back-edge-only Phi cannot initialize itself");
    bad = trace;
    bad.selectors.push_back(99);
    check(prove_fragment_uniform_trace(bad, leaves, {12}).empty(),
          "an unrelated varying selector invalidates the new trace certificate");
    bad = trace;
    bad.successors.emplace(9, std::vector<uint32_t>{9});
    check(prove_fragment_uniform_trace(bad, leaves, {12}).empty(),
          "unreachable cycles cannot create vacuous dominance authority");
    check(leaves == std::unordered_set<uint32_t>({5, 6}), "refused attempts leave prior certificates unchanged");
}
void dump(const std::filesystem::path& directory, const std::string& name,
          const std::vector<uint32_t>& words) {
    std::ofstream file(directory / (name + ".spv"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()),
               static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    check(file.good(), "fixture artifact written");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && !(argc == 3 && std::string_view(argv[1]) == "--dump")) return 2;
    const bool dumping = argc == 3;
    if (dumping) std::filesystem::create_directories(argv[2]);
    graph_controls();
    using namespace prosper::gpu;
    namespace f = prosper::test::fragment_votes;
    namespace loops = prosper::test::fragment_loop_votes;
    for (const auto& fixture : f::fixtures()) {
        const auto unchanged = fixture.words;
        const auto lowered = lower_fragment_votes(fixture.words, fixture.immutable_storage, true);
        check((lowered.refusal == FragmentVoteRefusal::None) == fixture.admitted &&
              lowered.uniform_votes == fixture.uniform_votes && lowered.dead_votes == fixture.dead_votes &&
              lowered.words.empty() == !fixture.admitted && fixture.words == unchanged, fixture.name.c_str());
        if (!fixture.admitted)
            check(lowered.refusal == (fixture.inventory_refusal ? FragmentVoteRefusal::UnsupportedWaveOperation :
                                                                FragmentVoteRefusal::UnprovedVote),
                  "counterexample refused for the intended proof rather than malformed input");
        if (dumping) {
            dump(argv[2], fixture.name, fixture.words);
            if (fixture.admitted) dump(argv[2], fixture.name + "_effective", lowered.words);
        }
    }
    const auto buffered = loops::make_module(loops::Shape::BufferBound);
    for (const bool immutable : {false, true}) for (const bool robust2 : {false, true})
        check((lower_fragment_votes(buffered, immutable, robust2).refusal == FragmentVoteRefusal::None) ==
              (immutable && robust2), "loop bounds still require both independent buffer authorities");
    std::printf("== %s: %d failures ==\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
