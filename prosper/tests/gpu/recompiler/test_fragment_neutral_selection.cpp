#include "gpu/recompiler/spirv_fragment_neutral_selection.hpp"
#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#include "../../fixtures/spirv_fragment_neutral_fixtures.hpp"
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
using namespace prosper::gpu;
using Type = FragmentNeutralType;

FragmentNeutralSelection graph() {
    FragmentNeutralSelection g;
    g.predicate = 10;
    g.preserve_f32 = true;
    g.values = {
        {1, {Type::Boolean, 41, 0, {}, true, 1}},
        {2, {Type::Boolean, 42, 0, {}, true, 0}},
        {3, {Type::Int32, 43, 0, {}, true, 0}},
        {4, {Type::Int32, 43, 0, {}, true, 1}},
        {5, {Type::Float32, 43, 0, {}}},
        {10, {Type::Boolean, 61, 0, {}}}, // varying, not an independently defined Input seed
        {11, {Type::Boolean, 61, 0, {}}},
        {12, {Type::Float32, 61, 0, {}}},
        {20, {Type::Int32, 169, 0, {10, 4, 3}}},
        {21, {Type::Boolean, 168, 0, {10}}},
    };
    g.body = {20, 21};
    g.exports = {{3, 20}, {1, 21}};
    return g;
}

void proof_controls() {
    const std::unordered_set<uint32_t> frozen{1, 2, 3, 4, 5};
    auto g = graph();
    check(prove_fragment_neutral_selection(g, frozen),
          "masked integer export and live termination state are neutral only under P=false");
    g.values[20] = {Type::Int32, 83, 0, {4}};
    check(!prove_fragment_neutral_selection(g, frozen), "one unmasked scalar export refuses");
    g = graph();
    g.values[22] = {Type::Boolean, 167, 0, {10, 11}};
    g.body.push_back(22); g.exports.push_back({2, 22});
    check(!prove_fragment_neutral_selection(g, frozen),
          "false AND unrelated possibly poisoned input is not a constant certificate");
    g.values[11].opcode = 1;
    check(prove_fragment_neutral_selection(g, frozen),
          "false AND actual undefined Boolean is constrained to stable false, unlike poison");
    g.values[11].opcode = 61;
    g.values[10] = {Type::Boolean, 167, 0, {1, 11}};
    check(prove_fragment_neutral_selection(g, frozen),
          "strict predicate dependence certifies scalar nonpoison without uniformizing the input");
    g = graph();
    g.values[30] = {Type::Other, 61, 0, {}};
    g.values[10] = {Type::Boolean, 81, 0, {30}};
    g.values[11] = {Type::Boolean, 81, 0, {30}};
    g.values[22] = {Type::Boolean, 167, 0, {10, 11}};
    g.body.push_back(22); g.exports.push_back({2, 22});
    check(!prove_fragment_neutral_selection(g, frozen),
          "one predicate component does not certify unrelated components of its vector");
    g = graph();
    g.values[24] = {Type::Float32, 12, 8, {12}};
    g.values[25] = {Type::Int32, 124, 0, {24}};
    g.values[20] = {Type::Int32, 169, 0, {10, 25, 3}};
    g.body = {24, 25, 20, 21};
    check(prove_fragment_neutral_selection(g, frozen),
          "unselected floating intrinsic poison is masked by an ordinary Select");
    g.values[24].extended_opcode = 14;
    check(prove_fragment_neutral_selection(g, frozen),
          "cosine execution is not a numerical definedness or special-value identity claim");
    g.values[26] = {Type::Boolean, 186, 0, {24, 5}};
    g.values[27] = {Type::Boolean, 167, 0, {10, 26}};
    g.body.insert(g.body.begin() + 2, {26, 27}); g.exports.push_back({2, 27});
    check(!prove_fragment_neutral_selection(g, frozen),
          "floating intrinsic result cannot acquire nonpoison from input typing or masking elsewhere");
    g = graph();
    g.values[22] = {Type::Int32, 134, 0, {4, 3}}; g.body.push_back(22);
    check(!prove_fragment_neutral_selection(g, frozen),
          "dead-result division by zero still refuses extra execution");
    g = graph();
    g.values[22] = {Type::Int32, 61, 0, {}}; g.body.push_back(22);
    check(!prove_fragment_neutral_selection(g, frozen), "even a dead body load requires separate authority");
    g = graph();
    g.values[22] = {Type::Float32, 207, 0, {5}}; g.body.push_back(22);
    check(!prove_fragment_neutral_selection(g, frozen), "even a dead body derivative refuses");
    g = graph();
    g.values[22] = {Type::Float32, 83, 0, {5}}; g.body.push_back(22); g.exports.push_back({5, 22});
    check(!prove_fragment_neutral_selection(g, frozen), "floating copies are not raw-bit export identity");
    g = graph();
    g.values[22] = {Type::Int32, 83, 0, {23}};
    g.values[23] = {Type::Int32, 1, 0, {}};
    g.body.push_back(22); g.exports.push_back({23, 22});
    check(!prove_fragment_neutral_selection(g, frozen), "undefined SSA atom equality is not stable identity");
    g = graph();
    g.values[22] = {Type::Boolean, 169, 0, {11, 1, 1}};
    g.body.push_back(22); g.exports.push_back({1, 22});
    check(!prove_fragment_neutral_selection(g, frozen), "equal Select alternatives do not mask a poisoned condition");
    g.values[11].opcode = 1;
    check(prove_fragment_neutral_selection(g, frozen), "equal defined Select alternatives constrain an undefined condition");
    g = graph();
    g.values[30] = {Type::Float32, 43, 0, {}, true, 0x7f800000u};
    g.values[31] = {Type::Int32, 124, 0, {30}};
    g.values[32] = {Type::Boolean, 171, 0, {31, 3}};
    g.values[33] = {Type::Boolean, 167, 0, {10, 32}};
    g.body = {31, 32, 33, 20, 21}; g.exports.push_back({2, 33});
    check(!prove_fragment_neutral_selection(g, frozen),
          "cross-float infinity bitcast may poison a live false conjunction even with preserve mode");
    g.values[33] = {Type::Boolean, 169, 0, {10, 32, 2}};
    g.values[34] = {Type::Boolean, 168, 0, {33}};
    g.body.push_back(34); g.exports = {{1, 34}};
    check(prove_fragment_neutral_selection(g, frozen),
          "inactive compare Select preserves defined live EXEC without nonpoison authority for its comparison");
    g.values[33].operands[0] = 1;
    g.values[35] = {Type::Boolean, 167, 0, {2, 33}};
    g.values[34].operands[0] = 35;
    g.body.insert(g.body.end() - 1, 35);
    check(!prove_fragment_neutral_selection(g, frozen),
          "active compare poison survives Select and cannot be absorbed by scalar FALSE AND");
    g.body.erase(g.body.end() - 2); g.values.erase(35); g.values[34].operands[0] = 33;
    g.values[33] = {Type::Boolean, 167, 0, {10, 32}};
    check(!prove_fragment_neutral_selection(g, frozen),
          "restoring strict AND loses the inactive live EXEC certificate");
    g.values[30].literal = 0;
    check(prove_fragment_neutral_selection(g, frozen),
          "explicit finite bits authorize scalar cross-float nonpoison, not general float identities");
    g = graph();
    g.values[30] = {Type::Float32, 43, 0, {}, true, 0x7f800000u};
    g.values[31] = {Type::Float32, 12, 4, {30}};
    g.values[32] = {Type::Boolean, 186, 0, {31, 5}};
    g.values[33] = {Type::Boolean, 167, 0, {10, 32}};
    g.values[34] = {Type::Boolean, 168, 0, {33}};
    g.body = {31, 32, 33, 34}; g.exports = {{1, 34}};
    check(!prove_fragment_neutral_selection(g, frozen),
          "floating compare propagates intrinsic poison through strict inactive AND");
    g.values[33] = {Type::Boolean, 169, 0, {10, 32, 2}};
    check(prove_fragment_neutral_selection(g, frozen),
          "inactive floating compare Select keeps defined live EXEC");
    g.values[33].operands[0] = 1;
    g.values[35] = {Type::Boolean, 167, 0, {2, 33}};
    g.values[34].operands[0] = 35;
    g.body.insert(g.body.end() - 1, 35);
    check(!prove_fragment_neutral_selection(g, frozen),
          "active floating compare poison survives Select and cannot be absorbed by scalar FALSE AND");
    g = graph();
    g.values[30] = {Type::Int32, 1, 0, {}};
    g.values[31] = {Type::Float32, 124, 0, {30}};
    g.values[32] = {Type::Boolean, 180, 0, {31, 5}};
    g.values[10] = {Type::Boolean, 167, 0, {2, 32}};
    g.values[33] = {Type::Float32, 124, 0, {30}};
    g.values[34] = {Type::Boolean, 180, 0, {33, 5}};
    g.values[35] = {Type::Boolean, 167, 0, {10, 34}};
    g.body = {33, 34, 35, 20, 21}; g.exports.push_back({2, 35});
    check(!prove_fragment_neutral_selection(g, frozen),
          "a finite header use cannot certify another independent consumption of Undef bits");
    check(frozen == std::unordered_set<uint32_t>({1, 2, 3, 4, 5}), "proof does not widen frozen load authority");
}

void dump(const std::filesystem::path& directory, const std::string& name,
          const std::vector<uint32_t>& words) {
    std::ofstream out(directory / (name + ".spv"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(uint32_t));
    check(out.good(), "strict fixture artifact written");
}

void module_controls(const char* directory) {
    namespace f = prosper::test::fragment_neutral;
    namespace base = prosper::test::fragment_votes;
    for (const bool floating_compare : {false, true}) {
        const auto strict_mask = f::make_module(floating_compare ? f::Shape::FloatPoisonConjunction :
                                                                  f::Shape::BitcastPoisonConjunction);
        auto selected_mask = strict_mask;
        const auto mask_at = base::find(selected_mask, 167);
        if (mask_at + 5 <= selected_mask.size()) {
            selected_mask.erase(selected_mask.begin() + mask_at, selected_mask.begin() + mask_at + 5);
            selected_mask.insert(selected_mask.begin() + mask_at, {(6u << 16) | 169u, 2, 84, 53, 81, 15});
        }
        check(selected_mask == f::make_module(floating_compare ? f::Shape::FloatPoisonSelection :
                                                                f::Shape::BitcastPoisonSelection),
              "poison AND/Select modules differ only in one mask instruction, with identical live EXEC Phi/Kill consumers");
    }
    for (const auto& fixture : f::fixtures()) {
        const auto words = f::make_module(fixture.shape), unchanged = words;
        const auto lowered = lower_fragment_votes(words);
        const bool admitted = lowered.refusal == FragmentVoteRefusal::None;
        check(admitted == fixture.admitted && words == unchanged &&
              lowered.words.empty() == !admitted && lowered.uniform_votes == 0 &&
              lowered.dead_votes == 0 && lowered.neutral_votes == (admitted ? 1u : 0u), fixture.name);
        if (!admitted) check(lowered.refusal == FragmentVoteRefusal::UnprovedVote,
                             "counterexample hits selection/definedness proof, not malformed inventory");
        else {
            namespace base = prosper::test::fragment_votes;
            const auto original_vote = base::find(words, 335);
            bool typed_true = false;
            for (size_t at = 5; at < lowered.words.size(); at += lowered.words[at] >> 16) {
                if ((lowered.words[at] & 0xffffu) != 83 || lowered.words[at + 2] != words[original_vote + 2]) continue;
                const auto true_id = lowered.words[at + 3];
                for (size_t decl = 5; decl < lowered.words.size(); decl += lowered.words[decl] >> 16)
                    if (lowered.words[decl] == ((3u << 16) | 41u) &&
                        lowered.words[decl + 1] == words[original_vote + 1] &&
                        lowered.words[decl + 2] == true_id) typed_true = true;
            }
            check(typed_true && base::find(lowered.words, 335) == lowered.words.size(),
                  "effective controller copies ordinary typed TRUE, not varying P or FALSE");
        }
        if (directory && fixture.strict) {
            dump(directory, fixture.name, words);
            if (admitted) dump(directory, std::string(fixture.name) + "_effective", lowered.words);
        }
    }
    for (const auto predicate : {f::Predicate::Helpers, f::Predicate::Visible,
                                f::Predicate::AllFalse, f::Predicate::AllTrue}) {
        const auto words = f::make_module(f::Shape::Masked, predicate);
        check(lower_fragment_votes(words).neutral_votes == 1,
              "all-false/all-true/visible/helper predicates gain no source uniformity fact");
    }
    const auto buffered = f::make_module(f::Shape::BufferPredicate), captured = buffered;
    const auto certified = lower_fragment_votes(buffered, true, true);
    const auto uncertified = lower_fragment_votes(buffered, false, true);
    check(certified.uniform_votes == 1 && certified.neutral_votes == 0 &&
          uncertified.uniform_votes == 0 && uncertified.neutral_votes == 1 &&
          !certified.words.empty() && !uncertified.words.empty() &&
          certified.words != uncertified.words && buffered == captured,
          "one source admits distinct owned Copy(P)/Copy(TRUE) buffer-certificate variants");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && !(argc == 3 && std::string_view(argv[1]) == "--dump")) return 2;
    if (argc == 3) std::filesystem::create_directories(argv[2]);
    proof_controls();
    module_controls(argc == 3 ? argv[2] : nullptr);
    std::printf("== %s: %d failures ==\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
