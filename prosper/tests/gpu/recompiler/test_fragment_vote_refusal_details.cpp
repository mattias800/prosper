#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#include "diagnostics/perf/wave64_refusal.hpp"
#include "../../fixtures/spirv_fragment_vote_fixtures.hpp"
#include "../../fixtures/test_scratch.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#ifdef _WIN32
#include <io.h>
#define detail_fileno _fileno
#define detail_dup _dup
#define detail_dup2 _dup2
#define detail_close _close
#else
#include <unistd.h>
#define detail_fileno fileno
#define detail_dup dup
#define detail_dup2 dup2
#define detail_close close
#endif

namespace {
int failures = 0, checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    failures += !ok;
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", what);
}
bool absent(const prosper::gpu::FragmentVoteFailureDetails& vote) {
    return !vote.available && vote.source_word == 0 && vote.vote_result_id == 0 &&
           vote.predicate_id == 0 && vote.predicate_opcode == UINT32_MAX;
}
bool same(const prosper::gpu::FragmentVoteLoweringDiagnostic& a,
          const prosper::gpu::FragmentVoteLoweringDiagnostic& b) {
    return a.attempted == b.attempted && a.refusal == b.refusal &&
        a.failed_vote.available == b.failed_vote.available &&
        a.failed_vote.source_word == b.failed_vote.source_word &&
        a.failed_vote.vote_result_id == b.failed_vote.vote_result_id &&
        a.failed_vote.predicate_id == b.failed_vote.predicate_id &&
        a.failed_vote.predicate_opcode == b.failed_vote.predicate_opcode;
}
}

int main(int argc, char** argv) {
    using namespace prosper::gpu;
    using namespace prosper::diagnostics::perf;
    namespace f = prosper::test::fragment_votes;
    const bool disabled = argc == 2 && std::strcmp(argv[1], "--disabled") == 0;
    if (argc != 1 && !disabled) return 2;
    FragmentVoteLoweringDiagnostic retained;
    for (const auto& fixture : f::fixtures()) {
        const auto source = fixture.words;
        const auto lowered = lower_fragment_votes(source, fixture.immutable_storage, true);
        const auto detail = lowered.diagnostic();
        check(detail.attempted && detail.refusal == lowered.refusal && source == fixture.words,
              "actual verdict metadata is observer-only and SOURCE is unchanged");
        check((lowered.refusal == FragmentVoteRefusal::None) == fixture.admitted &&
              lowered.words.empty() == !fixture.admitted,
              "admission and empty EFFECTIVE refusal remain the independent fixture contract");
        if (lowered.refusal != FragmentVoteRefusal::UnprovedVote) {
            check(absent(detail.failed_vote), "success/inventory refusal invents no failed vote");
            continue;
        }
        const auto& vote = detail.failed_vote;
        const bool location = vote.available && vote.source_word >= 5 &&
            vote.source_word <= source.size() - 5 && source[vote.source_word] == ((5u << 16) | 335u);
        check(location, "failed location names an actual SOURCE Any instruction, not guest PC");
        check(location && source[vote.source_word + 2] == vote.vote_result_id &&
              source[vote.source_word + 4] == vote.predicate_id,
              "vote result and predicate IDs are read from that actual source instruction");
        // Independent typed-Boolean definition walk: labels, decoration operands and literal
        // collisions are not definitions. Valid fixture predicates have one such definition.
        uint32_t definition = UINT32_MAX;
        const uint32_t predicate_type = location ? source[vote.source_word + 1] : UINT32_MAX;
        for (size_t at = 5; at < source.size(); at += source[at] >> 16) {
            const uint32_t count = source[at] >> 16, op = source[at] & 0xffffu;
            if (count >= 3 && source[at + 1] == predicate_type &&
                source[at + 2] == vote.predicate_id &&
                (op == 1 || op == 41 || op == 42 || op == 45 || op == 61 || op == 83 ||
                 (op >= 164 && op <= 191) || op == 245 || op == 335)) definition = op;
        }
        check(definition != UINT32_MAX && vote.predicate_opcode == definition,
              "reported predicate opcode is its independent typed source definition");
        check(lowered.uniform_votes == 0 && lowered.dead_votes == 0 && lowered.neutral_votes == 0,
              "a failed certificate still rolls back all admission counters");
        if (fixture.name == "safe_then_unsafe_vote_rolls_back") {
            const auto first = f::find(source, 335);
            size_t second = first + (source[first] >> 16);
            while (second < source.size() && (source[second] & 0xffffu) != 335)
                second += source[second] >> 16;
            check(second < source.size() && vote.source_word == second &&
                  vote.vote_result_id == 96 && vote.predicate_id == 51 && vote.predicate_opcode == 186,
                  "proved first vote is skipped: metadata identifies the actual second unproved vote");
            retained = detail;
            std::unordered_map<uint64_t, FragmentVoteLowering> memo;
            memo.emplace(1, lowered);
            check(same(memo.at(1).diagnostic(), retained), "copied memo retains exact source failure metadata");
            auto moved = std::move(memo.at(1));
            memo.clear();
            check(same(moved.diagnostic(), retained), "moved result survives memo owner destruction");
        }
    }
    check(retained.attempted && retained.failed_vote.available, "actual failed-result owner survived the fixture loop");
    auto malformed = f::fixtures().front().words;
    malformed.back() = 0;
    const auto bad = lower_fragment_votes(malformed);
    check(bad.refusal == FragmentVoteRefusal::MalformedModule && absent(bad.failed_vote),
          "malformed refusal does not fabricate vote details");
    auto inconsistent = f::fixtures().front().words;
    const auto marker = f::find(inconsistent, 330);
    inconsistent.erase(inconsistent.begin() + marker,
                       inconsistent.begin() + marker + (inconsistent[marker] >> 16));
    const auto contract = lower_fragment_votes(inconsistent);
    check(contract.refusal == FragmentVoteRefusal::InconsistentContract && absent(contract.failed_vote),
          "missing contract does not fabricate vote details");

    check(enabled() == !disabled, "ordinary and explicitly disabled observer modes are distinct");
    FILE* captured = std::fopen(
        prosper_test::test_scratch_file("fragment-vote-refusal-stderr.log").c_str(), "w+b");
    std::fflush(stderr);
    const int saved = detail_dup(detail_fileno(stderr));
    if (!captured || saved < 0 || detail_dup2(detail_fileno(captured), detail_fileno(stderr)) < 0) {
        if (captured) std::fclose(captured);
        if (saved >= 0) detail_close(saved);
        return 2;
    }
    const auto counter = [] { return ledger().counters[static_cast<size_t>(Counter::Wave64FragmentSubgroup)].load(); };
    const auto before = counter();
    const auto note = [&](uint64_t id, const FragmentVoteLoweringDiagnostic& diagnostic) {
        note_unsupported_wave64(Wave64Refusal::FragmentSubgroup, 64, 0x4071, id, 2, 32, 32, diagnostic);
    };
    note(0x40710001, retained);
    note(0x40710001, retained); // same identity: count every refusal, announce only once
    note(0x40710002, {});
    note(0x40710003, bad.diagnostic());
    auto missing_opcode = retained;
    missing_opcode.failed_vote.predicate_opcode = UINT32_MAX;
    note(0x40710004, missing_opcode);
    {
        const SuppressDrawDropCounting capture;
        note(0x40710005, retained); // must neither count nor consume its identity
    }
    note(0x40710005, retained);
    std::fflush(stderr);
    if (detail_dup2(saved, detail_fileno(stderr)) < 0) return 2;
    detail_close(saved);
    std::rewind(captured);
    std::string log;
    char buffer[512];
    while (const size_t n = std::fread(buffer, 1, sizeof buffer, captured)) log.append(buffer, n);
    std::fclose(captured);
    if (disabled) {
        check(log.empty() && counter() == before, "disabled observer emits nothing and counts no refused uses");
    } else {
        const std::string expected = "lowering=unproved-vote vote-source-word=" +
            std::to_string(retained.failed_vote.source_word) +
            " vote-result-id=96 vote-predicate-id=51 predicate-def-op=186";
        check(log.find(expected) != std::string::npos, "actual retained metadata reaches default-on stderr announcement");
        const auto first = log.find("identity=0x40710001");
        check(first != std::string::npos && log.find("identity=0x40710001", first + 1) == std::string::npos &&
              counter() == before + 6, "bounded announcement dedup preserves all actual refused-use counts");
        check(log.find("lowering=not-attempted") != std::string::npos &&
              log.find("lowering=malformed-module failed-vote=unavailable") != std::string::npos,
              "not-attempted and attempted refusal without vote metadata are explicitly different");
        check(log.find("predicate-def-op=unavailable") != std::string::npos,
              "missing predicate definition is not an invented numeric opcode");
        check(log.find("identity=0x40710005") != std::string::npos,
              "suppressed reanalysis cannot consume the later real announcement");
    }
    std::printf("== %s: %d checks, %d failures ==\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
