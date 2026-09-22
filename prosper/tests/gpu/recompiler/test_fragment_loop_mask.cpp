// The static admission contract is independent of SPIR-V and Vulkan. Every negative arm is a
// one-site mutation of an admitted decoded stream, and all arms run even after an earlier failure.
#include "gpu/recompiler/fragment_loop_mask.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

namespace {
int failures = 0;
#define CHECK(condition, name) do { \
    if (!(condition)) { std::printf("[FAIL] %s\n", name); ++failures; } \
    else std::printf("[ok] %s\n", name); \
} while (false)

constexpr uint32_t sop1(uint32_t op, uint32_t dst, uint32_t source) {
    return 0xbe800000u | (dst << 16) | (op << 8) | source;
}
constexpr uint32_t sop2(uint32_t op, uint32_t dst, uint32_t a, uint32_t b) {
    return 0x80000000u | (op << 23) | (dst << 16) | (b << 8) | a;
}
constexpr uint32_t branch(uint32_t op, int offset) {
    return 0xbf800000u | (op << 16) | (static_cast<uint32_t>(offset) & 0xffffu);
}
constexpr uint32_t end = 0xbf810000u;

std::vector<uint32_t> loop() {
    return {
        sop1(0x03, 84, 129),              // 0: s84 = 1
        sop1(0x03, 85, 130),              // 1: s85 = 2 (distinct high-word seed)
        sop1(0x04, 68, 126),              // 2: header: save EXEC in another pair
        sop2(0x15, 68, 84, 126),          // 3: s[68:69] = s[84:85] & ~EXEC
        sop2(0x11, 84, 84, 68),           // 4: s[84:85] |= s[68:69]
        branch(0x09, -4),                // 5: s_cbranch_execnz -> 2
        end,
    };
}

std::vector<uint32_t> alias_tail(std::initializer_list<uint32_t> body) {
    auto words = loop();
    words.resize(5);
    words.insert(words.end(), body.begin(), body.end());
    words.push_back(branch(0x09, 2 - static_cast<int>(words.size() + 1)));
    words.push_back(end);
    return words;
}

FragmentLoopMaskProof analyze(const std::vector<uint32_t>& words,
                              uint32_t header = 2, int pair = 84) {
    std::vector<Rdna2Inst> ins;
    const size_t consumed = rdna2_walk(words.data(), words.size(), ins);
    CHECK(consumed == words.size() && !ins.empty() && ins.back().is_end,
          "fixture fully decodes through S_ENDPGM");
    CHECK(std::all_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
              return in.fmt != Rdna2Format::Unknown;
          }), "fixture contains only recognized instruction encodings");
    return prove_fragment_loop_mask(ins, pair, header);
}

void reject(const std::vector<uint32_t>& words, const char* reason, const char* name,
            uint32_t header = 2, int pair = 84) {
    const auto proof = analyze(words, header, pair);
    if (proof.admitted || std::strcmp(proof.reason, reason) != 0)
        std::printf("  actual admitted=%d reason=%s pc=%u\n",
                    proof.admitted, proof.reason, proof.blocker_pc);
    CHECK(!proof.admitted && std::strcmp(proof.reason, reason) == 0, name);
}
} // namespace

int main(int argc, char** argv) {
    // Optional local witness: test_fragment_loop_mask <shader.bin> <pair-base> <header-pc>.
    // No title bytes or local artifact paths are committed with the synthetic regression tests.
    if (argc == 4) {
        std::ifstream stream(argv[1], std::ios::binary);
        const std::vector<char> bytes{std::istreambuf_iterator<char>(stream), {}};
        if (!stream || bytes.empty() || bytes.size() % sizeof(uint32_t)) return 2;
        std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
        std::memcpy(words.data(), bytes.data(), bytes.size());
        const auto proof = analyze(words, static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 0)),
                                   static_cast<int>(std::strtol(argv[2], nullptr, 0)));
        std::printf("witness admitted=%d reason=%s pc=%u definitions=%u/%u\n",
                    proof.admitted, proof.reason, proof.blocker_pc,
                    proof.definition_pc[0], proof.definition_pc[1]);
        return failures || !proof.admitted ? 1 : 0;
    }
    if (argc != 1) return 2;
    const auto control = analyze(loop());
    CHECK(control.admitted && control.definition_pc[0] == 0 && control.definition_pc[1] == 1,
          "two unconditional scalar definitions seed the mask lifetime before the loop");

    auto wide = loop();
    wide[0] = 0xf4000000u | (0x03u << 18) | (80u << 6) | 1u;
    wide[1] = 0xfa000000u;               // s_load_dwordx8 s[80:87], s[2:3], 0
    const auto wide_proof = analyze(wide);
    CHECK(wide_proof.admitted && wide_proof.definition_pc[0] == 0 &&
              wide_proof.definition_pc[1] == 0,
          "both words may be interior words of one wide SMEM definition");

    auto movk = loop();
    movk[0] = 0xb0000000u | (84u << 16) | 1u;
    movk[1] = 0xb0000000u | (85u << 16) | 2u;
    CHECK(analyze(movk).admitted, "MOVK immediate definitions are unconditional scalar words");

    auto prefix_read = loop();
    prefix_read.insert(prefix_read.begin() + 2, sop1(0x03, 10, 85));
    CHECK(analyze(prefix_read, 3).admitted,
          "scalar uses before the loop remain valid alongside the eagerly seeded Bool");
    auto prefix_compare = loop();
    prefix_compare.insert(prefix_compare.begin() + 2, 0xbf000000u | (0x12u << 16) | (128u << 8) | 84u);
    reject(prefix_compare, "derived-mask-data-or-reduction",
           "eager Bool seeding cannot redirect a prefix U64 compare through a wave vote", 3);
    auto prefix_quadmask = loop();
    prefix_quadmask.insert(prefix_quadmask.begin() + 2, sop1(0x2d, 70, 84));
    reject(prefix_quadmask, "derived-mask-data-or-reduction",
           "eager Bool seeding cannot redirect a prefix QUADMASK through a partial ballot", 3);
    auto prefix_arithmetic = loop();
    prefix_arithmetic.insert(prefix_arithmetic.begin() + 2, sop2(0x00, 10, 84, 129));
    CHECK(analyze(prefix_arithmetic, 3).admitted,
          "ordinary prefix B32 arithmetic keeps reading the loaded scalar word");
    auto unrelated_mulk = loop();
    unrelated_mulk.insert(unrelated_mulk.begin() + 2,
                         0xb0000000u | (0x10u << 23) | (106u << 16) | 100u);
    CHECK(analyze(unrelated_mulk, 3).admitted,
          "S_MULK's exact implicit VCC_LO read does not touch an unrelated promoted pair");
    auto prefix_buffer = loop();
    prefix_buffer.insert(prefix_buffer.begin() + 2,
        {0xf4000000u | (0x08u << 18) | (10u << 6) | 41u, 0xfa000000u});
    CHECK(analyze(prefix_buffer, 4).admitted,
          "prefix descriptors may consume either word of the retained scalar pair");

    // The seed precedes a forward conditional that may skip the entire loop. Admission therefore
    // remains sound only when the caller seeds eagerly, rather than inside the taken loop arm.
    auto skipped = loop();
    skipped.insert(skipped.begin() + 2, branch(0x04, 4)); // pc2 -> end pc7
    CHECK(analyze(skipped, 3).admitted, "an earlier forward if may skip the promoted loop");

    const std::vector<uint32_t> nested{
        sop1(0x03, 84, 129), sop1(0x03, 85, 130),
        sop1(0x04, 68, 126),             // outer header2
        sop2(0x15, 68, 84, 126),         // inner header3
        sop2(0x11, 84, 84, 68),
        branch(0x09, -3),               // inner -> 3
        branch(0x09, -5),               // outer -> 2
        sop1(0x04, 70, 84),             // mask consumer after both loop exits
        end,
    };
    CHECK(analyze(nested).admitted, "nested backedges and the loop exit retain one mask lifetime");

    auto undefined = loop();
    undefined[0] = sop1(0x03, 10, 129);
    reject(undefined, "pair-not-defined-in-entry-prefix", "missing low word cannot use a PHI zero");
    undefined = loop();
    undefined[1] = sop1(0x03, 10, 130);
    reject(undefined, "pair-not-defined-in-entry-prefix", "missing high word cannot use a stale input");

    auto conditional_definition = loop();
    conditional_definition.insert(conditional_definition.begin(), branch(0x04, 1));
    reject(conditional_definition, "prefix-pair-overwrite-or-unproved-definition",
           "a definition reachable only on one branch is not a prefix definition", 3);

    auto overwrite = loop();
    overwrite.insert(overwrite.begin() + 2, sop1(0x03, 85, 128));
    reject(overwrite, "prefix-pair-overwrite-or-unproved-definition",
           "later preheader overwrites invalidate the original pair proof", 3);

    auto copied = loop();
    copied[0] = sop1(0x03, 84, 10);
    reject(copied, "prefix-pair-overwrite-or-unproved-definition",
           "copying an unproved SGPR is not an unconditional scalar definition");

    for (uint32_t word : {84u, 85u}) {
        auto partial = loop();
        partial[4] = sop1(0x03, word, 129);
        reject(partial, "non-mask-or-partial-pair-write",
               word == 84 ? "scalar low-half write kills the mask lifetime"
                          : "scalar high-half write kills the mask lifetime");
    }
    auto overlapping = loop();
    overlapping[4] = sop2(0x11, 83, 84, 126);
    reject(overlapping, "non-mask-or-partial-pair-write",
           "overlapping B64 destination is not a complete replacement of this pair");

    auto scalar_copy = loop();
    scalar_copy[4] = sop1(0x04, 84, 90);
    reject(scalar_copy, "non-mask-or-partial-pair-write",
           "MOV_B64 from an unproved scalar pair cannot masquerade as a mask writer");

    auto wrong_root = loop();
    wrong_root[3] = sop2(0x15, 68, 85, 126);
    reject(wrong_root, "scalar-or-unmodelled-read",
           "a mask consumer rooted at the high half reads a different physical pair");

    for (uint32_t word : {84u, 85u}) {
        auto data_read = loop();
        data_read[2] = sop1(0x03, 10, word);
        reject(data_read, "scalar-or-unmodelled-read",
               word == 84 ? "a scalar low-word read cannot observe a promoted Bool"
                          : "a scalar high-word read cannot observe the old loaded high word");
    }
    auto exit_read = nested;
    exit_read[7] = sop1(0x03, 70, 85);
    reject(exit_read, "scalar-or-unmodelled-read", "scalar reads after loop exit are also refused");

    auto implicit_read = loop();
    implicit_read[2] = 0xb0000000u | (0x03u << 23) | (85u << 16) | 1u;
    reject(implicit_read, "scalar-or-unmodelled-read",
           "SOPK's implicit destination read is not hidden by an empty source list");

    const std::vector<uint32_t> buffer_address{
        sop1(0x03, 84, 129), sop1(0x03, 85, 130),
        0xf4000000u | (0x08u << 18) | (10u << 6) | 41u, 0xfa000000u,
        sop2(0x15, 68, 84, 126), sop2(0x11, 84, 84, 68),
        branch(0x09, -5), end,
    };
    reject(buffer_address, "scalar-memory-address-read",
           "S_BUFFER_LOAD reads all four descriptor words, including the upper pair");

    auto reenter = loop();
    reenter[5] = branch(0x09, -6);
    reject(reenter, "mask-lifetime-reenters-prefix", "a backedge cannot reenter scalar definitions");
    auto indirect = loop();
    indirect[2] = sop1(0x20, 0, 10);
    reject(indirect, "unmodelled-control-or-relative-sgpr", "indirect PC changes remain unproved");
    auto trap = loop();
    trap[2] = branch(0x12, 0);
    reject(trap, "unmodelled-control-flow", "unmodelled scalar control remains unproved");
    auto interpolation = loop();
    interpolation[2] = 0xc8000000u;
    reject(interpolation, "scalar-or-unmodelled-read", "unknown read inventories stay fail-closed");

    reject(loop(), "ordinary-aligned-pair-required", "odd SGPR pair roots are outside the contract", 2, 85);
    reject(loop(), "missing-loop-header", "the supplied header must name an instruction", 99);

    reject(alias_tail({sop1(0x04, 70, 84), sop1(0x03, 10, 70)}),
           "derived-mask-data-or-reduction", "a copied mask cannot escape through a scalar alias read");
    reject(alias_tail({sop1(0x04, 106, 84), branch(0x06, 0)}),
           "unbounded-derived-vcc-branch", "VCCZ observes unrepresented bits through a copied alias");
    CHECK(analyze(alias_tail({sop1(0x04, 106, 84), 0x7d840000u, branch(0x06, 0)})).admitted,
          "a fresh VOPC overwrites an unsafe VCC alias before its wave branch");
    CHECK(analyze(alias_tail({sop2(0x0f, 106, 84, 126), branch(0x06, 0)})).admitted,
          "AND with EXEC bounds a derived VCC mask for its wave branch");
    CHECK(analyze(alias_tail({sop2(0x15, 106, 126, 84), branch(0x06, 0)})).admitted,
          "ANDN2 is bounded when its left operand is EXEC");
    reject(alias_tail({sop2(0x15, 106, 84, 126), branch(0x06, 0)}),
           "unbounded-derived-vcc-branch", "ANDN2 is not bounded merely because its right operand is EXEC");
    reject(alias_tail({sop2(0x0f, 106, 84, 126), sop1(0x08, 106, 106), branch(0x06, 0)}),
           "unbounded-derived-vcc-branch", "NOT recreates omitted one bits after a bounding AND");
    reject(alias_tail({sop2(0x0f, 106, 84, 126), sop2(0x1d, 106, 106, 106), branch(0x06, 0)}),
           "unbounded-derived-vcc-branch", "XNOR recreates omitted one bits even from bounded operands");
    reject(alias_tail({sop1(0x04, 70, 84), sop1(0x0a, 72, 70)}),
           "derived-mask-wqm", "WQM cannot treat a derived arbitrary mask as an identity");
    reject(alias_tail({sop2(0x0f, 70, 84, 126), sop1(0x03, 10, 70)}),
           "derived-mask-data-or-reduction", "bounding a mask does not create a scalar SSA word");
    reject(alias_tail({sop1(0x04, 126, 84)}),
           "unbounded-derived-exec-write", "unbounded mask bits cannot enable EXEC lanes");
    reject(alias_tail({sop1(0x04, 70, 84), sop1(0x03, 71, 128)}),
           "partial-derived-mask-write", "a partial alias write cannot clear pair uncertainty");
    CHECK(analyze(alias_tail({sop1(0x04, 70, 84),
                             0xf4000000u | (1u << 18) | (70u << 6) | 1u, 0xfa000000u,
                             sop1(0x03, 10, 70)})).admitted,
          "a complete scalar load ends an alias lifetime before a later scalar read");
    reject(alias_tail({sop2(0x0a, 10, 128, 129)}),
           "derived-mask-scc-consumer", "implicit SCC consumers cannot use an uncertain mask reduction");
    reject(alias_tail({sop1(0x03, 10, 253)}),
           "derived-mask-scc-source", "SSRC253 is an explicit uncertain SCC consumer");
    reject(alias_tail({sop1(0x04, 106, 84), branch(0x08, 1), 0x7d840000u, branch(0x06, 0)}),
           "unbounded-derived-vcc-branch", "one safe predecessor cannot hide an unsafe VCC alias at a join");
    std::printf("fragment_loop_mask: %d failures\n", failures);
    return failures ? 1 : 0;
}
