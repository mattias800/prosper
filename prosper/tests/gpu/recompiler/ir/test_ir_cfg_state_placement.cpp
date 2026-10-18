// test_ir_cfg_state_placement -- IR_CONTROL_FLOW_PLAN step 6. The state placement's phi sets
// (state_placement.hpp) against the structured emitter's own, three ways: per instruction against
// loop_written_regs on every committed fixture, per join at every loop header and if merge the
// structured route reaches when the fixtures are recompiled, and the guard itself -- a program
// whose placement differs from the emitter's set must compile to the same module as with the
// placement withheld.
#include "gpu/recompiler/ir/cfg/state_placement.hpp"

#include "gpu/recompiler/ir/cfg/region_shadow.hpp"

#include "fixtures/test_data.h"
#include "gpu/recompiler/ir/cfg_streams.hpp"
#include "gpu/recompiler/ir/cfg_structured_cases.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_copy_continuation.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"
#include "gpu/recompiler/rdna2_lane_slot_carry.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir::cfg;
using prosper::test::cfg_streams::walk;

namespace {

uint8_t nibble(char c) {
    return static_cast<uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
}

// A tests/data program: `.hex` holds the little-endian bytes as hex text, `.bin` the raw bytes.
std::vector<uint32_t> load_words(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::vector<uint8_t> bytes;
    if (path.extension() == ".hex") {
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        for (size_t i = 0; i + 1 < text.size(); i += 2)
            bytes.push_back(static_cast<uint8_t>(nibble(text[i]) << 4 | nibble(text[i + 1])));
    } else {
        bytes.assign(text.begin(), text.end());
    }
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}

std::vector<std::filesystem::path> guest_program_fixtures() {
    std::vector<std::filesystem::path> out;
    const auto data = prosper::test::tests_root(__FILE__) / "data";
    for (const auto& entry : std::filesystem::directory_iterator(data)) {
        const auto ext = entry.path().extension();
        const auto name = entry.path().filename().string();
        if (ext == ".hex" || (ext == ".bin" && name.find("symtab") == std::string::npos))
            out.push_back(entry.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

PhiSets inventory(const Rdna2Inst& in, int max_vgpr = 255) {
    PhiSets out;
    add_phi_writes(in, max_vgpr, out);
    return out;
}

// The one documented class where the inventory charges more than loop_written_regs: a 64-bit SOP2
// scalar-data result other than s_cselect/s_lshl/s_lshr, whose high word the legacy scan drops,
// and a 64-bit shift other than s_lshr into EXEC (AGENTS.md, "State placement").
bool documented_difference(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP2) return false;
    const char* name = rdna2_instruction_effects(in).name;
    if (!name) return false;
    const std::string op = name;
    const bool into_exec = in.dst.value == 126 || in.dst.value == 127;
    return op == "s_ashr_i64" || op == "s_bfe_u64" || op == "s_bfe_i64" || op == "s_bfm_b64" ||
           (op == "s_lshl_b64" && into_exec);
}

TEST(IrStatePlacement, InventoryEqualsTheEmitterScanOnEveryFixtureInstruction) {
    size_t compared = 0, writers = 0;
    for (const auto& path : guest_program_fixtures()) {
        const auto ins = walk(load_words(path));
        const int max_vgpr = shader_max_vgpr(ins);
        for (const Rdna2Inst& in : ins) {
            PhiSets legacy;
            loop_written_regs(ins, in.pc, in.pc + 1, legacy.vgprs, legacy.sgprs);
            const PhiSets placed = inventory(in, max_vgpr);
            ++compared;
            writers += !legacy.vgprs.empty() || !legacy.sgprs.empty();
            if (placed == legacy) continue;
            EXPECT_TRUE(documented_difference(in))
                << path.filename() << " pc=" << in.pc << " fmt=" << static_cast<int>(in.fmt)
                << " opcode=0x" << std::hex << in.opcode;
        }
    }
    EXPECT_GT(compared, 2000u);
    EXPECT_GT(writers, 1500u);
}

Rdna2Inst decode_one(std::vector<uint32_t> words) {
    words.push_back(prosper::test::cfg_streams::kEndpgm);
    const auto ins = walk(words);
    EXPECT_GE(ins.size(), 2u);
    return ins.empty() ? Rdna2Inst{} : ins[0];
}

TEST(IrStatePlacement, EachInventoryRule) {
    struct Rule {
        std::vector<uint32_t> words;
        std::set<int> vgprs, sgprs;
        const char* what;
    };
    const Rule rules[] = {
        {{0x7E0A0501u}, {}, {5}, "v_readfirstlane_b32 s5: its SGPR"},
        {{0x7E068501u}, {3, 4, 5, 6}, {}, "v_movreld_b32 v3: every VGPR from v3 up"},
        {{0x7E082101u}, {4}, {}, "v_cvt_f64_f32 v[4:5]: VDST alone (legacy VOP1 width)"},
        {{0xBE860400u}, {}, {6}, "s_mov_b64 s[6:7]: one word (legacy SOP1 width)"},
        {{0x87880200u}, {}, {8}, "s_and_b64 s[8:9]: one word (lane-mask pair)"},
        {{0x8F8A8200u}, {}, {10, 11}, "s_lshl_b64 s[10:11]: the pair"},
        {{0x90FE817Eu}, {}, {}, "s_lshr_b64 exec: the mask domain only"},
        {{0x928C8081u}, {}, {12, 13}, "s_bfm_b64 s[12:13]: the pair (the scan drops s13)"},
        {{0xBF8C0000u}, {}, {}, "s_waitcnt: nothing"},
    };
    for (const Rule& rule : rules) {
        const PhiSets got = inventory(decode_one(rule.words), 6);
        EXPECT_EQ(got.vgprs, rule.vgprs) << rule.what;
        EXPECT_EQ(got.sgprs, rule.sgprs) << rule.what;
    }
}

// The tree the structured emitter builds its placement from, under the shadow's stage policy
// (region_shadow.cpp derives the same `safe` set).
StatePlacement stage_placement(const std::vector<Rdna2Inst>& ins, ShadowStage stage,
                               const std::vector<uint32_t>& words,
                               std::unordered_set<uint32_t>& safe) {
    safe = safe_execz_branches(ins);
    for (uint32_t pc : waterfall_branches(ins)) safe.insert(pc);
    if (stage == ShadowStage::Fragment)
        for (uint32_t pc : fragment_mask_test_branches(ins, false)) safe.insert(pc);
    const bool compute = stage == ShadowStage::Compute;
    RegionPolicy policy;
    policy.allow_vcc = !compute;
    policy.compute_wave_branches = compute;
    policy.linearized = &safe;
    policy.fragment_wave_breaks = stage == ShadowStage::Fragment;
    policy.code = words.data();
    policy.dwords = words.size();
    return place_state(ins, policy);
}

// Every if and loop the detectors find, not only the joins a table-less recompile reaches: the
// legacy sets from the detector records (the emitter's ranges; an else-arm to its merge) against
// the placement's. Where the region shadow agrees, every record has a placement.
TEST(IrStatePlacement, EveryDetectedJoinOfEveryFixtureIsEqual) {
    size_t ifs = 0, loops = 0;
    for (const auto& path : guest_program_fixtures()) {
        const auto words = load_words(path);
        const auto ins = walk(words);
        for (ShadowStage stage :
             {ShadowStage::Vertex, ShadowStage::Fragment, ShadowStage::Compute}) {
            const RegionShadow shadow = shadow_regions(ins, stage, words.data(), words.size());
            if (shadow.verdict != RegionShadow::Verdict::Agree || shadow.tree_unstructured)
                continue;
            std::unordered_set<uint32_t> safe;
            const StatePlacement placement = stage_placement(ins, stage, words, safe);
            const std::string at = path.filename().string() + " stage " +
                                   std::to_string(static_cast<int>(stage));
            for (const ShadowIf& f : shadow.detector) {
                PhiSets legacy;
                loop_written_regs(ins, f.branch_pc + 1, f.has_else ? f.sb_pc : f.target_pc,
                                  legacy.vgprs, legacy.sgprs);
                if (f.has_else)
                    loop_written_regs(ins, f.target_pc, f.merge_pc, legacy.vgprs, legacy.sgprs);
                const IfPlacement* placed = placement.if_at(f.branch_pc);
                ASSERT_NE(placed, nullptr) << at << " if pc=" << f.branch_pc;
                EXPECT_EQ(placed->merge, legacy) << at << " if pc=" << f.branch_pc;
                ++ifs;
            }
            for (const ShadowLoop& l : shadow.detector_loops) {
                PhiSets carried, condition;
                loop_written_regs(ins, l.header_pc, l.backedge_pc, carried.vgprs, carried.sgprs);
                loop_written_regs(ins, l.header_pc, l.exit_branch_pc, condition.vgprs,
                                  condition.sgprs);
                const LoopPlacement* placed = placement.loop_at(l.header_pc);
                ASSERT_NE(placed, nullptr) << at << " loop pc=" << l.header_pc;
                EXPECT_EQ(placed->carried, carried) << at << " loop pc=" << l.header_pc;
                EXPECT_EQ(placed->condition, condition) << at << " loop pc=" << l.header_pc;
                EXPECT_EQ(placed->lane_slots, lane_slots_written(ins, l.header_pc, l.backedge_pc))
                    << at << " loop pc=" << l.header_pc;
                ++loops;
            }
        }
    }
    EXPECT_GT(ifs, 50u);
    EXPECT_GT(loops, 5u);
    RecordProperty("ifs", static_cast<int>(ifs));
    RecordProperty("loops", static_cast<int>(loops));
}

struct Census {
    size_t sites = 0, equal = 0, absent = 0;
    std::vector<std::string> differs;
};

void add(Census& total, const StatePlacementCensus& census, const std::string& where) {
    for (const auto& site : census.sites()) {
        ++total.sites;
        if (site.verdict == PlacementSiteRecord::Verdict::Equal) ++total.equal;
        if (site.verdict == PlacementSiteRecord::Verdict::Absent) ++total.absent;
        if (site.verdict == PlacementSiteRecord::Verdict::Differs)
            total.differs.push_back(where + " " + phi_site_name(site.site) +
                                    " pc=" + std::to_string(site.pc) + " " + site.difference);
    }
}

TEST(IrStatePlacement, EveryVisitedJoinOfEveryFixtureIsEqual) {
    Census total;
    for (const auto& path : guest_program_fixtures()) {
        const auto words = load_words(path);
        const std::string name = path.filename().string();
        {
            StatePlacementCensus census;
            (void)recompile_vertex(words.data(), words.size());
            add(total, census, name + " vertex");
        }
        {
            StatePlacementCensus census;
            (void)recompile_fragment(words.data(), words.size());
            add(total, census, name + " fragment");
        }
        {
            StatePlacementCensus census;
            (void)recompile_compute(words.data(), words.size(), nullptr, ComputeShaderConfig{});
            add(total, census, name + " compute");
        }
    }
    for (const auto& line : total.differs) ADD_FAILURE() << line;
    EXPECT_GT(total.equal, 0u) << "no fixture reached a guarded join";
    RecordProperty("sites", static_cast<int>(total.sites));
    RecordProperty("absent", static_cast<int>(total.absent));
}

// s_bfm_b64 s[40:41] in the then-arm: the placement charges s41, the emitter's scan does not.
std::vector<uint32_t> bfm_in_then_arm() {
    prosper::test::ir::cfg_structured::Stream s;
    s.choose(true, [&] {
        s.words.push_back(0x92A88081u);   // s_bfm_b64 s[40:41], 1, 0
        s.move(2);
    });
    s.words.push_back(prosper::test::cfg_streams::kBarrier);
    return s.words;
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& ops) {
    const auto p = prosper::test::ir::lane_program(ops, 64, 64, true, true);
    return recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
}

TEST(IrStatePlacement, TheGuardKeepsTheEmitterSetWhereThePlacementDiffers) {
    const auto ops = bfm_in_then_arm();
    std::vector<uint32_t> guarded, withheld;
    size_t differs = 0;
    {
        StatePlacementCensus census;
        guarded = compile(ops);
        for (const auto& site : census.sites())
            if (site.verdict == PlacementSiteRecord::Verdict::Differs) {
                ++differs;
                EXPECT_EQ(site.difference, "placed-only=s41");
            }
    }
    {
        StatePlacementCensus census(/*withhold=*/true);
        withheld = compile(ops);
        EXPECT_TRUE(census.sites().empty());
    }
    ASSERT_FALSE(guarded.empty());
    // Without this the comparison below proves nothing: the case must reach a differing join.
    ASSERT_EQ(differs, 1u);
    EXPECT_EQ(guarded, withheld);
    EXPECT_EQ(compile(ops), withheld);
}

TEST(IrStatePlacement, AnEqualIfMergeIsTakenFromThePlacement) {
    StatePlacementCensus census;
    ASSERT_FALSE(compile(prosper::test::ir::cfg_structured::words({1, 1})).empty());
    size_t equal_merges = 0;
    for (const auto& site : census.sites()) {
        EXPECT_EQ(site.verdict, PlacementSiteRecord::Verdict::Equal) << site.difference;
        equal_merges += site.site == PhiSite::IfMerge;
    }
    EXPECT_EQ(equal_merges, 1u);
}

}   // namespace
