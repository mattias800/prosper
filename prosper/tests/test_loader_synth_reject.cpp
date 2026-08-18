// test_loader_synth_reject — the corrupted half of the loader fixtures (#2567).
//
// WHY THIS FILE EXISTS SEPARATELY FROM test_loader_synth_link.cpp
// --------------------------------------------------------------
// A fixture the loader ACCEPTS proves only that the generator and the loader agree with each other,
// which is a weaker claim than the one it is tempting to make. Every arm here is a module that
// differs from a linking control in exactly ONE field, so each arm can show its lever moved: if the
// rejection were unconditional, the control would fail; if it were absent, the arm would.
//
// The arms are grouped by what the loader must do, because "reject" is not one behaviour:
//   * REFUSE the module — `Module::load` fails and `link_program` returns false with the path in the
//     message (a wrong machine, a missing PT_DYNAMIC, a file truncated before its program headers);
//   * REFUSE the module at IMAGE-BUILD time — everything parses, and only the PT_LOAD copy can see
//     that the file is shorter than the segment claims (#2631). These are the arms whose control is
//     doing the most work: the corrupted module is byte-identical to the control up to the cut, so
//     an arm that "passes" by refusing broadly would take the control down with it;
//   * REFUSE the relocation but keep the module — a relocation whose target is outside the mapped
//     image, or straddles its end, must not be applied and must not write past the buffer. The
//     straddle pair is four bytes apart and lands on opposite sides of module.cpp's `p + 8 > end`
//     guard, so it is a razor rather than a smoke test.
//
// One arm is built BY HAND, outside `synth_prx.h`, for the reason recorded in `handmade_prx.h`: a
// corrupted variant produced by the same generator as the control inherits that generator's geometry,
// so a malformation it cannot express would read as a clean pass.
#include "../src/hle/nid.hpp"
#include "../src/loader/linker.hpp"
#include "../src/self/module.hpp"
#include "handmade_prx.h"
#include "synth_prx.h"
#include "test_scratch.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

using namespace prosper;
using prosper_test::SynthModuleSpec;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

// Typed constants rather than an enum: these are added to the fixtures' own layout enums, and
// enum + enum arithmetic is deprecated in C++20.
static constexpr uint64_t kBase0    = 0x400000000ull;
static constexpr uint64_t kBase1    = 0x500000000ull;
static constexpr uint64_t kStubBase = 0x700000000ull;

static bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

int main() {
    printf("== test_loader_synth_reject ==\n");

    const std::string dir = prosper_test::test_scratch_dir().string();
    const std::string kExport = nid_hash("prosperSynthRejectExport");
    const std::string kHandA  = nid_hash("prosperHandmadeExportA");
    const std::string kHandB  = nid_hash("prosperHandmadeExportB");
    const std::string kHandI  = nid_hash("prosperHandmadeImport");

    auto emit = [&](const char* name, const SynthModuleSpec& spec) {
        const std::string path = dir + "/" + name;
        std::string err;
        if (!prosper_test::write_synth_prx(path, spec, &err)) {
            printf("  [FAIL] cannot write fixture %s: %s\n", name, err.c_str());
            fails++;
        }
        return path;
    };

    // Every module-level arm links against this one, so a failure is attributable to the arm.
    SynthModuleSpec good_spec; good_spec.exports = { kExport };
    const std::string good_path = emit("reject_good.prx", good_spec);

    // Link `path` as the second module behind the known-good one. Returns the error text; empty on
    // success.
    auto link_error = [&](const std::string& path) {
        const std::vector<LinkInput> inputs = { { good_path, kBase0 }, { path, kBase1 } };
        Program p; std::string err;
        return link_program(inputs, kStubBase, p, &err) ? std::string() : err;
    };

    // ---- module-level rejections. Each spec differs from `good_spec` in exactly one field. -------
    CHECK(link_error(good_path).empty(), "control: an unmodified fixture links");

    {
        const std::string missing = dir + "/reject_absent.prx";
        const std::string e = link_error(missing);
        CHECK(!e.empty() && contains(e, "cannot open file") && contains(e, missing),
              "a path with no file is refused, and the message names it");
    }
    {
        SynthModuleSpec s = good_spec; s.e_machine = 0x28;   // EM_ARM
        const std::string p = emit("reject_machine.prx", s);
        const std::string e = link_error(p);
        CHECK(!e.empty() && contains(e, "not x86-64"), "a non-x86-64 module is refused");
    }
    {
        SynthModuleSpec s = good_spec; s.omit_dynamic_phdr = true;
        const std::string p = emit("reject_nodynamic.prx", s);
        const std::string e = link_error(p);
        CHECK(!e.empty() && contains(e, "no PT_DYNAMIC"), "a module with no PT_DYNAMIC is refused");
    }
    {
        // Truncated before the program header table is complete: the PT_DYNAMIC header cannot be
        // read, so the module has no dynamic segment at all.
        SynthModuleSpec s = good_spec; s.truncate_to = 0x60;
        const std::string p = emit("reject_truncated_phdr.prx", s);
        const std::string e = link_error(p);
        CHECK(!e.empty() && contains(e, "no PT_DYNAMIC"),
              "a file truncated inside its program header table is refused");
    }
    {
        SynthModuleSpec s = good_spec; s.truncate_to = 4;    // only the magic survives
        const std::string p = emit("reject_truncated_all.prx", s);
        const std::string e = link_error(p);
        CHECK(!e.empty(), "a four-byte file is refused");
    }

    // ---- truncated modules: the FILE is shorter than a PT_LOAD says it is (#2631) ---------------
    // These are the quiet ones. A cut inside the program header table (above) removes PT_DYNAMIC and
    // is refused by `Module::load`; a cut AFTER the tables removes nothing the parser reads, so every
    // earlier stage reports success and only the segment copy in `build_image` can see it. Before
    // #2631 that copy was skipped in full - the segment mapped all-zero, the bytes that DO exist were
    // discarded, and build_image returned true. Each arm differs from `good_spec` in `truncate_to`
    // alone, and the "control" line above (an unmodified fixture links) is that field at its default.
    {
        // The measurement recorded in #2631: 0x1000 declared, 0x400 present.
        SynthModuleSpec s = good_spec; s.truncate_to = 0x400;
        const std::string path = emit("reject_truncated_load.prx", s);

        std::string err;
        auto mod = Module::load(path, &err);
        CHECK(mod.has_value(),
              "truncated PT_LOAD: the module still PARSES - no earlier stage can see the cut");
        if (mod) {
            CHECK(mod->symbols.size() == 2,
                  "truncated PT_LOAD: its symbol table still reads back (null symbol + the export)");
            LoadedImage img;
            CHECK(!build_image(*mod, kBase1, img, &err),
                  "truncated PT_LOAD: build_image refuses instead of mapping the segment all-zero");
            CHECK(contains(err, "declares 0x1000 bytes"),
                  "truncated PT_LOAD: the refusal quotes the length the segment declared");
            CHECK(contains(err, "the file holds only 0x400 bytes"),
                  "truncated PT_LOAD: the refusal quotes what the file actually holds");
        }
        const std::string e = link_error(path);
        CHECK(!e.empty() && contains(e, path) && contains(e, "the file holds only"),
              "truncated PT_LOAD: link_program fails and the message names the path");
    }
    {
        // The razor: one 0x100 block short of the declared 0x1000, so EVERY table the loader reads -
        // dynamic, symtab, strtab, RELA, JMPREL, DT_INIT_ARRAY, DT_INIT - survives intact and the
        // module is fully parseable. Deliberately strict: the missing tail happens to be the
        // fixture's zero padding, and prosper still refuses it, because a file that does not contain
        // a segment cannot be shown to contain zeros there. Measured to cost nothing on real content
        // - across the 44 local dumps no PT_LOAD even ENDS at EOF, let alone runs past it.
        SynthModuleSpec s = good_spec; s.truncate_to = prosper_test::kSynthFileSize - 0x100;
        const std::string path = emit("reject_truncated_tail.prx", s);

        std::string err;
        auto mod = Module::load(path, &err);
        CHECK(mod.has_value(), "one-block-short PT_LOAD: the module still parses");
        if (mod) {
            CHECK(mod->symbols.size() == 2 && mod->symbols[1].nid == kExport,
                  "one-block-short PT_LOAD: its export still resolves by NID - nothing else notices");
            LoadedImage img;
            CHECK(!build_image(*mod, kBase1, img, &err),
                  "one-block-short PT_LOAD: build_image refuses");
            CHECK(contains(err, "the file holds only 0xf00 bytes"),
                  "one-block-short PT_LOAD: the refusal quotes the short length");
        }
    }
    {
        // ... and the same knob set to the file's own size truncates nothing, so the arms above
        // cannot be passing by refusing every module that carries a `truncate_to` at all.
        SynthModuleSpec s = good_spec; s.truncate_to = prosper_test::kSynthFileSize;
        const std::string path = emit("truncate_noop.prx", s);
        CHECK(link_error(path).empty(),
              "truncate control: a cut at exactly the declared size removes nothing and links");
    }

    // ---- relocation-level rejections ------------------------------------------------------------
    // These use Module::load / build_image / apply_relocations directly, because the discriminator is
    // apply_relocations' returned count — link_program discards it.
    //
    // The fixture has no imports and no exports, so its only relocation is the R_X86_64_RELATIVE that
    // fills DT_INIT_ARRAY[0]. `applied` is therefore 1 or 0 with nothing else moving.
    struct RelocArm { const char* name; uint64_t offset; bool expect_applied; };
    const uint64_t kEnd = prosper_test::kSynthImageSize;
    const RelocArm arms[] = {
        { "reloc_control.prx",  prosper_test::kSynthInitArray, true  },  // inside the file
        { "reloc_far.prx",      0x40000,                       false },  // past the whole image
        { "reloc_last8.prx",    kEnd - 8,                      true  },  // last legal 8-byte word
        { "reloc_straddle.prx", kEnd - 4,                      false },  // 4 bytes over the end
    };
    for (const auto& arm : arms) {
        SynthModuleSpec s;                       // no exports, no imports; DT_INIT_ARRAY only
        s.relative_reloc_offset = arm.offset;
        const std::string path = emit(arm.name, s);

        std::string err;
        auto mod = Module::load(path, &err);
        CHECK(mod.has_value(), (std::string("reloc arm ") + arm.name + ": the module still parses").c_str());
        if (!mod) { printf("  load error: %s\n", err.c_str()); continue; }
        CHECK(mod->relocs.size() == 1,
              (std::string("reloc arm ") + arm.name + ": exactly one relocation was emitted").c_str());

        LoadedImage img;
        CHECK(build_image(*mod, kBase1, img, &err),
              (std::string("reloc arm ") + arm.name + ": the image builds").c_str());
        CHECK(img.mem.size() == prosper_test::kSynthImageSize,
              (std::string("reloc arm ") + arm.name + ": the image is one 16 KiB page").c_str());

        const size_t applied = apply_relocations(*mod, img);
        CHECK(applied == (arm.expect_applied ? 1u : 0u),
              (std::string("reloc arm ") + arm.name + ": application matches the expectation").c_str());

        // Read back the in-bounds bytes at the relocation's own target. For the straddling arm the
        // first four bytes ARE inside the image, so "still zero" is direct, in-bounds evidence that
        // the write did not happen at all rather than happening partially.
        if (arm.offset < prosper_test::kSynthImageSize) {
            const uint8_t* q = img.at(kBase1 + arm.offset);
            CHECK(q != nullptr,
                  (std::string("reloc arm ") + arm.name + ": the target's first byte is in-image").c_str());
            if (q) {
                uint32_t low = 0; memcpy(&low, q, 4);
                CHECK(arm.expect_applied ? (low != 0) : (low == 0),
                      (std::string("reloc arm ") + arm.name +
                       ": the target word was written exactly when it should have been").c_str());
            }
        }
        if (arm.expect_applied && arm.offset == prosper_test::kSynthInitArray) {
            uint64_t v = 0; memcpy(&v, img.at(kBase1 + arm.offset), 8);
            CHECK(v == kBase1 + prosper_test::kSynthCtor,
                  "reloc control: DT_INIT_ARRAY[0] holds the relocated constructor address");
        }
    }

    // The module-level consequence: a relocation the loader refused must not leave a bogus entry in
    // the collected init functions. DT_INIT is still collected, so the count discriminates.
    {
        SynthModuleSpec ctl;  const std::string ctl_path = emit("initfn_control.prx", ctl);
        SynthModuleSpec bad;  bad.relative_reloc_offset = 0x40000;
        const std::string bad_path = emit("initfn_bad.prx", bad);

        auto init_fns_of = [&](const std::string& path) {
            const std::vector<LinkInput> inputs = { { good_path, kBase0 }, { path, kBase1 } };
            Program p; std::string err;
            if (!link_program(inputs, kStubBase, p, &err)) return std::vector<uint64_t>{ 0xbad };
            return p.init_fns;
        };
        const std::vector<uint64_t> ctl_fns = init_fns_of(ctl_path);
        const std::vector<uint64_t> bad_fns = init_fns_of(bad_path);
        CHECK(ctl_fns.size() == 2, "init control: DT_INIT plus the relocated DT_INIT_ARRAY entry");
        CHECK(bad_fns.size() == 1,
              "init: an unapplied DT_INIT_ARRAY relocation contributes no init function");
        CHECK(!bad_fns.empty() && bad_fns[0] == kBase1 + prosper_test::kSynthInit,
              "init: DT_INIT itself is unaffected by the refused relocation");
    }

    // ---- hand-built arms, outside the generator --------------------------------------------------
    {
        // (a) The smallest possible hand-written instance: a literal byte array. Nothing in
        // synth_prx.h produced this, and no field of it came from a spec.
        const std::vector<uint8_t> stub_bytes = { 0x7f, 'E', 'L', 'F', 2, 1, 1 };
        const std::string stub_path = dir + "/handmade_stub.prx";
        std::string err;
        if (!prosper_test::write_module_bytes(stub_path, stub_bytes, &err)) {
            printf("  [FAIL] cannot write the hand-written stub: %s\n", err.c_str());
            fails++;
        } else {
            const std::string e = link_error(stub_path);
            CHECK(!e.empty(), "hand-written: a seven-byte ELF fragment is refused");
        }

        // (b) The hand-laid module from handmade_prx.h, with ONE field corrupted by hand: its
        // PT_DYNAMIC program header retyped to PT_NULL. The uncorrupted bytes are the control, so
        // this arm cannot pass by rejecting everything.
        std::vector<uint8_t> hand = prosper_test::handmade_prx_bytes(kHandA, kHandB, kHandI);
        const std::string hand_good = dir + "/handmade_good.prx";
        CHECK(prosper_test::write_module_bytes(hand_good, hand, &err),
              "hand-built: the control fixture is written");
        CHECK(link_error(hand_good).empty(), "hand-built control: the uncorrupted module links");

        const uint32_t pt_null = 0;
        memcpy(&hand[prosper_test::kHandPhdr2Off], &pt_null, 4);   // p_type: PT_DYNAMIC -> PT_NULL
        const std::string hand_bad = dir + "/handmade_nodynamic.prx";
        CHECK(prosper_test::write_module_bytes(hand_bad, hand, &err),
              "hand-built: the corrupted fixture is written");
        const std::string e = link_error(hand_bad);
        CHECK(!e.empty() && contains(e, "no PT_DYNAMIC"),
              "hand-built: retyping the PT_DYNAMIC header to PT_NULL is refused");

        // (c) The truncation arm, in a geometry the generator cannot express (#2631). The hand-laid
        // module has TWO PT_LOADs - text at file 0x0000 and data at file 0x1000 - so cutting the file
        // at 0x1800 leaves program header 0 complete and takes 0x800 bytes off program header 1. Every
        // dynamic table lives below 0x1800, so the module parses in full and both exports and the
        // import still resolve: the ONLY stage that can see the damage is the segment copy. A
        // single-PT_LOAD fixture cannot pose this question at all - there, a short file always cuts
        // the one segment there is - which is the whole reason this file exists.
        std::vector<uint8_t> cut = prosper_test::handmade_prx_bytes(kHandA, kHandB, kHandI);
        cut.resize(0x1800);
        const std::string hand_cut = dir + "/handmade_truncated.prx";
        CHECK(prosper_test::write_module_bytes(hand_cut, cut, &err),
              "hand-built: the truncated fixture is written");

        auto cut_mod = Module::load(hand_cut, &err);
        CHECK(cut_mod.has_value(), "hand-built truncated: the module still parses");
        if (cut_mod) {
            CHECK(cut_mod->symbols.size() == 4, "hand-built truncated: all four symbols still read back");
            CHECK(cut_mod->symbols[1].nid == kHandA && cut_mod->symbols[2].nid == kHandB,
                  "hand-built truncated: both exports still resolve by NID");
            CHECK(cut_mod->imports.size() == 1, "hand-built truncated: the import still resolves");
            LoadedImage img;
            CHECK(!build_image(*cut_mod, kBase1, img, &err),
                  "hand-built truncated: build_image refuses the short data segment");
            CHECK(contains(err, "program header 1"),
                  "hand-built truncated: the refusal names the SHORT segment, not the intact one");
            CHECK(contains(err, "declares 0x1000 bytes at file offset 0x1000") &&
                      contains(err, "the file holds only 0x1800 bytes"),
                  "hand-built truncated: the refusal quotes the declared span and the file length");
        }
        // The control for that arm: the same module uncut maps its data segment's bytes, so the
        // refusal above is about the truncation and not about the second PT_LOAD existing.
        {
            auto ok_mod = Module::load(hand_good, &err);
            CHECK(ok_mod.has_value(), "hand-built control: the uncut module parses");
            if (ok_mod) {
                LoadedImage img;
                CHECK(build_image(*ok_mod, kBase1, img, &err),
                      "hand-built control: the uncut module's image builds");
                const uint8_t* strtab = img.at(kBase1 + prosper_test::kHandStrtabVa + 1);
                CHECK(strtab != nullptr && memcmp(strtab, kHandA.c_str(), kHandA.size()) == 0,
                      "hand-built control: the DATA segment's own bytes reach the image");
            }
        }
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
