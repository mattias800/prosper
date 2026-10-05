// test_va2foff — regression guard for issue #113 (PPSA02664 booted 0 imports).
//
// The bug: Module::va2foff resolves a guest VA to a file offset by scanning
// `loads`, which carries PT_LOAD *and* the non-LOAD file-backed phdrs
// (DYNAMIC/PROCPARAM/TLS). The dynamic tables physically live inside a PT_LOAD,
// but a DYNAMIC phdr's own logical p_offset points at UNMAPPED bytes. When the
// DYNAMIC entry precedes the covering LOAD in program-header order (as in
// PPSA02664, but not PPSA24651), a single-pass scan matched DYNAMIC first and
// returned its garbage self-offset — so the loader parsed 0 dynamic tags / 0
// imports and jumped to garbage. The fix: resolve PT_LOAD first, non-LOAD only
// as a fallback. This test needs no game dump — it is a pure va2foff unit check.
#include "../src/self/module.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

namespace {
Segment seg(uint32_t type, uint64_t vaddr, uint64_t filesz, uint64_t file_off) {
    Segment s; s.type = type; s.vaddr = vaddr; s.filesz = filesz; s.memsz = filesz; s.file_off = file_off;
    return s;
}
// PPSA02664's real numbers: LOAD [0x1aeabb0, +0xb42c0) @ file 0x19bfe60; DYNAMIC vaddr 0x1b9e350 -> correct
// file 0x1a73600. The DYNAMIC phdr's OWN p_offset (0x1a724f0) is the wrong, unmapped answer the
// single-pass scan used to return.
constexpr uint64_t kDynamicVa = 0x1b9e350, kCorrectFileOff = 0x1a73600,
                   kWrongSelfOffset = 0x1a724f0;
constexpr uint64_t kLoadVa = 0x1aeabb0, kLoadFileOff = 0x19bfe60, kLoadFileSz = 0xb42c0;
}  // namespace

TEST(Va2foff, DynamicBeforeLoadResolvesThroughTheLoad) {
    // Reproduce PPSA02664's ordering: DYNAMIC (vaddr inside the LOAD's range) is listed BEFORE the
    // PT_LOAD that actually contains its bytes.
    Module m;
    m.loads.push_back(seg(PT_DYNAMIC, kDynamicVa, 0xb20, kWrongSelfOffset));
    m.loads.push_back(seg(PT_LOAD, kLoadVa, kLoadFileSz, kLoadFileOff));
    EXPECT_EQ(m.va2foff(kDynamicVa), static_cast<int64_t>(kCorrectFileOff))
        << "DYNAMIC-before-LOAD must resolve through the LOAD, not through DYNAMIC's own "
           "self-offset";
}

TEST(Va2foff, LoadBeforeDynamicResolvesTheSameWay) {
    // Order-independence: same result when the LOAD is listed first (PPSA24651's ordering, which
    // worked before the fix and must still work).
    Module m;
    m.loads.push_back(seg(PT_LOAD, kLoadVa, kLoadFileSz, kLoadFileOff));
    m.loads.push_back(seg(PT_DYNAMIC, kDynamicVa, 0xb20, kWrongSelfOffset));
    EXPECT_EQ(m.va2foff(kDynamicVa), static_cast<int64_t>(kCorrectFileOff))
        << "the covering PT_LOAD wins regardless of program-header order";
}

TEST(Va2foff, NonLoadSegmentStillResolvesAVaNoLoadCovers) {
    // Fallback: a VA that no PT_LOAD covers must still resolve via a non-LOAD file-backed segment
    // (e.g. PROCPARAM sitting outside every LOAD).
    Module m;
    m.loads.push_back(seg(PT_LOAD, 0x1000, 0x1000, 0x8000));
    m.loads.push_back(seg(PT_SCE_PROCPARAM, 0x50000, 0x60, 0x40000));
    EXPECT_EQ(m.va2foff(0x50010), 0x40010) << "a non-LOAD phdr is still a fallback, not a dead end";
}

TEST(Va2foff, UnmappedVaReportsMinusOne) {
    Module m;
    m.loads.push_back(seg(PT_LOAD, 0x1000, 0x1000, 0x8000));
    EXPECT_EQ(m.va2foff(0x99999), -1) << "an unmapped VA must report -1, not a 0-offset guess";
}