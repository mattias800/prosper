// test_libc_qsort_bsearch -- qsort/bsearch through prosper's registered handlers (hle_libc.cpp
// h_qsort / h_bsearch), with the comparator compiled in the GUEST's calling convention.
//
// The comparator is the part prosper owns. A guest passes a pointer to its own System V function,
// and the handler has to get the host library to call it correctly. On Linux/macOS the host
// convention is the guest's, but on Windows it is Microsoft x64, so a comparator handed to the host
// library unconverted reads its arguments from the wrong registers. Every comparator below is
// therefore declared PROSPER_GUEST_ABI (System V on Windows, nothing elsewhere): that is the only
// way this suite can stand in for a guest, and on the Windows job it is what makes a missing
// conversion fail here rather than inside a game. Comparators own no object with a destructor, per
// the PROSPER_GUEST_ABI rule in dispatch.hpp.
//
// Expectations come from the C standard (N1570 7.22.5): qsort sorts nmemb objects of `size` bytes
// into ascending order as defined by the comparator, whose arguments point at array elements;
// bsearch returns a pointer to a matching element or a null pointer, calls the comparator with the
// key as its FIRST argument, and may return any one of several equal elements. Sorting results are
// checked against std::sort over a copy of the same input, which is an independent oracle.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper;

namespace {

uint64_t U(const void* p) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
}

// The only comparator type this suite passes: the guest's convention, as a guest would supply it.
using GuestCmp = PROSPER_GUEST_ABI int (*)(const void*, const void*);
uint64_t U(GuestCmp f) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(f));
}

HleFn g_qsort = nullptr;
HleFn g_bsearch = nullptr;

// Comparator bookkeeping. Plain globals: the comparators must not own destructible objects.
int g_calls = 0;
int g_out_of_array = 0;   // comparator saw a pointer outside [base, base + n * size)
int g_misaligned = 0;   // ...or one not on an element boundary
const unsigned char* g_base = nullptr;
size_t g_count = 0;
size_t g_size = 0;
const void* g_expected_key = nullptr;
int g_wrong_key_slot = 0;   // bsearch passed something other than the key as argument 1

void note_element(const void* p) {
    if (!g_base) return;
    const auto* b = static_cast<const unsigned char*>(p);
    if (b < g_base || b >= g_base + g_count * g_size) {
        ++g_out_of_array;
        return;
    }
    if ((static_cast<size_t>(b - g_base) % g_size) != 0) ++g_misaligned;
}

void reset_tracking(const void* base, size_t count, size_t size) {
    g_calls = g_out_of_array = g_misaligned = g_wrong_key_slot = 0;
    g_base = static_cast<const unsigned char*>(base);
    g_count = count;
    g_size = size;
    g_expected_key = nullptr;
}

PROSPER_GUEST_ABI int cmp_int_asc(const void* a, const void* b) {
    ++g_calls;
    note_element(a);
    note_element(b);
    const int x = *static_cast<const int*>(a), y = *static_cast<const int*>(b);
    return (x > y) - (x < y);
}

PROSPER_GUEST_ABI int cmp_int_desc(const void* a, const void* b) {
    ++g_calls;
    const int x = *static_cast<const int*>(a), y = *static_cast<const int*>(b);
    return (y > x) - (y < x);
}

// A 24-byte record whose payload must travel with its key.
struct Record {
    int key;
    char tag[12];
    uint64_t check;   // = key * 7919, so a torn swap is visible
};
static_assert(sizeof(Record) == 24, "test assumes a 24-byte record");

PROSPER_GUEST_ABI int cmp_record(const void* a, const void* b) {
    ++g_calls;
    note_element(a);
    note_element(b);
    const int x = static_cast<const Record*>(a)->key, y = static_cast<const Record*>(b)->key;
    return (x > y) - (x < y);
}

// An odd element size (3 bytes): sort by the big-endian 24-bit value.
PROSPER_GUEST_ABI int cmp_u24(const void* a, const void* b) {
    ++g_calls;
    note_element(a);
    note_element(b);
    const auto* p = static_cast<const unsigned char*>(a);
    const auto* q = static_cast<const unsigned char*>(b);
    const uint32_t x = (uint32_t(p[0]) << 16) | (uint32_t(p[1]) << 8) | p[2];
    const uint32_t y = (uint32_t(q[0]) << 16) | (uint32_t(q[1]) << 8) | q[2];
    return (x > y) - (x < y);
}

// bsearch key comparator: the key is an int, the elements are Records.
PROSPER_GUEST_ABI int cmp_key_record(const void* key, const void* elem) {
    ++g_calls;
    if (key != g_expected_key) ++g_wrong_key_slot;
    note_element(elem);
    const int k = *static_cast<const int*>(key), e = static_cast<const Record*>(elem)->key;
    return (k > e) - (k < e);
}

// Nested use: a qsort comparator that performs a bsearch with a DIFFERENT comparator before
// answering. A guest may legitimately do this, and the handlers must not lose track of which
// comparator belongs to which call.
int g_inner_hits = 0;
PROSPER_GUEST_ABI int cmp_int_desc_plain(const void* a, const void* b) {
    const int x = *static_cast<const int*>(a), y = *static_cast<const int*>(b);
    return (y > x) - (y < x);
}
const int kInnerTableDesc[] = {10, 8, 6, 4, 2};
PROSPER_GUEST_ABI int cmp_int_asc_nesting(const void* a, const void* b) {
    ++g_calls;
    const int probe = 6;
    // Descending table + descending comparator: only the inner comparator finds 6 here.
    if (g_bsearch(U(&probe), U(kInnerTableDesc), 5, sizeof(int), U(&cmp_int_desc_plain), 0) != 0)
        ++g_inner_hits;
    const int x = *static_cast<const int*>(a), y = *static_cast<const int*>(b);
    return (x > y) - (y > x);
}

class LibcQsortBsearch : public ::testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        g_qsort = Hle::lookup(nid_hash("qsort"));
        g_bsearch = Hle::lookup(nid_hash("bsearch"));
        ASSERT_NE(g_qsort, nullptr) << "qsort has no host-ABI handler registered";
        ASSERT_NE(g_bsearch, nullptr) << "bsearch has no host-ABI handler registered";
    }
    static void qsort_(void* base, size_t n, size_t size, GuestCmp cmp) {
        reset_tracking(base, n, size);
        g_qsort(U(base), n, size, U(cmp), 0, 0);
    }
    static const void* bsearch_(const void* key, const void* base, size_t n, size_t size,
                                GuestCmp cmp) {
        reset_tracking(base, n, size);
        g_expected_key = key;
        return reinterpret_cast<const void*>(
            static_cast<uintptr_t>(g_bsearch(U(key), U(base), n, size, U(cmp), 0)));
    }
};

// Deterministic input shapes: random, sorted, reversed, constant, sawtooth, few distinct values.
std::vector<int> shape(int which, size_t n) {
    std::vector<int> v(n);
    uint32_t s = 0x9e3779b9u ^ static_cast<uint32_t>(n * 131 + which);
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        switch (which) {
            case 0: v[i] = static_cast<int>(s >> 1) - (1 << 30); break;
            case 1: v[i] = static_cast<int>(i); break;
            case 2: v[i] = static_cast<int>(n - i); break;
            case 3: v[i] = 42; break;
            case 4: v[i] = static_cast<int>(i % 7); break;
            default: v[i] = static_cast<int>((s >> 16) % 3) - 1; break;
        }
    }
    return v;
}

TEST_F(LibcQsortBsearch, SortsIntsAscendingAcrossShapesAndSizes) {
    for (int which = 0; which < 6; ++which) {
        for (size_t n : {2u, 3u, 7u, 16u, 31u, 100u, 257u, 1000u}) {
            std::vector<int> v = shape(which, n), want = v;
            std::sort(want.begin(), want.end());
            qsort_(v.data(), n, sizeof(int), &cmp_int_asc);
            SCOPED_TRACE("shape=" + std::to_string(which) + " n=" + std::to_string(n));
            EXPECT_EQ(v, want);
            EXPECT_GT(g_calls, 0) << "the guest comparator was never called";
            EXPECT_EQ(g_out_of_array, 0) << "comparator received a pointer outside the array";
            EXPECT_EQ(g_misaligned, 0) << "comparator received a pointer between elements";
        }
    }
}

TEST_F(LibcQsortBsearch, OrderIsTheComparatorsNotTheHosts) {
    std::vector<int> v = shape(0, 200), want = v;
    std::sort(want.begin(), want.end(), [](int a, int b) { return a > b; });
    qsort_(v.data(), v.size(), sizeof(int), &cmp_int_desc);
    EXPECT_EQ(v, want);
}

TEST_F(LibcQsortBsearch, ZeroOrOneElementIsLeftAloneWithoutComparing) {
    int one = 7;
    qsort_(&one, 1, sizeof(int), &cmp_int_asc);
    EXPECT_EQ(one, 7);
    EXPECT_EQ(g_calls, 0);
    int none[2] = {9, 1};
    qsort_(none, 0, sizeof(int), &cmp_int_asc);
    EXPECT_EQ(none[0], 9);
    EXPECT_EQ(none[1], 1);
    EXPECT_EQ(g_calls, 0);
}

TEST_F(LibcQsortBsearch, MovesWholeRecordsOfANonWordSize) {
    std::vector<Record> v(97);
    std::vector<int> keys = shape(0, v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        v[i].key = keys[i] % 1000;
        snprintf(v[i].tag, sizeof v[i].tag, "r%d", v[i].key);
        v[i].check = static_cast<uint64_t>(static_cast<int64_t>(v[i].key)) * 7919u;
    }
    qsort_(v.data(), v.size(), sizeof(Record), &cmp_record);
    EXPECT_EQ(g_out_of_array, 0);
    EXPECT_EQ(g_misaligned, 0);
    for (size_t i = 0; i < v.size(); ++i) {
        char want_tag[12];
        snprintf(want_tag, sizeof want_tag, "r%d", v[i].key);
        EXPECT_STREQ(v[i].tag, want_tag) << "record " << i << " was torn";
        EXPECT_EQ(v[i].check, static_cast<uint64_t>(static_cast<int64_t>(v[i].key)) * 7919u);
        if (i) EXPECT_LE(v[i - 1].key, v[i].key) << "records out of order at " << i;
    }
}

TEST_F(LibcQsortBsearch, SortsThreeByteElements) {
    const size_t n = 50;
    std::vector<unsigned char> v(n * 3);
    std::vector<int> src = shape(0, n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t x = static_cast<uint32_t>(src[i]) & 0xffffffu;
        v[i * 3] = static_cast<unsigned char>(x >> 16);
        v[i * 3 + 1] = static_cast<unsigned char>(x >> 8);
        v[i * 3 + 2] = static_cast<unsigned char>(x);
    }
    std::vector<uint32_t> want(n);
    for (size_t i = 0; i < n; ++i) want[i] = static_cast<uint32_t>(src[i]) & 0xffffffu;
    std::sort(want.begin(), want.end());
    qsort_(v.data(), n, 3, &cmp_u24);
    EXPECT_EQ(g_misaligned, 0);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t got =
            (uint32_t(v[i * 3]) << 16) | (uint32_t(v[i * 3 + 1]) << 8) | v[i * 3 + 2];
        EXPECT_EQ(got, want[i]) << "element " << i;
    }
}

TEST_F(LibcQsortBsearch, BsearchFindsEveryElementAtItsOwnAddress) {
    std::vector<Record> table(40);
    for (size_t i = 0; i < table.size(); ++i) table[i].key = static_cast<int>(i * 3 + 1);
    for (size_t i = 0; i < table.size(); ++i) {
        const int key = table[i].key;
        const void* hit =
            bsearch_(&key, table.data(), table.size(), sizeof(Record), &cmp_key_record);
        EXPECT_EQ(hit, &table[i]) << "key " << key;
        EXPECT_EQ(g_wrong_key_slot, 0) << "comparator's first argument was not the key";
        EXPECT_EQ(g_out_of_array, 0);
        EXPECT_EQ(g_misaligned, 0);
    }
}

TEST_F(LibcQsortBsearch, BsearchMissesReturnNull) {
    // keys are 1, 4, 7, ..., 118 (i * 3 + 1)
    std::vector<Record> table(40);
    for (size_t i = 0; i < table.size(); ++i) table[i].key = static_cast<int>(i * 3 + 1);
    for (int key : {-5, 0, 2, 3, 59, 60, 119, 120, 1000}) {
        EXPECT_EQ(bsearch_(&key, table.data(), table.size(), sizeof(Record), &cmp_key_record),
                  nullptr)
            << "key " << key;
        EXPECT_GT(g_calls, 0);
    }
}

TEST_F(LibcQsortBsearch, BsearchOverAnEmptyTableNeverCompares) {
    Record table[1] = {};
    const int key = 0;
    EXPECT_EQ(bsearch_(&key, table, 0, sizeof(Record), &cmp_key_record), nullptr);
    EXPECT_EQ(g_calls, 0);
}

TEST_F(LibcQsortBsearch, BsearchOverDuplicatesReturnsAnEqualElement) {
    const int table[] = {1, 3, 3, 3, 3, 3, 3, 3, 9};
    const int key = 3;
    const void* hit = bsearch_(&key, table, 9, sizeof(int), &cmp_int_asc);
    ASSERT_NE(hit, nullptr);
    EXPECT_EQ(*static_cast<const int*>(hit), 3);
    const auto idx = static_cast<const int*>(hit) - table;
    EXPECT_GE(idx, 1);
    EXPECT_LE(idx, 7);
}

TEST_F(LibcQsortBsearch, ComparatorMayItselfCallBsearch) {
    g_inner_hits = 0;
    std::vector<int> v = shape(0, 64), want = v;
    std::sort(want.begin(), want.end());
    qsort_(v.data(), v.size(), sizeof(int), &cmp_int_asc_nesting);
    EXPECT_EQ(v, want) << "the outer sort lost its comparator to the nested bsearch";
    EXPECT_GT(g_inner_hits, 0);
    EXPECT_EQ(g_inner_hits, g_calls) << "a nested bsearch used the wrong comparator";
}

}   // namespace
