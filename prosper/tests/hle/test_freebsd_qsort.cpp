// test_freebsd_qsort — FreeBSD libc contract for qsort, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/stdlib/qsort_test.c (ATF) plus the
// shared fixtures lib/libc/tests/stdlib/test-sort.h (sorthelp comparator,
// ssort reference, initvector data).
// Original author: Maxim Sobolev / FreeBSD.
//
// Why this pins prosper: h_qsort forwards to the host qsort with a guest
// comparator. Guest code sorts save entries, trophy lists and shader keys;
// the contract is exact output equality with a reference sort over MANY
// sizes (the ATF original grows the table 2..N), plus sortedness and
// permutation on adversarial inputs (already sorted, reversed, all equal).
//
// The full 1024-element initvector is not vendored: a compact xorshift
// generates the same shapes (mixed signs, duplicates, extremes) without a
// 200-line data table. Sizes 2..256 keep the reference O(n^2) check fast.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

int cmp_int(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    // Never `return ia - ib`: it overflows (noted in test-sort.h itself).
    if (ia > ib) return 1;
    if (ia < ib) return -1;
    return 0;
}

// ssort from test-sort.h: the slow reference every qsort result is compared to.
void ref_sort(int* v, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (v[j] < v[i]) std::swap(v[i], v[j]);
}

uint32_t xorshift(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

bool is_sorted(const int* v, size_t n) {
    for (size_t i = 1; i < n; ++i)
        if (v[i] < v[i - 1]) return false;
    return true;
}

}  // namespace

TEST(FreebsdQsort, MatchesReferenceAllSizes) {
    // Port of the ATF loop: for each size, qsort and the reference must agree
    // element-wise. Deterministic pseudo-random data incl. extremes.
    for (size_t n = 2; n <= 256; ++n) {
        std::vector<int> a(n), b(n);
        uint32_t s = 0x12345678u;
        for (size_t i = 0; i < n; ++i) {
            uint32_t r = xorshift(s);
            int v = (int)(r ^ (r >> 3));
            if (i % 17 == 0) v = INT32_MIN + (int)i;  // extremes in the mix
            if (i % 13 == 0) v = INT32_MAX - (int)i;
            if (i % 7 == 0) v = 42;  // duplicates in the mix
            a[i] = b[i] = v;
        }
        qsort(a.data(), n, sizeof(int), cmp_int);
        ref_sort(b.data(), n);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_EQ(a[i], b[i]) << "size " << n << " index " << i;
            if (a[i] != b[i]) break;  // one report per size, not n
        }
    }
}

TEST(FreebsdQsort, AdversarialShapes) {
    for (size_t n : {1u, 2u, 16u, 128u}) {
        std::vector<int> v(n);
        // Already sorted.
        for (size_t i = 0; i < n; ++i) v[i] = (int)i;
        qsort(v.data(), n, sizeof(int), cmp_int);
        EXPECT_TRUE(is_sorted(v.data(), n)) << "sorted stays sorted, n=" << n;
        // Reversed.
        for (size_t i = 0; i < n; ++i) v[i] = (int)(n - i);
        qsort(v.data(), n, sizeof(int), cmp_int);
        EXPECT_TRUE(is_sorted(v.data(), n)) << "reversed sorts, n=" << n;
        // All equal.
        std::fill(v.begin(), v.end(), -7);
        qsort(v.data(), n, sizeof(int), cmp_int);
        EXPECT_TRUE(is_sorted(v.data(), n)) << "constant sorts, n=" << n;
        for (size_t i = 0; i < n; ++i) EXPECT_EQ(v[i], -7);
    }
}

TEST(FreebsdQsort, IsPermutation) {
    // Sorting must reorder, never invent or drop elements: multiset equality
    // against the reference on a duplicate-heavy input.
    std::vector<int> a(200), b(200);
    uint32_t s = 0xdeadbeefu;
    for (auto& x : a) x = (int)(xorshift(s) % 11) - 5;  // only 11 distinct values
    b = a;
    qsort(a.data(), a.size(), sizeof(int), cmp_int);
    ref_sort(b.data(), b.size());
    EXPECT_EQ(a, b) << "duplicate-heavy permutation matches reference";
}

TEST(FreebsdQsort, PositiveControl) {
    // Hand-built: the comparator order must be visible, else agreement with
    // a reference is vacuous (e.g. a no-op sort of an already-sorted table).
    int v[] = {3, 1, 2};
    qsort(v, 3, sizeof(int), cmp_int);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
}
