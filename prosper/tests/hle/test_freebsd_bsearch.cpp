// test_freebsd_bsearch — FreeBSD libc contract for bsearch, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/stdlib/bsearch_test.c (ATF) plus
// the shared oracle lib/libc/tests/stdlib/test-search.h.
// Original author: Faraz Vahedi / FreeBSD.
//
// Why this pins prosper: h_bsearch forwards to the host bsearch with a guest
// comparator. The contract arms here are what a guest binary search depends
// on: every element of a sorted array is found AT its own address, misses
// return NULL, an empty table never calls the comparator, and duplicates
// resolve inside the duplicate run.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>

namespace {

int cmp_int(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

int cmp_never_called(const void* /*a*/, const void* /*b*/) {
    ADD_FAILURE() << "comparator invoked on an empty table";
    return 0;
}

// check_sorted_search from test-search.h: fill v[i] = i, every element must
// be found at its own address, -1 and n must miss.
void check_sorted_search(int* v, size_t n) {
    for (size_t i = 0; i < n; ++i) v[i] = (int)i;
    for (size_t i = 0; i < n; ++i) {
        int key = v[i];
        void* found = bsearch(&key, v, n, sizeof(int), cmp_int);
        EXPECT_EQ(found, &v[i]) << "element " << i << " of " << n;
    }
    if (n != 0) {
        int key = -1;
        EXPECT_EQ(bsearch(&key, v, n, sizeof(int), cmp_int), nullptr) << "below range misses";
        key = (int)n;
        EXPECT_EQ(bsearch(&key, v, n, sizeof(int), cmp_int), nullptr) << "above range misses";
    }
}

}  // namespace

TEST(FreebsdBsearch, SortedTableAllSizes) {
    int v[64];
    for (size_t n = 1; n <= 64; ++n) check_sorted_search(v, n);
}

TEST(FreebsdBsearch, EmptyTableNeverCompares) {
    int v[1] = {0};
    int key = 0;
    EXPECT_EQ(bsearch(&key, v, 0, sizeof(int), cmp_never_called), nullptr);
}

TEST(FreebsdBsearch, DuplicatesResolveInsideRun) {
    int d[] = {1, 2, 2, 2, 3};
    int e[] = {7, 7, 7};
    int key = 2;
    int* found = (int*)bsearch(&key, d, 5, sizeof(int), cmp_int);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(*found, 2);
    EXPECT_GE(found, &d[1]) << "duplicate resolves inside the run";
    EXPECT_LE(found, &d[3]);
    key = 7;
    found = (int*)bsearch(&key, e, 3, sizeof(int), cmp_int);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(*found, 7);
    EXPECT_GE(found, &e[0]);
    EXPECT_LE(found, &e[2]);
}

TEST(FreebsdBsearch, PositiveControl) {
    int v[] = {1, 3, 5};
    int key = 4;
    EXPECT_EQ(bsearch(&key, v, 3, sizeof(int), cmp_int), nullptr) << "harness sees a miss";
    key = 3;
    EXPECT_NE(bsearch(&key, v, 3, sizeof(int), cmp_int), nullptr) << "harness sees a hit";
}
