// The pooled texture-decode scratch buffers, and the ONE property their speed depends on being
// safe: that a lease reused from the pool is byte-for-byte what a fresh value-initialised
// std::vector would have been, once the caller declares how much of it it filled.
//
// The equivalence arm below is written as a differential: the same partial fill is applied to a
// pooled lease and to a `std::vector<uint8_t>(n, 0)`, and the two are compared. It is deliberately
// run after the pool has served a DIFFERENT, larger surface, because that is the only state in
// which the fast path can differ — a pristine pool hands out zeroed pages and would pass whatever
// the tail contract said.
#include "shared/live/decode_scratch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using prosper::frontend::DecodeScratchPool;
using prosper::frontend::decode_scratch_budget_bytes;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

// The shape every converted call site has: stage `filled` bytes into a buffer of `n`, and leave the
// rest zero. Returns what the decoder would then read.
static std::vector<uint8_t> reference_fill(size_t n, size_t filled, uint8_t value) {
    std::vector<uint8_t> buffer(n, 0);
    for (size_t i = 0; i < filled && i < n; ++i) buffer[i] = value;
    return buffer;
}

int main() {
    // --- retention: a returned lease keeps its pages, a re-take of the same size reuses them -----
    {
        DecodeScratchPool pool(64u << 20);
        uint8_t* first = nullptr;
        {
            auto lease = pool.take(1u << 20);
            CHECK(lease.size() == (1u << 20));
            first = lease.data();
        }
        CHECK(pool.retained_buffers() == 1);
        CHECK(pool.retained_bytes() >= (1u << 20));
        {
            auto lease = pool.take(1u << 20);
            CHECK(lease.data() == first);   // same allocation, no fault-in
        }
    }

    // --- a smaller request never shrinks the buffer, and reports the requested extent ------------
    {
        DecodeScratchPool pool(64u << 20);
        uint8_t* big = nullptr;
        { auto lease = pool.take(4u << 20); big = lease.data(); }
        {
            auto lease = pool.take(1u << 20);
            CHECK(lease.data() == big);
            CHECK(lease.size() == (1u << 20));   // the extent is what was asked for...
        }
        CHECK(pool.retained_bytes() >= (4u << 20));   // ...while the mapping is kept whole
    }

    // --- EQUIVALENCE: a dirty reused lease + zero_tail == a fresh value-initialised vector -------
    {
        DecodeScratchPool pool(64u << 20);
        // Dirty the pool with a previous, larger surface so the reused buffer is NOT zero.
        { auto dirty = pool.take(8u << 20); std::memset(dirty.data(), 0xA5, dirty.size()); }

        const size_t n = 4u << 20;
        for (size_t filled : {size_t{0}, size_t{1}, n / 3, n - 1, n}) {
            auto lease = pool.take(n);
            CHECK(lease.size() == n);
            std::memset(lease.data(), 0x5C, filled);
            lease.zero_tail(filled);
            const std::vector<uint8_t> expected = reference_fill(n, filled, 0x5C);
            CHECK(std::memcmp(lease.data(), expected.data(), n) == 0);
        }
    }

    // --- zero_all is zero_tail(0), and clears a dirty reused buffer completely -------------------
    {
        DecodeScratchPool pool(64u << 20);
        { auto dirty = pool.take(1u << 20); std::memset(dirty.data(), 0xFF, dirty.size()); }
        auto lease = pool.take(1u << 20);
        lease.zero_all();
        const std::vector<uint8_t> zeros(1u << 20, 0);
        CHECK(std::memcmp(lease.data(), zeros.data(), zeros.size()) == 0);
    }

    // --- nesting: two live leases are distinct buffers (traw + hlin are alive together) ----------
    {
        DecodeScratchPool pool(64u << 20);
        auto a = pool.take(1u << 20);
        auto b = pool.take(1u << 20);
        CHECK(a.data() != b.data());
        std::memset(a.data(), 0x11, a.size());
        std::memset(b.data(), 0x22, b.size());
        CHECK(a.data()[0] == 0x11 && b.data()[0] == 0x22);
    }

    // --- moving a lease transfers ownership and returns the buffer exactly once ------------------
    {
        DecodeScratchPool pool(64u << 20);
        {
            auto a = pool.take(1u << 20);
            auto b = std::move(a);
            CHECK(b.size() == (1u << 20));
            CHECK(a.size() == 0);
        }
        CHECK(pool.retained_buffers() == 1);
    }

    // --- the budget bounds retention, and drops the SMALLEST buffers first -----------------------
    //
    // The leases are held SIMULTANEOUSLY on purpose. Taken one at a time the pool would hand the
    // same buffer back each time and never hold more than one, so the eviction path this asserts
    // would never run and the test would pass without exercising anything.
    {
        DecodeScratchPool pool(5u << 20, /*max_retained=*/8);
        {
            auto a = pool.take(4u << 20);
            auto b = pool.take(1u << 20);
            auto c = pool.take(1u << 20);
            CHECK(a.data() != b.data() && b.data() != c.data() && a.data() != c.data());
        }
        // 6 MiB returned against a 5 MiB budget: one 1 MiB buffer goes, the 4 MiB one stays.
        CHECK(pool.retained_buffers() == 2);
        CHECK(pool.retained_bytes() == (5u << 20));
        auto big = pool.take(4u << 20);
        CHECK(big.size() == (4u << 20));
        CHECK(pool.retained_bytes() == (1u << 20));   // the survivor really was the big one
    }

    // --- max_retained bounds the COUNT independently of the byte budget --------------------------
    {
        DecodeScratchPool pool(1024u << 20, /*max_retained=*/2);
        {
            auto a = pool.take(1024);
            auto b = pool.take(2048);
            auto c = pool.take(4096);
            (void)a.size(); (void)b.size(); (void)c.size();
        }
        CHECK(pool.retained_buffers() == 2);
        CHECK(pool.retained_bytes() == 2048 + 4096);   // the smallest was the one dropped
    }

    // --- retention disabled: every lease is a fresh allocation, i.e. the pre-pool behaviour ------
    {
        DecodeScratchPool pool(0);
        { auto a = pool.take(1u << 20); (void)a.size(); }
        CHECK(pool.retained_buffers() == 0);
        CHECK(pool.retained_bytes() == 0);
    }

    // --- the knob, and WHICH WAY a typo fails ----------------------------------------------------
    //
    // This block is the record of a claim that was made here and was false. The header used to say
    // a mistyped value "selects the SAFE sentinel"; these assertions say otherwise and always did.
    // A malformed value keeps the 512 MiB DEFAULT, which leaves pooling fully ON -- and since this
    // knob's main use is disarming the pool for an A/B, the dangerous typo is one meant to turn the
    // optimisation off. Anyone restating the old claim has to delete a passing assertion to do it.
    CHECK(decode_scratch_budget_bytes(nullptr) == (512ull << 20));
    CHECK(decode_scratch_budget_bytes("") == (512ull << 20));
    CHECK(decode_scratch_budget_bytes("64") == (64ull << 20));
    // Only the exact spelling disarms the pool...
    CHECK(decode_scratch_budget_bytes("0") == 0);
    // ...and every near-miss of it does NOT. These are the spellings an operator actually types.
    CHECK(decode_scratch_budget_bytes("0mb") == (512ull << 20));
    CHECK(decode_scratch_budget_bytes("0MB") == (512ull << 20));
    CHECK(decode_scratch_budget_bytes(" 0") == (512ull << 20));
    CHECK(decode_scratch_budget_bytes("64mb") == (512ull << 20));
    CHECK(decode_scratch_budget_bytes(" 64") == (512ull << 20));

    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("decode scratch pool: all checks passed\n");
    return 0;
}
