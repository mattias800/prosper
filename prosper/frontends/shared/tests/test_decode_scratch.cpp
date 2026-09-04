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
using prosper::frontend::ScratchBuffer;
using prosper::frontend::decode_scratch_budget_bytes;
using prosper::frontend::decode_scratch_pool;

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

    // --- the poison arm actually poisons -----------------------------------------------------
    //
    // The whole equivalence claim for the `make_unique_for_overwrite` call sites is "indeterminate
    // bytes stayed indeterminate". PROSPER_DECODE_SCRATCH_POISON exists to make a call site that
    // secretly relied on the old allocation's incidental zeroing fail loudly, and a poison arm that
    // does not poison would launder exactly that. Assert it in whichever direction this run is.
    {
        DecodeScratchPool pool(64u << 20);
        auto lease = pool.take(4096);
        if (prosper::frontend::decode_scratch_poison_enabled()) {
            bool all_poison = true;
            for (size_t i = 0; i < lease.size(); ++i) all_poison &= lease.data()[i] == 0xCD;
            CHECK(all_poison);
        } else {
            // A pristine pool's first buffer is value-initialised, so the default arm can state the
            // complement rather than merely skipping.
            bool any_poison = false;
            for (size_t i = 0; i < lease.size(); ++i) any_poison |= lease.data()[i] == 0xCD;
            CHECK(!any_poison);
        }
    }

    // --- ScratchBuffer: the adapter live_compute.cpp's per-dispatch intermediates use ------------
    //
    // Asserted against the THREAD pool the production call sites use, not a local one. An adapter
    // that quietly kept allocating would pass every local-pool assertion above and change nothing
    // in production, so the discriminator has to be the shared pool's own retention counter.
    {
        const bool retaining =
            decode_scratch_budget_bytes(std::getenv("PROSPER_DECODE_SCRATCH_MB")) != 0;
        // The poison arm deliberately overwrites the previous tenant, so the "same pages still hold
        // what I wrote" half of this only means anything when poison is off. Reuse of the pointer
        // itself is asserted in both arms.
        const bool tenant_survives = retaining && !prosper::frontend::decode_scratch_poison_enabled();
        const size_t n = 1u << 20;
        const size_t retained_before = decode_scratch_pool().retained_buffers();
        uint8_t* first = nullptr;
        {
            ScratchBuffer scratch;
            scratch.reset(n, /*zero_fill=*/false);
            CHECK(static_cast<bool>(scratch));
            first = scratch.get();
            CHECK(first != nullptr);
            std::memset(first, 0x3C, n);
        }
        if (retaining) {
            // The lease went back to the shared pool: that is the one observable fact separating
            // "ScratchBuffer uses the pool" from "ScratchBuffer allocates and frees".
            CHECK(decode_scratch_pool().retained_buffers() > retained_before);
            ScratchBuffer scratch;
            scratch.reset(n, /*zero_fill=*/false);
            CHECK(scratch.get() == first);        // same pages: no mmap, no fault-in
            if (tenant_survives)
                CHECK(scratch.get()[0] == 0x3C);  // ...holding the previous tenant, which is
                                                  //    precisely why zero_fill exists
        } else {
            CHECK(decode_scratch_pool().retained_buffers() == 0);
        }
    }

    // --- ScratchBuffer zero_fill is load-bearing, in both directions -----------------------------
    {
        const bool tenant_survives =
            decode_scratch_budget_bytes(std::getenv("PROSPER_DECODE_SCRATCH_MB")) != 0 &&
            !prosper::frontend::decode_scratch_poison_enabled();
        const size_t n = 1u << 20;
        { ScratchBuffer dirty; dirty.reset(n, false); std::memset(dirty.get(), 0xA5, n); }
        {
            ScratchBuffer scratch;
            scratch.reset(n, /*zero_fill=*/true);
            bool all_zero = true;
            for (size_t i = 0; i < n; ++i) all_zero &= scratch.get()[i] == 0;
            CHECK(all_zero);
        }
        { ScratchBuffer dirty; dirty.reset(n, false); std::memset(dirty.get(), 0xA5, n); }
        {
            ScratchBuffer scratch;
            scratch.reset(n, /*zero_fill=*/false);
            // MUTATION ARM: without the flag the previous tenant survives. If this ever reads zero
            // the flag has stopped discriminating and the zero_fill=true assertion above is vacuous.
            if (tenant_survives) CHECK(scratch.get()[0] == 0xA5);
            // ...and under the poison arm the complement: the lease must NOT arrive zeroed either,
            // or `zero_fill=false` would be indistinguishable from `zero_fill=true` there too.
            if (prosper::frontend::decode_scratch_poison_enabled())
                CHECK(scratch.get()[0] == 0xCD);
        }
    }

    // --- a zero-extent reset leases nothing and stays null ---------------------------------------
    //
    // The call sites guard on a non-zero extent, but `take(0)` would still claim a retained buffer
    // and hand it straight back, so state the contract here rather than relying on the guards.
    {
        const size_t retained_before = decode_scratch_pool().retained_buffers();
        ScratchBuffer scratch;
        scratch.reset(0, /*zero_fill=*/true);
        CHECK(scratch.get() == nullptr);
        CHECK(!static_cast<bool>(scratch));
        CHECK(decode_scratch_pool().retained_buffers() == retained_before);
    }

    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("decode scratch pool: all checks passed\n");
    return 0;
}
