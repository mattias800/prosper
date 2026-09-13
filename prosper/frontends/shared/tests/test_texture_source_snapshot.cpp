#include "shared/live/texture_source_snapshot.hpp"

#include <cstdio>

int main()
{
    int failures = 0;
    auto check = [&](bool ok, const char* description) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", description); ++failures; }
    };
    for (bool transfer : {false, true}) {
        std::vector<uint8_t> scratch{1, 2, 3, 4, 0xee, 0xee};
        std::vector<uint8_t> inherited(32, 0xdd);
        const auto* populated = scratch.data();
        const auto* outgoing = inherited.data();
        auto handoff = prosper::frontend::take_texture_source_snapshot(scratch, inherited, 4, transfer);
        check(handoff.transferred == transfer, "eligible capacity honors the transfer control");
        auto snapshot = std::move(handoff.bytes);
        check(snapshot == std::vector<uint8_t>({1, 2, 3, 4}), "short read stores only actual prefix");
        check(snapshot.data() == (transfer ? populated : outgoing), "selected path owns the expected allocation");
        check(scratch.data() == (transfer ? outgoing : populated), "outgoing allocation is reusable scratch on transfer");
        scratch.assign(64, 0xab);
        check(snapshot == std::vector<uint8_t>({1, 2, 3, 4}), "later scratch writes cannot mutate cached snapshot");

        inherited = std::move(snapshot);
        populated = scratch.data();
        snapshot = prosper::frontend::take_texture_source_snapshot(scratch, inherited, 64, transfer).bytes;
        check(snapshot == std::vector<uint8_t>(64, 0xab), "larger replacement keeps all current bytes");
        if (transfer) check(snapshot.data() == populated, "larger replacement needs no snapshot copy");
        inherited = std::move(snapshot);
        scratch.assign(8, 0xfe);
        snapshot = prosper::frontend::take_texture_source_snapshot(scratch, inherited, 0, transfer).bytes;
        check(snapshot.empty(), "failed guest read does not publish scratch or inherited stale bytes");
    }
    for (size_t old_capacity : {size_t{0}, size_t{4}, size_t{1024}}) {
        std::vector<uint8_t> scratch(1024, 0xef);
        std::vector<uint8_t> inherited(old_capacity, 0xdd);
        const size_t allowed_capacity = std::max(size_t{4}, inherited.capacity());
        const auto* populated = scratch.data();
        auto handoff = prosper::frontend::take_texture_source_snapshot(scratch, inherited, 4, true);
        check(handoff.bytes == std::vector<uint8_t>(4, 0xef), "heterogeneous short-prefix handoff stays exact");
        check(handoff.bytes.capacity() <= allowed_capacity, "tiny entry does not absorb oversized global scratch");
        check(handoff.transferred == (old_capacity == 1024), "oversized scratch moves only into an already-large owner");
        if (handoff.transferred)
            check(handoff.bytes.data() == populated, "already-large owner still avoids the redundant copy");
    }
    return failures ? 1 : 0;
}
