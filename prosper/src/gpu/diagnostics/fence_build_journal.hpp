#pragma once
// Build-time label observation. Only a complete eight-byte sample is available to lookup;
// unreadable/revoked/partial samples replace old entries without manufacturing a zero value.
// This does not validate, rewrite or complete the guest's packet or its fence.
#include <cstdint>

namespace prosper::gpu {
struct FenceBuildRecord {
    uint64_t pkt = 0, addr = 0, pre = 0, t_ms = 0;
    uint32_t fold = 0;
};
void fence_build_journal_record(uint64_t pkt, uint64_t addr, uint64_t t_ms, uint32_t fold);
// False leaves out untouched, including on collisions and unavailable samples.
bool fence_build_journal_lookup(uint64_t pkt, FenceBuildRecord& out);
}
