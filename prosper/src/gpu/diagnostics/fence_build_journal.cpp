#include "gpu/diagnostics/fence_build_journal.hpp"
#include "host/memory/guest_memory_copy.hpp"
#include "host/platform/immortal.hpp"
#include <array>
#include <mutex>

namespace prosper::gpu {
namespace {
constexpr uint32_t journal_size = 65536;
struct Slot {
    std::mutex mutex;
    FenceBuildRecord record;
    bool available = false;
};
using Journal = std::array<Slot, journal_size>;
Journal& journal() {
    // Guest builders can still run during host static destruction. The bounded observer
    // needs no destructor side effects and must retain both its records and locks.
    static Immortal<Journal> storage;
    static_assert(std::is_trivially_destructible_v<decltype(storage)>);
    return *storage;
}
Slot& slot_for(uint64_t pkt) {
    return journal()[static_cast<uint32_t>((pkt >> 2) * 2654435761u) & (journal_size - 1)];
}
}   // namespace
void fence_build_journal_record(uint64_t pkt, uint64_t addr, uint64_t t_ms, uint32_t fold) {
    if (!pkt) return;
    Slot& slot = slot_for(pkt);
    // Lock only this direct-mapped slot: concurrent packet builders cannot publish a torn
    // record. Invalidate before sampling so a failed replacement never exposes an old pre.
    std::lock_guard lock(slot.mutex);
    slot.available = false;
    slot.record = {pkt, addr, 0, t_ms, fold};
    uint64_t pre = 0;
    if (addr < 0x10000 || (addr & 3) || !host::guest_read_exact(addr, &pre, sizeof pre)) return;
    slot.record.pre = pre;
    slot.available = true;
}
bool fence_build_journal_lookup(uint64_t pkt, FenceBuildRecord& out) {
    if (!pkt) return false;
    Slot& slot = slot_for(pkt);
    std::lock_guard lock(slot.mutex);
    if (!slot.available || slot.record.pkt != pkt) return false;
    out = slot.record;
    return true;
}
}   // namespace prosper::gpu
