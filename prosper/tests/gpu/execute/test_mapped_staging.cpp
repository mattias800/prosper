// Actual Vulkan allocations exercise budget pressure and acquisition ownership.
#include "fixtures/render_runner.h"
using namespace prosper::test;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}
int main() {
    constexpr VkDeviceSize mib = 1024 * 1024;
    const auto& ctx = render_vk_ctx();
    if (!ctx.ok || mapped_staging_limit() != 2 * mib) return 1;
    auto& cache = mapped_staging_cache();
    auto acquire = [&](VkDeviceSize bytes, VkBufferUsageFlags usage, bool* reused = nullptr) {
        return acquire_mapped_staging(ctx.dev, bytes, usage, [&](uint32_t bits) {
            return render_memory_type(ctx.phys, bits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }, reused);
    };
    auto release = [&](const MappedStagingBlock& b, bool reuse = true) {
        return release_mapped_staging(b.device, b.buffer, b.memory, b.mapped, b.lease, reuse);
    };
    auto active = acquire(mib / 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto oldest = acquire(mib, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto newer = acquire(mib, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto target = acquire(mib / 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (!active.mapped || !oldest.mapped || !newer.mapped || !target.mapped) return 1;
    std::memset(active.mapped, 0x67, mib / 4);
    release(oldest); release(newer);
    check(cache.cached_bytes == 2 * mib, "two different idle shapes fill the configured budget");
    release(target);
    check(cache.evictions == 1 && !cache.owned.count(oldest.lease) &&
              cache.owned.count(newer.lease) && cache.cached_bytes == 3 * mib / 2,
          "newly completed shape replaces the oldest idle block within budget");
    check(cache.in_use.count(active.lease) &&
              static_cast<const uint8_t*>(active.mapped)[mib / 4 - 1] == 0x67,
          "pressure preserves the active allocation and its bytes");
    const auto stale = cache.stale_releases;
    check(release(oldest, false) && cache.stale_releases == stale + 1,
          "duplicate release after eviction is handled without generic destruction");
    const auto duplicates = cache.double_releases;
    check(release(target, false) && cache.double_releases == duplicates + 1,
          "retained duplicate cannot destroy the idle mapping");
    bool reused = false;
    auto current = acquire(mib / 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &reused);
    check(reused && current.buffer == target.buffer && current.mapped == target.mapped &&
              current.lease && current.lease != target.lease,
          "reacquisition retains raw handles but gets a new ownership identity");
    if (!current.mapped) return 1;
    std::memset(current.mapped, 0x39, mib / 2);
    check(release(target, false) && cache.in_use.count(current.lease) &&
              static_cast<const uint8_t*>(current.mapped)[mib / 2 - 1] == 0x39,
          "stale lease cannot release or overwrite the reacquired allocation");
    check(release_mapped_staging(ctx.dev, current.buffer, current.memory, current.mapped, 0, false) &&
              cache.in_use.count(current.lease), "missing lease cannot free known pool memory");
    const auto wrong_device = cache.cross_device_skips;
    check(release_mapped_staging(VK_NULL_HANDLE, current.buffer, current.memory,
              current.mapped, current.lease, false) &&
              cache.cross_device_skips == wrong_device + 1 && cache.in_use.count(current.lease),
          "wrong device cannot surrender a valid ownership identity");
    const auto before_bytes = cache.cached_bytes;
    const auto before_evictions = cache.evictions;
    auto oversized = acquire(3 * mib, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (!oversized.mapped) return 1;
    check(release(oversized) && cache.cached_bytes == before_bytes &&
              cache.evictions == before_evictions && cache.owned.count(newer.lease),
          "oversized input does not evict useful idle entries");
    check(release(oversized, false), "already destroyed oversized input cannot fall through to free");
    const auto sequence = cache.next_sequence;
    cache.next_sequence = UINT64_MAX;
    auto last = acquire(64 * 1024, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    check(last.mapped && last.lease == UINT64_MAX && cache.next_sequence == 0,
          "the final nonzero identity exhausts the sequence without wrapping to a reused token");
    if (!last.mapped) return 1;
    const auto before_owners = cache.owned.size(), before_live = cache.in_use.size();
    auto exhausted = acquire(mib, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    check(!exhausted.memory && cache.cached_bytes == before_bytes &&
              cache.owned.size() == before_owners && cache.in_use.size() == before_live &&
              cache.owned.count(newer.lease), "exhausted identities fail before changing cache state");
    check(release(last) && !cache.owned.count(last.lease) && cache.cached_bytes == before_bytes,
          "release after identity exhaustion destroys only its own allocation");
    cache.next_sequence = sequence; // test-local seam; production never resets this counter

    // Metadata-only refusal: no Vulkan object is fabricated or destroyed. Foreign
    // occupancy makes retention impossible even after reclaiming every local byte.
    MappedStagingCache foreign;
    foreign.cached_bytes = 2 * mib;
    foreign.free_blocks[{VK_NULL_HANDLE, 3 * mib / 2, 1}].push_back({});
    foreign.free_blocks[{ctx.dev, mib / 2, 1}].push_back({});
    check(!make_mapped_staging_room(foreign, ctx.dev, mib) && foreign.evictions == 0 &&
              foreign.free_blocks.size() == 2 && foreign.cached_bytes == 2 * mib,
          "foreign-device occupancy refuses retention without evicting any device's blocks");
    release(current, false); release(active, false);
    return failures ? 1 : 0;
}
