#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <version>

// Apple's PMR runtime starts at macOS 14. Keep the ordinary allocator for older
// deployment targets rather than adding an unconditional runtime dependency.
#if defined(__cpp_lib_memory_resource) && \
    (!defined(__APPLE__) || (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && \
                            __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 140000))
#include <memory_resource>
#define PROSPER_PASS_LOOKUP_HAS_PMR 1
#else
#define PROSPER_PASS_LOOKUP_HAS_PMR 0
#endif

namespace prosper::frontend {

struct PassBufferLookupStats {
    uint64_t observed_passes = 0;
    uint64_t arena_passes = 0;
    uint64_t allocations = 0;
    uint64_t allocation_bytes = 0;
    uint64_t deallocations = 0;
    uint64_t deallocation_bytes = 0;
    void add(const PassBufferLookupStats& other) {
        observed_passes += other.observed_passes;
        arena_passes += other.arena_passes;
        allocations += other.allocations;
        allocation_bytes += other.allocation_bytes;
        deallocations += other.deallocations;
        deallocation_bytes += other.deallocation_bytes;
    }
};

// Only CPU lookup nodes/buckets live here, never uploaded bytes or submission
// owners. Declare this before its maps; all maps must die before this object.
// Four KiB is initial storage, not a cap. Rehashing retains old buckets until
// destruction, and larger passes use the ordinary heap. No state crosses calls.
class PassBufferLookupMemory {
public:
    static constexpr bool supported = PROSPER_PASS_LOOKUP_HAS_PMR;
#if PROSPER_PASS_LOOKUP_HAS_PMR
    class Upstream final : public std::pmr::memory_resource {
    public:
        Upstream(PassBufferLookupStats& stats, std::pmr::memory_resource* resource)
            : stats_(stats), resource_(resource) {}
    private:
        void* do_allocate(size_t bytes, size_t alignment) override {
            void* result = resource_->allocate(bytes, alignment);
            ++stats_.allocations;
            stats_.allocation_bytes += bytes;
            return result;
        }
        void do_deallocate(void* p, size_t bytes, size_t alignment) override {
            resource_->deallocate(p, bytes, alignment);
            ++stats_.deallocations;
            stats_.deallocation_bytes += bytes;
        }
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return this == &other;
        }
        PassBufferLookupStats& stats_;
        std::pmr::memory_resource* resource_;
    };
    explicit PassBufferLookupMemory(PassBufferLookupStats& stats, bool arena,
            std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
        : upstream_(stats, upstream), arena_(seed_.data(), seed_.size(), &upstream_),
          selected_(arena ? static_cast<std::pmr::memory_resource*>(&arena_) : &upstream_) {
        ++stats.observed_passes;
        stats.arena_passes += arena;
    }
    template<class Key>
    using Allocator = std::pmr::polymorphic_allocator<std::pair<const Key, size_t>>;
    template<class Key> Allocator<Key> allocator() { return Allocator<Key>{selected_}; }
private:
    alignas(std::max_align_t) std::array<std::byte, 4096> seed_; // intentionally uninitialized
    Upstream upstream_;
    std::pmr::monotonic_buffer_resource arena_;
    std::pmr::memory_resource* selected_;
#else
    explicit PassBufferLookupMemory(PassBufferLookupStats&, bool) {}
    template<class Key> using Allocator = std::allocator<std::pair<const Key, size_t>>;
    template<class Key> Allocator<Key> allocator() { return {}; }
#endif
public:
    PassBufferLookupMemory(const PassBufferLookupMemory&) = delete;
    PassBufferLookupMemory& operator=(const PassBufferLookupMemory&) = delete;
};

template<class Key, class Hash>
using PassBufferLookupMap = std::unordered_map<Key, size_t, Hash, std::equal_to<Key>,
    PassBufferLookupMemory::Allocator<Key>>;

} // namespace prosper::frontend

#undef PROSPER_PASS_LOOKUP_HAS_PMR
