#pragma once
// unchanged_publication_census.hpp -- how often does a compute result published to the renderer carry
// exactly the bytes it carried last time?
//
// WHY. Black Flag runs at 3-4 fps on its static health-warning page, with the compute display chain
// republishing a full 1920x1080 target every submit. If most of those publications are byte-identical to
// the previous one for the same target, the avoidable work is the whole republish and everything the
// renderer does with it, which is far larger than any loop inside it. Nothing measured that fraction.
//
// WHAT. For each publication (target address, extent, format, bytes) the census hashes the bytes and
// compares the hash with the last one seen for that target. It counts publications and bytes, and how many
// of each were identical. It is a pure class: the F8-window gate and the hook live in
// unchanged_publication_hook.hpp, so this file has no dependency on the live renderer.
//
// The hash is 64 bits over every byte (four independent lanes, so it runs at memory speed): a collision
// would report a changed picture as unchanged, which at 2^-64 per comparison is not a concern for a census
// and would only ever overstate the avoidable fraction by a publication.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace prosper::perf {

// 64-bit hash over `bytes` bytes. Every input byte influences the result (tested at the first, last and
// every lane boundary), and the length is folded in so a zero-padded buffer does not collide with its prefix.
inline uint64_t hash_published_bytes(const uint8_t* data, size_t bytes) {
    constexpr uint64_t kMul = 0x9E3779B97F4A7C15ULL;
    uint64_t lane[4] = {0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL, 0xA4093822299F31D0ULL,
                        0x082EFA98EC4E6C89ULL};
    size_t i = 0;
    for (; i + 32 <= bytes; i += 32) {
        for (int l = 0; l < 4; ++l) {
            uint64_t v;
            std::memcpy(&v, data + i + static_cast<size_t>(l) * 8, sizeof(v));
            lane[l] = (lane[l] ^ v) * kMul;
            lane[l] ^= lane[l] >> 29;
        }
    }
    // The 0..31 byte tail, zero-padded into one more block so every tail byte participates.
    if (i < bytes) {
        uint8_t block[32] = {};
        std::memcpy(block, data + i, bytes - i);
        for (int l = 0; l < 4; ++l) {
            uint64_t v;
            std::memcpy(&v, block + l * 8, sizeof(v));
            lane[l] = (lane[l] ^ v) * kMul;
            lane[l] ^= lane[l] >> 29;
        }
    }
    uint64_t h = static_cast<uint64_t>(bytes) * kMul;
    for (int l = 0; l < 4; ++l) {
        h = (h ^ lane[l]) * kMul;
        h ^= h >> 32;
    }
    return h;
}

class UnchangedPublicationCensus {
public:
    struct Totals {
        uint64_t publications = 0;
        uint64_t identical = 0;
        uint64_t bytes = 0;
        uint64_t identical_bytes = 0;
    };

    struct TargetRow {
        uint64_t address = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t format = 0;
        uint64_t publications = 0;
        uint64_t identical = 0;
        uint64_t bytes = 0;
        uint64_t identical_bytes = 0;
    };

    // One publication. `identical` means: the same target (address, extent and format) was published
    // before and its bytes hash the same. The first publication of a target is never identical.
    bool note(uint64_t address, uint32_t width, uint32_t height, uint32_t format, const uint8_t* data,
              size_t bytes) {
        if (!data || !bytes) return false;
        const uint64_t hash = hash_published_bytes(data, bytes);
        std::lock_guard lock(mutex_);
        Entry& e = targets_[address];
        const bool same_shape = e.seen && e.width == width && e.height == height && e.format == format &&
                                e.size == bytes;
        const bool identical = same_shape && e.hash == hash;
        e.seen = true;
        e.width = width;
        e.height = height;
        e.format = format;
        e.size = bytes;
        e.hash = hash;
        ++e.publications;
        e.bytes += bytes;
        ++totals_.publications;
        totals_.bytes += bytes;
        if (identical) {
            ++e.identical;
            e.identical_bytes += bytes;
            ++totals_.identical;
            totals_.identical_bytes += bytes;
        }
        return identical;
    }

    Totals totals() const {
        std::lock_guard lock(mutex_);
        return totals_;
    }

    // The `top` targets by bytes published, largest first.
    std::vector<TargetRow> rows(size_t top) const {
        std::vector<TargetRow> out;
        {
            std::lock_guard lock(mutex_);
            for (const auto& [address, e] : targets_) {
                TargetRow r;
                r.address = address;
                r.width = e.width;
                r.height = e.height;
                r.format = e.format;
                r.publications = e.publications;
                r.identical = e.identical;
                r.bytes = e.bytes;
                r.identical_bytes = e.identical_bytes;
                out.push_back(r);
            }
        }
        std::sort(out.begin(), out.end(), [](const TargetRow& a, const TargetRow& b) {
            return a.bytes != b.bytes ? a.bytes > b.bytes : a.address < b.address;
        });
        if (out.size() > top) out.resize(top);
        return out;
    }

    // Empty when nothing was published.
    std::string format(size_t top = 6) const {
        const Totals t = totals();
        if (!t.publications) return {};
        std::string out;
        char line[320];
        std::snprintf(line, sizeof(line),
                      "[unchanged-census] publications=%llu identical=%llu (%.1f%%) bytes=%.2f GB "
                      "identical_bytes=%.2f GB (%.1f%%)\n",
                      static_cast<unsigned long long>(t.publications),
                      static_cast<unsigned long long>(t.identical),
                      100.0 * static_cast<double>(t.identical) / static_cast<double>(t.publications),
                      static_cast<double>(t.bytes) / 1e9, static_cast<double>(t.identical_bytes) / 1e9,
                      t.bytes ? 100.0 * static_cast<double>(t.identical_bytes) / static_cast<double>(t.bytes)
                              : 0.0);
        out += line;
        for (const TargetRow& r : rows(top)) {
            std::snprintf(line, sizeof(line),
                          "[unchanged-census]   addr=0x%llx %ux%u format=%u publications=%llu identical=%llu "
                          "(%.1f%%) bytes=%.2f GB\n",
                          static_cast<unsigned long long>(r.address), r.width, r.height, r.format,
                          static_cast<unsigned long long>(r.publications),
                          static_cast<unsigned long long>(r.identical),
                          r.publications ? 100.0 * static_cast<double>(r.identical) /
                                               static_cast<double>(r.publications) : 0.0,
                          static_cast<double>(r.bytes) / 1e9);
            out += line;
        }
        return out;
    }

private:
    struct Entry {
        bool seen = false;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t format = 0;
        size_t size = 0;
        uint64_t hash = 0;
        uint64_t publications = 0;
        uint64_t identical = 0;
        uint64_t bytes = 0;
        uint64_t identical_bytes = 0;
    };

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, Entry> targets_;
    Totals totals_;
};

}  // namespace prosper::perf
