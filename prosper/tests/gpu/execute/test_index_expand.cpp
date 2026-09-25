#include "gpu/execute/index_expand.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

bool check_one(size_t count, size_t source_offset, size_t output_offset, size_t peak) {
    constexpr uint32_t guard = 0xfeedfaceu;
    std::vector<uint16_t> source(count + source_offset + 8, 0xabcdu);
    std::vector<uint32_t> output(count + output_offset + 8, guard);
    uint16_t* src = source.data() + source_offset;
    uint32_t* dst = output.data() + output_offset;
    for (size_t i = 0; i < count; ++i)
        src[i] = static_cast<uint16_t>((i * 31337u + 7919u) & 0xffffu);
    if (count) src[peak % count] = 0xffffu;

    const uint32_t expected_max = count ? *std::max_element(src, src + count) : 0u;
    const auto check_output = [&](uint32_t maximum) {
        if (maximum != expected_max) return false;
        for (size_t i = 0; i < output.size(); ++i) {
            const uint32_t expected = i >= output_offset && i < output_offset + count
                ? src[i - output_offset] : guard;
            if (output[i] != expected) return false;
        }
        return true;
    };
    if (!check_output(prosper::gpu::copy_indices_u16_max_scalar(dst, src, count))) return false;
    std::fill(output.begin(), output.end(), guard);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (prosper::gpu::index_expand_avx2_available()) {
        if (!check_output(prosper::gpu::copy_indices_u16_max_avx2(dst, src, count))) return false;
        std::fill(output.begin(), output.end(), guard);
    }
#endif
    return check_output(prosper::gpu::copy_indices_u16_max(dst, src, count));
}

#if defined(__linux__)
bool check_mapping_edge() {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return false;
    void* pages = mmap(nullptr, static_cast<size_t>(page_size) * 2,
                       PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) return false;
    const bool protected_page = mprotect(static_cast<uint8_t*>(pages) + page_size,
                                         static_cast<size_t>(page_size), PROT_NONE) == 0;
    bool valid = protected_page;
    if (valid) {
        for (size_t count : {8u, 9u, 15u, 16u, 17u}) {
            auto* src = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(pages) + page_size)
                        - count;
            for (size_t i = 0; i < count; ++i) src[i] = static_cast<uint16_t>(i * 67u);
            std::vector<uint32_t> dst(count + 1, 0xfeedfaceu);
            const uint32_t maximum = prosper::gpu::copy_indices_u16_max(dst.data(), src, count);
            valid &= maximum == (count - 1) * 67u && dst[count] == 0xfeedfaceu;
            for (size_t i = 0; i < count; ++i) valid &= dst[i] == i * 67u;
        }
    }
    munmap(pages, static_cast<size_t>(page_size) * 2);
    return valid;
}
#endif

} // namespace

int main() {
    constexpr std::array<size_t, 19> sizes = {
        0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 8193};
    for (size_t count : sizes)
        for (size_t source_offset : {0u, 1u, 3u})
            for (size_t output_offset : {0u, 1u, 3u})
                for (size_t peak : {0u, 7u, 8u, 15u, 257u})
                    if (!check_one(count, source_offset, output_offset, peak)) {
                        std::fprintf(stderr, "index expansion failed count=%zu source=%zu output=%zu peak=%zu\n",
                                     count, source_offset, output_offset, peak);
                        return 1;
                    }
#if defined(__linux__)
    if (!check_mapping_edge()) {
        std::fputs("index expansion crossed a validated guest mapping edge\n", stderr);
        return 1;
    }
#endif
    std::puts("index expansion and maximum controls passed");
    return 0;
}
