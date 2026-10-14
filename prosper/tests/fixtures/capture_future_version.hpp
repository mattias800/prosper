#pragma once
// Capture writers select the oldest format needed by their fields. A version offset from that
// header can therefore be supported by the reader and exercise a missing tail instead of its
// unsupported-version contract. Both capture tests use this synthetic header producer.
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace prosper::test {
inline uint32_t
unsupported_capture_version([[maybe_unused]] const std::vector<uint8_t>& writer_bytes) noexcept {
    // The wire field's largest value is unsupported independently of conditional writer tails.
    return std::numeric_limits<uint32_t>::max();
}
inline uint32_t install_unsupported_capture_version(std::vector<uint8_t>& bytes) {
    if (bytes.size() < 12) throw std::invalid_argument("fixture needs a complete version header");
    const uint32_t version = unsupported_capture_version(bytes);
    for (uint32_t i = 0; i < 4; ++i) bytes[8 + i] = uint8_t(version >> (8u * i));
    return version;
}
} // namespace prosper::test
