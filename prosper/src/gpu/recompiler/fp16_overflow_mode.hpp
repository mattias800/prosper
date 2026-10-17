// Independent producing arithmetic state: an absent observation is not a cleared FP16_OVFL.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct Fp16OverflowMode {
    bool available = false;
    bool saturate = false;

    constexpr bool canonical() const { return available || !saturate; }
    bool operator==(const Fp16OverflowMode&) const = default;
};

// GC10.3 COMPUTE_PGM_RSRC1[26]. Old raw configs may be defaulted, so presence is explicit.
// A physical compute wrapper for graphics must keep its graphics-producing carrier instead.
constexpr Fp16OverflowMode compute_fp16_overflow_from_rsrc1(bool observed, uint32_t word) {
    return observed ? Fp16OverflowMode{true, (word & (1u << 26)) != 0} : Fp16OverflowMode{};
}
} // namespace prosper::gpu
