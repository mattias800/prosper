#pragma once
// Observation of actual guest F32 ADD/MUL lowering, not an instruction inventory or permission.
#include "gpu/recompiler/fragment_float_mode.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace prosper::gpu {

enum class FragmentArithmeticFamily : uint8_t { Add, Mul };
struct FragmentArithmeticSite {
    uint32_t pc = 0;
    FragmentArithmeticFamily family = FragmentArithmeticFamily::Add;
    bool operator==(const FragmentArithmeticSite&) const = default;
};
struct FragmentArithmeticObservation {
    static constexpr size_t kSiteLimit = 8;
    uint64_t producing_program = 0;
    uint64_t source_fingerprint = 0; // not an exact-content or unique-shader identity
    FragmentFloatMode float_mode{};
    uint8_t families = 0;
    uint8_t site_count = 0;
    uint64_t truncated_emissions = 0; // repeated emission beyond the prefix may be counted again
    std::array<FragmentArithmeticSite, kSiteLimit> sites{};

    void record(uint32_t pc, FragmentArithmeticFamily family) {
        families |= static_cast<uint8_t>(1u << static_cast<uint8_t>(family));
        const FragmentArithmeticSite site{pc, family};
        for (size_t i = 0; i < site_count; ++i)
            if (sites[i] == site) return;
        if (site_count == kSiteLimit) { ++truncated_emissions; return; }
        sites[site_count++] = site;
    }
};

} // namespace prosper::gpu
