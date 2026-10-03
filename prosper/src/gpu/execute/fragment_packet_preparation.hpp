#pragma once
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct RasterQuadInputs;
struct FragmentPacketVgprRequirements;
struct FragmentPacketMaskRequirements;
struct FragmentPacketScalarReadRequirements;
// A bounded immutable copy of the producing normalized binding contract. Only genuinely owned
// buffer bytes are copied; live guest/image content never gains authority from nominal addresses.
// Owned byte membership is not a fetch-PC/read-point, producer epoch or image/sampler witness.
struct FragmentPacketResources {
    std::shared_ptr<const ShaderResourceTable> table;
    bool observed = false;
    std::vector<bool> host_backing_owned;
    std::string rejection;
};
FragmentPacketResources own_fragment_packet_resources(const ShaderResourceTable*);

struct FragmentPacketPreparation {
    std::shared_ptr<const RasterQuadInputs> inputs;
    std::vector<std::string> unmet;
    // Only the source-associated, canonical PS launch can supply the user prefix. These owned
    // raw words are suitable for FragmentInvocationPacket::sgprs, but do not initialize system
    // SGPRs (the first is s[user_sgpr_count]), M0, VGPRs or logical-wave/coverage state.
    bool user_sgpr_count_available = false;
    uint32_t user_sgpr_count = 0;
    uint32_t missing_user_sgprs = 0;
    std::vector<std::pair<uint32_t, uint32_t>> initial_user_sgprs;
    // Code-only inventory from the producing immutable shader-analysis owner. Scratch allocation
    // never asks for launch values; actual masked/peer/raw-export reads need runtime validity.
    std::shared_ptr<const FragmentPacketVgprRequirements> vgpr_requirements;
    std::shared_ptr<const FragmentPacketMaskRequirements> mask_requirements;
    // Exact code demands plus observed descriptor words. These supply neither ordered byte reads
    // nor source extent/alias/upload/completion authority; all live resource guards remain closed.
    std::shared_ptr<const FragmentPacketScalarReadRequirements> scalar_read_requirements;
    struct ScalarDescriptorObservation {
        uint32_t site_index = 0;
        std::array<uint32_t, 4> descriptor{};
    };
    std::vector<ScalarDescriptorObservation> scalar_descriptors;
    // This provenance/preflight slice cannot grant kernel or graphics admission.
    bool ready = false;
};
// Actual refused-draw consumer. Missing registers are never zero-filled; observed resources do
// not establish host-subgroup pairing, guest coverage or attachment effects.
std::shared_ptr<const FragmentPacketPreparation> prepare_fragment_packet_inputs(
    std::shared_ptr<const RasterQuadInputs>, bool producing_modules_match);
}   // namespace prosper::gpu
