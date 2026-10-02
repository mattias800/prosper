#pragma once
#include "gpu/resources/shader_resources.hpp"
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {
struct RasterQuadInputs;
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
    // This provenance/preflight slice cannot grant kernel or graphics admission.
    bool ready = false;
};
// Actual refused-draw consumer. Observed registers/resources are retained, never converted into
// zero-seeded guest registers, host-subgroup pairing, guest coverage or attachment effects.
std::shared_ptr<const FragmentPacketPreparation> prepare_fragment_packet_inputs(
    std::shared_ptr<const RasterQuadInputs>, bool producing_modules_match);
} // namespace prosper::gpu
