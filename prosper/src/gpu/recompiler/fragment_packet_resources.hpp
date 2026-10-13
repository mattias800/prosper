#pragma once
#include "gpu/resources/shader_resources.hpp"
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {
// A bounded immutable copy of the producing normalized binding contract. Only genuinely owned
// buffer bytes are copied; live guest/image content never gains authority from nominal addresses.
// Owned byte membership is not a fetch-PC/read-point, producer epoch or image/sampler witness.
struct FragmentPacketResources {
    std::shared_ptr<const ShaderResourceTable> table;
    bool observed = false;
    std::vector<bool> host_backing_owned;
    std::string rejection;
};
}   // namespace prosper::gpu
