#pragma once
#include "gpu/state/raster_launch_facts.hpp"

namespace prosper::gpu {
// Necessary producing-state predicates for the first live/helper bridge. This is not a public
// launch token: the plan additionally requires exact draw/code ownership, its original quad-local
// program proof and the backend's same geometry/viewport/scissor, one-sample, disabled-DS contract.
const char* fragment_raster_workitem_gap(const RasterLaunchFacts&);
}   // namespace prosper::gpu
