// service_census.cpp -- selected block-end nodes against the dispatcher's separate tail pointers.
#include "gpu/recompiler/ir/cfg/service_census.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/ir/cfg/service_nodes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace prosper::gpu::ir::cfg {
namespace {
thread_local std::vector<ServiceRegionRecord>* g_capture = nullptr;
}   // namespace

ServiceCapture::ServiceCapture() : previous_(g_capture) {
    g_capture = &records_;
}
ServiceCapture::~ServiceCapture() {
    g_capture = previous_;
}

ServiceObservation::ServiceObservation(std::span<const Rdna2Inst> ins, const SplitPolicy& policy,
                                       size_t auxiliary_raw_mask_sites)
    : capture_(g_capture), ins_(ins) {
    if (!capture_) return;
    cfg_ = build_guest_cfg(ins, policy);
    record_.entry_pc = ins.empty() ? 0 : ins.front().pc;
    record_.split_sites = policy.services ? policy.services->size() : 0;
    record_.auxiliary_raw_mask_sites = auxiliary_raw_mask_sites;
}

void ServiceObservation::block(uint32_t dispatch, uint32_t block_pc,
                               const SelectedServicePointers& selected) {
    if (!capture_) return;
    ++record_.blocks;
    for (size_t k = 0; k < selected.size(); ++k)
        if (selected[k])
            selections_.push_back(
                {dispatch, block_pc, {selected[k]->pc, static_cast<ServiceKind>(k)}});
}

void ServiceObservation::group(uint32_t dispatch, const SelectedServicePointers& selected) {
    if (!capture_) return;
    groups_.push_back(dispatch);
    ++record_.groups;
    for (size_t k = 0; k < selected.size(); ++k)
        if (selected[k])
            record_.dispatcher.push_back(
                {dispatch, {selected[k]->pc, static_cast<ServiceKind>(k)}});
}

ServiceObservation::~ServiceObservation() {
    if (!capture_) return;
    std::vector<BlockServiceSelection> completed;
    for (const auto& selection : selections_) {
        if (std::find(groups_.begin(), groups_.end(), selection.dispatch) != groups_.end())
            completed.push_back(selection);
        else
            ++record_.pending_selections;
    }
    record_.model = build_service_nodes(cfg_, ins_, completed);
    record_.difference = service_nodes_difference(record_.model, record_.dispatcher);
    capture_->push_back(std::move(record_));
}

}   // namespace prosper::gpu::ir::cfg
