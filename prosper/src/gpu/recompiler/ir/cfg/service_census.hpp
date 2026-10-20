// service_census.hpp -- observation only; no compiler decision reads these records.
#pragma once

#include "gpu/recompiler/ir/cfg/service_nodes.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace prosper::gpu::ir::cfg {

// The already selected pointers, indexed by ServiceKind. A slot is null when not selected.
using SelectedServicePointers = std::array<const Rdna2Inst*, kServiceKindCount>;

template <class... Pointer>
SelectedServicePointers selected_service_pointers(Pointer... selected) {
    static_assert(sizeof...(Pointer) == kServiceKindCount);
    return {selected...};
}

struct ServiceRegionRecord {
    uint32_t entry_pc = 0;
    size_t blocks = 0;
    size_t groups = 0;
    size_t split_sites = 0;   // syntactic candidates; selection can use ordinary ALU instead
    size_t auxiliary_raw_mask_sites = 0;
    size_t pending_selections = 0;   // aborted before that group's tail assignments
    bool complete = false;   // every dispatch group reached its service selection
    ServiceNodes model;
    std::vector<DispatchServiceEvent> dispatcher;
    std::string difference;
};

class ServiceCapture {
public:
    ServiceCapture();
    ~ServiceCapture();
    ServiceCapture(const ServiceCapture&) = delete;
    ServiceCapture& operator=(const ServiceCapture&) = delete;
    const std::vector<ServiceRegionRecord>& records() const { return records_; }

private:
    std::vector<ServiceRegionRecord> records_;
    std::vector<ServiceRegionRecord>* previous_ = nullptr;
};

// Scope around the existing case-selection walk. Reads only decoded instructions, split policy,
// group identity and the already selected block/dispatch pointers. No RegState or proof reads.
// A refused case can leave a partial region: only groups whose tail assignments were observed
// enter the comparison, and pending selections are reported separately.
class ServiceObservation {
public:
    ServiceObservation(std::span<const Rdna2Inst> ins, const SplitPolicy& policy,
                       size_t auxiliary_raw_mask_sites = 0);
    ~ServiceObservation();
    ServiceObservation(const ServiceObservation&) = delete;
    ServiceObservation& operator=(const ServiceObservation&) = delete;
    void block(uint32_t dispatch, uint32_t block_pc, const SelectedServicePointers& selected);
    void group(uint32_t dispatch, const SelectedServicePointers& selected);
    void complete() { record_.complete = true; }

private:
    std::vector<ServiceRegionRecord>* capture_ = nullptr;
    std::span<const Rdna2Inst> ins_;
    GuestCfg cfg_;
    std::vector<BlockServiceSelection> selections_;
    std::vector<uint32_t> groups_;
    ServiceRegionRecord record_;
};

}   // namespace prosper::gpu::ir::cfg
