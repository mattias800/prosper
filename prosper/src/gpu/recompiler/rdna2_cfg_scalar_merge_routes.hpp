// Borrowed variables for the original scalar CFG worklist step.
#pragma once
#include "gpu/recompiler/rdna2_cfg_scalar_merge.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace prosper::gpu::detail {
struct CfgScalarMergeRoutes {
    const Rdna2Inst& in;
    int& R;
    const bool& data_read_ok;
    const std::function<bool(const Rdna2Inst&, const char*)>& block;
    std::unordered_map<uint32_t, std::size_t>& pc_to_index;
    std::vector<std::size_t>& pending;
    std::size_t& index;
    const std::vector<Rdna2Inst>& ins;
    ScalarMergeMaskWriteWidths& mask_widths;
    bool& vopc_sgpr_pair_kills;
    std::optional<bool> step();

private:
    bool implicit_reads();
    bool alu_reads();
    std::optional<bool> sopk();
    std::optional<bool> sopp();
    std::optional<bool> smem();
    std::optional<bool> alu();
    std::optional<bool> descriptors();
    std::optional<bool> flat_vintrp();
};
} // namespace prosper::gpu::detail
