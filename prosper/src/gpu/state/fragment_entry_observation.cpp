#include "gpu/state/fragment_entry_observation.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"

namespace prosper::gpu {
void observe_fragment_entry(const GpuState& ds, bool has_source, FragmentEntryFacts& entry) {
    if (has_source) {
        entry.observed = true;
        for (uint32_t i = 0; i < entry.user_data.size(); ++i) {
            const auto word = ds.sh.find(prosper::agc::Pm4::SPI_SHADER_USER_DATA_PS_0 + i);
            if (word != ds.sh.end()) {
                entry.user_data_available |= uint32_t(1) << i;
                entry.user_data[i] = word->second;
            }
        }
        const auto rsrc2 = ds.sh.find(prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_PS);
        if (rsrc2 != ds.sh.end()) {
            entry.rsrc2_available = true;
            entry.rsrc2 = rsrc2->second;
        }
    }
}
}   // namespace prosper::gpu
