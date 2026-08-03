// hle_addcontent.hpp — validated host inventory for installed local PS5 add-content.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace prosper {

enum class AddcontentInventoryState {
    None,       // no dlc_emu.ini: this title has no locally declared add-content
    Ready,
    Invalid,    // manifest exists but failed validation; never masquerade as "no DLC"
};

struct InstalledAddcontent {
    int64_t service_label = -1;       // -1 = producer-defined wildcard; otherwise uint32_t
    std::string entitlement_label;   // 1..16 validated ASCII characters
    uint32_t package_type = 0;       // PSAC=2, PSAL=3 (SceNpEntitlementAccess ABI)
    uint32_t download_status = 4;    // INSTALLED
    std::string guest_mount_point;   // empty for licence-only add-content
    std::array<uint8_t, 16> entitlement_key{};
    bool has_entitlement_key = false;
};

struct AddcontentInventorySnapshot {
    AddcontentInventoryState state = AddcontentInventoryState::None;
    std::vector<InstalledAddcontent> entries;
};

// Called whenever the host /app0 root changes. The manifest is parsed once, and only fully validated
// records are published to HLE readers. Diagnostics deliberately omit the host root (public logs must
// not expose the developer's private directory layout).
void addcontent_configure_for_app0(const std::string& app0_root);
AddcontentInventorySnapshot addcontent_inventory_snapshot();

} // namespace prosper
