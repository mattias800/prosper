// hle_addcontent.hpp — validated host inventory for installed local PS5 add-content.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace prosper {

// The SKU flag of the running application, as the published AppContent contract enumerates it. There
// is deliberately no "unknown" enumerator: an application whose local declaration prosper cannot
// recognise has no SKU to report, which is an absent std::optional, not a third value a guest could
// mistake for data.
//
// Guest code in the project's local test dumps pins both values independently of any header: Crisis
// Core (PPSA07809) holds a predicate that is true only when sceNpEntitlementAccessGetSkuFlag yields
// exactly 3, Little Nightmares II (PPSA02154) holds one that is true only when it yields 1, and both
// GTA V (PPSA04263) and Little Nightmares III (PPSA05143) route a 1 into their restricted-SKU branch.
enum class AppSkuFlag : int32_t {
    Trial = 1,   // SCE_APP_CONTENT_APPPARAM_SKU_FLAG_TRIAL
    Full  = 3,   // SCE_APP_CONTENT_APPPARAM_SKU_FLAG_FULL
};

// What the installed application declares about ITSELF in sce_sys/param.json. Sony asks the same SKU
// question through two libraries — sceAppContentAppParamGetInt(SKU_FLAG) and
// sceNpEntitlementAccessGetSkuFlag — so both read this one derivation and cannot disagree.
struct AppParamDeclaration {
    // sce_sys/param.json was present and parsed. False means there is no local declaration to answer
    // from at all, and every app-param query must fail rather than invent one.
    bool declared = false;
    // Absent when param.json declares no applicationDrmType, or declares one prosper has no evidence
    // for. Either way the SKU is unknown and the queries fail visibly instead of reporting a value.
    std::optional<AppSkuFlag> sku_flag;
    // Exactly as declared, for diagnostics; empty when the key is absent.
    std::string declared_drm_type;
    // USER_DEFINED_PARAM_1..4. param.json omits a userDefinedParamN the publishing tool left at its
    // default, so an absent key is a declared zero rather than missing information.
    int32_t user_param[4]{};
};

// Published by addcontent_configure_for_app0 from the same single param.json parse the add-content
// inventory uses. Safe to call from any guest thread.
AppParamDeclaration app_param_declaration();

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
    std::string guest_mount_point;   // validated declared path; mountable gates package policy
    std::array<uint8_t, 16> entitlement_key{};
    bool mountable = false;           // producer default: installed PSAC with a declared path
    bool mounted = false;             // process-local AppContent mount state
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

enum class AddcontentMountResult {
    Mounted,
    NotFound,
    Busy,
    OutputError,
};

using AddcontentMountWriter = bool (*)(uint64_t, const void*, std::size_t);

// Atomically validate, write, and claim a configured mount. Successful output is the complete
// zero-padded 16-byte SceAppContentMountPoint object. A failed guest write does not consume the
// mount. The claim is released by addcontent_unmount below.
AddcontentMountResult addcontent_mount(uint32_t service_label, std::string_view entitlement_label,
                                       uint64_t output_address, AddcontentMountWriter writer);

enum class AddcontentUnmountResult {
    Unmounted,
    NotMounted,   // the path is not a currently-claimed mount point (or is not a mount point at all)
};

// Release a claim taken by addcontent_mount, identified by the mount point it handed back.
//
// This is the inverse of the claim and NOTHING else: it clears an entry's `mounted` flag. It grants
// no entitlement, changes no ownership answer, and cannot make absent content present — an unknown
// or already-free mount point is reported as NotMounted rather than quietly accepted. That
// fail-visible direction is also what makes the handler safe under residual uncertainty about the
// guest's argument: an argument we fail to recognise yields an error, never a false success.
AddcontentUnmountResult addcontent_unmount(std::string_view guest_mount_point);

} // namespace prosper
