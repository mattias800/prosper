#include "rdna2_gta5_cf9200_contract.hpp"

#include "rdna2_decode.hpp"
#include "rdna2_to_spirv.hpp"
#include "shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace prosper::gpu {

bool guest_readable(uint64_t address, uint32_t bytes);

namespace {

constexpr uint32_t kProgramDwords = 135u;
constexpr uint32_t kSourceDescriptorWord = 48u;
constexpr uint32_t kOptionalOutputPointerWord = 20u;
constexpr uint32_t kApplicationRecordWords = 4u;
constexpr uint32_t kApplicationWordMax = 0x00ffffffu;
constexpr uint32_t kMinimumApplicationRecords = 2u;

// Exact consumed prefix of routed GTA V PPSA04263 kernel 0x413cf9200. The bytes are declarative
// identity, not another decoder: the contract depends on the whole producer/branch/consumer shape.
constexpr char kProgramHex[] =
"0300a0bf8902007e000108f4c00000fa7fc08cbf000030e00011018001ff09880000e000000388be81038abeff038bbe04620100703f8cbf8a22022c1105187e82222428f9228a7d80828606c0020228c124204abf020436c002064a8004847dff060436c0ffffff02030202010d047e0105207e0c2070e0000101800257027e8010ea81ff020210feff7f4f010f027e0105d67e6a6b6a936a6bea9a6b6a6a816a0c919a11106a936a22024e1002044ef902867d108c86061004867d020001d5010532006a0cea87f904867d108e86061004064ef902030280068686020001d502073a00016a28d51102320080048a7d800202500204eabe002070e000010180740074e000100280330087bfc102007ec102027ec102047ec102067eff02087e0000807fff020a7e0000807fff020c7e0000807fff020e7e000080ffff02107e000080ffff02127e000080ffff02147e0000807fff02167e0000807fff02187e0000807fff021a7e000080ffff021c7e000080ffff021e7e000080ff000204f4500000fa81038abeff038bbe046201107fc08cbf961d89be000078e000000280700078e000000280100078e000040280200078e000080280300078e0000c0280400078e000040280500078e000080280600078e0000c02809f22084a8402007e010046d7120309038302047e8102067e8508082c82020a7e002070e002010180002070e003100180002070e000040180002070e005120180000081bf";

uint8_t nibble(char value) {
    return value >= '0' && value <= '9' ? static_cast<uint8_t>(value - '0')
                                        : static_cast<uint8_t>(value - 'a' + 10);
}

const std::array<uint8_t, kProgramDwords * sizeof(uint32_t)>& program_bytes() {
    static const auto bytes = [] {
        static_assert(sizeof(kProgramHex) - 1u == kProgramDwords * sizeof(uint32_t) * 2u);
        std::array<uint8_t, kProgramDwords * sizeof(uint32_t)> result{};
        for (size_t index = 0; index < result.size(); ++index)
            result[index] = static_cast<uint8_t>(
                nibble(kProgramHex[index * 2u]) << 4u | nibble(kProgramHex[index * 2u + 1u]));
        return result;
    }();
    return bytes;
}

struct Site {
    uint32_t pc;
    uint32_t word0;
    uint32_t word1;
    GtaCf9200NoBackingAccess access;
};

constexpr std::array<Site, 15> kNoBackingSites{{
    {5u,   0xe0300000u, 0x80011100u, GtaCf9200NoBackingAccess::LoadZero},
    {29u,  0xe070200cu, 0x80010100u, GtaCf9200NoBackingAccess::DropStore},
    {62u,  0xe0702000u, 0x80010100u, GtaCf9200NoBackingAccess::DropStore},
    {102u, 0xe0780000u, 0x80020000u, GtaCf9200NoBackingAccess::DropStore},
    {104u, 0xe0780070u, 0x80020000u, GtaCf9200NoBackingAccess::DropStore},
    {106u, 0xe0780010u, 0x80020400u, GtaCf9200NoBackingAccess::DropStore},
    {108u, 0xe0780020u, 0x80020800u, GtaCf9200NoBackingAccess::DropStore},
    {110u, 0xe0780030u, 0x80020c00u, GtaCf9200NoBackingAccess::DropStore},
    {112u, 0xe0780040u, 0x80020400u, GtaCf9200NoBackingAccess::DropStore},
    {114u, 0xe0780050u, 0x80020800u, GtaCf9200NoBackingAccess::DropStore},
    {116u, 0xe0780060u, 0x80020c00u, GtaCf9200NoBackingAccess::DropStore},
    {126u, 0xe0702000u, 0x80010102u, GtaCf9200NoBackingAccess::DropStore},
    {128u, 0xe0702000u, 0x80011003u, GtaCf9200NoBackingAccess::DropStore},
    {130u, 0xe0702000u, 0x80010400u, GtaCf9200NoBackingAccess::DropStore},
    {132u, 0xe0702000u, 0x80011205u, GtaCf9200NoBackingAccess::DropStore},
}};

constexpr std::array<uint32_t, 7> kSourceSites{{5u, 29u, 62u, 126u, 128u, 130u, 132u}};
constexpr std::array<uint32_t, 8> kOutputSites{{102u, 104u, 106u, 108u,
                                                110u, 112u, 114u, 116u}};

const ShaderResource* unique_fetch_resource(const ShaderResourceTable& resources,
                                            uint32_t fetch_pc) {
    const ShaderResource* result = nullptr;
    for (const ShaderResource& resource : resources.resources) {
        if (resource.fetch_pc != fetch_pc) continue;
        if (result) return nullptr;
        result = &resource;
    }
    return result;
}

bool root_shape(const ShaderResource& root, uint64_t expected_address) {
    const bool complete_host = root.host_data
        ? root.host_data_size >= kGtaCf9200RootBytes
        : root.host_data_size == 0u;
    return root.cls == ResourceClass::ConstantBuffer &&
           root.format == DataFormat::Float32 && root.num_components == 1u &&
           root.fetch_pc == kGtaCf9200RootPc && root.gpu_addr == expected_address &&
           root.gpu_addr > 0x10000u && root.size == kGtaCf9200RootBytes &&
           root.stride == kGtaCf9200RootBytes && root.table_index_count == 0u &&
           root.srt_offset == UINT32_MAX && root.sgpr_base == UINT32_MAX && complete_host;
}

const uint8_t* root_bytes(const ShaderResource& root) {
    if (root.host_data)
        return root.host_data_size >= kGtaCf9200RootBytes ? root.host_data : nullptr;
    if (!guest_readable(root.gpu_addr, kGtaCf9200RootBytes)) return nullptr;
    return reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(root.gpu_addr));
}

bool application_record(const uint32_t* words) {
    return words[0] != 0u && words[0] < words[1] &&
           words[0] <= kApplicationWordMax && words[1] <= kApplicationWordMax &&
           words[2] == UINT32_MAX && words[3] == UINT32_MAX;
}

struct RootDomain {
    bool source_no_backing = false;
    bool output_no_backing = false;
};

bool root_domain(const uint8_t* bytes, RootDomain& domain) {
    if (!bytes) return false;
    std::array<uint32_t, kGtaCf9200RootBytes / sizeof(uint32_t)> words{};
    std::memcpy(words.data(), bytes, kGtaCf9200RootBytes);
    uint32_t records = 0;
    for (size_t index = 0; index + kApplicationRecordWords <= words.size(); ++index)
        records += application_record(words.data() + index) ? 1u : 0u;
    if (records < kMinimumApplicationRecords) return false;
    domain.source_no_backing = application_record(words.data() + kSourceDescriptorWord);
    domain.output_no_backing = words[kOptionalOutputPointerWord] == 0u &&
                               words[kOptionalOutputPointerWord + 1u] == 0u;
    return domain.source_no_backing || domain.output_no_backing;
}

bool expected_site(uint32_t pc, const RootDomain& domain) {
    if (domain.source_no_backing &&
        std::find(kSourceSites.begin(), kSourceSites.end(), pc) != kSourceSites.end())
        return true;
    return domain.output_no_backing &&
           std::find(kOutputSites.begin(), kOutputSites.end(), pc) != kOutputSites.end();
}

size_t expected_site_count(const RootDomain& domain) {
    return (domain.source_no_backing ? kSourceSites.size() : 0u) +
           (domain.output_no_backing ? kOutputSites.size() : 0u);
}

uint64_t root_address(const ComputeShaderConfig& config) {
    if (config.user_sgprs.size() != 2u) return 0u;
    return static_cast<uint64_t>(config.user_sgprs[0]) |
           (static_cast<uint64_t>(config.user_sgprs[1]) << 32u);
}

} // namespace

bool rdna2_gta5_cf9200_shader(const uint32_t* code, size_t dwords) {
    if (!code || dwords < kProgramDwords ||
        std::memcmp(code, program_bytes().data(), program_bytes().size()) != 0)
        return false;
    std::vector<Rdna2Inst> instructions;
    const size_t consumed = rdna2_walk(code, dwords, instructions);
    return consumed == kProgramDwords && !instructions.empty() && instructions.back().is_end;
}

bool rdna2_gta5_cf9200_launch(
        const uint32_t* code, size_t dwords, const ComputeShaderConfig& config) {
    return rdna2_gta5_cf9200_shader(code, dwords) && config.user_sgprs.size() == 2u &&
           root_address(config) > 0x10000u &&
           config.local_x == 1u && config.local_y == 1u && config.local_z == 1u &&
           config.threads_x == 1u && config.threads_y == 1u && config.threads_z == 1u &&
           config.wave_size == 64u && !config.tgid_x_en && !config.tgid_y_en &&
           !config.tgid_z_en && !config.tg_size_en && config.tidig_comp_cnt == 0u &&
           config.lds_bytes == 0u;
}

GtaCf9200NoBackingAccess rdna2_gta5_cf9200_no_backing_site(
        const Rdna2Inst& instruction) {
    const auto found = std::find_if(kNoBackingSites.begin(), kNoBackingSites.end(),
        [&](const Site& site) {
            return instruction.pc == site.pc && instruction.fmt == Rdna2Format::MUBUF &&
                   instruction.len_dwords == 2u && instruction.words[0] == site.word0 &&
                   instruction.words[1] == site.word1;
        });
    return found == kNoBackingSites.end() ? GtaCf9200NoBackingAccess::None : found->access;
}

bool is_gta5_cf9200_no_backing_marker_candidate(const ShaderResource& resource) {
    return resource.stride == kGtaCf9200NoBackingStride;
}

bool is_proven_gta5_cf9200_no_backing(const ShaderResource& resource) {
    const bool complete_host = resource.host_data
        ? resource.host_data_size >= kGtaCf9200RootBytes
        : resource.host_data_size == 0u;
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.format == DataFormat::Unknown && resource.num_components == 0u &&
           resource.gpu_addr > 0x10000u && resource.size == kGtaCf9200RootBytes &&
           is_gta5_cf9200_no_backing_marker_candidate(resource) &&
           resource.srt_offset == UINT32_MAX && resource.sgpr_base == UINT32_MAX &&
           resource.fetch_pc != UINT32_MAX && resource.table_index_count == 0u && complete_host;
}

bool discover_rdna2_gta5_cf9200_no_backing(
        const uint32_t* code, size_t dwords, const ComputeShaderConfig& config,
        ShaderResourceTable& resources) {
    if (!rdna2_gta5_cf9200_launch(code, dwords, config)) return false;
    const uint64_t address = root_address(config);
    const ShaderResource* root = unique_fetch_resource(resources, kGtaCf9200RootPc);
    if (!root || !root_shape(*root, address)) return false;
    RootDomain domain;
    if (!root_domain(root_bytes(*root), domain)) return false;

    const std::vector<ShaderResource> original = resources.resources;
    std::erase_if(resources.resources, [&](const ShaderResource& resource) {
        return is_gta5_cf9200_no_backing_marker_candidate(resource) ||
               expected_site(resource.fetch_pc, domain);
    });
    auto add_markers = [&](const auto& sites, bool enabled) {
        if (!enabled) return;
        for (uint32_t pc : sites) {
            ShaderResource marker;
            marker.cls = ResourceClass::ConstantBuffer;
            marker.format = DataFormat::Unknown;
            marker.num_components = 0u;
            marker.gpu_addr = address;
            marker.size = kGtaCf9200RootBytes;
            marker.stride = kGtaCf9200NoBackingStride;
            marker.fetch_pc = pc;
            resources.resources.push_back(marker);
        }
    };
    add_markers(kSourceSites, domain.source_no_backing);
    add_markers(kOutputSites, domain.output_no_backing);
    if (rdna2_gta5_cf9200_no_backing_dispatch(code, dwords, config, resources)) return true;
    resources.resources = original;
    return false;
}

bool rdna2_gta5_cf9200_no_backing_dispatch(
        const uint32_t* code, size_t dwords, const ComputeShaderConfig& config,
        const ShaderResourceTable& resources) {
    if (!rdna2_gta5_cf9200_launch(code, dwords, config)) return false;
    const uint64_t address = root_address(config);
    const ShaderResource* root = unique_fetch_resource(resources, kGtaCf9200RootPc);
    if (!root || !root_shape(*root, address)) return false;
    RootDomain domain;
    if (!root_domain(root_bytes(*root), domain)) return false;

    std::array<bool, kNoBackingSites.size()> found{};
    size_t marker_count = 0;
    for (const ShaderResource& resource : resources.resources) {
        const bool candidate = is_gta5_cf9200_no_backing_marker_candidate(resource);
        if (expected_site(resource.fetch_pc, domain) && !candidate) return false;
        if (!candidate) continue;
        if (!is_proven_gta5_cf9200_no_backing(resource) || resource.gpu_addr != address ||
            !expected_site(resource.fetch_pc, domain))
            return false;
        const auto site = std::find_if(kNoBackingSites.begin(), kNoBackingSites.end(),
            [&](const Site& candidate_site) { return candidate_site.pc == resource.fetch_pc; });
        if (site == kNoBackingSites.end()) return false;
        const size_t index = static_cast<size_t>(site - kNoBackingSites.begin());
        if (found[index]) return false;
        found[index] = true;
        ++marker_count;
    }
    if (marker_count != expected_site_count(domain)) return false;
    for (const Site& site : kNoBackingSites) {
        if (expected_site(site.pc, domain) !=
            found[static_cast<size_t>(&site - kNoBackingSites.data())])
            return false;
    }
    return true;
}

} // namespace prosper::gpu
