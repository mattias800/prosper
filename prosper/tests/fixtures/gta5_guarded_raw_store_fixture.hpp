#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace prosper::test {

// Existing checked-in 81 consumed dwords of the null guarded-store fixture. Keep the input
// independent of the production recognizer so same-site mutations test its trust boundary.
inline constexpr std::array<uint32_t, 81> kGta5GuardedRawStoreProgram = {
    0xbfa00003u, 0xd7460000u, 0x04010c07u, 0xf4040500u, 0xfa000090u, 0xbe960304u, 0xbe9703ffu,
    0x00016204u, 0xbf8cc07fu, 0xbe951d92u, 0x816ac104u, 0xe0302000u, 0x80050100u, 0xf4080300u,
    0xfa0000c0u, 0xbf8cc07fu, 0xf4200406u, 0xfa000000u, 0xbf8cc07fu, 0x7e040210u, 0xbe9703ffu,
    0x00016204u, 0xbe92047eu, 0x7da8006au, 0xbf880003u, 0x4a040081u, 0xe0302000u, 0x80050202u,
    0xbefe0412u, 0xbf8c3f70u, 0x7da80302u, 0xbf880009u, 0xf4040200u, 0xfa000098u, 0xbe8a0304u,
    0xbe8b03ffu, 0x00016204u, 0xbf8cc07fu, 0xbe891d92u, 0xe0702000u, 0x80020001u, 0xbefe0412u,
    0xbf128002u, 0x7d8a00f9u, 0x06868080u, 0x85ea8012u, 0x8dea006au, 0x87fe126au, 0xbf88001fu,
    0x7e0e0210u, 0x936b6a10u, 0x936a056au, 0x87048106u, 0x816b6a6bu, 0x9aea0510u, 0x93000510u,
    0x81016b6au, 0xbf078004u, 0x4a0620f9u, 0x86860681u, 0x85ea1000u, 0x7e000210u, 0x7e10026au,
    0x7e020210u, 0x7e040280u, 0x7e080210u, 0x7e0a0281u, 0x7e0c0281u, 0x8801ff03u, 0x00380000u,
    0xbe800302u, 0xbe820381u, 0xbe8303ffu, 0x00016204u, 0xe0740030u, 0x80000700u, 0xe0780020u,
    0x80000000u, 0xe07c0000u, 0x80000400u, 0xbf810000u,
};

inline prosper::gpu::ComputeShaderConfig
guarded_raw_store_config(std::span<const uint32_t> user_sgprs) {
    prosper::gpu::ComputeShaderConfig config;
    config.user_sgprs.assign(user_sgprs.begin(), user_sgprs.end());
    config.local_x = 64u;
    config.local_y = config.local_z = 1u;
    config.wave_size = 64u;
    config.native_subgroup_size = 64u;
    config.tidig_comp_cnt = 0u;
    return config;
}

} // namespace prosper::test
