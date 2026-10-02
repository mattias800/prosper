#pragma once
// Project-owned raw compute programs and integer storage oracles. No title
// data.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace prosper::test::bpermute {
constexpr uint32_t sentinel = 0xdeadbeefu;
struct Case {
  uint32_t local = 64, wave = 64, threads = 64, offset = 0, selector = 0;
  uint32_t mask = 0;
  bool loop = false, second = false, phased = false, in_place = false;
  bool different_waves = false, different_sites = false;
};
inline uint32_t data(uint32_t lane) { return 0x51000000u + lane * 17u; }
inline bool enabled(const Case &c, uint32_t lane) {
  if (lane >= c.threads)
    return false;
  if (c.mask == 1)
    return lane % 32 != 0;
  if (c.mask == 2)
    return lane % 32 == 31;
  if (c.mask == 3)
    return false;
  if (c.mask == 4)
    return lane % 2 == 0;
  return true;
}
inline uint32_t address(const Case &c, uint32_t lane) {
  if (c.selector == 1)
    return 124;
  if (c.selector == 2)
    return 0xfffffffcu;
  if (c.selector == 3)
    return 0;
  // Reverse each 32-lane half; nonzero high index bits must not escape it.
  return (31 - lane % 32) * 4 + 0x800;
}
inline std::vector<uint32_t> input(const Case &c) {
  std::vector<uint32_t> v(c.local * 4, sentinel);
  for (uint32_t lane = 0; lane < c.local; ++lane) {
    v[lane] = data(lane);
    v[c.local + lane] = address(c, lane);
    v[c.local * 2 + lane] = enabled(c, lane);
  }
  return v;
}
inline std::vector<uint32_t> expected(const Case &c) {
  auto v = input(c);
  std::vector<uint32_t> values(c.local), dst(c.local, sentinel);
  for (uint32_t i = 0; i < c.local; ++i)
    values[i] = data(i);
  for (uint32_t iteration = 0;
       iteration < (c.loop ? (c.different_waves ? c.local / c.wave : 3u) : 1u);
       ++iteration) {
    // Independently enumerate the hardware's byte-index slot and physical half.
    // This is not a call to, or copy of, the emitter's integer expression.
    for (uint32_t lane = 0; lane < c.local; ++lane) {
      if (!enabled(c, lane))
        continue;
      if (c.loop && c.different_waves && iteration >= lane / c.wave + 1)
        continue;
      const bool alternate = c.different_sites && lane / c.wave == 1;
      const uint32_t byte = address(c, lane) + c.offset + (alternate ? 4u : 0u);
      const uint32_t slot = (byte % 128) / 4;
      const uint32_t peer = (lane / 32) * 32 + slot;
      dst[lane] = peer < c.local && enabled(c, peer)
                      ? values[peer] + (alternate ? 0x10000u : 0u)
                      : 0;
    }
    if (c.in_place)
      values = dst;
    if (c.second) {
      const auto first = dst;
      for (uint32_t lane = 0; lane < c.local; ++lane) {
        if (!enabled(c, lane))
          continue;
        const uint32_t slot = ((address(c, lane) + c.offset + 4) % 128) / 4;
        const uint32_t peer = (lane / 32) * 32 + slot;
        dst[lane] = peer < c.local && enabled(c, peer) ? first[peer] : 0;
      }
    }
    if (c.loop)
      for (uint32_t lane = 0; lane < c.local; ++lane)
        if (enabled(c, lane) &&
            (!c.different_waves || iteration < lane / c.wave + 1))
          ++values[lane];
  }
  for (uint32_t lane = 0; lane < c.threads; ++lane)
    v[c.local * 3 + lane] = dst[lane];
  return v;
}
inline void add(std::vector<uint32_t> &p, uint32_t dst, uint32_t src,
                uint32_t n) {
  p.insert(p.end(), {0x4a0000ffu | (dst << 17) | (src << 9), n});
}
inline void load(std::vector<uint32_t> &p, uint32_t dst, uint32_t index) {
  p.insert(p.end(), {0xe0302000u, 0x80020000u | (dst << 8) | index});
}
inline std::vector<uint32_t> guest(const Case &c) {
  std::vector<uint32_t> p;
  load(p, 1, 0);
  add(p, 3, 0, c.local);
  load(p, 2, 3);
  add(p, 3, 0, c.local * 2);
  load(p, 6, 3);
  p.insert(p.end(),
           {0x7e0802ffu, sentinel}); // v4 sentinel (live inactive VDST)
  if (c.different_waves || c.different_sites) {
    p.insert(p.end(), {0xd7600000u, 0x00010100u, // s0=V_READLANE(v0,0)
                       0x90008600u,              // s_lshr_b32 s0,s0,6
                       0x80008100u,              // s_add_u32 s0,s0,1
                       0xbe810300u}); // preserve wave ordinal+1 in s1
  } else if (c.loop)
    p.push_back(0xbe800383u); // s0=3
  if (c.phased)
    p.push_back(0xbf8a0000u); // S_BARRIER before first gather
  p.insert(p.end(), {0xd4c20006u, 262u | (129u << 9), // CMP_EQ_U32 s[6:7],v6,1
                     0xbe942406u}); // save full EXEC to s[20:21], then narrow
  const uint32_t loop_pc = static_cast<uint32_t>(p.size());
  if (c.different_sites) {
    add(p, 7, 1, 0x10000);
    p.insert(p.end(),
             {0xbf068201u, 0xbf840003u, // s1==2? one wave uses a distinct site
              0xdacc0000u | (c.offset + 4), (4u << 24) | (7u << 8) | 2u,
              0xbf820002u});
  }
  p.insert(p.end(), {0xdacc0000u | c.offset,
                     ((c.in_place ? 1u : 4u) << 24) | (1u << 8) | 2u});
  if (c.in_place)
    p.push_back(0x7e080301u); // v4=v1
  if (c.second)
    p.insert(p.end(),
             {0xdacc0000u | (c.offset + 4), (4u << 24) | (4u << 8) | 2u});
  if (c.loop) {
    p.push_back(0x4a020281u); // v_add_nc_u32 v1,1,v1
    p.push_back(0x80808100u); // s_sub_u32 s0,s0,1
    p.push_back(0xbf078000u); // s_cmp_lg_u32 s0,0
    const int32_t displacement =
        static_cast<int32_t>(loop_pc) - static_cast<int32_t>(p.size() + 1);
    p.push_back(0xbf850000u | (static_cast<uint32_t>(displacement) & 0xffff));
  }
  p.push_back(
      0xbefe0414u); // restore full EXEC for inactive-destination observation
  if (c.phased)
    p.push_back(0xbf8a0000u);
  add(p, 5, 0, c.local * 3);
  p.insert(p.end(), {0xe0702000u, 0x80020405u, 0xbf810000u});
  return p;
}
inline prosper::gpu::ShaderResourceTable resources(const Case &c) {
  prosper::gpu::ShaderResourceTable rt;
  prosper::gpu::ShaderResource r;
  r.cls = prosper::gpu::ResourceClass::ConstantBuffer;
  r.format = prosper::gpu::DataFormat::Uint32;
  r.num_components = 1;
  r.binding = 3;
  r.stride = 4;
  r.sgpr_base = 8;
  r.size = c.local * 16;
  rt.resources.push_back(r);
  return rt;
}
inline std::vector<uint32_t> compile(const Case &c, uint32_t native = 0) {
  const auto p = guest(c);
  const auto rt = resources(c);
  prosper::gpu::ComputeShaderConfig cfg;
  cfg.local_x = c.local;
  cfg.wave_size = c.wave;
  cfg.native_subgroup_size = native;
  cfg.exact_thread_extent = c.threads != c.local;
  cfg.threads_x = c.threads;
  cfg.threads_y = cfg.threads_z = 1;
  return prosper::gpu::recompile_compute(
      p.data(), p.size(), &rt, cfg,
      {prosper::gpu::RecompileDiagnosticStage::Compute, 0x4089u});
}
inline std::vector<Case> cases() {
  std::vector<Case> v;
  for (uint32_t wave : {32u, 64u})
    for (uint32_t selector = 0; selector < 4; ++selector)
      for (uint32_t offset : {0u, 4u})
        v.push_back({64, wave, 64, offset, selector});
  for (uint32_t wave : {32u, 64u})
    for (uint32_t mask = 1; mask <= 4; ++mask)
      v.push_back({128, wave, 128, 4, 0, mask});
  for (uint32_t local : {1u, 16u, 33u, 65u, 96u, 256u})
    v.push_back({local, 64, local, 4, 1});
  for (uint32_t threads : {1u, 16u, 33u, 65u})
    v.push_back({96, 64, threads, 0, 0});
  for (uint32_t shape = 0; shape < 4; ++shape) {
    Case c{128, 64, 128, 4, 0, 4};
    c.loop = shape == 0;
    c.second = shape == 1;
    if (c.second)
      c.mask = 1; // the second gather must consume live nonzero first-site data
    c.phased = shape == 2;
    c.in_place = shape == 3;
    v.push_back(c);
  }
  Case diverged{256, 64, 256, 0, 0, 1};
  diverged.loop = diverged.different_waves = true;
  v.push_back(diverged);
  diverged.different_sites = true;
  v.push_back(diverged);
  return v;
}
} // namespace prosper::test::bpermute
