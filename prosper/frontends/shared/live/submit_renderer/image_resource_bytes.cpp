#include "shared/live/submit_renderer/image_resources.hpp"
#include "shared/live/submit_renderer/guest_reads.hpp"
#include "shared/live/decode_scratch.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <cstring>

namespace prosper::frontend::submit_renderer {
// clang-format off: preserve the byte-identical moved definitions.

size_t copy_shader_resource(const prosper::gpu::ShaderResource& r, uint8_t* dst, uint64_t addr, size_t n) {
              if (!r.host_data) return safe_copy(dst, addr, n);
              if (addr < r.gpu_addr) return 0;
              uint64_t off = addr - r.gpu_addr;
              if (off >= r.host_data_size) return 0;
              size_t take = static_cast<size_t>(std::min<uint64_t>(n, r.host_data_size - off));
              std::memcpy(dst, r.host_data + off, take);
              return take;
}

const uint8_t* direct_shader_resource_source(const prosper::gpu::ShaderResource& r, const bool& no_direct_texture_source, uint64_t addr, size_t n) {
              if (!n || no_direct_texture_source) return nullptr;
              if (r.host_data) {
                  // Mirrors copy_resource's host_data arm: the same bounds, and `take == n`.
                  if (addr < r.gpu_addr) return nullptr;
                  const uint64_t off = addr - r.gpu_addr;
                  if (off >= r.host_data_size || n > r.host_data_size - off) return nullptr;
                  return r.host_data + off;
              }
              if (addr < 0x1000 || addr > UINT64_MAX - n) return nullptr;
              return safe_span(addr, n) == n
                  ? reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(addr))
                  : nullptr;
}

const uint8_t* stage_tiled_shader_resource(const prosper::gpu::ShaderResource& r, const bool& no_direct_texture_source, prosper::frontend::DecodeScratchPool::Lease& lease, uint64_t addr, size_t n, size_t& got) {
    const auto direct_resource_source = [&](uint64_t addr, size_t n) {
        return direct_shader_resource_source(r, no_direct_texture_source, addr, n);
    };
    const auto copy_resource = [&](uint8_t* dst, uint64_t addr, size_t n) {
        return copy_shader_resource(r, dst, addr, n);
    };
              if (const uint8_t* direct = direct_resource_source(addr, n)) {
                  got = n;
                  return direct;
              }
              lease = prosper::frontend::decode_scratch_pool().take(n);
              got = copy_resource(lease.data(), addr, n);
              lease.zero_tail(got);   // restores what `std::vector<uint8_t>(n, 0)` gave
              return lease.data();
}

size_t copy_shader_dcc_metadata(const prosper::gpu::ShaderResource& r, uint8_t* dst, size_t n) {
              if (!r.dcc_metadata_host_data)
                  return safe_copy(dst, r.metadata_addr, n);
              const size_t take = static_cast<size_t>(std::min<uint64_t>(
                  n, r.dcc_metadata_host_data_size));
              std::memcpy(dst, r.dcc_metadata_host_data, take);
              return take;
}
// clang-format on

} // namespace prosper::frontend::submit_renderer
