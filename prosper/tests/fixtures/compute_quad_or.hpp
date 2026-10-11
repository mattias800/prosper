#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <atomic>
#include <vector>

namespace prosper::test::compute_quad_or {
using namespace prosper::gpu;
inline constexpr uint32_t lanes = 128, sentinel = 0xdeadbeefu;
enum class Shape {
    Plain,
    Complement,
    MissingArm,
    NewGeneration,
    Absent,
    Partial,
    Cyclic,
    Masked,
    DistinctSites,
    TooManyGenerations
};
inline std::vector<uint32_t> program(Shape shape = Shape::Plain) {
    std::vector<uint32_t> p;
    p.reserve(80);
    const auto vop2 = [&](uint32_t op, uint32_t dst, uint32_t a, uint32_t v) {
        p.push_back((op << 25u) | (dst << 17u) | (v << 9u) | a);
    };
    const auto literal = [&](uint32_t dst, uint32_t value) {
        p.insert(p.end(), {0x7e0002ffu | (dst << 17u), value});
    };
    const auto branch = [&](uint32_t op) {
        const auto at = p.size();
        p.push_back(0xbf800000u | (op << 16u));
        return at;
    };
    const auto target = [&](size_t at, size_t to) {
        p[at] |= uint16_t(int32_t(to) - int32_t(at + 1));
    };
    const auto quad = [&](uint32_t ctrl) {
        p.insert(p.end(), {0x380000fau | (3u << 17u) | (2u << 9u), 0xff080001u | (ctrl << 8u)});
    };
    literal(6, 17);
    vop2(0x25, 1, 262, 0); // v1 = local_id + 17
    literal(6, 257);
    vop2(0x25, 2, 262, 0); // SRC1 is distinct, unpermuted
    literal(6, 4096);
    vop2(0x25, 3, 262, 0); // genuine old destination
    if (shape == Shape::Absent) p.push_back(0x7e02035au); // v1 = absent v90
    if (shape == Shape::Partial) {
        // Partial replacement of an absent word must not invent the other half.
        p.push_back(0x7e02035au);
        p.insert(p.end(), {0x7e0202f9u, 0x00051480u}); // WORD0 / UNUSED_PRESERVE
    }
    if (shape == Shape::Complement || shape == Shape::MissingArm || shape == Shape::NewGeneration) {
        // Kill old validity before deriving complementary, instruction-order lane definitions.
        p.push_back(0x7e02035au);
        literal(8, 32);
        p.push_back(0x7da21100u);   // CMPX v0 < v8
        const auto empty = branch(8);
        literal(6, 17);
        vop2(0x25, 1, 262, 0);
        target(empty, p.size());
        if (shape == Shape::NewGeneration) {
            p.push_back(0xbefe04c1u);
            literal(8, 64);
            p.push_back(0x7da21100u);   // distinct larger mask generation
        }
        p.push_back(0xbefe087eu);   // NOT EXEC, EXEC
        const auto complement_empty = branch(8);
        if (shape != Shape::MissingArm) literal(1, 512);
        target(complement_empty, p.size());
        p.push_back(0xbefe04c1u);   // restore FULL
    }
    if (shape == Shape::TooManyGenerations)
        for (uint32_t i = 0; i < 9; ++i) p.insert(p.end(), {0x7da40e80u, 0xbefe04c1u});
    if (shape == Shape::Masked) {
        vop2(0x1b, 7, 129, 0);   // v7 = lane & 1
        p.push_back(0x7da40e80u);   // CMPX v7 == 0
    }
    size_t loop = 0;
    if (shape == Shape::Cyclic) {
        p.push_back(0xbe800382u);   // s0=2
        loop = p.size();
    }
    if (shape == Shape::DistinctSites) {
        // Native READFIRSTLANE's legacy lane-local substitution is exact for this genuine
        // wave-uniform source. Divergent READFIRSTLANE is a separate lowering gap (#153).
        vop2(0x16, 7, 134, 0);   // v7 = local_id >> 6
        p.push_back(0x7e000507u);   // READFIRSTLANE s0,v7
        p.push_back(0xbf0a8100u);   // s_cmp_lt_u32 s0,1 (each wave takes one site)
        const auto alternate = branch(4);
        quad(0xb1);
        const auto joined = branch(2);
        target(alternate, p.size());
        quad(0x4e);
        target(joined, p.size());
    } else {
        // The final masked permutation must select inactive peers; otherwise an earlier
        // incorrect inactive-peer result is overwritten before the output can observe it.
        quad(shape == Shape::Masked ? 0x4e : 0xb1);
        quad(shape == Shape::Masked ? 0xb1 : 0x4e);
    }
    if (shape == Shape::Cyclic) {
        p.insert(p.end(), {0x80808100u, 0xbf068000u});
        target(branch(4), loop);
    }
    if (shape == Shape::Masked) p.push_back(0xbefe04c1u);
    p.insert(p.end(), {0xe0702000u, 0x80020300u, 0xbf810000u});   // RAW store v3 at local_id
    return p;
}
inline ComputeShaderConfig config(uint32_t native = 0, uint32_t local = lanes) {
    ComputeShaderConfig c;
    c.local_x = c.threads_x = local;
    c.local_y = c.local_z = c.threads_y = c.threads_z = 1;
    c.wave_size = 64;
    c.native_subgroup_size = native;
    return c;
}
inline ShaderResourceTable table(uint32_t count = lanes) {
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.binding = 3;
    r.sgpr_base = 8;
    r.stride = 4;
    r.size = count * 4;
    ShaderResourceTable t;
    t.resources.push_back(r);
    return t;
}
inline uint64_t address() {
    static std::atomic<uint64_t> next{0x4925000000ull};
    return next.fetch_add(0x100, std::memory_order_relaxed);
}
inline std::vector<uint32_t> compile(const std::vector<uint32_t>& p, const ShaderResourceTable& t,
                                     const ComputeShaderConfig& c, uint64_t addr) {
    return recompile_compute(p.data(), p.size(), &t, c, {RecompileDiagnosticStage::Compute, addr});
}
inline std::vector<uint32_t> expected(Shape shape, uint32_t count = lanes) {
    std::vector<uint32_t> value(count), source(count), own(count);
    for (uint32_t lane = 0; lane < count; ++lane) {
        value[lane] = lane + 4096;
        source[lane] = shape == Shape::Complement && lane >= 32 ? 512u : lane + 17;
        own[lane] = lane + 257;
    }
    const std::array<uint32_t, 2> controls =
        shape == Shape::Masked ? std::array{0x4eu, 0xb1u} : std::array{0xb1u, 0x4eu};
    for (uint32_t control : controls) {
        const auto& previous = source;
        for (uint32_t lane = 0; lane < count; ++lane) {
            if (shape == Shape::Masked && lane % 2) continue;
            if (shape == Shape::DistinctSites && ((lane < 64) != (control == 0xb1))) continue;
            const uint32_t peer = (lane & ~3u) | ((control >> (2 * (lane & 3))) & 3u);
            const uint32_t payload =
                peer < count && (shape != Shape::Masked || peer % 2 == 0) ? previous[peer] : 0;
            value[lane] = payload | own[lane];
        }
        // SRC0 and SRC1 are not the destination: later sites still read the original inputs.
    }
    return value;
}
}   // namespace prosper::test::compute_quad_or
