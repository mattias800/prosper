#include "gpu/recompiler/fragment_packet_services.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <bitset>
#include <tuple>

namespace prosper::gpu {
namespace {
bool modifiers(const Rdna2Inst& in) {
    return in.has_modifier || in.has_sdwa || in.has_dpp || in.clamp || in.omod ||
        std::any_of(std::begin(in.src_abs), std::end(in.src_abs), [](bool v) { return v; }) ||
        std::any_of(std::begin(in.src_neg), std::end(in.src_neg), [](bool v) { return v; });
}
bool normal_or_zero(uint32_t bits) {
    const auto mag = bits & 0x7fffffffu;
    return !mag || (mag >= 0x00800000u && mag < 0x7f800000u);
}
uint32_t scalar_width(const Rdna2Inst& in, uint32_t source) {
    if (in.fmt == Rdna2Format::SMEM) return source == 0 ? 4 : 1;
    if (in.fmt == Rdna2Format::MIMG) return source == 1 ? 8 : source == 2 ? 4 : 1;
    return scalar_alu_source_words(in, source);
}
const char* image_gap(const FragmentPacketImageRead& image) {
    const auto& t = image.descriptor;
    const uint32_t width = ((t[1] >> 30) | ((t[2] & 0xfffu) << 2)) + 1;
    const uint32_t height = ((t[2] >> 14) & 0x3fffu) + 1;
    const uint32_t last = (t[3] >> 16) & 15u;
    // Exact first T# domain: linear plain 2D, base view 0, identity RGBA32F, no compression,
    // pitch/array/performance/unknown controls. A normalized default is NOT descriptor authority.
    if (!(t[0] || (t[1] & 255u)) || (t[1] & 0x3ff00000u) != (77u << 20) ||
        (t[1] & 0x000fff00u) || (t[2] & 0xf0003000u) ||
        t[3] != (0x90000facu | (last << 16)) || t[4] || t[5] != (last << 4) || t[6] || t[7])
        return "packet-image-descriptor-domain-unimplemented";
    if (!width || !height ||
        (width & (width - 1)) || (height & (height - 1)) ||
        image.mips.size() != last + 1)
        return "packet-image-mip-domain-unimplemented";
    uint32_t extent = std::max(width, height), levels = 0;
    do { ++levels; extent >>= 1; } while (extent);
    if (last >= levels) return "packet-image-mip-domain-unimplemented";
    // AMD 70648 Table46: mip_filter=0 DISABLES mipmapping, 1=POINT, 2=LINEAR.
    // Do not inherit apply_sampler_descriptor's nonzero->linear collapse. Here XY and mip POINT,
    // CLAMP_LAST_TEXEL on all axes, zero bias/minLOD, exact last-level maxLOD, no hidden controls.
    if (image.sampler != std::array<uint32_t, 4>{0xdbu, (last * 256u) << 12, 1u << 26, 0u})
        return "packet-sampler-domain-unimplemented";
    for (uint32_t level = 0; level <= last; ++level) {
        const auto& mip = image.mips[level];
        const auto w = std::max(1u, width >> level), h = std::max(1u, height >> level);
        if (mip.width != w || mip.height != h || mip.texels.size() != w * h)
            return "packet-image-owned-mip-unavailable";
        for (const auto& texel : mip.texels)
            for (const auto bits : texel)
                // Nearest R32F data: no conversion/NaN/denormal/signed-zero sampling claim.
                if (bits == 0x80000000u || !normal_or_zero(bits))
                    return "packet-image-texel-domain-unimplemented";
    }
    return nullptr;
}
} // namespace

const char* packet_resource_instruction_gap(const Rdna2Inst& in) {
    if (modifiers(in)) return "packet-resource-modifier-unimplemented";
    if (in.fmt == Rdna2Format::VINTRP) {
        // RDNA2 70648 12.11: P1 destination/source alias is not independent of HALF_LDS mode.
        // This API owns parameter tuples but does not supply that launch-mode authority.
        if (in.opcode == 0 && in.dst.value == in.src[0].value)
            return "packet-parameter-p1-alias-mode-unavailable";
        return in.opcode <= 1 && in.vintrp_attr < 32 && in.src[0].kind == OperandKind::VGPR
            ? nullptr : "packet-parameter-form-unimplemented";
    }
    if (in.fmt == Rdna2Format::SMEM) {
        if (in.opcode >= 8 && in.opcode <= 10 &&
            (in.dst.value & ((1u << (in.opcode - 8)) - 1u)))
            return "packet-smem-destination-alignment-invalid";
        if (in.opcode < 8 || in.opcode > 10 || in.src[0].kind != OperandKind::SGPR ||
            (in.src[0].value & 3) || in.src[0].value > 102 || in.dst.value > 102 ||
            in.src[1].kind != OperandKind::Special || in.src[1].value != 125 ||
            (in.literal & 3u) || in.literal > 0xfffffu ||
            // Plain GFX10 load controls only; DLC/GLC/IMM/reserved bits must not be ignored.
            (in.words[0] & 0x0003e000u) || (in.words[1] & 0x01e00000u))
            return "packet-smem-form-unimplemented";
        return nullptr;
    }
    if (in.fmt == Rdna2Format::MIMG)
        return (in.opcode == 0x27 || in.opcode == 0x24) && in.mimg_dim == 1 &&
            in.mimg_dmask == 15 && !in.mimg_nsa && !in.mimg_unorm && !in.mimg_a16 &&
            !in.mimg_d16 && !in.mimg_glc && !in.mimg_dlc && !in.mimg_r128 &&
            !in.mimg_tfe && !in.mimg_lwe && !in.mimg_slc && !in.mimg_reserved &&
            in.dst.value <= 252 && in.src[1].value <= 98 && in.src[2].value <= 102
            ? nullptr : "packet-image-explicit-lod-form-unimplemented";
    if (in.fmt == Rdna2Format::VOP2 && (in.opcode == 3 || in.opcode == 8))
        return nullptr;
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 && in.dst.value == 124)
        return in.src[0].kind == OperandKind::SGPR || in.src[0].kind == OperandKind::InlineInt ||
               in.src[0].kind == OperandKind::Literal ? nullptr : "packet-m0-writer-form-unimplemented";
    if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0c)
        return in.simm16 == 0 ? nullptr : "packet-waitcnt-nonzero-unimplemented";
    return "packet-resource-instruction-unimplemented";
}

const char* packet_resource_preflight(const FragmentResourcePacket& packet,
    const std::vector<Rdna2Inst>& ins, uint32_t& failure_pc) {
    const auto& guest = packet.invocation;
    if (!packet.device.device_identity || !packet.device.shader_int64_enabled)
        return "packet-enabled-int64-device-unavailable";
    if (!guest.float_mode.available || !guest.float_flags.available)
        return "packet-f32-launch-mode-or-flags-unavailable";
    if (!packet.launch_rsrc1.available || !packet.launch_rsrc1.canonical())
        return "packet-f32-launch-rsrc1-unavailable";
    const auto raw = packet.launch_rsrc1.value;
    if (((raw >> 12) & 255u) != guest.float_mode.value ||
        ((raw >> 21) & 1u) != guest.float_flags.dx10_clamp ||
        ((raw >> 23) & 1u) != guest.float_flags.ieee_mode)
        return "packet-f32-launch-rsrc1-association-mismatch";
    if (!guest.float_flags.ieee_mode)
        return "packet-f32-non-ieee-mode-unimplemented";
    if (packet.buffers.size() > 64 || packet.images.size() > 16 ||
        packet.parameter_cache.parameters.size() > 16 * 32 * 4)
        return "packet-resource-input-budget";
    std::map<uint32_t, const FragmentPacketBufferRead*> buffers;
    std::map<uint32_t, const FragmentPacketImageRead*> images;
    uint64_t total_words = 0;
    for (const auto& buffer : packet.buffers) {
        // Raw S_BUFFER load does not consume format/swizzle. This first backing domain admits
        // a byte-sized unstrided V#, exact whole allocation and no unknown resource controls.
        if (!buffers.emplace(buffer.pc, &buffer).second || buffer.words.empty() ||
            buffer.words.size() > 262144 || buffer.descriptor[1] & 0xffff0000u ||
            buffer.descriptor[2] != buffer.words.size() * 4 || buffer.descriptor[3] ||
            !(buffer.descriptor[0] || buffer.descriptor[1]))
            return "packet-buffer-owned-descriptor-unavailable";
        total_words += buffer.words.size();
    }
    if (total_words > 1048576) return "packet-resource-input-budget";
    for (const auto& image : packet.images) {
        if (!images.emplace(image.pc, &image).second) return "packet-image-readpoint-duplicate";
        // Bound owned bytes, not a convenient test dimension. Device image limits are separately
        // checked by the executing owner. Inspect this bound before scanning any texel payload.
        for (const auto& mip : image.mips) {
            if (mip.texels.size() > 262144) return "packet-resource-input-budget";
            total_words += uint64_t(mip.texels.size()) * 4u;
            if (total_words > 1048576) return "packet-resource-input-budget";
        }
        if (const auto* gap = image_gap(image)) return gap;
    }
    if (!images.empty() && !packet.device.rgba32_sfloat_sampled)
        return "packet-enabled-image-format-device-unavailable";
    const bool interpolation = std::any_of(ins.begin(), ins.end(), [](const auto& in) {
        return in.fmt == Rdna2Format::VINTRP;
    });
    if (interpolation) {
        const auto& cache = packet.parameter_cache;
        if (!cache.available || guest.quad_topology != FragmentPacketQuadTopology::ConsecutiveLogicalQuads)
            return "packet-parameter-cache-or-quad-ownership-unavailable";
        if (cache.m0 & 0x80000000u) return "packet-parameter-m0-invalid";
        for (uint32_t quad = 1; quad < 16; ++quad) {
            const bool different = cache.quad_primitive[quad] != cache.quad_primitive[quad - 1];
            if (different != (((cache.m0 >> (16 + quad - 1)) & 1u) != 0))
                return "packet-parameter-m0-primitive-association-unavailable";
        }
        std::set<std::tuple<uint32_t, uint32_t, uint32_t>> keys;
        for (const auto& p : cache.parameters)
            if (p.attribute >= 32 || p.channel >= 4 || !normal_or_zero(p.p0) ||
                !normal_or_zero(p.p10) || !normal_or_zero(p.p20) ||
                !keys.emplace(p.primitive, p.attribute, p.channel).second)
                return "packet-parameter-tuple-invalid";
    }
    std::set<uint32_t> used_buffers, used_images;
    struct Pending { std::bitset<106> scalar; std::bitset<256> vector; bool m0 = false; };
    std::vector<Pending> entry(ins.size());
    std::vector<bool> reachable(ins.size());
    std::map<uint32_t, size_t> index;
    for (size_t i = 0; i < ins.size(); ++i) index.emplace(ins[i].pc, i);
    entry.front().m0 = packet.parameter_cache.entry_m0_available;
    reachable.front() = true;
    const auto meet = [&](size_t next, const Pending& pending) {
        if (!reachable[next]) { entry[next] = pending; reachable[next] = true; }
        else {
            entry[next].scalar |= pending.scalar; entry[next].vector |= pending.vector;
            entry[next].m0 &= pending.m0;
        }
    };
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i]; failure_pc = in.pc;
        // Association is WHOLE-program, even for a dead instruction; completion is all structurally
        // reachable paths. No actual entry EXEC/SCC mask specializes away a hazardous arm.
        if (in.fmt == Rdna2Format::SMEM) {
            const auto found = buffers.find(in.pc);
            if (found == buffers.end()) return "packet-buffer-readpoint-unavailable";
            const auto count = 1u << (in.opcode - 8);
            if (in.literal / 4u + count > found->second->words.size()) return "packet-smem-range-unavailable";
            used_buffers.insert(in.pc);
        }
        if (in.fmt == Rdna2Format::MIMG) {
            if (!images.contains(in.pc)) return "packet-image-readpoint-unavailable";
            used_images.insert(in.pc);
        }
        if (!reachable[i]) continue;
        auto pending = entry[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0c) {
            pending.scalar.reset(); pending.vector.reset();
        } else {
            for (uint32_t source = 0; source < in.n_src; ++source) {
                if (in.fmt == Rdna2Format::EXP && !(in.exp_en & (1u << source))) continue;
                const auto& op = in.src[source];
                if (op.kind == OperandKind::SGPR) {
                    const auto count = scalar_width(in, source);
                    for (uint32_t c = 0; c < count; ++c)
                        if (op.value + c < 106 && pending.scalar.test(op.value + c))
                            return "packet-smem-result-read-before-wait";
                }
                if (op.kind == OperandKind::VGPR) {
                    const auto count = in.fmt == Rdna2Format::MIMG ? (in.opcode == 0x24 ? 3u : 2u) : 1u;
                    for (uint32_t c = 0; c < count; ++c)
                        if (op.value + c < 256 && pending.vector.test(op.value + c))
                            return "packet-image-result-read-before-wait";
                }
            }
            if (in.fmt == Rdna2Format::VINTRP) {
                if (!pending.m0) return "packet-parameter-m0-read-before-definition";
                if (in.opcode == 1 && pending.vector.test(in.dst.value))
                    return "packet-image-result-read-before-wait";
            }
            bool overwrite = false;
            for_each_scalar_write(in, [&](int first, uint32_t count) {
                for (uint32_t c = 0; c < count; ++c)
                    if (first + c < 106 && pending.scalar.test(first + c)) overwrite = true;
            });
            if (overwrite) return "packet-smem-result-overwrite-before-wait";
            for (uint32_t c = 0; c < rdna2_vgpr_write_count(in); ++c)
                if (pending.vector.test(in.dst.value + c)) return "packet-image-result-overwrite-before-wait";
            if (in.fmt == Rdna2Format::SMEM)
                for (uint32_t c = 0; c < (1u << (in.opcode - 8)); ++c) pending.scalar.set(in.dst.value + c);
            if (in.fmt == Rdna2Format::MIMG)
                for (uint32_t c = 0; c < 4; ++c) pending.vector.set(in.dst.value + c);
            if (in.fmt == Rdna2Format::SOP1 && in.dst.value == 124) pending.m0 = true;
        }
        if (in.is_end) {
            if (pending.scalar.any() || pending.vector.any()) return "packet-result-pending-at-end";
            continue;
        }
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        if (branch) meet(index.at(branch_target(in)), pending);
        if ((!branch || in.opcode != 2) && i + 1 < ins.size()) meet(i + 1, pending);
    }
    if (used_buffers.size() != buffers.size() || used_images.size() != images.size())
        return "packet-resource-readpoint-unused";
    failure_pc = UINT32_MAX;
    return nullptr;
}
} // namespace prosper::gpu
