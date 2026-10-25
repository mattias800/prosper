#include "gpu/recompiler/rdna2_native_image_results.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"

#include <bit>
#include <cstdint>
#include <utility>

namespace prosper::gpu {
namespace {
bool source_proved(const RegState& rs, const Rdna2Inst& in) {
    if (!rs.vector_native_interpolation || rs.vector_wave_size != 64 || !rs.exec ||
        !rs.exec_narrowed || in.fmt != Rdna2Format::MIMG || in.opcode != 0xa0 || in.mimg_dim != 1 ||
        !in.mimg_dmask || (in.mimg_dmask & ~15u) || in.mimg_d16 || in.mimg_tfe || in.mimg_lwe ||
        in.mimg_r128 || in.mimg_reserved || in.has_modifier || in.has_dpp || in.has_sdwa ||
        in.len_dwords < 2 || in.len_dwords > in.words.size() || in.dst.kind != OperandKind::VGPR ||
        in.dst.value < 0 || in.dst.value > 255)
        return false;
    auto decoded = rdna2_decode_one(in.words.data(), in.len_dwords);
    decoded.pc = in.pc;
    const auto& source = rs.mimg_sources;
    return decoded == in && source.pc == in.pc && source.enabled && source.defined &&
           !source.words.empty() && arbitrary_read_resource_genuine(rs, in);
}
} // namespace

void record_native_image_result(RegState& rs, const Rdna2Inst& in) {
    rs.native_image_result_publication = {};
    if (!source_proved(rs, in)) return;
    NativeImageResultPublication publication;
    publication.pc = in.pc;
    publication.exec = rs.exec;
    publication.destination = in.dst.value;
    publication.count = static_cast<uint32_t>(std::popcount(in.mimg_dmask));
    if (static_cast<uint32_t>(in.dst.value) + publication.count > 256) return;
    for (uint32_t word = 0; word < publication.count; ++word) {
        const auto value = rs.vreg.find(in.dst.value + static_cast<int>(word));
        if (value == rs.vreg.end() || !value->second) return;
        publication.values[word] = value->second;
    }
    rs.native_image_result_publication = publication;
}

bool consume_native_image_result(RegState& rs, const Rdna2Inst& in) {
    if (!rs.native_image_result_publication.exec) return false;
    const auto publication = std::exchange(rs.native_image_result_publication, {});
    if (publication.pc != in.pc || publication.exec != rs.exec ||
        publication.destination != in.dst.value || !source_proved(rs, in) ||
        publication.count != static_cast<uint32_t>(std::popcount(in.mimg_dmask)) ||
        publication.count > publication.values.size())
        return false;
    for (uint32_t word = 0; word < publication.count; ++word) {
        const auto value = rs.vreg.find(in.dst.value + static_cast<int>(word));
        if (value == rs.vreg.end() || !value->second || value->second != publication.values[word])
            return false;
    }
    return true;
}
} // namespace prosper::gpu
