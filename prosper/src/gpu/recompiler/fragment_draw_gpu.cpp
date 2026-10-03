#include "gpu/recompiler/fragment_draw_gpu.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
namespace {
struct Words {
    SpirvCompute& b;
    uint32_t variable;
    uint32_t address(uint32_t index) const {
        const auto pointer = b.id();
        b.put(b.code, Op_AccessChain, {b.t_ptr_sb_u32, pointer, variable, b.uconst(0), index});
        return pointer;
    }
    uint32_t load(uint32_t index) const {
        const auto value = b.id();
        b.put(b.code, Op_Load, {b.t_u32, value, address(index)});
        return value;
    }
    uint32_t length() const {
        const auto value = b.id();
        b.put(b.code, 68, {b.t_u32, value, variable, 0});
        return value;
    }
    void store(uint32_t index, uint32_t value) const {
        b.put(b.code, Op_Store, {address(index), value});
    }
};
void readonly(SpirvCompute& b, uint32_t variable) {
    b.put(b.deco, Op_Decorate, {variable, 24});
}
uint32_t selection(SpirvCompute& b, uint32_t predicate) {
    const auto body = b.id(), end = b.id();
    b.put(b.code, Op_SelectionMerge, {end, 0});
    b.put(b.code, Op_BranchConditional, {predicate, body, end});
    b.emit_label(body);
    return end;
}
void end_selection(SpirvCompute& b, uint32_t end) {
    b.emit_branch(end);
    b.emit_label(end);
}
uint32_t authority_valid(SpirvCompute& b, const Words& authority,
                         const FragmentDrawCapacity& capacity) {
    auto valid = b.btrue();
    for (uint32_t offset = 0; offset < kFragmentDrawAuthorityWords; ++offset)
        valid = b.land(valid, b.ucmp(Op_IEqual, authority.load(b.uconst(offset)),
                                     b.uconst(capacity.authority()[offset])));
    return valid;
}
} // namespace

std::vector<uint32_t> build_fragment_draw_count(const FragmentDrawCapacity& capacity,
                                                const RasterQuadCollector& collector) {
    if (!capacity.matches_collector(collector)) return {};
    SpirvCompute b;
    b.begin(0, nullptr, 1, 1, 1, 64, 0, true, true);
    const Words source{b, b.v_in}, packet{b, b.v_out}, authority{b, b.v_cbuf}, entry{b, b.v_cbuf1};
    readonly(b, source.variable);
    readonly(b, authority.variable);
    readonly(b, entry.variable);
    auto lengths = b.land(b.ucmp(Op_IEqual, source.length(), b.uconst(capacity.collector_words())),
                          b.ucmp(Op_IEqual, packet.length(), b.uconst(capacity.input_words())));
    lengths = b.land(lengths,
                     b.ucmp(Op_IEqual, authority.length(), b.uconst(kFragmentDrawAuthorityWords)));
    lengths = b.land(lengths, b.ucmp(Op_IEqual, entry.length(),
                                     b.uconst(4 + capacity.kernel()->layout.input_words)));
    for (const auto component : b.groupid)
        lengths = b.land(lengths, b.ucmp(Op_IEqual, component, b.uconst(0)));
    const auto outer = selection(b, lengths);
    const auto count = source.load(b.uconst(0));
    const auto primitives = entry.load(b.uconst(0)), width = entry.load(b.uconst(1)),
               height = entry.load(b.uconst(2));
    auto valid = authority_valid(b, authority, capacity);
    valid = b.land(valid,
                   b.ucmp(Op_IEqual, source.load(b.uconst(2)), b.uconst(collector.record_words)));
    valid = b.land(valid, b.ucmp(Op_IEqual, source.load(b.uconst(3)), b.uconst(kRasterQuadMagic)));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, count, b.uconst(capacity.max_quads())));
    for (const auto dimension : {width, height})
        valid = b.land(valid, b.land(b.ucmp(Op_UGreaterThan, dimension, b.uconst(0)),
                                     b.ucmp(Op_ULessThanEqual, dimension, b.uconst(8192))));
    valid = b.land(valid, b.land(b.ucmp(Op_UGreaterThan, primitives, b.uconst(0)),
                                 b.ucmp(Op_IEqual, entry.load(b.uconst(3)),
                                        b.uconst(kFragmentDrawInputMagic))));
    const auto overflow = b.ucmp(Op_INotEqual, source.load(b.uconst(1)), b.uconst(0));
    const auto good = b.land(valid, b.logical_not(overflow));
    const auto quads = b.sel(good, count, b.uconst(0));
    const auto waves = b.ibin(Op_UDiv, b.ibin(Op_IAdd, quads, b.uconst(15)), b.uconst(16));
    const uint32_t values[kFragmentDrawHeaderWords] = {
        b.uconst(kFragmentDrawInputMagic),
        waves,
        b.uconst(capacity.input_words()),
        b.uconst(capacity.output_words()),
        quads,
        waves,
        b.uconst(1),
        b.uconst(1),
        b.sel(valid,
              b.sel(overflow, b.uconst(uint32_t(FragmentDrawFailure::CollectionOverflow)),
                    b.uconst(0)),
              b.uconst(uint32_t(FragmentDrawFailure::CollectionHeader))),
        width,
        height,
        primitives};
    for (uint32_t offset = 0; offset < kFragmentDrawHeaderWords; ++offset)
        packet.store(b.uconst(offset), values[offset]);
    end_selection(b, outer);
    return b.finish();
}

std::vector<uint32_t> build_fragment_draw_assembly(const FragmentDrawCapacity& capacity,
                                                   const RasterQuadCollector& collector) {
    if (!capacity.matches_collector(collector)) return {};
    const auto& kernel = *capacity.kernel();
    const auto& layout = kernel.layout;
    // This first recipe has no external read-PC or guest parameter coefficients. Adding those
    // needs producer/epoch ownership, not copying observed host varyings into presumed registers.
    if (!layout.buffers.empty() || !layout.parameters.empty() || !kernel.program.images.empty() ||
        layout.entry_m0_available)
        return {};
    SpirvCompute b;
    b.begin(0, nullptr, 64, 1, 1, 64, 0, true, true);
    const Words source{b, b.v_in}, packet{b, b.v_out}, authority{b, b.v_cbuf}, entry{b, b.v_cbuf1};
    readonly(b, source.variable);
    readonly(b, authority.variable);
    readonly(b, entry.variable);
    auto lengths = b.land(b.ucmp(Op_IEqual, source.length(), b.uconst(capacity.collector_words())),
                          b.ucmp(Op_IEqual, packet.length(), b.uconst(capacity.input_words())));
    lengths =
        b.land(lengths,
               b.land(b.ucmp(Op_IEqual, authority.length(), b.uconst(kFragmentDrawAuthorityWords)),
                      b.ucmp(Op_IEqual, entry.length(), b.uconst(4 + layout.input_words))));
    const auto outer = selection(b, lengths);
    const auto quads = packet.load(b.uconst(4)), count = packet.load(b.uconst(1));
    auto valid = authority_valid(b, authority, capacity);
    valid = b.land(valid,
                   b.ucmp(Op_IEqual, packet.load(b.uconst(0)), b.uconst(kFragmentDrawInputMagic)));
    // Word8 becomes an atomic failure accumulator during THIS dispatch. Do not race plain
    // loads against another workgroup's fault publication. The preceding count phase represents
    // a refused collection by count0; the later original-kernel phase reads word8 after a barrier.
    valid = b.land(valid, b.ucmp(Op_UGreaterThan, entry.load(b.uconst(0)), b.uconst(0)));
    valid = b.land(valid,
                   b.ucmp(Op_IEqual, entry.load(b.uconst(3)), b.uconst(kFragmentDrawInputMagic)));
    for (uint32_t field = 0; field < 3; ++field)
        valid = b.land(valid, b.ucmp(Op_IEqual, entry.load(b.uconst(field)),
                                     packet.load(b.uconst(field == 0 ? 11 : field + 8))));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, quads, b.uconst(capacity.max_quads())));
    valid =
        b.land(valid, b.ucmp(Op_IEqual, count,
                             b.ibin(Op_UDiv, b.ibin(Op_IAdd, quads, b.uconst(15)), b.uconst(16))));
    valid = b.land(valid, b.land(b.ucmp(Op_ULessThan, b.groupid[0], count),
                                 b.land(b.ucmp(Op_IEqual, b.groupid[1], b.uconst(0)),
                                        b.ucmp(Op_IEqual, b.groupid[2], b.uconst(0)))));
    const auto inner = selection(b, valid);   // whole-WG uniform, before any address/worker effect
    const auto wave_base = b.ibin(Op_IAdd, b.uconst(kFragmentDrawHeaderWords),
                                  b.ibin(Op_IMul, b.groupid[0], b.uconst(capacity.input_span())));
    const auto data_base = b.ibin(Op_IAdd, wave_base, b.uconst(2));
    const auto leader = selection(b, b.ucmp(Op_IEqual, b.linear_localid, b.uconst(0)));
    packet.store(wave_base, b.uconst(kPacketWaveInputMagic));
    packet.store(b.ibin(Op_IAdd, wave_base, b.uconst(1)), b.groupid[0]);
    for (size_t scalar = 0; scalar < layout.sgprs.size(); ++scalar) {
        for (uint32_t offset :
             {layout.scalar_offsets[scalar], layout.scalar_available_offsets[scalar]})
            packet.store(b.ibin(Op_IAdd, data_base, b.uconst(offset)),
                         entry.load(b.uconst(4 + offset)));
    }
    // Absence is explicit. A cached program demanding real entry M0 cannot enter this recipe.
    packet.store(b.ibin(Op_IAdd, data_base, b.uconst(layout.entry_m0_available_offset)),
                 b.uconst(0));
    end_selection(b, leader);
    const auto quad = b.ibin(Op_IAdd, b.ibin(Op_IMul, b.groupid[0], b.uconst(16)),
                             b.ibin(Op_UDiv, b.linear_localid, b.uconst(4)));
    const auto occupied = b.ucmp(Op_ULessThan, quad, quads);
    const auto safe_quad = b.sel(occupied, quad, b.uconst(0));
    const auto origin = b.ibin(Op_IAdd, b.uconst(kRasterQuadBufferHeaderWords),
                               b.ibin(Op_IMul, safe_quad, b.uconst(collector.record_words)));
    const auto read = [&](uint32_t lane, uint32_t word) {
        return source.load(b.ibin(Op_IAdd, origin, b.uconst(lane * collector.lane_words + word)));
    };
    const auto primitive = read(0, 4), facing = read(0, 3);
    auto record_valid = b.land(b.ucmp(Op_ULessThan, primitive, entry.load(b.uconst(0))),
                               b.ucmp(Op_ULessThanEqual, facing, b.uconst(1)));
    auto helpers_absent = b.btrue();
    for (uint32_t lane = 0; lane < 4; ++lane) {
        record_valid = b.land(record_valid, b.land(b.ucmp(Op_IEqual, read(lane, 4), primitive),
                                                   b.ucmp(Op_IEqual, read(lane, 3), facing)));
        record_valid = b.land(record_valid, b.land(b.ucmp(Op_IEqual, read(lane, 1), b.uconst(1)),
                                                   b.ucmp(Op_IEqual, read(lane, 2), b.uconst(1))));
        helpers_absent = b.land(helpers_absent, b.ucmp(Op_IEqual, read(lane, 0), b.uconst(0)));
    }
    const auto bad =
        selection(b, b.land(occupied, b.logical_not(b.land(record_valid, helpers_absent))));
    const auto ignored = b.id();
    const auto reason =
        b.sel(helpers_absent, b.uconst(uint32_t(FragmentDrawFailure::CollectionRecord)),
              b.uconst(uint32_t(FragmentDrawFailure::HelperEntryUnavailable)));
    b.put(b.code, Op_AtomicCompareExchange,
          {b.t_u32, ignored, packet.address(b.uconst(8)), b.uconst(Scope_Device), b.uconst(0),
           b.uconst(0), reason, b.uconst(0)});
    end_selection(b, bad);
    // Raw storage is cleared on-device before assembly. Zero scratch is NEVER made available.
    // Post-PS export eligibility is separate from guest EXEC and only granted by this recipe's
    // disabled-DS, single-sample, genuine nonhelper contract. Initial mask values remain absent.
    const auto n = static_cast<uint32_t>(layout.vgprs.size());
    const bool validity = kernel.program.packet.input_stride == 2 * n + 4;
    const auto lane_base =
        b.ibin(Op_IAdd, data_base,
               b.ibin(Op_IMul, b.linear_localid, b.uconst(kernel.program.packet.input_stride)));
    packet.store(b.ibin(Op_IAdd, lane_base, b.uconst(n * (validity ? 2u : 1u) + 3)),
                 b.sel(occupied, b.uconst(1), b.uconst(0)));
    end_selection(b, inner);
    end_selection(b, outer);
    return b.finish();
}
}   // namespace prosper::gpu
