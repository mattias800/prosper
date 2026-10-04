#include "gpu/recompiler/fragment_draw_gpu.hpp"
#include "gpu/recompiler/fragment_draw_spirv_flow.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>

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
// A driver may run one 2x2 quad of a primitive as several helper-backed invocations whose
// nonhelper lanes are disjoint (llvmpipe does; RADV does not, #4277). This reassembles such split
// scopes IN PLACE in the collector buffer, before anything counts or reads its records, with the
// same contract as decode_raster_quad_records on the CPU path: a later record whose key (lane-0
// primitive and origin) matches an earlier one is folded into it, each lane taken from the record
// that ran it as a nonhelper, and removed; survivors keep first-occurrence order and the header
// count is rewritten. Overlapping nonhelper lanes, a lane that is a helper in both but disagrees,
// a helper-only record or any malformed lane (helper/coverage/mask words, primitive or facing
// identity, quad topology) refuses the whole collection and zeroes the count.
//
// Cost is bounded: keys live in a linear-probing hash table in the commit plane's pixel slots
// (zero-filled before the transaction, >= 8x max_quads entries, so the load factor is <= 1/8),
// probing at most kFragmentDrawPixelProbeLimit slots per record; an exhausted probe refuses.
// Every slot written is zeroed again before validation reuses the plane. A key names one
// triangle only while each PrimitiveId does. Two guards make that hold today: a draw with more
// than one instance is refused (PrimitiveId restarts per instance;
// tests/fixtures/fragment_draw_backend_transaction.h), and so is any interpolation layout that
// requires a geometry stage, which could emit several triangles per input primitive
// (input_free_layout in fragment_draw_plan.cpp). Lifting either needs the instance index or the
// geometry stage's own primitive id in the key.
// Returns a function variable holding 1 when the collection was refused.
uint32_t reassemble_split_scopes(SpirvCompute& b, const Words& source, const Words& scratch,
                                 const FragmentDrawCapacity& capacity,
                                 const RasterQuadCollector& collector, uint32_t primitives) {
    using namespace fragment_draw_flow;
    uint32_t pointer_type = 0;
    const auto failed = b.function_var(b.t_u32, pointer_type);
    const auto out = b.function_var(b.t_u32, pointer_type);
    const auto merged = b.function_var(b.t_u32, pointer_type);
    const auto mismatch = b.function_var(b.t_u32, pointer_type);
    const auto probe = b.function_var(b.t_u32, pointer_type);
    const auto found = b.function_var(b.t_u32, pointer_type);
    const auto resolved = b.function_var(b.t_u32, pointer_type);
    b.store_function(failed, b.uconst(0));
    b.store_function(out, b.uconst(0));
    const auto ok = [&] {
        return b.ucmp(Op_IEqual, b.load_function(b.t_u32, failed), b.uconst(0));
    };
    const auto fail_if = [&](uint32_t condition) {
        b.store_function(failed, b.sel(condition, b.uconst(1), b.load_function(b.t_u32, failed)));
    };
    const auto lane_words = collector.lane_words, record_words = collector.record_words;
    const auto word = [&](uint32_t record, uint32_t offset) {
        return b.ibin(Op_IAdd, b.uconst(kRasterQuadBufferHeaderWords),
                      b.ibin(Op_IAdd, b.ibin(Op_IMul, record, b.uconst(record_words)), offset));
    };
    const auto lane_word = [&](uint32_t record, uint32_t lane, uint32_t field) {
        return word(record, b.uconst(lane * lane_words + field));
    };
    // Nonhelper lane mask, refusing exactly the lanes the CPU decoder refuses.
    const auto checked_mask = [&](uint32_t record) {
        const auto facing = source.load(lane_word(record, 0, 3));
        const auto primitive = source.load(lane_word(record, 0, 4));
        const auto x = pixel_center(b, source.load(lane_word(record, 0, 5)), b.uconst(8192));
        const auto y = pixel_center(b, source.load(lane_word(record, 0, 6)), b.uconst(8192));
        auto mask = b.uconst(0);
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const auto helper = source.load(lane_word(record, lane, 0));
            const auto nonhelper = b.ucmp(Op_IEqual, helper, b.uconst(0));
            auto bad = b.ucmp(Op_UGreaterThan, helper, b.uconst(1));
            bad = b.lor(bad, b.ucmp(Op_INotEqual, source.load(lane_word(record, lane, 1)),
                                    b.sel(nonhelper, b.uconst(1), b.uconst(0))));
            bad = b.lor(bad, b.land(nonhelper, b.ucmp(Op_INotEqual,
                                                      source.load(lane_word(record, lane, 2)),
                                                      b.uconst(1))));
            bad = b.lor(bad, b.ucmp(Op_UGreaterThan, source.load(lane_word(record, lane, 3)),
                                    b.uconst(1)));
            bad = b.lor(bad, b.ucmp(Op_INotEqual, source.load(lane_word(record, lane, 3)), facing));
            bad = b.lor(bad, b.ucmp(Op_INotEqual, source.load(lane_word(record, lane, 4)),
                                    primitive));
            bad = b.lor(bad, b.ucmp(Op_UGreaterThanEqual, primitive, primitives));
            // Quad topology on decoded pixel indices (integer-only, as validation decodes them):
            // lane = origin + (lane & 1, lane >> 1). Helper lanes may lie outside the target, so
            // only the encoding range bounds them here.
            const auto lx = pixel_center(b, source.load(lane_word(record, lane, 5)), b.uconst(8192));
            const auto ly = pixel_center(b, source.load(lane_word(record, lane, 6)), b.uconst(8192));
            bad = b.lor(bad, b.logical_not(b.land(lx.valid, ly.valid)));
            bad = b.lor(bad, b.ucmp(Op_INotEqual, lx.pixel,
                                    b.ibin(Op_IAdd, x.pixel, b.uconst(lane & 1))));
            bad = b.lor(bad, b.ucmp(Op_INotEqual, ly.pixel,
                                    b.ibin(Op_IAdd, y.pixel, b.uconst(lane >> 1))));
            fail_if(bad);
            mask = b.ibin(Op_BitwiseOr, mask, b.sel(nonhelper, b.uconst(1u << lane), b.uconst(0)));
        }
        fail_if(b.ucmp(Op_IEqual, mask, b.uconst(0)));
        return mask;
    };
    const auto slots = capacity.pixel_slots();
    // Hash DECODED pixel indices, as validation does: raw half-integer float bits have nearly
    // constant low bits and cluster every key into a few probe chains.
    const auto home = [&](uint32_t record) {
        const auto x = pixel_center(b, source.load(lane_word(record, 0, 5)), b.uconst(8192));
        const auto y = pixel_center(b, source.load(lane_word(record, 0, 6)), b.uconst(8192));
        const auto a = b.ibin(Op_IMul, source.load(lane_word(record, 0, 4)), b.uconst(0x9e3779b1u));
        const auto c = b.ibin(Op_IMul, x.pixel, b.uconst(0x85ebca6bu));
        const auto d = b.ibin(Op_IMul, y.pixel, b.uconst(0xc2b2ae35u));
        return b.ibin(Op_BitwiseAnd, b.ibin(Op_BitwiseXor, a, b.ibin(Op_BitwiseXor, c, d)),
                      b.uconst(slots - 1));
    };
    const auto table = [&](uint32_t slot) {
        return b.ibin(Op_IAdd, b.uconst(kFragmentDrawCommitHeaderWords), slot);
    };
    const auto same_key = [&](uint32_t a, uint32_t c) {
        auto same = b.btrue();
        for (uint32_t field : {4u, 5u, 6u})
            same = b.land(same, b.ucmp(Op_IEqual, source.load(lane_word(a, 0, field)),
                                       source.load(lane_word(c, 0, field))));
        return same;
    };
    const auto count = source.load(b.uconst(0));
    auto header = b.ucmp(Op_IEqual, source.load(b.uconst(2)), b.uconst(record_words));
    header = b.land(header, b.ucmp(Op_IEqual, source.load(b.uconst(3)), b.uconst(kRasterQuadMagic)));
    header = b.land(header, b.ucmp(Op_ULessThanEqual, count, b.uconst(capacity.max_quads())));
    header = b.land(header, b.ucmp(Op_IEqual, source.load(b.uconst(1)), b.uconst(0)));
    const auto run = begin_if(b, header);
    bounded_loop(b, count, [&](uint32_t record) {
        const auto live = begin_if(b, ok());   // stop working once the collection is refused
        const auto mask = checked_mask(record);
        // Probe for an earlier record with this key: found -> its kept index + 1 in `found`;
        // otherwise `probe` ends on the empty slot this record will claim.
        b.store_function(probe, home(record));
        b.store_function(found, b.uconst(0));
        b.store_function(resolved, b.uconst(0));
        bounded_loop(b, b.uconst(kFragmentDrawPixelProbeLimit), [&](uint32_t) {
            const auto open =
                begin_if(b, b.ucmp(Op_IEqual, b.load_function(b.t_u32, resolved), b.uconst(0)));
            const auto slot = b.load_function(b.t_u32, probe);
            const auto value = scratch.load(table(slot));
            const auto empty = b.ucmp(Op_IEqual, value, b.uconst(0));
            const auto safe = b.sel(empty, b.uconst(0), b.ibin(Op_ISub, value, b.uconst(1)));
            const auto hit = b.land(b.logical_not(empty), same_key(safe, record));
            b.store_function(found, b.sel(hit, value, b.uconst(0)));
            b.store_function(resolved, b.sel(b.lor(empty, hit), b.uconst(1), b.uconst(0)));
            const auto next = b.ibin(Op_BitwiseAnd, b.ibin(Op_IAdd, slot, b.uconst(1)),
                                     b.uconst(slots - 1));
            b.store_function(probe, b.sel(b.lor(empty, hit), slot, next));
            end_if(b, open);
        });
        fail_if(b.ucmp(Op_IEqual, b.load_function(b.t_u32, resolved), b.uconst(0)));
        const auto fold = begin_if(
            b, b.land(ok(), b.ucmp(Op_INotEqual, b.load_function(b.t_u32, found), b.uconst(0))));
        const auto earlier = b.ibin(Op_ISub, b.load_function(b.t_u32, found), b.uconst(1));
        const auto earlier_mask = checked_mask(earlier);
        fail_if(b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, mask, earlier_mask), b.uconst(0)));
        b.store_function(mismatch, b.uconst(0));
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const auto bit = b.uconst(1u << lane);
            const auto take = b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, mask, bit), b.uconst(0));
            const auto helper_in_both = b.land(
                b.logical_not(take),
                b.ucmp(Op_IEqual, b.ibin(Op_BitwiseAnd, earlier_mask, bit), b.uconst(0)));
            bounded_loop(b, b.uconst(lane_words), [&](uint32_t field) {
                const auto offset = b.ibin(Op_IAdd, b.uconst(lane * lane_words), field);
                const auto from = source.load(word(record, offset));
                const auto into = source.load(word(earlier, offset));
                source.store(word(earlier, offset), b.sel(take, from, into));
                const auto differs = b.land(helper_in_both, b.ucmp(Op_INotEqual, from, into));
                b.store_function(mismatch, b.sel(differs, b.uconst(1),
                                                 b.load_function(b.t_u32, mismatch)));
            });
        }
        fail_if(b.ucmp(Op_INotEqual, b.load_function(b.t_u32, mismatch), b.uconst(0)));
        end_if(b, fold);
        // Not folded: keep the record at the next survivor slot (never past its own position)
        // and claim the empty table slot the probe ended on.
        const auto keep = begin_if(
            b, b.land(ok(), b.ucmp(Op_IEqual, b.load_function(b.t_u32, found), b.uconst(0))));
        const auto kept = b.load_function(b.t_u32, out);
        const auto move = begin_if(b, b.ucmp(Op_INotEqual, kept, record));
        bounded_loop(b, b.uconst(record_words), [&](uint32_t offset) {
            source.store(word(kept, offset), source.load(word(record, offset)));
        });
        end_if(b, move);
        scratch.store(table(b.load_function(b.t_u32, probe)), b.ibin(Op_IAdd, kept, b.uconst(1)));
        b.store_function(out, b.ibin(Op_IAdd, kept, b.uconst(1)));
        end_if(b, keep);
        end_if(b, live);
    });
    // Return the borrowed slots to zero, then publish the compacted count (zero on refusal, so no
    // later stage can read a partially folded collection as records).
    bounded_loop(b, b.load_function(b.t_u32, out), [&](uint32_t kept) {
        b.store_function(probe, home(kept));
        b.store_function(resolved, b.uconst(0));
        bounded_loop(b, b.uconst(kFragmentDrawPixelProbeLimit), [&](uint32_t) {
            const auto open =
                begin_if(b, b.ucmp(Op_IEqual, b.load_function(b.t_u32, resolved), b.uconst(0)));
            const auto slot = b.load_function(b.t_u32, probe);
            const auto mine = b.ucmp(Op_IEqual, scratch.load(table(slot)),
                                     b.ibin(Op_IAdd, kept, b.uconst(1)));
            const auto clear = begin_if(b, mine);
            scratch.store(table(slot), b.uconst(0));
            end_if(b, clear);
            b.store_function(resolved, b.sel(mine, b.uconst(1), b.uconst(0)));
            b.store_function(probe, b.ibin(Op_BitwiseAnd, b.ibin(Op_IAdd, slot, b.uconst(1)),
                                           b.uconst(slots - 1)));
            end_if(b, open);
        });
    });
    source.store(b.uconst(0), b.sel(ok(), b.load_function(b.t_u32, out), b.uconst(0)));
    end_if(b, run);
    return failed;
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
    // The collector is writable here only: split scopes are reassembled in place (#4277), using
    // the zero-filled commit plane's pixel slots as a bounded scratch table (binding 4).
    const Words scratch{b, b.id()};
    b.declare_external_storage_buffer(b.t_ptr_sb_struct_u, scratch.variable);
    b.put(b.deco, Op_Decorate, {scratch.variable, Dec_DescriptorSet, b.desc_set});
    b.put(b.deco, Op_Decorate, {scratch.variable, Dec_Binding, 4});
    readonly(b, authority.variable);
    readonly(b, entry.variable);
    auto lengths = b.land(b.ucmp(Op_IEqual, source.length(), b.uconst(capacity.collector_words())),
                          b.ucmp(Op_IEqual, packet.length(), b.uconst(capacity.input_words())));
    lengths = b.land(lengths,
                     b.ucmp(Op_IEqual, authority.length(), b.uconst(kFragmentDrawAuthorityWords)));
    lengths = b.land(lengths, b.ucmp(Op_IEqual, entry.length(),
                                     b.uconst(4 + capacity.kernel()->layout.input_words)));
    lengths = b.land(lengths, b.ucmp(Op_IEqual, scratch.length(), b.uconst(capacity.commit_words())));
    for (const auto component : b.groupid)
        lengths = b.land(lengths, b.ucmp(Op_IEqual, component, b.uconst(0)));
    const auto outer = selection(b, lengths);
    const auto refused = reassemble_split_scopes(b, source, scratch, capacity, collector,
                                                 entry.load(b.uconst(0)));
    const auto reassembled =
        b.ucmp(Op_IEqual, b.load_function(b.t_u32, refused), b.uconst(0));
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
    const auto good = b.land(b.land(valid, b.logical_not(overflow)), reassembled);
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
                    b.sel(reassembled, b.uconst(0),
                          b.uconst(uint32_t(FragmentDrawFailure::CollectionRecord)))),
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
    const bool raster_entry = !capacity.raster_inputs().empty();
    // Old full-buffer/parameter payloads are not per-wave packing authority. The distinct scalar
    // bank stays outside this input plane; only its genuine user descriptor words travel here.
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
    auto any_live = b.bfalse();
    for (uint32_t lane = 0; lane < 4; ++lane) {
        record_valid = b.land(record_valid, b.land(b.ucmp(Op_IEqual, read(lane, 4), primitive),
                                                   b.ucmp(Op_IEqual, read(lane, 3), facing)));
        if (raster_entry) {
            const auto helper = read(lane, 0);
            const auto live = b.ucmp(Op_IEqual, helper, b.uconst(0));
            any_live = b.lor(any_live, live);
            record_valid = b.land(record_valid, b.ucmp(Op_ULessThanEqual, helper, b.uconst(1)));
            record_valid = b.land(record_valid, b.ucmp(Op_IEqual, read(lane, 1),
                                                       b.sel(live, b.uconst(1), b.uconst(0))));
            record_valid =
                b.land(record_valid,
                       b.lor(b.logical_not(live), b.ucmp(Op_IEqual, read(lane, 2), b.uconst(1))));
        } else {
            record_valid =
                b.land(record_valid, b.land(b.ucmp(Op_IEqual, read(lane, 1), b.uconst(1)),
                                            b.ucmp(Op_IEqual, read(lane, 2), b.uconst(1))));
        }
        helpers_absent = b.land(helpers_absent, b.ucmp(Op_IEqual, read(lane, 0), b.uconst(0)));
    }
    if (raster_entry) record_valid = b.land(record_valid, any_live);
    const auto bad = selection(
        b, b.land(occupied, b.logical_not(raster_entry ? record_valid
                                                       : b.land(record_valid, helpers_absent))));
    const auto ignored = b.id();
    const auto reason =
        raster_entry
            ? b.uconst(uint32_t(FragmentDrawFailure::CollectionRecord))
            : b.sel(helpers_absent, b.uconst(uint32_t(FragmentDrawFailure::CollectionRecord)),
                    b.uconst(uint32_t(FragmentDrawFailure::HelperEntryUnavailable)));
    b.put(b.code, Op_AtomicCompareExchange,
          {b.t_u32, ignored, packet.address(b.uconst(8)), b.uconst(Scope_Device), b.uconst(0),
           b.uconst(0), reason, b.uconst(0)});
    end_selection(b, bad);
    // Raw storage is cleared on-device before assembly. Zero scratch is NEVER made available.
    // Post-PS export eligibility is separate from guest EXEC and only granted by this recipe's
    // disabled-DS, single-sample, genuine nonhelper contract. The old recipe keeps initial
    // masks absent; the distinct raster recipe supplies its independently certified live mask.
    const auto n = static_cast<uint32_t>(layout.vgprs.size());
    const bool validity = kernel.program.packet.input_stride == 2 * n + 4;
    const auto lane_base =
        b.ibin(Op_IAdd, data_base,
               b.ibin(Op_IMul, b.linear_localid, b.uconst(kernel.program.packet.input_stride)));
    if (raster_entry) {
        const auto quad_lane = b.ibin(Op_UMod, b.linear_localid, b.uconst(4));
        const auto lane_address =
            b.ibin(Op_IAdd, origin, b.ibin(Op_IMul, quad_lane, b.uconst(collector.lane_words)));
        const auto live =
            b.land(occupied, b.ucmp(Op_IEqual, source.load(lane_address), b.uconst(0)));
        for (uint32_t column = 0; column < n; ++column) {
            const auto found =
                std::find_if(capacity.raster_inputs().begin(), capacity.raster_inputs().end(),
                             [&](const auto& row) { return row.reg == layout.vgprs[column]; });
            if (found == capacity.raster_inputs().end()) continue;
            // QuadBroadcast retained this exact helper invocation's actual FragCoord BEFORE
            // the nonhelper store/election. Helpers do not publish their own SSBO stores.
            const auto value =
                source.load(b.ibin(Op_IAdd, lane_address, b.uconst(found->collector_word)));
            packet.store(b.ibin(Op_IAdd, lane_base, b.uconst(column)), value);
            packet.store(b.ibin(Op_IAdd, lane_base, b.uconst(n + column)),
                         b.sel(occupied, b.uconst(1), b.uconst(0)));
        }
        // Independent facts: storage/backing includes all four genuine lanes; initial EXEC
        // and final attachment eligibility include only the recipe-certified live pixels.
        // WQM may expand EXEC, but cannot expand this final eligibility field.
        packet.store(b.ibin(Op_IAdd, lane_base, b.uconst(2 * n)),
                     b.sel(live, b.uconst(1), b.uconst(0)));
        packet.store(b.ibin(Op_IAdd, lane_base, b.uconst(2 * n + 3)),
                     b.sel(live, b.uconst(1), b.uconst(0)));
    } else {
        packet.store(b.ibin(Op_IAdd, lane_base, b.uconst(n * (validity ? 2u : 1u) + 3)),
                     b.sel(occupied, b.uconst(1), b.uconst(0)));
    }
    end_selection(b, inner);
    end_selection(b, outer);
    return b.finish();
}
}   // namespace prosper::gpu
