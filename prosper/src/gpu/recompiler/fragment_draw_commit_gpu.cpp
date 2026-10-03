#include "gpu/recompiler/fragment_draw_gpu.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
namespace {
// Internal raw storage for the new transaction only. No descriptor fallback or aliased mutable
// route table is used; the immutable capacity fixes every extent before a shader is cached.
struct DrawWords {
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
    void store(uint32_t index, uint32_t value) const {
        b.put(b.code, Op_Store, {address(index), value});
    }
    uint32_t length() const {
        const auto result = b.id();
        b.put(b.code, 68, {b.t_u32, result, variable, 0});
        return result;
    }
};

DrawWords buffer(SpirvCompute& b, uint32_t binding, bool writable = false) {
    const auto variable = b.id();
    b.declare_external_storage_buffer(b.t_ptr_sb_struct_u, variable);
    b.put(b.deco, Op_Decorate, {variable, Dec_DescriptorSet, b.desc_set});
    b.put(b.deco, Op_Decorate, {variable, Dec_Binding, binding});
    if (!writable) b.put(b.deco, Op_Decorate, {variable, 24});
    return {b, variable};
}

uint32_t begin_if(SpirvCompute& b, uint32_t condition) {
    const auto body = b.id(), end = b.id();
    b.emit_selmerge(end);
    b.emit_condbranch(condition, body, end);
    b.emit_label(body);
    return end;
}
void end_if(SpirvCompute& b, uint32_t end) {
    b.emit_branch(end);
    b.emit_label(end);
}
void discard_unless(SpirvCompute& b, uint32_t condition) {
    const auto end = begin_if(b, b.logical_not(condition));
    b.put(b.code, Op_Kill, {});
    b.emit_label(end);
}

template <typename Body>
void bounded_loop(SpirvCompute& b, uint32_t count, Body body) {
    uint32_t pointer_type = 0;
    const auto index = b.function_var(b.t_u32, pointer_type);
    b.store_function(index, b.uconst(0));
    const auto header = b.id(), active = b.id(), next = b.id(), end = b.id();
    b.emit_branch(header);
    b.emit_label(header);
    const auto value = b.load_function(b.t_u32, index);
    const auto condition = b.ucmp(Op_ULessThan, value, count);
    b.emit_loopmerge(end, next);
    b.emit_condbranch(condition, active, end);
    b.emit_label(active);
    body(value);
    b.emit_branch(next);
    b.emit_label(next);
    b.store_function(index, b.ibin(Op_IAdd, value, b.uconst(1)));
    b.emit_branch(header);
    b.emit_label(end);
}

uint32_t capacity_matches(SpirvCompute& b, const DrawWords& authority,
                          const FragmentDrawCapacity& capacity) {
    auto valid = b.btrue();
    for (uint32_t field = 0; field < kFragmentDrawAuthorityWords; ++field)
        valid = b.land(valid, b.ucmp(Op_IEqual, authority.load(b.uconst(field)),
                                     b.uconst(capacity.authority()[field])));
    return valid;
}

// This first attachment recipe is deliberately single-MRT/uncompressed. This restriction belongs
// to the replay recipe, not the logical64 compiler or full-support goal. The live admission owner
// must additionally require the Architectural export policy once that dependency is accepted;
// LegacyRaw is retained here solely for same-code CPU wire/fault controls.
uint32_t export_recipe(const FragmentDrawCapacity& capacity, Rdna2Inst& site) {
    const auto& kernel = *capacity.kernel();
    uint32_t sites = 0;
    for (const auto& instruction : kernel.instructions)
        if (instruction.fmt == Rdna2Format::EXP) {
            site = instruction;
            ++sites;
        }
    if (sites != 1 || site.exp_target != 0 || site.exp_en != 15 || site.exp_compr ||
        !(site.words[0] & (1u << 11)) || !(site.words[0] & (1u << 12)) ||
        kernel.program.packet.exports_per_lane != 1 || kernel.program.status_offset % 64)
        return 0;
    const auto words = kernel.program.status_offset / 64;
    if (words != kFragmentPacketExportWords && words != 14) return 0;
    const auto validity_words = kernel.program.packet.vgpr_status_offset == UINT32_MAX
                                    ? 0u
                                    : 64 * kFragmentPacketVgprStatusWords;
    if (kernel.layout.output_words !=
            64 * words + 64 * kFragmentResourceStatusWords + validity_words ||
        (validity_words && kernel.program.packet.vgpr_status_offset !=
                               kernel.program.status_offset + 64 * kFragmentResourceStatusWords))
        return 0;
    return words;
}

uint32_t collector_address(SpirvCompute& b, const RasterQuadCollector& collector, uint32_t record,
                           uint32_t field) {
    return b.ibin(Op_IAdd, b.uconst(kRasterQuadBufferHeaderWords + field),
                  b.ibin(Op_IMul, record, b.uconst(collector.lane_words)));
}
uint32_t pixel_hash(SpirvCompute& b, uint32_t primitive, uint32_t x, uint32_t y,
                    const FragmentDrawCapacity& capacity) {
    const auto a = b.ibin(Op_IMul, primitive, b.uconst(0x9e3779b1u));
    const auto c = b.ibin(Op_IMul, x, b.uconst(0x85ebca6bu));
    const auto d = b.ibin(Op_IMul, y, b.uconst(0xc2b2ae35u));
    return b.ibin(Op_BitwiseAnd, b.ibin(Op_BitwiseXor, a, b.ibin(Op_BitwiseXor, c, d)),
                  b.uconst(capacity.pixel_slots() - 1));
}
struct PixelCoordinate {
    uint32_t valid, pixel;
};
PixelCoordinate pixel_center(SpirvCompute& b, uint32_t raw, uint32_t extent) {
    // IEEE binary32 encoding of positive i+0.5, for 0<=i<8192. This is a transport/index
    // check on a host raster observation, not guest floating arithmetic or interpolation.
    // Integer decoding avoids float-to-integer undefined behavior on malformed NaNs/infinities.
    const auto exponent =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, raw, b.uconst(23)), b.uconst(255));
    auto valid = b.land(b.ucmp(Op_UGreaterThanEqual, exponent, b.uconst(126)),
                        b.ucmp(Op_ULessThanEqual, exponent, b.uconst(139)));
    valid = b.land(
        valid, b.ucmp(Op_IEqual, b.ibin(Op_BitwiseAnd, raw, b.uconst(0x80000000u)), b.uconst(0)));
    const auto safe_exponent = b.sel(valid, exponent, b.uconst(126));
    const auto shift = b.ibin(Op_ISub, b.uconst(150), safe_exponent);   // checked 11..24
    const auto significand =
        b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, raw, b.uconst(0x7fffffu)), b.uconst(0x800000u));
    const auto fraction_mask =
        b.ibin(Op_ISub, b.ibin(Op_ShiftLeftLogical, b.uconst(1), shift), b.uconst(1));
    const auto half = b.ibin(Op_ShiftLeftLogical, b.uconst(1), b.ibin(Op_ISub, shift, b.uconst(1)));
    const auto pixel = b.ibin(Op_ShiftRightLogical, significand, shift);
    valid =
        b.land(valid, b.ucmp(Op_IEqual, b.ibin(Op_BitwiseAnd, significand, fraction_mask), half));
    valid = b.land(valid, b.ucmp(Op_ULessThan, pixel, extent));
    return {valid, pixel};
}
}   // namespace

std::vector<uint32_t> build_fragment_draw_validation(const FragmentDrawCapacity& capacity,
                                                     const RasterQuadCollector& collector) {
    if (!capacity.matches_collector(collector)) return {};
    Rdna2Inst export_site{};
    const auto export_words = export_recipe(capacity, export_site);
    if (!export_words || collector.record_words != collector.lane_words * 4) return {};
    const auto& program = capacity.kernel()->program;
    SpirvCompute b;
    // One device invocation owns the entire bounded transaction/index. O(workers * 32) maximum,
    // no all-wave quadratic scan, no cross-workgroup publication race, no CPU synchronization.
    // Parallel validators can replace this owner later without weakening the all-draw commit gate.
    b.begin(0, nullptr, 1, 1, 1, 64, 0, true, true);
    const DrawWords input{b, b.v_in}, commit{b, b.v_out}, authority{b, b.v_cbuf},
        output{b, b.v_cbuf1};
    const auto source = buffer(b, 4);
    for (const auto variable : {input.variable, authority.variable, output.variable})
        b.put(b.deco, Op_Decorate, {variable, 24});
    auto lengths = b.btrue();
    for (const auto pair : {std::pair{input.length(), capacity.input_words()},
                            std::pair{commit.length(), capacity.commit_words()},
                            std::pair{authority.length(), kFragmentDrawAuthorityWords},
                            std::pair{output.length(), capacity.output_words()},
                            std::pair{source.length(), capacity.collector_words()}})
        lengths = b.land(lengths, b.ucmp(Op_IEqual, pair.first, b.uconst(pair.second)));
    for (const auto component : b.groupid)
        lengths = b.land(lengths, b.ucmp(Op_IEqual, component, b.uconst(0)));
    const auto outer =
        begin_if(b, lengths);   // BEFORE even the first header or pointer dereference
    const auto count = input.load(b.uconst(1)), quads = input.load(b.uconst(4));
    const auto width = input.load(b.uconst(9)), height = input.load(b.uconst(10)),
               primitives = input.load(b.uconst(11));
    auto valid = capacity_matches(b, authority, capacity);
    valid = b.land(valid,
                   b.ucmp(Op_IEqual, input.load(b.uconst(0)), b.uconst(kFragmentDrawInputMagic)));
    valid =
        b.land(valid, b.ucmp(Op_IEqual, input.load(b.uconst(2)), b.uconst(capacity.input_words())));
    valid = b.land(valid,
                   b.ucmp(Op_IEqual, input.load(b.uconst(3)), b.uconst(capacity.output_words())));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, quads, b.uconst(capacity.max_quads())));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, count, b.uconst(capacity.max_waves())));
    valid =
        b.land(valid, b.ucmp(Op_IEqual, count,
                             b.ibin(Op_UDiv, b.ibin(Op_IAdd, quads, b.uconst(15)), b.uconst(16))));
    valid = b.land(valid, b.ucmp(Op_IEqual, source.load(b.uconst(0)), quads));
    valid = b.land(valid, b.ucmp(Op_IEqual, source.load(b.uconst(1)), b.uconst(0)));
    valid = b.land(valid,
                   b.ucmp(Op_IEqual, source.load(b.uconst(2)), b.uconst(collector.record_words)));
    valid = b.land(valid, b.ucmp(Op_IEqual, source.load(b.uconst(3)), b.uconst(kRasterQuadMagic)));
    for (const auto dimension : {width, height})
        valid = b.land(valid, b.land(b.ucmp(Op_UGreaterThan, dimension, b.uconst(0)),
                                     b.ucmp(Op_ULessThanEqual, dimension, b.uconst(8192))));
    valid = b.land(valid, b.ucmp(Op_UGreaterThan, primitives, b.uconst(0)));
    valid = b.land(valid, b.ucmp(Op_IEqual, input.load(b.uconst(5)), count));
    valid = b.land(valid, b.ucmp(Op_IEqual, input.load(b.uconst(6)), b.uconst(1)));
    valid = b.land(valid, b.ucmp(Op_IEqual, input.load(b.uconst(7)), b.uconst(1)));
    uint32_t pointer_type = 0;
    const auto failure = b.function_var(b.t_u32, pointer_type);
    b.store_function(failure, b.sel(valid, input.load(b.uconst(8)),
                                    b.uconst(uint32_t(FragmentDrawFailure::CollectionHeader))));
    const auto note = [&](uint32_t bad, FragmentDrawFailure reason) {
        const auto previous = b.load_function(b.t_u32, failure);
        b.store_function(failure, b.sel(b.land(bad, b.ucmp(Op_IEqual, previous, b.uconst(0))),
                                        b.uconst(uint32_t(reason)), previous));
    };
    const auto scan = begin_if(b, valid);
    const auto pixels = b.ibin(Op_IMul, quads, b.uconst(4));
    bounded_loop(b, b.ibin(Op_IMul, count, b.uconst(64)), [&](uint32_t worker) {
        const auto wave = b.ibin(Op_UDiv, worker, b.uconst(64));
        const auto lane = b.ibin(Op_UMod, worker, b.uconst(64));
        const auto base = b.ibin(Op_IMul, wave, b.uconst(capacity.output_span()));
        const auto raw = b.ibin(Op_IAdd, base, b.uconst(kPacketWaveOutputPrefix));
        const auto prefix = b.ibin(Op_IAdd, base, b.ibin(Op_IMul, lane, b.uconst(2)));
        auto completed =
            b.land(b.ucmp(Op_IEqual, output.load(prefix), b.uconst(kPacketWaveOutputMagic)),
                   b.ucmp(Op_IEqual, output.load(b.ibin(Op_IAdd, prefix, b.uconst(1))), wave));
        note(b.logical_not(completed), FragmentDrawFailure::WorkerCompletion);
        const auto status = b.ibin(
            Op_IAdd, raw,
            b.ibin(Op_IAdd, b.uconst(program.status_offset), b.ibin(Op_IMul, lane, b.uconst(3))));
        auto success =
            b.land(b.ucmp(Op_IEqual, output.load(status), b.uconst(kFragmentResourceStatusMagic)),
                   b.land(b.ucmp(Op_IEqual, output.load(b.ibin(Op_IAdd, status, b.uconst(1))),
                                 b.uconst(UINT32_MAX)),
                          b.ucmp(Op_IEqual, output.load(b.ibin(Op_IAdd, status, b.uconst(2))),
                                 b.uconst(0))));
        if (program.packet.vgpr_status_offset != UINT32_MAX) {
            const auto validity =
                b.ibin(Op_IAdd, raw,
                       b.ibin(Op_IAdd, b.uconst(program.packet.vgpr_status_offset),
                              b.ibin(Op_IMul, lane, b.uconst(4))));
            const uint32_t fields[] = {kFragmentPacketVgprStatusMagic, UINT32_MAX, UINT32_MAX, 0};
            for (uint32_t field = 0; field < 4; ++field)
                success =
                    b.land(success, b.ucmp(Op_IEqual,
                                           output.load(b.ibin(Op_IAdd, validity, b.uconst(field))),
                                           b.uconst(fields[field])));
        }
        note(b.logical_not(success), FragmentDrawFailure::GuestRuntime);
        const auto record = b.ibin(Op_IAdd, raw, b.ibin(Op_IMul, lane, b.uconst(export_words)));
        const auto value = [&](uint32_t field) {
            return output.load(b.ibin(Op_IAdd, record, b.uconst(field)));
        };
        const auto reached = b.ucmp(Op_IEqual, value(0), b.uconst(1));
        const auto occupied = b.ucmp(Op_ULessThan, worker, pixels);
        auto export_valid = b.ucmp(Op_ULessThanEqual, value(0), b.uconst(1));
        const uint32_t fields[] = {0, 0, 0, 0, 15, 0, 1, 1};
        for (uint32_t field = 1; field < 8; ++field) {
            const auto expected =
                field == 2 ? b.sel(occupied, b.uconst(1), b.uconst(0)) : b.uconst(fields[field]);
            const auto matching = field == 1 ? b.ucmp(Op_ULessThanEqual, value(field), b.uconst(1))
                                             : b.ucmp(Op_IEqual, value(field), expected);
            export_valid =
                b.land(export_valid, b.lor(b.land(reached, matching),
                                           b.land(b.logical_not(reached),
                                                  b.ucmp(Op_IEqual, value(field), b.uconst(0)))));
        }
        for (uint32_t field = 8; field < export_words; ++field) {
            auto matching = b.btrue();
            if (field == 12) matching = b.ucmp(Op_IEqual, value(field), b.uconst(export_site.pc));
            if (field == 13)
                matching = b.ucmp(
                    Op_IEqual, value(field),
                    b.sel(b.ucmp(Op_IEqual, value(1), b.uconst(1)), b.uconst(15), b.uconst(0)));
            if (export_words == 14 && field < 12)
                matching = b.lor(b.ucmp(Op_IEqual, value(1), b.uconst(1)),
                                 b.ucmp(Op_IEqual, value(field), b.uconst(0)));
            export_valid =
                b.land(export_valid, b.lor(b.land(reached, matching),
                                           b.land(b.logical_not(reached),
                                                  b.ucmp(Op_IEqual, value(field), b.uconst(0)))));
        }
        note(b.logical_not(export_valid), FragmentDrawFailure::GuestExport);
        const auto active = begin_if(b, occupied);
        const auto coordinate = [&](uint32_t field) {
            return source.load(collector_address(b, collector, worker, field));
        };
        const auto x = coordinate(5), y = coordinate(6), primitive = coordinate(4);
        const auto px = pixel_center(b, x, width), py = pixel_center(b, y, height);
        auto topology = b.land(px.valid, py.valid);
        topology = b.land(topology, b.ucmp(Op_ULessThan, primitive, primitives));
        const auto quad_start = b.ibin(Op_IMul, b.ibin(Op_UDiv, worker, b.uconst(4)), b.uconst(4));
        const auto quad_lane = b.ibin(Op_UMod, worker, b.uconst(4));
        // Revalidate the original host-observation provenance independently of assembly. Even a
        // correctly completed guest kernel cannot authorize a helper, absent coverage or a
        // collector scope whose primitive/facing identity changed before the whole-draw gate.
        const auto helper = coordinate(0), coverage_available = coordinate(1),
                   coverage = coordinate(2), facing = coordinate(3);
        const auto backed = b.land(b.ucmp(Op_IEqual, helper, b.uconst(0)),
                                   b.ucmp(Op_IEqual, coverage_available, b.uconst(1)));
        note(b.logical_not(backed), FragmentDrawFailure::HelperEntryUnavailable);
        auto provenance = b.land(backed, b.ucmp(Op_IEqual, coverage, b.uconst(1)));
        provenance = b.land(provenance, b.ucmp(Op_ULessThanEqual, facing, b.uconst(1)));
        provenance =
            b.land(provenance, b.ucmp(Op_IEqual, facing,
                                      source.load(collector_address(b, collector, quad_start, 3))));
        provenance =
            b.land(provenance, b.ucmp(Op_IEqual, primitive,
                                      source.load(collector_address(b, collector, quad_start, 4))));
        topology = b.land(topology, provenance);
        const auto origin_x =
            pixel_center(b, source.load(collector_address(b, collector, quad_start, 5)), width);
        const auto origin_y =
            pixel_center(b, source.load(collector_address(b, collector, quad_start, 6)), height);
        topology = b.land(topology, b.land(origin_x.valid, origin_y.valid));
        topology = b.land(topology, b.ucmp(Op_IEqual, px.pixel,
                                           b.ibin(Op_IAdd, origin_x.pixel,
                                                  b.ibin(Op_BitwiseAnd, quad_lane, b.uconst(1)))));
        topology =
            b.land(topology, b.ucmp(Op_IEqual, py.pixel,
                                    b.ibin(Op_IAdd, origin_y.pixel,
                                           b.ibin(Op_ShiftRightLogical, quad_lane, b.uconst(1)))));
        note(b.logical_not(topology), FragmentDrawFailure::CollectionRecord);
        const auto indexable = begin_if(b, topology);
        uint32_t slot_type = 0;
        const auto slot_var = b.function_var(b.t_u32, slot_type);
        const auto found_var = b.function_var(b.t_u32, slot_type);
        b.store_function(slot_var, pixel_hash(b, primitive, px.pixel, py.pixel, capacity));
        b.store_function(found_var, b.uconst(0));
        bounded_loop(b, b.uconst(kFragmentDrawPixelProbeLimit), [&](uint32_t) {
            const auto searching =
                begin_if(b, b.ucmp(Op_IEqual, b.load_function(b.t_u32, found_var), b.uconst(0)));
            const auto slot = b.load_function(b.t_u32, slot_var);
            const auto address = b.ibin(Op_IAdd, b.uconst(kFragmentDrawCommitHeaderWords), slot);
            const auto previous = commit.load(address);
            const auto empty = begin_if(b, b.ucmp(Op_IEqual, previous, b.uconst(0)));
            commit.store(address, b.ibin(Op_IAdd, worker, b.uconst(1)));
            b.store_function(found_var, b.uconst(1));
            end_if(b, empty);
            const auto taken = begin_if(b, b.ucmp(Op_INotEqual, previous, b.uconst(0)));
            const auto bounded = b.ucmp(Op_ULessThanEqual, previous, pixels);
            const auto other = b.sel(bounded, b.ibin(Op_ISub, previous, b.uconst(1)), b.uconst(0));
            auto same = b.btrue();
            const uint32_t key[] = {primitive, x, y}, key_fields[] = {4, 5, 6};
            for (uint32_t field = 0; field < 3; ++field)
                same = b.land(same, b.ucmp(Op_IEqual, key[field],
                                           source.load(collector_address(b, collector, other,
                                                                         key_fields[field]))));
            note(b.logical_not(bounded), FragmentDrawFailure::PixelIndexCapacity);
            note(b.land(bounded, same), FragmentDrawFailure::DuplicatePixel);
            b.store_function(found_var,
                             b.sel(b.lor(b.logical_not(bounded), same), b.uconst(1), b.uconst(0)));
            end_if(b, taken);
            b.store_function(slot_var, b.ibin(Op_BitwiseAnd, b.ibin(Op_IAdd, slot, b.uconst(1)),
                                              b.uconst(capacity.pixel_slots() - 1)));
            end_if(b, searching);
        });
        note(b.ucmp(Op_IEqual, b.load_function(b.t_u32, found_var), b.uconst(0)),
             FragmentDrawFailure::PixelIndexCapacity);
        end_if(b, indexable);
        end_if(b, active);
    });
    end_if(b, scan);
    const auto reason = b.load_function(b.t_u32, failure);
    commit.store(b.uconst(1), reason);
    commit.store(b.uconst(2), quads);
    commit.store(b.uconst(3), primitives);
    // The subsequent graphics barrier publishes ALL preceding shader writes. No invocation can
    // observe gate1 until every wave/status/export/geometry/index record has been inspected.
    commit.store(b.uconst(0),
                 b.sel(b.ucmp(Op_IEqual, reason, b.uconst(0)), b.uconst(1), b.uconst(0)));
    end_if(b, outer);
    return b.finish();
}

std::vector<uint32_t> build_fragment_draw_replay(const FragmentDrawCapacity& capacity,
                                                 const RasterQuadCollector& collector) {
    if (!capacity.matches_collector(collector)) return {};
    Rdna2Inst export_site{};
    const auto export_words = export_recipe(capacity, export_site);
    if (!export_words) return {};
    SpirvCompute b;
    b.float_transport = capacity.kernel()->transport;
    b.begin_fragment(nullptr, 1);
    b.declare_cbufs();   // raw-u32 Block types and retained authority/source at binding2/3
    // Set1/binding0 belongs to normal persistent GDS. Private replay planes use distinct,
    // complete retained binding1..5 owners and never replace its persistent allocation.
    const auto input = buffer(b, 1), output = buffer(b, 4), commit = buffer(b, 5);
    const DrawWords authority{b, b.v_cbuf}, source{b, b.v_cbuf1};
    for (const auto variable : {authority.variable, source.variable})
        b.put(b.deco, Op_Decorate, {variable, 24});
    auto lengths = b.btrue();
    for (const auto pair : {std::pair{input.length(), capacity.input_words()},
                            std::pair{output.length(), capacity.output_words()},
                            std::pair{authority.length(), kFragmentDrawAuthorityWords},
                            std::pair{source.length(), capacity.collector_words()},
                            std::pair{commit.length(), capacity.commit_words()}})
        lengths = b.land(lengths, b.ucmp(Op_IEqual, pair.first, b.uconst(pair.second)));
    discard_unless(b, lengths);   // No header dereference before actual length checks
    auto valid = capacity_matches(b, authority, capacity);
    valid = b.land(valid, b.ucmp(Op_IEqual, commit.load(b.uconst(0)), b.uconst(1)));
    valid = b.land(valid, b.ucmp(Op_IEqual, commit.load(b.uconst(1)), b.uconst(0)));
    valid = b.land(valid,
                   b.ucmp(Op_IEqual, input.load(b.uconst(0)), b.uconst(kFragmentDrawInputMagic)));
    valid = b.land(valid, b.ucmp(Op_IEqual, input.load(b.uconst(8)), b.uconst(0)));
    const auto quads = input.load(b.uconst(4));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, quads, b.uconst(capacity.max_quads())));
    valid = b.land(valid, b.ucmp(Op_IEqual, quads, commit.load(b.uconst(2))));
    valid = b.land(valid, b.ucmp(Op_IEqual, input.load(b.uconst(11)), commit.load(b.uconst(3))));
    discard_unless(b, valid);
    b.put(b.caps, Op_Capability, {Cap_Geometry});
    const auto primitive_pointer = b.id(), primitive_variable = b.id(), primitive = b.id();
    b.put(b.types, Op_TypePointer, {primitive_pointer, SC_Input, b.t_u32});
    b.put(b.types, Op_Variable, {primitive_pointer, primitive_variable, SC_Input});
    b.put(b.deco, Op_Decorate, {primitive_variable, Dec_BuiltIn, 7});
    b.put(b.deco, Op_Decorate, {primitive_variable, Dec_Flat});
    b.iface.push_back(primitive_variable);
    b.put(b.code, Op_Load, {b.t_u32, primitive, primitive_variable});
    const auto x = b.fragcoord_component(0), y = b.fragcoord_component(1);
    const auto px = pixel_center(b, x, input.load(b.uconst(9))),
               py = pixel_center(b, y, input.load(b.uconst(10)));
    discard_unless(b, b.land(b.logical_not(b.helper_invocation()), b.land(px.valid, py.valid)));
    uint32_t variable_type = 0;
    const auto slot_var = b.function_var(b.t_u32, variable_type);
    const auto record_var = b.function_var(b.t_u32, variable_type);
    b.store_function(slot_var, pixel_hash(b, primitive, px.pixel, py.pixel, capacity));
    b.store_function(record_var, b.uconst(0));
    const auto pixels = b.ibin(Op_IMul, quads, b.uconst(4));
    bounded_loop(b, b.uconst(kFragmentDrawPixelProbeLimit), [&](uint32_t) {
        const auto search =
            begin_if(b, b.ucmp(Op_IEqual, b.load_function(b.t_u32, record_var), b.uconst(0)));
        const auto slot = b.load_function(b.t_u32, slot_var);
        const auto candidate =
            commit.load(b.ibin(Op_IAdd, b.uconst(kFragmentDrawCommitHeaderWords), slot));
        const auto bounded = b.land(b.ucmp(Op_UGreaterThan, candidate, b.uconst(0)),
                                    b.ucmp(Op_ULessThanEqual, candidate, pixels));
        const auto safe = b.sel(bounded, b.ibin(Op_ISub, candidate, b.uconst(1)), b.uconst(0));
        auto matches = bounded;
        const uint32_t key[] = {primitive, x, y}, fields[] = {4, 5, 6};
        for (uint32_t field = 0; field < 3; ++field)
            matches = b.land(
                matches, b.ucmp(Op_IEqual, key[field],
                                source.load(collector_address(b, collector, safe, fields[field]))));
        b.store_function(record_var, b.sel(matches, candidate, b.uconst(0)));
        b.store_function(slot_var, b.ibin(Op_BitwiseAnd, b.ibin(Op_IAdd, slot, b.uconst(1)),
                                          b.uconst(capacity.pixel_slots() - 1)));
        end_if(b, search);
    });
    const auto found = b.load_function(b.t_u32, record_var);
    discard_unless(b, b.ucmp(Op_UGreaterThan, found, b.uconst(0)));
    const auto worker = b.ibin(Op_ISub, found, b.uconst(1));
    const auto wave = b.ibin(Op_UDiv, worker, b.uconst(64)),
               lane = b.ibin(Op_UMod, worker, b.uconst(64));
    const auto record =
        b.ibin(Op_IAdd,
               b.ibin(Op_IAdd, b.ibin(Op_IMul, wave, b.uconst(capacity.output_span())),
                      b.uconst(kPacketWaveOutputPrefix)),
               b.ibin(Op_IMul, lane, b.uconst(export_words)));
    const auto value = [&](uint32_t field) {
        return output.load(b.ibin(Op_IAdd, record, b.uconst(field)));
    };
    discard_unless(b, b.land(b.ucmp(Op_IEqual, value(0), b.uconst(1)),
                             b.land(b.ucmp(Op_IEqual, value(1), b.uconst(1)),
                                    b.ucmp(Op_IEqual, value(2), b.uconst(1)))));
    b.export_color(0, value(8), value(9), value(10), value(11));
    return b.finish();
}
}   // namespace prosper::gpu
