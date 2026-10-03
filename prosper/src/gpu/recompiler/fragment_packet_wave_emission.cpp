#include "gpu/recompiler/fragment_packet_services.hpp"

namespace prosper::gpu {
uint32_t SpirvCompute::load_packet_word(uint32_t index) {
    if (packet_input_base) index = ibin(Op_IAdd, packet_input_base, index);
    const auto pointer = id(), value = id();
    put(code, Op_AccessChain, {t_ptr_sb_f32, pointer, v_in, uconst(0), index});
    put(code, Op_Load, {t_u32, value, pointer});
    return value;
}
void configure_packet_wave_data(const FragmentInvocationPacket& packet,
                                const std::vector<Rdna2Inst>& ins, FragmentPacketProgram& program,
                                PacketResourceServices& services, PacketWaveDataLayout& layout) {
    for (const auto& column : packet.vgprs) layout.vgprs.push_back(column.reg);
    auto scalars = packet.sgprs;
    std::sort(scalars.begin(), scalars.end());
    const auto append = [&](uint32_t value) {
        const auto offset = static_cast<uint32_t>(program.input_words.size());
        program.input_words.push_back(value);
        return offset;
    };
    for (const auto& [reg, value] : scalars) {
        layout.sgprs.push_back(reg);
        layout.scalar_offsets.push_back(append(value));
        layout.scalar_available_offsets.push_back(append(1));
    }
    layout.expected_m0 = append(services.input.parameter_cache.m0);
    layout.entry_m0 = append(services.input.parameter_cache.entry_m0);
    layout.entry_m0_available = services.input.parameter_cache.entry_m0_available;
    layout.entry_m0_available_offset = append(layout.entry_m0_available ? 1 : 0);
    for (const auto& buffer : services.input.buffers) {
        const auto descriptor = static_cast<uint32_t>(program.input_words.size());
        for (const auto word : buffer.descriptor) append(word);
        layout.buffers.push_back({buffer.pc, services.buffer_offsets.at(buffer.pc),
                                  static_cast<uint32_t>(buffer.words.size()), descriptor});
    }
    for (const auto& in : ins)
        if (in.fmt == Rdna2Format::VINTRP)
            layout.parameters.push_back(
                {in.pc, services.parameter_offsets.at(in.pc), in.vintrp_attr, in.vintrp_chan});
    for (const auto& image : services.input.images) {
        layout.image_descriptor_offsets.push_back(
            static_cast<uint32_t>(program.input_words.size()));
        for (const auto word : image.descriptor) append(word);
        layout.sampler_offsets.push_back(static_cast<uint32_t>(program.input_words.size()));
        for (const auto word : image.sampler) append(word);
    }
    layout.input_words = static_cast<uint32_t>(program.input_words.size());
    layout.output_words = static_cast<uint32_t>(program.output_words.size());
    services.wave_data = &layout;
}
PacketWaveEmission begin_packet_wave_data(SpirvCompute& b, const PacketWaveDataLayout& layout) {
    // All predicates use uniform shared wave metadata and WorkgroupId only. They are identical
    // for all64 workers. Malformed metadata skips the ENTIRE workgroup, never one participant.
    // No guest load/barrier/output is reached until actual descriptor extents and tags agree.
    const auto input_length = b.id(), output_length = b.id(), authority_length = b.id();
    b.put(b.code, 68, {b.t_u32, input_length, b.v_in, 0});   // OpArrayLength, raw-u32 Block member0
    b.put(b.code, 68, {b.t_u32, output_length, b.v_out, 0});
    b.put(b.code, 68, {b.t_u32, authority_length, b.v_cbuf, 0});
    b.put(b.deco, Op_Decorate, {b.v_cbuf, 24});   // NonWritable: separate dispatcher authority
    const auto authority_word = [&](uint32_t index) {
        const auto pointer = b.id(), value = b.id();
        b.put(b.code, Op_AccessChain, {b.t_ptr_sb_u32, pointer, b.v_cbuf, b.uconst(0), index});
        b.put(b.code, Op_Load, {b.t_u32, value, pointer});
        return value;
    };
    const auto merge = b.id();
    const auto enter = [&](uint32_t predicate) {
        const auto body = b.id();
        b.put(b.code, Op_SelectionMerge, {merge, 0});
        b.put(b.code, Op_BranchConditional, {predicate, body, merge});
        b.put(b.code, Op_Label, {body});
        b.cur_block = body;
    };
    // Separate nested selections have their own merge blocks, unwound after guest services.
    enter(b.land(
        b.ucmp(Op_UGreaterThanEqual, input_length, b.uconst(kPacketWaveHeaderWords)),
        b.ucmp(Op_UGreaterThanEqual, authority_length, b.uconst(kPacketWaveAuthorityHeaderWords))));
    const auto magic = b.load_packet_word(b.uconst(0));
    const auto count = b.load_packet_word(b.uconst(1));
    const auto declared_input = b.load_packet_word(b.uconst(2));
    const auto declared_output = b.load_packet_word(b.uconst(3));
    const auto valid_count = b.land(b.ucmp(Op_UGreaterThan, count, b.uconst(0)),
                                    b.ucmp(Op_ULessThanEqual, count, b.uconst(4096)));
    const auto safe_count = b.sel(valid_count, count, b.uconst(0));
    const auto table_end = b.ibin(Op_IAdd, b.uconst(4), b.ibin(Op_IMul, safe_count, b.uconst(2)));
    const auto authority_end = b.ibin(Op_IAdd, b.uconst(kPacketWaveAuthorityHeaderWords),
                                      b.ibin(Op_IMul, safe_count, b.uconst(2)));
    auto valid = b.land(b.ucmp(Op_IEqual, magic, b.uconst(kPacketWaveTableMagic)), valid_count);
    valid = b.land(valid, b.land(b.ucmp(Op_IEqual, input_length, declared_input),
                                 b.ucmp(Op_IEqual, output_length, declared_output)));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, table_end, input_length));
    valid = b.land(valid, b.ucmp(Op_IEqual, authority_length, authority_end));
    valid = b.land(
        valid, b.ucmp(Op_IEqual, authority_word(b.uconst(0)), b.uconst(kPacketWaveAuthorityMagic)));
    valid = b.land(valid, b.ucmp(Op_IEqual, authority_word(b.uconst(1)), count));
    valid = b.land(valid, b.ucmp(Op_IEqual, authority_word(b.uconst(2)), input_length));
    valid = b.land(valid, b.ucmp(Op_IEqual, authority_word(b.uconst(3)), output_length));
    valid =
        b.land(valid, b.ucmp(Op_IEqual, authority_word(b.uconst(4)), b.uconst(layout.input_words)));
    valid = b.land(valid,
                   b.ucmp(Op_IEqual, authority_word(b.uconst(5)), b.uconst(layout.output_words)));
    valid = b.land(valid, b.land(b.ucmp(Op_ULessThan, b.groupid[0], safe_count),
                                 b.land(b.ucmp(Op_IEqual, b.groupid[1], b.uconst(0)),
                                        b.ucmp(Op_IEqual, b.groupid[2], b.uconst(0)))));
    // Invalid table selects existing header words, avoiding even an eager out-of-range table read.
    const auto table_index =
        b.sel(valid, b.ibin(Op_IAdd, b.uconst(4), b.ibin(Op_IMul, b.groupid[0], b.uconst(2))),
              b.uconst(0));
    const auto input_base = b.load_packet_word(table_index);
    const auto output_base = b.load_packet_word(b.ibin(Op_IAdd, table_index, b.uconst(1)));
    // Invalid metadata selects existing authority header words before eager record loads. Valid
    // metadata has exact record extent. No quadratic scan or mutable completion-prefix authority.
    const auto authority_index = b.sel(valid,
                                       b.ibin(Op_IAdd, b.uconst(kPacketWaveAuthorityHeaderWords),
                                              b.ibin(Op_IMul, b.groupid[0], b.uconst(2))),
                                       b.uconst(0));
    valid = b.land(valid, b.ucmp(Op_IEqual, input_base, authority_word(authority_index)));
    valid = b.land(valid, b.ucmp(Op_IEqual, output_base,
                                 authority_word(b.ibin(Op_IAdd, authority_index, b.uconst(1)))));
    const auto input_span = b.uconst(layout.input_words + 2);
    const auto output_span = b.uconst(layout.output_words + kPacketWaveOutputPrefix);
    auto input_valid =
        b.land(b.ucmp(Op_UGreaterThanEqual, input_length, input_span),
               b.ucmp(Op_ULessThanEqual, input_base, b.ibin(Op_ISub, input_length, input_span)));
    input_valid = b.land(input_valid, b.ucmp(Op_UGreaterThanEqual, input_base, table_end));
    const auto output_valid =
        b.land(b.ucmp(Op_UGreaterThanEqual, output_length, output_span),
               b.ucmp(Op_ULessThanEqual, output_base, b.ibin(Op_ISub, output_length, output_span)));
    const auto safe_input_base = b.sel(b.land(valid, input_valid), input_base, b.uconst(0));
    valid = b.land(valid, b.land(input_valid, output_valid));
    valid = b.land(
        valid,
        b.land(
            b.ucmp(Op_IEqual, b.load_packet_word(safe_input_base), b.uconst(kPacketWaveInputMagic)),
            b.ucmp(Op_IEqual, b.load_packet_word(b.ibin(Op_IAdd, safe_input_base, b.uconst(1))),
                   b.groupid[0])));
    // Presence words are uniform owned metadata. Sanitize every eager address to an existing
    // header word when the full region/tag is invalid; no worker may leave before a barrier.
    const auto present = [&](uint32_t offset, uint32_t expected) {
        const auto index =
            b.sel(valid, b.ibin(Op_IAdd, input_base, b.uconst(2 + offset)), b.uconst(0));
        return b.ucmp(Op_IEqual, b.load_packet_word(index), b.uconst(expected));
    };
    for (const auto offset : layout.scalar_available_offsets)
        valid = b.land(valid, present(offset, 1));
    valid =
        b.land(valid, present(layout.entry_m0_available_offset, layout.entry_m0_available ? 1 : 0));
    const auto inner_merge = b.id(), body = b.id();
    b.put(b.code, Op_SelectionMerge, {inner_merge, 0});
    b.put(b.code, Op_BranchConditional, {valid, body, inner_merge});
    b.put(b.code, Op_Label, {body});
    b.cur_block = body;
    b.packet_input_base = b.ibin(Op_IAdd, input_base, b.uconst(2));
    b.packet_output_base = b.ibin(Op_IAdd, output_base, b.uconst(kPacketWaveOutputPrefix));
    b.gidx = b.linear_localid;   // every record remains per-logical-lane INSIDE this wave's region
    std::vector<uint32_t> marker;
    b.pstr(marker, "Prosper.GuestFragmentPacket.WaveData=WAV1;NoRasterPackingAuthority");
    b.putv(b.debug, Op_ModuleProcessed, marker);
    return {inner_merge, output_base, merge};
}
void finish_packet_wave_data(SpirvCompute& b, PacketWaveEmission state) {
    const auto base =
        b.ibin(Op_IAdd, state.output_base, b.ibin(Op_IMul, b.linear_localid, b.uconst(2)));
    for (uint32_t word = 0; word < 2; ++word) {
        const auto pointer = b.id();
        b.put(
            b.code, Op_AccessChain,
            {b.t_ptr_sb_u32, pointer, b.v_out, b.uconst(0), b.ibin(Op_IAdd, base, b.uconst(word))});
        b.put(b.code, Op_Store, {pointer, word ? b.groupid[0] : b.uconst(kPacketWaveOutputMagic)});
    }
    b.put(b.code, Op_Branch, {state.merge});
    b.put(b.code, Op_Label, {state.merge});
    b.cur_block = state.merge;
    b.put(b.code, Op_Branch, {state.outer_merge});
    b.put(b.code, Op_Label, {state.outer_merge});
    b.cur_block = state.outer_merge;
}
}   // namespace prosper::gpu
