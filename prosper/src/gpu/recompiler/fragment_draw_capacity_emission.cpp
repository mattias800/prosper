#include "gpu/recompiler/fragment_draw_capacity.hpp"
#include "gpu/recompiler/fragment_packet_services.hpp"

namespace prosper::gpu {
PacketWaveEmission begin_fragment_draw_capacity(SpirvCompute& b,
                                                const PacketWaveDataLayout& layout) {
    const auto input_length = b.id(), output_length = b.id(), authority_length = b.id();
    b.put(b.code, 68, {b.t_u32, input_length, b.v_in, 0});
    b.put(b.code, 68, {b.t_u32, output_length, b.v_out, 0});
    b.put(b.code, 68, {b.t_u32, authority_length, b.v_cbuf, 0});
    b.put(b.deco, Op_Decorate, {b.v_cbuf, 24}); // NonWritable private placement authority
    const auto authority = [&](uint32_t offset) {
        const auto pointer = b.id(), word = b.id();
        b.put(b.code, Op_AccessChain,
              {b.t_ptr_sb_u32, pointer, b.v_cbuf, b.uconst(0), b.uconst(offset)});
        b.put(b.code, Op_Load, {b.t_u32, word, pointer});
        return word;
    };
    const auto outer_merge = b.id(), header = b.id();
    // Length gate precedes even the first header read. Every predicate is whole-WG uniform.
    b.put(b.code, Op_SelectionMerge, {outer_merge, 0});
    b.put(b.code, Op_BranchConditional,
          {b.land(b.ucmp(Op_UGreaterThanEqual, input_length, b.uconst(kFragmentDrawHeaderWords)),
                  b.ucmp(Op_IEqual, authority_length, b.uconst(kFragmentDrawAuthorityWords))),
           header, outer_merge});
    b.emit_label(header);
    const auto capacity = authority(1);
    const auto quads = b.load_packet_word(b.uconst(4));
    const auto count = b.load_packet_word(b.uconst(1));
    auto valid = b.land(
        b.ucmp(Op_IEqual, authority(0), b.uconst(kFragmentDrawAuthorityMagic)),
        b.ucmp(Op_IEqual, b.load_packet_word(b.uconst(0)), b.uconst(kFragmentDrawInputMagic)));
    valid = b.land(valid, b.land(b.ucmp(Op_UGreaterThan, capacity, b.uconst(0)),
                                 b.ucmp(Op_ULessThanEqual, capacity, b.uconst(256))));
    valid = b.land(valid, b.land(b.ucmp(Op_UGreaterThan, authority(6), b.uconst(0)),
                                 b.ucmp(Op_ULessThanEqual, authority(6), b.uconst(4096))));
    valid = b.land(
        valid, b.ucmp(Op_IEqual, capacity,
                      b.ibin(Op_UDiv, b.ibin(Op_IAdd, authority(6), b.uconst(15)), b.uconst(16))));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, quads, authority(6)));
    valid =
        b.land(valid, b.ucmp(Op_IEqual, count,
                             b.ibin(Op_UDiv, b.ibin(Op_IAdd, quads, b.uconst(15)), b.uconst(16))));
    valid = b.land(valid, b.ucmp(Op_ULessThanEqual, count, capacity));
    const auto input_span = b.uconst(layout.input_words + 2);
    const auto output_span = b.uconst(layout.output_words + kPacketWaveOutputPrefix);
    valid = b.land(valid, b.ucmp(Op_IEqual, authority(4), input_span));
    valid = b.land(valid, b.ucmp(Op_IEqual, authority(5), output_span));
    valid = b.land(valid, b.land(b.ucmp(Op_IEqual, authority(2), input_length),
                                 b.ucmp(Op_IEqual, authority(3), output_length)));
    valid = b.land(
        valid, b.land(b.ucmp(Op_IEqual, input_length,
                             b.ibin(Op_IAdd, b.uconst(kFragmentDrawHeaderWords),
                                    b.ibin(Op_IMul, capacity, input_span))),
                      b.ucmp(Op_IEqual, output_length, b.ibin(Op_IMul, capacity, output_span))));
    valid =
        b.land(valid, b.land(b.ucmp(Op_IEqual, b.load_packet_word(b.uconst(2)), input_length),
                             b.ucmp(Op_IEqual, b.load_packet_word(b.uconst(3)), output_length)));
    valid = b.land(valid,
                   b.land(b.ucmp(Op_IEqual, b.load_packet_word(b.uconst(5)), count),
                          b.land(b.ucmp(Op_IEqual, b.load_packet_word(b.uconst(6)), b.uconst(1)),
                                 b.ucmp(Op_IEqual, b.load_packet_word(b.uconst(7)), b.uconst(1)))));
    valid = b.land(valid, b.ucmp(Op_IEqual, b.load_packet_word(b.uconst(8)), b.uconst(0)));
    valid = b.land(valid, b.land(b.ucmp(Op_ULessThan, b.groupid[0], count),
                                 b.land(b.ucmp(Op_IEqual, b.groupid[1], b.uconst(0)),
                                        b.ucmp(Op_IEqual, b.groupid[2], b.uconst(0)))));
    // No GPU-writable route table: actual bases are functions of immutable checked strides.
    const auto input_base = b.ibin(Op_IAdd, b.uconst(kFragmentDrawHeaderWords),
                                   b.ibin(Op_IMul, b.groupid[0], input_span));
    const auto output_base = b.ibin(Op_IMul, b.groupid[0], output_span);
    const auto safe_base = b.sel(valid, input_base, b.uconst(0));
    valid = b.land(
        valid,
        b.land(b.ucmp(Op_IEqual, b.load_packet_word(safe_base), b.uconst(kPacketWaveInputMagic)),
               b.ucmp(Op_IEqual, b.load_packet_word(b.ibin(Op_IAdd, safe_base, b.uconst(1))),
                      b.groupid[0])));
    const auto present = [&](uint32_t offset, uint32_t expected) {
        const auto address =
            b.sel(valid, b.ibin(Op_IAdd, input_base, b.uconst(2 + offset)), b.uconst(0));
        return b.ucmp(Op_IEqual, b.load_packet_word(address), b.uconst(expected));
    };
    for (const auto offset : layout.scalar_available_offsets)
        valid = b.land(valid, present(offset, 1));
    valid =
        b.land(valid, present(layout.entry_m0_available_offset, layout.entry_m0_available ? 1 : 0));
    const auto merge = b.id(), body = b.id();
    b.put(b.code, Op_SelectionMerge, {merge, 0});
    b.put(b.code, Op_BranchConditional, {valid, body, merge});
    b.emit_label(body);
    b.packet_input_base = b.ibin(Op_IAdd, input_base, b.uconst(2));
    b.packet_output_base = b.ibin(Op_IAdd, output_base, b.uconst(kPacketWaveOutputPrefix));
    b.gidx = b.linear_localid;
    std::vector<uint32_t> marker;
    b.pstr(marker, "Prosper.GuestFragmentPacket.WaveData=WAV2;GPUCapacityNotGuestEntry");
    b.putv(b.debug, Op_ModuleProcessed, marker);
    return {merge, output_base, outer_merge};
}
} // namespace prosper::gpu
