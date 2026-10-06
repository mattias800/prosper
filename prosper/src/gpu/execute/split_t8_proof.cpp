#include <algorithm>
#include "gpu/execute/split_t8_proof.hpp"
#include "gpu/agc/agc_shader_layout.hpp"

#include "gpu/execute/sopp_cfg.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <unordered_map>
#include <vector>

namespace prosper::gpu {

// A split image descriptor needs a stronger proof than the scalar fold's linear walk. In
// particular, a conditional branch can skip one of two adjacent loads while the walk still sees
// both. This deliberately admits only direct, entry-rooted scalar loads; computed pointers and
// SOFFSETs remain on the existing unresolved path.
//
// The proof is a forward must-dataflow over the program's CFG, not a scan of index ranges. Every
// SGPR carries a tag: the entry value of that register, word k of the scalar load at instruction i,
// or unknown. A load from an entry-valued pointer tags its destination words, s_mov_b32 copies a
// tag, any other possible write makes the register unknown, and a join keeps a tag only where
// every incoming path agrees. The consumer is admitted only if, on EVERY path that reaches it
// (including a later loop iteration), each descriptor word still carries the tag of the load word
// the linear fold read. That one condition covers what interval scans kept missing: a path that
// skips a load or a copy, a loop that replays a copy after its source changed, and a loop path
// that leaves the lexical loop body to rewrite a register (#4203/#4214/#4218 reviews).
//
// The loaded BYTES must also be the ones the CPU snapshot read, so no instruction from which the
// consumer is reachable may write guest memory. Two narrower contracts are kept from the interval
// proof, because no title evidence needs them relaxed: a producer load inside a cycle is refused
// outright, and only a pointer register that was never written counts as entry-rooted (a copy,
// even of itself, drops that identity). A loop that re-runs a COPY is admitted when the copied
// word still carries the same load tag, since a copy captures bits.
bool mapped_split_t8_reaches_use(const uint32_t* code, size_t dwords, uint32_t use_pc, int tbase,
                                 const std::array<uint32_t, 8>& source_pc,
                                 const std::array<uint64_t, 8>& source_addr,
                                 const uint32_t* user_sgprs, uint32_t nsgpr,
                                 uint32_t user_sgpr_base,
                                 const std::vector<ImageWriteExtent>& image_writes) {
    constexpr int kSgprs = 106;
    // This rare fallback reparses the owned code. Keep its CFG analysis bounded; larger programs
    // retain the ordinary unresolved path until they have a cached analysis.
    if (!code || !user_sgprs || dwords > 2048 || tbase < 0 || tbase + 7 >= kSgprs)
        return false;
    std::vector<Rdna2Inst> full;
    rdna2_walk(code, dwords, full);
    if (full.empty() || !full.back().is_end || !rdna2_append_closed_tail_blocks(code, dwords, full) ||
        has_indirect_control_flow(full)) return false;
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < full.size(); ++i) {
        if (full[i].fmt == Rdna2Format::Unknown || !full[i].len_dwords ||
            !by_pc.emplace(full[i].pc, i).second) return false;
    }
    const auto use_it = by_pc.find(use_pc);
    if (use_it == by_pc.end()) return false;
    const size_t use = use_it->second;
    constexpr size_t no_edge = SIZE_MAX;
    std::vector<std::array<size_t, 2>> edges(full.size());
    for (size_t i = 0; i < full.size(); ++i) {
        edges[i] = {no_edge, no_edge};
        const auto& in = full[i];
        if (in.is_end) continue;
        // The debug conditional branches have different predicates, and an indirect transfer
        // has no statically enumerable successor. Decline rather than treating either as fallthrough.
        if (in.fmt == Rdna2Format::SOPP && in.opcode >= 0x17 && in.opcode <= 0x1a)
            return false;
        // s_subvector_loop_begin/end (SOPK 0x1b/0x1c) branch by their SIMM16, which this CFG does
        // not model.
        if (in.fmt == Rdna2Format::SOPK && (in.opcode == 0x1b || in.opcode == 0x1c))
            return false;
        if (sopp_is_branch(in)) {
            const int64_t target = sopp_branch_target(in);
            if (target < 0 || target > UINT32_MAX) return false;
            const auto branch = by_pc.find(static_cast<uint32_t>(target));
            if (branch == by_pc.end()) return false;
            edges[i][0] = branch->second;
            if (sopp_is_unconditional_branch(in)) continue;
            if (i + 1 >= full.size()) return false;
            edges[i][1] = i + 1;
        } else if (i + 1 < full.size()) {
            edges[i][0] = i + 1;
        }
    }
    auto may_write = [&](const Rdna2Inst& in, int reg) {
        // Unknown relative SGPR destinations cannot be excluded by comparing the decoded base.
        if ((in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCallB64)) return true;
        // A plain B32 move writes only its named word. Treating it as a pair incorrectly
        // clobbers the adjacent destination while assembling a T# one lane at a time.
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
            in.dst.kind == OperandKind::SGPR)
            return in.dst.value == reg;
        // v_readfirstlane_b32 (VOP1 0x02) and v_readlane_b32 (VOP3 0x360) write the SGPR their
        // VDST field names, which the decoder carries as a VGPR operand. 0x182 is the VOP3 slot of
        // v_readfirstlane; gfx1030 has no such encoding, so it is refused the same way rather than
        // trusted to be absent.
        if (((in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02u) ||
             (in.fmt == Rdna2Format::VOP3 && (in.opcode == 0x360u || in.opcode == 0x182u))) &&
            in.dst.kind == OperandKind::VGPR)
            return in.dst.value == reg;
        auto overlaps = [reg](const Operand& dst, uint32_t width) {
            return dst.kind == OperandKind::SGPR && reg >= dst.value &&
                   static_cast<uint32_t>(reg - dst.value) < width;
        };
        uint32_t width = 2; // conservative for ordinary scalar and vector-carry pair writes
        // A 32-bit SOP2 result is one SGPR: counting a neighbour overlapped a pointer register
        // that sat directly above an unrelated scalar add and refused a split T# for nothing.
        if (in.fmt == Rdna2Format::SOP2 && in.dst.kind == OperandKind::SGPR)
            width = rdna2_sop2_dest_dwords(in.opcode);
        if (in.fmt == Rdna2Format::SMEM) {
            switch (in.opcode & 7u) {
                case 0: width = 1; break;
                case 1: width = 2; break;
                case 2: width = 4; break;
                case 3: width = 8; break;
                case 4: width = 16; break;
                default: return true;
            }
        }
        return overlaps(in.dst, width) || overlaps(in.sdst, 2);
    };

    // Validate each lane's producer and name the tag its descriptor word must carry at the use.
    // A tag is (instruction index << 8 | word) for a load word, and the two sentinels below.
    constexpr uint64_t kUnknown = ~0ull;
    constexpr uint64_t kEntryBit = 1ull << 63;
    auto load_tag = [](size_t producer, uint32_t word) {
        return (static_cast<uint64_t>(producer) << 8u) | word;
    };
    std::vector<uint8_t> is_producer(full.size());
    std::array<uint64_t, 8> expected{};
    for (int lane = 0; lane < 8; ++lane) {
        const auto found = by_pc.find(source_pc[static_cast<size_t>(lane)]);
        if (found == by_pc.end()) return false;
        const size_t producer = found->second;
        const Rdna2Inst& load = full[producer];
        if (load.fmt != Rdna2Format::SMEM || load.opcode > 4u ||
            load.opcode < 2u || load.dst.kind != OperandKind::SGPR ||
            load.src[0].kind != OperandKind::SGPR ||
            ((load.words[1] >> 25u) & 0x7fu) != 125u ||
            static_cast<int32_t>(load.literal) < 0) return false;
        const uint32_t width = load.opcode == 2u ? 4u : load.opcode == 3u ? 8u : 16u;
        const int base_reg = load.src[0].value;
        if (base_reg < static_cast<int>(user_sgpr_base) ||
            base_reg + 1 >= static_cast<int>(user_sgpr_base + nsgpr) ||
            base_reg + 1 >= kSgprs) return false;
        const size_t seed = static_cast<size_t>(base_reg - static_cast<int>(user_sgpr_base));
        const uint64_t base = static_cast<uint64_t>(user_sgprs[seed]) |
                              (static_cast<uint64_t>(user_sgprs[seed + 1]) << 32u);
        if (base > UINT64_MAX - load.literal) return false;
        const uint64_t first_addr = base + load.literal;
        const uint64_t addr = source_addr[static_cast<size_t>(lane)];
        if (addr < first_addr || addr - first_addr >= width * sizeof(uint32_t) ||
            ((addr - first_addr) & 3u)) return false;
        const uint32_t word = static_cast<uint32_t>((addr - first_addr) / sizeof(uint32_t));
        if (load.dst.value < 0 || load.dst.value + static_cast<int>(word) >= kSgprs)
            return false;
        is_producer[producer] = 1;
        expected[static_cast<size_t>(lane)] = load_tag(producer, word);
    }

    using State = std::array<uint64_t, kSgprs>;
    auto transfer = [&](size_t i, State& s) {
        const Rdna2Inst& in = full[i];
        if (is_producer[i]) {
            const int base_reg = in.src[0].value;
            const uint32_t width = in.opcode == 2u ? 4u : in.opcode == 3u ? 8u : 16u;
            const bool entry_pointer = s[static_cast<size_t>(base_reg)] ==
                                           (kEntryBit | static_cast<uint64_t>(base_reg)) &&
                                       s[static_cast<size_t>(base_reg + 1)] ==
                                           (kEntryBit | static_cast<uint64_t>(base_reg + 1));
            for (uint32_t k = 0; k < width; ++k) {
                const int reg = in.dst.value + static_cast<int>(k);
                if (reg >= kSgprs) break;
                s[static_cast<size_t>(reg)] = entry_pointer ? load_tag(i, k) : kUnknown;
            }
            return;
        }
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
            in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value < kSgprs) {
            const bool from_sgpr = in.src[0].kind == OperandKind::SGPR &&
                                   in.src[0].value >= 0 && in.src[0].value < kSgprs;
            const uint64_t tag = from_sgpr ? s[static_cast<size_t>(in.src[0].value)] : kUnknown;
            s[static_cast<size_t>(in.dst.value)] = (tag & kEntryBit) ? kUnknown : tag;
            return;
        }
        for (int reg = 0; reg < kSgprs; ++reg)
            if (may_write(in, reg)) s[static_cast<size_t>(reg)] = kUnknown;
    };

    // A producer that can reach itself is refused (see above).
    for (size_t producer = 0; producer < full.size(); ++producer) {
        if (!is_producer[producer]) continue;
        std::vector<uint8_t> seen(full.size());
        std::vector<size_t> queue;
        for (size_t next : edges[producer])
            if (next != no_edge) { seen[next] = 1; queue.push_back(next); }
        for (size_t q = 0; q < queue.size(); ++q) {
            if (queue[q] == producer) return false;
            for (size_t next : edges[queue[q]])
                if (next != no_edge && !seen[next]) { seen[next] = 1; queue.push_back(next); }
        }
    }

    // in_state[i] is the meet over every path from entry to instruction i.
    std::vector<State> in_state(full.size());
    std::vector<uint8_t> reached(full.size());
    for (int reg = 0; reg < kSgprs; ++reg)
        in_state[0][static_cast<size_t>(reg)] = kEntryBit | static_cast<uint64_t>(reg);
    reached[0] = 1;
    std::vector<size_t> worklist{0};
    std::vector<uint8_t> queued(full.size());
    queued[0] = 1;
    while (!worklist.empty()) {
        const size_t at = worklist.back();
        worklist.pop_back();
        queued[at] = 0;
        State out = in_state[at];
        transfer(at, out);
        for (size_t next : edges[at]) {
            if (next == no_edge) continue;
            bool changed = false;
            if (!reached[next]) {
                in_state[next] = out;
                reached[next] = 1;
                changed = true;
            } else {
                // Tags only ever fall to unknown, so the iteration terminates.
                for (size_t reg = 0; reg < static_cast<size_t>(kSgprs); ++reg) {
                    if (in_state[next][reg] != kUnknown && in_state[next][reg] != out[reg]) {
                        in_state[next][reg] = kUnknown;
                        changed = true;
                    }
                }
            }
            if (changed && !queued[next]) {
                queued[next] = 1;
                worklist.push_back(next);
            }
        }
    }
    if (!reached[use]) return false;
    for (int lane = 0; lane < 8; ++lane)
        if (in_state[use][static_cast<size_t>(tbase + lane)] != expected[static_cast<size_t>(lane)])
            return false;

    // A guest-visible write that can execute before the consumer could alter descriptor backing
    // after the CPU snapshot. That is every reachable instruction from which the consumer can be
    // reached, so a write after the consumer inside a loop counts too. LDS writes are a separate
    // address space; global/buffer and storage-image writes are left unresolved because proving
    // non-aliasing would need resource ownership analysis.
    std::vector<std::vector<size_t>> preds(full.size());
    for (size_t i = 0; i < full.size(); ++i)
        for (size_t next : edges[i])
            if (next != no_edge) preds[next].push_back(i);
    std::vector<uint8_t> before_use(full.size());
    std::vector<size_t> back{use};
    while (!back.empty()) {
        const size_t at = back.back();
        back.pop_back();
        for (size_t p : preds[at]) {
            if (before_use[p]) continue;
            before_use[p] = 1;
            back.push_back(p);
        }
    }
    // A storage-image write whose footprint is known and disjoint from every descriptor byte read here
    // cannot change what the loads observed, so it does not revoke the proof.
    auto cannot_reach_descriptor = [&](const Rdna2Inst& writer) {
        if (writer.fmt != Rdna2Format::MIMG) return false;
        for (const ImageWriteExtent& extent : image_writes) {
            if (extent.pc != writer.pc) continue;
            for (int lane = 0; lane < 8; ++lane) {
                const uint64_t addr = source_addr[static_cast<size_t>(lane)];
                if (addr < extent.hi && extent.lo < addr + sizeof(uint32_t)) return false;
            }
            return true;
        }
        return false;
    };
    for (size_t i = 0; i < full.size(); ++i)
        if (before_use[i] && reached[i] && rdna2_instruction_may_write_memory(full[i]) &&
            !cannot_reach_descriptor(full[i]))
            return false;
    return true;
}

bool storage_image_write_extent(const std::array<uint32_t, 8>& t8, uint64_t& lo, uint64_t& hi) {
    const DecodedImageDescriptor d = decode_image_descriptor(t8.data());
    Gen5ImageFormatInfo format;
    if (!d.base || !d.width || !d.height || !gen5_image_format(d.format, &format) ||
        !format.bytes_per_block || d.compression_enabled || d.write_compress_enabled ||
        d.metadata_addr)
        return false;
    // Only layouts whose slices are separate 2D surfaces are bounded here. A 3D surface (TYPE 10) in
    // a thick swizzle mode is tiled in 3D blocks (a 64 KiB block at 4 bytes per texel is 32x32x16
    // texels), so its depth pads to the block depth: 1024x1024x2 R32 occupies 64 MiB, not 8. A 3D
    // UAV view's DEPTH is also only the selected slice range, not the volume. Neither is modelled,
    // so a 3D image, and any TYPE outside the GFX10 image range, has no sound bound.
    if (!valid_image_type(d.type) || d.type == 10) return false;
    // A mip chain has no cheap sound bound either. GFX10 stores a tiled chain smallest level first
    // and level 0 LAST, each level outside the tail padded to whole swizzle blocks, and BASE is the
    // allocation base -- so level 0 can end past twice its own padded size. An R8 1280x256 chain
    // with LAST_LEVEL 3 in a 64 KiB swizzle occupies 5+3+2+1 = 11 blocks (704 KiB), while 2 x the
    // padded level 0 is 640 KiB. The measured shape is single-level, so refusing loses nothing.
    if (d.max_mip || d.last_level) return false;
    constexpr uint64_t kTile = 256;
    const uint64_t w = (static_cast<uint64_t>(d.width) + kTile - 1) / kTile * kTile;
    const uint64_t h = (static_cast<uint64_t>(d.height) + kTile - 1) / kTile * kTile;
    const uint64_t slices = std::max<uint64_t>(d.depth, 1) + d.base_array;
    const uint64_t samples = std::max<uint32_t>(d.sample_count, 1u);
    // The block size is per block of block_width x block_height texels; one byte per texel of block
    // size is a superset for every format (blocks never hold fewer texels than bytes_per_block / 16).
    const uint64_t bytes_per_texel = format.bytes_per_block;
    if (w > UINT64_MAX / h) return false;
    uint64_t bytes = w * h;
    if (bytes > UINT64_MAX / slices) return false;
    bytes *= slices;
    if (bytes > UINT64_MAX / samples) return false;
    bytes *= samples;
    if (bytes > UINT64_MAX / bytes_per_texel) return false;
    bytes *= bytes_per_texel;
    if (d.base > UINT64_MAX - bytes) return false;
    lo = d.base;
    hi = d.base + bytes;
    return true;
}

} // namespace prosper::gpu
