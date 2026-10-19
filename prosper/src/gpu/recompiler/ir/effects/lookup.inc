// The decoder's opcode spaces, minus encodings another format claims first (rdna2_decode_one).
// SOP2: bits[29:23]; 0x60-0x7F are the SOPK/SOP1/SOPC/SOPP prefixes.
constexpr Rdna2OpcodeRange kSop2Range{0, 0x60};
// SOP1: bits[15:8].
constexpr Rdna2OpcodeRange kSop1Range{0, 0x100};
// SOPK: bits[27:23]; 0x1D/0x1E/0x1F are the SOP1/SOPC/SOPP prefixes.
constexpr Rdna2OpcodeRange kSopkRange{0, 0x1d};
// SOPC and SOPP: bits[22:16].
constexpr Rdna2OpcodeRange kSopcRange{0, 0x80};
constexpr Rdna2OpcodeRange kSoppRange{0, 0x80};
// SMEM: bits[25:18].
constexpr Rdna2OpcodeRange kSmemRange{0, 0x100};
// VOP2: bits[30:25]; 0x3E/0x3F are the VOPC/VOP1 prefixes.
constexpr Rdna2OpcodeRange kVop2Range{0, 0x3e};
// VOP1 and VOPC: 8-bit opcode fields (a VOPC e64 compare keeps its e32 number).
constexpr Rdna2OpcodeRange kVop1Range{0, 0x100};
constexpr Rdna2OpcodeRange kVopcRange{0, 0x100};
// VOP3: bits[25:16]; 0x000-0x0FF are VOPC e64 and decode as VOPC.
constexpr Rdna2OpcodeRange kVop3Range{0x100, 0x400};
// VOP3P: bits[22:16].
constexpr Rdna2OpcodeRange kVop3pRange{0, 0x80};

template <size_t N>
struct FormatTable {
    std::array<Rdna2InstructionEffects, N> by_opcode{};
};

// Dense lookup for one format: its named rows plus its explicit reserved ranges. Every other slot
// stays Unknown.
template <size_t N, size_t R>
constexpr FormatTable<N> dense(Rdna2Format fmt, const Row (&rows)[R]) {
    FormatTable<N> table;
    for (const ReservedRange& range : kReserved)
        if (range.fmt == fmt)
            for (uint32_t opcode = range.first; opcode <= range.last; ++opcode)
                table.by_opcode[opcode] = reserved();
    for (const Row& row : rows) table.by_opcode[row.opcode] = row.effects;
    return table;
}

constexpr auto kSop2Table = dense<0x100>(F::SOP2, kSop2);
constexpr auto kSop1Table = dense<0x100>(F::SOP1, kSop1);
constexpr auto kSopkTable = dense<0x100>(F::SOPK, kSopk);
constexpr auto kSopcTable = dense<0x100>(F::SOPC, kSopc);
constexpr auto kSoppTable = dense<0x100>(F::SOPP, kSopp);
constexpr auto kSmemTable = dense<0x100>(F::SMEM, kSmem);
constexpr auto kVop2Table = dense<0x100>(F::VOP2, kVop2);
constexpr auto kVop1Table = dense<0x100>(F::VOP1, kVop1);
constexpr auto kVopcTable = dense<0x100>(F::VOPC, kVopc);
constexpr auto kVop3Table = dense<0x400>(F::VOP3, kVop3);
constexpr auto kVop3pTable = dense<0x100>(F::VOP3P, kVop3p);

constexpr auto kDsTable = dense<0x100>(F::DS, kDs);
constexpr auto kMubufTable = dense<0x100>(F::MUBUF, kMubuf);
constexpr auto kMtbufTable = dense<0x10>(F::MTBUF, kMtbuf);
constexpr auto kMimgTable = dense<0x100>(F::MIMG, kMimg);
constexpr auto kFlatTable = dense<0x80>(F::FLAT, kFlat);
constexpr auto kVintrpTable = dense<0x4>(F::VINTRP, kVintrp);
constexpr auto kExpTable = dense<0x1>(F::EXP, kExp);

Rdna2OpcodeRange range_for(Rdna2Format fmt) {
    switch (fmt) {
        case F::SOP2: return kSop2Range;
        case F::SOP1: return kSop1Range;
        case F::SOPK: return kSopkRange;
        case F::SOPC: return kSopcRange;
        case F::SOPP: return kSoppRange;
        case F::SMEM: return kSmemRange;
        case F::VOP2: return kVop2Range;
        case F::VOP1: return kVop1Range;
        case F::VOPC: return kVopcRange;
        case F::VOP3: return kVop3Range;
        case F::VOP3P: return kVop3pRange;
        case F::DS: return {0, 0x100};
        case F::MUBUF: return {0, 0x100};
        case F::MTBUF: return {0, 0x10};
        case F::MIMG: return {0, 0x100};
        case F::FLAT: return {0, 0x80};
        case F::VINTRP: return {0, 0x4};
        case F::EXP: return {0, 0x1};
        default: return {};
    }
}

template <size_t N>
const Rdna2InstructionEffects* lookup(const FormatTable<N>& table, uint32_t opcode) {
    return opcode < N ? &table.by_opcode[opcode] : nullptr;
}

const Rdna2InstructionEffects* entry_for(Rdna2Format fmt, uint32_t opcode) {
    const Rdna2OpcodeRange range = range_for(fmt);
    if (opcode < range.first || opcode >= range.end) return nullptr;
    switch (fmt) {
        case F::SOP2: return lookup(kSop2Table, opcode);
        case F::SOP1: return lookup(kSop1Table, opcode);
        case F::SOPK: return lookup(kSopkTable, opcode);
        case F::SOPC: return lookup(kSopcTable, opcode);
        case F::SOPP: return lookup(kSoppTable, opcode);
        case F::SMEM: return lookup(kSmemTable, opcode);
        case F::VOP2: return lookup(kVop2Table, opcode);
        case F::VOP1: return lookup(kVop1Table, opcode);
        case F::VOPC: return lookup(kVopcTable, opcode);
        case F::VOP3: return lookup(kVop3Table, opcode);
        case F::VOP3P: return lookup(kVop3pTable, opcode);
        case F::DS: return lookup(kDsTable, opcode);
        case F::MUBUF: return lookup(kMubufTable, opcode);
        case F::MTBUF: return lookup(kMtbufTable, opcode);
        case F::MIMG: return lookup(kMimgTable, opcode);
        case F::FLAT: return lookup(kFlatTable, opcode);
        case F::VINTRP: return lookup(kVintrpTable, opcode);
        case F::EXP: return lookup(kExpTable, opcode);
        default: return nullptr;
    }
}

std::vector<Row> all_rows() {
    std::vector<Row> rows;
    auto append = [&](const auto& list) {
        for (const Row& row : list) rows.push_back(row);
    };
    append(kSop2);
    append(kSop1);
    append(kSopk);
    append(kSopc);
    append(kSopp);
    append(kSmem);
    append(kVop2);
    append(kVop1);
    append(kVopc);
    append(kVop3);
    append(kVop3p);
    append(kDs);
    append(kMubuf);
    append(kMtbuf);
    append(kMimg);
    append(kFlat);
    append(kVintrp);
    append(kExp);
    for (const ReservedRange& range : kReserved)
        for (uint32_t opcode = range.first; opcode <= range.last; ++opcode)
            rows.push_back({range.fmt, opcode, reserved()});
    return rows;
}

const Rdna2InstructionEffects kUnknown{};

// Formats with no scalar-ALU encoding. None of them reads or writes SCC except through an explicit
// operand (ISA 70648 3.8: "Most scalar ALU instructions set the Scalar Condition Code"; VALU
// compares write VCC/EXEC instead, 3.9).
bool never_touches_scc(Rdna2Format fmt) {
    switch (fmt) {
        case F::SMEM:
        case F::VOP2:
        case F::VOP1:
        case F::VOPC:
        case F::VOP3:
        case F::VOP3P:
        case F::VINTRP:
        case F::DS:
        case F::MUBUF:
        case F::MTBUF:
        case F::MIMG:
        case F::FLAT:
        case F::EXP: return true;
        default: return false;
    }
}
