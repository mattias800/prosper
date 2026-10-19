// AMD ISA 70648: register-file widths, not backend support or validity policy.
using MA = Rdna2MemoryAccess;
using AD = Rdna2MemoryAddress;
using MD = Rdna2MemoryData;
using MH = Rdna2MemoryHalf;

constexpr Rdna2InstructionEffects memory(const char* name, MA access, AD address, uint8_t read,
                                         uint8_t write, uint8_t bits = 0) {
    Rdna2InstructionEffects e;
    e.status = Rdna2EffectsStatus::Memory;
    e.name = name;
    e.memory.described = true;
    e.memory.access = access;
    e.memory.address = address;
    e.memory.data_read_dwords = read;
    e.memory.data_write_dwords = write;
    e.memory.element_bits = bits;
    e.memory.implicit_reads = access == MA::None ? 0 : kExec;
    return e;
}
constexpr Rdna2InstructionEffects ds(const char* name, MA access, uint8_t addr, uint8_t d0,
                                     uint8_t d1, uint8_t result, uint8_t bits) {
    auto e = memory(name, access, AD::Fixed, 0, result, bits);
    e.memory.vsrc_dwords[0] = addr;
    e.memory.vsrc_dwords[1] = d0;
    e.memory.vsrc_dwords[2] = d1;
    e.memory.gds_reads_m0 = access != MA::None && access != MA::CrossLane;
    return e;
}
constexpr Rdna2InstructionEffects mem_m0(Rdna2InstructionEffects e) {
    e.memory.implicit_reads |= kM0;
    return e;
}
constexpr Rdna2InstructionEffects mem_half(Rdna2InstructionEffects e, MH half) {
    e.memory.result_half = half;
    return e;
}
constexpr Rdna2InstructionEffects buffer(const char* name, MA access, uint8_t read, uint8_t write,
                                         uint8_t bits, bool lds_load) {
    auto e = memory(name, access, AD::Buffer, read, write, bits);
    e.memory.ssrc_dwords[1] = 4;
    e.memory.ssrc_dwords[2] = 1;
    e.memory.write_if_glc = access == MA::Atomic;
    e.memory.lds_redirect = lds_load && access == MA::Load;
    e.memory.status = access == MA::Load;
    return e;
}
constexpr Rdna2InstructionEffects flat(const char* name, MA access, uint8_t read, uint8_t write,
                                       uint8_t bits, uint8_t segments) {
    auto e = memory(name, access, AD::Flat, read, write, bits);
    e.memory.flat_segments = segments;
    e.memory.write_if_glc = access == MA::Atomic;
    e.memory.lds_redirect = access == MA::Load;
    return e;
}
constexpr Rdna2InstructionEffects mem_addtid(Rdna2InstructionEffects e) {
    e.memory.address = AD::FlatAddtid;
    return e;
}
// These LLVM-recognized encodings are absent from ISA 70648. Naming them is not an effects
// claim: a future transfer must refuse or make all possible writes hard, never assume no effect.
constexpr Rdna2InstructionEffects unmodeled(const char* name) {
    Rdna2InstructionEffects e;
    e.status = Rdna2EffectsStatus::Unmodeled;
    e.name = name;
    return e;
}
