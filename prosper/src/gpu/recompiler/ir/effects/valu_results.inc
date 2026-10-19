// Scalar destinations kept separate from legacy scalar-ALU fields (ISA 3.9, 6.2.3, 12.7-12.12).
constexpr Rdna2InstructionEffects scalar_output(Rdna2InstructionEffects e,
                                                Rdna2ScalarDestination destination, uint8_t words,
                                                bool mask = true) {
    e.scalar_result = {destination, words, mask};
    return e;
}
// VOP3 interpolation uses ordinary VALU operand fields for its explicit register operands;
// the LDS attribute selector is not a register (ISA 12.11, 12.12, 10.3).
constexpr Rdna2InstructionEffects interpolated(Rdna2InstructionEffects e, uint8_t attribute_bits,
                                               bool old_vdst = false) {
    e.memory.described = true;
    e.memory.access = Rdna2MemoryAccess::Interpolate;
    for (uint32_t i = 0; i < 3; ++i) {
        e.memory.vsrc_dwords[i] = e.ssrc_dwords[i];
        e.memory.ssrc_dwords[i] = e.ssrc_dwords[i];
    }
    e.memory.data_write_dwords = e.vdst_dwords;
    e.memory.implicit_reads = kM0 | kExec;
    e.memory.element_bits = attribute_bits;
    e.memory.reads_old_vdst = old_vdst;
    return e;
}
constexpr Rdna2InstructionEffects vcmpx(const char* name, uint8_t s0, uint8_t s1) {
    return scalar_output(vcmp(name, s0, s1), Rdna2ScalarDestination::Exec, 2);
}
