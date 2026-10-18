// ISA 70648 12.11 (VINTRP instruction descriptions), 13.4 Table 93 and chapter 11 EXP.
// CONFIDENCE: HIGH. MOV's VSRC is a selector, not a VGPR; P2 reads its old destination.
constexpr Rdna2InstructionEffects interpolate(const char* name, uint8_t vsrc, bool accumulator) {
    auto e = memory(name, MA::Interpolate, AD::Fixed, 0, 1, 32);
    e.memory.vsrc_dwords[0] = vsrc;
    e.memory.implicit_reads |= kM0;
    e.memory.reads_old_vdst = accumulator;
    return e;
}
constexpr Row kVintrp[] = {
    {F::VINTRP, 0, interpolate("v_interp_p1_f32", 1, false)},
    {F::VINTRP, 1, interpolate("v_interp_p2_f32", 1, true)},
    {F::VINTRP, 2, interpolate("v_interp_mov_f32", 0, false)},
};
constexpr Row kExp[] = {
    {F::EXP, 0, memory("exp", MA::Export, AD::Export, 0, 0)},
};
