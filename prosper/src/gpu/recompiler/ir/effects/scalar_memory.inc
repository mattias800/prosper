// ---- SMEM: ISA 12.6 (loads, s_gl1_inv, s_dcache_inv, s_memtime, s_memrealtime, s_atc_probe*)
// plus the gfx10.1 forms ISA 70648 dropped (scalar stores and atomics, s_scratch_*,
// s_dcache_wb/discard, s_get_waveid_in_workgroup): PS5 code is gfx10.1-era where the two differ
// (see the VALU note below), so the rows are the UNION of llvm-mc -mcpu=gfx1030 and gfx1013, swept
// over the whole 8-bit opcode field with and without operands. The local corpus uses only the
// loads. No SMEM instruction touches SCC, EXEC or VCC. CONFIDENCE: HIGH for the widths: each is
// the register class llvm-mc prints, and the ISA says "Return current 64-bit timestamp" for
// s_memtime and s_memrealtime. A gfx10.1 atomic's SDATA is its data (cmpswap: data and compare,
// twice the returned width) and receives the old value only with GLC; the row records that
// returned width as the destination, so a may-write question sees it.
constexpr Row kSmem[] = {
    {F::SMEM, 0x00, smem("s_load_dword", Rdna2ScalarMemory::Load, 1, 2, 1)},
    {F::SMEM, 0x01, smem("s_load_dwordx2", Rdna2ScalarMemory::Load, 2, 2, 1)},
    {F::SMEM, 0x02, smem("s_load_dwordx4", Rdna2ScalarMemory::Load, 4, 2, 1)},
    {F::SMEM, 0x03, smem("s_load_dwordx8", Rdna2ScalarMemory::Load, 8, 2, 1)},
    {F::SMEM, 0x04, smem("s_load_dwordx16", Rdna2ScalarMemory::Load, 16, 2, 1)},
    {F::SMEM, 0x05,
     smem("s_scratch_load_dword", Rdna2ScalarMemory::ScratchLoad, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x06,
     smem("s_scratch_load_dwordx2", Rdna2ScalarMemory::ScratchLoad, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0x07,
     smem("s_scratch_load_dwordx4", Rdna2ScalarMemory::ScratchLoad, 4, 2, 1)},   // gfx1013
    {F::SMEM, 0x08, smem("s_buffer_load_dword", Rdna2ScalarMemory::Load, 1, 4, 1)},
    {F::SMEM, 0x09, smem("s_buffer_load_dwordx2", Rdna2ScalarMemory::Load, 2, 4, 1)},
    {F::SMEM, 0x0a, smem("s_buffer_load_dwordx4", Rdna2ScalarMemory::Load, 4, 4, 1)},
    {F::SMEM, 0x0b, smem("s_buffer_load_dwordx8", Rdna2ScalarMemory::Load, 8, 4, 1)},
    {F::SMEM, 0x0c, smem("s_buffer_load_dwordx16", Rdna2ScalarMemory::Load, 16, 4, 1)},
    {F::SMEM, 0x10, smem("s_store_dword", Rdna2ScalarMemory::Store, 0, 2, 1)},   // gfx1013
    {F::SMEM, 0x11, smem("s_store_dwordx2", Rdna2ScalarMemory::Store, 0, 2, 1)},   // gfx1013
    {F::SMEM, 0x12, smem("s_store_dwordx4", Rdna2ScalarMemory::Store, 0, 2, 1)},   // gfx1013
    {F::SMEM, 0x15, smem("s_scratch_store_dword", Rdna2ScalarMemory::Store, 0, 2, 1)},   // gfx1013
    {F::SMEM, 0x16,
     smem("s_scratch_store_dwordx2", Rdna2ScalarMemory::Store, 0, 2, 1)},   // gfx1013
    {F::SMEM, 0x17,
     smem("s_scratch_store_dwordx4", Rdna2ScalarMemory::Store, 0, 2, 1)},   // gfx1013
    {F::SMEM, 0x18, smem("s_buffer_store_dword", Rdna2ScalarMemory::Store, 0, 4, 1)},   // gfx1013
    {F::SMEM, 0x19, smem("s_buffer_store_dwordx2", Rdna2ScalarMemory::Store, 0, 4, 1)},   // gfx1013
    {F::SMEM, 0x1a, smem("s_buffer_store_dwordx4", Rdna2ScalarMemory::Store, 0, 4, 1)},   // gfx1013
    {F::SMEM, 0x1f, smem("s_gl1_inv", Rdna2ScalarMemory::CacheControl, 0, 0, 0)},
    {F::SMEM, 0x20, smem("s_dcache_inv", Rdna2ScalarMemory::CacheControl, 0, 0, 0)},
    {F::SMEM, 0x21, smem("s_dcache_wb", Rdna2ScalarMemory::CacheControl, 0, 0, 0)},   // gfx1013
    {F::SMEM, 0x24, smem("s_memtime", Rdna2ScalarMemory::HardwareRead, 2, 0, 0)},
    {F::SMEM, 0x25, smem("s_memrealtime", Rdna2ScalarMemory::HardwareRead, 2, 0, 0)},
    {F::SMEM, 0x26, smem("s_atc_probe", Rdna2ScalarMemory::Probe, 0, 2, 1)},
    {F::SMEM, 0x27, smem("s_atc_probe_buffer", Rdna2ScalarMemory::Probe, 0, 4, 1)},
    {F::SMEM, 0x28,
     smem("s_dcache_discard", Rdna2ScalarMemory::CacheControl, 0, 2, 1)},   // gfx1013
    {F::SMEM, 0x29,
     smem("s_dcache_discard_x2", Rdna2ScalarMemory::CacheControl, 0, 2, 1)},   // gfx1013
    {F::SMEM, 0x2a,
     smem("s_get_waveid_in_workgroup", Rdna2ScalarMemory::HardwareRead, 1, 0, 0)},   // gfx1013
    {F::SMEM, 0x40, smem("s_buffer_atomic_swap", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x41,
     smem("s_buffer_atomic_cmpswap", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x42, smem("s_buffer_atomic_add", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x43, smem("s_buffer_atomic_sub", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x44, smem("s_buffer_atomic_smin", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x45, smem("s_buffer_atomic_umin", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x46, smem("s_buffer_atomic_smax", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x47, smem("s_buffer_atomic_umax", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x48, smem("s_buffer_atomic_and", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x49, smem("s_buffer_atomic_or", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x4a, smem("s_buffer_atomic_xor", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x4b, smem("s_buffer_atomic_inc", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x4c, smem("s_buffer_atomic_dec", Rdna2ScalarMemory::Atomic, 1, 4, 1)},   // gfx1013
    {F::SMEM, 0x60,
     smem("s_buffer_atomic_swap_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x61,
     smem("s_buffer_atomic_cmpswap_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x62,
     smem("s_buffer_atomic_add_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x63,
     smem("s_buffer_atomic_sub_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x64,
     smem("s_buffer_atomic_smin_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x65,
     smem("s_buffer_atomic_umin_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x66,
     smem("s_buffer_atomic_smax_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x67,
     smem("s_buffer_atomic_umax_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x68,
     smem("s_buffer_atomic_and_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x69, smem("s_buffer_atomic_or_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x6a,
     smem("s_buffer_atomic_xor_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x6b,
     smem("s_buffer_atomic_inc_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x6c,
     smem("s_buffer_atomic_dec_x2", Rdna2ScalarMemory::Atomic, 2, 4, 1)},   // gfx1013
    {F::SMEM, 0x80, smem("s_atomic_swap", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x81, smem("s_atomic_cmpswap", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x82, smem("s_atomic_add", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x83, smem("s_atomic_sub", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x84, smem("s_atomic_smin", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x85, smem("s_atomic_umin", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x86, smem("s_atomic_smax", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x87, smem("s_atomic_umax", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x88, smem("s_atomic_and", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x89, smem("s_atomic_or", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x8a, smem("s_atomic_xor", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x8b, smem("s_atomic_inc", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0x8c, smem("s_atomic_dec", Rdna2ScalarMemory::Atomic, 1, 2, 1)},   // gfx1013
    {F::SMEM, 0xa0, smem("s_atomic_swap_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xa1, smem("s_atomic_cmpswap_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xa2, smem("s_atomic_add_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xa3, smem("s_atomic_sub_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xa4, smem("s_atomic_smin_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xa5, smem("s_atomic_umin_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xa6, smem("s_atomic_smax_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xa7, smem("s_atomic_umax_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xa8, smem("s_atomic_and_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xa9, smem("s_atomic_or_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xaa, smem("s_atomic_xor_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xab, smem("s_atomic_inc_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
    {F::SMEM, 0xac, smem("s_atomic_dec_x2", Rdna2ScalarMemory::Atomic, 2, 2, 1)},   // gfx1013
};

