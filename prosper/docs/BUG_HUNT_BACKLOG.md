# History bug-hunt: verified-active findings backlog

A full review of all 405 mainline commits (oldest → newest, 14 parallel review passes,
2026-07-08) hunting defects, hacks, hardcoded stand-ins, and silent simplifications.
Every finding below was **verified to still exist on master @ 96c80b2** before being
recorded; candidates that later history already fixed were discarded (~60 of them).

Severity legend: how likely it is to bite + how silent the failure is.
Tags: [defect] wrong behavior · [hack] shortcut standing in for real behavior ·
[hardcoded] magic value where a real parsed/computed value belongs ·
[simplification] silently narrower semantics than the real API/ISA.

## Fixed on branch `fix/history-bug-hunt` (18 bugs, 15 commits)

- MUBUF SOFFSET decoded with a 7-bit mask — inline-0 (0x80) read as SGPR s0.
- R_X86_64_DTPOFF64 computed as S *or* A instead of S + A.
- v_mul_legacy_f32 emitted as IEEE FMul (0×Inf gave NaN, not 0).
- sceSystemServiceGetStatus wrote 4 bytes through stale RSI, real out-struct unwritten.
- __cxa_guard_acquire / _Execute_once raced (concurrent double-construction).
- FreeBSD→Linux open(2) flag words passed raw (O_CREAT became O_TRUNC).
- sceKernelWaitEventFlag / WaitSema ignored their timeout (silent forever-block).
- scePthread(Attr)Getschedparam/Getprio returned success with unwritten out-params.
- EVFILT_VIDEO_OUT was -10 (FreeBSD EVFILT_LIO); FreeBSD-derived constants and independent implementations agree on -13.
- guest_readable/probe_readable probed via /dev/null, which never faults → all guards
  were no-ops (now a drained O_NONBLOCK pipe; raw syscalls in the fault handler).
- Lazy GPU-VA backing ignored si_code (clobbered live RO mappings; infinite fault loop
  on in-window instruction fetches). Now SEGV_MAPERR-only and addr != RIP.
- SceAudioOutPortState wrote volume into the flag field instead of the ABI's volume field; audio
  errors were generic -1 instead of 0x80260003/0x80260005.
- Guest-%fs swap stubs shifted stack args by 2 qwords — ReleaseMem's (data_sel, fence
  value) read a TCB pointer + return address under PROSPER_GUEST_FS. Stub now forwards
  the first two stack args (verified live: data_sel=0x2/0x3 with real fence values).
- DrawIndexAuto dropped its modifier and left cmd[3..6] uninitialized.
- WriteData silently clamped to 60 dwords (now accepts PM4 14-bit max, rejects loudly).
- AGC shader registry iterated lock-free against push_back (UAF on realloc).
- s_waitcnt_vscnt mapped as nonexistent SOPP 0x7d — real one (SOPK 0x17) failed the
  shader; waitcnt family now no-op'd in SOPK. Plus: execz linearization now rejects
  VALU ops with unpredicated scalar writes (readfirstlane/carry ops); v_cndmask_e64
  honors neg/abs modifiers; VOP3B untracked carry-in rejects instead of assuming 0.
- Direct vertex-buffer provenance keyed without user_sgpr_base (off by 8 in VS).
- munmap/BatchMap-UNMAP never untracked g_maps; mprotect never re-tagged (stale
  "committed" state defeated safe_copy and the lazy-commit fault probe).

## High-priority remaining

1. **[hack] APR read resolves the file by BYTE-SIZE match** — `hle_file.cpp`
   f_apr_read_submit: request matched against stat'd sizes of known containers;
   whole-file reads at offset 0 only. SAFE-FALLBACK LANDED (#62): size collisions and
   partial/chunk reads are now detected and refused loudly (EINVAL + unconditional
   stderr), resolve dedups re-registered paths, and PROSPER_FILELOG dumps the 0x90-byte
   descriptor buffer for the remaining RE. Still open: decode the real (file id, offset)
   from the descriptor buffer / Ampr CB read command and read via
   prosper_apr_path_for_id — required for the compressed .ucas chunk reads.
   (introduced 41b01f3)
2. **[simplification] Indexed draws silently dropped** — hle_agc.cpp emits R_DRAW_INDEX
   (0x03) but pm4_decode never decodes it: every sceAgcDcbDrawIndex vanishes downstream
   while the guest sees success. Decode it (verify a2/a3 roles from wrapper disassembly and live calls), push a Draw
   with index data, round-trip test. Directly relevant to scene-content rendering.
   (8d26489)
3. **[hack] Quad-fan topology heuristic** — gpu_execute.hpp:107-112 rewrites any draw
   whose bound VB has 4 records to vertex_count=4 TRIANGLE_FAN. Stands in for real
   index-buffer support; wrong for real 4-vertex meshes. Supersede with #2 then delete.
   (b1f3420)
4. **[hardcoded] Sampled textures assumed RGBA8** — agc_shader_layout.cpp:144,151: the
   decoded 9-bit T# format is thrown away; everything uploads as Unorm8 ×4 with
   size = w*h*4 (over-reads BCn allocations up to 8×). Add a Gen5 IMG_FMT→DataFormat
   mapper from public format definitions and captures, size from block size, loud skip on unmapped. (13d70f7)
5. **[defect] release-mem data_sel default write** — command_processor.cpp:58-67 writes
   8 bytes for ANY unrecognized data_sel and cases 1/2 lack the rel_value_valid guard.
   With the stub-frame fix in, data_sel now decodes correctly — make the default
   log-and-skip and guard cases 1/2. (858c47e)
6. **[defect] Equeue lifetime + semantics cluster** — hle_kernel_time.cpp:
   (a) k_eq_delete deletes EqState while waiters may sit in wait (UAF; registrations in
   g_*_regs never purged — address reuse resurrects them); (b) WaitEqueue returns
   SUCCESS with *out=0 on timeout/unknown queue (the platform-compatible result is ETIMEDOUT) and
   caps NULL timeout at 100 ms returning success; (c) eq_post drops the NEWEST event
   when 4 queued (a flip event can be the one dropped — coalesce by ident/filter
   instead); (d) detached HR-timer threads post to possibly-deleted queues and
   Delete*Event are no-ops. shared_ptr EqState + purge + ETIMEDOUT + coalescing.
   (61e7ee8, 534bcc1)
7. **[defect] TLS DTV keyed by recyclable thread id, never purged** — hle_kernel.cpp
   __tls_get_addr: pthread id reuse hands a new thread the dead thread's dirty TLS
   block; blocks leak. Purge at thread exit (trampoline) or key by a monotonic token.
   (a76140b/6495dbc; flagged by three passes)
8. **[simplification] pthread_once / _Execute_once-style global-lock deadlock** —
   k_pthread_once holds ONE global recursive mutex across the guest init routine;
   cross-thread-dependent inits deadlock. Per-control state (3-state word + condvar),
   callback outside the lock (h_execute_once now shows the shape). (5fae501)
9. **[defect] WaitRegMem args destroyed at build time** — hle_agc.cpp zeroes the whole
   9-dword payload; the wait can never be honored downstream. Encode the captured wrapper/packet layout;
   CP should at least assert the condition already holds at fold time. (2e4b9de)
10. **[hardcoded] Ampr mirror view at fixed −0x540000000** — hle_kernel_mem.cpp:431:
    one-title constant, MAP_FIXED over whatever lives there. Track the guest's real
    second view via the phys-keyed memfd instead. (a51d7ce)

## Medium

- [defect] init-fault dump derefs unchecked g_fault_rip (mprotect result ignored;
  forces R|X on RW pages) — exec_image_linux.cpp:~1290. Probe first; restore prot. (b51ad10)
- [hardcoded] RTC/time: wall clock = uptime + 1700000000 (frozen Nov 2023), three
  disagreeing time sources; LocalTime ignores tz — hle_kernel_time.cpp:43-96. Base off
  host CLOCK_REALTIME once at startup. (e1473f8/efaf7cf)
- [simplification] forward-branch clamp swallows real code past s_endpgm —
  rdna2_to_spirv.cpp:~968: clamp only tgt == end_pc+1, else reject. (4e65a94)
- [defect] flip-status triple (g_flip_count/g_current_buffer/g_last_flip_arg) is
  read/written by 3 threads with no sync — hle_graphics.cpp. Mutex or atomic seq. (dcff92e)
- [simplification] untyped MUBUF loads/stores hardcode binding 2 (never resolve SRSRC)
  — rdna2_to_spirv.cpp:~1648. Mirror the format-load provenance resolution. (e9b62dd)
- [hardcoded] LDS fixed at 16 KB (RDNA2 allows 64 KB; size is per-shader RSRC2 data) —
  rdna2_to_spirv.cpp declare_lds. (fbfd8da)
- [defect] robustImageAccess required by the recompiler's storage-image contract but
  never enabled at either vkCreateDevice — render_runner.h:113, image_compute_runner.h:57. (e3951c1)
- [simplification] EventWrite drops its address arg (label-carrying events can never
  write) — hle_agc.cpp:136. (2e4b9de)
- [hardcoded] vk_color_format maps ONLY 8_8_8_8 (else silently Undefined) —
  vk_translate.cpp:20. Fill rows from public format definitions and validated captures; log unmapped. (822cd24)
- [simplification] Special-operand ALU sources (VCC/EXEC/M0 as data) silently read 0 —
  rdna2_to_spirv.cpp operand_bits default. Route to tracked bools or reject. (e13f469)
- [defect] v_cvt_u32_f32/i32_f32 emit bare OpConvertFToU/S (UB out-of-range; hardware
  saturates, NaN→0) — rdna2_to_spirv.cpp:~1266. Clamp + NaN→0. (ac1e536)
- [simplification] DTPMOD64 always resolves to the local module; the promised
  "unhandled" log for cross-module TLS doesn't exist — module.cpp:237. (d499c1d)
- [defect] mmap hint forces MAP_FIXED (silently clobbers live mappings) — map_at,
  hle_kernel_mem.cpp:124. MAP_FIXED_NOREPLACE unless replacing own reservation. (cf2495c)
- [defect] guest thread stacks (8 MiB) never freed; g_stacks never erased; attr
  stacksize ignored — hle_kernel.cpp:281. (caa9443)
- [defect] EventFlag/Sema delete frees under active waiters (UAF); Sony wakes with
  WAIT_DELETE first — hle_kernel.cpp:335/369. (caa9443)
- [defect] sceKernelAvailableDirectMemorySize aliased to GetSize (out-params never
  written) — hle_kernel_mem.cpp. Walk g_dmem gaps. (cf2495c)
- [simplification] printf-family forwards only integer regs (floats/stack args read
  garbage; %s can fault) — hle_libc.cpp:199. Needs a va-marshaling shim. (3e16200)
- [simplification] alloc_dmem ignores searchStart/End — constrain the gap walk. (efaf7cf)
- [simplification] wait_on_address timeout honor still gated OFF behind
  PROSPER_WAIT_TIMEOUT with an obsolete justification (unwinder long fixed) + 5 s cap —
  hle_kernel_mem.cpp:334. Flip the default. (0e121a5)
- [simplification] folded multi-draw uses draws[0].index_count with the LAST draw's
  state (chimera draw) — gpu_execute.hpp:136. Use consistent provenance / PERDRAW default. (d99e3c6)
- [defect] detached timer threads + eq delete UAF (see cluster #6). (534bcc1)
- [simplification] type-0 AGC data packets get the type-1 payload offset (never RE'd) —
  hle_agc.cpp agc_get_data_packet_payload. Per-type offset or loud diagnostic. (54b7b81)
- [defect] mb_cur_max returns a POINTER (address of static) where the value is the
  contract — hle_libc.cpp:233. (5b781e4)
- [simplification] Windows k_wait_on_address fallback drops the timeout + global cv —
  hle_kernel_mem.cpp #else branch. (8e7a661)
- [simplification] guest static-TLS layout hardcodes 16-byte alignment instead of
  PT_TLS p_align (Variant II tpoff mismatch for p_align>16) — guest_tls.cpp:47. (45016d8)

## Lower priority / latent

- vblank count advances per POLL, not per 16.67 ms — hle_graphics.cpp:248. (d83b2e6)
- sceVideoOutGetOutputStatus returns success without writing the out-struct —
  hle_graphics.cpp:311. (d83b2e6)
- MsgDialog reports FINISHED before any Open — hle_service.cpp. 3-state lifecycle. (1b14b79)
- all guest mutexes forced RECURSIVE (settype discarded; static sentinels too) —
  hle_kernel.cpp:55,73-89. Map sentinel/attr → real type. (cf2495c/8d26489)
- LoadStartModule returns a fake success handle for unknown paths — ENOENT for misses. (e1473f8)
- dlsym ignores the module handle (global first-wins table) — per-module exports. (b51ad10)
- pad handle ignored (always backend index 0); getenv per poll — hle_pad.cpp:59. (8564d73)
- MUBUF idxen+offen drops the second VADDR (per-lane byte offset) — reject or decode
  both. (66e82a0)
- SMEM register SOFFSET never decoded (silently dropped — can't even reject) and the
  21-bit imm treated unsigned — rdna2_decode.cpp:149. (3bff9d0)
- packed vertex attributes assume dword-aligned element addresses — rdna2_to_spirv.cpp
  packed load/store paths. Fold addr&3 or reject. (08b1f5c)
- image_sample → ImplicitLod even in compute/vertex shells (invalid SPIR-V) — gate on
  fragment, else explicit LOD 0. (f8bcad1)
- v_interp_mov ignores its P0/P10/P20 selector (flat reads interpolated). (636cf47)
- v_readfirstlane models "this lane" (marked SPECULATIVE) — use
  OpGroupNonUniformBroadcastFirst where available. (836394c)
- NGG vertex index hardcoded to v5 via an s_sendmsg heuristic — derive from
  SPI_SHADER_PGM_RSRC/GE state. (6b29017)
- s_bfe_u64 fold zero-extends negative inline constants (ISA sign-extends). (f8c7371)
- gpu_clock64 writes steady_clock ns, not GPU-tick units (data_sel=3 deltas wrong). (3685104)
- detile treats all tile modes except 5 as linear, silently; 32 bpp only — log
  unrecognized modes once. (1ecc260)
- VS texture-first binding collision with hardwired SB bindings 2/3 —
  assign_convention_bindings should start at 4. (d99e3c6)
- dynfetch: fixed 0x4000-dword code walk (bound by AgcShaderHeader size), Unknown
  format → Float32×4 default, size fallback "stride*4" — gpu_executor.cpp:344-386. (21589ee)
- two glibc write() + one getenv() left inside the fault handler's HWBP diag paths
  (guest-%fs unsafe) — exec_image_linux.cpp:~489/~580. (b51ad10)
- shader-capture dump path hardcoded to this machine's WSL layout — hle_graphics.cpp:384. (7089e48)
- three disagreeing "now" sources (see RTC above) — clock_gettime ignores clockid. (e1473f8)
