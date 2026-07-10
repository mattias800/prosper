cmake_minimum_required(VERSION 3.20)
project(prosper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Agentic-first: binaries must run with no PATH/DLL setup so CI (ctest) is hermetic.
# On mingw, statically link the GCC/C++ runtimes.
if(MINGW OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
  add_link_options(-static -static-libgcc -static-libstdc++)
endif()

# --- tools ---------------------------------------------------------------
# self_dump: SELF/ELF inspector and NID import extractor (M0).
add_executable(self_dump tools/self_dump/self_dump.cpp)

# --- core library (M1+) --------------------------------------------------
# Loader / HLE / kernel / gpu sources accrete here per the roadmap.
file(GLOB_RECURSE PROSPER_SRC CONFIGURE_DEPENDS src/*.cpp)
if(PROSPER_SRC)
  add_library(prosper_core STATIC ${PROSPER_SRC})
  target_include_directories(prosper_core PUBLIC src)
  target_compile_definitions(prosper_core PRIVATE PROSPER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  find_package(Threads REQUIRED)
  target_link_libraries(prosper_core PUBLIC Threads::Threads)
endif()

# --- tests (agentic-first: self-checking, exit-code = truth) -------------
enable_testing()
# Path to the (gitignored) local game dump; override with -DGAME_DUMP=...
set(GAME_DUMP "${CMAKE_SOURCE_DIR}/../PPSA24651-app0" CACHE PATH "Local PS5 game dump root")
set(HAVE_GAME_DUMP OFF)
if(EXISTS "${GAME_DUMP}/eboot.bin")
  set(HAVE_GAME_DUMP ON)
else()
  message(STATUS "Game dump not found at ${GAME_DUMP} — skipping dump-backed tests")
endif()

if(TARGET prosper_core)
  add_executable(test_module tests/test_module.cpp)
  target_link_libraries(test_module PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    add_test(NAME module_loads_eboot
             COMMAND test_module "${GAME_DUMP}/eboot.bin")
  endif()

  add_executable(test_nid tests/test_nid.cpp)
  target_link_libraries(test_nid PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    add_test(NAME nid_hash_matches_imports
             COMMAND test_nid "${GAME_DUMP}/eboot.bin")
  endif()

  add_executable(test_hle_registered tests/test_hle_registered.cpp)
  target_link_libraries(test_hle_registered PRIVATE prosper_core)
  add_test(NAME hle_functions_registered COMMAND test_hle_registered)

  # No NID registered twice with a DIFFERENT handler (the #330 shadow class): a later register_*()
  # silently overwriting a real impl with a stub returns garbage to the guest. Pure, no game dump.
  add_executable(test_hle_no_shadow tests/test_hle_no_shadow.cpp)
  target_link_libraries(test_hle_no_shadow PRIVATE prosper_core)
  add_test(NAME hle_no_shadow COMMAND test_hle_no_shadow)

  # Cooperative stop signal (issue #164 P0): request_stop/stop_requested the frontend uses to wind
  # the guest run-loop down on window-close. Pure, no deps.
  add_executable(test_lifecycle tests/test_lifecycle.cpp)
  target_link_libraries(test_lifecycle PRIVATE prosper_core)
  add_test(NAME lifecycle_stop_signal COMMAND test_lifecycle)

  # va2foff resolution order (issue #113): DYNAMIC/PROCPARAM/TLS must never shadow
  # the PT_LOAD that truly maps a VA. Pure unit test — no game dump required.
  add_executable(test_va2foff tests/test_va2foff.cpp)
  target_link_libraries(test_va2foff PRIVATE prosper_core)
  add_test(NAME va2foff_load_priority COMMAND test_va2foff)

  # SELF data-segment key masks flag bits >= 32 (#339): the SCE segment id is bits [31:20]; without
  # the mask a data segment keys the map wrong and every phdr lookup misses. Pure, no game dump.
  add_executable(test_self_segment_key tests/test_self_segment_key.cpp)
  target_link_libraries(test_self_segment_key PRIVATE prosper_core)
  add_test(NAME self_segment_key COMMAND test_self_segment_key)

  # Guest static-TLS Variant II layout uses each module's real PT_TLS p_align (#143). The layout
  # code is Linux-only (guest %fs), so gate the test on it too.
  if(UNIX AND NOT APPLE)
    add_executable(test_guest_tls_align tests/test_guest_tls_align.cpp)
    target_link_libraries(test_guest_tls_align PRIVATE prosper_core)
    add_test(NAME guest_tls_align COMMAND test_guest_tls_align)
  endif()

  # DTPMOD64 cross-module TLS resolution (#136): a TLS import resolves to the DEFINING module's
  # tls_modid, not the relocating module's. Pure unit test (synthetic Module/LoadedImage).
  add_executable(test_dtpmod_tls tests/test_dtpmod_tls.cpp)
  target_link_libraries(test_dtpmod_tls PRIVATE prosper_core)
  add_test(NAME dtpmod_tls_cross_module COMMAND test_dtpmod_tls)

  # AMD SSE4a INSERTQ/EXTRQ emulation (the SIGILL handler emulates these Zen2-only ops on Intel
  # hosts). Golden bit-math vectors — header-only, no game dump required.
  add_executable(test_sse4a tests/test_sse4a.cpp)
  add_test(NAME sse4a_insertq_extrq COMMAND test_sse4a)

  # sceAudioOut HLE: format decode, port lifecycle, PCM forwarding, volume, batch, error paths
  # (via a capturing AudioSink — headless, no device).
  add_executable(test_audio tests/test_audio.cpp)
  target_link_libraries(test_audio PRIVATE prosper_core)
  add_test(NAME audio_hle COMMAND test_audio)

  # libSceAjm (#187): headless audio-decode lifecycle — handle out-params filled with valid values
  # (context/instance/batch), inputs validated, batch reports complete. Pure, no game dump.
  add_executable(test_ajm tests/test_ajm.cpp)
  target_link_libraries(test_ajm PRIVATE prosper_core)
  add_test(NAME ajm_hle COMMAND test_ajm)

  add_executable(test_avplayer tests/test_avplayer.cpp)
  target_link_libraries(test_avplayer PRIVATE prosper_core)
  add_test(NAME avplayer_hle COMMAND test_avplayer)

  add_executable(test_equeue_events tests/test_equeue_events.cpp)
  target_link_libraries(test_equeue_events PRIVATE prosper_core)
  add_test(NAME equeue_events COMMAND test_equeue_events)

  # Variadic libc formatters (issue #122): real va_list capture so %f/XMM + stack + %s-past-4th
  # args format correctly (was a 3-integer-register forward). Cross-platform (host libc printf).
  add_executable(test_printf tests/test_printf.cpp)
  target_link_libraries(test_printf PRIVATE prosper_core)
  add_test(NAME printf_varargs COMMAND test_printf)

  # EventFlag/Sema delete-under-waiter safety (issue #104): delete wakes parked waiters with EACCES
  # and defers the free to the last one out, instead of free()ing the object under a live wait (UAF).
  add_executable(test_sync_delete tests/test_sync_delete.cpp)
  target_link_libraries(test_sync_delete PRIVATE prosper_core)
  add_test(NAME sync_delete COMMAND test_sync_delete)

  # Anchored wall-clock sources (#92): CLOCK_REALTIME ≈ host now, one shared epoch across
  # time()/gettimeofday/sceRtc*, FreeBSD clockid mapping, valid LocalTime calendar dates.
  add_executable(test_wall_clock tests/test_wall_clock.cpp)
  target_link_libraries(test_wall_clock PRIVATE prosper_core)
  add_test(NAME wall_clock COMMAND test_wall_clock)

  # UserService/libkernel startup stubs (HLE audit): getters write deterministic defaults; the
  # libkernel registration/hook calls resolve to benign OK no-ops.
  add_executable(test_service_getters tests/test_service_getters.cpp)
  target_link_libraries(test_service_getters PRIVATE prosper_core)
  add_test(NAME service_getters COMMAND test_service_getters)

  # PlatformUi hook (#347): the app frontend can service the guest's ImeDialog with real UI; with no
  # backend registered the headless auto-complete default is unchanged. Pure, no game dump.
  add_executable(test_platform_ui tests/test_platform_ui.cpp)
  target_link_libraries(test_platform_ui PRIVATE prosper_core)
  add_test(NAME platform_ui COMMAND test_platform_ui)

  # SDL3 dialog frontend's guest-struct parsing (#347): pure, SDL-free, so it runs in normal CI even
  # though the SDL_ShowMessageBox glue is display-only. Verifies the MsgDialog/ErrorDialog layouts.
  add_executable(test_dialog_helpers frontends/dialog_sdl3/test_dialog_helpers.cpp
                                     frontends/dialog_sdl3/dialog_helpers.cpp)
  add_test(NAME dialog_helpers COMMAND test_dialog_helpers)

  # libSceImeDialog lifecycle + libSceSaveData memory API round-trip (#191): auto-completing dialog
  # (no hang) + a real per-(user,slot) Set->Get memory block with verified shadPS4 struct layouts.
  add_executable(test_savedata_ime tests/test_savedata_ime.cpp)
  target_link_libraries(test_savedata_ime PRIVATE prosper_core)
  add_test(NAME savedata_ime COMMAND test_savedata_ime)

  # libSceIme keyboard API (#186): reports a consistent "no keyboard connected" state (empty resource
  # array + disconnected info, bounds-checked) so the per-frame input poll never reads garbage.
  add_executable(test_ime_keyboard tests/test_ime_keyboard.cpp)
  target_link_libraries(test_ime_keyboard PRIVATE prosper_core)
  add_test(NAME ime_keyboard COMMAND test_ime_keyboard)

  # pthread scheduling getters fill their out-param (not a shadowing no-op): scePthreadGetprio/
  # Getaffinity must WRITE priority/mask, else the caller reads uninitialized stack (harmful-stub class).
  add_executable(test_pthread_sched_getters tests/test_pthread_sched_getters.cpp)
  target_link_libraries(test_pthread_sched_getters PRIVATE prosper_core)
  add_test(NAME pthread_sched_getters COMMAND test_pthread_sched_getters)

  # libScePad controller input: ScePadData layout + HostPadState mapping + HLE fill contract (pure).
  add_executable(test_pad tests/test_pad.cpp)
  target_link_libraries(test_pad PRIVATE prosper_core)
  add_test(NAME pad_input COMMAND test_pad)

  add_executable(test_sync_on_address tests/test_sync_on_address.cpp)
  target_link_libraries(test_sync_on_address PRIVATE prosper_core)
  add_test(NAME sync_on_address COMMAND test_sync_on_address)

  # sceKernelDlsym honors its module handle (#147): per-module export tables, real handles from
  # LoadStartModule for linked-module paths, global first-wins fallback. Pure, no game dump needed.
  add_executable(test_dlsym_handle tests/test_dlsym_handle.cpp)
  target_link_libraries(test_dlsym_handle PRIVATE prosper_core)
  add_test(NAME dlsym_handle COMMAND test_dlsym_handle)

  # Guest mutex TYPE semantics (#145): settype honored, Sony/FreeBSD ERRORCHECK default (incl.
  # static-sentinel self-init), FreeBSD errno values (EDEADLK=11). Pure, no game dump needed.
  add_executable(test_mutex_types tests/test_mutex_types.cpp)
  target_link_libraries(test_mutex_types PRIVATE prosper_core)
  add_test(NAME mutex_types COMMAND test_mutex_types)

  add_executable(test_agc_shader tests/test_agc_shader.cpp)
  target_link_libraries(test_agc_shader PRIVATE prosper_core)
  add_test(NAME agc_shader_create COMMAND test_agc_shader)

  add_executable(test_tile tests/test_tile.cpp)
  target_link_libraries(test_tile PRIVATE prosper_core)
  add_test(NAME gpu_surface_detile COMMAND test_tile)

  # BC1/BC2/BC3 block decompression to RGBA8 (#121): hand-built blocks with known output. Pure.
  add_executable(test_bc_decode tests/test_bc_decode.cpp)
  target_link_libraries(test_bc_decode PRIVATE prosper_core)
  add_test(NAME bc_decode COMMAND test_bc_decode)

  add_executable(test_stat tests/test_stat.cpp)
  target_link_libraries(test_stat PRIVATE prosper_core)
  add_test(NAME sce_stat_layout COMMAND test_stat)

  # APR container registry + size-keyed read matching (issue #62): stable ids, path dedup, and
  # the ambiguity/chunk-read refusal the read-submit path relies on. Pure, no game dump needed.
  add_executable(test_apr_registry tests/test_apr_registry.cpp)
  target_link_libraries(test_apr_registry PRIVATE prosper_core)
  add_test(NAME apr_registry_size_keying COMMAND test_apr_registry)

  # Direct-memory allocate/available (issue #99): AvailableDirectMemorySize reports the largest
  # free aligned block in the search window and writes both out-params (was a total-size stub).
  add_executable(test_dmem tests/test_dmem.cpp)
  target_link_libraries(test_dmem PRIVATE prosper_core)
  add_test(NAME dmem_available COMMAND test_dmem)

  # Hinted map must not clobber a live mapping (#137): MAP_FIXED_NOREPLACE unless the hint is our
  # own uncommitted reservation. Drives the real map HLE handlers. Linux-only (mmap semantics).
  if(UNIX AND NOT APPLE)
    add_executable(test_map_noreplace tests/test_map_noreplace.cpp)
    target_link_libraries(test_map_noreplace PRIVATE prosper_core)
    add_test(NAME map_noreplace COMMAND test_map_noreplace)

    # A guest mprotect(prot=0) makes the page PROT_NONE, not a default RW (#342): guard pages must
    # actually guard. Drives the real sceKernelMprotect + a SIGSEGV probe. Linux-only.
    add_executable(test_prot_none tests/test_prot_none.cpp)
    target_link_libraries(test_prot_none PRIVATE prosper_core)
    add_test(NAME prot_none COMMAND test_prot_none)
  endif()

  # PM4 command-stream decoder (front of the CommandProcessor): decode a Dcb built by the AGC
  # Dcb functions and assert the ops. Pure/cross-platform, no game dump needed.
  add_executable(test_pm4_decode tests/test_pm4_decode.cpp)
  target_link_libraries(test_pm4_decode PRIVATE prosper_core)
  add_test(NAME pm4_decode_stream COMMAND test_pm4_decode)

  # CommandProcessor apply stage: replay a decoded stream into a GpuState (register files + draws).
  add_executable(test_command_processor tests/test_command_processor.cpp)
  target_link_libraries(test_command_processor PRIVATE prosper_core)
  add_test(NAME command_processor_apply COMMAND test_command_processor)

  # Stage B: CommandProcessor honors RELEASE_MEM/WRITE_DATA memory writes (synchronous end-of-pipe).
  add_executable(test_eop_write tests/test_eop_write.cpp)
  target_link_libraries(test_eop_write PRIVATE prosper_core)
  add_test(NAME eop_write COMMAND test_eop_write)

  # #312 per-queue WAIT_REG_MEM barrier model: unsatisfied waits gate downstream memory effects,
  # streams release independently, jumps don't un-gate, timeout backstop preserves liveness.
  add_executable(test_wait_barrier tests/test_wait_barrier.cpp)
  target_link_libraries(test_wait_barrier PRIVATE prosper_core)
  add_test(NAME wait_barrier COMMAND test_wait_barrier)

  # Render-state extraction: fold a stream into a GpuState, then interpret RDNA2 registers.
  add_executable(test_render_state tests/test_render_state.cpp)
  target_link_libraries(test_render_state PRIVATE prosper_core)
  add_test(NAME render_state_extract COMMAND test_render_state)

  # Messenger recompile guard (#228): committed real Messenger VS/PS RDNA2 binaries must keep
  # recompiling — a recompiler change that re-rejects one of their instructions (the #199/#201
  # blue-screen class) fails here instead of shipping. Pure (no Vulkan); needs no game dump.
  add_executable(test_messenger_recompile_guard tests/test_messenger_recompile_guard.cpp)
  target_link_libraries(test_messenger_recompile_guard PRIVATE prosper_core)
  target_compile_definitions(test_messenger_recompile_guard PRIVATE
      PROSPER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  add_test(NAME messenger_recompile_guard COMMAND test_messenger_recompile_guard)

  # libkernel/libScePosix signal + time-conversion surface (#190): NID match, registration, and real
  # behavior (UTC<->localtime round-trip fills its out-param; pthread_setcancelstate old-state + EINVAL).
  add_executable(test_kernel_signal_time tests/test_kernel_signal_time.cpp)
  target_link_libraries(test_kernel_signal_time PRIVATE prosper_core)
  add_test(NAME kernel_signal_time COMMAND test_kernel_signal_time)

  # Pipeline-state resolution: RenderState (RDNA2 enums) -> Vulkan-ready pipeline state (pure, no Vulkan).
  add_executable(test_pipeline_state tests/test_pipeline_state.cpp)
  target_link_libraries(test_pipeline_state PRIVATE prosper_core)
  add_test(NAME pipeline_state_resolve COMMAND test_pipeline_state)

  # Recompiler coverage metric (pure): per-instruction support report used by shader_histo.
  add_executable(test_recompile_coverage tests/test_recompile_coverage.cpp)
  target_link_libraries(test_recompile_coverage PRIVATE prosper_core)
  add_test(NAME recompile_coverage COMMAND test_recompile_coverage)

  # RDNA2 shader instruction-stream walker (first stage of the RDNA2->SPIR-V recompiler).
  add_executable(test_rdna2_decode tests/test_rdna2_decode.cpp)
  target_link_libraries(test_rdna2_decode PRIVATE prosper_core)
  add_test(NAME rdna2_decode_walk COMMAND test_rdna2_decode)

  # Structural RDNA2->SPIR-V validation that does not require Vulkan runtime support.
  add_executable(test_rdna2_spirv_struct tests/test_rdna2_spirv_struct.cpp)
  target_link_libraries(test_rdna2_spirv_struct PRIVATE prosper_core)
  add_test(NAME rdna2_spirv_struct COMMAND test_rdna2_spirv_struct)

  # Scalar const-fold interpreter behind bindless-dynamic vertex-fetch resolution (no Vulkan needed).
  add_executable(test_dynfetch_fold tests/test_dynfetch_fold.cpp)
  target_link_libraries(test_dynfetch_fold PRIVATE prosper_core)
  add_test(NAME dynfetch_fold COMMAND test_dynfetch_fold)

  # Unannounced-32-bit index-buffer fingerprint (#304 — DOLL's Slate/UMG quads). No Vulkan needed.
  add_executable(test_index_size_detect tests/test_index_size_detect.cpp)
  target_link_libraries(test_index_size_detect PRIVATE prosper_core)
  add_test(NAME index_size_detect COMMAND test_index_size_detect)

  # shader_histo: data-driven recompiler-coverage report over the game's real embedded RDNA2 shaders.
  add_executable(shader_histo tools/shader_histo/shader_histo.cpp)
  target_link_libraries(shader_histo PRIVATE prosper_core)

  # spv_validate: recompile one shader from every path and run spirv-val on each — a strict-validation
  # gate (the render tests only prove llvmpipe accepts them; llvmpipe is lenient). No Vulkan needed;
  # skips validation gracefully if spirv-val is not on PATH (still checks recompilation succeeds).
  add_executable(spv_validate tools/spv_validate/spv_validate.cpp)
  target_link_libraries(spv_validate PRIVATE prosper_core)
  add_test(NAME spv_validate COMMAND spv_validate ${CMAKE_CURRENT_BINARY_DIR})

  # AGC→Vulkan resource layer: the front-half↔back-half contract (gpu_resources.hpp). Registry
  # behavior is fixed here so both halves integrate against a stable API. Pure/cross-platform.
  add_executable(test_gpu_resources tests/test_gpu_resources.cpp)
  target_link_libraries(test_gpu_resources PRIVATE prosper_core)
  add_test(NAME gpu_resource_registry COMMAND test_gpu_resources)

  # Resource-binding contract (shader_resources.hpp): format sizing + the provenance/binding lookups
  # the recompiler and pipeline share. Pure/cross-platform. See docs/RESOURCE_BINDING.md.
  add_executable(test_shader_resources tests/test_shader_resources.cpp)
  target_link_libraries(test_shader_resources PRIVATE prosper_core)
  add_test(NAME shader_resource_contract COMMAND test_shader_resources)

  # Front-half resource-table builder + V# descriptor decode (agc_shader_layout.cpp): shader user_data
  # + user-data SGPRs -> ShaderResourceTable (constant buffers, stage 1). Pure/cross-platform.
  add_executable(test_build_shader_resources tests/test_build_shader_resources.cpp)
  target_link_libraries(test_build_shader_resources PRIVATE prosper_core)
  add_test(NAME build_shader_resources COMMAND test_build_shader_resources)

  # Descriptor-binding policy (#157): textures/storage images never land on the recompiler's hardwired
  # cbuf bindings 2/3; constant/vertex buffers assigned first. Pure, no game dump / Vulkan needed.
  add_executable(test_convention_bindings tests/test_convention_bindings.cpp)
  target_link_libraries(test_convention_bindings PRIVATE prosper_core)
  add_test(NAME convention_bindings COMMAND test_convention_bindings)

  # M2+: real memory mapping, trap, boot, and the boot-trace debug tool (Linux only).
  if(UNIX AND NOT APPLE)
    add_executable(boot_trace tools/boot_trace/boot_trace.cpp)
    target_link_libraries(boot_trace PRIVATE prosper_core)

    # Zero-dependency Linux evdev gamepad frontend (outside prosper_core). Always available on Linux;
    # boot_trace installs it (gated by PROSPER_PAD) when the SDL3 gamepad frontend isn't built.
    add_library(prosper_pad_evdev STATIC frontends/pad_evdev/pad_evdev.cpp)
    target_include_directories(prosper_pad_evdev PUBLIC frontends/pad_evdev src)
    target_link_libraries(prosper_pad_evdev PUBLIC prosper_core)
    target_link_libraries(boot_trace PRIVATE prosper_pad_evdev)
    target_compile_definitions(boot_trace PRIVATE PROSPER_PAD_EVDEV=1)

    add_executable(imgdump tools/imgdump/imgdump.cpp)
    target_link_libraries(imgdump PRIVATE prosper_core)

    add_executable(test_trap_linux tests/test_trap_linux.cpp)
    target_link_libraries(test_trap_linux PRIVATE prosper_core)
    if(HAVE_GAME_DUMP)
      add_test(NAME trap_identifies_imports
               COMMAND test_trap_linux "${GAME_DUMP}")
    endif()

    # The init-fault report must survive a wild/null faulting rip (#128). No game dump needed.
    add_executable(test_initfault_dump tests/test_initfault_dump.cpp)
    target_link_libraries(test_initfault_dump PRIVATE prosper_core)
    add_test(NAME initfault_dump_survives COMMAND test_initfault_dump)

    add_executable(test_boot_linux tests/test_boot_linux.cpp)
    target_link_libraries(test_boot_linux PRIVATE prosper_core)
    if(HAVE_GAME_DUMP)
      add_test(NAME boot_reaches_first_syscall
               COMMAND test_boot_linux "${GAME_DUMP}")
    endif()

    # Regression test for the real setjmp/longjmp impl (x86-64 asm; used by the GC's root scan).
    add_executable(test_setjmp tests/test_setjmp.cpp)
    target_link_libraries(test_setjmp PRIVATE prosper_core)
    add_test(NAME setjmp_saves_and_restores COMMAND test_setjmp)

    add_executable(test_stack_registry tests/test_stack_registry.cpp)
    target_link_libraries(test_stack_registry PRIVATE prosper_core)
    add_test(NAME thread_stack_registry COMMAND test_stack_registry)

    add_executable(test_rwlock_once tests/test_rwlock_once.cpp)
    target_link_libraries(test_rwlock_once PRIVATE prosper_core)
    add_test(NAME rwlock_once_semantics COMMAND test_rwlock_once)

    # __tls_get_addr per-thread DTV is purged (blocks freed) on both thread-exit paths (#68):
    # a recycled pthread id must get fresh zero/tdata-initialized TLS, never a dead thread's bytes.
    add_executable(test_tls_dtv_purge tests/test_tls_dtv_purge.cpp)
    target_link_libraries(test_tls_dtv_purge PRIVATE prosper_core)
    add_test(NAME tls_dtv_purge_on_thread_exit COMMAND test_tls_dtv_purge)

    add_executable(test_agc_dcb tests/test_agc_dcb.cpp)
    target_link_libraries(test_agc_dcb PRIVATE prosper_core)
    add_test(NAME agc_dcb_pm4 COMMAND test_agc_dcb)

    # End-to-end AGC submit: build a Dcb via the HLE fns -> sceAgcDriverSubmitDcb -> CommandProcessor
    # folds it into a GpuState. Locks in the "first real command buffer executes" milestone.
    add_executable(test_agc_submit tests/test_agc_submit.cpp)
    target_link_libraries(test_agc_submit PRIVATE prosper_core)
    add_test(NAME agc_submit_to_gpustate COMMAND test_agc_submit)

    # libSceVideoOut: query/config functions return real 1080p60 values + swapchain-scaffolding
    # display-buffer registry records the game's registered framebuffers.
    add_executable(test_videoout tests/test_videoout.cpp)
    target_link_libraries(test_videoout PRIVATE prosper_core)
    add_test(NAME videoout_queries COMMAND test_videoout)

    # libSceVideoOut flip->present: SubmitFlip scans out the registered buffer; present_readback
    # returns the presented frame's real pixels. The surface the renderer presents frames to.
    add_executable(test_present tests/test_present.cpp)
    target_link_libraries(test_present PRIVATE prosper_core)
    add_test(NAME videoout_present COMMAND test_present)

    # Graphics-phase foundation + agentic-first verification harness: headless offscreen Vulkan
    # (render/clear -> readback -> pixel/CRC assert). Only built if Vulkan dev files are present.
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
      # Live game rendering: with Vulkan available, boot_trace can register the live submit renderer
      # (PROSPER_RENDER=1) so the game's SubmitDcb draws execute + present + dump frames to disk.
      target_link_libraries(boot_trace PRIVATE Vulkan::Vulkan)
      target_compile_definitions(boot_trace PRIVATE PROSPER_HAVE_VULKAN=1)

      # Shared live renderer (frontends/shared): the DrawItem->Vulkan compositor, factored out of
      # boot_trace so BOTH boot_trace and prosper-app register the SAME renderer instead of
      # duplicating it. Needs render_runner.h (tests/) and refvs.inc (tools/boot_trace/).
      add_library(prosper_live_renderer STATIC frontends/shared/live_renderer.cpp)
      target_include_directories(prosper_live_renderer PRIVATE src tests tools/boot_trace)
      target_link_libraries(prosper_live_renderer PUBLIC prosper_core Vulkan::Vulkan)
      target_include_directories(prosper_live_renderer PUBLIC frontends/shared)
      target_link_libraries(boot_trace PRIVATE prosper_live_renderer)

      # screenshot: run a game and capture a PNG every N frames (default 60) until M shots (default
      # 30). Reuses boot_program + the shared live renderer; writes zlib-compressed PNGs (needs zlib).
      find_package(ZLIB QUIET)
      if(ZLIB_FOUND)
        add_executable(screenshot tools/screenshot/screenshot.cpp)
        target_include_directories(screenshot PRIVATE src)
        target_link_libraries(screenshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan ZLIB::ZLIB)
      else()
        message(STATUS "zlib not found — skipping the screenshot tool")
      endif()

      add_executable(test_vulkan_offscreen tests/test_vulkan_offscreen.cpp)
      target_link_libraries(test_vulkan_offscreen PRIVATE Vulkan::Vulkan)
      add_test(NAME vulkan_offscreen_clear COMMAND test_vulkan_offscreen)

      # First real rendered output: a VkPipeline whose topology/format come from a RenderState via
      # vk_translate, drawing a triangle (embedded placeholder SPIR-V), verified by pixel readback.
      add_executable(test_vulkan_triangle tests/test_vulkan_triangle.cpp)
      target_link_libraries(test_vulkan_triangle PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME vulkan_triangle_render COMMAND test_vulkan_triangle)

      # Execution-differential harness (verification layer 4): run a compute shader with known
      # inputs, assert numeric outputs. The RDNA2->SPIR-V recompiler's output will plug in here.
      add_executable(test_compute_exec tests/test_compute_exec.cpp)
      target_link_libraries(test_compute_exec PRIVATE Vulkan::Vulkan)
      add_test(NAME compute_exec_differential COMMAND test_compute_exec)

      # Prove our own SPIR-V emitter produces valid, correct, runnable SPIR-V (execution-differential).
      add_executable(test_spirv_builder tests/test_spirv_builder.cpp)
      target_link_libraries(test_spirv_builder PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME spirv_builder_exec COMMAND test_spirv_builder)

      # End-to-end: recompile real RDNA2 (llvm-mc) -> SPIR-V -> run -> assert numeric result.
      add_executable(test_rdna2_to_spirv tests/test_rdna2_to_spirv.cpp)
      target_link_libraries(test_rdna2_to_spirv PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME rdna2_to_spirv_exec COMMAND test_rdna2_to_spirv)

      # Render a triangle whose PIXEL shader is recompiled from RDNA2 (EXP MRT0 -> color output).
      add_executable(test_recompiled_fragment tests/test_recompiled_fragment.cpp)
      target_link_libraries(test_recompiled_fragment PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME recompiled_fragment_render COMMAND test_recompiled_fragment)

      # Render a frame where BOTH vertex + pixel shaders are recompiled from RDNA2.
      add_executable(test_recompiled_shaders tests/test_recompiled_shaders.cpp)
      target_link_libraries(test_recompiled_shaders PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME recompiled_shaders_render COMMAND test_recompiled_shaders)

      # Vertex fetch: a VS that pulls positions from a bound vertex buffer via buffer_load_format,
      # keyed by a ShaderResourceTable — real game vertex shaders work this way.
      add_executable(test_vertex_fetch_render tests/test_vertex_fetch_render.cpp)
      target_link_libraries(test_vertex_fetch_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME vertex_fetch_render COMMAND test_vertex_fetch_render)

      # Texture sampling: a recompiled pixel shader does MIMG image_sample against a bound 2x2 texture.
      add_executable(test_texture_sample_render tests/test_texture_sample_render.cpp)
      target_link_libraries(test_texture_sample_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME texture_sample_render COMMAND test_texture_sample_render)

      # Multi-draw backend: N draws composited into ONE cleared-once framebuffer (red opaque + green
      # additive -> yellow center proves accumulation into a single target).
      add_executable(test_multidraw_render tests/test_multidraw_render.cpp)
      target_link_libraries(test_multidraw_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME multidraw_render COMMAND test_multidraw_render)

      # Indexed draws (#64): vkCmdDrawIndexed from a BackendDraw's index data — the [0,1,2,2,3,0] quad
      # must match the retired perimeter-fan rendering pixel-exact, and indices must really be applied.
      add_executable(test_indexed_render tests/test_indexed_render.cpp)
      target_link_libraries(test_indexed_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME indexed_render COMMAND test_indexed_render)

      # Storage images: a recompiled compute kernel does MIMG image_load->image_store (no sampler) to
      # copy a 1D storage image bit-exact (the game's blit/copy compute-shader class).
      add_executable(test_storage_image_copy tests/test_storage_image_copy.cpp)
      target_link_libraries(test_storage_image_copy PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME storage_image_copy COMMAND test_storage_image_copy)

      # Interpolation: a recompiled VS exports EXP PARAM0, a recompiled PS reads it via v_interp.
      add_executable(test_interp_render tests/test_interp_render.cpp)
      target_link_libraries(test_interp_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME interp_render COMMAND test_interp_render)

      # Full textured-draw path: VS exports interpolated UVs, PS samples a texture at them (+screenshot).
      add_executable(test_textured_interp_render tests/test_textured_interp_render.cpp)
      target_link_libraries(test_textured_interp_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME textured_interp_render COMMAND test_textured_interp_render)

      # Pipeline realization: build a VkPipeline whose fixed-function state comes from
      # resolve_pipeline_state(RenderState) and prove the resolved state drives rendering.
      add_executable(test_pipeline_render tests/test_pipeline_render.cpp)
      target_link_libraries(test_pipeline_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME pipeline_render COMMAND test_pipeline_render)

      # Full back-half spine: a GpuState -> recompile shaders from their PGM addresses -> resolve
      # pipeline state -> rendered frame (the path a real submitted Dcb will drive).
      add_executable(test_gpustate_render tests/test_gpustate_render.cpp)
      target_link_libraries(test_gpustate_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME gpustate_render COMMAND test_gpustate_render)

      # GPU-executor core (Stage A): GpuState -> execute_gpustate() -> render -> present round-trip.
      add_executable(test_gpu_execute tests/test_gpu_execute.cpp)
      target_link_libraries(test_gpu_execute PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME gpu_execute COMMAND test_gpu_execute)

      # Payoff demo: recompile the GAME'S OWN embedded shaders and render them on real Vulkan.
      # Needs both Vulkan and the (gitignored) game dump, so it's gated on HAVE_GAME_DUMP.
      if(HAVE_GAME_DUMP)
        add_executable(test_real_shader_render tests/test_real_shader_render.cpp)
        target_link_libraries(test_real_shader_render PRIVATE prosper_core Vulkan::Vulkan)
        add_test(NAME real_shader_render COMMAND test_real_shader_render "${GAME_DUMP}/eboot.bin")
      endif()
      message(STATUS "Vulkan found (${Vulkan_LIBRARY}) — offscreen graphics harness enabled")
    else()
      message(STATUS "Vulkan not found — skipping offscreen graphics harness")
    endif()
  endif()
endif()

# --- SDL3 frontends (optional; headless core stays dependency-free) --------------------------
# The sceAudioOut sink (frontends/audio_sdl3) and the libScePad gamepad backend (frontends/pad_sdl3)
# can both use SDL3. They must share ONE SDL3 build: if each fetched independently, the first fetch's
# option set would win — e.g. an AUDIO+PAD build would get audio's joystick-less SDL and the gamepad
# backend would silently have no device. So acquire SDL3 ONCE here, with the UNION of the subsystems
# the enabled frontends need (JOYSTICK/HIDAPI only when the pad frontend is on). Both consumer blocks
# below just use the resulting SDL3::SDL3 target.
option(PROSPER_AUDIO_SDL3 "Build the optional SDL3 audio frontend"   OFF)
option(PROSPER_PAD_SDL3   "Build the optional SDL3 gamepad frontend" OFF)
option(PROSPER_APP        "Build the prosper-app frontend (SDL3 window + Vulkan present)" OFF)
if((PROSPER_AUDIO_SDL3 OR PROSPER_PAD_SDL3 OR PROSPER_APP) AND TARGET prosper_core AND NOT TARGET SDL3::SDL3)
  find_package(SDL3 QUIET CONFIG)
  if(NOT SDL3_FOUND)
    message(STATUS "SDL3 not found — fetching release-3.2.30 (audio=${PROSPER_AUDIO_SDL3} pad=${PROSPER_PAD_SDL3} app=${PROSPER_APP})")
    include(FetchContent)
    set(SDL_TEST OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    # The window frontend (PROSPER_APP) needs a windowing system (Wayland/X11) for the Vulkan
    # surface; audio/pad-only builds do not — use SDL's console (no-video) escape hatch there. This
    # is the same "union of subsystems" logic as the joystick block below: enable VIDEO iff any
    # enabled frontend needs a window.
    if(PROSPER_APP)
      set(SDL_UNIX_CONSOLE_BUILD OFF CACHE BOOL "" FORCE)
      set(SDL_VIDEO   ON CACHE BOOL "" FORCE)
      set(SDL_WAYLAND ON CACHE BOOL "" FORCE)
      set(SDL_X11     ON CACHE BOOL "" FORCE)
    else()
      set(SDL_UNIX_CONSOLE_BUILD ON CACHE BOOL "" FORCE)
      set(SDL_VIDEO   OFF CACHE BOOL "" FORCE)
      set(SDL_WAYLAND OFF CACHE BOOL "" FORCE)
      set(SDL_X11     OFF CACHE BOOL "" FORCE)
    endif()
    set(SDL_RENDER OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
    set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
    set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
    set(SDL_POWER OFF CACHE BOOL "" FORCE)
    set(SDL_DIALOG OFF CACHE BOOL "" FORCE)
    # Gamepad support is needed ONLY by the pad frontend — enable JOYSTICK + HIDAPI whenever it's on,
    # regardless of which frontend triggers the (single) fetch. This is the union that fixes the
    # combined-build trap: an AUDIO+PAD build now gets a joystick-capable SDL.
    if(PROSPER_PAD_SDL3)
      set(SDL_JOYSTICK ON CACHE BOOL "" FORCE)
      set(SDL_HIDAPI   ON CACHE BOOL "" FORCE)
    else()
      set(SDL_JOYSTICK OFF CACHE BOOL "" FORCE)
      set(SDL_HIDAPI   OFF CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(SDL3
      GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
      GIT_TAG release-3.2.30
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(SDL3)
  endif()
endif()

# --- prosper-app: SDL3 window + Vulkan swapchain present frontend (issue #164 P0) -----------
# Presents the frame prosper's renderer hands to the present layer (present_readback) in a real
# window. A SEPARATE Vulkan context from the core's headless render device (docs/FRONTEND_APP.md);
# frames cross as CPU pixels. Entirely outside prosper_core. Default OFF; needs SDL3 (video) + Vulkan.
if(PROSPER_APP AND TARGET SDL3::SDL3)
  find_package(Vulkan QUIET)
  if(Vulkan_FOUND)
    add_executable(prosper-app frontends/prosper-app/main.cpp)
    target_include_directories(prosper-app PRIVATE src)
    target_link_libraries(prosper-app PRIVATE prosper_core SDL3::SDL3 Vulkan::Vulkan)
    # Share the live renderer with boot_trace so the window shows composited game frames (not blank).
    if(TARGET prosper_live_renderer)
      target_link_libraries(prosper-app PRIVATE prosper_live_renderer)
      target_compile_definitions(prosper-app PRIVATE PROSPER_HAVE_LIVE_RENDERER=1)
    endif()
    # SDL3 dialog frontend (#347): real SDL_ShowMessageBox for the guest's MsgDialog/ErrorDialog via
    # the PlatformUi hook. The layout parsing (dialog_helpers) is pure + unit-tested (see below); this
    # module adds the SDL glue and needs SDL3 video (already required by prosper-app).
    add_library(prosper_dialog_sdl3 STATIC
                frontends/dialog_sdl3/platform_ui_sdl3.cpp frontends/dialog_sdl3/dialog_helpers.cpp)
    target_include_directories(prosper_dialog_sdl3 PUBLIC frontends/dialog_sdl3 src)
    target_link_libraries(prosper_dialog_sdl3 PUBLIC prosper_core SDL3::SDL3)
    target_link_libraries(prosper-app PRIVATE prosper_dialog_sdl3)
    target_compile_definitions(prosper-app PRIVATE PROSPER_HAVE_DIALOG_SDL3=1)
    message(STATUS "prosper-app frontend enabled (SDL3 window + Vulkan present)")
  else()
    message(WARNING "PROSPER_APP=ON but Vulkan not found — skipping prosper-app")
  endif()
endif()

# --- optional SDL3 audio frontend (headless core stays dependency-free) ---------------------
# The sceAudioOut HLE is backend-agnostic (src/hle/audio.hpp). This builds an SDL3-backed sink,
# entirely OUTSIDE prosper_core, so the guest's audio plays on the host.
if(PROSPER_AUDIO_SDL3 AND TARGET SDL3::SDL3)
  add_library(prosper_audio_sdl3 STATIC frontends/audio_sdl3/audio_sink_sdl3.cpp)
  target_include_directories(prosper_audio_sdl3 PUBLIC frontends/audio_sdl3 src)
  target_link_libraries(prosper_audio_sdl3 PUBLIC prosper_core SDL3::SDL3)
  target_compile_definitions(prosper_audio_sdl3 PUBLIC PROSPER_AUDIO_SDL3=1)

  # Headless self-test of the full SDL3 path (run under the SDL "dummy" audio driver in CI):
  # install the sink, drive the sceAudioOut HLE, assert grains flow through SDL with no device.
  add_executable(test_audio_sdl3 frontends/audio_sdl3/test_audio_sdl3.cpp)
  target_link_libraries(test_audio_sdl3 PRIVATE prosper_audio_sdl3)
  add_test(NAME audio_sdl3 COMMAND test_audio_sdl3)
  set_tests_properties(audio_sdl3 PROPERTIES ENVIRONMENT "SDL_AUDIO_DRIVER=dummy")

  # Route the live game's audio to the host when the runner is built (guarded by the macro).
  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_audio_sdl3)
    target_compile_definitions(boot_trace PRIVATE PROSPER_AUDIO_SDL3=1)
  endif()

  message(STATUS "SDL3 audio frontend enabled (prosper_audio_sdl3)")
endif()

# --- optional SDL3 gamepad frontend (headless core stays dependency-free) --------------------
# The libScePad HLE is backend-agnostic (src/input/pad.hpp). This builds an SDL_Gamepad-backed
# PadBackend, entirely OUTSIDE prosper_core, so a real controller drives the guest cross-platform.
# Off by default.  Enable with:  -DPROSPER_PAD_SDL3=ON  (SDL3 acquired by the shared step above,
# with JOYSTICK/HIDAPI ON because this option is set).
if(PROSPER_PAD_SDL3 AND TARGET SDL3::SDL3)
  add_library(prosper_pad_sdl3 STATIC frontends/pad_sdl3/pad_sink_sdl3.cpp)
  target_include_directories(prosper_pad_sdl3 PUBLIC frontends/pad_sdl3 src)
  target_link_libraries(prosper_pad_sdl3 PUBLIC prosper_core SDL3::SDL3)
  target_compile_definitions(prosper_pad_sdl3 PUBLIC PROSPER_PAD_SDL3=1)

  # Headless self-test of the full SDL3 gamepad path (no controller attached in CI): install the
  # backend, poll -> disconnected/neutral, drive the pure mapping. Uses the SDL dummy video driver.
  add_executable(test_pad_sdl3 frontends/pad_sdl3/test_pad_sdl3.cpp)
  target_link_libraries(test_pad_sdl3 PRIVATE prosper_pad_sdl3)
  add_test(NAME pad_sdl3 COMMAND test_pad_sdl3)
  set_tests_properties(pad_sdl3 PROPERTIES ENVIRONMENT "SDL_VIDEO_DRIVER=dummy;SDL_AUDIO_DRIVER=dummy")

  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_pad_sdl3)
    target_compile_definitions(boot_trace PRIVATE PROSPER_PAD_SDL3=1)
  endif()

  message(STATUS "SDL3 gamepad frontend enabled (prosper_pad_sdl3)")
endif()

# --- wire the SDL3 audio/pad frontends into prosper-app (after their targets exist) ---------
# prosper-app is defined above (before these libs), so link them here once they're built. Enable
# both alongside the app for a full experience:  -DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON
if(TARGET prosper-app)
  if(TARGET prosper_audio_sdl3)
    target_link_libraries(prosper-app PRIVATE prosper_audio_sdl3)
    target_include_directories(prosper-app PRIVATE frontends/audio_sdl3)
    target_compile_definitions(prosper-app PRIVATE PROSPER_AUDIO_SDL3=1)
    message(STATUS "prosper-app: audio out enabled")
  endif()
  if(TARGET prosper_pad_sdl3)
    target_link_libraries(prosper-app PRIVATE prosper_pad_sdl3)
    target_include_directories(prosper-app PRIVATE frontends/pad_sdl3)
    target_compile_definitions(prosper-app PRIVATE PROSPER_PAD_SDL3=1)
    message(STATUS "prosper-app: controller input enabled")
  endif()
endif()
