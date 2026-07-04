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

  add_executable(test_sync_on_address tests/test_sync_on_address.cpp)
  target_link_libraries(test_sync_on_address PRIVATE prosper_core)
  add_test(NAME sync_on_address COMMAND test_sync_on_address)

  add_executable(test_agc_shader tests/test_agc_shader.cpp)
  target_link_libraries(test_agc_shader PRIVATE prosper_core)
  add_test(NAME agc_shader_create COMMAND test_agc_shader)

  add_executable(test_stat tests/test_stat.cpp)
  target_link_libraries(test_stat PRIVATE prosper_core)
  add_test(NAME sce_stat_layout COMMAND test_stat)

  # PM4 command-stream decoder (front of the CommandProcessor): decode a Dcb built by the AGC
  # Dcb functions and assert the ops. Pure/cross-platform, no game dump needed.
  add_executable(test_pm4_decode tests/test_pm4_decode.cpp)
  target_link_libraries(test_pm4_decode PRIVATE prosper_core)
  add_test(NAME pm4_decode_stream COMMAND test_pm4_decode)

  # CommandProcessor apply stage: replay a decoded stream into a GpuState (register files + draws).
  add_executable(test_command_processor tests/test_command_processor.cpp)
  target_link_libraries(test_command_processor PRIVATE prosper_core)
  add_test(NAME command_processor_apply COMMAND test_command_processor)

  # Render-state extraction: fold a stream into a GpuState, then interpret RDNA2 registers.
  add_executable(test_render_state tests/test_render_state.cpp)
  target_link_libraries(test_render_state PRIVATE prosper_core)
  add_test(NAME render_state_extract COMMAND test_render_state)

  # Pipeline-state resolution: RenderState (RDNA2 enums) -> Vulkan-ready pipeline state (pure, no Vulkan).
  add_executable(test_pipeline_state tests/test_pipeline_state.cpp)
  target_link_libraries(test_pipeline_state PRIVATE prosper_core)
  add_test(NAME pipeline_state_resolve COMMAND test_pipeline_state)

  # RDNA2 shader instruction-stream walker (first stage of the RDNA2->SPIR-V recompiler).
  add_executable(test_rdna2_decode tests/test_rdna2_decode.cpp)
  target_link_libraries(test_rdna2_decode PRIVATE prosper_core)
  add_test(NAME rdna2_decode_walk COMMAND test_rdna2_decode)

  # Structural RDNA2->SPIR-V validation that does not require Vulkan runtime support.
  add_executable(test_rdna2_spirv_struct tests/test_rdna2_spirv_struct.cpp)
  target_link_libraries(test_rdna2_spirv_struct PRIVATE prosper_core)
  add_test(NAME rdna2_spirv_struct COMMAND test_rdna2_spirv_struct)

  # shader_histo: data-driven recompiler-coverage report over the game's real embedded RDNA2 shaders.
  add_executable(shader_histo tools/shader_histo/shader_histo.cpp)
  target_link_libraries(shader_histo PRIVATE prosper_core)

  # AGC→Vulkan resource layer: the front-half↔back-half contract (gpu_resources.hpp). Registry
  # behavior is fixed here so both halves integrate against a stable API. Pure/cross-platform.
  add_executable(test_gpu_resources tests/test_gpu_resources.cpp)
  target_link_libraries(test_gpu_resources PRIVATE prosper_core)
  add_test(NAME gpu_resource_registry COMMAND test_gpu_resources)

  # M2+: real memory mapping, trap, boot, and the boot-trace debug tool (Linux only).
  if(UNIX AND NOT APPLE)
    add_executable(boot_trace tools/boot_trace/boot_trace.cpp)
    target_link_libraries(boot_trace PRIVATE prosper_core)

    add_executable(imgdump tools/imgdump/imgdump.cpp)
    target_link_libraries(imgdump PRIVATE prosper_core)

    add_executable(test_trap_linux tests/test_trap_linux.cpp)
    target_link_libraries(test_trap_linux PRIVATE prosper_core)
    if(HAVE_GAME_DUMP)
      add_test(NAME trap_identifies_imports
               COMMAND test_trap_linux "${GAME_DUMP}")
    endif()

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

    # Graphics-phase foundation + agentic-first verification harness: headless offscreen Vulkan
    # (render/clear -> readback -> pixel/CRC assert). Only built if Vulkan dev files are present.
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
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

      # Pipeline realization: build a VkPipeline whose fixed-function state comes from
      # resolve_pipeline_state(RenderState) and prove the resolved state drives rendering.
      add_executable(test_pipeline_render tests/test_pipeline_render.cpp)
      target_link_libraries(test_pipeline_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME pipeline_render COMMAND test_pipeline_render)
      message(STATUS "Vulkan found (${Vulkan_LIBRARY}) — offscreen graphics harness enabled")
    else()
      message(STATUS "Vulkan not found — skipping offscreen graphics harness")
    endif()
  endif()
endif()
