cmake_minimum_required(VERSION 3.20)
project(prosper_diagnostic_controls LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
get_filename_component(PROSPER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(PROSPER_SANITIZER "address-undefined" CACHE STRING "address-undefined, thread, or none")
set_property(CACHE PROSPER_SANITIZER PROPERTY STRINGS address-undefined thread none)
add_library(host_checks INTERFACE)
target_include_directories(host_checks INTERFACE "${PROSPER_ROOT}/src" "${PROSPER_ROOT}/frontends")
if(NOT PROSPER_SANITIZER STREQUAL "none")
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "This sanitizer workflow is validated for Linux GCC/Clang only")
  endif()
  if(PROSPER_SANITIZER STREQUAL "address-undefined")
    set(_san "address,undefined")
  elseif(PROSPER_SANITIZER STREQUAL "thread")
    set(_san "thread")
  else()
    message(FATAL_ERROR "Unknown PROSPER_SANITIZER: ${PROSPER_SANITIZER}")
  endif()
  target_compile_options(host_checks INTERFACE -fsanitize=${_san} -fno-sanitize-recover=all
                         -fno-omit-frame-pointer -g)
  target_link_options(host_checks INTERFACE -fsanitize=${_san})
endif()
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
enable_testing()
# GoogleTest for the migrated suites below, from the same vendored tree the main build uses
# (cmake/ProsperGTest.cmake). This narrow build cannot reuse prosper targets, but the TEST
# sources need gtest_main like everywhere else.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory("${PROSPER_ROOT}/third_party/googletest"
                 "${CMAKE_BINARY_DIR}/_deps/googletest-build" EXCLUDE_FROM_ALL)
function(host_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE host_checks Threads::Threads)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT 30)
endfunction()
# Existing production sources and existing regression arms, without guest execution or Vulkan.
# These pairs are listed BY HAND on purpose: this build is deliberately narrow, so it cannot
# reuse the main tree's `file(GLOB_RECURSE ... src/*.cpp)`. The cost lands on someone else --
# splitting any source below into a second .cpp reddens the `Host ASan + UBSan` CI job in an
# unrelated PR. If that is you: add the new source to the same host_test() line. Nothing here
# is title- or dump-specific, so a link error is a stale list and never a missing fixture.
host_test(self_segment_key "${PROSPER_ROOT}/tests/test_self_segment_key.cpp"
                          "${PROSPER_ROOT}/src/self/module.cpp")
target_link_libraries(self_segment_key PRIVATE GTest::gtest GTest::gtest_main)
# Run the production SELF/ELF parser and image builder through their existing malformed-input
# bounds regression suite, and exercise VA-to-file-offset resolution under the same sanitizers.
# These synthetic tests need no game dump or Vulkan; keeping them here puts ELF metadata reads and
# loader-facing image construction under ASan + UBSan in CI (#4338).
host_test(self_bounds "${PROSPER_ROOT}/tests/test_self_bounds.cpp"
                      "${PROSPER_ROOT}/src/self/module.cpp")
target_link_libraries(self_bounds PRIVATE GTest::gtest GTest::gtest_main)
host_test(va2foff "${PROSPER_ROOT}/tests/test_va2foff.cpp"
                  "${PROSPER_ROOT}/src/self/module.cpp")
target_link_libraries(va2foff PRIVATE GTest::gtest GTest::gtest_main)
# Run the synthetic link and rejection contracts through the production linker, SELF/ELF parser,
# relocation applier, and NID hash implementation under ASan + UBSan. The fixtures are generated
# from synthetic ELF bytes and exercise import/export resolution and malformed relocation targets;
# no game dump or Vulkan is required. Keep the two contracts separate so a healthy link control and
# malformed-input refusal each remain visible in CTest.
host_test(loader_synth_link "${PROSPER_ROOT}/tests/misc/test_loader_synth_link.cpp"
                            "${PROSPER_ROOT}/src/self/module.cpp"
                            "${PROSPER_ROOT}/src/loader/linker.cpp"
                            "${PROSPER_ROOT}/src/hle/dispatch/nid.cpp")
target_include_directories(loader_synth_link PRIVATE "${PROSPER_ROOT}/tests")
target_link_libraries(loader_synth_link PRIVATE GTest::gtest GTest::gtest_main)
host_test(loader_synth_reject "${PROSPER_ROOT}/tests/misc/test_loader_synth_reject.cpp"
                              "${PROSPER_ROOT}/src/self/module.cpp"
                              "${PROSPER_ROOT}/src/loader/linker.cpp"
                              "${PROSPER_ROOT}/src/hle/dispatch/nid.cpp")
target_include_directories(loader_synth_reject PRIVATE "${PROSPER_ROOT}/tests")
target_link_libraries(loader_synth_reject PRIVATE GTest::gtest GTest::gtest_main)
host_test(rdna2_decode "${PROSPER_ROOT}/tests/gpu/recompiler/test_rdna2_decode.cpp"
                      "${PROSPER_ROOT}/src/gpu/recompiler/rdna2_decode.cpp")
target_link_libraries(rdna2_decode PRIVATE GTest::gtest GTest::gtest_main)
host_test(bc_decode "${PROSPER_ROOT}/tests/gpu/texture/test_bc_decode.cpp"
                   "${PROSPER_ROOT}/src/gpu/texture/bc_decode.cpp")
host_test(gpu_submit_gate "${PROSPER_ROOT}/tests/host/platform/test_gpu_submit_gate.cpp"
                         "${PROSPER_ROOT}/src/host/platform/gpu_submit_gate.cpp")
add_test(NAME diagnostic_doctor COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_doctor.py")
if(NOT PROSPER_SANITIZER STREQUAL "none")
  add_executable(sanitizer_control sanitizer_control.cpp)
  target_link_libraries(sanitizer_control PRIVATE host_checks Threads::Threads)
  add_test(NAME sanitizer_instrument_control COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/check_sanitizer.py" "$<TARGET_FILE:sanitizer_control>"
    "${PROSPER_SANITIZER}")
  set_tests_properties(sanitizer_instrument_control PROPERTIES TIMEOUT 30)
endif()
option(PROSPER_RENDERDOC_CONTROL "Build the opt-in RenderDoc capture control" OFF)
if(PROSPER_RENDERDOC_CONTROL)
  find_package(Vulkan REQUIRED)
  find_path(RENDERDOC_INCLUDE_DIR renderdoc_app.h REQUIRED)
  add_executable(renderdoc_control "${PROSPER_ROOT}/tools/index_fetch_probe/probe.c")
  target_compile_definitions(renderdoc_control PRIVATE PROSPER_RENDERDOC_CONTROL)
  target_include_directories(renderdoc_control PRIVATE "${RENDERDOC_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(renderdoc_control PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})
  # Same bracket, different subject: this one produces the capture that validates
  # tools/pixel_history/pixel_history.py against a construction with a known answer.
  add_executable(pixel_history_capture "${PROSPER_ROOT}/tools/pixel_history/control.c")
  target_compile_definitions(pixel_history_capture PRIVATE PROSPER_PIXHIST_CAPTURE)
  target_include_directories(pixel_history_capture PRIVATE "${RENDERDOC_INCLUDE_DIR}")
  target_link_libraries(pixel_history_capture PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})
endif()
