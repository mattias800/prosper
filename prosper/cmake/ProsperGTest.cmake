# GoogleTest (GTest + GMock), vendored verbatim under third_party/googletest (v1.18.0; see its README).
# Usage: include(cmake/ProsperGTest.cmake) once after enable_testing(), then
#   prosper_add_gtest(<target> SOURCES ... [INCLUDES ...] [LIBRARIES ...] [LABELS ...] [PROPERTIES <name> <value> ...])
# PROPERTIES are applied to every discovered test (e.g. TIMEOUT 60). ENVIRONMENT takes NAME=value
# words (ENVIRONMENT A=1 B=2), set for every run via `cmake -E env`.
# Each TEST() becomes its own ctest case (<Suite>.<Name>).
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest"
                 "${CMAKE_BINARY_DIR}/_deps/googletest-build" EXCLUDE_FROM_ALL)
include(GoogleTest)

function(prosper_add_gtest target)
  cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES;LIBRARIES;LABELS;PROPERTIES;ENVIRONMENT" ${ARGN})
  add_executable(${target} ${ARG_SOURCES})
  if(ARG_INCLUDES)
    target_include_directories(${target} PRIVATE ${ARG_INCLUDES})
  endif()
  target_link_libraries(${target} PRIVATE ${ARG_LIBRARIES} GTest::gtest GTest::gmock GTest::gtest_main)
  if(NOT ARG_LABELS)
    set(ARG_LABELS unit)
  endif()
  if(ARG_ENVIRONMENT)
    # A multi-variable ENVIRONMENT test property cannot be expressed through gtest_discover_tests
    # (it emits each ';'-separated item as a separate argument), so launch the binary through
    # `cmake -E env`, which sets the variables for discovery and for every run.
    set_property(TARGET ${target} PROPERTY CROSSCOMPILING_EMULATOR
                 ${CMAKE_COMMAND} -E env ${ARG_ENVIRONMENT})
  endif()
  # 60 s, not the 5 s default: a large test binary linked against prosper_core can take longer than
  # that to list its cases on a first run (Rosetta, cold cache), and a discovery timeout fails ctest.
  gtest_discover_tests(${target} PROPERTIES LABELS "${ARG_LABELS}" ${ARG_PROPERTIES}
                       DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60)
endfunction()
