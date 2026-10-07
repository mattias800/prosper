# GoogleTest (GTest + GMock), vendored verbatim under third_party/googletest (v1.18.0; see its README).
# Usage: include(cmake/ProsperGTest.cmake) once after enable_testing(), then
#   prosper_add_gtest(<target> SOURCES ... [INCLUDES ...] [LIBRARIES ...] [LABELS ...] [PROPERTIES <name> <value> ...])
# PROPERTIES are applied to every discovered test (e.g. TIMEOUT 60). A list-valued property is one
# quoted argument, exactly as with set_tests_properties:
#   PROPERTIES ENVIRONMENT_MODIFICATION "A=unset:;B=unset:"
# ENVIRONMENT takes NAME=value words (ENVIRONMENT A=1 B=2), set for every run via `cmake -E env`.
# Each TEST() becomes its own ctest case (<Suite>.<Name>).
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest"
                 "${CMAKE_BINARY_DIR}/_deps/googletest-build" EXCLUDE_FROM_ALL)
include(GoogleTest)

# Quote `value` as a CMake bracket argument, choosing a bracket level the value does not contain.
# A bracket argument is one argument whatever it holds, so a ';' inside it stays a list separator
# WITHIN the property value instead of splitting the call's arguments.
function(_prosper_gtest_bracket out value)
  set(eq "")
  # The appended ']' also catches a value that ENDS in ']=...', which would close the bracket early.
  while("${value}]" MATCHES "\\]${eq}\\]")
    string(APPEND eq "=")
  endwhile()
  set(${out} "[${eq}[${value}]${eq}]" PARENT_SCOPE)
endfunction()

function(prosper_add_gtest target)
  cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES;LIBRARIES;LABELS;PROPERTIES;ENVIRONMENT" ${ARGN})
  # PROPERTIES is re-read with PARSE_ARGV, which keeps a quoted list ("A;B") as ONE element. The
  # plain form above flattens it into separate words, so a list-valued property would arrive as
  # several name/value pairs (#4682). The other keywords keep the plain form: they are word lists,
  # so flattening is what they mean.
  cmake_parse_arguments(PARSE_ARGV 1 PROP "" "" "SOURCES;INCLUDES;LIBRARIES;LABELS;PROPERTIES;ENVIRONMENT")
  add_executable(${target} ${ARG_SOURCES})
  if(ARG_INCLUDES)
    target_include_directories(${target} PRIVATE ${ARG_INCLUDES})
  endif()
  target_link_libraries(${target} PRIVATE ${ARG_LIBRARIES} GTest::gtest GTest::gmock GTest::gtest_main)
  if(ARG_ENVIRONMENT)
    # A multi-variable ENVIRONMENT test property cannot be expressed through gtest_discover_tests
    # (it emits each ';'-separated item as a separate argument), so launch the binary through
    # `cmake -E env`, which sets the variables for discovery and for every run.
    set_property(TARGET ${target} PROPERTY CROSSCOMPILING_EMULATOR
                 ${CMAKE_COMMAND} -E env ${ARG_ENVIRONMENT})
  endif()

  # Test properties are NOT forwarded through gtest_discover_tests(PROPERTIES ...) (#4682). The
  # GoogleTest module joins that list with "]] [[" into the generated discovery script, so a
  # list-valued value (ENVIRONMENT_MODIFICATION "A=unset:;B=unset:", LABELS "a;b") is split into
  # separate words and every later name/value pair shifts by one. Measured on #4682's
  # four-element ENVIRONMENT_MODIFICATION: CMake 3.31 (CI) silently applied only the first
  # element, and CTest 4.4 segfaulted while reading the tests, so no test in the tree ran.
  # Instead, a second test-include file applies them with set_tests_properties() to the tests
  # discovery just registered (the TEST_LIST variable), writing each value as one bracket
  # argument. This depends only on documented behaviour of both CMake generations.
  # Labels are words, so the flattening parse is the right one for them: LABELS a b == LABELS "a;b".
  set(labels "${ARG_LABELS}")
  if(NOT labels)
    set(labels unit)
  endif()
  list(LENGTH PROP_PROPERTIES prop_count)
  math(EXPR prop_odd "${prop_count} % 2")
  if(prop_odd)
    message(FATAL_ERROR
      "prosper_add_gtest(${target}): PROPERTIES must be <name> <value> pairs, got ${prop_count} "
      "words: ${PROP_PROPERTIES}. Quote a list-valued value as one argument (\"A;B\").")
  endif()
  set(pairs "")
  _prosper_gtest_bracket(quoted "${labels}")
  string(APPEND pairs " LABELS ${quoted}")
  if(prop_count GREATER 0)
    math(EXPR last "${prop_count} - 1")
    foreach(i RANGE 0 ${last} 2)
      math(EXPR j "${i} + 1")
      list(GET PROP_PROPERTIES ${i} name)
      list(GET PROP_PROPERTIES ${j} value)
      _prosper_gtest_bracket(quoted "${value}")
      string(APPEND pairs " ${name} ${quoted}")
    endforeach()
  endif()

  # 60 s, not the 5 s default: a large test binary linked against prosper_core can take longer than
  # that to list its cases on a first run (Rosetta, cold cache), and a discovery timeout fails ctest.
  gtest_discover_tests(${target} TEST_LIST ${target}_TESTS
                       DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60)

  # Runs after the discovery include (TEST_INCLUDE_FILES is ordered), so the list is populated.
  # Empty when the binary is not built yet: gtest then registers <target>_NOT_BUILT, which is left
  # without properties exactly as before.
  set(props_file "${CMAKE_CURRENT_BINARY_DIR}/${target}_prosper_props.cmake")
  file(GENERATE OUTPUT "${props_file}" CONTENT
    "if(${target}_TESTS)\n  set_tests_properties(\${${target}_TESTS} PROPERTIES${pairs})\nendif()\n")
  set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${props_file}")
endfunction()
