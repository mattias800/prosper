include_guard(GLOBAL)

function(prosper_add_build_revision_library target_name)
  set(one_value_args WORK_TREE)
  cmake_parse_arguments(PBR "" "${one_value_args}" "" ${ARGN})
  if(NOT PBR_WORK_TREE)
    message(FATAL_ERROR "prosper_add_build_revision_library requires WORK_TREE")
  endif()

  find_package(Git QUIET)
  find_package(Python3 COMPONENTS Interpreter QUIET)

  set(_revision_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${target_name}")
  set(_revision_source "${_revision_dir}/build_revision.cpp")
  set(_revision_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerateBuildRevision.cmake")
  set(_revision_template "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/build_revision.cpp.in")
  set(_revision_include "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src")
  # Only digests are embedded, never host paths. Configuration options are resolved by this
  # configure; source bytes are re-read by the existing refresh target on EVERY relevant build.
  set(_identity_config "${CMAKE_CXX_COMPILER_ID}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CXX_COMPILER_TARGET}|${CMAKE_SYSTEM_NAME}|${CMAKE_SYSTEM_PROCESSOR}|${CMAKE_SIZEOF_VOID_P}|${CMAKE_CXX_STANDARD}|${CMAKE_CXX_FLAGS}|${CMAKE_CXX_FLAGS_DEBUG}|${CMAKE_CXX_FLAGS_RELEASE}|${CMAKE_CXX_FLAGS_RELWITHDEBINFO}|${CMAKE_CXX_FLAGS_MINSIZEREL}")
  if(EXISTS "${CMAKE_CXX_COMPILER}")
    file(SHA256 "${CMAKE_CXX_COMPILER}" _compiler_hash)
    string(APPEND _identity_config "|compiler=${_compiler_hash}")
  endif()
  get_cmake_property(_identity_variables VARIABLES)
  list(SORT _identity_variables)
  foreach(_variable IN LISTS _identity_variables)
    if(_variable MATCHES "^PROSPER_")
      string(APPEND _identity_config "|${_variable}=${${_variable}}")
    endif()
  endforeach()
  string(SHA256 _identity_config_hash "${_identity_config}")

  # This target intentionally runs whenever a consumer is built. Depending only on .git/HEAD is
  # insufficient: linked worktrees use a gitdir indirection, branch refs may be packed, and a
  # checkout can move between refs without changing the file a generator happened to watch.
  add_custom_target(${target_name}_refresh
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_revision_dir}"
    COMMAND "${CMAKE_COMMAND}"
      "-DPROSPER_REVISION_GIT_EXECUTABLE=${GIT_EXECUTABLE}"
      "-DPROSPER_REVISION_WORK_TREE=${PBR_WORK_TREE}"
      "-DPROSPER_REVISION_TEMPLATE=${_revision_template}"
      "-DPROSPER_REVISION_OUTPUT=${_revision_source}"
      "-DPROSPER_REVISION_CONFIG_ID=${_identity_config_hash}"
      "-DPROSPER_REVISION_COMPILE_COMMANDS=${CMAKE_BINARY_DIR}/compile_commands.json"
      "-DPROSPER_REVISION_GNU_VARIADIC_REQUIRED=${PROSPER_NATIVE_CLANG_TLS}"
      "-DPROSPER_REVISION_PYTHON=${Python3_EXECUTABLE}"
      "-DPROSPER_REVISION_BUILD_CONFIG=$<CONFIG>"
      -P "${_revision_script}"
    BYPRODUCTS "${_revision_source}"
    VERBATIM)

  set_source_files_properties("${_revision_source}" PROPERTIES GENERATED TRUE)
  add_library(${target_name} STATIC "${_revision_source}")
  add_dependencies(${target_name} ${target_name}_refresh)
  target_include_directories(${target_name} PUBLIC "${_revision_include}")
endfunction()
