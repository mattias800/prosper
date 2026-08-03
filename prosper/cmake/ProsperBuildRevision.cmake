include_guard(GLOBAL)

function(prosper_add_build_revision_library target_name)
  set(one_value_args WORK_TREE)
  cmake_parse_arguments(PBR "" "${one_value_args}" "" ${ARGN})
  if(NOT PBR_WORK_TREE)
    message(FATAL_ERROR "prosper_add_build_revision_library requires WORK_TREE")
  endif()

  find_package(Git QUIET)

  set(_revision_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${target_name}")
  set(_revision_source "${_revision_dir}/build_revision.cpp")
  set(_revision_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerateBuildRevision.cmake")
  set(_revision_template "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/build_revision.cpp.in")
  set(_revision_include "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src")

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
      -P "${_revision_script}"
    BYPRODUCTS "${_revision_source}"
    VERBATIM)

  set_source_files_properties("${_revision_source}" PROPERTIES GENERATED TRUE)
  add_library(${target_name} STATIC "${_revision_source}")
  add_dependencies(${target_name} ${target_name}_refresh)
  target_include_directories(${target_name} PUBLIC "${_revision_include}")
endfunction()
