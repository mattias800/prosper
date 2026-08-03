set(PROSPER_GENERATED_REVISION "unknown")

# Both ordinary clones (.git directory) and linked worktrees (.git file) have an entry at their
# root. Requiring that entry prevents Git from walking up into an unrelated ancestor when this runs
# in an exported/non-Git source directory, without relying on platform-sensitive path comparisons.
if(PROSPER_REVISION_GIT_EXECUTABLE AND PROSPER_REVISION_WORK_TREE AND
   EXISTS "${PROSPER_REVISION_WORK_TREE}/.git")
  execute_process(
    COMMAND "${PROSPER_REVISION_GIT_EXECUTABLE}" rev-parse --verify HEAD
    WORKING_DIRECTORY "${PROSPER_REVISION_WORK_TREE}"
    RESULT_VARIABLE _revision_result
    OUTPUT_VARIABLE _revision_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  # A Git object id contains only hexadecimal digits. Reject unexpected output before placing it in
  # generated C++ rather than trying to quote arbitrary command output safely.
  if(_revision_result EQUAL 0 AND _revision_output MATCHES "^[0-9a-fA-F]+$")
    set(PROSPER_GENERATED_REVISION "${_revision_output}")
  endif()
endif()

# configure_file preserves the destination timestamp when its contents did not change. Therefore
# the inexpensive revision query runs before each relevant build, but compilation and relinking
# happen only after the resolved revision actually changes.
configure_file(
  "${PROSPER_REVISION_TEMPLATE}"
  "${PROSPER_REVISION_OUTPUT}"
  @ONLY
  NEWLINE_STYLE UNIX)
